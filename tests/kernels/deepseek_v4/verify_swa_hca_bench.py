"""Device check + microbenchmark of TPU_DSV4_VERIFY_FUSED_HCA at decode shapes.

An HCA layer's attention (`_attention_hca`, stack7 settings) with and without
`verify_fused_hca`, per TP rank at TP8 (16 heads x 512), served layouts (SWA
cache int32 words with 128-token pages, HCA cache uint8 slabs with 8-row
pages). Checks, on batches mixing decode, verify and mixed sequences: the
verify rows against an f32 reference (both paths), the decode/mixed rows
bit-identical between the paths, and the SWA cache identical. Then times
both per layer (jitted loop of steps of 8 dependent layer calls that share
the metadata glue, as the model's 61 layers do; best of 5).

Run: python tests/kernels/deepseek_v4/verify_swa_hca_bench.py [ctx_k ...]
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# JAX owns the TPU runtime: initialize it before anything imports torch_tpu.
print(jax.devices())

import verify_window_ref as ref  # noqa: E402

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (  # noqa: E402
    deepseek_v4_attention_op as attn_op,
)

H, D, VQ, WINDOW = 16, 512, 7, 128
SWA_PAGE, HCA_PAGE, RATIO = 128, 8, 128
MAX_SEQS = 8
SM_SCALE = D**-0.5


def finite_bytes(key, shape):
    """Random uint8 whose bf16 pairs are finite (exponent top bit cleared)."""
    x = jax.random.randint(key, shape, 0, 256, jnp.int32).astype(jnp.uint8)
    return x & jnp.uint8(0xBF)


def perm_table(pages_per_seq, num_pages, seed):
    rng = np.random.default_rng(seed)
    t = rng.permutation(max(num_pages, MAX_SEQS * pages_per_seq)) % num_pages
    return jnp.asarray(t[: MAX_SEQS * pages_per_seq].astype(np.int32))


def inputs(T, seqs, key):
    """seqs: [(kind, q_len, kv_len)], kinds in decode/verify/mixed order."""
    k = jax.random.split(key, 5)
    q_lens = [n for _, n, _ in seqs]
    cu = np.zeros(MAX_SEQS + 1, np.int32)
    cu[1 : len(seqs) + 1] = np.cumsum(q_lens)
    cu[len(seqs) + 1 :] = cu[len(seqs)]
    kinds = [kd for kd, _, _ in seqs]
    nd, nv = kinds.count("d"), kinds.count("v")
    dist = np.array([nd, nd + nv, len(seqs)], np.int32)
    kv_lens = np.zeros(MAX_SEQS, np.int32)
    kv_lens[: len(seqs)] = [L for _, _, L in seqs]
    attend = np.zeros(T, np.int32)
    for s, (_, n, L) in enumerate(seqs):
        for i in range(n):
            attend[cu[s] + i] = (L - n + i + 1) // RATIO
    return dict(
        q=jax.random.normal(k[0], (T, H, D), jnp.bfloat16),
        new_kv=jax.random.normal(k[1], (T, D), jnp.bfloat16),
        sw=jax.lax.bitcast_convert_type(
            finite_bytes(k[2], (512, 2 * SWA_PAGE, 128, 4)), jnp.int32
        ),
        hca=finite_bytes(k[3], (640, 2 * HCA_PAGE, 4, 128)),
        sinks=jax.random.normal(k[4], (H,), jnp.float32),
        kv_lens=jnp.asarray(kv_lens), hlens=jnp.asarray(kv_lens // RATIO),
        attend=jnp.asarray(attend), cu=jnp.asarray(cu), dist=jnp.asarray(dist),
        swa_pt=perm_table(8192, 512, 11), hca_pt=perm_table(1024, 640, 12),
        seqs=seqs, cu_np=cu,
    )  # fmt: skip


def hca(x, q, sw, fused):
    return attn_op._attention_hca(
        q, x["new_kv"], sw, x["kv_lens"], x["swa_pt"], x["cu"], x["dist"],
        x["hca"], x["hlens"], x["attend"], x["hca_pt"], x["cu"], x["dist"],
        x["sinks"], sm_scale=SM_SCALE, sliding_window=WINDOW,
        logical_page_size=SWA_PAGE, swa_only=False, two_caches_same_buffer=False,
        verify_q_len=VQ, hca_mixed_q_block=128, hca_pipelined=True,
        swa_padded_lm=True, verify_fused_hca=fused,
    )  # fmt: skip


def reference_verify(x, s):
    """f32 attention of verify sequence s: [VQ, H, D]."""
    hca_words = jax.lax.bitcast_convert_type(
        jnp.swapaxes(x["hca"], 2, 3), jnp.int32
    )  # uint8 slab sub-rows are the bytes of one word, little endian
    c0 = int(x["cu_np"][s])
    out = ref.swa_hca_reference(
        x["q"][c0 : c0 + VQ][None], x["new_kv"][c0 : c0 + VQ][None], x["sw"],
        x["swa_pt"].reshape(MAX_SEQS, -1)[s : s + 1], hca_words,
        x["hca_pt"].reshape(MAX_SEQS, -1)[s : s + 1], x["kv_lens"][s : s + 1],
        x["sinks"], sm_scale=SM_SCALE, window=WINDOW, swa_page=SWA_PAGE,
        ratio=RATIO, hca_page=HCA_PAGE,
    )  # fmt: skip
    return np.asarray(out)[0]


def check(name, T, seqs):
    x = inputs(T, seqs, jax.random.PRNGKey(T + len(seqs)))
    o_s, c_s = jax.jit(lambda q, sw: hca(x, q, sw, False))(x["q"], x["sw"] + 0)
    o_f, c_f = jax.jit(lambda q, sw: hca(x, q, sw, True))(x["q"], x["sw"] + 0)
    o_s, o_f = np.asarray(o_s, np.float32), np.asarray(o_f, np.float32)
    e_s = e_f = 0.0
    other_same = True
    for s, (kind, n, _) in enumerate(seqs):
        rows = slice(int(x["cu_np"][s]), int(x["cu_np"][s]) + n)
        if kind == "v":
            r = reference_verify(x, s)
            e_s = max(e_s, np.max(np.abs(o_s[rows] - r)))
            e_f = max(e_f, np.max(np.abs(o_f[rows] - r)))
        else:
            other_same &= np.array_equal(o_s[rows], o_f[rows])
    same_cache = np.array_equal(np.asarray(c_s), np.asarray(c_f))
    ok = same_cache and other_same and e_f <= max(2 * e_s, 1e-4)
    print(
        f"NUM {name}: maxerr vs f32 ref, verify rows: stock {e_s:.3g} fused {e_f:.3g}; "
        f"decode/mixed rows identical {other_same}; swa cache identical {same_cache}  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def per_call_us(step, state, n1=2, n2=10, reps=5):
    def loop(n):
        return jax.jit(lambda s: jax.lax.fori_loop(0, n, lambda i, s: step(s), s))

    r1, r2 = loop(n1), loop(n2)
    t = {}
    for n, r in ((n1, r1), (n2, r2)):
        jax.block_until_ready(r(state))
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(r(state))
            best = min(best, time.perf_counter() - t0)
        t[n] = best
    return (t[n2] - t[n1]) / (n2 - n1) * 1e6


LAYERS = 8  # per timed step, sharing the metadata glue as the model's layers do


def bench(L, T=8):
    x = inputs(T, [("v", VQ, L)], jax.random.PRNGKey(L))
    res = {}
    for fused in (False, True):

        def step(s, fused=fused):
            q, sw = s
            for _ in range(LAYERS):
                q, sw = hca(x, q, sw, fused)
            return q, sw

        res[fused] = per_call_us(step, (x["q"], x["sw"] + 0)) / LAYERS
    print(
        f"TIME ctx={L // 1000}k T{T}: stock {res[False]:.2f} us  fused {res[True]:.2f} us"
        f"  delta {res[True] - res[False]:+.2f} us"
    )


def main():
    ctxs = [int(a) * 1000 for a in sys.argv[1:]] or [20_000, 80_000, 160_000, 480_000]
    ok = True
    ok &= check("T8 verify 480k", 8, [("v", VQ, 480_011)])
    ok &= check("T8 verify 2k", 8, [("v", VQ, 2_003)])
    ok &= check("T16 2 verify", 16, [("v", VQ, 300_123), ("v", VQ, 90_001)])
    ok &= check("T16 decode+verify", 16, [("d", 1, 70_000), ("v", VQ, 200_005)])
    ok &= check("T16 decode x3 + verify", 16,
                [("d", 1, 5_000), ("d", 1, 70_000), ("d", 1, 9), ("v", VQ, 200_005)])
    ok &= check("T32 verify+mixed", 32, [("v", VQ, 150_000), ("m", 20, 5_000)])
    ok &= check("T32 4 verify", 32, [("v", VQ, L) for L in (100_000, 260_000, 7_000, 480_000)])
    ok &= check("T16 mixed only", 16, [("m", 13, 4_000)])
    for L in ctxs:
        bench(L)
    bench(160_000, T=16)
    print("ALL NUMERICS PASS" if ok else "NUMERICS FAILURES")


if __name__ == "__main__":
    main()
