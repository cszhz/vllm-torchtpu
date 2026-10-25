"""Device check + microbenchmark of TPU_DSV4_VERIFY_FUSED_HCA at decode shapes.

An HCA layer's attention for a batch of 7-token verify windows: the stock
fast path (SWA static pass, then the HCA kernel's static pass) against
`verify_swa_hca` (one kernel). Served layouts: SWA cache int32 words with
128-token pages, HCA cache uint8 slabs with 8-row pages. Reports per-call
device time and, against an f32 reference, the max |error| of both paths, and
checks the SWA cache writes are identical.

Run: python tests/kernels/deepseek_v4/verify_swa_hca_bench.py [ctx_k ...]
"""

import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np

# JAX owns the TPU runtime: initialize it before anything imports torch_tpu.
print(jax.devices())

import verify_fast_path_bench as vb  # noqa: E402
import verify_window_ref as ref  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.core_attention import (  # noqa: E402
    mla,
    mla_swa,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention.verify_window import (  # noqa: E402
    attn_swa_hca,
)

H, D, VQ, WINDOW = vb.H, vb.D, vb.VQ, vb.WINDOW
SWA_PAGE, HCA_PAGE, RATIO = 128, 8, 128
SM_SCALE = vb.SM_SCALE


def inputs(T, ctxs, key):
    k = jax.random.split(key, 5)
    swa_pps, swa_pages = 8192, 512
    hca_pps, hca_pages = 1024, 640
    q = jax.random.normal(k[0], (T, H, D), jnp.bfloat16)
    new_kv = jax.random.normal(k[1], (T, D), jnp.bfloat16)
    sw = jax.lax.bitcast_convert_type(
        vb.finite_bytes(k[2], (swa_pages, 2 * SWA_PAGE, 128, 4)), jnp.int32
    )
    hca = vb.finite_bytes(k[3], (hca_pages, 2 * HCA_PAGE, 4, 128))
    sinks = jax.random.normal(k[4], (H,), jnp.float32)
    n = len(ctxs)
    kv_lens = np.zeros(vb.MAX_SEQS, np.int32)
    kv_lens[:n] = ctxs
    cu, dist = vb.batch([VQ] * n, ["v"] * n, T)
    attend = np.zeros(T, np.int32)
    for s, L in enumerate(ctxs):
        for i in range(VQ):
            attend[s * VQ + i] = (L - VQ + i + 1) // RATIO
    return dict(
        q=q, new_kv=new_kv, sw=sw, hca=hca, sinks=sinks,
        kv_lens=jnp.asarray(kv_lens), hlens=jnp.asarray(kv_lens // RATIO),
        attend=jnp.asarray(attend), cu=cu, dist=dist,
        swa_pt=vb.perm_table(swa_pps, swa_pages, 11),
        hca_pt=vb.perm_table(hca_pps, hca_pages, 12),
    )  # fmt: skip


def stock(x, q, sw):
    swa_out, sw, swa_l, swa_m = mla_swa.mla_sliding_window_ragged_paged_attention(
        q=q, new_kv=x["new_kv"], cache_kv=sw, kv_lens=x["kv_lens"],
        page_indices=x["swa_pt"], cu_q_lens=x["cu"], distribution=x["dist"],
        attention_sinks=x["sinks"], sm_scale=SM_SCALE, sliding_window=WINDOW,
        logical_page_size=SWA_PAGE, num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 32, 32), q_compute_block_size=4,
        unnormalized_output=True, chunk_prefill_size=VQ, padded_lm_out=True,
        verify_fast_path=True,
    )  # fmt: skip
    out = mla.mla_ragged_paged_attention(
        q=q, cache_kv=x["hca"], kv_lens=x["hlens"], kv_lens_to_attend=x["attend"],
        page_indices=x["hca_pt"], cu_q_lens=x["cu"], distribution=x["dist"],
        attention_sinks=x["sinks"], swa_accumution=swa_out, swa_l=swa_l,
        swa_m=swa_m, sm_scale=SM_SCALE, num_kv_pages_per_block=(32, 64, 64),
        num_queries_per_block=(1, 32, 32), chunk_prefill_size=VQ,
        bkv_prefetch="after_wait", fast_dma=True, verify_fast_path=True,
    )  # fmt: skip
    return out, sw


def fused(x, q, sw, blk_pages=None):
    return attn_swa_hca.verify_swa_hca(
        q, x["new_kv"], sw, x["kv_lens"], x["swa_pt"], x["hca"], x["hca_pt"],
        x["dist"][1], x["sinks"], q_len=VQ, sm_scale=SM_SCALE, window=WINDOW,
        blk_pages=blk_pages,
    )  # fmt: skip


def reference(x, n):
    hca_words = jax.lax.bitcast_convert_type(
        jnp.swapaxes(x["hca"], 2, 3), jnp.int32
    )  # uint8 slab sub-rows are the bytes of one word, little endian
    swa_pps = x["swa_pt"].shape[0] // vb.MAX_SEQS
    hca_pps = x["hca_pt"].shape[0] // vb.MAX_SEQS
    out = ref.swa_hca_reference(
        x["q"][: n * VQ].reshape(n, VQ, H, D),
        x["new_kv"][: n * VQ].reshape(n, VQ, D),
        x["sw"], x["swa_pt"].reshape(vb.MAX_SEQS, swa_pps)[:n], hca_words,
        x["hca_pt"].reshape(vb.MAX_SEQS, hca_pps)[:n], x["kv_lens"][:n],
        x["sinks"], sm_scale=SM_SCALE, window=WINDOW, swa_page=SWA_PAGE,
        ratio=RATIO, hca_page=HCA_PAGE,
    )  # fmt: skip
    return np.asarray(out).reshape(n * VQ, H, D)


def check(name, T, ctxs):
    x = inputs(T, ctxs, jax.random.PRNGKey(len(ctxs) + T))
    n = len(ctxs)
    o_s, c_s = jax.jit(lambda q, sw: stock(x, q, sw))(x["q"], x["sw"] + 0)
    o_f, c_f = jax.jit(lambda q, sw: fused(x, q, sw))(x["q"], x["sw"] + 0)
    r = reference(x, n)
    o_s = np.asarray(o_s, np.float32)[: n * VQ]
    o_f = np.asarray(o_f, np.float32)[: n * VQ]
    scale = np.max(np.abs(r))
    e_s, e_f = np.max(np.abs(o_s - r)), np.max(np.abs(o_f - r))
    d_sf = np.max(np.abs(o_s - o_f))
    rms_s = np.sqrt(np.mean((o_s - r) ** 2))
    rms_f = np.sqrt(np.mean((o_f - r) ** 2))
    same_cache = np.array_equal(np.asarray(c_s), np.asarray(c_f))
    ok = same_cache and e_f <= max(2 * e_s, 1e-2 * scale)
    print(
        f"NUM {name}: |ref|max {scale:.3g}  maxerr stock {e_s:.3g} fused {e_f:.3g}  "
        f"rms stock {rms_s:.3g} fused {rms_f:.3g}  max|stock-fused| {d_sf:.3g}  "
        f"swa cache identical {same_cache}  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def bench(L, T=8, blks=(32, 64, 128)):
    x = inputs(T, [L], jax.random.PRNGKey(L))
    res = {}
    variants = [("stock", stock)] + [
        (f"fused{b}", functools.partial(fused, blk_pages=b)) for b in blks
    ]
    for name, f in variants:

        def step(s, f=f):
            q, sw = s
            o, sw = f(x, q, sw)
            return (o.astype(q.dtype), sw)

        res[name] = vb.per_call_us(step, (x["q"], x["sw"] + 0))
    print(f"TIME ctx={L // 1000}k T{T}: " + "  ".join(
        f"{k} {v:.2f}" for k, v in res.items()) + " us")  # fmt: skip


def main():
    ctxs = [int(a) * 1000 for a in sys.argv[1:]] or [
        20_000, 80_000, 130_000, 160_000, 240_000, 330_000, 480_000
    ]
    ok = True
    ok &= check("T8 1x480k", 8, [480_011])
    ok &= check("T8 1x2k", 8, [2_003])
    ok &= check("T16 2 windows", 16, [300_123, 90_001])
    ok &= check("T16 1 window", 16, [150_000])
    ok &= check("T32 4 windows", 32, [100_000, 260_000, 7_000, 480_000])
    for L in ctxs:
        bench(L)
    bench(160_000, T=16)
    print("ALL NUMERICS PASS" if ok else "NUMERICS FAILURES")


if __name__ == "__main__":
    main()
