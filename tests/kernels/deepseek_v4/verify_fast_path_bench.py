"""Device check + microbenchmark of TPU_DSV4_VERIFY_FAST_PATH at decode shapes.

Per TP rank at TP8: 16 attention heads x 512, indexer 64 heads x 128, one
speculative verify window of 7 tokens in a T8 bucket. For each op (SWA pass of
a CSA layer, the HCA layer op, the indexer top-k) it checks the fast path is
bit-identical to the default path on several batch distributions, then times
both: per-call device time = (wall(N2) - wall(N1)) / (N2 - N1) of a jitted loop
of N calls whose inputs depend on the previous call's output.

Run: python tests/kernels/deepseek_v4/verify_fast_path_bench.py [ctx_k ...]
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# JAX owns the TPU runtime: initialize it before anything imports torch_tpu.
print(jax.devices())

from vllm_torchtpu.kernels.deepseek_v4.core_attention import (  # noqa: E402
    mla,
    mla_swa,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (  # noqa: E402
    streamindex_topk,
)

H, D = 16, 512
IH, ID = 64, 128
WINDOW, SWA_PAGE, SWA_PHYS = 128, 64, 128
HCA_RATIO, HCA_PAGE = 128, 8
IDX_RATIO, IDX_PAGE, TOPK = 4, 256, 1024
MAX_SEQS = 8
VQ = 7
SM_SCALE = D**-0.5


def batch(q_lens, kinds, T):
    """cu_q_lens [MAX_SEQS+1] and distribution for sequences of q_lens."""
    cu = np.zeros(MAX_SEQS + 1, np.int32)
    cu[1 : len(q_lens) + 1] = np.cumsum(q_lens)
    cu[len(q_lens) + 1 :] = cu[len(q_lens)]
    assert cu[-1] <= T
    n_dec = kinds.count("d")
    n_ver = kinds.count("v")
    dist = np.array([n_dec, n_dec + n_ver, len(q_lens)], np.int32)
    return jnp.asarray(cu), jnp.asarray(dist)


def perm_table(pages_per_seq, num_pages, seed):
    rng = np.random.default_rng(seed)
    t = rng.permutation(max(num_pages, MAX_SEQS * pages_per_seq)) % num_pages
    return jnp.asarray(t[: MAX_SEQS * pages_per_seq].astype(np.int32))


def finite_bytes(key, shape):
    """Random uint8 whose bf16 pairs are finite (exponent top bit cleared)."""
    return jax.random.randint(key, shape, 0, 256, jnp.int32).astype(
        jnp.uint8
    ) & jnp.uint8(0xBF)


def swa_inputs(T, ctxs, key):
    k = jax.random.split(key, 4)
    pages_per_seq = 8192
    num_pages = 1024
    q = jax.random.normal(k[0], (T, H, D), jnp.bfloat16)
    new_kv = jax.random.normal(k[1], (T, D), jnp.bfloat16)
    cache = finite_bytes(k[2], (num_pages, SWA_PHYS * 2, 4, 128))
    kv_lens = np.zeros(MAX_SEQS, np.int32)
    kv_lens[: len(ctxs)] = ctxs
    sinks = jax.random.normal(k[3], (H,), jnp.float32)
    pages = perm_table(pages_per_seq, num_pages, 1)
    return q, new_kv, cache, jnp.asarray(kv_lens), pages, sinks


def hca_inputs(T, ctxs, cu, key):
    pages_per_seq = 1024
    num_pages = 640
    cache = finite_bytes(key, (num_pages, HCA_PAGE * 2, 4, 128))
    kv_lens = np.zeros(MAX_SEQS, np.int32)
    kv_lens[: len(ctxs)] = np.asarray(ctxs) // HCA_RATIO
    # Tokens attend to the compressed rows complete at their position.
    cu_np = np.asarray(cu)
    attend = np.zeros(T, np.int32)
    for s, L in enumerate(ctxs):
        n = cu_np[s + 1] - cu_np[s]
        for i in range(n):
            attend[cu_np[s] + i] = (L - n + i + 1) // HCA_RATIO
    pages = perm_table(pages_per_seq, num_pages, 2)
    return cache, jnp.asarray(kv_lens), jnp.asarray(attend), pages


def idx_inputs(T, ctxs, key):
    k = jax.random.split(key, 3)
    pages_per_seq = 1024
    num_pages = 600
    q = jax.random.normal(k[0], (T, IH, ID), jnp.float32).astype(jnp.float8_e4m3fn)
    w = jax.random.normal(k[1], (T, IH), jnp.float32)
    keys = jax.random.normal(k[2], (num_pages, 32, 4, IDX_PAGE), jnp.float32)
    key_bytes = jax.lax.bitcast_convert_type(
        keys.astype(jnp.float8_e4m3fn), jnp.uint8
    )
    scale = jnp.zeros((num_pages, 1, 4, IDX_PAGE), jnp.uint8).at[:, :, 0, :].set(127)
    spare = jnp.zeros((num_pages, 31, 4, IDX_PAGE), jnp.uint8)
    cache = jnp.concatenate([key_bytes, scale, spare], axis=1)
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[: len(ctxs)] = ctxs
    pages = perm_table(pages_per_seq, num_pages, 3)
    return q, w, cache, jnp.asarray(seq_lens), pages


def make_swa(fast):
    """The SWA pass of a CSA/HCA layer, as `_run_swa` calls it."""

    def f(q, new_kv, cache, kv_lens, pages, cu, dist, sinks):
        return mla_swa.mla_sliding_window_ragged_paged_attention(
            q=q, new_kv=new_kv, cache_kv=cache, kv_lens=kv_lens,
            page_indices=pages, cu_q_lens=cu, distribution=dist,
            attention_sinks=sinks, sm_scale=SM_SCALE, sliding_window=WINDOW,
            logical_page_size=SWA_PAGE, num_kv_pages_per_block=(2, 2, 2),
            num_queries_per_block=(1, 32, 32), q_compute_block_size=4,
            unnormalized_output=True, chunk_prefill_size=VQ, padded_lm_out=True,
            verify_fast_path=fast,
        )  # fmt: skip

    return f


def make_hca(fast):
    """An HCA layer's attention, as `_attention_hca` runs it (stack7 flags)."""
    swa = make_swa(fast)

    def f(q, new_kv, cache, kv_lens, pages, cu, dist, sinks, hcache, hlens, attend,
          hpages):  # fmt: skip
        swa_out, cache, swa_l, swa_m = swa(
            q, new_kv, cache, kv_lens, pages, cu, dist, sinks
        )
        out = mla.mla_ragged_paged_attention(
            q=q, cache_kv=hcache, kv_lens=hlens, kv_lens_to_attend=attend,
            page_indices=hpages, cu_q_lens=cu, distribution=dist,
            attention_sinks=sinks, swa_accumution=swa_out, swa_l=swa_l,
            swa_m=swa_m, sm_scale=SM_SCALE, num_kv_pages_per_block=(32, 64, 64),
            num_queries_per_block=(1, 32, 32), chunk_prefill_size=VQ,
            bkv_prefetch="after_wait", fast_dma=True, verify_fast_path=fast,
        )  # fmt: skip
        return out, cache

    return f


def make_idx(fast):
    def f(q, w, cache, seq_lens, pages, cu, dist):
        return streamindex_topk(
            q=q, indexer_weights=w, cache_kv=cache, seq_lens=seq_lens,
            page_indices=pages, cu_q_lens=cu, distribution=dist, k=TOPK,
            compression_ratio=IDX_RATIO, num_kv_pages_per_block=(32, 32, 16),
            num_queries_per_block=(1, 128, 32), decode_req_batch_size=1,
            uniform_q_len=VQ, kv_layout="seq_along_lane", strict_causal=True,
            sc_topk_one_launch=True, sc_topk_coop_max_rows=16,
            verify_fast_path=fast,
        )  # fmt: skip

    return f


def real_rows(x, cu):
    n = int(np.asarray(cu)[-1])
    return np.asarray(x)[:n]


def check_exact(name, outs_a, outs_b, cu, cache_idx=None):
    ok = True
    for i, (a, b) in enumerate(zip(outs_a, outs_b)):
        if i == cache_idx:
            a, b = np.asarray(a), np.asarray(b)
        else:
            a, b = real_rows(a, cu), real_rows(b, cu)
        a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
        same = np.array_equal(a.view(np.uint8), b.view(np.uint8))
        ok &= same
        if not same:
            print(f"  MISMATCH {name} output {i}: {np.sum(a != b)} elements differ")
    print(f"EXACT {name}: {'PASS' if ok else 'FAIL'}")
    return ok


def timed_loop(step, n):
    @jax.jit
    def run(state):
        return jax.lax.fori_loop(0, n, lambda i, s: step(s), state)

    return run


def per_call_us(step, state, n1=8, n2=72, reps=5):
    r1, r2 = timed_loop(step, n1), timed_loop(step, n2)
    for r in (r1, r2):
        jax.block_until_ready(r(state))
    t = {}
    for n, r in ((n1, r1), (n2, r2)):
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(r(state))
            best = min(best, time.perf_counter() - t0)
        t[n] = best
    return (t[n2] - t[n1]) / (n2 - n1) * 1e6


def bench_ctx(L):
    key = jax.random.PRNGKey(L)
    T = 8
    cu, dist = batch([VQ], ["v"], T)
    q, new_kv, cache, kv_lens, pages, sinks = swa_inputs(T, [L], key)
    hcache, hlens, attend, hpages = hca_inputs(T, [L], cu, key)
    iq, iw, icache, ilens, ipages = idx_inputs(T, [L], key)

    res = {}
    for fast in (False, True):
        swa = make_swa(fast)

        def swa_step(s, swa=swa):
            q_, cache_ = s
            o, c, _, _ = swa(q_, new_kv, cache_, kv_lens, pages, cu, dist, sinks)
            return (o.astype(q_.dtype), c)

        hca = make_hca(fast)

        def hca_step(s, hca=hca):
            q_, cache_ = s
            o, c = hca(
                q_, new_kv, cache_, kv_lens, pages, cu, dist, sinks, hcache, hlens,
                attend, hpages,
            )  # fmt: skip
            return (o, c)

        idx = make_idx(fast)

        def idx_step(s, idx=idx):
            lens_, acc = s
            out = idx(iq, iw, icache, lens_ + jnp.minimum(acc, 0), ipages, cu, dist)
            return (lens_, jnp.maximum(out[0, 0], 0))

        tag = "fast" if fast else "base"
        res[("swa", tag)] = per_call_us(swa_step, (q, cache + 0))
        res[("hca", tag)] = per_call_us(hca_step, (q, cache + 0))
        res[("idx", tag)] = per_call_us(idx_step, (ilens, jnp.int32(0)))
    for op in ("swa", "hca", "idx"):
        b, f = res[(op, "base")], res[(op, "fast")]
        print(
            f"TIME ctx={L // 1000}k {op}: base {b:.2f} us  fast {f:.2f} us  "
            f"delta {f - b:+.2f} us"
        )


def exactness():
    key = jax.random.PRNGKey(7)
    cases = [  # (name, T, contexts, q_lens, kinds)
        ("T8 verify", 8, [480_000], [VQ], ["v"]),
        ("T16 2 verify", 16, [300_123, 90_001], [VQ, VQ], ["v", "v"]),
        ("T16 decode+verify", 16, [70_000, 200_005], [1, VQ], ["d", "v"]),
        ("T16 verify+mixed", 16, [150_000, 5_000], [VQ, 9], ["v", "m"]),
        ("T16 decode+mixed", 16, [64_000, 5_003], [1, 12], ["d", "m"]),
    ]
    ok = True
    for name, T, ctxs, q_lens, kinds in cases:
        cu, dist = batch(q_lens, kinds, T)
        q, new_kv, cache, kv_lens, pages, sinks = swa_inputs(T, ctxs, key)
        hcache, hlens, attend, hpages = hca_inputs(T, ctxs, cu, key)
        iq, iw, icache, ilens, ipages = idx_inputs(T, ctxs, key)
        outs = {}
        for fast in (False, True):
            outs[("swa", fast)] = list(
                jax.jit(make_swa(fast))(
                    q, new_kv, cache + 0, kv_lens, pages, cu, dist, sinks
                )
            )
            outs[("hca", fast)] = jax.jit(make_hca(fast))(
                q, new_kv, cache + 0, kv_lens, pages, cu, dist, sinks, hcache,
                hlens, attend, hpages,
            )  # fmt: skip
            outs[("idx", fast)] = (
                jax.jit(make_idx(fast))(iq, iw, icache, ilens, ipages, cu, dist),
            )
            # l/m are lane-padded; only the real heads are defined.
            for i in (2, 3):
                outs[("swa", fast)][i] = outs[("swa", fast)][i][:, :H]
        for op in ("swa", "hca", "idx"):
            ok &= check_exact(
                f"{op} {name}",
                outs[(op, False)],
                outs[(op, True)],
                cu,
                cache_idx=None if op == "idx" else 1,
            )
    return ok


def main():
    ctxs = [int(a) * 1000 for a in sys.argv[1:]] or [80_000, 160_000, 480_000]
    ok = exactness()
    for L in ctxs:
        bench_ctx(L)
    print("ALL EXACT" if ok else "EXACTNESS FAILURES")


if __name__ == "__main__":
    main()
