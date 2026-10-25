"""Step-shaped microbenchmark of the DSv4 indexer's TP kv-split at decode buckets.

Every TP rank holds the whole indexer K cache and the same queries (the indexer
is replicated). Variants, per CSA layer:
  unsplit    each rank scores every key and runs the SparseCore top-k
  split_own  each rank scores 1/TP of the keys (its top-k with scores), the
             shares are all-gathered, each rank merges T/TP rows, the merged
             rows are all-gathered (today's TPU_DSV4_INDEXER_KV_SPLIT path)
  split_all  as split_own but every rank merges all rows: one all-gather
Each timed step chains LAYERS layers; every layer also runs an independent
bf16 matmul (OTHER_N columns, ~9.5 us of weight traffic at 2048) that the async SparseCore work
and collectives can hide behind, as the q projection and SWA do in the model.
The next layer's indexer weights depend on this layer's top-k.
Also checks that both split variants return the unsplit top-k exactly.

Run: python tests/kernels/deepseek_v4/indexer_kv_split_bench.py
"""

import os
import re
import time

import jax
import jax.numpy as jnp
import numpy as np

# JAX owns the TPU runtime: initialize it before anything imports torch_tpu.
print(jax.devices())

from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (  # noqa: E402
    merge_kv_split_topk,
    streamindex_topk,
)

try:
    shard_map = jax.shard_map
except AttributeError:  # older jax
    from jax.experimental.shard_map import shard_map

IH, ID, RATIO, PAGE, TOPK, VQ = 64, 128, 4, 256, 1024, 7
MAX_SEQS, PAGES_PER_SEQ, NUM_PAGES = 8, 1024, 2048
LAYERS = 8
OTHER_N = int(os.environ.get("OTHER_N", "2048"))  # [8,7168]x[7168,N] bf16 per layer


def inputs(T, ctxs, key):
    k = jax.random.split(key, 4)
    q = jax.random.normal(k[0], (T, IH, ID), jnp.float32).astype(jnp.float8_e4m3fn)
    w = jax.random.normal(k[1], (T, IH), jnp.float32)
    keys = jax.random.normal(k[2], (NUM_PAGES, 32, 4, PAGE), jnp.float32)
    key_bytes = jax.lax.bitcast_convert_type(keys.astype(jnp.float8_e4m3fn), jnp.uint8)
    scale = jnp.zeros((NUM_PAGES, 1, 4, PAGE), jnp.uint8).at[:, :, 0, :].set(127)
    cache = jnp.concatenate(
        [key_bytes, scale, jnp.zeros((NUM_PAGES, 31, 4, PAGE), jnp.uint8)], axis=1
    )
    n = len(ctxs)
    q_lens = [VQ] * n
    cu = np.zeros(MAX_SEQS + 1, np.int32)
    cu[1 : n + 1] = np.cumsum(q_lens)
    cu[n + 1 :] = cu[n]
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:n] = ctxs
    rng = np.random.default_rng(5)
    pages = rng.permutation(MAX_SEQS * PAGES_PER_SEQ) % NUM_PAGES
    return dict(
        q=q, w=w, cache=cache, seq_lens=jnp.asarray(seq_lens),
        pages=jnp.asarray(pages.astype(np.int32)), cu=jnp.asarray(cu),
        dist=jnp.asarray(np.array([0, n, n], np.int32)),
    )  # fmt: skip


def topk(x, w, kv_split=1, rank=0):
    return streamindex_topk(
        q=x["q"], indexer_weights=w, cache_kv=x["cache"], seq_lens=x["seq_lens"],
        page_indices=x["pages"], cu_q_lens=x["cu"], distribution=x["dist"], k=TOPK,
        compression_ratio=RATIO, num_kv_pages_per_block=(32, 32, 16),
        num_queries_per_block=(1, 128, 32), decode_req_batch_size=1, uniform_q_len=VQ,
        kv_layout="seq_along_lane", strict_causal=True,
        sc_topk_one_launch=kv_split == 1, sc_topk_coop_max_rows=16 if kv_split == 1 else 0,
        kv_split=kv_split, kv_split_rank=rank,
    )  # fmt: skip


def indexer(variant, x, w, tp):
    """Per-rank body (inside shard_map): i32[T, TOPK]."""
    if variant == "unsplit":
        return topk(x, w)
    rank = jax.lax.axis_index("tp")
    idx, sc = topk(x, w, kv_split=tp, rank=rank)
    share = jnp.concatenate([idx, sc], axis=1)  # [T, 2k]
    shares = jax.lax.all_gather(share, "tp")  # [tp, T, 2k]
    T = share.shape[0]
    if variant == "split_all" or T % tp:
        return merge_kv_split_topk(shares, TOPK)
    rows = T // tp
    mine = jax.lax.dynamic_slice_in_dim(shares, rank * rows, rows, axis=1)
    merged = merge_kv_split_topk(mine, TOPK)  # [T/tp, k]
    return jax.lax.all_gather(merged, "tp", tiled=True)


def make_step(variant, x, tp, mesh, layers=LAYERS):
    def body(w, h, wo):
        outs = []
        for _ in range(layers):
            out = indexer(variant, x, w, tp) if variant != "none" else None
            if OTHER_N:  # independent work
                h = h.at[:, :OTHER_N].set(jnp.tanh(h @ wo).astype(h.dtype))
            if out is not None:
                w = w + jnp.minimum(out[0, 0], 0).astype(w.dtype)
                outs.append(out)
            else:
                w = w + jnp.minimum(h[0, 0], 0).astype(w.dtype)
        return w, h, (outs[-1] if outs else jnp.zeros((1, TOPK), jnp.int32))

    f = shard_map(
        body, mesh=mesh, in_specs=(P(), P(), P()), out_specs=(P(), P(), P()),
        check_vma=False,
    )  # fmt: skip
    return f


def per_step_us(f, args, name, reps=6, top_ops=0):
    """Median device time of one step: the XLA module span on TPU 0 in a
    profiler trace (host dispatch of an 8-device program is ~1.3 ms, longer
    than the steps measured here)."""
    import glob
    import gzip
    import json
    import tempfile

    step = jax.jit(f)
    w, h, wo = args
    jax.block_until_ready(step(w, h, wo))
    d = tempfile.mkdtemp()
    with jax.profiler.trace(d, create_perfetto_trace=True):
        for _ in range(reps):
            jax.block_until_ready(step(w, h, wo))
    (path,) = glob.glob(f"{d}/**/perfetto_trace.json.gz", recursive=True)
    ev = json.load(gzip.open(path))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    pids = {e["pid"]: e["args"]["name"] for e in ev
            if e.get("ph") == "M" and e.get("name") == "process_name"}
    tids = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev
            if e.get("ph") == "M" and e.get("name") == "thread_name"}
    tpu0 = [p for p, n in pids.items() if n.endswith("TPU:0")]
    mods = [e["dur"] for e in ev
            if e.get("ph") == "X" and e["pid"] in tpu0
            and "Module" in tids.get((e["pid"], e["tid"]), "")]
    if top_ops:
        tot = {}
        for e in ev:
            if (e.get("ph") == "X" and e["pid"] in tpu0
                    and "Ops" in tids.get((e["pid"], e["tid"]), "")):
                key = re.sub(r"\.\d+", "", e["name"])  # one entry per op kind
                tot[key] = tot.get(key, 0.0) + e["dur"] / reps / LAYERS
        top = sorted(tot.items(), key=lambda kv: -kv[1])[:top_ops]
        print(f"OPS {name} (us per layer): " + "; ".join(f"{k[:60]} {v:.2f}" for k, v in top))
    if not mods:
        names = sorted(set(pids.values()))[:20]
        threads = sorted(set(tids.values()))[:30]
        raise RuntimeError(f"no module spans on TPU:0; processes {names} threads {threads}")
    return float(np.median(mods))


def run_case(tp, T, ctx, check, top_ops=0):
    devs = np.array(jax.devices()[:tp])
    mesh = Mesh(devs, ("tp",))
    n_req = T // 8 if T >= 8 else 1
    n_req = max(1, min(T // VQ, n_req))
    ctxs = [ctx] * n_req
    x = inputs(T, ctxs, jax.random.PRNGKey(T + ctx))
    h = jnp.ones((8, 7168), jnp.bfloat16) * 0.01
    wo = jnp.ones((7168, max(OTHER_N, 1)), jnp.bfloat16) * 1e-4
    if check:
        outs = {}
        for v in ("unsplit", "split_own", "split_all"):
            print(f"... check tp{tp} T{T} {ctx // 1000}k {v}", flush=True)
            f = jax.jit(make_step(v, x, tp, mesh, layers=1))
            outs[v] = np.asarray(f(x["w"], h, wo)[2])
        n_rows = n_req * VQ
        for v in ("split_own", "split_all"):
            same = np.array_equal(outs[v][:n_rows], outs["unsplit"][:n_rows])
            print(f"EXACT tp{tp} T{T} {ctx // 1000}k {v}: {'PASS' if same else 'FAIL'}")
    res = {}
    for v in ("none", "unsplit", "split_all", "split_own"):
        print(f"... timing tp{tp} T{T} {ctx // 1000}k {v}", flush=True)
        res[v] = per_step_us(
            make_step(v, x, tp, mesh), (x["w"], h, wo), v, top_ops=top_ops
        ) / LAYERS
    base = res["none"]
    print(
        f"TIME tp{tp} T{T} {n_req}x{ctx // 1000}k per layer: other-only {base:.2f}  "
        + "  ".join(f"{v} +{res[v] - base:.2f}" for v in ("unsplit", "split_own", "split_all"))
        + " us"
    )


def main():
    if os.environ.get("BREAKDOWN"):
        run_case(8, 8, 480_000, False, top_ops=40)
        return
    run_case(8, 8, 250_000, True)
    for tp, Ts in ((8, (8, 16, 32)), (4, (16, 32))):
        for T in Ts:
            for ctx in (80_000, 250_000, 480_000):
                run_case(tp, T, ctx, ctx == 250_000 and (tp, T) != (8, 8))


if __name__ == "__main__":
    main()
