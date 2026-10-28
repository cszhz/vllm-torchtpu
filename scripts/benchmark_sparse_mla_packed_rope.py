# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Microbenchmark: `TPU_SPARSE_MLA_PACKED_ROPE` off vs on, one sparse MLA layer.

Times the standalone non-DCP `sparse_mla_attention` entry point (cache write
+ SparseCore gather + attention kernel) at GLM-5.3's serving geometry, with
the padded 128 B/token fp8 RoPE cache (640 B/token latent) against the packed
64 B/token one (576 B/token), and checks the two produce bit-identical
outputs: the caches hold the same fp8 bytes, only the rope rows are packed.

Token buckets `T <= 64` are decode steps (`T` one-token sequences over a
3,071-token context); larger buckets are one sequence's prefill chunk at a
2,048-token prefix. Every token has a full `index_topk` selection, so the
gather moves the maximum bytes each bucket can ask for.

Run (SparseCore layouts, split-q and the 128-token chunk as the GLM-5.3 config):

    python scripts/benchmark_sparse_mla_packed_rope.py
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from pathlib import Path

# The serving config under test; the caller's environment wins.
os.environ.setdefault("TPU_MLA_SPARSE_SPLIT_Q", "1")
os.environ.setdefault("TPU_SPARSE_MLA_CHUNK_TOKENS", "128")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vllm_torchtpu.kernels.mla import kv_cache_utils  # noqa: E402
from vllm_torchtpu.kernels.mla.kv_cache_utils import (  # noqa: E402
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
from vllm_torchtpu.layers.core import attention_interface  # noqa: E402

# GLM-5.3 at TP=1: every rank carries all 64 heads.
NUM_HEADS = 64
LKV_DIM = 512  # kv_lora_rank
ROPE_DIM = 64  # qk_rope_head_dim
TOPK = 2048  # index_topk
PAGE_SIZE = 1024
NUM_PAGES = 229  # tpu7x-8, gpu_memory_utilization=0.95, 640 B/token
SPARSE_LAYERS = 75
# Per sequence, so 64 decode sequences fit the 229-page cache.
PAGES_PER_SEQ = 3
DECODE_CONTEXT = PAGES_PER_SEQ * PAGE_SIZE - 1  # kv_len after the decode token
PREFILL_PREFIX = 2 * PAGE_SIZE  # a prefill chunk starts after this many tokens
BUCKETS = (16, 32, 64, 256, 512, 1024)
DECODE_MAX = 64


def _spec(cache_type, head_dim, packed_rope=False):
    return SparseMLAKVCacheSpec.create(
        cache_type,
        KVCacheLayout.SPARSECORE,
        NUM_PAGES,
        PAGE_SIZE,
        head_dim,
        kv_cache_utils.get_dtype_packing(jnp.float8_e4m3fn),
        packed_rope=packed_rope,
    )


def _topk_rows(rng, positions):
    """Full `TOPK` selection per token from `[0, pos]` (pos + 1 >= TOPK)."""
    rows = np.empty((len(positions), TOPK), np.int32)
    for i, pos in enumerate(positions):
        assert pos + 1 >= TOPK, pos
        rows[i] = np.sort(rng.choice(pos + 1, size=TOPK, replace=False))
    return rows


def _case(num_tokens: int, rng):
    """Inputs for one bucket: `T` decodes, or one `T`-token prefill chunk."""
    if num_tokens <= DECODE_MAX:
        num_seqs = num_tokens
        seq_lens = [DECODE_CONTEXT + 1] * num_seqs
        query_start_loc = list(range(num_seqs + 1))
        distribution = [num_seqs, num_seqs, num_seqs]
        positions = [DECODE_CONTEXT] * num_seqs
    else:
        num_seqs = 1
        seq_lens = [PREFILL_PREFIX + num_tokens]
        query_start_loc = [0, num_tokens]
        distribution = [0, 1, 1]
        positions = list(range(PREFILL_PREFIX, PREFILL_PREFIX + num_tokens))
    assert num_seqs * PAGES_PER_SEQ <= NUM_PAGES
    block_tables = rng.permutation(NUM_PAGES)[: num_seqs * PAGES_PER_SEQ]
    f32 = np.float32
    fp8 = jnp.float8_e4m3fn
    return dict(
        ql_nope=jnp.asarray(
            rng.standard_normal((num_tokens, NUM_HEADS, LKV_DIM)).astype(f32),
            jnp.bfloat16,
        ),
        q_pe=jnp.asarray(
            rng.standard_normal((num_tokens, NUM_HEADS, ROPE_DIM)).astype(f32),
            jnp.bfloat16,
        ),
        kv_c=jnp.asarray(rng.standard_normal((num_tokens, LKV_DIM)), f32).astype(fp8),
        k_pe=jnp.asarray(rng.standard_normal((num_tokens, ROPE_DIM)), f32).astype(fp8),
        topk=jnp.asarray(_topk_rows(rng, positions)),
        seq_lens=jnp.asarray(seq_lens, jnp.int32),
        block_tables=jnp.asarray(block_tables.astype(np.int32)),
        query_start_loc=jnp.asarray(query_start_loc, jnp.int32),
        distribution=jnp.asarray(distribution, jnp.int32),
    )


def _build(mesh, nope_spec, rope_spec):
    """One compiled step: write this step's latents, gather, attend."""
    sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)

    def step(ql_nope, q_pe, kv_c, k_pe, nope, rope, topk, lens, pages, starts, dist):
        return attention_interface.sparse_mla_attention(
            ql_nope,
            q_pe,
            kv_c,
            k_pe,
            nope,
            rope,
            topk,
            lens,
            pages,
            starts,
            dist,
            mesh,
            nope_spec,
            rope_spec,
            sm_scale=sm_scale,
            k_scale=1.0,
        )

    # The caches are carried step to step in place, as the runner does.
    return jax.jit(step, donate_argnums=(4, 5))


class _Arm:
    def __init__(self, name, mesh, nope_spec, rope_spec):
        self.name = name
        self.step = _build(mesh, nope_spec, rope_spec)
        self.caches = (
            jnp.zeros(nope_spec.shape, nope_spec.jax_dtype),
            jnp.zeros(rope_spec.shape, rope_spec.jax_dtype),
        )
        self.cache_bytes = sum(c.nbytes for c in self.caches)

    def run(self, c):
        nope, rope, out = self.step(
            c["ql_nope"],
            c["q_pe"],
            c["kv_c"],
            c["k_pe"],
            *self.caches,
            c["topk"],
            c["seq_lens"],
            c["block_tables"],
            c["query_start_loc"],
            c["distribution"],
        )
        self.caches = (nope, rope)
        return out


def _time_pair(arms, c, warmup, iters):
    """Paired medians (us) with alternating arm order against TPU drift."""
    for _ in range(warmup):
        for arm in arms:
            jax.block_until_ready(arm.run(c))
    samples = {arm.name: [] for arm in arms}
    for i in range(iters):
        order = arms if i % 2 == 0 else arms[::-1]
        for arm in order:
            start = time.perf_counter()
            jax.block_until_ready(arm.run(c))
            samples[arm.name].append((time.perf_counter() - start) * 1e6)
    return {name: statistics.median(v) for name, v in samples.items()}


def _headroom(num_pages, padded_bytes, packed_bytes, extra_bytes=0):
    """Pages at the same HBM budget once a token costs `packed` not `padded`."""
    budget = num_pages * PAGE_SIZE * (padded_bytes + extra_bytes)
    return budget // (PAGE_SIZE * (packed_bytes + extra_bytes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--buckets", type=int, nargs="*", default=list(BUCKETS))
    args = parser.parse_args()

    mesh = Mesh(np.asarray(jax.local_devices()[:1]), ("model",))
    nope_spec = _spec(KVCacheType.NOPE, LKV_DIM)
    padded_spec = _spec(KVCacheType.ROPE, ROPE_DIM)
    packed_spec = _spec(KVCacheType.ROPE, ROPE_DIM, packed_rope=True)
    arms = [
        _Arm("off", mesh, nope_spec, padded_spec),
        _Arm("on", mesh, nope_spec, packed_spec),
    ]
    tokens = NUM_PAGES * PAGE_SIZE
    off_bpt, on_bpt = (arm.cache_bytes // tokens for arm in arms)
    print(
        f"GLM-5.3 sparse MLA layer: {NUM_HEADS} heads, lkv {LKV_DIM}, rope "
        f"{ROPE_DIM}, topk {TOPK}, page {PAGE_SIZE}, {NUM_PAGES} pages, fp8 "
        f"SparseCore caches; TPU_MLA_SPARSE_SPLIT_Q="
        f"{os.environ['TPU_MLA_SPARSE_SPLIT_Q']} TPU_SPARSE_MLA_CHUNK_TOKENS="
        f"{os.environ['TPU_SPARSE_MLA_CHUNK_TOKENS']}"
    )
    print(
        f"latent cache: {off_bpt} B/token (padded rope) -> {on_bpt} B/token "
        f"(packed rope), {arms[0].cache_bytes / 2**20:.1f} MiB -> "
        f"{arms[1].cache_bytes / 2**20:.1f} MiB per layer"
    )
    print()
    header = (
        f"{'T':>5} {'shape':>8} {'off us':>9} {'on us':>9} {'delta us':>9} "
        f"{'delta %':>8} {'off ms/step':>12} {'on ms/step':>11} {'d ms/step':>10}"
    )
    print(header)
    print("-" * len(header))
    rng = np.random.default_rng(0)
    for t in args.buckets:
        c = _case(t, rng)
        # Same inputs, same written bytes: the outputs must match exactly.
        outs = [np.asarray(jax.block_until_ready(arm.run(c))) for arm in arms]
        np.testing.assert_array_equal(outs[1], outs[0], err_msg=f"T={t}")
        us = _time_pair(arms, c, args.warmup, args.iters)
        off, on = us["off"], us["on"]
        scale = SPARSE_LAYERS / 1e3
        print(
            f"{t:>5} {'decode' if t <= DECODE_MAX else 'prefill':>8} "
            f"{off:>9.1f} {on:>9.1f} {on - off:>+9.1f} "
            f"{100 * (on - off) / off:>+7.1f}% {off * scale:>12.3f} "
            f"{on * scale:>11.3f} {(on - off) * scale:>+10.3f}"
        )
    print()
    print(f"ms/step = us/call x {SPARSE_LAYERS} sparse layers; outputs bit-identical.")
    print()
    for label, extra in (("latent only", 0), ("latent + 132 B/token indexer", 132)):
        pages = _headroom(NUM_PAGES, off_bpt, on_bpt, extra)
        print(
            f"KV pages at the same HBM budget ({label}): {NUM_PAGES} -> {pages} "
            f"(+{pages - NUM_PAGES}, {100 * (pages / NUM_PAGES - 1):+.1f}%)"
        )


if __name__ == "__main__":
    main()
