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
"""Microbenchmark: GDN v3 prefill with and without the prefill checkpoint.

Times the pooled single-chip kernel (`run_jax_gdn_attention_pooled_local`,
i.e. `fused_conv1d_gdn` over the unified pool) and, when enough TPU devices
are present, the fused PCP prefill (`fused_qkvz_projection_pcp_gdn`) on the
Qwen3.5-397B GDN geometry for

  * one 9455-token request (checkpoint at 9216, column 2 of block size 3072),
  * two 9455-token requests in one batch (the second starts at flat row
    9455, so its checkpoint row 18671 is not tile aligned).

"off" passes no checkpoint arrays (the exact pre-feature schedule); "on"
passes them with the checkpoint active, which is what the model runner does
with TPU_GDN_PREFILL_CHECKPOINT=1 during the final prefill chunk. Each arm
reports the best of several runs of one jitted call. The reported step
count per token bucket also shows the padding the runner's token buckets add
when a 9455-token chunk replaces a 9216 + 239 split (local tokens under
PCP8: 1182 -> padded to the next bucket), which is a cost outside this
kernel.

Usage (on a TPU host, from the repo root):

    python scripts/benchmark_gdn_prefill_checkpoint.py --pcp-size 8
    python scripts/benchmark_gdn_prefill_checkpoint.py --skip-pcp
"""

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
    pcp_local_token_counts,
)
from vllm_torchtpu.layers.core import gdn_attention

N_KQ, N_V = 16, 64
D_K, D_V = 128, 128
KERNEL_SIZE = 4
QKV_DIM = N_KQ * D_K * 2 + N_V * D_V
HIDDEN_SIZE = 4096
BLOCK = 3072
PROMPT = 9455
CKPT_OFFSET = 9216  # get_mamba_prefill_checkpoint_position(9455, 3072)
POOL_KERNEL_BLOCK_TOKENS = 256
REPEATS = 5


def _case_arrays(num_reqs: int, *, with_ckpt: bool, max_num_seqs: int):
    lengths = [PROMPT] * num_reqs
    qsl = np.zeros((max_num_seqs + 1,), dtype=np.int32)
    qsl[1 : num_reqs + 1] = np.cumsum(lengths)
    qsl[num_reqs + 1 :] = qsl[num_reqs]
    seq_lens = np.zeros((max_num_seqs,), dtype=np.int32)
    seq_lens[:num_reqs] = lengths
    # Request i owns manager blocks [1 + 4i, 1 + 4i + 4): running state in
    # column 3, checkpoint column 2.
    state_indices = np.zeros((max_num_seqs,), dtype=np.int32)
    ckpt_blocks = np.zeros((max_num_seqs,), dtype=np.int32)
    ckpt_offsets = np.zeros((max_num_seqs,), dtype=np.int32)
    for i in range(num_reqs):
        state_indices[i] = 1 + 4 * i + 3
        ckpt_blocks[i] = 1 + 4 * i + 2
        ckpt_offsets[i] = CKPT_OFFSET if with_ckpt else 0
    distribution = np.asarray([0, num_reqs, num_reqs], dtype=np.int32)
    return (
        jnp.asarray(qsl),
        jnp.asarray(seq_lens),
        jnp.asarray(state_indices),
        jnp.asarray(distribution),
        (jnp.asarray(ckpt_offsets), jnp.asarray(ckpt_blocks)) if with_ckpt else None,
    )


def _best_of(fn, make_args, repeats=REPEATS):
    best = None
    for _ in range(repeats):
        args = make_args()
        start = time.perf_counter()
        out = fn(*args)
        jax.tree.map(lambda x: x.block_until_ready(), out)
        best = min(best or 1e9, time.perf_counter() - start)
    return best


def bench_pooled_local(num_reqs: int, with_ckpt: bool) -> float:
    """Single-chip pooled kernel (the non-PCP production path)."""
    num_tokens = PROMPT * num_reqs
    max_num_seqs = 8
    keys = iter(jax.random.split(jax.random.key(1), 8))
    scale = jnp.asarray(0.05, dtype=jnp.bfloat16)
    qkv = jax.random.normal(next(keys), (num_tokens, QKV_DIM), jnp.bfloat16) * scale
    b = jax.random.normal(next(keys), (num_tokens, N_V), jnp.bfloat16) * scale
    a = jax.random.normal(next(keys), (num_tokens, N_V), jnp.bfloat16) * scale
    conv_weight = jax.random.normal(next(keys), (QKV_DIM, 1, KERNEL_SIZE), jnp.bfloat16)
    conv_bias = jax.random.normal(next(keys), (QKV_DIM,), jnp.bfloat16)
    a_log = jax.random.normal(next(keys), (N_V,), jnp.float32)
    dt_bias = jax.random.normal(next(keys), (N_V,), jnp.float32) - 2.0
    qsl, seq_lens, state_indices, distribution, ckpt = _case_arrays(
        num_reqs, with_ckpt=with_ckpt, max_num_seqs=max_num_seqs
    )
    num_blocks = 1 + 4 * max_num_seqs
    pool_split = BLOCK // POOL_KERNEL_BLOCK_TOKENS
    pool_shape = (num_blocks * pool_split, POOL_KERNEL_BLOCK_TOKENS, 1, 2 * N_V, 128)

    @functools.partial(jax.jit, donate_argnums=(0,))
    def run(pool, qkv, b, a, offsets, blocks):
        return gdn_attention.run_jax_gdn_attention_pooled_local(
            qkv,
            b,
            a,
            pool,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            qsl,
            state_indices,
            distribution,
            seq_lens,
            None,
            None,
            offsets,
            blocks,
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D_K,
            d_v=D_V,
            kernel_size=KERNEL_SIZE,
            pool_block_tokens=BLOCK,
        )

    offsets, blocks = ckpt if ckpt is not None else (None, None)

    def make_args():
        pool = jnp.zeros(pool_shape, dtype=jnp.float8_e4m3fn).block_until_ready()
        return pool, qkv, b, a, offsets, blocks

    jax.tree.map(lambda x: x.block_until_ready(), run(*make_args()))
    return _best_of(run, make_args)


def bench_fused_pcp(num_reqs: int, with_ckpt: bool, *, pcp_size: int, cc: int):
    """Fused QKVZ-projection PCP prefill (the pooled PCP production path)."""
    lengths = np.asarray([PROMPT] * num_reqs, dtype=np.int32)
    num_tokens = int(lengths.sum())
    max_num_seqs = 8
    keys = iter(jax.random.split(jax.random.key(2), 8))
    hidden = jax.random.normal(next(keys), (num_tokens, HIDDEN_SIZE), jnp.bfloat16)
    qkvz_weight = (
        0.02 * jax.random.normal(next(keys), (QKV_DIM + N_V * D_V, HIDDEN_SIZE))
    ).astype(jnp.bfloat16)
    b = jax.random.normal(next(keys), (num_tokens, N_V), jnp.bfloat16)
    a = jax.random.normal(next(keys), (num_tokens, N_V), jnp.bfloat16)
    conv_weight = jax.random.normal(next(keys), (QKV_DIM, 1, KERNEL_SIZE), jnp.bfloat16)
    conv_bias = jax.random.normal(next(keys), (QKV_DIM,), jnp.bfloat16)
    a_log = jax.random.normal(next(keys), (N_V,), jnp.float32)
    dt_bias = jax.random.normal(next(keys), (N_V,), jnp.float32)
    qsl, seq_lens, state_indices, distribution, ckpt = _case_arrays(
        num_reqs, with_ckpt=with_ckpt, max_num_seqs=max_num_seqs
    )

    owner_starts = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.int32)
    local_required = int(
        pcp_local_token_counts(
            lengths, pcp_size, cc, token_owner_start_offsets_per_req=owner_starts
        ).max()
    )
    projection_block = 2 * cc
    local_padded = max(
        8 * projection_block,
        -(-local_required // projection_block) * projection_block,
    )
    padded_num_tokens = local_padded * pcp_size
    token_order, _ = build_pcp_rank_major_token_order(
        lengths,
        pcp_size,
        cc,
        padded_num_tokens,
        token_owner_start_offsets_per_req=owner_starts,
    )
    valid = token_order >= 0
    valid_rows = np.where(valid)[0]
    source_rows = token_order[valid]

    def pack(tensor):
        padded = jnp.pad(tensor, ((0, padded_num_tokens - num_tokens), (0, 0)))
        return jnp.zeros_like(padded).at[valid_rows].set(padded[source_rows])

    mesh = Mesh(np.asarray(jax.devices()[:pcp_size]), ("pcp",))

    def shard_tokens(t):
        return jax.device_put(t, NamedSharding(mesh, P("pcp", None)))

    def replicate(t):
        return jax.device_put(t, NamedSharding(mesh, P()))

    num_blocks = 1 + 4 * max_num_seqs
    pool_split = BLOCK // POOL_KERNEL_BLOCK_TOKENS
    local_pool_shape = (
        num_blocks * pool_split,
        POOL_KERNEL_BLOCK_TOKENS,
        1,
        max(4, 2 * (N_V // pcp_size)),
        128,
    )
    global_pool_shape = (pcp_size * local_pool_shape[0], *local_pool_shape[1:])
    packed_hidden = shard_tokens(pack(hidden))
    packed_b = shard_tokens(pack(b))
    packed_a = shard_tokens(pack(a))
    rep = {
        "w": replicate(qkvz_weight),
        "cw": replicate(conv_weight),
        "cb": replicate(conv_bias),
        "al": replicate(a_log),
        "db": replicate(dt_bias),
        "si": replicate(state_indices),
        "qsl": replicate(qsl),
        "dist": replicate(distribution),
        "sl": replicate(seq_lens),
    }
    offsets, blocks = (
        (replicate(ckpt[0]), replicate(ckpt[1])) if ckpt is not None else (None, None)
    )

    @functools.partial(jax.jit, donate_argnums=(0,))
    def run(pool, hidden_, b_, a_, offsets_, blocks_):
        return gdn_attention.run_jax_gdn_attention_pooled_pcp_prefill_projection(
            hidden_,
            rep["w"],
            None,
            b_,
            a_,
            pool,
            rep["cw"],
            rep["cb"],
            rep["al"],
            rep["db"],
            rep["si"],
            rep["qsl"],
            rep["dist"],
            rep["sl"],
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D_K,
            d_v=D_V,
            kernel_size=KERNEL_SIZE,
            pool_block_tokens=BLOCK,
            pcp_size=pcp_size,
            interleave_size=cc,
            mesh=mesh,
            prefill_ckpt_offsets=offsets_,
            prefill_ckpt_blocks=blocks_,
        )

    def make_args():
        pool = jax.device_put(
            jnp.zeros(global_pool_shape, dtype=jnp.float8_e4m3fn),
            NamedSharding(mesh, P("pcp")),
        ).block_until_ready()
        return pool, packed_hidden, packed_b, packed_a, offsets, blocks

    jax.tree.map(lambda x: x.block_until_ready(), run(*make_args()))
    return _best_of(run, make_args), local_padded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcp-size", type=int, default=8)
    parser.add_argument("--interleave-size", type=int, default=512)
    parser.add_argument("--skip-pcp", action="store_true")
    parser.add_argument("--skip-local", action="store_true")
    args = parser.parse_args()

    print(f"devices={len(jax.devices())} prompt={PROMPT} ckpt_offset={CKPT_OFFSET}")
    if not args.skip_local:
        for num_reqs in (1, 2):
            off = bench_pooled_local(num_reqs, with_ckpt=False)
            on = bench_pooled_local(num_reqs, with_ckpt=True)
            print(
                f"pooled_local reqs={num_reqs}: off {off * 1e3:.3f} ms, "
                f"on {on * 1e3:.3f} ms ({on / off:.3f}x)"
            )
    if not args.skip_pcp:
        if len(jax.devices()) < args.pcp_size:
            print(f"skipping fused PCP: need {args.pcp_size} devices")
            return
        for num_reqs in (1, 2):
            off, local_padded = bench_fused_pcp(
                num_reqs,
                with_ckpt=False,
                pcp_size=args.pcp_size,
                cc=args.interleave_size,
            )
            on, _ = bench_fused_pcp(
                num_reqs,
                with_ckpt=True,
                pcp_size=args.pcp_size,
                cc=args.interleave_size,
            )
            print(
                f"fused_pcp{args.pcp_size} reqs={num_reqs} local_padded_tokens="
                f"{local_padded}: off {off * 1e3:.3f} ms, on {on * 1e3:.3f} ms "
                f"({on / off:.3f}x)"
            )


if __name__ == "__main__":
    main()
