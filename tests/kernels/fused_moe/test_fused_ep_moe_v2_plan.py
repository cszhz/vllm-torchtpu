# SPDX-License-Identifier: Apache-2.0
"""The one-pass plan tables equal the builders they replace, element for element.

CPU contract tests: the tables are integer index arithmetic, so a wrong
identity is silent at rank 0 and on a uniform routing. Both geometries that
ship are covered -- the DP32 C768 decode shape (ep 32, 13 slots a shard, 144
local tokens) and the 2048-row prefill bucket -- at every shard, because
`shard_small_tables` selects a shard by mask where the builder slices.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2.host import (
    GatheredPairs,
    build_routing_tables_sharded,
    expert_visit_list,
    pair_block_hist,
    ragged_stride_bound,
    routing_block,
    shard_count_vector,
    shard_expert_slabs,
    shard_push_tables_in_rows,
    shard_transport_tables_in_blocks,
)
from vllm_torchtpu.kernels.fused_moe.v2.plan_kernel import (
    shard_pair_tables,
    shard_small_tables,
)

CAPACITY = 128

# (ep, local tokens, top-k, slots a shard) -- the DP32 decode shape, the
# prefill bucket at the same expert layout, and two small shards that exercise
# a different block width and an empty expert.
GEOMETRIES = [
    (32, 144, 6, 13),
    (32, 2048, 6, 13),
    (8, 144, 6, 13),
    (4, 32, 2, 3),
]


def _case(ep, t_local, topk, g_local, seed):
    e_total = ep * g_local
    rng = np.random.default_rng(seed)
    weights = rng.gamma(2.0, 1.0, size=e_total)
    idx = np.stack([
        rng.choice(e_total, size=topk, replace=False, p=weights / weights.sum())
        for _ in range(ep * t_local)
    ]).astype(np.int32)
    counts = np.stack([
        np.bincount(idx[s * t_local:(s + 1) * t_local].reshape(-1), minlength=e_total)
        for s in range(ep)
    ]).astype(np.int32)
    return idx, counts


def _reference(idx, counts, me, *, ep, t_local, topk, g_local):
    e_total = ep * g_local
    block = routing_block(t_local, topk)
    n_blocks = idx.size // block
    expert_blocks = jnp.asarray(idx.reshape(n_blocks, block))
    per_dest = t_local * topk // block
    my_blocks = expert_blocks[me * per_dest:(me + 1) * per_dest]
    routing = build_routing_tables_sharded(
        None,
        jnp.int32(me),
        gathered=GatheredPairs(
            expert_blocks=expert_blocks,
            local_blocks=my_blocks,
            block_hist=pair_block_hist(my_blocks, e_total),
            rows_by_dest=jnp.asarray(counts),
            n_tokens=ep * t_local,
            topk=topk,
        ),
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=block,
        tile_m=CAPACITY,
        shard_stride=ragged_stride_bound(ep * t_local, topk, e_total, CAPACITY),
    )
    bt = shard_transport_tables_in_blocks(routing, jnp.int32(me), e_total=e_total, ep=ep)
    rt = shard_push_tables_in_rows(routing, jnp.int32(me), e_total=e_total, ep=ep)
    expert_rows, slab_base = shard_expert_slabs(routing, jnp.int32(me), e_total=e_total, ep=ep)
    visit, _ = expert_visit_list(expert_rows, g_local)
    return routing, dict(
        kernel_tables=(*bt[:3], bt[5], rt[0]),
        expert_rows=expert_rows,
        slab_base=slab_base,
        visit=visit,
        counts=shard_count_vector(
            routing, expert_rows, jnp.int32(me), e_total=e_total, ep=ep
        ),
    )


@pytest.mark.parametrize("ep,t_local,topk,g_local", GEOMETRIES)
def test_small_tables_match_the_builders(ep, t_local, topk, g_local):
    e_total = ep * g_local
    stride = ragged_stride_bound(ep * t_local, topk, e_total, CAPACITY)
    idx, counts = _case(ep, t_local, topk, g_local, seed=ep * 1000 + t_local)
    for me in range(ep):
        routing, want = _reference(
            idx, counts, me, ep=ep, t_local=t_local, topk=topk, g_local=g_local
        )
        got = jax.jit(
            shard_small_tables,
            static_argnames=("e_total", "ep", "shard_stride", "tile_m"),
        )(
            jnp.asarray(counts),
            jnp.int32(me),
            e_total=e_total,
            ep=ep,
            shard_stride=stride,
            tile_m=CAPACITY,
        )
        for a, b in zip(got.kernel_tables, want["kernel_tables"]):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        for name in ("expert_rows", "slab_base", "visit", "counts"):
            np.testing.assert_array_equal(
                np.asarray(getattr(got, name)), np.asarray(want[name])
            )

        # The per-pair pass's three inputs, against the slices it replaces.
        first = me * g_local
        np.testing.assert_array_equal(
            np.asarray(got.recv_base_e), np.asarray(routing.recv_base[me].reshape(e_total))
        )
        run_base = np.asarray(routing.expert_base)[first:first + g_local][None, :] + np.asarray(
            routing.run_start_aligned
        )[first:first + g_local, :].T
        np.testing.assert_array_equal(np.asarray(got.run_base), run_base)
        shift = int(np.log2(CAPACITY))
        mirror = (
            (np.asarray(routing.recv_base[me].reshape(e_total)) >> shift)
            + 2 * np.arange(e_total, dtype=np.int32)
        ) * CAPACITY + (np.asarray(routing.run_start_aligned)[:, me] & (CAPACITY - 1))
        np.testing.assert_array_equal(np.asarray(got.mirror_base_e), mirror)


@pytest.mark.parametrize("ep,t_local,topk,g_local", GEOMETRIES)
def test_pair_tables_match_the_builder(ep, t_local, topk, g_local):
    e_total = ep * g_local
    stride = ragged_stride_bound(ep * t_local, topk, e_total, CAPACITY)
    block = routing_block(t_local, topk)
    per_dest = t_local * topk // block
    idx, counts = _case(ep, t_local, topk, g_local, seed=ep * 97 + t_local)
    expert_blocks = jnp.asarray(idx.reshape(idx.size // block, block))
    for me in (0, 1, ep - 1):
        routing, _ = _reference(
            idx, counts, me, ep=ep, t_local=t_local, topk=topk, g_local=g_local
        )
        my_blocks = expert_blocks[me * per_dest:(me + 1) * per_dest]
        small = shard_small_tables(
            jnp.asarray(counts), jnp.int32(me), e_total=e_total, ep=ep,
            shard_stride=stride, tile_m=CAPACITY,
        )
        pos, mirror_pos, slab_row = jax.jit(
            shard_pair_tables,
            static_argnames=("e_total", "ep", "t_local", "topk", "block", "shard_stride"),
        )(
            expert_blocks, my_blocks, pair_block_hist(my_blocks, e_total), small,
            jnp.int32(me), e_total=e_total, ep=ep, t_local=t_local, topk=topk,
            block=block, shard_stride=stride,
        )
        np.testing.assert_array_equal(np.asarray(pos), np.asarray(routing.pos))
        np.testing.assert_array_equal(
            np.asarray(mirror_pos), np.asarray(routing.mirror_pos)
        )
        np.testing.assert_array_equal(
            np.asarray(slab_row), np.asarray(routing.slab_row)
        )
