# SPDX-License-Identifier: Apache-2.0
"""The routing plan's small tables, built in one pass over the count table.

Every table the expert kernel takes that is not per-pair is a function of the
exchanged ``[ep, e_total]`` count table alone. `build_routing_tables_sharded`
plus the four ``shard_*`` builders compute them as ~35 separate XLA ops --
cumsums that lower to reduce-windows, transposes that relayout, and dynamic
slices that each copy a row into scalar memory. At the DP32 decode shape that
is ~10 us a layer on 13 312 int32s: launch and layout cost, not arithmetic.

This module computes the same values in one pass, with two changes that remove
the op count rather than the work:

* a prefix sum is a strictly-lower-triangular select-sum, not `jnp.cumsum`
  (which lowers to a reduce-window, a module of its own -- the reasoning
  `_window_table` already carries);
* a shard's own slice is a mask-and-reduce, not `lax.dynamic_slice`, so
  nothing moves through scalar memory to index it.

Integer adds are exact and associative, so every value here is bit-identical
to the builder it replaces; `tests/kernels/fused_moe/test_fused_ep_moe_v2_plan.py`
asserts that element for element.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.fused_moe.v2.host import (
    N_COUNTS,
    ROWBLK,
    ROWBLK_SHIFT,
    _lookup_per_pair,
    _lookup_per_sub_pair,
    _over_sub_blocks,
    _rank_within_block,
    _rank_within_window,
    _window_hist,
    _window_table,
    expert_visit_list,
    mirror_run_tiles,
    off_shard_slab_row,
    pow2_shift,
    rank_window,
)


def _exclusive_prefix(x, axis):
    """``cumsum(x, axis) - x`` over a small triangle, for the kernel body.

    Inside one Pallas program this is a handful of vector ops; `jnp.cumsum`
    there would have to lower a reduce-window. Measured the other way round
    in XLA, where the triangle's padded intermediates cost 4 us a call at the
    DP32 shape -- so the per-pair pass below keeps `jnp.cumsum`.
    """
    n = x.shape[axis]
    y = jnp.moveaxis(x, axis, -1)
    lower = jnp.arange(n, dtype=jnp.int32)[None, :] < jnp.arange(n, dtype=jnp.int32)[:, None]
    y = jnp.sum(jnp.where(lower, y[..., None, :], 0), axis=-1)
    return jnp.moveaxis(y, -1, axis)


class SmallTables(NamedTuple):
    """What one shard needs from the count table, and nothing else.

    ``kernel_tables`` is the five-tuple `compute_local` hands the expert
    kernel. ``recv_base_e`` and ``mirror_base_e`` are the per-expert halves of
    the arrival and mirror lookups the per-pair pass adds its own offsets to,
    and ``run_base`` is the slab-row pass's per (source, local expert) base.
    """

    kernel_tables: tuple
    expert_rows: jax.Array
    slab_base: jax.Array
    visit: jax.Array
    counts: jax.Array
    recv_base_e: jax.Array
    mirror_base_e: jax.Array
    run_base: jax.Array


def shard_small_tables(rows_by_dest, me, *, e_total, ep, shard_stride, tile_m,
                       groups=0, pallas=False):
    """Shard ``me``'s non-per-pair plan tables from the ``[ep, e_total]`` counts.

    ``pallas`` runs the same body as one Pallas kernel. The arithmetic is a
    few dozen vector ops on 13 312 int32s; spelled as XLA it is ~35 separate
    programs, and the launch cost of those is what this costs today.
    """
    if pallas:
        return _as_pallas(
            rows_by_dest, me, e_total=e_total, ep=ep, shard_stride=shard_stride,
            tile_m=tile_m,
        )
    g_local = e_total // ep
    counts_se = rows_by_dest.astype(jnp.int32)  # [source, expert]
    run_rows = counts_se.T  # [E, source]
    aligned = (run_rows + (ROWBLK - 1)) & jnp.int32(-ROWBLK)
    run_start = _exclusive_prefix(aligned, axis=1)
    expert_rows_aligned = aligned.sum(axis=1)  # [E]
    rows_by_shard = expert_rows_aligned.reshape(ep, g_local)
    expert_base = (
        _exclusive_prefix(rows_by_shard, axis=1)
        + jnp.arange(ep, dtype=jnp.int32)[:, None] * jnp.int32(shard_stride)
    )  # [owner, j]
    region = aligned.reshape(ep, g_local, ep)  # [owner, j, source]
    by_source = region.transpose(2, 0, 1)  # [source, owner, j]
    per_owner = by_source.sum(axis=2)  # [source, owner]
    recv_base = (
        _exclusive_prefix(by_source, axis=2)
        + _exclusive_prefix(per_owner, axis=1)[:, :, None]
    )  # [source, owner, j]
    recv_rows = per_owner.sum(axis=1)  # [source]
    run_tiles = mirror_run_tiles(run_start, aligned, tile_m)  # [E, source]

    is_me = jnp.arange(ep, dtype=jnp.int32) == me
    mine = is_me.astype(jnp.int32)

    def over_owner(x):
        """``x[me * g_local : (me + 1) * g_local]`` of an ``[E, ...]`` table."""
        owner = x.reshape((ep, g_local) + x.shape[1:])
        return jnp.sum(jnp.where(is_me.reshape((ep,) + (1,) * x.ndim), owner, 0), axis=0)

    my_aligned = over_owner(aligned)  # [g, source]
    my_true = over_owner(run_rows)
    my_start = over_owner(run_start)
    my_tiles = over_owner(run_tiles)
    my_rows = over_owner(expert_rows_aligned)  # [g]
    my_base = over_owner(expert_base.reshape(e_total))  # [g]
    slab_base = my_base - my_base[0]
    my_region = jnp.sum(jnp.where(is_me[:, None, None], region, 0), axis=0)  # [g, source]
    # recv_base[:, me, :] -- the arrival row each (local expert, source) push
    # lands on -- and recv_base[me] -- this shard's own tokens' arrival base.
    push_dst = jnp.sum(jnp.where(is_me[None, :, None], recv_base, 0), axis=1).T  # [g, src]
    recv_base_e = jnp.sum(jnp.where(is_me[:, None, None], recv_base, 0), axis=0).reshape(
        e_total
    )
    my_run_start = jnp.sum(jnp.where(mine[None, :], run_start, 0), axis=1)  # [E]

    out_total = my_region.sum(axis=0)  # [dest]
    contrib_off = _exclusive_prefix(out_total, axis=0)[None, :] + _exclusive_prefix(
        my_region, axis=0
    )

    shift = pow2_shift(tile_m, "tile_m")
    mirror_base_e = (
        jnp.right_shift(recv_base_e, shift) + 2 * jnp.arange(e_total, dtype=jnp.int32)
    ) * tile_m + (my_run_start & (tile_m - 1))

    def blocks(a):
        return jnp.right_shift(a, ROWBLK_SHIFT).astype(jnp.int32)

    kernel_tables = (
        blocks(my_start),
        blocks(my_aligned),
        blocks(contrib_off),
        blocks(push_dst),
        my_true.astype(jnp.int32),
    )
    visit, _ = expert_visit_list(my_rows, g_local)
    counts = _count_vector(
        my_true, my_aligned, my_tiles, my_rows, run_rows, run_tiles, recv_rows, mine, g_local, ep
    )
    return SmallTables(
        kernel_tables=kernel_tables,
        expert_rows=my_rows.astype(jnp.int32),
        slab_base=slab_base.astype(jnp.int32),
        visit=visit,
        counts=counts,
        recv_base_e=recv_base_e,
        mirror_base_e=mirror_base_e,
        run_base=my_base[None, :] + my_start.T,  # [source, g]
    )


def _count_vector(
    true_rows, aligned, tiles, expert_rows, run_rows, run_tiles, recv_rows, mine, g_local, ep
):
    """`shard_count_vector`'s eight rows, off tables this pass already holds."""
    other = 1 - mine
    first = (jnp.arange(g_local, dtype=jnp.int32) == 0).astype(jnp.int32)[:, None]
    recv_rows_all = run_rows.sum(axis=0).astype(jnp.int32)
    recv_tiles_all = run_tiles.sum(axis=0).astype(jnp.int32)
    active = (expert_rows > 0).astype(jnp.int32)[:, None]

    def over_experts(term):
        return term.sum(axis=0, keepdims=True)

    return jnp.concatenate(
        [
            over_experts(true_rows * other[None, :]),
            over_experts(
                (recv_rows_all * mine)[None, :] * first - true_rows * mine[None, :]
            ),
            over_experts(aligned * other[None, :]),
            over_experts(
                (recv_rows.astype(jnp.int32) * mine)[None, :] * first
                - aligned * mine[None, :]
            ),
            over_experts(active * mine[None, :]),
            over_experts(tiles * other[None, :]),
            over_experts(
                (recv_tiles_all * mine)[None, :] * first - tiles * mine[None, :]
            ),
            over_experts(tiles * mine[None, :]),
        ],
        axis=0,
    ).astype(jnp.int32)


assert N_COUNTS == 8


def shard_pair_tables(expert_blocks, my_blocks, my_hist, small, me, *, e_total, ep,
                      t_local, topk, block, shard_stride):
    """The three per-pair tables, over the bases `shard_small_tables` built.

    `pos` and `mirror_pos` are combine inputs; `slab_row` is what both slab
    scatters write through. Identical to the second half of
    `build_routing_tables_sharded`, with its per-expert bases taken from
    `small` instead of being rebuilt from the full plan.
    """
    g_local = e_total // ep
    n_blocks = expert_blocks.shape[0]
    blocks_per_dest = my_blocks.shape[0]
    bins = jnp.arange(e_total, dtype=jnp.int32)
    window = rank_window(block, blocks_per_dest * block)

    my_off = jnp.cumsum(my_hist, axis=0) - my_hist
    my_rank = _rank_within_block(my_blocks, block)
    pos = (
        _lookup_per_pair(my_off + small.recv_base_e[None, :], my_blocks, bins) + my_rank
    ).reshape(t_local, topk)
    mirror_pos = (
        _lookup_per_pair(my_off + small.mirror_base_e[None, :], my_blocks, bins) + my_rank
    ).reshape(t_local, topk)

    first = me * g_local
    my_experts = first + jnp.arange(g_local, dtype=jnp.int32)
    win_hist = _window_hist(expert_blocks, my_experts, window)
    by_source = _over_sub_blocks(win_hist).reshape(ep, blocks_per_dest, g_local)
    row_table = _window_table(
        (jnp.cumsum(by_source, axis=1) - by_source + small.run_base[:, None, :]).reshape(
            n_blocks, g_local
        ),
        win_hist,
    )
    on_me = (expert_blocks >= first) & (expert_blocks < first + g_local)
    my_row = _lookup_per_sub_pair(
        row_table, expert_blocks, my_experts, window
    ) + _rank_within_window(expert_blocks, window)
    slab_row = jnp.where(
        on_me, my_row, jnp.int32(off_shard_slab_row(shard_stride, ep))
    ).reshape(-1)
    return pos, mirror_pos, slab_row


def _as_pallas(rows_by_dest, me, *, e_total, ep, shard_stride, tile_m):
    """`shard_small_tables` as one kernel: one launch instead of ~35."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    g_local = e_total // ep
    i32 = jnp.int32
    shapes = (
        (jax.ShapeDtypeStruct((g_local, ep), i32),) * 5
        + (
            jax.ShapeDtypeStruct((g_local,), i32),
            jax.ShapeDtypeStruct((g_local,), i32),
            jax.ShapeDtypeStruct((g_local,), i32),
            jax.ShapeDtypeStruct((N_COUNTS, ep), i32),
            jax.ShapeDtypeStruct((e_total,), i32),
            jax.ShapeDtypeStruct((e_total,), i32),
            jax.ShapeDtypeStruct((ep, g_local), i32),
        )
    )

    def body(me_ref, counts_ref, *outs):
        t = shard_small_tables(
            counts_ref[...], me_ref[0], e_total=e_total, ep=ep,
            shard_stride=shard_stride, tile_m=tile_m,
        )
        flat = (
            *t.kernel_tables, t.expert_rows, t.slab_base, t.visit, t.counts,
            t.recv_base_e, t.mirror_base_e, t.run_base,
        )
        for ref, value in zip(outs, flat):
            ref[...] = value

    flat = pl.pallas_call(
        body,
        out_shape=shapes,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec()],
        out_specs=[pl.BlockSpec()] * len(shapes),
        name="moe_plan_small_tables",
    )(jnp.asarray(me, i32).reshape(1), rows_by_dest.astype(i32))
    return SmallTables(
        kernel_tables=tuple(flat[:5]),
        expert_rows=flat[5],
        slab_base=flat[6],
        visit=flat[7],
        counts=flat[8],
        recv_base_e=flat[9],
        mirror_base_e=flat[10],
        run_base=flat[11],
    )
