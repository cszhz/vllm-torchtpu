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
from jax import lax

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
    """``cumsum(x, axis) - x`` over a small triangle.

    Inside one Pallas program this is a handful of vector ops; `jnp.cumsum`
    there would have to lower a reduce-window. Measured the other way round
    in XLA, where the triangle's padded intermediates cost 4 us a call at the
    DP32 shape -- so the per-pair pass below keeps `jnp.cumsum`.

    The triangle is two iotas at the full rank rather than a reshaped
    [n, n]: Mosaic's layout inferer aborts the compiler on a reshape that
    changes the minormost dimension, and a mask built by reshape does.
    """
    n = x.shape[axis]
    shape = x.shape[:axis] + (n, n) + x.shape[axis + 1:]
    lower = lax.broadcasted_iota(jnp.int32, shape, axis + 1) < lax.broadcasted_iota(
        jnp.int32, shape, axis
    )
    return jnp.sum(jnp.where(lower, jnp.expand_dims(x, axis), 0), axis=axis + 1)


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


# What `_small_tables` returns, in order. Every one is two-dimensional:
# a Pallas body that holds a one-dimensional vector cannot keep the minormost
# dimension fixed across the broadcasts it needs, and Mosaic aborts the
# compiler rather than refusing when a reshape moves it.
_RAW = (
    "run_start",      # [g, source] in ROWBLK blocks
    "aligned_rows",   # [g, source] in blocks
    "contrib_off",    # [g, source] in blocks
    "push_dst",       # [g, source] in blocks
    "true_rows",      # [g, source] in rows
    "my_tiles",       # [g, source] mirror sublanes per push
    "recv_base_og",   # [owner, j] this shard's tokens' arrival base
    "mirror_base_og", # [owner, j] the same for the mirror
    "run_base_gs",    # [g, source]
    "totals",         # [3, source]: recv rows, rows received, tiles received
)


def shard_small_tables(rows_by_dest, me, *, e_total, ep, shard_stride, tile_m,
                       groups=0, pallas=False):
    """Shard ``me``'s non-per-pair plan tables from the ``[ep, e_total]`` counts.

    ``pallas`` runs the body as one Pallas program. The arithmetic is a few
    dozen vector ops on 13 312 int32s; spelled as XLA it is ~35 separate
    programs, and the launch cost of those is what this costs today.

    The transpose in and the reductions out stay outside the body on both
    paths, so the two produce the same values by construction.
    """
    g_local = e_total // ep
    run_rows_e = rows_by_dest.astype(jnp.int32).T  # [E, source]
    raw = dict(zip(_RAW, (
        _as_pallas(run_rows_e, me, e_total=e_total, ep=ep,
                   shard_stride=shard_stride, tile_m=tile_m)
        if pallas
        else _small_tables(run_rows_e, me, e_total=e_total, ep=ep,
                           shard_stride=shard_stride, tile_m=tile_m)
    )))
    aligned, true_rows = raw["aligned_rows"], raw["true_rows"]
    # Rows per local expert, and where each one's slab starts: both are the
    # row sums of a table the body already built.
    expert_rows = (aligned.sum(axis=1) << ROWBLK_SHIFT).astype(jnp.int32)
    slab_base = (jnp.cumsum(expert_rows) - expert_rows).astype(jnp.int32)
    visit, _ = expert_visit_list(expert_rows, g_local)
    recv_rows, rows_recv_all, tiles_recv_all = raw["totals"]
    mine = (jnp.arange(ep, dtype=jnp.int32) == me).astype(jnp.int32)
    counts = _count_vector(
        true_rows,
        (aligned << ROWBLK_SHIFT).astype(jnp.int32),
        raw["my_tiles"],
        expert_rows,
        rows_recv_all,
        tiles_recv_all,
        recv_rows,
        mine,
        g_local,
        ep,
    )
    return SmallTables(
        kernel_tables=(
            raw["run_start"], aligned, raw["contrib_off"], raw["push_dst"], true_rows
        ),
        expert_rows=expert_rows,
        slab_base=slab_base,
        visit=visit,
        counts=counts,
        recv_base_e=raw["recv_base_og"].reshape(e_total),
        mirror_base_e=raw["mirror_base_og"].reshape(e_total),
        run_base=raw["run_base_gs"].T,
    )


def _small_tables(run_rows_e, me, *, e_total, ep, shard_stride, tile_m):
    """The body, in the one orientation that needs no relayout.

    [E, source] throughout, so every intermediate keeps `source` minor. The
    only reshape splits the leading axis into (owner, local expert), which
    leaves the minor dimension alone; a reshape that regroups a table into
    [ep, g_local] does not (g_local is 13 at the served layout, not a whole
    number of sublane tiles) and aborts the Mosaic layout inferer.
    """
    g_local = e_total // ep
    run_rows = run_rows_e.astype(jnp.int32)
    aligned = (run_rows + (ROWBLK - 1)) & jnp.int32(-ROWBLK)
    run_start = _exclusive_prefix(aligned, axis=1)
    run_tiles = mirror_run_tiles(run_start, aligned, tile_m)
    region = aligned.reshape(ep, g_local, ep)  # [owner, j, source]
    per_owner = region.sum(axis=1)  # [owner, source]
    # recv_base[source, owner, j] is the prefix of `region` over (owner, j),
    # held here as [owner, j, source] so nothing transposes.
    recv_ojs = (
        _exclusive_prefix(region, axis=1)
        + _exclusive_prefix(per_owner, axis=0)[:, None, :]
    )

    def owner_is_me(rank, axis):
        return lax.broadcasted_iota(jnp.int32, (1,) * axis + (ep,) + (1,) * (rank - axis - 1), axis) == me

    is3 = owner_is_me(3, 0)  # [ep, 1, 1]
    src3 = owner_is_me(3, 2)  # [1, 1, ep]

    def over_owner(x):
        """``x[me * g_local : (me + 1) * g_local]`` of an ``[E, source]`` table."""
        return jnp.sum(jnp.where(is3, x.reshape(ep, g_local, ep), 0), axis=0)

    my_aligned = over_owner(aligned)  # [g, source]
    my_true = over_owner(run_rows)
    my_start = over_owner(run_start)
    my_tiles = over_owner(run_tiles)
    # This shard's slab bases, [g, 1] so they broadcast over the source axis.
    my_base = _exclusive_prefix(
        my_aligned.sum(axis=1, keepdims=True), axis=0
    ) + me * jnp.int32(shard_stride)
    my_region = jnp.sum(jnp.where(is3, region, 0), axis=0)  # [g, source]
    push_dst = jnp.sum(jnp.where(is3, recv_ojs, 0), axis=0)  # [g, source]
    recv_base_og = jnp.sum(jnp.where(src3, recv_ojs, 0), axis=2)  # [owner, j]
    my_run_start_og = jnp.sum(
        jnp.where(src3, run_start.reshape(ep, g_local, ep), 0), axis=2
    )

    contrib_off = _exclusive_prefix(
        my_region.sum(axis=0, keepdims=True), axis=1
    ) + _exclusive_prefix(my_region, axis=0)

    shift = pow2_shift(tile_m, "tile_m")
    expert_id = g_local * lax.broadcasted_iota(
        jnp.int32, (ep, g_local), 0
    ) + lax.broadcasted_iota(jnp.int32, (ep, g_local), 1)
    mirror_base_og = (
        jnp.right_shift(recv_base_og, shift) + 2 * expert_id
    ) * tile_m + (my_run_start_og & (tile_m - 1))

    def blocks(a):
        return jnp.right_shift(a, ROWBLK_SHIFT).astype(jnp.int32)

    totals = jnp.concatenate(
        [
            per_owner.sum(axis=0, keepdims=True),
            run_rows.sum(axis=0, keepdims=True),
            run_tiles.sum(axis=0, keepdims=True),
        ],
        axis=0,
    ).astype(jnp.int32)  # [3, source]
    return (
        blocks(my_start),
        blocks(my_aligned),
        blocks(contrib_off),
        blocks(push_dst),
        my_true.astype(jnp.int32),
        my_tiles.astype(jnp.int32),
        recv_base_og.astype(jnp.int32),
        mirror_base_og.astype(jnp.int32),
        (my_base + my_start).astype(jnp.int32),  # [g, source]; the caller transposes
        totals,
    )


def _count_vector(
    true_rows, aligned, tiles, expert_rows, rows_recv_all, tiles_recv_all,
    recv_rows, mine, g_local, ep,
):
    """`shard_count_vector`'s eight rows, off tables the body already holds."""
    other = 1 - mine
    first = (jnp.arange(g_local, dtype=jnp.int32) == 0).astype(jnp.int32)[:, None]
    active = (expert_rows > 0).astype(jnp.int32)[:, None]

    def over_experts(term):
        return term.sum(axis=0, keepdims=True)

    return jnp.concatenate(
        [
            over_experts(true_rows * other[None, :]),
            over_experts(
                (rows_recv_all * mine)[None, :] * first - true_rows * mine[None, :]
            ),
            over_experts(aligned * other[None, :]),
            over_experts(
                (recv_rows * mine)[None, :] * first - aligned * mine[None, :]
            ),
            over_experts(active * mine[None, :]),
            over_experts(tiles * other[None, :]),
            over_experts(
                (tiles_recv_all * mine)[None, :] * first - tiles * mine[None, :]
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


def _as_pallas(run_rows_e, me, *, e_total, ep, shard_stride, tile_m):
    """`_small_tables` as one kernel: one launch instead of ~35."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    g_local = e_total // ep
    i32 = jnp.int32
    gs = jax.ShapeDtypeStruct((g_local, ep), i32)
    og = jax.ShapeDtypeStruct((ep, g_local), i32)
    shapes = (gs,) * 6 + (og, og, gs, jax.ShapeDtypeStruct((3, ep), i32))
    assert len(shapes) == len(_RAW)

    def body(me_ref, rows_ref, *outs):
        for ref, value in zip(outs, _small_tables(
            rows_ref[...], me_ref[0], e_total=e_total, ep=ep,
            shard_stride=shard_stride, tile_m=tile_m,
        )):
            ref[...] = value

    return pl.pallas_call(
        body,
        out_shape=shapes,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec()],
        out_specs=[pl.BlockSpec()] * len(shapes),
        name="moe_plan_small_tables",
    )(jnp.asarray(me, i32).reshape(1), run_rows_e)
