# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Masked FP8 MQA logits over contiguous candidate unions after gathering."""

from dataclasses import replace
from functools import partial, reduce
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .fp8_mqa_common import (
    QueryTile,
    _dma_rows,
    _head_order,
    _next_query_tile,
    _reduce_heads,
)
from .packed_fp8_cache import PAGE_RECORDS, PAGE_ROWS
from .tuned_block_sizes import PACKED_FP8_PREPARED_MQA_CONFIG


def _kernel(
    offsets,
    lengths,
    query_start,
    inputs,
    out,
    scratch,
    *,
    config,
    info,
    membership_block_size,
):
    mxu_size, sublanes = info.mxu_column_size, info.num_sublanes
    q, weights, pool, membership = inputs
    buffers, semaphores = scratch
    queries, keys, scores, counts = buffers
    read_sem, write_sem, q_sem = semaphores
    weights = pltpu.annotate(weights, no_store=True)
    keys = pltpu.annotate(keys, no_store=True)
    tokens, heads, dim = q.shape
    begin = query_start[()]
    end = begin + out.shape[0]
    q_tile, key_tile = config.query_tile, config.key_tile
    tile_rows = key_tile // PAGE_RECORDS * PAGE_ROWS
    group_rows = out.shape[1] // PAGE_RECORDS * PAGE_ROWS
    tiles, fragments = out.shape[1] // key_tile, key_tile // mxu_size

    def copy_queries(query, slot, wait=False, zero=False):
        dst = queries.at[pl.ds(slot * q_tile, q_tile)]
        if zero:
            dst[...] = jnp.zeros(dst.shape, dst.dtype)
        _dma_rows(q, dst, q_sem.at[slot], query.count, wait, src_start=query.start)

    def release_outputs(pair):
        count = counts[pair]
        for slot in (2 * pair, 2 * pair + 1)[:tiles]:
            src = scores.at[slot]
            _dma_rows(src, src, write_sem.at[slot], count, wait=True)

    def load_keys(tile, group, slot):
        pltpu.make_async_copy(
            pool.at[pl.ds(group * group_rows + tile * tile_rows, tile_rows)],
            keys.at[slot],
            read_sem.at[slot],
        ).start()

    next_query = partial(_next_query_tile, offsets=offsets, tile_size=q_tile, end=end)

    def row_ids(query):
        return [jnp.minimum(query.start + i, tokens - 1) for i in range(q_tile)]

    def row_lengths(query):
        rows = enumerate(row_ids(query))
        return [jnp.where(i < query.count, lengths[r], 0) for i, r in rows]

    def live_tiles(query):
        limit = reduce(jnp.maximum, row_lengths(query), jnp.int32(0))
        return pl.cdiv(jnp.minimum(limit, out.shape[1]), key_tile)

    owners = _head_order(
        jnp.repeat(jnp.arange(q_tile), heads)[:, None], q_tile, sublanes
    )[:, 0]
    selector = (jnp.arange(q_tile)[:, None] == owners[None, :]).astype(jnp.bfloat16)

    def query_block(state):
        query, following, lookahead, tile_counts, step, key_base, ready = state
        tile_count, next_tiles, ahead_tiles = tile_counts
        q_slot = step & 1
        farthest = next_query(lookahead)

        def key_slot(tile):
            return (key_base + tile) % config.key_buffers

        def output_slot(tile):
            return 2 * q_slot + (tile & 1)

        copy_queries(following, 1 - q_slot, zero=True)
        lengths_vector = jnp.stack(row_lengths(query))

        def publish(tile, wait=False, slot=None):
            slot = output_slot(tile) if slot is None else slot
            src, sem = scores.at[slot], write_sem.at[slot]
            dst = out.at[:, pl.ds(tile * key_tile, key_tile)]
            _dma_rows(src, dst, sem, query.count, wait, dst_start=query.start - begin)

        def tile_source(future):
            first, after = future < tile_count, future - tile_count
            third = after >= next_tiles
            return (
                jnp.where(first, future, after - jnp.where(third, next_tiles, 0)),
                jnp.where(
                    first,
                    query.page_row,
                    jnp.where(third, lookahead.page_row, following.page_row),
                ),
            )

        release_outputs(q_slot)
        for i in range(2):

            @pl.when((i >= ready) & (i < tile_count + next_tiles + ahead_tiles))
            def prefill(i=i):
                target, owner = tile_source(i)
                load_keys(target, owner, key_slot(i))

        copy_queries(query, q_slot, wait=True)

        def emit_tile(tile):
            weight = jnp.concatenate([weights[r][:, None] for r in row_ids(query)])
            head_weights = _head_order(weight, q_tile, sublanes)
            query_slice = pl.ds(q_slot * q_tile, q_tile)
            lhs = _head_order(
                queries[query_slice].reshape(q_tile * heads, dim), q_tile, sublanes
            )

            def key_rows(start, count):
                return keys[key_slot(tile), pl.ds(start, count)]

            block_flags = None
            if membership is not None:
                blocks_per_tile = key_tile // membership_block_size
                block_flags = jnp.stack(
                    [
                        membership[r, pl.ds(tile * blocks_per_tile, blocks_per_tile)]
                        for r in row_ids(query)
                    ]
                )
                block_flags = block_flags.astype(jnp.int32)
            base = tile * key_tile + jnp.arange(PAGE_RECORDS)
            remaining = jnp.maximum(lengths_vector, 0)[:, None] - base[None, :]

            def qk(fragment):
                parts = []
                for page in range(
                    fragment * mxu_size // PAGE_RECORDS,
                    (fragment + 1) * mxu_size // PAGE_RECORDS,
                ):
                    words = key_rows(page * PAGE_ROWS, PAGE_ROWS - 1)
                    parts.append(pltpu.bitcast(words, q.dtype))
                logits = lax.dot(
                    lhs,
                    jnp.concatenate(parts, axis=1),
                    preferred_element_type=jnp.float32,
                )
                return jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))

            def store(fragment, values):
                for offset in range(0, mxu_size, PAGE_RECORDS):
                    local_page = (fragment * mxu_size + offset) // PAGE_RECORDS
                    scale_row = (local_page + 1) * PAGE_ROWS - 1
                    scale = lax.bitcast_convert_type(
                        key_rows(scale_row, 1), jnp.float32
                    )
                    scaled = values[:, offset : offset + PAGE_RECORDS] * scale
                    column = local_page * PAGE_RECORDS
                    scaled = jnp.where(remaining > column, scaled, -jnp.inf)
                    if block_flags is not None:
                        columns = (
                            column + jnp.arange(PAGE_RECORDS)
                        ) // membership_block_size
                        first_flag = (column // membership_block_size // 128) * 128
                        local_flags = block_flags[:, first_flag : first_flag + 128]
                        allowed = (
                            jnp.take_along_axis(
                                local_flags,
                                jnp.broadcast_to(
                                    (columns - first_flag)[None, :],
                                    (block_flags.shape[0], columns.shape[0]),
                                ),
                                axis=1,
                            )
                            != 0
                        )
                        scaled = jnp.where(allowed, scaled, -jnp.inf)
                    scores[output_slot(tile), :, pl.ds(column, PAGE_RECORDS)] = scaled

            pending = {}
            for stage in range(fragments + config.issue_depth):
                if stage < fragments:
                    pending[stage] = qk(stage)
                if stage >= config.issue_depth:
                    fragment = stage - config.issue_depth
                    values = _reduce_heads(
                        pending.pop(fragment) * head_weights,
                        selector,
                        sublanes=sublanes,
                        use_mxu=False,
                    )
                    store(fragment, values)

        def step_tile(tile, _):
            src = keys.at[key_slot(tile)]
            pltpu.make_async_copy(src, src, read_sem.at[key_slot(tile)]).wait()
            pl.when(tile >= 2)(lambda: publish(tile - 2, wait=True))

            @pl.when(tile + 2 < tile_count + next_tiles + ahead_tiles)
            def prefetch():
                target, owner = tile_source(tile + 2)
                load_keys(target, owner, key_slot(tile + 2))

            emit_tile(tile)
            publish(tile)

        if tiles == 1:
            pl.when(tile_count > 0)(lambda: step_tile(jnp.int32(0), None))
        else:
            lax.fori_loop(0, tile_count, step_tile, None)

        counts[q_slot] = jnp.where(tile_count == tiles, query.count, 0)

        @pl.when(tile_count < tiles)
        def fill_padding():
            @pl.when(tile_count > 0)
            def drain():
                publish(tile_count - 1, wait=True)
                pl.when(tile_count > 1)(lambda: publish(tile_count - 2, wait=True))

            scores[output_slot(0)] = jnp.full(scores.shape[1:], -jnp.inf, jnp.float32)
            for wait in (False, True):

                def body(tile, _, wait=wait):
                    publish(tile, wait=wait, slot=output_slot(0))

                lax.fori_loop(tile_count, tiles, body, None)

        return (
            following,
            lookahead,
            farthest,
            (next_tiles, ahead_tiles, live_tiles(farthest)),
            step + 1,
            key_slot(tile_count),
            jnp.minimum(2, next_tiles + ahead_tiles),
        )

    zero = jnp.int32(0)
    first = next_query(QueryTile(begin, zero, zero))
    second = next_query(first)
    third = next_query(second)
    initial = (first, second, third)
    for pair in range(2):
        counts[pair] = jnp.int32(0)
    copy_queries(first, 0, zero=True)
    lax.while_loop(
        lambda state: state[0].start < end,
        query_block,
        (
            *initial,
            tuple(live_tiles(q) for q in initial),
            zero,
            zero,
            zero,
        ),
    )
    release_outputs(0)
    release_outputs(1)


@jax.jit(
    static_argnames=(
        "max_model_len",
        "membership_block_size",
        "query_count",
        "key_tile",
    )
)
def fp8_gathered_mqa_logits(
    q: jax.Array,
    keys: jax.Array,
    weights: jax.Array,
    context_lens: jax.Array,
    block_membership: jax.Array,
    *,
    query_start_loc: jax.Array,
    max_model_len: int,
    membership_block_size: int,
    query_start: jax.Array | int = 0,
    query_count: int | None = None,
    key_tile: int | None = None,
) -> jax.Array:
    """Score gathered K without a page table or cache-bank lookup.

    Each group owns a contiguous, equally sized K region in packed 33x128
    format. Offsets map one or two adjacent queries to that region; membership
    is per query, in union-block order. Weights already include query scales.
    Context lengths count visible tokens in the sorted union. Returned column
    IDs remain compact; the outer top-k maps them through canonical block IDs.
    """
    rows = q.shape[0] if query_count is None else query_count
    if rows == 0 or max_model_len == 0:
        return jnp.empty((rows, max_model_len), jnp.float32)
    info = cast(pltpu.TpuInfo, pltpu.get_tpu_info())
    groups = query_start_loc.shape[0] - 1
    group_rows = keys.shape[0] // groups
    capacity = group_rows // PAGE_ROWS * PAGE_RECORDS
    if keys.shape != (groups * group_rows, PAGE_RECORDS) or group_rows % PAGE_ROWS:
        raise ValueError("Gathered K must contain equal packed regions per group")
    if max_model_len > capacity:
        raise ValueError("Score width exceeds gathered K capacity")
    config = PACKED_FP8_PREPARED_MQA_CONFIG
    tile = min(
        config.key_tile if key_tile is None else key_tile,
        pl.cdiv(max_model_len, info.mxu_column_size) * info.mxu_column_size,
    )
    if tile % info.mxu_column_size or tile % membership_block_size:
        raise ValueError("K tile must align to MXU columns and membership blocks")
    config = replace(config, key_tile=tile)
    width = pl.cdiv(max_model_len, tile) * tile
    padded_rows = width // PAGE_RECORDS * PAGE_ROWS
    keys = keys.reshape(groups, group_rows, PAGE_RECORDS)[:, :padded_rows]
    keys = jnp.pad(keys, ((0, 0), (0, max(0, padded_rows - group_rows)), (0, 0)))
    keys = keys.reshape(-1, PAGE_RECORDS)
    columns = width // membership_block_size
    membership = jnp.pad(
        block_membership[:, :columns].astype(jnp.int32),
        ((0, 0), (0, max(0, columns - block_membership.shape[1]))),
    )
    q_tile, heads, dim = config.query_tile, q.shape[1], q.shape[2]
    scratch = (
        (
            pltpu.VMEM((2 * q_tile, heads, dim), q.dtype),
            pltpu.VMEM(
                (config.key_buffers, tile // PAGE_RECORDS * PAGE_ROWS, dim), jnp.int32
            ),
            pltpu.VMEM((4, q_tile, tile), jnp.float32),
            pltpu.SMEM((2,), jnp.int32),
        ),
        tuple(
            pltpu.SemaphoreType.DMA(shape)
            for shape in ((config.key_buffers,), (4, q_tile), (2,))
        ),
    )
    hbm, vmem = (pl.BlockSpec(memory_space=s) for s in (pltpu.HBM, pltpu.VMEM))
    return pl.pallas_call(
        partial(
            _kernel,
            config=config,
            info=info,
            membership_block_size=membership_block_size,
        ),
        out_shape=jax.ShapeDtypeStruct((rows, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=((hbm, vmem, hbm, vmem),),
            out_specs=hbm,
            scratch_shapes=(scratch,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=info.vmem_capacity_bytes,
            disable_bounds_checks=True,
        ),
        name="fp8_gathered_mqa_logits",
    )(
        query_start_loc,
        context_lens,
        jnp.asarray(query_start, jnp.int32),
        (q, weights, keys, membership),
    )[:, :max_model_len]
