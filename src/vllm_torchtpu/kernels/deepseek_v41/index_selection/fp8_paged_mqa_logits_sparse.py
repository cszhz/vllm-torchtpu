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
    q, weights, pool, membership, output_map = inputs
    buffers, semaphores = scratch
    queries, keys, counts, projected, maps = buffers
    read_sem, write_sem, q_sem, map_sem = semaphores
    weights = pltpu.annotate(weights, no_store=True)
    keys = pltpu.annotate(keys, no_store=True)
    tokens, heads, dim = q.shape
    begin = query_start[()]
    end = begin + out.shape[0]
    q_tile, key_tile = config.query_tile, config.key_tile
    tile_rows = key_tile // PAGE_RECORDS * PAGE_ROWS
    group_rows = pool.shape[0] // (offsets.shape[0] - 1)
    union_width = group_rows // PAGE_ROWS * PAGE_RECORDS
    tiles, fragments = union_width // key_tile, key_tile // mxu_size
    map_columns = output_map.shape[1]

    def copy_queries(query, slot, wait=False, zero=False):
        dst = queries.at[pl.ds(slot * q_tile, q_tile)]
        if zero:
            dst[...] = jnp.zeros(dst.shape, dst.dtype)
        _dma_rows(q, dst, q_sem.at[slot], query.count, wait, src_start=query.start)

    def output_ref(pair, row):
        return projected.at[
            pl.ds(0, 1), pl.ds((pair * q_tile + row) * out.shape[1], out.shape[1])
        ]

    def release_outputs(pair):
        for i in range(q_tile):

            @pl.when(i < counts[pair])
            def wait(i=i):
                src = output_ref(pair, i)
                pltpu.make_async_copy(src, src, write_sem.at[pair, i]).wait()

    def copy_map(query, slot, wait=False):
        for i in range(q_tile):

            @pl.when(i < query.count)
            def copy(i=i):
                dst = maps.at[pl.ds((slot * q_tile + i) * map_columns, map_columns)]
                dma = pltpu.make_async_copy(
                    output_map.at[query.start + i], dst, map_sem.at[slot]
                )
                (dma.wait if wait else dma.start)()

    def key_rows_to_read(tile, limit):
        valid = jnp.minimum(key_tile, jnp.maximum(limit - tile * key_tile, 0))
        return pl.cdiv(valid, PAGE_RECORDS) * PAGE_ROWS

    def load_keys(tile, group, slot, limit):
        count = key_rows_to_read(tile, limit)
        pltpu.make_async_copy(
            pool.at[pl.ds(group * group_rows + tile * tile_rows, count)],
            keys.at[slot, pl.ds(0, count)],
            read_sem.at[slot],
        ).start()

    next_query = partial(_next_query_tile, offsets=offsets, tile_size=q_tile, end=end)

    def row_ids(query):
        return [jnp.minimum(query.start + i, tokens - 1) for i in range(q_tile)]

    def row_lengths(query):
        rows = enumerate(row_ids(query))
        return [jnp.where(i < query.count, lengths[r], 0) for i, r in rows]

    def query_limit(query):
        limit = reduce(jnp.maximum, row_lengths(query), jnp.int32(0))
        return jnp.minimum(limit, union_width)

    def live_tiles(query):
        return pl.cdiv(query_limit(query), key_tile)

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

        copy_queries(following, 1 - q_slot, zero=True)
        lengths_vector = jnp.stack(row_lengths(query))

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
                jnp.where(
                    first,
                    query_limit(query),
                    jnp.where(third, query_limit(lookahead), query_limit(following)),
                ),
            )

        release_outputs(q_slot)
        projected[
            pl.ds(0, 1), pl.ds(q_slot * q_tile * out.shape[1], q_tile * out.shape[1])
        ] = jnp.full((1, q_tile * out.shape[1]), -jnp.inf, jnp.float32)
        copy_map(query, q_slot)
        for i in range(2):

            @pl.when((i >= ready) & (i < tile_count + next_tiles + ahead_tiles))
            def prefill(i=i):
                target, owner, limit = tile_source(i)
                load_keys(target, owner, key_slot(i), limit)

        copy_queries(query, q_slot, wait=True)
        copy_map(query, q_slot, wait=True)

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
                    for i in range(q_tile):
                        for start in range(0, PAGE_RECORDS, membership_block_size):
                            union_block = (
                                tile * key_tile + column + start
                            ) // membership_block_size
                            destination = maps[
                                (q_slot * q_tile + i) * map_columns + union_block
                            ]

                            @pl.when((i < query.count) & (destination >= 0))
                            def write(
                                i=i, start=start, destination=destination, scaled=scaled
                            ):
                                output_column = destination * membership_block_size
                                page = output_column // PAGE_RECORDS * PAGE_RECORDS
                                previous = projected[
                                    pl.ds(0, 1),
                                    pl.ds(
                                        (q_slot * q_tile + i) * out.shape[1] + page,
                                        PAGE_RECORDS,
                                    ),
                                ]
                                shift = (output_column - start) % PAGE_RECORDS
                                aligned = pltpu.roll(scaled[i : i + 1], shift, axis=1)
                                active = jnp.arange(
                                    PAGE_RECORDS
                                ) // membership_block_size == destination % (
                                    PAGE_RECORDS // membership_block_size
                                )
                                projected[
                                    pl.ds(0, 1),
                                    pl.ds(
                                        (q_slot * q_tile + i) * out.shape[1] + page,
                                        PAGE_RECORDS,
                                    ),
                                ] = jnp.where(active[None, :], aligned, previous)

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
            count = key_rows_to_read(tile, query_limit(query))
            src = keys.at[key_slot(tile), pl.ds(0, count)]
            pltpu.make_async_copy(src, src, read_sem.at[key_slot(tile)]).wait()

            @pl.when(tile + 2 < tile_count + next_tiles + ahead_tiles)
            def prefetch():
                target, owner, limit = tile_source(tile + 2)
                load_keys(target, owner, key_slot(tile + 2), limit)

            emit_tile(tile)

        if tiles == 1:
            pl.when(tile_count > 0)(lambda: step_tile(jnp.int32(0), None))
        else:
            lax.fori_loop(0, tile_count, step_tile, None)

        for i in range(q_tile):

            @pl.when(i < query.count)
            def publish(i=i):
                dst = out.at[pl.ds(query.start - begin + i, 1), :]
                pltpu.make_async_copy(
                    output_ref(q_slot, i), dst, write_sem.at[q_slot, i]
                ).start()

        counts[q_slot] = query.count

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
    candidate_slots: jax.Array,
    output_slots: jax.Array,
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
    DMA reads only the visible union prefix, rounded to a packed 128-token page.
    Context lengths count visible tokens in the sorted union. Returned column
    IDs are per-query candidate slots; the existing top-k maps them to tokens.
    candidate_slots fixes the original per-query candidate count. output_slots
    maps union blocks to those original slots; -1 means no write for this query.
    Each FP32 fragment writes directly into query-local VMEM candidate slots.
    No union score array is materialized; completed query rows are DMAed to HBM.
    """
    rows = q.shape[0] if query_count is None else query_count
    output_width = candidate_slots.shape[1] * membership_block_size
    if rows == 0 or max_model_len == 0:
        return jnp.full((rows, output_width), -jnp.inf, jnp.float32)
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
    if PAGE_RECORDS % membership_block_size:
        raise ValueError("Candidate block size must divide the packed page")
    output_padded = pl.cdiv(output_width, PAGE_RECORDS) * PAGE_RECORDS
    map_columns = (
        pl.cdiv(width // membership_block_size, info.num_lanes) * info.num_lanes
    )
    output_slots = jnp.pad(
        output_slots,
        ((0, 0), (0, map_columns - output_slots.shape[1])),
        constant_values=-1,
    )
    q_tile, heads, dim = config.query_tile, q.shape[1], q.shape[2]
    scratch = (
        (
            pltpu.VMEM((2 * q_tile, heads, dim), q.dtype),
            pltpu.VMEM(
                (config.key_buffers, tile // PAGE_RECORDS * PAGE_ROWS, dim), jnp.int32
            ),
            pltpu.SMEM((2,), jnp.int32),
            pltpu.VMEM((1, 2 * q_tile * output_padded), jnp.float32),
            pltpu.SMEM((2 * q_tile * map_columns,), jnp.int32),
        ),
        tuple(
            pltpu.SemaphoreType.DMA(shape)
            for shape in ((config.key_buffers,), (2, q_tile), (2,), (2,))
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
        out_shape=jax.ShapeDtypeStruct((rows, output_padded), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=((hbm, vmem, hbm, vmem, hbm),),
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
        (q, weights, keys, membership, output_slots),
    )[:, :output_width]
