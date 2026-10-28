# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32 sparse MQA tile streaming and query-local slot restoration before top-k."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from .fp8_mqa_tile_stream import (
    FILL_BATCH,
    RECORD_SLOTS,
    TILE_KEYS,
    TILE_PAGES,
    TILE_ROWS,
    UNROLL,
    pack_query_records,
    score_tiles,
)
from .packed_fp8_cache import PAGE_RECORDS, PAGE_ROWS


def _schedule(offsets, lengths, rows, begin, count, width):
    groups = offsets.shape[0] - 1
    tiles = width // TILE_KEYS
    size = offsets[1:] - offsets[:-1]
    first = jnp.minimum(offsets[:-1], rows - 1)
    second = jnp.minimum(first + 1, rows - 1)
    active = (size > 0) & (offsets[:-1] < begin + count) & (offsets[1:] > begin)
    limit = jnp.maximum(lengths[first], jnp.where(size == 2, lengths[second], 0))
    live = active[:, None] & (jnp.arange(tiles)[None, :] * TILE_KEYS < limit[:, None])
    pair = live & (size[:, None] == 2)
    single = live & (size[:, None] == 1)
    capacity = groups * tiles
    pair_count, single_count = pair.sum(dtype=jnp.int32), single.sum(dtype=jnp.int32)
    pair_ids = jnp.nonzero(pair.reshape(-1), size=capacity, fill_value=0)[0]
    single_ids = jnp.nonzero(single.reshape(-1), size=capacity, fill_value=0)[0]
    pair_steps = pl.cdiv(pair_count, UNROLL) * UNROLL
    single_steps = pl.cdiv(single_count, UNROLL) * UNROLL
    i = jnp.arange(2 * capacity + 2 * UNROLL, dtype=jnp.int32)
    is_pair = i < pair_steps
    pi = jnp.clip(jnp.minimum(i, pair_count - 1), 0, capacity - 1)
    si = jnp.clip(jnp.minimum(i - pair_steps, single_count - 1), 0, capacity - 1)
    tile_id = jnp.where(is_pair, pair_ids[pi], single_ids[si]).astype(jnp.int32)
    real = jnp.where(
        is_pair, i < pair_count, (i >= pair_steps) & (i - pair_steps < single_count)
    )
    new = real & (tile_id % tiles == 0)
    record_slot = jnp.maximum(jnp.cumsum(new, dtype=jnp.int32) - 1, 0) % RECORD_SLOTS
    descriptor = jnp.stack(
        (
            tile_id,
            tile_id,
            first[tile_id // tiles],
            (record_slot << 1) | new.astype(jnp.int32),
        ),
        axis=1,
    )
    meta = jnp.stack((pair_steps, pair_steps + single_steps, jnp.int32(0)))
    return meta, descriptor.reshape(-1), jnp.zeros((FILL_BATCH,), jnp.int32)


def _query_scores(single, pairs, slots, control, *, block):
    """SparseCore restores compact query slots; top-k sees only 16K FP32 rows."""
    rows, count = slots.shape
    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    group = min(256, pl.cdiv(count, 128) * 128)
    padded = pl.cdiv(count, group) * group
    steps = padded // group
    slots = jnp.pad(slots, ((0, 0), (0, padded - count)), constant_values=-1)
    blocks_per_tile = TILE_KEYS // block
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(
            (rows * padded * block // PAGE_RECORDS, PAGE_RECORDS), jnp.int32
        ),
        scratch_types=(
            pltpu.VMEM((padded,), jnp.int32),
            pltpu.VMEM((128,), jnp.int32),
            pltpu.VMEM((2, group, block), jnp.int32),
            pltpu.SemaphoreType.DMA((6,)),
        ),
        mesh=mesh,
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="fp8_query_slot_logits",
    )
    def project(
        single_ref, pair_ref, slot_ref, control_ref, out, indices, row_info, values, sem
    ):
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        @pl.loop(unit, rows, step=units)
        def row(row_id):
            read_slots = pltpu.async_copy(slot_ref.at[row_id], indices, sem.at[0])
            pltpu.async_copy(control_ref.at[row_id], row_info, sem.at[1]).wait()
            fields = row_info[pl.ds(0, lanes)]
            tile_base, member, paired, length = (fields[i] for i in range(4))
            read_slots.wait()

            @plsc.parallel_loop(0, padded, lanes, unroll=4)
            def addresses(i):
                slot = indices[pl.ds(i, lanes)]
                tile = tile_base + jnp.maximum(slot, 0) // blocks_per_tile
                address = (
                    jnp.where(paired != 0, 2 * tile + member, tile) * blocks_per_tile
                )
                address += jnp.maximum(slot, 0) % blocks_per_tile
                indices[pl.ds(i, lanes)] = jnp.where(
                    (slot >= 0) & (slot * block < length), address, -1
                )

            def read(step, wait=False):
                slot = step % 2
                if not wait:

                    @plsc.parallel_loop(0, group, lanes, unroll=4)
                    def clear(i):
                        row_ids = i + jnp.arange(lanes)
                        for field in range(block):
                            plsc.store_scatter(
                                values.at[slot],
                                [row_ids, jnp.full((lanes,), field, jnp.int32)],
                                jnp.full((lanes,), -8388608, jnp.int32),
                            )

                def transfer(src):
                    copy = pltpu.make_async_copy(
                        src.reshape(-1, block).at[
                            plsc.Indices(
                                indices.at[pl.ds(step * group, group)], ignored_value=-1
                            )
                        ],
                        values.at[slot],
                        sem.at[2 + slot],
                    )
                    (copy.wait if wait else copy.start)()

                @pl.when(length > 0)
                def gather():
                    lax.cond(
                        paired != 0,
                        lambda: transfer(pair_ref),
                        lambda: transfer(single_ref),
                    )

            def write(step, wait=False):
                slot = step % 2
                target = out.reshape(-1, block).at[
                    pl.ds(row_id * padded + step * group, group)
                ]
                copy = pltpu.make_async_copy(values.at[slot], target, sem.at[4 + slot])
                (copy.wait if wait else copy.start)()

            read(jnp.int32(0))

            @pl.loop(0, steps)
            def stream(step):
                @pl.when(step + 1 < steps)
                def prefetch():
                    pl.when(step >= 1)(lambda: write(step - 1, wait=True))
                    read(step + 1)

                read(step, wait=True)
                write(step)

            for last in range(max(0, steps - 2), steps):
                write(jnp.int32(last), wait=True)

    bits = project(
        lax.bitcast_convert_type(single.reshape(-1, PAGE_RECORDS), jnp.int32),
        lax.bitcast_convert_type(pairs.reshape(-1, PAGE_RECORDS), jnp.int32),
        slots,
        control,
    ).reshape(rows, padded * block)[:, : count * block]
    return lax.bitcast_convert_type(bits, jnp.float32)


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
    """Return FP32 scores in the original per-query sparse-block order.

    The TensorCore stream writes consecutive 4K score tiles. An internal
    SparseCore epilogue applies pos_a/pos_b before returning [queries, C*block].
    The existing top-k receives the original query-local layout and is unchanged.
    context_lens counts the visible prefix of the sorted gathered union.
    """
    del output_slots, key_tile
    rows = q.shape[0] if query_count is None else query_count
    output_width = candidate_slots.shape[1] * membership_block_size
    if rows == 0 or output_width == 0 or max_model_len == 0:
        return jnp.full((rows, output_width), -jnp.inf, jnp.float32)
    if q.shape[1:] != (32, 128) or TILE_KEYS % membership_block_size:
        raise ValueError("Expected H32/D128 and candidate blocks dividing 4096")
    groups = query_start_loc.shape[0] - 1
    group_rows = keys.shape[0] // groups
    capacity = group_rows // PAGE_ROWS * PAGE_RECORDS
    if max_model_len > capacity or group_rows % PAGE_ROWS:
        raise ValueError("Gathered K capacity does not cover the union")
    width = pl.cdiv(max_model_len, TILE_KEYS) * TILE_KEYS
    padded_rows = width // PAGE_RECORDS * PAGE_ROWS
    keys = keys.reshape(groups, group_rows, PAGE_RECORDS)[:, :padded_rows]
    keys = jnp.pad(keys, ((0, 0), (0, max(0, padded_rows - group_rows)), (0, 0)))
    keys = keys.reshape(-1, TILE_ROWS, PAGE_RECORDS)
    tiles = width // TILE_KEYS
    columns = width // membership_block_size
    allowed = jnp.pad(
        block_membership[:, :columns],
        ((0, 0), (0, max(0, columns - block_membership.shape[1]))),
    )
    size = query_start_loc[1:] - query_start_loc[:-1]
    first = jnp.minimum(query_start_loc[:-1], q.shape[0] - 1)
    second = jnp.minimum(first + 1, q.shape[0] - 1)
    positions = jnp.arange(width)[None, :]
    a = jnp.repeat(allowed[first], membership_block_size, axis=1)
    a &= (size[:, None] > 0) & (positions < context_lens[first, None])
    b = jnp.repeat(allowed[second], membership_block_size, axis=1)
    b &= (size[:, None] == 2) & (positions < context_lens[second, None])
    visibility = (a.astype(jnp.int32) | (b.astype(jnp.int32) << 1)).reshape(
        -1, TILE_PAGES, PAGE_RECORDS
    )
    meta, desc, dead = _schedule(
        query_start_loc, context_lens, q.shape[0], query_start, rows, width
    )
    records = pack_query_records(q, weights)
    single, pairs = score_tiles(
        meta,
        desc,
        dead,
        records,
        keys,
        visibility,
        heads=q.shape[1],
        rows=groups * tiles,
        pair_rows=groups * tiles,
    )
    query_rows = jnp.asarray(query_start, jnp.int32) + jnp.arange(rows)
    group = jnp.searchsorted(query_start_loc[1:], query_rows, side="right")
    member = query_rows - query_start_loc[group]
    control = jnp.stack(
        (
            group * tiles,
            member,
            (size[group] == 2).astype(jnp.int32),
            context_lens[query_rows],
        ),
        axis=1,
    )
    control = jnp.pad(control, ((0, 0), (0, 124)))
    return _query_scores(
        single,
        pairs,
        lax.dynamic_slice_in_dim(candidate_slots, query_start, rows),
        control,
        block=membership_block_size,
    )
