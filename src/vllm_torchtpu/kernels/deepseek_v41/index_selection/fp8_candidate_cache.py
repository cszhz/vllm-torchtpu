# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare and gather packed FP8 candidate caches for cross-layer reuse."""

from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from .packed_fp8_cache import PAGE_RECORDS, PAGE_ROWS, PackedFP8Layout
from .tuned_block_sizes import (
    PACKED_FP8_DENSE_MIN_KEY_TILE,
    PACKED_FP8_MQA_CONFIG,
    PAGED_MQA_GATHER_CONFIG,
)


class PreparedCandidates(NamedTuple):
    keys: jax.Array
    block_ids: jax.Array
    offsets: jax.Array
    order: jax.Array
    pages: jax.Array
    valid_queries: jax.Array
    membership: jax.Array


def compact_table_width(capacity: int) -> int:
    tile = min(
        PACKED_FP8_MQA_CONFIG.key_tile,
        max(PACKED_FP8_DENSE_MIN_KEY_TILE, 1 << (capacity - 1).bit_length()),
    )
    return max(128, (capacity + tile - 1) // tile * tile // PAGE_RECORDS)


def prepare_candidates(
    banks: tuple[jax.Array, ...],
    pages: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    layout: PackedFP8Layout,
    block: int,
    width: int,
    *,
    grouped: bool = False,
) -> PreparedCandidates:
    """Share gathered keys between adjacent queries of even-sized requests.

    As in vLLM v0.31.1rc0's attention/dsa/sparse_mqa_logits.py, candidates
    remain per query. Pairing only shares reads; membership preserves each
    query's selection domain. Odd-sized requests use individual gathers.
    """
    tokens, count = candidates.shape
    owners = jnp.clip(requests, 0, pages.shape[0] - 1)
    order = (
        jnp.arange(tokens, dtype=jnp.int32)
        if grouped
        else jnp.argsort(
            jnp.where(
                (requests >= 0) & (requests < pages.shape[0]), owners, pages.shape[0]
            ),
            stable=True,
        )
    )
    if not grouped:
        candidates, owners = candidates[order], owners[order]
    valid = (requests[order] >= 0) & (requests[order] < pages.shape[0])
    counts = jnp.bincount(
        owners, weights=valid.astype(jnp.int32), length=pages.shape[0]
    )
    rows = jnp.arange(tokens, dtype=jnp.int32)
    first = jax.ops.segment_min(jnp.where(valid, rows, tokens), owners, pages.shape[0])
    paired = valid & (counts[owners] % 2 == 0)
    starts = ~paired | ((rows - first[owners]) % 2 == 0)
    groups = jnp.cumsum(starts.astype(jnp.int32)) - 1
    sizes = jnp.bincount(groups, length=tokens)
    offsets = jnp.concatenate((jnp.zeros(1, jnp.int32), jnp.cumsum(sizes)))
    representatives = jax.ops.segment_min(rows, groups, tokens)
    representatives = jnp.minimum(representatives, tokens - 1)
    group_candidates = candidates[representatives]
    if tokens > 1:
        following = candidates[jnp.minimum(representatives + 1, tokens - 1)]
        group_candidates = jnp.concatenate(
            (group_candidates, jnp.where(sizes[:, None] == 2, following, -1)), axis=1
        )
    group_candidates = jnp.where(
        (sizes > 0)[:, None] & valid[representatives, None], group_candidates, -1
    )
    canonical, physical = canonical_candidate_blocks(
        group_candidates,
        pages[owners[representatives]],
        offsets,
        block=block,
        width=width,
        cache_pages=layout.pages,
        page_size=layout.page_size,
    )
    canonical = canonical[groups]
    sorted_candidates = jnp.sort(candidates, axis=1)
    slots = jax.vmap(jnp.searchsorted)(sorted_candidates, canonical)
    membership = (canonical >= 0) & (
        jnp.take_along_axis(sorted_candidates, jnp.minimum(slots, count - 1), axis=1)
        == canonical
    )
    fragment = min(block, layout.page_size)
    per_page = PAGE_RECORDS // fragment
    padded = (physical.shape[1] + per_page - 1) // per_page * per_page
    physical = jnp.pad(
        physical, ((0, 0), (0, padded - physical.shape[1])), constant_values=-1
    )
    keys = gather_candidate_keys(
        banks, physical, offsets, block=fragment, layout=layout
    )
    compact_pages = jnp.arange(tokens * padded // per_page, dtype=jnp.int32).reshape(
        tokens, -1
    )
    table_width = compact_table_width(group_candidates.shape[1] * block)
    compact_pages = jnp.pad(
        compact_pages,
        ((0, 0), (0, table_width - compact_pages.shape[1])),
        constant_values=-1,
    )
    return PreparedCandidates(
        keys,
        canonical,
        offsets,
        order,
        compact_pages,
        (requests >= 0) & (requests < pages.shape[0]),
        membership,
    )


def canonical_candidate_blocks(
    candidates: jax.Array,
    table: jax.Array,
    offsets: jax.Array,
    *,
    block: int,
    width: int,
    cache_pages: int,
    page_size: int = PAGE_RECORDS,
) -> tuple[jax.Array, jax.Array]:
    """Sort, deduplicate and resolve request rows through the last live group."""
    blocks = pl.cdiv(width, block)
    # Keep half of subcore VMEM free for compiler temporaries and alignment.
    scratch_words = (
        pl.cdiv(blocks, 128)
        + pl.cdiv(table.shape[1], 128)
        + 3 * pl.cdiv(candidates.shape[1], 128)
        + 3
    ) * 128
    if jax.default_backend() != "tpu" or (
        scratch_words * 4 > plsc.get_sparse_core_info().vmem_capacity_bytes // 2
    ):
        physical = resolve_candidate_blocks(
            candidates,
            table,
            sparse_block_kv=block,
            cache_pages=cache_pages,
            page_size=page_size,
        )
        parts = max(1, block // page_size)
        needed = (
            candidates[..., None] * block + jnp.arange(parts) * min(block, page_size)
            < width
        )
        valid = jnp.all(
            (physical.reshape(*candidates.shape, parts) >= 0) | ~needed, axis=-1
        )
        valid &= (candidates >= 0) & (candidates < blocks)
        canonical = jnp.sort(jnp.where(valid, candidates, blocks), axis=-1)
        unique = jnp.diff(canonical, axis=-1, prepend=-1) != 0
        canonical = jnp.sort(jnp.where(unique, canonical, blocks), axis=-1)
        canonical = jnp.where(canonical < blocks, canonical, -1)
        return canonical, resolve_candidate_blocks(
            canonical,
            table,
            sparse_block_kv=block,
            cache_pages=cache_pages,
            page_size=page_size,
        )

    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    rows, count = candidates.shape
    alignment = pltpu.get_tpu_info().num_lanes
    columns = pl.cdiv(count, alignment) * alignment
    histogram_width = pl.cdiv(blocks, alignment) * alignment
    candidates = jnp.pad(candidates, ((0, 0), (0, columns - count)), constant_values=-1)
    table = jnp.pad(
        table, ((0, 0), (0, -table.shape[1] % alignment)), constant_values=-1
    )
    live = jnp.max(jnp.where(offsets[1:] > offsets[:-1], jnp.arange(1, rows + 1), 0))
    control = jnp.full((lanes,), live, jnp.int32)
    per_page, parts = max(1, page_size // block), max(1, block // page_size)
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        out_type=(jax.ShapeDtypeStruct((rows, columns), jnp.int32),) * 2,
        scratch_types=(
            pltpu.VMEM((histogram_width,), jnp.int32),
            pltpu.VMEM((columns,), jnp.int32),
            pltpu.VMEM((table.shape[1],), jnp.int32),
            pltpu.VMEM((columns + lanes,), jnp.int32),
            pltpu.VMEM((columns + lanes,), jnp.int32),
            pltpu.VMEM((lanes,), jnp.int32),
            pltpu.SemaphoreType.DMA((4,)),
        ),
        mesh=mesh,
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="paged_mqa_candidate_metadata",
    )
    def prepare(
        src,
        pages,
        control_ref,
        canonical,
        physical,
        hist,
        ids,
        page_ids,
        logical_ids,
        physical_ids,
        active,
        sem,
    ):
        pltpu.async_copy(control_ref, active, sem.at[0]).wait()
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        @pl.loop(unit, active[...][0], step=units)
        def row(row_id):
            read_ids = pltpu.async_copy(src.at[row_id], ids, sem.at[0])
            read_pages = pltpu.async_copy(pages.at[row_id], page_ids, sem.at[1])

            @plsc.parallel_loop(0, histogram_width, lanes, unroll=alignment // lanes)
            def clear_hist(i):
                hist[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

            @plsc.parallel_loop(0, columns + lanes, lanes)
            def clear_result(i):
                logical_ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)
                physical_ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

            read_ids.wait()
            read_pages.wait()

            @plsc.parallel_loop(0, columns, lanes, unroll=alignment // lanes)
            def scatter(i):
                keys = ids[pl.ds(i, lanes)]
                inc, mask = plsc.scan_count(keys, mask=(keys >= 0) & (keys < blocks))
                plsc.addupdate_scatter(
                    hist, [jnp.clip(keys, 0, histogram_width - 1)], inc, mask=mask
                )

            def compact(i, count):
                logical = i + jnp.arange(lanes)
                page = plsc.load_gather(
                    page_ids,
                    [jnp.minimum(logical // per_page * parts, table.shape[1] - 1)],
                )
                valid = (
                    (hist[pl.ds(i, lanes)] != 0)
                    & (logical < blocks)
                    & (logical * block < table.shape[1] * page_size)
                    & (page >= 0)
                    & (page < cache_pages)
                )
                for part in range(1, parts):
                    column = logical * parts + part
                    following = plsc.load_gather(
                        page_ids, [jnp.minimum(column, table.shape[1] - 1)]
                    )
                    valid &= (logical * block + part * page_size >= width) | (
                        (column < table.shape[1])
                        & (following >= 0)
                        & (following < cache_pages)
                    )
                plsc.store_compressed(
                    logical_ids.at[pl.ds(count, lanes)], logical, mask=valid
                )
                plsc.store_compressed(
                    physical_ids.at[pl.ds(count, lanes)],
                    page * per_page + logical % per_page,
                    mask=valid,
                )
                return count + plsc.all_reduce_population_count(valid)[0]

            loop = cast(
                Callable[[Callable[[jax.Array, jax.Array], jax.Array]], jax.Array],
                plsc.parallel_loop(
                    0,
                    histogram_width,
                    lanes,
                    # Fully unrolling one histogram vector miscompiles the
                    # page-ID gather in libtpu 0.0.50.
                    unroll=1 if histogram_width == alignment else alignment // lanes,
                    carry=jnp.int32(0),
                ),
            )
            loop(compact)
            copy_ids = pltpu.async_copy(
                logical_ids.at[:columns], canonical.at[row_id], sem.at[2]
            )
            copy_physical = pltpu.async_copy(
                physical_ids.at[:columns], physical.at[row_id], sem.at[3]
            )
            copy_ids.wait()
            copy_physical.wait()

    canonical, physical = prepare(candidates, table, control)
    canonical, physical = canonical[:, :count], physical[:, :count]
    if parts > 1:
        physical = resolve_candidate_blocks(
            canonical,
            table,
            sparse_block_kv=block,
            cache_pages=cache_pages,
            page_size=page_size,
        )
    return canonical, physical


def _resolve_blocks_sc(blocks, table, block, cache_pages, page_size):
    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    rows, count = blocks.shape
    table_width, per_page = table.shape[1], page_size // block
    columns = pl.cdiv(count, 128) * 128
    blocks = jnp.pad(blocks, ((0, 0), (0, columns - count)), constant_values=-1)
    table = jnp.pad(table, ((0, 0), (0, -table_width % 128)), constant_values=-1)
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(blocks.shape, jnp.int32),
        scratch_types=(
            pltpu.VMEM((columns,), jnp.int32),
            pltpu.VMEM((table.shape[1],), jnp.int32),
            pltpu.VMEM((columns,), jnp.int32),
            pltpu.SemaphoreType.DMA((3,)),
        ),
        mesh=mesh,
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="paged_mqa_candidate_addresses",
    )
    def resolve(src, pages, out, ids, page_ids, physical, sem):
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        @pl.loop(unit, rows, step=units)
        def row(row_id):
            read_ids = pltpu.async_copy(src.at[row_id], ids, sem.at[0])
            read_pages = pltpu.async_copy(pages.at[row_id], page_ids, sem.at[1])
            read_ids.wait()
            read_pages.wait()

            @plsc.parallel_loop(0, columns, lanes, unroll=8)
            def address(i):
                selected = ids[pl.ds(i, lanes)]
                safe = jnp.clip(selected, 0, table_width * per_page - 1)
                page = plsc.load_gather(page_ids, [safe // per_page])
                valid = (
                    (selected >= 0)
                    & (selected < table_width * per_page)
                    & (page >= 0)
                    & (page < cache_pages)
                )
                physical[pl.ds(i, lanes)] = jnp.where(
                    valid, page * per_page + safe % per_page, -1
                )

            pltpu.async_copy(physical, out.at[row_id], sem.at[2]).wait()

    return resolve(blocks, table)[:, :count]


@jax.jit(static_argnames=("sparse_block_kv", "cache_pages", "page_size"))
def resolve_candidate_blocks(
    sparse_blocks: jax.Array,
    block_table: jax.Array,
    *,
    sparse_block_kv: int,
    cache_pages: int,
    page_size: int = PAGE_RECORDS,
) -> jax.Array:
    """Resolve request-shared [G,C] blocks, preserving order and duplicates.

    Blocks spanning pages expand into page-sized fragments in candidate order.
    Per-query causal lengths are applied by the scorer, not this shared metadata.
    """
    if sparse_block_kv > page_size:
        parts = sparse_block_kv // page_size
        sparse_blocks = jnp.where(
            sparse_blocks[..., None] >= 0,
            sparse_blocks[..., None] * parts + jnp.arange(parts),
            -1,
        ).reshape(sparse_blocks.shape[0], -1)
        sparse_block_kv = page_size
    per_page = page_size // sparse_block_kv
    limit = block_table.shape[1] * per_page
    safe = jnp.clip(sparse_blocks, 0, limit - 1)
    scratch_words = (
        2 * pl.cdiv(sparse_blocks.shape[1], 128) + pl.cdiv(block_table.shape[1], 128)
    ) * 128
    if jax.default_backend() == "tpu" and (
        scratch_words * 4 <= plsc.get_sparse_core_info().vmem_capacity_bytes // 2
    ):
        return _resolve_blocks_sc(
            sparse_blocks, block_table, sparse_block_kv, cache_pages, page_size
        )
    pages = jnp.take_along_axis(block_table, safe // per_page, axis=1)
    valid = (
        (sparse_blocks >= 0)
        & (sparse_blocks < limit)
        & (pages >= 0)
        & (pages < cache_pages)
    )
    return jnp.where(valid, pages * per_page + safe % per_page, -1)


def gather_candidate_keys(
    banks: tuple[jax.Array, ...],
    physical: jax.Array,
    offsets: jax.Array,
    *,
    block: int,
    layout: PackedFP8Layout,
) -> jax.Array:
    """Gather whole compact pages; trailing empty query groups do no DMA."""
    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    group = max(lanes, min(128, 1024 // block))
    buffers = min(
        2,
        info.vmem_capacity_bytes
        // (group * PAGE_ROWS * (block + len(banks)) * 4 + 4096),
    )
    group, buffers = PAGED_MQA_GATHER_CONFIG.get((block, len(banks)), (group, buffers))
    distance = buffers - 1
    count, per_page = physical.size, PAGE_RECORDS // block
    source_per_page = layout.page_size // block
    per_unit = pl.cdiv(count, units * group) * group
    scratch_words = buffers * group * PAGE_ROWS * (block + len(banks)) + lanes
    metadata_rows = min(
        per_unit,
        (info.vmem_capacity_bytes // 4 - scratch_words - 4096) // group * group,
    )
    metadata_steps = metadata_rows // group
    ids = jnp.pad(
        physical.reshape(-1), (0, units * per_unit - count), constant_values=-1
    )
    ids = ids.reshape(-1, units, group).transpose(1, 0, 2).reshape(units, per_unit)
    ids = jnp.pad(ids, ((0, 0), (0, -per_unit % metadata_rows)), constant_values=-1)
    live_rows = jnp.max(
        jnp.where(offsets[1:] > offsets[:-1], jnp.arange(physical.shape[0]) + 1, 0)
    )
    active_steps = pl.cdiv(live_rows * physical.shape[1], units * group)
    control = jnp.full((lanes,), active_steps, jnp.int32)
    span, steps = group * PAGE_ROWS, per_unit // group
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(
            (units * per_unit // per_page * PAGE_ROWS, PAGE_RECORDS), jnp.int32
        ),
        scratch_types=(
            pltpu.VMEM((metadata_rows,), jnp.int32),
            pltpu.VMEM((lanes,), jnp.int32),
            pltpu.VMEM((buffers, span, block), jnp.int32),
            pltpu.VMEM((buffers, len(banks), span), jnp.int32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((buffers, len(banks))),
            pltpu.SemaphoreType.DMA((buffers,)),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False, needs_layout_passes=False
        ),
        mesh=mesh,
        name="paged_mqa_candidate_gather",
    )
    def gather(
        pool,
        physical_ref,
        control_ref,
        out,
        selected,
        control_vmem,
        words,
        read_ids,
        metadata_sem,
        read_sem,
        write_sem,
    ):
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        pltpu.async_copy(control_ref, control_vmem, metadata_sem).wait()
        active = control_vmem[...][0]
        metadata = physical_ref.at[unit]
        if metadata_rows < per_unit:
            metadata = metadata.at[:metadata_rows]
        pltpu.async_copy(metadata, selected, metadata_sem).wait()

        def read(step, slot, wait=False):
            if not wait:
                index = step
                if metadata_rows < per_unit:
                    index = step % metadata_steps

                    @pl.when((step > 0) & (index == 0))
                    def next_metadata():
                        pltpu.async_copy(
                            physical_ref.at[unit, pl.ds(step * group, metadata_rows)],
                            selected,
                            metadata_sem,
                        ).wait()

                @plsc.parallel_loop(0, span, lanes, unroll=8)
                def addresses(i):
                    segment = i + jnp.arange(lanes)
                    field = segment // per_page % PAGE_ROWS
                    column = (
                        segment // (PAGE_ROWS * per_page) * per_page
                        + segment % per_page
                    )
                    chosen = plsc.load_gather(selected, [index * group + column])
                    safe = jnp.maximum(chosen, 0)
                    source = (
                        (
                            layout.page_offset
                            + safe // source_per_page * layout.page_stride
                        )
                        * per_page
                        + field * source_per_page
                        + safe % source_per_page
                    )
                    bank_base = 0
                    for bank, ref in enumerate(pool):
                        size = ref.shape[0] * per_page
                        valid = (
                            (chosen >= 0)
                            & (source >= bank_base)
                            & (source < bank_base + size)
                        )
                        read_ids[slot, bank, pl.ds(i, lanes)] = jnp.where(
                            valid, source - bank_base, -1
                        )
                        bank_base += size

            for bank, ref in enumerate(pool):
                copy = pltpu.make_async_copy(
                    ref.reshape(-1, block).at[
                        plsc.Indices(read_ids.at[slot, bank], ignored_value=-1)
                    ],
                    words.at[slot],
                    read_sem.at[slot, bank],
                )
                (copy.wait if wait else copy.start)()

        def write(step, wait=False):
            slot = step % buffers
            start = (step * units + unit) * group
            copy = pltpu.make_async_copy(
                words.at[slot],
                out.reshape(-1, block).at[pl.ds(start * PAGE_ROWS, span)],
                write_sem.at[slot],
            )
            (copy.wait if wait else copy.start)()

        for step in range(min(distance, steps)):
            pl.when(step < active)(
                lambda step=step: read(jnp.int32(step), jnp.int32(step))
            )

        @pl.loop(0, active)
        def body(step):
            slot = step % buffers
            if buffers == 1:
                read(step, slot)
            read(step, slot, wait=True)
            write(step)
            if buffers == 1:
                write(step, wait=True)

            @pl.when((distance > 0) & (step + distance < active))
            def prefetch():
                future = (step + distance) % buffers
                pl.when(step + distance >= buffers)(
                    lambda: write(step + distance - buffers, wait=True)
                )
                read(step + distance, future)

        for last in range(buffers if buffers > 1 else 0):
            pl.when(last < active)(
                lambda last=last: write(active - last - 1, wait=True)
            )

    return gather(banks, ids, control)[: count // per_page * PAGE_ROWS]
