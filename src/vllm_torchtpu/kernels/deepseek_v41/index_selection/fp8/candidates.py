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

from .cache import PAGE_RECORDS, PAGE_ROWS, PackedFP8Layout
from .tuned import (
    PAGED_MQA_GATHER_CONFIG,
    PAGED_MQA_WHOLE_PAGE_GATHER_CONFIG,
)


class PreparedCandidates(NamedTuple):
    """Group-shared K and per-query mappings for candidate scoring."""

    keys: jax.Array  # Contiguous packed FP8 K region per group.
    block_ids: jax.Array  # Sorted union IDs per query; -1 is padding.
    offsets: jax.Array  # Group g owns queries [offsets[g], offsets[g + 1]).
    valid_queries: jax.Array  # Valid request-ID mask in input query order.
    membership: jax.Array  # Per-query boolean membership aligned with block_ids.


class CandidateMetadata(NamedTuple):
    """Pre-gather metadata; remaining fields match PreparedCandidates."""

    physical: jax.Array  # Physical fragment IDs per group; -1 is invalid/padding.
    block_ids: jax.Array
    offsets: jax.Array
    valid_queries: jax.Array
    membership: jax.Array


def prepare_candidates(
    banks: tuple[jax.Array, ...],
    pages: jax.Array,
    candidates: jax.Array,
    requests_ids: jax.Array,
    layout: PackedFP8Layout,
    block: int,
    width: int,
) -> PreparedCandidates:
    """Gather packed FP8 K for candidate unions shared by query groups.

    Args:
        banks: Physical packed FP8 cache banks.
        pages: Dense [requests, max_pages_per_request] physical page table;
            valid page counts vary by request, bounded by sequence lengths.
        candidates: [queries, 2048] logical block IDs; -1 is padding.
        requests_ids: [queries] request IDs; each request's queries are contiguous.
        layout: Packed cache storage layout.
        block: Tokens per candidate block.
        width: Logical token range upper bound.

    Returns:
        PreparedCandidates with gathered K, logical block mappings, query group
        offsets, validity and per-query membership masks.
    """
    metadata = prepare_candidate_metadata(
        pages, candidates, requests_ids, layout, block, width
    )
    keys = gather_candidate_keys(
        banks,
        metadata.physical,
        metadata.offsets,
        block=min(block, layout.page_size),
        layout=layout,
    )
    return PreparedCandidates(keys, *metadata[1:])


def prepare_candidate_metadata(
    pages: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    layout: PackedFP8Layout,
    block: int,
    width: int,
) -> CandidateMetadata:
    """Pair queries of even-sized requests; odd-sized requests stay unpaired.

    Args:
        pages: Dense [requests, max_pages_per_request] physical page table;
            valid page counts vary by request, bounded by sequence lengths.
        candidates: [queries, 2048] logical block IDs; -1 is padding.
        requests: [queries] request IDs; each request's queries are contiguous.
        layout: Packed cache storage layout.
        block: Tokens per candidate block.
        width: Logical token range upper bound.

    Returns:
        Per-group physical fragments and per-query union IDs, membership,
        grouping offsets and validity mask, all in input query order.
    """
    # The caller supplies contiguous queries per request, followed by padding.
    tokens, count = candidates.shape
    owners = jnp.clip(requests, 0, pages.shape[0] - 1)
    valid = (requests >= 0) & (requests < pages.shape[0])

    # Pair adjacent queries in even-sized requests; leave odd-sized ones unpaired.
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

    # Concatenate each group's candidates; mask missing partners and empty groups.
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

    # Sort/deduplicate each union and resolve physical Index-K fragments.
    # width bounds logical Index-K record positions, not candidate count.
    canonical, physical = canonical_candidate_blocks(
        group_candidates,
        pages[owners[representatives]],
        offsets,
        block=block,
        width=width,
        cache_pages=layout.pages,
        page_size=layout.page_size,
    )

    # Expand union IDs to queries and preserve each query's original candidates.
    canonical = canonical[groups]
    membership = candidate_membership(candidates, canonical, pl.cdiv(width, block))

    # SparseCore metadata only initializes live group rows.
    physical = jnp.where(sizes[:, None] > 0, physical, -1)
    # Align fragment capacity so each group's gathered K fills packed pages.
    fragment = min(block, layout.page_size)
    per_page = PAGE_RECORDS // fragment
    padded = (physical.shape[1] + per_page - 1) // per_page * per_page
    physical = jnp.pad(
        physical, ((0, 0), (0, padded - physical.shape[1])), constant_values=-1
    )
    return CandidateMetadata(
        physical,
        canonical,
        offsets,
        valid,
        membership,
    )


def candidate_membership(candidates, canonical, blocks):
    """Test per-query union membership using a SparseCore presence histogram.

    A batched binary search lowers to dependent TensorCore gathers, making
    metadata preparation dominate the layer. Each subcore instead builds a
    private histogram and reads the union's flags from local VMEM.

    Args:
        candidates: [queries, 2048] original logical candidate block IDs.
        canonical: [queries, 2048 or 4096] shared union IDs in the same row order.
        blocks: Logical block count; valid IDs are in [0, blocks).

    Returns:
        Boolean mask shaped like canonical, true for valid IDs in candidates.
    """
    rows, count = candidates.shape
    columns = canonical.shape[1]
    padded_count = pl.cdiv(count, 128) * 128
    padded_columns = pl.cdiv(columns, 128) * 128
    histogram_width = pl.cdiv(blocks, 128) * 128
    scratch_words = histogram_width + padded_count + 2 * padded_columns
    if jax.default_backend() != "tpu" or (
        scratch_words * 4 > plsc.get_sparse_core_info().vmem_capacity_bytes // 2
    ):
        ordered = jnp.sort(candidates, axis=1)
        slots = jax.vmap(lambda a, v: jnp.searchsorted(a, v, method="sort"))(
            ordered, canonical
        )
        found = jnp.take_along_axis(ordered, jnp.minimum(slots, count - 1), axis=1)
        return (canonical >= 0) & (canonical < blocks) & (found == canonical)

    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    candidates = jnp.pad(
        candidates, ((0, 0), (0, padded_count - count)), constant_values=-1
    )
    canonical = jnp.pad(
        canonical, ((0, 0), (0, padded_columns - columns)), constant_values=-1
    )
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(canonical.shape, jnp.int32),
        scratch_types=(
            pltpu.VMEM((histogram_width,), jnp.int32),
            pltpu.VMEM((padded_count,), jnp.int32),
            pltpu.VMEM((padded_columns,), jnp.int32),
            pltpu.VMEM((padded_columns,), jnp.int32),
            pltpu.SemaphoreType.DMA((3,)),
        ),
        mesh=mesh,
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="paged_mqa_candidate_membership",
    )
    def membership(src, unions, out, hist, ids, union_ids, flags, sem):
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        @pl.loop(unit, rows, step=units)
        def row(row_id):
            read_ids = pltpu.async_copy(src.at[row_id], ids, sem.at[0])
            read_union = pltpu.async_copy(unions.at[row_id], union_ids, sem.at[1])

            @plsc.parallel_loop(0, histogram_width, lanes, unroll=128 // lanes)
            def clear(i):
                hist[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

            read_ids.wait()

            @plsc.parallel_loop(0, padded_count, lanes, unroll=128 // lanes)
            def scatter(i):
                keys = ids[pl.ds(i, lanes)]
                inc, valid = plsc.scan_count(keys, mask=(keys >= 0) & (keys < blocks))
                plsc.addupdate_scatter(
                    hist, [jnp.clip(keys, 0, blocks - 1)], inc, mask=valid
                )

            read_union.wait()

            @plsc.parallel_loop(0, padded_columns, lanes, unroll=128 // lanes)
            def lookup(i):
                keys = union_ids[pl.ds(i, lanes)]
                present = plsc.load_gather(hist, [jnp.clip(keys, 0, blocks - 1)])
                flags[pl.ds(i, lanes)] = (
                    (keys >= 0) & (keys < blocks) & (present != 0)
                ).astype(jnp.int32)

            pltpu.async_copy(flags, out.at[row_id], sem.at[2]).wait()

    return membership(candidates, canonical)[:, :columns] != 0


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
    """Sort, deduplicate and resolve candidate blocks per query group.

    Args:
        candidates: [groups, 2048 or 4096] logical candidate block IDs.
        table: [groups, max_pages_per_request] physical page table rows.
        offsets: Query boundaries identifying live group rows.
        block: Tokens per candidate block.
        width: Logical token range upper bound.
        cache_pages: Physical cache page count.
        page_size: Tokens per cache page.

    Returns:
        Sorted unique logical IDs and corresponding physical fragment IDs;
        invalid entries are -1. SparseCore leaves trailing empty groups unset.
    """
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
    """Resolve sub-page or page-sized blocks using SparseCore.

    Args:
        blocks: [groups, slots] logical block IDs.
        table: [groups, max_pages_per_request] physical page table rows.
        block: Tokens per block; divides page_size.
        cache_pages: Physical cache page count.
        page_size: Tokens per cache page.

    Returns:
        Physical block IDs shaped like blocks; invalid entries are -1.
    """
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

    Args:
        sparse_blocks: [groups, slots] logical candidate block IDs.
        block_table: [groups, max_pages_per_request] physical page table rows.
        sparse_block_kv: Tokens per candidate block.
        cache_pages: Physical cache page count.
        page_size: Tokens per cache page.

    Returns:
        [groups, slots * max(1, sparse_block_kv // page_size)] physical fragment
        IDs, preserving order and duplicates; invalid entries are -1.
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
    """Gather compact pages, skipping empty tiles inside SparseCore.

    Args:
        banks: Physical packed FP8 cache banks.
        physical: [groups, fragments] physical fragment IDs; -1 is padding.
        offsets: Query boundaries identifying live group rows.
        block: Tokens per fragment, at most layout.page_size.
        layout: Packed cache storage layout.

    Returns:
        Packed int32 K of shape [packed_rows, PAGE_RECORDS], with equal-sized
        contiguous regions per group; skipped tiles remain uninitialized.
    """
    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    group = max(lanes, min(128, 1024 // block))
    buffers = min(
        2,
        info.vmem_capacity_bytes
        // (group * PAGE_ROWS * (block + len(banks)) * 4 + 4096),
    )
    group, buffers = PAGED_MQA_GATHER_CONFIG.get((block, len(banks)), (group, buffers))
    whole_pages = (
        (block, len(banks)) in PAGED_MQA_WHOLE_PAGE_GATHER_CONFIG
        and block == layout.page_size == 8
        and layout.page_offset % layout.page_rows == 0
        and layout.page_stride % layout.page_rows == 0
        and all(bank.shape[0] % layout.page_rows == 0 for bank in banks)
    )
    if whole_pages:
        group, buffers = PAGED_MQA_WHOLE_PAGE_GATHER_CONFIG[block, len(banks)]
    page_words = layout.page_rows * PAGE_RECORDS
    read_span = group if whole_pages else group * PAGE_ROWS
    storage_span = group * page_words // block if whole_pages else group * PAGE_ROWS
    distance = buffers - 1
    count, per_page = physical.size, PAGE_RECORDS // block
    source_per_page = layout.page_size // block
    per_unit = pl.cdiv(count, units * group) * group
    flags_width = pl.cdiv(per_unit // group, 128) * 128
    scratch_words = (
        buffers * (storage_span * block + read_span * len(banks))
        + ((buffers + 1) * storage_span if whole_pages else 0)
        + lanes
        + flags_width
    )
    metadata_rows = min(
        per_unit,
        (info.vmem_capacity_bytes // 4 - scratch_words - 4096) // group * group,
    )
    metadata_steps = metadata_rows // group
    ids = jnp.pad(
        physical.reshape(-1), (0, units * per_unit - count), constant_values=-1
    )
    tiles = ids.reshape(-1, group)
    live_rows = jnp.max(
        jnp.where(offsets[1:] > offsets[:-1], jnp.arange(physical.shape[0]) + 1, 0)
    )
    live_tiles = jnp.any(tiles >= 0, axis=1) & (
        jnp.arange(tiles.shape[0]) * group < live_rows * physical.shape[1]
    )
    ids = ids.reshape(-1, units, group).transpose(1, 0, 2).reshape(units, per_unit)
    tile_flags = live_tiles.astype(jnp.int32).reshape(-1, units).T
    tile_flags = jnp.pad(
        tile_flags,
        ((0, 0), (0, flags_width - tile_flags.shape[1])),
    )
    ids = jnp.pad(ids, ((0, 0), (0, -per_unit % metadata_rows)), constant_values=-1)

    active_steps = pl.cdiv(live_rows * physical.shape[1], units * group)
    control = jnp.full((lanes,), active_steps, jnp.int32)
    span, steps = group * PAGE_ROWS, per_unit // group
    if whole_pages:
        segments = jnp.arange(storage_span, dtype=jnp.int32)
        record, field = (
            segments // (page_words // block),
            segments % (page_words // block),
        )
        pattern = jnp.where(
            field < PAGE_ROWS,
            (record // per_page * PAGE_ROWS + field) * per_page + record % per_page,
            -1,
        )
    else:
        pattern = jnp.zeros((lanes,), jnp.int32)
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(
            (units * per_unit // per_page * PAGE_ROWS, PAGE_RECORDS), jnp.int32
        ),
        scratch_types=(
            pltpu.VMEM((metadata_rows,), jnp.int32),
            pltpu.VMEM((lanes,), jnp.int32),
            pltpu.VMEM((flags_width,), jnp.int32),
            pltpu.VMEM((buffers, storage_span, block), jnp.int32),
            pltpu.VMEM((buffers, len(banks), read_span), jnp.int32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((buffers, len(banks))),
            pltpu.SemaphoreType.DMA((buffers,)),
        )
        + (
            (
                pltpu.VMEM((buffers, storage_span), jnp.int32),
                pltpu.VMEM((storage_span,), jnp.int32),
            )
            if whole_pages
            else ()
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
        tile_flags_ref,
        pattern_ref,
        out,
        selected,
        control_vmem,
        flags,
        words,
        read_ids,
        metadata_sem,
        read_sem,
        write_sem,
        *scatter,
    ):
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        pltpu.async_copy(control_ref, control_vmem, metadata_sem).wait()
        active = control_vmem[...][0]
        pltpu.async_copy(tile_flags_ref.at[unit], flags, metadata_sem).wait()
        if whole_pages:
            pltpu.async_copy(pattern_ref, scatter[1], metadata_sem).wait()
        metadata = physical_ref.at[unit]
        if metadata_rows < per_unit:
            metadata = metadata.at[:metadata_rows]
        pltpu.async_copy(metadata, selected, metadata_sem).wait()

        def tile_is_live(step):
            return (
                plsc.load_gather(flags, [jnp.full((lanes,), step, jnp.int32)])[0] != 0
            )

        def store_read_ids(slot, columns, chosen, source):
            bank_base = 0
            for bank, ref in enumerate(pool):
                size = (
                    ref.shape[0] // layout.page_rows
                    if whole_pages
                    else ref.shape[0] * per_page
                )
                valid = (
                    (chosen >= 0) & (source >= bank_base) & (source < bank_base + size)
                )
                read_ids[slot, bank, columns] = jnp.where(valid, source - bank_base, -1)
                bank_base += size

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

            @pl.when(tile_is_live(step))
            def transfer():
                if not wait:
                    if whole_pages:

                        @plsc.parallel_loop(0, group, lanes)
                        def page_addresses(i):
                            chosen = selected[pl.ds(index * group + i, lanes)]
                            source = (
                                layout.page_offset
                                + jnp.maximum(chosen, 0) * layout.page_stride
                            ) // layout.page_rows
                            store_read_ids(slot, pl.ds(i, lanes), chosen, source)
                    elif per_page == lanes:

                        @plsc.parallel_loop(0, group, lanes)
                        def block_addresses(i):
                            chosen = selected[pl.ds(index * group + i, lanes)]
                            safe = jnp.maximum(chosen, 0)
                            base = (
                                layout.page_offset
                                + safe // source_per_page * layout.page_stride
                            ) * per_page + safe % source_per_page
                            for field in range(PAGE_ROWS):
                                source = base + field * source_per_page
                                store_read_ids(
                                    slot,
                                    pl.ds(i * PAGE_ROWS + field * lanes, lanes),
                                    chosen,
                                    source,
                                )
                    else:

                        @plsc.parallel_loop(0, span, lanes, unroll=8)
                        def addresses(i):
                            segment = i + jnp.arange(lanes)
                            field = segment // per_page % PAGE_ROWS
                            column = (
                                segment // (PAGE_ROWS * per_page) * per_page
                                + segment % per_page
                            )
                            chosen = plsc.load_gather(
                                selected, [index * group + column]
                            )
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
                            store_read_ids(slot, pl.ds(i, lanes), chosen, source)

                for bank, ref in enumerate(pool):
                    copy = pltpu.make_async_copy(
                        ref.reshape(-1, page_words if whole_pages else block).at[
                            plsc.Indices(read_ids.at[slot, bank], ignored_value=-1)
                        ],
                        words.at[slot].reshape(group, page_words)
                        if whole_pages
                        else words.at[slot],
                        read_sem.at[slot, bank],
                    )
                    (copy.wait if wait else copy.start)()

        def write(step, wait=False):
            slot = step % buffers
            destination = step * units + unit

            @pl.when(tile_is_live(step))
            def copy_tile():
                if whole_pages:
                    if not wait:

                        @plsc.parallel_loop(0, storage_span, lanes, unroll=8)
                        def scatter_addresses(i):
                            local = scatter[1][pl.ds(i, lanes)]
                            scatter[0][slot, pl.ds(i, lanes)] = jnp.where(
                                local >= 0, destination * group * PAGE_ROWS + local, -1
                            )

                    target = out.reshape(-1, block).at[
                        plsc.Indices(scatter[0].at[slot], ignored_value=-1)
                    ]
                else:
                    target = out.reshape(-1, block).at[
                        pl.ds(destination * group * PAGE_ROWS, span)
                    ]
                copy = pltpu.make_async_copy(words.at[slot], target, write_sem.at[slot])
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

    return gather(banks, ids, control, tile_flags, pattern)[
        : count // per_page * PAGE_ROWS
    ]
