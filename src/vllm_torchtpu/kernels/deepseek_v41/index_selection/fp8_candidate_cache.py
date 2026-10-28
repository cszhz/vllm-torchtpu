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
    PAGED_MQA_GATHER_CONFIG,
    PAGED_MQA_WHOLE_PAGE_GATHER_CONFIG,
)


class PreparedCandidates(NamedTuple):
    """Internal union storage and original per-query output-slot metadata."""

    keys: jax.Array
    block_ids: jax.Array
    offsets: jax.Array
    order: jax.Array
    valid_queries: jax.Array
    membership: jax.Array
    candidates: jax.Array
    candidate_slots: jax.Array
    output_slots: jax.Array


class CandidateMetadata(NamedTuple):
    """Union IDs plus forward/reverse maps to each query's original slots."""

    physical: jax.Array
    block_ids: jax.Array
    offsets: jax.Array
    order: jax.Array
    valid_queries: jax.Array
    membership: jax.Array
    candidates: jax.Array
    candidate_slots: jax.Array
    output_slots: jax.Array


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
    """Prepare candidate metadata and gather its compact key pages."""
    metadata = prepare_candidate_metadata(
        pages, candidates, requests, layout, block, width, grouped=grouped
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
    *,
    grouped: bool = False,
) -> CandidateMetadata:
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
    slots, output_candidates, output_slots = candidate_output_slots(
        candidates, canonical, pl.cdiv(width, block)
    )
    membership = output_slots >= 0
    # SparseCore metadata only initializes live group rows.
    physical = jnp.where(sizes[:, None] > 0, physical, -1)
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
        order,
        (requests >= 0) & (requests < pages.shape[0]),
        membership,
        output_candidates,
        slots,
        output_slots,
    )


def _candidate_output_slots_reference(candidates, canonical):
    """Map original per-query slots to the union; retain only the first duplicate."""
    sentinel = jnp.iinfo(jnp.int32).max
    sorted_union = jnp.where(canonical >= 0, canonical, sentinel)
    slots = jax.vmap(lambda a, b: jnp.searchsorted(a, b, method="scan"))(
        sorted_union, candidates
    )
    safe = jnp.minimum(slots, canonical.shape[1] - 1)
    valid = (candidates >= 0) & (slots < canonical.shape[1])
    valid &= jnp.take_along_axis(canonical, safe, axis=1) == candidates
    order = jnp.argsort(candidates, axis=1, stable=True)
    ordered = jnp.take_along_axis(candidates, order, axis=1)
    first = jnp.concatenate(
        (
            jnp.ones((candidates.shape[0], 1), jnp.bool_),
            ordered[:, 1:] != ordered[:, :-1],
        ),
        axis=1,
    )
    first = jnp.take_along_axis(first, jnp.argsort(order, axis=1), axis=1)
    valid &= first
    return jnp.where(valid, slots, -1), jnp.where(valid, candidates, -1)


def candidate_output_slots(
    candidates: jax.Array, canonical: jax.Array, blocks: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build both slot maps, keeping the first occurrence in original query order."""
    rows, count = candidates.shape
    union_count = canonical.shape[1]
    columns = pl.cdiv(count, 128) * 128
    union_columns = pl.cdiv(union_count, 128) * 128
    histogram_width = pl.cdiv(blocks, 128) * 128
    words = histogram_width + 2 * columns + 2 * union_columns
    if (
        jax.default_backend() != "tpu"
        or words * 4 > plsc.get_sparse_core_info().vmem_capacity_bytes // 2
    ):
        slots, ids = _candidate_output_slots_reference(candidates, canonical)
        inverse = (
            jnp.full_like(canonical, -1)
            .at[jnp.arange(rows)[:, None], jnp.maximum(slots, 0)]
            .max(jnp.where(slots >= 0, jnp.arange(count)[None, :], -1))
        )
        return slots, ids, inverse
    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    candidates = jnp.pad(candidates, ((0, 0), (0, columns - count)), constant_values=-1)
    canonical = jnp.pad(
        canonical, ((0, 0), (0, union_columns - union_count)), constant_values=-1
    )
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        out_type=(
            jax.ShapeDtypeStruct((rows, columns), jnp.int32),
            jax.ShapeDtypeStruct((rows, columns), jnp.int32),
            jax.ShapeDtypeStruct((rows, union_columns), jnp.int32),
        ),
        scratch_types=(
            pltpu.VMEM((histogram_width,), jnp.int32),
            pltpu.VMEM((columns,), jnp.int32),
            pltpu.VMEM((union_columns,), jnp.int32),
            pltpu.VMEM((columns,), jnp.int32),
            pltpu.VMEM((union_columns,), jnp.int32),
            pltpu.SemaphoreType.DMA((5,)),
        ),
        mesh=mesh,
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="fp8_query_candidate_slots",
    )
    def build(
        src,
        unions,
        forward_out,
        ids_out,
        reverse_out,
        hist,
        ids,
        union_ids,
        forward,
        reverse,
        sem,
    ):
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        @pl.loop(unit, rows, step=units)
        def row(row_id):
            read_ids = pltpu.async_copy(src.at[row_id], ids, sem.at[0])
            read_union = pltpu.async_copy(unions.at[row_id], union_ids, sem.at[1])

            @plsc.parallel_loop(0, histogram_width, lanes, unroll=4)
            def clear(i):
                hist[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

            read_union.wait()

            @plsc.parallel_loop(0, union_columns, lanes, unroll=4)
            def index_union(i):
                keys = union_ids[pl.ds(i, lanes)]
                plsc.store_scatter(
                    hist,
                    [jnp.clip(keys, 0, blocks - 1)],
                    i + jnp.arange(lanes) + 1,
                    mask=(keys >= 0) & (keys < blocks),
                )

            read_ids.wait()

            # Sequential vectors preserve the first original slot across vectors.
            @pl.loop(0, columns, step=lanes)
            def query_slots(i):
                keys = ids[pl.ds(i, lanes)]
                addresses = jnp.clip(keys, 0, blocks - 1)
                found = plsc.load_gather(hist, [addresses])
                valid = (keys >= 0) & (keys < blocks) & (found > 0)
                occurrences, _ = plsc.scan_count(keys, mask=valid)
                first = valid & (occurrences == 1)
                forward[pl.ds(i, lanes)] = jnp.where(first, found - 1, -1)
                ids[pl.ds(i, lanes)] = jnp.where(first, keys, -1)
                plsc.store_scatter(
                    hist, [addresses], -(i + jnp.arange(lanes) + 1), mask=first
                )

            @plsc.parallel_loop(0, union_columns, lanes, unroll=4)
            def inverse(i):
                keys = union_ids[pl.ds(i, lanes)]
                value = plsc.load_gather(hist, [jnp.clip(keys, 0, blocks - 1)])
                reverse[pl.ds(i, lanes)] = jnp.where(
                    (keys >= 0) & (keys < blocks) & (value < 0), -value - 1, -1
                )

            writes = (
                pltpu.async_copy(forward, forward_out.at[row_id], sem.at[2]),
                pltpu.async_copy(ids, ids_out.at[row_id], sem.at[3]),
                pltpu.async_copy(reverse, reverse_out.at[row_id], sem.at[4]),
            )
            for transfer in writes:
                transfer.wait()

    forward, ids, reverse = build(candidates, canonical)
    return forward[:, :count], ids[:, :count], reverse[:, :union_count]


def candidate_membership(candidates, canonical, blocks):
    """Test per-query union membership using a SparseCore presence histogram.

    A batched binary search lowers to dependent TensorCore gathers, making
    metadata preparation dominate the layer. Each subcore instead builds a
    private histogram and reads the union's flags from local VMEM.
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
    """Gather compact pages, skipping empty tiles inside SparseCore."""
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
        + (buffers * group * PAGE_ROWS * block if whole_pages else 0)
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
                pltpu.VMEM(
                    (buffers, group // per_page * PAGE_ROWS, PAGE_RECORDS), jnp.int32
                ),
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
        out,
        selected,
        control_vmem,
        flags,
        words,
        read_ids,
        metadata_sem,
        read_sem,
        write_sem,
        *packed,
    ):
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        pltpu.async_copy(control_ref, control_vmem, metadata_sem).wait()
        active = control_vmem[...][0]
        pltpu.async_copy(tile_flags_ref.at[unit], flags, metadata_sem).wait()
        metadata = physical_ref.at[unit]
        if metadata_rows < per_unit:
            metadata = metadata.at[:metadata_rows]
        pltpu.async_copy(metadata, selected, metadata_sem).wait()

        def tile_is_live(step):
            return (
                plsc.load_gather(flags, [jnp.full((lanes,), step, jnp.int32)])[0] != 0
            )

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
                            bank_base = 0
                            for bank, ref in enumerate(pool):
                                size = ref.shape[0] // layout.page_rows
                                valid = (
                                    (chosen >= 0)
                                    & (source >= bank_base)
                                    & (source < bank_base + size)
                                )
                                read_ids[slot, bank, pl.ds(i, lanes)] = jnp.where(
                                    valid, source - bank_base, -1
                                )
                                bank_base += size
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
                                bank_base = 0
                                for bank, ref in enumerate(pool):
                                    size = ref.shape[0] * per_page
                                    valid = (
                                        (chosen >= 0)
                                        & (source >= bank_base)
                                        & (source < bank_base + size)
                                    )
                                    read_ids[
                                        slot,
                                        bank,
                                        pl.ds(i * PAGE_ROWS + field * lanes, lanes),
                                    ] = jnp.where(valid, source - bank_base, -1)
                                    bank_base += size
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

                        @plsc.parallel_loop(
                            0, group // per_page * PAGE_ROWS, 1, unroll=2
                        )
                        def pack_fields(row):
                            base = (
                                row // PAGE_ROWS * per_page * (page_words // block)
                                + row % PAGE_ROWS
                            )
                            for col in range(0, PAGE_RECORDS, lanes):
                                columns = col + jnp.arange(lanes)
                                source = base + columns // block * (page_words // block)
                                values = plsc.load_gather(
                                    words.at[slot], [source, columns % block]
                                )
                                packed[0][slot, row, pl.ds(col, lanes)] = values

                    output_rows = group // per_page * PAGE_ROWS
                    target = out.at[pl.ds(destination * output_rows, output_rows)]
                    source = packed[0].at[slot]
                else:
                    target = out.reshape(-1, block).at[
                        pl.ds(destination * group * PAGE_ROWS, span)
                    ]
                    source = words.at[slot]
                copy = pltpu.make_async_copy(source, target, write_sem.at[slot])
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

    return gather(banks, ids, control, tile_flags)[: count // per_page * PAGE_ROWS]
