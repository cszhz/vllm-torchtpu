# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2024-2025 SGLang Team
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SparseCore DeepSelect-style Top-K over score columns.

Adapted from deepseek-ai/DeepSelect's scan/filter/compact algorithm:
https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.zh.md
Uses shuffled block visits, pipelined DMA and exact score/index radix selection.
"""

from __future__ import annotations

import functools
import math
import random
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v41.index_selection import tuned_block_sizes

# Byte-wise radix passes over IEEE FP32 keys. These describe the representation,
# independently of TensorCore/SparseCore lane counts.
RADIX_BITS = jnp.iinfo(jnp.uint8).bits
RADIX_BUCKETS = 1 << RADIX_BITS
RADIX_MASK = RADIX_BUCKETS - 1
FP32_BITS = jnp.finfo(jnp.float32).bits
FP32_SIGN_BIT = 1 << (FP32_BITS - 1)
UINT32_MAX = jnp.iinfo(jnp.uint32).max
FP32_NEG_INF_BITS = 0xFF800000
ORDERED_NEG_INF = UINT32_MAX ^ FP32_NEG_INF_BITS


class _Selection(NamedTuple):
    count: jax.Array
    threshold: jax.Array
    threshold_id: jax.Array
    have_threshold: jax.Array


class TopKEpilogue(NamedTuple):
    """Compile-time row output fusion; apply must invoke the supplied sort.

    Extra index scratch starts after the aligned output; row metadata follows
    the length lanes. Both reuse selection storage after selection finishes.
    """

    inputs: tuple[jax.Array, ...]
    row_metadata: jax.Array | None
    index_scratch_size: int
    apply: Callable[..., None]


class _TopKPlan(NamedTuple):
    block_size: int
    capacity: int
    id_capacity: int
    candidate_buffers: int
    buffers: int
    scratch_bytes: int


def _topk_plan(
    width: int,
    k: int,
    metadata_width: int,
    index_scratch_size: int,
    block_size: int | None,
    reconstruct_threshold: int,
    bootstrap_multiplier: int,
    *,
    lanes: int,
    hbm_alignment: int,
    vmem_bytes: int,
    score_bytes: int,
    id_bytes: int,
    merge_capacity: int = 0,
    single_shot: bool = False,
) -> _TopKPlan:
    output_width = pl.cdiv(k, hbm_alignment) * hbm_alignment

    def plan(tile):
        bootstrap = min(tile, bootstrap_multiplier * k)
        # A single-shot slice is compacted once; partitions also merge P * k.
        streamed = max(k + reconstruct_threshold + math.gcd(tile, bootstrap), bootstrap)
        capacity = max(tile if single_shot else streamed, output_width, merge_capacity)
        id_capacity = max(capacity, output_width + index_scratch_size)
        order_width = pl.cdiv(pl.cdiv(width, tile), lanes) * lanes
        fixed = k * (score_bytes + id_bytes) + id_bytes * (RADIX_BUCKETS + order_width)

        def scratch(candidates, inputs):
            return (
                fixed
                + candidates * (capacity * score_bytes + id_capacity * id_bytes)
                + inputs * tile * score_bytes
                + metadata_width * id_bytes
            )

        candidates = 2 if scratch(2, 1) <= vmem_bytes else 1
        # A single-shot slice has nothing to prefetch.
        inputs = 2 if not single_shot and scratch(candidates, 2) <= vmem_bytes else 1
        return _TopKPlan(
            tile,
            capacity,
            id_capacity,
            candidates,
            inputs,
            scratch(candidates, inputs),
        )

    if block_size is not None:
        result = plan(block_size)
    else:
        # Limit each input DMA to one candidate-bank payload to bound startup.
        # Within that cap, prefer prefetch banks and larger aligned tiles.
        result = plan(min(width, hbm_alignment))
        tile = result.block_size * 2
        while tile <= width:
            candidate = plan(tile)
            bank_bytes = (
                candidate.capacity * score_bytes + candidate.id_capacity * id_bytes
            )
            if (
                tile * score_bytes <= bank_bytes
                and candidate.scratch_bytes <= vmem_bytes
                and (
                    candidate.candidate_buffers,
                    candidate.buffers,
                    tile,
                )
                > (result.candidate_buffers, result.buffers, result.block_size)
            ):
                result = candidate
            tile *= 2
    if result.scratch_bytes > vmem_bytes:
        raise ValueError("TopK scratch exceeds the SparseCore VMEM capacity")
    return result


def _pick_streaming_partition(
    batch_size: int,
    width: int,
    k: int,
    *,
    lanes: int,
    num_cores: int,
    num_subcores: int,
    plan_fits: Callable[[int], bool],
) -> int:
    """Pick the column-partition factor P so small batches fill the subcores.

    A row's P column slices run on P subcores of one SparseCore (the barrier
    and VMEM_SHARED are per core) and their slice Top-Ks merge exactly. P is
    a power of two bounded by subcore occupancy, slices wider than Top-K and
    VMEM; 1 leaves the single-row path unchanged.
    """
    if batch_size == 0:
        return 1
    max_p = min(num_subcores, num_cores * num_subcores // batch_size)
    p = 1
    while p * 2 <= max_p:
        nxt = p * 2
        if width // nxt < max(k, lanes) or not plan_fits(nxt):
            break
        p = nxt
    return p


def _ordered_bits(keys: jax.Array) -> jax.Array:
    """Preserve FP32 radix ordering, including signed zero."""
    bits = keys.view(jnp.uint32)
    sign = (bits.view(jnp.int32) >> (FP32_BITS - 1)).view(jnp.uint32)
    return bits ^ (sign | jnp.uint32(FP32_SIGN_BIT))


def _store_candidates(candidates, candidate_ids, count, keys, ids, mask):
    """Cache exact FP32 order keys once for all subsequent radix passes."""
    lanes = keys.shape[0]
    plsc.store_compressed(
        candidates.at[pl.ds(count, lanes)], _ordered_bits(keys), mask=mask
    )
    plsc.store_compressed(candidate_ids.at[pl.ds(count, lanes)], ids, mask=mask)


def _clear_histogram(histogram, *, unroll: int = 1):
    lanes = plsc.get_sparse_core_info().num_lanes
    histogram_size = histogram.shape[0]

    @plsc.parallel_loop(0, histogram_size, lanes, unroll=unroll)
    def clear(i):
        histogram[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)


def _sort_selected(
    candidate_ids, selected_ids, histogram, *, id_digits: int, unroll: int = 1
):
    lanes = plsc.get_sparse_core_info().num_lanes
    k = selected_ids.shape[0]
    histogram_size = histogram.shape[0]
    # Reuse selection scratch. Stable byte passes keep padding behind
    # valid IDs even when their low bytes coincide with the sentinel.

    def sort_digit(digit):
        _clear_histogram(histogram, unroll=unroll)

        @plsc.parallel_loop(0, k, lanes, unroll=unroll)
        def count_keys(i):
            bits = candidate_ids[pl.ds(i, lanes)].astype(jnp.uint32)
            digits = (bits >> digit * RADIX_BITS & jnp.uint32(RADIX_MASK)).astype(
                jnp.int32
            )
            counts, last = plsc.scan_count(digits)
            plsc.addupdate_scatter(histogram, [digits], counts, mask=last)

        @plsc.parallel_loop(
            0, histogram_size, lanes, unroll=unroll, carry=jnp.asarray(0, jnp.int32)
        )
        def prefix(i, total):
            counts = histogram[pl.ds(i, lanes)]
            cumulative = jnp.cumsum(counts)
            histogram[pl.ds(i, lanes)] = total + cumulative - counts
            return total + cumulative[-1]

        def scatter(vector, carry):
            i = vector * lanes
            ids = candidate_ids[pl.ds(i, lanes)]
            digits = (
                ids.astype(jnp.uint32) >> digit * RADIX_BITS & jnp.uint32(RADIX_MASK)
            ).astype(jnp.int32)
            counts, last = plsc.scan_count(digits)
            positions = plsc.load_gather(histogram, [digits]) + counts - 1
            plsc.store_scatter(selected_ids, [positions], ids)
            plsc.addupdate_scatter(histogram, [digits], counts, mask=last)
            return carry

        jax.lax.fori_loop(0, k // lanes, scatter, ())

        @plsc.parallel_loop(0, k, lanes, unroll=unroll)
        def copy_sorted(i):
            candidate_ids[pl.ds(i, lanes)] = selected_ids[pl.ds(i, lanes)]

    for digit in range(id_digits):
        sort_digit(digit)


def _compact(
    candidate_banks,
    id_banks,
    selected,
    selected_ids,
    histogram,
    num_candidates,
    *,
    candidate_buffers: int,
    id_digits: int,
    unroll: int = 1,
):
    """Radix-select exact (score, -column index) winners into the first bank."""
    lanes = plsc.get_sparse_core_info().num_lanes
    lane_ids = jnp.arange(lanes, dtype=jnp.int32)
    k = selected_ids.shape[0]
    histogram_size = histogram.shape[0]
    capacity = candidate_banks.shape[0] // candidate_buffers
    id_capacity = id_banks.shape[0] // candidate_buffers
    candidates = candidate_banks.at[pl.ds(0, capacity)]
    candidate_ids = id_banks.at[pl.ds(0, id_capacity)]

    @pl.when(num_candidates < k)
    def clear_padding():
        @plsc.parallel_loop(0, k, lanes, unroll=unroll)
        def clear_selected(i):
            selected[pl.ds(i, lanes)] = jnp.full(lanes, ORDERED_NEG_INF, jnp.uint32)
            selected_ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

    def cond(state):
        digit, remaining, found, bank = state
        return (digit >= 0) & (remaining > k - found)

    def radix_pass(state, *, by_id):
        digit, remaining, found, bank = state
        limit = pl.cdiv(remaining, lanes) * lanes

        next_bank = 1 - bank if candidate_buffers == 2 else bank
        source_keys = candidate_banks.at[pl.ds(bank * capacity, capacity)]
        source_ids = id_banks.at[pl.ds(bank * id_capacity, id_capacity)]
        destination_keys = candidate_banks.at[pl.ds(next_bank * capacity, capacity)]
        destination_ids = id_banks.at[pl.ds(next_bank * id_capacity, id_capacity)]

        _clear_histogram(histogram, unroll=unroll)

        def digits_for(bits):
            return bits >> (digit * RADIX_BITS).astype(jnp.uint32) & jnp.uint32(
                RADIX_MASK
            )

        @plsc.parallel_loop(0, limit, lanes, unroll=unroll)
        def fill_histogram(i):
            bits = (
                ~source_ids[pl.ds(i, lanes)].astype(jnp.uint32)
                if by_id
                else source_keys[pl.ds(i, lanes)]
            )
            digits = digits_for(bits)
            valid = i + lane_ids < remaining
            counts, mask = plsc.scan_count(digits, mask=valid)
            plsc.addupdate_scatter(histogram, [digits], counts, mask=mask)

        below = remaining - (k - found)
        init = (jnp.zeros(lanes, jnp.int32), jnp.zeros(lanes, jnp.int32))

        @plsc.parallel_loop(0, histogram_size, lanes, unroll=unroll, carry=init)
        def find_pivot(i, carry):
            pivot, prefix = carry
            cumulative = jnp.cumsum(histogram[pl.ds(i, lanes)])
            offset = plsc.all_reduce_ffs(prefix + cumulative > below)
            return (pivot + offset, prefix + cumulative[-1])

        pivot, _ = find_pivot

        def filter_candidates(i, carry):
            survivors, outputs = carry
            keys = source_keys[pl.ds(i, lanes)]
            ids = source_ids[pl.ds(i, lanes)]
            bits = ~ids.astype(jnp.uint32) if by_id else keys
            digits = digits_for(bits)
            valid = i + lane_ids < remaining
            accept = valid & (digits > pivot)
            survive = valid & (digits == pivot)
            plsc.store_compressed(selected.at[pl.ds(outputs, lanes)], keys, mask=accept)
            plsc.store_compressed(
                selected_ids.at[pl.ds(outputs, lanes)], ids, mask=accept
            )
            plsc.store_compressed(
                destination_keys.at[pl.ds(survivors, lanes)], keys, mask=survive
            )
            plsc.store_compressed(
                destination_ids.at[pl.ds(survivors, lanes)], ids, mask=survive
            )
            return (
                survivors + plsc.all_reduce_population_count(survive)[0],
                outputs + plsc.all_reduce_population_count(accept)[0],
            )

        initial = (jnp.asarray(0, jnp.int32), found)
        if candidate_buffers == 2:
            survivors, outputs = plsc.parallel_loop(
                0, limit, lanes, unroll=unroll, carry=initial
            )(filter_candidates)
        else:
            survivors, outputs = jax.lax.fori_loop(
                0,
                pl.cdiv(remaining, lanes),
                lambda i, carry: filter_candidates(i * lanes, carry),
                initial,
            )
        return (digit - 1, survivors, outputs, next_bank)

    _, remaining, found, bank = jax.lax.while_loop(
        cond,
        functools.partial(radix_pass, by_id=False),
        (
            jnp.int32(FP32_BITS // RADIX_BITS - 1),
            num_candidates,
            jnp.int32(0),
            jnp.int32(0),
        ),
    )
    _, remaining, found, bank = jax.lax.while_loop(
        cond,
        functools.partial(radix_pass, by_id=True),
        (jnp.int32(id_digits - 1), remaining, found, bank),
    )
    needed = jnp.minimum(remaining, k - found)
    initial = (
        jnp.full(lanes, jnp.uint32(UINT32_MAX)),
        jnp.full(lanes, -1, jnp.int32),
    )

    @plsc.parallel_loop(
        0, pl.cdiv(needed, lanes) * lanes, lanes, unroll=unroll, carry=initial
    )
    def postfill(i, carry):
        minimum, last_id = carry
        mask = i + lane_ids < needed
        keys = candidate_banks[pl.ds(bank * capacity + i, lanes)]
        ids = id_banks[pl.ds(bank * id_capacity + i, lanes)]
        plsc.store_compressed(selected.at[pl.ds(found + i, lanes)], keys, mask=mask)
        plsc.store_compressed(selected_ids.at[pl.ds(found + i, lanes)], ids, mask=mask)
        bits = jnp.where(mask, keys, jnp.uint32(UINT32_MAX))
        last_id = jnp.where(
            bits < minimum,
            ids,
            jnp.where(bits == minimum, jnp.maximum(last_id, ids), last_id),
        )
        return (jnp.minimum(minimum, bits), last_id)

    minimum, last_id = postfill
    threshold = jnp.min(minimum)
    threshold_id = jnp.max(jnp.where(minimum == threshold, last_id, -1))

    @plsc.parallel_loop(0, k, lanes, unroll=unroll)
    def retain_topk(i):
        candidates[pl.ds(i, lanes)] = selected[pl.ds(i, lanes)]
        candidate_ids[pl.ds(i, lanes)] = selected_ids[pl.ds(i, lanes)]

    return _Selection(
        jnp.minimum(num_candidates, k),
        threshold,
        threshold_id,
        num_candidates >= k,
    )


def _transfer_block(
    scores_ref,
    blocks,
    block_sems,
    row,
    block_id,
    slot,
    *,
    block_size: int,
    wait: bool,
    col_base: int = 0,
    slice_width: int | None = None,
):
    # col_base offsets the slice this subcore owns; slice_width bounds its tail.
    width = scores_ref.shape[1] if slice_width is None else slice_width
    num_blocks = pl.cdiv(width, block_size)

    def move(source, destination):
        copy = pltpu.make_async_copy(source, destination, block_sems.at[slot])
        if wait:
            copy.wait()
        else:
            copy.start()

    def full():
        move(
            scores_ref.at[row, pl.ds(col_base + block_id * block_size, block_size)],
            blocks.at[slot]
            if blocks.ndim == 2
            else blocks.at[pl.ds(slot * block_size, block_size)],
        )

    tail_size = width % block_size
    if tail_size:

        def tail():
            move(
                scores_ref.at[row, pl.ds(col_base + width - tail_size, tail_size)],
                blocks.at[slot, pl.ds(0, tail_size)],
            )

        if width < block_size:
            tail()
        else:
            jax.lax.cond(block_id == num_blocks - 1, tail, full)
    else:
        full()


def _filter_chunk(
    blocks,
    candidates,
    candidate_ids,
    slot,
    chunk_start,
    offset,
    length,
    selection: _Selection,
    *,
    chunk_size: int,
    exclude_neg_inf: bool,
    unroll: int = 1,
):
    lanes = plsc.get_sparse_core_info().num_lanes
    lane_ids = jnp.arange(lanes, dtype=jnp.int32)
    count, threshold, threshold_id, have_threshold = selection
    cutoff = jnp.where(have_threshold, threshold, jnp.uint32(ORDERED_NEG_INF))
    initial_id = -1 if exclude_neg_inf else jnp.iinfo(jnp.int32).max
    tie_id = jnp.where(have_threshold, threshold_id, initial_id)

    def append(full_block):
        # Match the packed scan's throughput; avoid predicate spills on tails.
        @plsc.parallel_loop(
            0,
            chunk_size,
            lanes,
            carry=count,
            unroll=2 * unroll if full_block else unroll,
        )
        def scan(i, count):
            keys = blocks[slot, pl.ds(chunk_start + i, lanes)]
            ids = offset + i + lane_ids
            bits = _ordered_bits(keys)
            accept = (bits > cutoff) | ((bits == cutoff) & (ids < tie_id))
            if not full_block:
                accept = accept & (ids < length)
            _store_candidates(candidates, candidate_ids, count, keys, ids, accept)
            return count + plsc.all_reduce_population_count(accept)[0]

        return scan

    return jax.lax.cond(
        offset + chunk_size <= length, lambda: append(True), lambda: append(False)
    )


def _logical_blocks_epilogue(
    logical_blocks: jax.Array,
    valid_lengths: jax.Array,
    start_request: jax.Array | None,
    width: int,
    k: int,
) -> TopKEpilogue:
    """Map compact winners using the existing request/query length window."""
    if (
        valid_lengths.ndim != 2
        or logical_blocks.ndim != 2
        or logical_blocks.dtype != jnp.int32
        or logical_blocks.shape[0] < valid_lengths.shape[0]
        or not logical_blocks.shape[1]
        or width % logical_blocks.shape[1]
    ):
        raise ValueError(
            "logical_blocks must be int32 [requests, blocks] covering "
            "the request/query length window, with blocks dividing N"
        )
    rows, queries = valid_lengths.size, valid_lengths.shape[1]
    records_per_block = width // logical_blocks.shape[1]
    alignment = pltpu.get_tpu_info().num_lanes
    lanes = plsc.get_sparse_core_info().num_lanes
    output_width = pl.cdiv(k, alignment) * alignment
    mapping_width = pl.cdiv(logical_blocks.shape[1], alignment) * alignment
    logical_blocks = jnp.pad(
        logical_blocks,
        ((0, 0), (0, mapping_width - logical_blocks.shape[1])),
        constant_values=-1,
    )
    start = jnp.asarray(0 if start_request is None else start_request, jnp.int32)
    owners = start + jnp.arange(rows, dtype=jnp.int32) // queries
    row_metadata = jnp.broadcast_to(owners[:, None], (rows, lanes))

    def finish_row(refs, candidate_ids, row_metadata_ref, sem, sort_selected):
        # Reuse the ID tail after selection and overlap mapping DMA with sorting.
        mapping_copy = pltpu.make_async_copy(
            refs[0].at[row_metadata_ref[pl.ds(lanes, lanes)][0]],
            candidate_ids.at[pl.ds(output_width, mapping_width)],
            sem,
        )
        mapping_copy.start()
        sort_selected()
        mapping_copy.wait()

        @plsc.parallel_loop(0, k, lanes)
        def map_selected(i):
            slots = candidate_ids[pl.ds(i, lanes)]
            blocks = plsc.load_gather(
                candidate_ids,
                [output_width + jnp.maximum(slots, 0) // records_per_block],
            )
            logical = blocks * records_per_block + slots % records_per_block
            candidate_ids[pl.ds(i, lanes)] = jnp.where(slots >= 0, logical, -1)

    return TopKEpilogue((logical_blocks,), row_metadata, mapping_width, finish_row)


def streaming_topk(
    scores: jax.Array,
    *,
    valid_lengths: jax.Array | None = None,
    k: int = 512,
    block_size: int | None = None,
    reconstruct_threshold: int = 8192,
    exclude_neg_inf: bool = False,
    loop_unroll: int = 1,
    bootstrap_multiplier: int = 4,
    scheduling_group_id: int | None = None,
    epilogue: TopKEpilogue | None = None,
    sorted_indices: bool = True,
    logical_blocks: jax.Array | None = None,
    start_request: jax.Array | None = None,
) -> jax.Array:
    """Return Top-K column indices for 2-D scores, ascending then -1 padded.

    Scores follow exact FP32 radix order (+0 above -0); equal bit patterns
    prefer smaller column indices regardless of block visitation order.
    The reproducible block shuffle is input-independent, not a fresh random
    permutation per invocation. NaNs are unsupported. Negative infinity is
    eligible unless exclude_neg_inf is set. Lengths are clipped to [0, N].
    Scores have shape [rows, N]; lengths contain one int32 entry per row.
    An optional epilogue transforms selected indices before writeback.
    Set sorted_indices=False when the consumer already sorts the selected IDs.
    Alternatively, logical_blocks maps compact blocks to logical IDs using
    valid_lengths [requests, queries] and an optional request window offset.
    Valid block IDs must increase; padded blocks must have ineligible scores.
    """
    if scores.ndim != 2 or scores.dtype != jnp.float32:
        raise ValueError("scores must be a rank-2 float32 array")
    batch_size, width = scores.shape
    if not 1 <= k <= width:
        raise ValueError("k must be in [1, N]")
    if (block_size is not None and block_size <= 0) or reconstruct_threshold <= 0:
        raise ValueError("block_size and reconstruct_threshold must be positive")
    if valid_lengths is not None and (
        valid_lengths.size != batch_size or valid_lengths.dtype != jnp.int32
    ):
        raise ValueError("valid_lengths must contain one int32 entry per score row")
    if logical_blocks is not None:
        if epilogue is not None or valid_lengths is None:
            raise ValueError("logical_blocks requires valid_lengths and no epilogue")
        epilogue = _logical_blocks_epilogue(
            logical_blocks, valid_lengths, start_request, width, k
        )
    valid_width = width
    # HBM arrays retain TensorCore tiling when SparseCore consumes them.
    hbm_alignment = pltpu.get_tpu_info().num_lanes
    input_alignment = (
        hbm_alignment
        if block_size is None or block_size % hbm_alignment == 0
        else block_size
    )
    padding = -width % input_alignment
    if padding:
        scores = jnp.pad(
            scores,
            ((0, 0), (0, padding)),
            constant_values=-jnp.inf,
        )
        width += padding
    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    if any(n % lanes for n in (k, block_size or lanes, reconstruct_threshold)):
        raise ValueError("k, block_size and reconstruct_threshold must be lane-aligned")
    lengths = (
        jnp.full((batch_size,), valid_width, jnp.int32)
        if valid_lengths is None
        else jnp.clip(valid_lengths.reshape(-1), 0, valid_width)
    )
    row_metadata = None if epilogue is None else epilogue.row_metadata
    metadata_only = row_metadata is not None and valid_lengths is None
    if row_metadata is not None and valid_lengths is None:
        lengths = row_metadata
        metadata_width = lanes + lengths.shape[1]
    elif epilogue is None:
        lengths = jnp.pad(lengths, (0, -batch_size % hbm_alignment))
        metadata_width = lanes
    else:
        lengths = jnp.broadcast_to(lengths[:, None], (batch_size, lanes))
        if epilogue.row_metadata is not None:
            lengths = jnp.concatenate((lengths, epilogue.row_metadata), axis=1)
        metadata_width = lengths.shape[1]
    output_width = pl.cdiv(k, hbm_alignment) * hbm_alignment
    index_scratch_size = 0 if epilogue is None else epilogue.index_scratch_size
    score_bytes, id_bytes = scores.dtype.itemsize, jnp.dtype(jnp.int32).itemsize
    num_subcores = info.num_subcores

    def slice_width_for(p: int) -> int:
        # Slice starts are HBM DMA offsets, so they keep TensorCore tiling.
        if p == 1:
            return width
        return pl.cdiv(pl.cdiv(width, p), hbm_alignment) * hbm_alignment

    def plan_for(p: int, single_shot: bool = False) -> _TopKPlan:
        return _topk_plan(
            slice_width_for(p),
            k,
            metadata_width,
            index_scratch_size,
            slice_width_for(p) if single_shot else block_size,
            reconstruct_threshold,
            bootstrap_multiplier,
            lanes=lanes,
            hbm_alignment=hbm_alignment,
            vmem_bytes=info.vmem_capacity_bytes,
            score_bytes=score_bytes,
            id_bytes=id_bytes,
            merge_capacity=p * k if p > 1 else 0,
            single_shot=single_shot,
        )

    def try_plan(p: int, single_shot: bool = False) -> _TopKPlan | None:
        try:
            return plan_for(p, single_shot)
        except ValueError:
            return None

    # Column-partition small batches so idle subcores share each row's scan.
    partition = _pick_streaming_partition(
        batch_size,
        width,
        k,
        lanes=lanes,
        num_cores=info.num_cores,
        num_subcores=num_subcores,
        plan_fits=lambda p: try_plan(p) is not None,
    )
    slice_width = slice_width_for(partition)
    # A slice that fits one tile takes one DMA and one compaction; streaming
    # it would serialize the next block's DMA behind the bootstrap compaction.
    single = try_plan(partition, True) if partition > 1 and block_size is None else None
    single_shot = single is not None
    config = single if single_shot else plan_for(partition)
    extra = partition * slice_width - width
    if extra:
        scores = jnp.pad(scores, ((0, 0), (0, extra)), constant_values=-jnp.inf)
        width += extra
    block_size = config.block_size
    capacity, id_capacity = config.capacity, config.id_capacity
    candidate_buffers, buffers = config.candidate_buffers, config.buffers
    # Block layout, shuffle and bootstrap are per column slice; all P slices in
    # a cooperative group share the same slice length and block count.
    num_blocks = pl.cdiv(slice_width, block_size)
    order_width = pl.cdiv(num_blocks, lanes) * lanes
    order = list(range(1, num_blocks))
    random.Random(0).shuffle(order)
    order.insert(0, 0)
    order.extend([0] * (order_width - num_blocks))
    # Bootstrap winners remain resident: its width tunes the initial cutoff
    # and scan granularity without approximating the final selection.
    bootstrap_width = (
        block_size if single_shot else min(block_size, bootstrap_multiplier * k)
    )
    chunk_size = math.gcd(block_size, bootstrap_width)
    histogram_size = RADIX_BUCKETS
    id_digits = max(1, pl.cdiv((valid_width - 1).bit_length(), RADIX_BITS))
    groups_per_core = num_subcores // partition
    num_cores = (
        min(info.num_cores, pl.cdiv(batch_size, groups_per_core)) if batch_size else 1
    )
    num_waves = pl.cdiv(batch_size, num_cores * groups_per_core) if batch_size else 0
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=num_cores,
    )
    out_type = jax.ShapeDtypeStruct((batch_size, output_width), jnp.int32)
    # Per-wave-parity slots let a subcore publish wave w + 1 while its group
    # leader may still be merging wave w; one barrier per wave then suffices.
    shared_scratch = (
        (
            pltpu.VMEM_SHARED((2 * num_subcores * k,), jnp.uint32),
            pltpu.VMEM_SHARED((2 * num_subcores * k,), jnp.int32),
        )
        if partition > 1
        else ()
    )

    @pl.kernel(
        out_type=out_type,
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((buffers, block_size), jnp.float32),
            pltpu.VMEM((candidate_buffers * capacity,), jnp.uint32),
            pltpu.VMEM((candidate_buffers * id_capacity,), jnp.int32),
            pltpu.VMEM((k,), jnp.uint32),
            pltpu.VMEM((k,), jnp.int32),
            pltpu.VMEM((histogram_size,), jnp.int32),
            pltpu.VMEM((metadata_width,), jnp.int32),
            pltpu.VMEM((order_width,), jnp.int32),
            pltpu.SemaphoreType.DMA((buffers,)),
            pltpu.SemaphoreType.DMA,
            *shared_scratch,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="streaming_topk",
    )
    def kernel(scores_ref, lengths_ref, *refs):
        metadata_inputs = 0 if epilogue is None else len(epilogue.inputs)
        out_indices_ref, *scratch = refs[metadata_inputs:]
        (
            blocks,
            candidate_banks,
            id_banks,
            selected,
            selected_ids,
            histogram,
            row_lengths,
            visit_order,
            block_sems,
            sem,
            *shared,
        ) = scratch
        candidates = candidate_banks.at[pl.ds(0, capacity)]
        candidate_ids = id_banks.at[pl.ds(0, id_capacity)]
        lane_ids = jnp.arange(lanes, dtype=jnp.int32)
        for start in range(0, order_width, lanes):
            values = jnp.zeros((lanes,), jnp.int32)
            for lane, block in enumerate(order[start : start + lanes]):
                if block:
                    values = jnp.where(lane_ids == lane, block, values)
            visit_order[pl.ds(start, lanes)] = values

        store_candidates = functools.partial(
            _store_candidates, candidates, candidate_ids
        )
        sort_selected = (
            functools.partial(
                _sort_selected,
                candidate_ids,
                selected_ids,
                histogram,
                id_digits=id_digits,
                unroll=loop_unroll,
            )
            if sorted_indices
            else lambda: None
        )
        compact = functools.partial(
            _compact,
            candidate_banks,
            id_banks,
            selected,
            selected_ids,
            histogram,
            candidate_buffers=candidate_buffers,
            id_digits=id_digits,
            unroll=loop_unroll,
        )
        filter_chunk = functools.partial(
            _filter_chunk,
            blocks,
            candidates,
            candidate_ids,
            chunk_size=chunk_size,
            exclude_neg_inf=exclude_neg_inf,
            unroll=loop_unroll,
        )

        def lookup(position):
            indices = jnp.full((lanes,), jnp.minimum(position, num_blocks - 1))
            return plsc.load_gather(visit_order, [indices])[0]

        def next_valid(position, length):
            def cond(state):
                step, block_id = state
                return (step < num_blocks) & (block_id * block_size >= length)

            def advance(state):
                step = state[0] + 1
                return (step, lookup(step))

            return jax.lax.while_loop(cond, advance, (position, lookup(position)))

        subcore = jax.lax.axis_index("subcore")
        core = jax.lax.axis_index("core")

        def load_length(row):
            target = row_lengths
            if metadata_only:
                source = lengths_ref.at[row]
                target = target.at[lanes:]
            elif epilogue is None:
                source = lengths_ref.at[pl.ds(row // lanes * lanes, lanes)]
            else:
                source = lengths_ref.at[row]
            copy = pltpu.make_async_copy(source, target, sem)
            copy.start()
            copy.wait()
            if metadata_only:
                row_lengths[:lanes] = jnp.full(lanes, valid_width, jnp.int32)
                return jnp.int32(valid_width)
            if epilogue is None:
                return plsc.load_gather(row_lengths, [jnp.full(lanes, row % lanes)])[0]
            return jnp.min(row_lengths[:lanes])

        def write_row(row):
            if epilogue is None:
                sort_selected()
            else:
                epilogue.apply(
                    refs[:metadata_inputs],
                    candidate_ids,
                    row_lengths,
                    sem,
                    sort_selected,
                )
            pltpu.async_copy(
                candidate_ids.at[:output_width], out_indices_ref.at[row], sem
            ).wait()

        def select_slice(row, length, col_base=0):
            """Leave the slice's exact Top-K, sentinel padded, in candidates[:k].

            length counts valid columns from col_base; emitted IDs are global.
            """
            transfer = functools.partial(
                _transfer_block,
                scores_ref,
                blocks,
                block_sems,
                block_size=block_size,
                col_base=col_base,
                slice_width=slice_width,
            )
            # Global bound: also masks stale lanes of a slice's partial tail block.
            limit = col_base + length
            row_blocks = pl.cdiv(length, block_size)

            def prime():
                position, block_id = next_valid(jnp.int32(0), length)
                transfer(row, block_id, jnp.int32(0), wait=False)
                return (position, block_id)

            position, block_id = jax.lax.cond(
                row_blocks > 0, prime, lambda: (jnp.int32(0), jnp.int32(0))
            )

            @plsc.parallel_loop(0, output_width, lanes, unroll=loop_unroll)
            def clear_outputs(i):
                candidates[pl.ds(i, lanes)] = jnp.full(
                    lanes, ORDERED_NEG_INF, jnp.uint32
                )
                candidate_ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

            def scan_block(iteration, state):
                selection, position, block_id = state
                slot = iteration % buffers

                def prefetch():
                    next_position, next_id = next_valid(position + 1, length)
                    transfer(row, next_id, (iteration + 1) % buffers, wait=False)
                    return (next_position, next_id)

                def advance():
                    return jax.lax.cond(
                        iteration + 1 < row_blocks,
                        prefetch,
                        lambda: (position, block_id),
                    )

                next_position, next_id = (
                    advance() if buffers == 2 else (position, block_id)
                )
                jax.lax.cond(
                    iteration > 0,
                    lambda: transfer(row, block_id, slot, wait=True),
                    lambda: None,
                )

                def scan_chunk(visit, selection: _Selection):
                    chunk_start = (block_size // chunk_size - 1 - visit) * chunk_size
                    offset = col_base + block_id * block_size + chunk_start
                    count = filter_chunk(slot, chunk_start, offset, limit, selection)
                    should_compact = (
                        count >= k + reconstruct_threshold
                    ) | ~selection.have_threshold & (count >= k)
                    return jax.lax.cond(
                        should_compact,
                        lambda: compact(count),
                        lambda: selection._replace(count=count),
                    )

                # Bootstrap consumes a single-chunk block on its first visit.
                # Express the remaining zero-or-one scan directly, without a loop.
                if block_size == chunk_size:
                    selection = jax.lax.cond(
                        iteration > 0,
                        lambda: scan_chunk(0, selection),
                        lambda: selection,
                    )
                else:
                    chunks_to_scan = block_size // chunk_size - jnp.where(
                        iteration == 0, bootstrap_width // chunk_size, 0
                    )
                    selection = jax.lax.fori_loop(
                        jnp.int32(0), chunks_to_scan, scan_chunk, selection
                    )
                selection = jax.lax.cond(
                    (iteration + 1 == row_blocks) & (selection.count > k),
                    lambda: compact(selection.count),
                    lambda: selection,
                )
                if buffers == 1:
                    next_position, next_id = advance()
                return (selection, next_position, next_id)

            initial_threshold = jnp.uint32(ORDERED_NEG_INF if exclude_neg_inf else 0)

            def establish_threshold():
                transfer(row, block_id, jnp.int32(0), wait=True)

                @plsc.parallel_loop(
                    0,
                    bootstrap_width,
                    lanes,
                    carry=jnp.asarray(0, jnp.int32),
                    unroll=loop_unroll,
                )
                def bootstrap(i, count):
                    keys = blocks[0, pl.ds(i, lanes)]
                    ids = col_base + block_id * block_size + i + lane_ids
                    accept = ids < limit
                    if exclude_neg_inf:
                        accept = accept & (keys > -jnp.inf)
                    store_candidates(count, keys, ids, accept)
                    return count + plsc.all_reduce_population_count(accept)[0]

                selection = compact(bootstrap)
                have = selection.have_threshold
                return selection._replace(
                    threshold=jnp.where(have, selection.threshold, initial_threshold),
                )

            seed = jax.lax.cond(
                row_blocks > 0,
                establish_threshold,
                lambda: _Selection(
                    jnp.int32(0),
                    initial_threshold,
                    jnp.int32(-1),
                    jnp.bool_(False),
                ),
            )
            if bootstrap_width < min(valid_width, slice_width):
                jax.lax.fori_loop(0, row_blocks, scan_block, (seed, position, block_id))

        if partition == 1:
            first_row = core * mesh.num_subcores + subcore
            row_stride = mesh.num_cores * mesh.num_subcores

            @pl.loop(first_row, batch_size, step=row_stride)
            def process_row(row):
                select_slice(row, load_length(row))
                write_row(row)

        else:
            # Exact under the strict (key desc, ID asc) order: every row winner
            # is also a winner of its slice, so merging P slice Top-Ks suffices.
            shared_keys, shared_ids = shared
            group = subcore // partition
            part = subcore % partition
            col_base = part * slice_width
            merged = partition * k
            pairs = ((candidates, shared_keys), (candidate_ids, shared_ids))

            # A fixed wave count keeps every subcore's barrier count equal.
            @pl.loop(0, num_waves)
            def process_wave(wave):
                row = (wave * mesh.num_cores + core) * groups_per_core + group
                active = row < batch_size
                slot = wave % 2
                own = pl.multiple_of((slot * num_subcores + subcore) * k, lanes)

                @pl.when(active)
                def select_part():
                    length = jnp.clip(load_length(row) - col_base, 0, slice_width)
                    select_slice(row, length, col_base)

                    # Padding must rank below every real key, including -inf,
                    # whose ID bytes could otherwise tie the -1 sentinel.
                    @plsc.parallel_loop(0, k, lanes, unroll=loop_unroll)
                    def demote_padding(i):
                        keys = candidates[pl.ds(i, lanes)]
                        ids = candidate_ids[pl.ds(i, lanes)]
                        candidates[pl.ds(i, lanes)] = jnp.where(
                            ids < 0, jnp.uint32(0), keys
                        )

                    for local, peer in pairs:
                        pltpu.sync_copy(local.at[pl.ds(0, k)], peer.at[pl.ds(own, k)])

                plsc.subcore_barrier()

                @pl.when(active & (part == 0))
                def merge():
                    peers = pl.multiple_of(
                        (slot * num_subcores + group * partition) * k, lanes
                    )
                    for local, peer in pairs:
                        pltpu.sync_copy(
                            peer.at[pl.ds(peers, merged)], local.at[pl.ds(0, merged)]
                        )
                    compact(jnp.int32(merged))
                    write_row(row)

    inputs = () if epilogue is None else epilogue.inputs
    result = kernel(scores, lengths, *inputs)
    if output_width != k:
        result = result[:, :k]
    if scheduling_group_id is not None:
        result = xla_metadata.set_xla_metadata(
            result, _scheduling_group_id=scheduling_group_id
        )
    return result


def select_candidate_tokens(
    scores: jax.Array,
    candidates: jax.Array,
    *,
    k: int,
    block_size: int,
    unordered: bool,
    sorted_indices: bool = True,
    scheduling_group_id: int | None = None,
) -> jax.Array:
    """Select compact scores and map winners to logical token IDs."""
    if k == 0:
        return jnp.empty((scores.shape[0], 0), jnp.int32)
    if (
        unordered
        and jax.default_backend() == "tpu"
        and k % plsc.get_sparse_core_info().num_lanes == 0
    ):
        return streaming_topk(
            scores,
            valid_lengths=jnp.full((scores.shape[0], 1), scores.shape[1], jnp.int32),
            logical_blocks=candidates,
            k=k,
            exclude_neg_inf=True,
            sorted_indices=sorted_indices,
            reconstruct_threshold=tuned_block_sizes.PACKED_FP8_CANDIDATE_TOPK_CONFIG.get(
                (*scores.shape, k), 8192
            ),
            scheduling_group_id=scheduling_group_id,
        )
    values, slots = jax.lax.top_k(scores, k)
    logical = (
        jnp.take_along_axis(candidates, slots // block_size, axis=1) * block_size
        + slots % block_size
    )
    valid = jnp.isfinite(values) if unordered else values > -jnp.inf
    sentinel = jnp.iinfo(jnp.int32).max
    logical = jnp.sort(jnp.where(valid, logical, sentinel))
    return jnp.where(logical < sentinel, logical, -1)


__all__ = ["TopKEpilogue", "select_candidate_tokens", "streaming_topk"]
