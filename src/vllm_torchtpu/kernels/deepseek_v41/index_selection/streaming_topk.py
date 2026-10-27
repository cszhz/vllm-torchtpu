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
    sample_width: int
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
) -> _TopKPlan:
    output_width = pl.cdiv(k, hbm_alignment) * hbm_alignment

    def plan(tile, sample=0):
        bootstrap = min(tile, bootstrap_multiplier * k)
        capacity = (
            width
            if sample
            else max(
                k + reconstruct_threshold + math.gcd(tile, bootstrap),
                bootstrap,
                output_width,
            )
        )
        id_capacity = max(capacity, output_width + index_scratch_size)
        order_width = pl.cdiv(pl.cdiv(width, tile), lanes) * lanes
        fixed = k * (score_bytes + id_bytes) + id_bytes * (RADIX_BUCKETS + order_width)

        def scratch(candidates, inputs):
            return (
                fixed
                + candidates * (capacity * score_bytes + id_capacity * id_bytes)
                + inputs * tile * score_bytes
                + (inputs if sample else 1) * metadata_width * id_bytes
            )

        candidates = 2 if scratch(2, 1) <= vmem_bytes else 1
        inputs = 2 if scratch(candidates, 2) <= vmem_bytes else 1
        return _TopKPlan(
            tile,
            sample,
            capacity,
            id_capacity,
            candidates,
            inputs,
            scratch(candidates, inputs),
        )

    # Representative independent ranks: reserve a 1/lanes rescan-work budget.
    # For expected sample hits mu, P(hits >= lanes) <= exp(-(lanes-mu)^2/(2*lanes)).
    # Fixed stratified samples need not follow that model; exact rescans remain.
    margin = math.ceil(math.sqrt(2 * lanes * math.log(lanes)))
    sample = width * max(0, lanes - margin) // k // lanes * lanes
    if lanes < sample < min(width, bootstrap_multiplier * k):
        resident = plan(width, sample)
        if resident.candidate_buffers == resident.buffers == 2:
            return resident

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


def _last_radix_digit(difference, radix_bits):
    """Find the highest differing digit; identical keys need no radix pass."""
    lanes = plsc.get_sparse_core_info().num_lanes
    lane = jnp.arange(lanes, dtype=jnp.int32)
    digits = pl.cdiv(FP32_BITS, radix_bits)
    shifts = (jnp.minimum(lane, digits - 1) * radix_bits).astype(jnp.uint32)
    nonzero = (jnp.full(lanes, difference) >> shifts) != 0
    return plsc.all_reduce_population_count(nonzero & (lane < digits))[0] - 1


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
    ordered_ids: bool = False,
    key_bounds=None,
):
    """Radix-select exact (score, -column index) winners into the first bank."""
    lanes = plsc.get_sparse_core_info().num_lanes
    lane_ids = jnp.arange(lanes, dtype=jnp.int32)
    k = selected_ids.shape[0]
    radix_bits = (
        max(lanes.bit_length() - 1, pl.cdiv(FP32_BITS, lanes))
        if ordered_ids
        else RADIX_BITS
    )
    histogram = histogram.at[: 1 << radix_bits]
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
            return bits >> (digit * radix_bits).astype(jnp.uint32) & jnp.uint32(
                (1 << radix_bits) - 1
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
            survivors, outputs, *bounds = carry
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
            if ordered_ids:
                bounds = (
                    jnp.minimum(
                        bounds[0], jnp.where(survive, keys, jnp.uint32(UINT32_MAX))
                    ),
                    jnp.maximum(bounds[1], jnp.where(survive, keys, jnp.uint32(0))),
                )
            return (
                survivors + plsc.all_reduce_population_count(survive)[0],
                outputs + plsc.all_reduce_population_count(accept)[0],
                *bounds,
            )

        initial = (jnp.asarray(0, jnp.int32), found)
        if ordered_ids:
            initial += (
                jnp.full(lanes, UINT32_MAX, jnp.uint32),
                jnp.zeros(lanes, jnp.uint32),
            )
        if candidate_buffers == 2:
            survivors, outputs, *bounds = plsc.parallel_loop(
                0, limit, lanes, unroll=unroll, carry=initial
            )(filter_candidates)
        else:
            survivors, outputs, *bounds = jax.lax.fori_loop(
                0,
                pl.cdiv(remaining, lanes),
                lambda i, carry: filter_candidates(i * lanes, carry),
                initial,
            )
        next_digit = digit - 1
        if ordered_ids:
            difference = jnp.min(bounds[0]) ^ jnp.max(bounds[1])
            next_digit = jnp.minimum(
                next_digit, _last_radix_digit(difference, radix_bits)
            )
        return (next_digit, survivors, outputs, next_bank)

    first_digit = jnp.int32(pl.cdiv(FP32_BITS, radix_bits) - 1)
    if key_bounds is not None:
        difference = key_bounds[0] ^ key_bounds[1]
        first_digit = _last_radix_digit(difference, radix_bits)
    _, remaining, found, bank = jax.lax.while_loop(
        cond,
        functools.partial(radix_pass, by_id=False),
        (
            first_digit,
            num_candidates,
            jnp.int32(0),
            jnp.int32(0),
        ),
    )
    # Stable filtering preserves ascending IDs in the resident candidate bank.
    if not ordered_ids:
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
    scores_ref, blocks, block_sems, row, block_id, slot, *, block_size: int, wait: bool
):
    width = scores_ref.shape[1]
    num_blocks = pl.cdiv(width, block_size)

    def move(source, destination):
        copy = pltpu.make_async_copy(source, destination, block_sems.at[slot])
        if wait:
            copy.wait()
        else:
            copy.start()

    def full():
        move(
            scores_ref.at[row, pl.ds(block_id * block_size, block_size)],
            blocks.at[slot]
            if blocks.ndim == 2
            else blocks.at[pl.ds(slot * block_size, block_size)],
        )

    tail_size = width % block_size
    if tail_size:

        def tail():
            move(
                scores_ref.at[row, pl.ds(width - tail_size, tail_size)],
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


def _logical_block_epilogue(blocks, lengths, width, k, start):
    """Preserve the original candidate indexer's compact-to-logical mapping."""
    if (
        lengths is None
        or lengths.ndim != 2
        or blocks.ndim != 2
        or blocks.dtype != jnp.int32
        or blocks.shape[0] < lengths.shape[0]
        or not blocks.shape[1]
        or width % blocks.shape[1]
    ):
        raise ValueError(
            "Logical blocks require request-major lengths and aligned scores"
        )
    requests, queries = lengths.shape
    records_per_block = width // blocks.shape[1]
    alignment = pltpu.get_tpu_info().num_lanes
    lanes = plsc.get_sparse_core_info().num_lanes
    output_width = pl.cdiv(k, alignment) * alignment
    mapping_width = pl.cdiv(blocks.shape[1], alignment) * alignment
    blocks = jnp.pad(
        blocks, ((0, 0), (0, mapping_width - blocks.shape[1])), constant_values=-1
    )
    owners = jnp.repeat(jnp.arange(requests, dtype=jnp.int32), queries)
    owners += jnp.asarray(0 if start is None else start, jnp.int32)
    metadata = jnp.broadcast_to(owners[:, None], (requests * queries, lanes))

    def finish(refs, ids, row_metadata, sem, sort_selected):
        copy = pltpu.make_async_copy(
            refs[0].at[row_metadata[pl.ds(lanes, lanes)][0]],
            ids.at[pl.ds(output_width, mapping_width)],
            sem,
        )
        copy.start()
        sort_selected()
        copy.wait()

        @plsc.parallel_loop(0, k, lanes)
        def remap(i):
            slots = ids[pl.ds(i, lanes)]
            selected = plsc.load_gather(
                ids, [output_width + jnp.maximum(slots, 0) // records_per_block]
            )
            ids[pl.ds(i, lanes)] = jnp.where(
                slots >= 0, selected * records_per_block + slots % records_per_block, -1
            )

    return TopKEpilogue((blocks,), metadata, mapping_width, finish)


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
        if epilogue is not None:
            raise ValueError("Use logical_blocks or an explicit epilogue, not both")
        epilogue = _logical_block_epilogue(
            logical_blocks, valid_lengths, width, k, start_request
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
    config = _topk_plan(
        width,
        k,
        metadata_width,
        index_scratch_size,
        block_size,
        reconstruct_threshold,
        bootstrap_multiplier,
        lanes=lanes,
        hbm_alignment=hbm_alignment,
        vmem_bytes=info.vmem_capacity_bytes,
        score_bytes=score_bytes,
        id_bytes=id_bytes,
    )
    block_size, sample_width = config.block_size, config.sample_width
    capacity, id_capacity = config.capacity, config.id_capacity
    candidate_buffers, buffers = config.candidate_buffers, config.buffers
    resident_sample = sample_width > 0
    scan_unroll = max(1, hbm_alignment // lanes) * loop_unroll
    num_blocks = pl.cdiv(width, block_size)
    order_width = pl.cdiv(num_blocks, lanes) * lanes
    order = list(range(1, num_blocks))
    random.Random(0).shuffle(order)
    order.insert(0, 0)
    order.extend([0] * (order_width - num_blocks))
    # Bootstrap winners remain resident: its width tunes the initial cutoff
    # and scan granularity without approximating the final selection.
    bootstrap_width = min(block_size, bootstrap_multiplier * k)
    chunk_size = math.gcd(block_size, bootstrap_width)
    histogram_size = RADIX_BUCKETS
    id_digits = max(1, pl.cdiv((valid_width - 1).bit_length(), RADIX_BITS))
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=min(info.num_cores, pl.cdiv(batch_size, info.num_subcores))
        if batch_size
        else 1,
    )
    out_type = jax.ShapeDtypeStruct((batch_size, output_width), jnp.int32)

    @pl.kernel(
        out_type=out_type,
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM(
                (buffers * block_size,) if resident_sample else (buffers, block_size),
                jnp.float32,
            ),
            pltpu.VMEM((candidate_buffers * capacity,), jnp.uint32),
            pltpu.VMEM((candidate_buffers * id_capacity,), jnp.int32),
            pltpu.VMEM((k,), jnp.uint32),
            pltpu.VMEM((k,), jnp.int32),
            pltpu.VMEM((histogram_size,), jnp.int32),
            pltpu.VMEM(
                (buffers * metadata_width if resident_sample else metadata_width,),
                jnp.int32,
            ),
            pltpu.VMEM((order_width,), jnp.int32),
            pltpu.SemaphoreType.DMA((buffers,)),
            pltpu.SemaphoreType.DMA,
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
        transfer = functools.partial(
            _transfer_block, scores_ref, blocks, block_sems, block_size=block_size
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
        first_row = core * mesh.num_subcores + subcore
        row_stride = mesh.num_cores * mesh.num_subcores

        def metadata_copy(row, slot):
            target = row_lengths.at[pl.ds(slot * metadata_width, metadata_width)]
            if metadata_only:
                source = lengths_ref.at[row]
                target = target.at[lanes:]
            elif epilogue is None:
                source = lengths_ref.at[pl.ds(row // lanes * lanes, lanes)]
            else:
                source = lengths_ref.at[row]
            semaphore = block_sems.at[slot] if resident_sample else sem
            return pltpu.make_async_copy(source, target, semaphore)

        if resident_sample:

            @pl.when(first_row < batch_size)
            def prime_row():
                transfer(first_row, jnp.int32(0), jnp.int32(0), wait=False)
                metadata_copy(first_row, jnp.int32(0)).start()

        @pl.loop(first_row, batch_size, step=row_stride)
        def process_row(row):
            score_slot = (
                ((row - first_row) // row_stride) % buffers
                if resident_sample
                else jnp.int32(0)
            )
            row_length = row_lengths.at[
                pl.ds(score_slot * metadata_width, metadata_width)
            ]
            if resident_sample:
                # Both waits share a counter; read either buffer only after both.
                transfer(row, jnp.int32(0), score_slot, wait=True)
                metadata_copy(row, score_slot).wait()
            else:
                copy = metadata_copy(row, score_slot)
                copy.start()
                copy.wait()
            if metadata_only:
                row_length[:lanes] = jnp.full(lanes, valid_width, jnp.int32)
                length = jnp.int32(valid_width)
            elif epilogue is None:
                length = plsc.load_gather(row_length, [jnp.full(lanes, row % lanes)])[0]
            else:
                length = jnp.min(row_length[:lanes])
            row_blocks = pl.cdiv(length, block_size)

            def prime():
                position, block_id = next_valid(jnp.int32(0), length)
                transfer(row, block_id, jnp.int32(0), wait=False)
                return (position, block_id)

            if resident_sample:

                @pl.when(row + row_stride < batch_size)
                def prefetch_row():
                    transfer(
                        row + row_stride,
                        jnp.int32(0),
                        (score_slot + 1) % buffers,
                        wait=False,
                    )
                    metadata_copy(row + row_stride, (score_slot + 1) % buffers).start()

                position, block_id = jnp.int32(0), jnp.int32(0)
            else:
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
                    offset = block_id * block_size + chunk_start
                    count = filter_chunk(slot, chunk_start, offset, length, selection)
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

            if resident_sample:

                @pl.when(row_blocks > 0)
                def resident_select():
                    floor = jnp.uint32(ORDERED_NEG_INF + int(exclude_neg_inf))
                    initial = jnp.full(lanes, floor, jnp.uint32)

                    @plsc.parallel_loop(0, sample_width, lanes, carry=initial)
                    def seed_top(i, top):
                        ids = ((i + lane_ids) * 2 + 1) * width // (2 * sample_width)
                        keys = plsc.load_gather(
                            blocks.at[pl.ds(score_slot * block_size, block_size)], [ids]
                        )
                        bits = jnp.where(ids < length, _ordered_bits(keys), floor)
                        incoming, _ = plsc.sort_key_val(bits, bits)
                        merged = jnp.maximum(top, incoming)
                        winners, _ = plsc.sort_key_val(merged, merged, descending=True)
                        return winners

                    cutoff = jnp.maximum(seed_top[-1], floor)

                    def scan(cutoff):
                        cutoff = jnp.full(lanes, cutoff, jnp.uint32)
                        mask = jnp.where(
                            cutoff & jnp.uint32(FP32_SIGN_BIT),
                            jnp.uint32(FP32_SIGN_BIT),
                            jnp.uint32(UINT32_MAX),
                        )
                        limit = (cutoff ^ mask).view(jnp.float32)

                        @plsc.parallel_loop(
                            0, width, lanes, unroll=scan_unroll, carry=jnp.int32(0)
                        )
                        def append(i, count):
                            keys = blocks[pl.ds(score_slot * block_size + i, lanes)]
                            ids = i + lane_ids
                            # Inclusive FP32 screening also retains both signed zeros.
                            accept = keys >= limit
                            if valid_lengths is not None or valid_width != width:
                                accept &= ids < length
                            plsc.store_compressed(
                                candidate_ids.at[pl.ds(count, lanes)], ids, mask=accept
                            )
                            return count + plsc.all_reduce_population_count(accept)[0]

                        return append

                    # Sampling only screens: fall back before exact FP32 selection.
                    count = scan(cutoff)
                    count = jax.lax.cond(
                        (count < k) & (cutoff > floor),
                        lambda: scan(floor),
                        lambda: count,
                    )

                    @plsc.parallel_loop(
                        0,
                        pl.cdiv(count, lanes) * lanes,
                        lanes,
                        unroll=loop_unroll,
                        carry=(
                            jnp.full(lanes, UINT32_MAX, jnp.uint32),
                            jnp.zeros(lanes, jnp.uint32),
                        ),
                    )
                    def order_keys(i, bounds):
                        ids = candidate_ids[pl.ds(i, lanes)]
                        keys = plsc.load_gather(
                            blocks.at[pl.ds(score_slot * block_size, block_size)],
                            [jnp.where(i + lane_ids < count, ids, 0)],
                        )
                        bits = _ordered_bits(keys)
                        candidates[pl.ds(i, lanes)] = bits
                        valid = i + lane_ids < count
                        return (
                            jnp.minimum(
                                bounds[0],
                                jnp.where(valid, bits, jnp.uint32(UINT32_MAX)),
                            ),
                            jnp.maximum(
                                bounds[1], jnp.where(valid, bits, jnp.uint32(0))
                            ),
                        )

                    compact(
                        count,
                        ordered_ids=True,
                        key_bounds=(jnp.min(order_keys[0]), jnp.max(order_keys[1])),
                    )
            else:
                initial_threshold = jnp.uint32(
                    ORDERED_NEG_INF if exclude_neg_inf else 0
                )

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
                        ids = block_id * block_size + i + lane_ids
                        accept = ids < length
                        if exclude_neg_inf:
                            accept = accept & (keys > -jnp.inf)
                        store_candidates(count, keys, ids, accept)
                        return count + plsc.all_reduce_population_count(accept)[0]

                    selection = compact(bootstrap)
                    have = selection.have_threshold
                    return selection._replace(
                        threshold=jnp.where(
                            have, selection.threshold, initial_threshold
                        ),
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
                if bootstrap_width < valid_width:
                    jax.lax.fori_loop(
                        0, row_blocks, scan_block, (seed, position, block_id)
                    )
            finish_row = (
                None
                if epilogue is None
                else functools.partial(
                    epilogue.apply,
                    refs[:metadata_inputs],
                    candidate_ids,
                    row_length,
                    sem,
                )
            )
            if finish_row is None:
                sort_selected()
            else:
                finish_row(sort_selected)
            pltpu.async_copy(
                candidate_ids.at[:output_width], out_indices_ref.at[row], sem
            ).wait()

    inputs = () if epilogue is None else epilogue.inputs
    result = kernel(scores, lengths, *inputs)
    if output_width != k:
        result = result[:, :k]
    if scheduling_group_id is not None:
        result = xla_metadata.set_xla_metadata(
            result, _scheduling_group_id=scheduling_group_id
        )
    return result


__all__ = ["TopKEpilogue", "streaming_topk"]
