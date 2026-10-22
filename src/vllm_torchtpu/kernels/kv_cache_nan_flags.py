# SPDX-License-Identifier: Apache-2.0
"""Read-only NaN/Inf reductions over a scheduled chunk's existing cache inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from vllm_torchtpu.kernels.gdn.v3.config import StateSourcePlan


def _active_requests(seq_lens, query_start_loc):
    query_lens = jnp.diff(query_start_loc)
    prefix_lens = seq_lens - query_lens
    active = (query_lens > 0) & (prefix_lens > 0)
    indices = jnp.nonzero(active, size=seq_lens.size, fill_value=0)[0]
    return indices, prefix_lens, active.sum(dtype=jnp.int32)


def _scan_requests(seq_lens, query_start_loc, inspect):
    indices, prefix_lens, count = _active_requests(seq_lens, query_start_loc)

    def visit(state):
        position, _ = state
        i = indices[position]
        return position + 1, inspect(i, prefix_lens[i])

    return jax.lax.while_loop(
        lambda state: (state[0] < count) & ~state[1],
        visit,
        (jnp.array(0, jnp.int32), jnp.array(False)),
    )[1]


def attention_cache_has_nan(
    cache, block_tables, seq_lens, query_start_loc, *, page_size=None, read_page=None
):
    """Canonical cache: [page, token, live KV head, live head dimension].

    Each loop reads one page, bounding temporary storage independently of the
    context length and batch size. Page increments until ceil(prefix / page_size)
    or a NaN/Inf is found; fresh and padded requests never enter the page loop.
    """
    if page_size is None:
        page_size = cache.shape[1]
    if read_page is None:

        def read_page(index):
            return cache[index]

    def inspect(i, prefix):
        pages = (prefix + page_size - 1) // page_size

        def visit(state):
            page, _ = state
            values = read_page(block_tables[i, page])
            valid = jnp.arange(page_size) < prefix - page * page_size
            bad = jnp.any(~jnp.isfinite(values), where=valid[:, None, None])
            return page + 1, bad

        return jax.lax.while_loop(
            lambda state: (state[0] < pages) & ~state[1],
            visit,
            (jnp.array(0, jnp.int32), jnp.array(False)),
        )[1]

    return _scan_requests(seq_lens, query_start_loc, inspect)


def _read_slots(state_indices, slot_read_offsets, ckpt_indices):
    if slot_read_offsets is None:
        return state_indices
    offsets = slot_read_offsets[state_indices]
    if ckpt_indices is not None:
        return jnp.take_along_axis(ckpt_indices, offsets[:, None], axis=1)[:, 0]
    return state_indices + offsets


def dense_state_has_nan(
    states,
    state_indices,
    seq_lens,
    query_start_loc,
    slot_read_offsets=None,
    ckpt_indices=None,
):
    """Dense Conv/SSM slots; checkpoint IDs can be noncontiguous in a pool."""
    slots = _read_slots(state_indices, slot_read_offsets, ckpt_indices)

    def inspect(i, _prefix):
        bad = jnp.array(False)
        for state in states:
            bad |= (~jnp.isfinite(state[slots[i]])).any()
        return bad

    return _scan_requests(seq_lens, query_start_loc, inspect)


def _typed_rows(raw, dtype):
    """JAX equivalent of typed_ldst's ref.bitcast on the second-minor axis.

    Keep lanes independent: adjacent stored rows form each typed word, rather
    than adjacent lanes. This matters when an FP8 pool carries FP32/BF16 state.
    """
    rows = raw.reshape(-1, raw.shape[-1])
    old_size, new_size = rows.dtype.itemsize, jnp.dtype(dtype).itemsize
    assert rows.shape[0] * old_size % new_size == 0, (
        f"Typed cache region must contain whole rows: {raw.shape}, "
        f"{raw.dtype} -> {dtype}"
    )
    if old_size == new_size:
        return jax.lax.bitcast_convert_type(rows, dtype)
    lanes_first = rows.T
    if old_size < new_size:
        lanes_first = lanes_first.reshape(rows.shape[-1], -1, new_size // old_size)
    converted = jax.lax.bitcast_convert_type(lanes_first, dtype)
    return converted.reshape(rows.shape[-1], -1).T


def pooled_state_has_nan(
    pool,
    state_indices,
    seq_lens,
    query_start_loc,
    slot_read_offsets=None,
    ckpt_indices=None,
    *,
    plan: StateSourcePlan,
):
    """Inspect the same typed regions and checkpoint as the fused GDN kernel.

    The caller supplies the existing GDN StateSourcePlan. Neither unused pool
    bytes nor typed region padding participates in the reduction.
    """
    slots = _read_slots(state_indices, slot_read_offsets, ckpt_indices)

    def inspect(i, _prefix):
        bad = jnp.array(False)
        for region in (plan.conv, plan.recurrent):
            base = slots[i] * plan.stride + region.kb0
            raw = jax.lax.dynamic_slice_in_dim(pool, base, region.nblocks, axis=0)
            if not plan.whole_block_dma:
                raw = raw[:, region.row0 : region.row0 + region.nrows]
            # uint32 is only GDN's VMEM transport for pairs of BF16 values.
            # Read the physical BF16 values directly: a chained FP8 -> uint32
            # -> BF16 bitcast triggers an invalid XLA simplification in JAX 0.10.2.
            unpack = 2 if region.view_dtype == jnp.uint32 else 1
            dtype = jnp.bfloat16 if unpack == 2 else region.view_dtype
            parts = [_typed_rows(raw[j], dtype) for j in range(region.nblocks)]
            values = jnp.concatenate(parts)
            lanes = raw.shape[-1]
            if unpack == 2:
                assert parts[0].shape[0] % 2 == 0, (
                    "GDN's uint32 view must contain whole pairs of BF16 rows"
                )
                # Match uint32 narrowing/truncation *before* unpacking. Each
                # selected word owns two BF16 values at the same source lane;
                # truncating BF16 rows first could select other lanes instead.
                pairs = values.reshape(-1, 2, lanes).transpose(0, 2, 1)
                words_used = region.rows_used * (lanes // region.lane_split)
                values = pairs.reshape(-1, 2)[:words_used]
            else:
                values = values.reshape(-1, lanes // region.lane_split)
                values = values[: region.rows_used]
            bad |= (~jnp.isfinite(values)).any()
        return bad

    return _scan_requests(seq_lens, query_start_loc, inspect)
