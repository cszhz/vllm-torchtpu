# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU-Friendly MLA Ragged Paged Attention kernel."""

import functools
from enum import Enum
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def narrow_page_indices(
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    *,
    sliding_window: int,
    logical_page_size: int,
    max_q_len: int,
    max_kv_pages_per_block: int,
) -> tuple[jax.Array, jax.Array]:
    """Keep only the pages each sequence's sliding window can touch.

    The kernels copy `page_indices` into SMEM, and at long max model lengths
    the full table is large: 8,192 int32 entries per sequence at 1M tokens and
    128-token pages. The kernels compute every position from `kv_lens`, so
    dropping whole pages before the window and reducing `kv_lens` by the same
    number of tokens keeps every read and write on the same page.

    Each sequence keeps `window_pages` entries, starting at the page of the
    first key any of its queries can see. They cover the keys and new tokens,
    which can start partway into a page, the pages a kv block can read past
    them, and one page of margin.

    Returns the shifted `kv_lens` and a `[max_num_seqs * window_pages]` table,
    or the inputs unchanged when the table is already that narrow.
    """
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    span_pages = cdiv(max_q_len + sliding_window, logical_page_size) + 1
    window_pages = span_pages + max_kv_pages_per_block + 1
    if window_pages >= pages_per_seq:
        return kv_lens, page_indices

    # Same bound as the kernels' `_start_offset` for the first query block.
    q_lens = jnp.maximum(cu_q_lens[1:] - cu_q_lens[:-1], 0)
    first_key = jnp.maximum(kv_lens - q_lens - sliding_window + 1, 0)
    first_page = first_key // logical_page_size

    window = jnp.arange(window_pages, dtype=first_page.dtype)
    cols = jnp.minimum(first_page[:, None] + window[None, :], pages_per_seq - 1)
    table = page_indices.reshape(max_num_seqs, pages_per_seq)
    narrowed = jnp.take_along_axis(table, cols, axis=1).reshape(-1)
    return kv_lens - first_page * logical_page_size, narrowed


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


class MlaCase(Enum):
    """Represents the different cases for MLA.

    - DECODE: Sequences are in decode-only mode (q_len = 1).
    - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
    - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
    """

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        return {
            MlaCase.DECODE: "d",
            MlaCase.PREFILL: "p",
            MlaCase.MIXED: "m",
        }[self]


def _mla_sliding_window_ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bo_ids_ref,
    # [6] (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset,
    #     bkv_sem_0_sz, bkv_sem_1_sz)
    bkv_update_ids_ref,
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    new_kv_hbm_ref,  # [max_num_tokens, token_bytes // 128, 128] uint8
    # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    cache_kv_hbm_ref,
    in_output_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    in_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    in_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    updated_cache_kv_hbm_ref,
    l_hbm_ref,  # [max_num_tokens, num_l_heads]
    m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Scratch
    bkv_x2_ref,  # [2, bkv_sz, token_bytes // 128, 128] uint8
    bq_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    bo_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    bl_x2_ref,  # [2, bq_sz, num_l_heads]
    bm_x2_ref,  # [2, bq_sz, num_l_heads]
    sems,  # [6, 2]
    l_ref,  # [bq_sz * num_q_heads],
    m_ref,  # [bq_sz * num_q_heads],
    acc_ref,  # [bq_sz * num_q_heads, head_dim],
    *,
    static_q_len: int | None,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    unnormalized_output: bool,
    non_causal_block: bool,
    q_compute_block_size: int | None,
    bkv_p,
    bq_sz,
    # With `hca`, fold each sequence's HCA kv (one block of `hca_bkv_p` pages)
    # into every bq block's SWA state and write the normalized output; no l/m.
    hca: "HcaDecodeRefs | None" = None,
    hca_bkv_p: int = 0,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape
    assert sliding_window > 0

    _, num_q_heads, head_dim = q_hbm_ref.shape

    # An int32 CSA host array is viewed as the uint8 slabs it holds.
    cache_kv_hbm_ref = csa_cache_layout.as_u8_slabs(cache_kv_hbm_ref)
    updated_cache_kv_hbm_ref = csa_cache_layout.as_u8_slabs(updated_cache_kv_hbm_ref)
    total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
        cache_kv_hbm_ref.shape
    )
    q_dtype = q_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert head_dim % 128 == 0
    token_bytes = head_dim * get_dtype_bitwidth(q_dtype) // 8
    slot_bytes = kv_packing * lkv_dim
    assert token_bytes % slot_bytes == 0
    slots_per_token = token_bytes // slot_bytes
    phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token

    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    bkv_sz = bkv_p * logical_page_size
    page_size = logical_page_size

    # Static upper bound on the number of bkv blocks any (seq, bq) block can
    # need
    max_bkv_blocks = cdiv(bq_sz + sliding_window - 1, bkv_sz)
    single_bkv_block = max_bkv_blocks == 1

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    seq_idx = pl.program_id(0) + start_seq_idx
    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]

    if hca is not None:
        assert unnormalized_output
        hca_total_pages, hca_slots_per_page, *hca_slot_shape = hca.cache_hbm_ref.shape
        assert tuple(hca_slot_shape) == (kv_packing, lkv_dim)
        hca_page_size = hca_slots_per_page // slots_per_token
        hca_bkv_sz = hca_bkv_p * hca_page_size
        hca_pages_per_seq = hca.page_indices_ref.shape[0] // max_num_seqs
        assert hca_pages_per_seq <= hca_bkv_p, "HCA kv must fit one block"
        hca_kv_len_cap = hca_pages_per_seq * hca_page_size
        reshaped_hca_cache_hbm_ref = hca.cache_hbm_ref.reshape(
            hca_total_pages * hca_page_size,
            slots_per_token * kv_packing,
            lkv_dim,
        )
        # One grid step per sequence; sequence i's HCA kv sits in slot i % 2.
        hca_slot = pl.program_id(0) % 2

    def flash_attention(
        q,  # [bq_sz * num_q_heads, head_dim]
        kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_x2_ref
        *,
        bq_idx,
        bkv_idx,
        start_offset,
    ):
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape == (bkv_sz, head_dim)
        n = q.shape[0] // num_q_heads

        # split bq into chunks of size q_compute_block_size
        if q_compute_block_size is None:
            chunk_sz = n
        else:
            chunk_sz = q_compute_block_size if n % q_compute_block_size == 0 else n
        num_chunks = n // chunk_sz

        def load_with_init(ref, init_val):
            if single_bkv_block:
                return jnp.full_like(ref, init_val)
            else:
                return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val), ref[...])

        # k_span is shared across all chunks (only depends on the bkv column).
        k_span = (
            start_offset
            + bkv_idx * bkv_sz
            + lax.broadcasted_iota(jnp.int32, (1, bkv_sz), 1)
        )

        # Process the bq rows in `num_chunks` independent slices for better
        # parallelism.
        chunk_size = chunk_sz * num_q_heads
        for c in range(num_chunks):
            start_row = c * chunk_size
            qc = q[
                start_row : start_row + chunk_size
            ]  # [chunk_sz * num_q_heads, head_dim]
            cl_ref = l_ref.at[start_row : start_row + chunk_size]
            cm_ref = m_ref.at[start_row : start_row + chunk_size]
            cacc_ref = acc_ref.at[start_row : start_row + chunk_size]

            # Follow FlashAttention-2 forward pass.
            # q_span is per-row (identical across the bkv columns and across the
            # num_q_heads rows of one query).
            q_span = (
                kv_len
                - q_len
                + bq_idx * bq_sz
                + (start_row + lax.broadcasted_iota(jnp.int32, (chunk_size, 1), 0))
                // num_q_heads
            )
            # A key is valid iff 0 <= q_span - k_span < sliding_window (causal +
            # within window).
            # When int32 -> uint32, negative values become large positive values.
            # So we only need to check the upper bound.
            keep = (q_span - k_span).astype(jnp.uint32) < jnp.uint32(sliding_window)

            if non_causal_block:
                # DSpark draft block mode: the last q_len positions of the sequence
                # are one parallel-drafting block attending bidirectionally within
                # itself. The upper bound is NOT free here: the causal uint32 trick
                # excluded the final page's tail garbage (k_span >= kv_len), so the
                # block clause must re-impose it.
                keep = keep | ((k_span >= kv_len - q_len) & (k_span < kv_len))

            s = jnp.einsum("nd,md->nm", qc, kv, preferred_element_type=jnp.float32)
            s *= sm_scale

            s = jnp.where(keep, s, jnp.finfo(s.dtype).min)
            s_rowmax = jnp.max(s, axis=1)
            if single_bkv_block:
                m_curr = s_rowmax
                cm_ref[...] = m_curr
                p = jnp.exp(s - m_curr[:, None])
                pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)
                cl_ref[...] = jnp.sum(p, axis=1)
                cacc_ref[...] = pv
            else:
                m_prev = load_with_init(cm_ref, jnp.finfo(jnp.float32).min)
                m_curr = jnp.maximum(m_prev, s_rowmax)
                cm_ref[...] = m_curr
                p = jnp.exp(s - m_curr[:, None])
                pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)
                p_rowsum = jnp.sum(p, axis=1)
                exp_m_diff = jnp.exp(m_prev - m_curr)
                l_prev = load_with_init(cl_ref, 0.0)
                l_curr = exp_m_diff * l_prev + p_rowsum
                cl_ref[...] = l_curr
                o_prev = load_with_init(cacc_ref, 0.0)
                o_curr = exp_m_diff[:, None] * o_prev + pv
                cacc_ref[...] = o_curr

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _get_kv_len(seq_idx):
        return jnp.where(seq_idx < end_seq_idx, kv_lens_ref[seq_idx], 0)

    def _get_q_len(seq_idx):
        return jnp.where(
            seq_idx < end_seq_idx,
            cu_q_lens_ref[seq_idx + 1] - cu_q_lens_ref[seq_idx],
            0,
        )

    def _start_offset(seq_idx, bq_idx):
        # Exact start of the sliding window for this (seq, bq) block, in tokens.
        # The first query row of this bq block sits at position
        #   q_pos_min = kv_len - q_len + bq_idx * bq_sz
        # and attends to keys k with 0 <= q_pos_min - k < sliding_window, so the
        # first key it can ever need is q_pos_min - sliding_window + 1.
        return jnp.maximum(
            _get_kv_len(seq_idx)
            - _get_q_len(seq_idx)
            + bq_idx * bq_sz
            - sliding_window
            + 1,
            0,
        )

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        # bkv_x2_ref shape: [2, bkv_sz, slots_per_token * kv_packing, lkv_dim]
        bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        # View the physical uint8 cache [total_num_pages, slots, kv_packing, lkv_dim]
        # as one contiguous row per token: [total_num_pages * phys_tokens_per_page,
        # token_bytes].
        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * phys_tokens_per_page,
            slots_per_token * kv_packing,
            lkv_dim,
        )

        kv_len = kv_lens_ref[seq_idx]
        # Window base for this (seq, bq) block plus the relative block. `start_offset`
        # is an exact token position, so `kv_len_start` is generally not a multiple
        # of page_size; `page_off` is where inside its page the window starts.
        kv_len_start = start_offset + bkv_idx * bkv_sz
        kv_p_start = kv_len_start // page_size
        page_off = kv_len_start - kv_p_start * page_size

        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = jnp.maximum(kv_len - kv_len_start, 0)
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        new_kv_len_start = q_end - kv_left_frm_new
        dma_bkv_sz = bkv_sz_frm_cache + bkv_sz_frm_new

        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

            # Fetch effective kv from kv cache. To pipeline multiple DMA calls, we
            # utilize static for loop instead of dynamic for loop.
            #
            # The window is not page-aligned, so it spans bkv_p + 1 source pages: a
            # partial head page (starting at `page_off`), bkv_p - 1 full pages, and a
            # partial tail page contributing `page_off` tokens. When page_off == 0
            # the tail copy degenerates to size 0.
            for i in range(bkv_p + 1):
                if i == 0:
                    # Head page: skip the first `page_off` tokens of the page.
                    in_page_off = page_off
                    vmem_off = jnp.int32(0)
                    avail = page_size - page_off
                else:
                    in_page_off = jnp.int32(0)
                    vmem_off = i * page_size - page_off
                    avail = jnp.int32(page_size)
                # Ensure only effective kvs are copied, that we don't go negative, and
                # that we never write past the end of the bkv buffer.
                sz = jnp.clip(bkv_sz_frm_cache - vmem_off, 0, avail)
                # If the page index is out of bound, we set page_idx to the last page.
                # And there will be no copy since sz will be 0.
                page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                _async_copy(
                    reshaped_cache_hbm_ref.at[
                        pl.ds(
                            page_indices_ref[page_idx] * phys_tokens_per_page
                            + in_page_off,
                            sz,
                        ),
                    ],
                    bkv_vmem_ref.at[pl.ds(vmem_off, sz)],
                    sem,
                    wait,
                )

            # Fetch new KVs by appending them right after the last cache token. Token
            # rows are self-contained, so new KV begins exactly where cache KV ends.
            _async_copy(
                new_kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                bkv_vmem_ref.at[pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)],
                sem,
                wait,
            )

        else:
            # When we wait, we can use a dummy copy to wait for DMAs to complete where
            # src == dst. However, the dma size must be correct.
            dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
            _async_copy(
                src=dst_kv,
                dst=dst_kv,
                sem=sem,
                wait=True,
            )

        # This returns (offset, size, vmem_start) in units of tokens:
        #   offset: starting token index where the new KV should be stored
        #   size: number of tokens of the new KV, which is 1 in decode.
        #   vmem_start: where the new KV starts within the bkv VMEM buffer.
        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new, bkv_sz_frm_cache

    def _update_kv_cache(
        seq_idx,
        bkv_sem_idx,
        offset,  # In units of tokens.
        update_sz,  # In units of tokens.
        *,
        in_vmem_start=0,  # New-KV start within the bkv VMEM buffer.
        wait=False,
    ):
        sem = sems.at[3, bkv_sem_idx]
        bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        update_kv_packing_iters = update_sz

        # View the cache as one contiguous row per token (see _fetch_bkv).
        reshaped_cache_kv_hbm_ref = updated_cache_kv_hbm_ref.reshape(
            total_num_pages * phys_tokens_per_page,
            slots_per_token * kv_packing,
            lkv_dim,
        )

        if not wait:
            # Issue DMA copy for the updated parts, page by page.
            kv_p_start = offset // page_size
            kv_p_end = cdiv(offset + update_sz, page_size)
            start_word_in_page = offset % page_size
            start_word_in_vmem = in_vmem_start
            words_to_transfer = update_kv_packing_iters
            page_indices_offset = seq_idx * pages_per_seq + kv_p_start

            def loop_body(i, states):
                curr_word_in_page, words_to_transfer, curr_word_in_vmem = states
                sz = jnp.minimum(page_size - curr_word_in_page, words_to_transfer)
                page_idx = page_indices_ref[page_indices_offset + i]

                _async_copy(
                    # bkv_vmem_ref shape: [bkv_sz_per_kv_packing, token_bytes]
                    bkv_vmem_ref.at[pl.ds(curr_word_in_vmem, sz)],
                    reshaped_cache_kv_hbm_ref.at[
                        pl.ds(
                            page_idx * phys_tokens_per_page + curr_word_in_page,
                            sz,
                        ),
                    ],
                    sem,
                    wait=False,
                )
                return 0, words_to_transfer - sz, curr_word_in_vmem + sz

            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (
                    start_word_in_page,
                    words_to_transfer,
                    start_word_in_vmem,
                ),  # initial states
                unroll=False,
            )
        else:  # Wait
            dma_sz_words = update_kv_packing_iters
            dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_sz_words)]
            _async_copy(
                src=dst_kv,
                dst=dst_kv,
                sem=sem,
                wait=True,
            )

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_vmem_ref = bq_x2_ref.at[bq_sem_idx]

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            q_hbm_ref.at[pl.ds(q_len_start, sz)],
            bq_vmem_ref.at[pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            o_hbm_ref.at[pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def _send_l(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[4, bo_sem_idx]
        vmem_ref = bl_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            l_hbm_ref.at[pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def _send_m(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[5, bo_sem_idx]
        vmem_ref = bm_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            m_hbm_ref.at[pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)
        if hca is None:
            _send_l(seq_idx, bo_idx, bo_sem_idx)
            _send_m(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(old_seq_idx >= 0, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)
            if hca is None:
                _send_l(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)
                _send_m(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, vmem_start):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(
            seq_idx, bkv_sem_idx, offset, update_sz, in_vmem_start=vmem_start
        )

    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, wait=True)

    def load_bq(bq_sem_idx):
        return bq_x2_ref.at[bq_sem_idx][...].reshape(bq_sz * num_q_heads, head_dim)

    def load_bkv(bkv_sem_idx, bkv_idx, start_offset):
        bkv_u8 = bkv_x2_ref.at[bkv_sem_idx][...]
        # [bkv_sz, slots_per_token * kv_packing, lkv_dim] -> [bkv_sz, token_bytes]
        # bkv_u8: [bkv_sz, slots_per_token * kv_packing, lkv_dim] uint8. The bytes
        # were stored (see `prepare_kv_inputs`) so that each pair of consecutive
        # sublanes already holds one bf16 value in little-endian order (lo byte then
        # hi byte). `pltpu.bitcast` fuses consecutive sublanes little-endian, so the
        # reconstruction is a plain bitcast + reshape.
        bkv = pltpu.bitcast(bkv_u8, jnp.bfloat16).reshape(bkv_sz, head_dim)
        return bkv

    def _fetch_hca_bkv(seq_idx, slot, *, wait=False):
        kv_len = hca.kv_lens_ref[seq_idx]
        sem = hca.sems.at[slot]
        bkv_ref = hca.bkv_x2_vmem.at[slot]
        if wait:
            # One wait for all page copies: the semaphore counts bytes.
            sz = jnp.clip(kv_len, 0, hca_kv_len_cap)
            _async_copy(bkv_ref.at[pl.ds(0, sz)], bkv_ref.at[pl.ds(0, sz)], sem, True)
            return
        for i in range(hca_pages_per_seq):
            sz = jnp.clip(kv_len - i * hca_page_size, 0, hca_page_size)
            page_idx = hca.page_indices_ref[seq_idx * hca_pages_per_seq + i]
            _async_copy(
                reshaped_hca_cache_hbm_ref.at[pl.ds(page_idx * hca_page_size, sz)],
                bkv_ref.at[pl.ds(i * hca_page_size, sz)],
                sem,
                False,
            )

    def fold_hca(bq_idx, bq_sem_idx, acc):
        """One flash-attention step over the HCA block from the SWA state,
        then the sinks; the same ops as `mla._mla_ragged_paged_attention_kernel`
        runs on the state this kernel would hand it through HBM."""
        num_tokens = hca.kv_lens_to_attend_ref.shape[0]
        tokens_attend = jnp.stack(
            [
                hca.kv_lens_to_attend_ref[
                    jnp.minimum(q_start + bq_idx * bq_sz + i, num_tokens - 1)
                ]
                for i in range(bq_sz)
            ]
        )
        kv_lens_to_attend = jnp.broadcast_to(
            tokens_attend[:, None, None], (bq_sz, num_q_heads, hca_bkv_sz)
        ).reshape(1, bq_sz * num_q_heads, hca_bkv_sz)
        kv = pltpu.bitcast(hca.bkv_x2_vmem[hca_slot], jnp.bfloat16).reshape(
            1, hca_bkv_sz, head_dim
        )
        q = load_bq(bq_sem_idx)[None]

        # The HBM handoff stores the SWA accumulator in q_dtype.
        acc = acc.astype(q_dtype).astype(jnp.float32)[None]
        m_prev = m_ref[...][None]
        l_prev = l_ref[...][None]

        s = jnp.einsum("bnd,bmd->bnm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
        k_span = lax.broadcasted_iota(jnp.int32, s.shape, 2)
        s = jnp.where(kv_lens_to_attend <= k_span, jnp.finfo(s.dtype).min, s)
        m_curr = jnp.maximum(m_prev, jnp.max(s, axis=2))
        p = jnp.exp(s - m_curr[:, :, None])
        pv = jnp.einsum("bnm,bmd->bnd", p, kv, preferred_element_type=jnp.float32)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_curr = exp_m_diff * l_prev + jnp.sum(p, axis=2)
        acc = exp_m_diff[:, :, None] * acc + pv

        attention_sinks = jnp.concat([attention_sinks_ref[...] for _ in range(bq_sz)])
        exp_attention_sinks = jnp.exp(attention_sinks[None, :] - m_curr)
        l_sum = (l_curr + exp_attention_sinks)[:, :, None]
        if q_dtype == jnp.float32:
            return lax.div(acc, l_sum)
        return (acc * pl.reciprocal(l_sum, approx=True)).astype(q_dtype)

    def process():
        if static_q_len is None:
            num_bq = jnp.maximum(1, cdiv(q_len, bq_sz))
        else:
            num_bq = jnp.maximum(1, cdiv(static_q_len, bq_sz))

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def compute_with_bq(bq_idx, _):
            # given it's sliding window attention, we only need to scan over the
            # needed bkv indices within the window of the current bq block.
            # Block indices are relative to the page-aligned window base
            # (_start_offset), so they always start at 0 for each bq block.
            #
            # _start_offset only depends on (seq_idx, bq_idx), so it is constant for
            # the whole current bq block. Compute it once here.
            cur_start_offset = _start_offset(seq_idx, bq_idx)
            start_bkv_idx = 0
            if single_bkv_block:
                end_bkv_idx = 1
            else:
                if non_causal_block:
                    # Draft block mode: every query in the block may attend to the
                    # whole block, which ends at kv_len, not at this bq block's last
                    # query.
                    q_pos_end = kv_len
                else:
                    # Causal: no query sees past its own position, so the last query
                    # of this bq block bounds the keys needed.
                    q_pos_end = jnp.minimum(
                        kv_len - q_len + (bq_idx + 1) * bq_sz, kv_len
                    )
                end_bkv_idx = jnp.maximum(cdiv(q_pos_end - cur_start_offset, bkv_sz), 1)

            def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
                next_bkv_idx = bkv_idx + 1
                is_last_bkv = next_bkv_idx == end_bkv_idx
                next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
                is_last_bq = next_bq_idx == num_bq
                next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
                next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
                # Next bq block restarts its relative block index at 0.
                next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
                next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
                return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx
            )

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):
                # Get next bkv ids.
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx = (
                    get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)
                )

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    # The next block may belong to a different (seq, bq), so it needs
                    # its own start offset.
                    next_start_offset = _start_offset(next_seq_idx, next_bq_idx)
                    start_fetch_bkv(
                        next_seq_idx,
                        next_bkv_idx,
                        next_bkv_sem_idx,
                        next_start_offset,
                    )

                # Wait for cur bkv
                offset, update_sz, vmem_start = wait_fetch_bkv(
                    seq_idx, bkv_idx, bkv_sem_idx, cur_start_offset
                )

                # Start updating bkv to kv cache if applicable.
                # Ensure only the query chunk corresponding to the exact new KV tokens
                # writes them to HBM to avoid duplicated writes.
                q_pos_start = kv_len - q_len + bq_idx * bq_sz
                q_pos_end = jnp.minimum(kv_len - q_len + (bq_idx + 1) * bq_sz, kv_len)
                write_start = jnp.maximum(offset, q_pos_start)
                write_end = jnp.minimum(offset + update_sz, q_pos_end)
                actual_update_sz = jnp.maximum(0, write_end - write_start)
                actual_vmem_start = vmem_start + (write_start - offset)

                @pl.when(actual_update_sz > 0)
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(
                        seq_idx,
                        bkv_sem_idx,
                        write_start,
                        actual_update_sz,
                        actual_vmem_start,
                    )

                # Load bkv into vreg. There is no need to mask out invalid k/v entries,
                # because the score of invalid Q.K^T pairs are masked (to be zero) in
                # flash attention, so that the invalid kv entries
                # (as long as they are not NaN or inf) won't affect to the output.
                bkv = load_bkv(
                    bkv_sem_idx,
                    bkv_idx,
                    cur_start_offset,
                )

                bq = load_bq(bq_sem_idx)

                flash_attention(
                    bq,
                    bkv,
                    bq_idx=bq_idx,
                    bkv_idx=bkv_idx,
                    start_offset=cur_start_offset,
                )

            # Wait for cur bq if not ready yet
            wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)
            if single_bkv_block:
                compute_with_bkv(0, None)
            else:
                lax.fori_loop(
                    start_bkv_idx,
                    end_bkv_idx,
                    compute_with_bkv,
                    None,
                    unroll=False,
                )

            # Load acc and calculate final output.
            acc = acc_ref[...]

            if hca is not None:
                out = fold_hca(bq_idx, bq_sem_idx, acc)
            elif unnormalized_output:
                out = acc.astype(q_dtype)
            else:
                attention_sinks = jnp.concat(
                    [attention_sinks_ref[...] for _ in range(bq_sz)]
                )
                exp_attention_sinks = jnp.exp(attention_sinks - m_ref[...])
                l = l_ref[...] + exp_attention_sinks  # noqa: E741
                out = (
                    lax.div(acc, l[:, None])
                    if q_dtype == jnp.float32
                    else (acc * pl.reciprocal(l[:, None], approx=True)).astype(q_dtype)
                )

            # Wait for previous bo to be fully sent before storing new bo.
            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            # Store output from acc to bo.
            bo_x2_ref.at[bo_sem_idx][...] = out.reshape(bq_sz, num_q_heads, head_dim)
            if hca is not None:
                pass  # The output is final; there is no l/m to hand on.
            elif num_q_heads % 128 == 0:
                bl_x2_ref.at[bo_sem_idx][...] = l_ref[...].reshape(bq_sz, num_q_heads)
                bm_x2_ref.at[bo_sem_idx][...] = m_ref[...].reshape(bq_sz, num_q_heads)
            else:
                # if num_q_heads can't be divided by 128, the reshape can't compile,
                # so we need to loop through and copy row by row.
                for i in range(bq_sz):
                    start = i * num_q_heads
                    end = start + num_q_heads
                    bl_x2_ref.at[bo_sem_idx, i][:num_q_heads] = l_ref[start:end]
                    bm_x2_ref.at[bo_sem_idx, i][:num_q_heads] = m_ref[start:end]

            # Send cur bo
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(start_seq_idx, 0, 0)

        # Initialize bkv_x2_ref to avoid NaN issues from accessing uninitialized
        # memory
        bkv_zeros = jnp.zeros(bkv_x2_ref.shape[1:], bkv_x2_ref.dtype)
        bkv_x2_ref[0] = bkv_zeros
        # bq_idx=0, relative bkv_idx=0 (page-aligned base via _start_offset).
        start_fetch_bkv(start_seq_idx, 0, 0, _start_offset(start_seq_idx, 0))
        bkv_x2_ref[1] = bkv_zeros

        if hca is not None:
            # Masked HCA slots are multiplied by p == 0; stale VMEM may be NaN.
            hca.bkv_x2_vmem[...] = jnp.zeros(hca.bkv_x2_vmem.shape, jnp.uint8)
            _fetch_hca_bkv(start_seq_idx, 0)

    if hca is not None:

        @pl.when(seq_idx + 1 < end_seq_idx)
        def prefetch_next_hca_bkv():
            _fetch_hca_bkv(seq_idx + 1, 1 - hca_slot)

        _fetch_hca_bkv(seq_idx, hca_slot, wait=True)

    process()

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(i)
            wait_update_kv_cache(i)

    ### ------- Kernel end ------- ###


def prepare_q_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    return q


def prepare_kv_inputs(kv: jax.Array):
    """Reinterprets bf16 new KV as one raw uint8 byte row per token.

    Each token's `actual_head_dim` bf16 values become `actual_head_dim * 2`
    contiguous bytes. The bytes are laid out so that, once a token row is viewed
    as ``[slots_per_token * kv_packing, 128]`` (128 = the kernel's lane width),
    every pair of consecutive sublanes holds one bf16 value in little-endian byte
    order (lo byte then hi byte).
    """
    assert kv.dtype == jnp.bfloat16
    tokens, head_dim = kv.shape
    assert head_dim % 128 == 0
    kv_u16 = jax.lax.bitcast_convert_type(kv, jnp.uint16)
    kv_hi = ((kv_u16 >> 8) & 0xFF).astype(jnp.uint8)
    kv_lo = (kv_u16 & 0xFF).astype(jnp.uint8)
    nb = head_dim // 128
    kv_hi = kv_hi.reshape(tokens, nb, 128)
    kv_lo = kv_lo.reshape(tokens, nb, 128)
    # (tokens, nb, 2, 128)
    interleaved = jnp.stack([kv_lo, kv_hi], axis=2)
    return interleaved.reshape(tokens, head_dim * 2)


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
    return out[:, :actual_num_q_heads, :actual_head_dim]


def uses_batched_decode(
    sliding_window: int, logical_page_size: int, decode_kv_pages_per_block: int
) -> bool:
    """Whether decode runs `run_mla_batched_decode_kernel` (one kv block)."""
    return sliding_window <= decode_kv_pages_per_block * logical_page_size


def get_physical_layout(
    cache_kv_shape, q_dtype, head_dim, max_num_seqs, num_page_indices
):
    total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
        cache_kv_shape
    )
    token_bytes = head_dim * get_dtype_bitwidth(q_dtype) // 8
    slot_bytes = kv_packing * lkv_dim
    slots_per_token = token_bytes // slot_bytes
    phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token
    pages_per_seq = num_page_indices // max_num_seqs

    return (
        total_num_pages,
        kv_packing,
        lkv_dim,
        slots_per_token,
        phys_tokens_per_page,
        pages_per_seq,
    )


class HcaDecodeInputs(NamedTuple):
    """The HCA pass the SWA kernels fuse, as `mla.mla_ragged_paged_attention`
    would run it on the same sequences."""

    # [total_num_pages, page_size * slots_per_token, kv_packing, lkv_dim] uint8
    cache_kv: jax.Array
    kv_lens: jax.Array  # i32[max_num_seqs]
    kv_lens_to_attend: jax.Array  # i32[max_num_tokens]
    page_indices: jax.Array  # i32[max_num_seqs * pages_per_seq]
    # Pages per kv block; every sequence's page table must fit in one.
    num_kv_pages_per_block: int


class SwaDecodeInputs(NamedTuple):
    """The decode rows `mla_sliding_window_ragged_paged_attention` leaves to
    a fused kernel with `decode_by_caller=True`: the operands the batched
    decode kernel would have taken (see `SwaDecodeWindowRefs`)."""

    # [max_num_tokens, slots_per_token * kv_packing, lkv_dim] uint8
    new_kv: jax.Array
    # The cache with the other rows' new kv written.
    cache_kv: jax.Array
    kv_lens: jax.Array  # i32[max_num_seqs], narrowed
    copy_table: jax.Array  # `swa_decode_copy_table` of the decode rows
    num_decode_tokens: jax.Array  # i32, one token per decode sequence


class HcaDecodeRefs(NamedTuple):
    """The HCA compressed-KV block the SWA kernels fold in."""

    kv_lens_ref: Any  # [max_num_seqs]
    kv_lens_to_attend_ref: Any  # [max_num_tokens]
    page_indices_ref: Any  # [max_num_seqs * pages_per_seq]
    # [total_num_pages, page_size * slots_per_token, kv_packing, lkv_dim] uint8
    cache_hbm_ref: Any
    # [2, (seq_batch_sz,) bkv_sz, slots_per_token * kv_packing, lkv_dim] uint8;
    # the batched decode kernel holds one block per sequence of its batch.
    bkv_x2_vmem: Any
    sems: Any  # [2, (seq_batch_sz,)]


def _mla_sliding_window_hca_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    start_end_seq_idx_ref,
    sem_ids_ref,
    bo_ids_ref,
    bkv_update_ids_ref,
    hca_kv_lens_ref,
    hca_kv_lens_to_attend_ref,
    hca_page_indices_ref,
    attention_sinks_ref,
    q_hbm_ref,
    new_kv_hbm_ref,
    cache_kv_hbm_ref,
    in_l_hbm_ref,
    in_m_hbm_ref,
    hca_cache_hbm_ref,
    *outputs_and_scratch,
    **kwargs,
):
    """`_mla_sliding_window_ragged_paged_attention_kernel` with the HCA
    operands of `HcaDecodeRefs`. q is aliased with the output: each bq block's
    q rows are fetched before its output rows are written over them."""
    *refs, hca_bkv_x2_vmem, hca_sems = outputs_and_scratch
    _mla_sliding_window_ragged_paged_attention_kernel(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        start_end_seq_idx_ref,
        sem_ids_ref,
        bo_ids_ref,
        bkv_update_ids_ref,
        attention_sinks_ref,
        q_hbm_ref,
        new_kv_hbm_ref,
        cache_kv_hbm_ref,
        None,  # in_output: q is the aliased buffer.
        in_l_hbm_ref,
        in_m_hbm_ref,
        *refs,
        hca=HcaDecodeRefs(
            hca_kv_lens_ref,
            hca_kv_lens_to_attend_ref,
            hca_page_indices_ref,
            hca_cache_hbm_ref,
            hca_bkv_x2_vmem,
            hca_sems,
        ),
        **kwargs,
    )


def _mla_batched_decode_hca_kernel(
    kv_lens_ref,
    copy_table_ref,
    cu_q_lens_ref,
    start_end_seq_idx_ref,
    hca_kv_lens_ref,
    hca_kv_lens_to_attend_ref,
    hca_page_indices_ref,
    attention_sinks_ref,
    q_hbm_ref,
    new_kv_hbm_ref,
    cache_kv_hbm_ref,
    in_l_hbm_ref,
    in_m_hbm_ref,
    hca_cache_hbm_ref,
    *outputs_and_scratch,
    **kwargs,
):
    """`_mla_batched_decode_kernel` with the HCA operands of `HcaDecodeRefs`.

    q is aliased with the output: each batch's q rows are fetched before its
    output rows are written over them."""
    *refs, hca_bkv_x2_vmem, hca_sems = outputs_and_scratch
    _mla_batched_decode_kernel(
        kv_lens_ref,
        copy_table_ref,
        cu_q_lens_ref,
        start_end_seq_idx_ref,
        attention_sinks_ref,
        q_hbm_ref,
        new_kv_hbm_ref,
        cache_kv_hbm_ref,
        None,  # in_output: q is the aliased buffer.
        in_l_hbm_ref,
        in_m_hbm_ref,
        *refs,
        hca=HcaDecodeRefs(
            hca_kv_lens_ref,
            hca_kv_lens_to_attend_ref,
            hca_page_indices_ref,
            hca_cache_hbm_ref,
            hca_bkv_x2_vmem,
            hca_sems,
        ),
        **kwargs,
    )


def _swa_decode_window_span(kv_len, *, sliding_window: int, bkv_sz: int):
    """First window key of a decode token, and how many of its window keys
    come from the cache (the new token itself comes from `new_kv`)."""
    start_offset = jnp.maximum(kv_len - sliding_window, jnp.int32(0))
    kv_left = jnp.maximum(kv_len - start_offset, jnp.int32(0))
    kv_left_frm_cache = jnp.maximum(kv_left - 1, jnp.int32(0))
    return start_offset, jnp.minimum(kv_left_frm_cache, bkv_sz)


def _swa_window_copy_chunks(
    sliding_window: int, logical_page_size: int, bkv_p: int
) -> int:
    """Pages a decode token's window keys can span: at most
    min(sliding_window - 1, bkv_sz) keys, from any offset in a page."""
    max_keys = min(sliding_window - 1, bkv_p * logical_page_size)
    return min(bkv_p + 1, cdiv(logical_page_size - 1 + max_keys, logical_page_size))


# Rows `swa_decode_copy_table` adds past the last sequence, so a batch of up
# to this many sequences may start at any sequence index without a clamp.
SWA_COPY_TABLE_PAD = 32


class SwaCopyCols:
    """Columns of one sequence's `swa_decode_copy_table` row, after its
    (cache row, rows) pair per page copy."""

    NEW_SRC = 0  # the new token's row in new_kv
    NEW_ROWS = 1  # 1 for a decode sequence, else 0 (an empty row)
    STORE_DST = 2  # the new token's row in the cache
    SLOT = 3  # the new token's slot in the window block: the cache keys

    @staticmethod
    def width(num_chunks: int) -> int:
        return 2 * num_chunks + 4


def swa_decode_copy_table(
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    end_seq_idx: jax.Array,  # i32, decode sequences are [0, end_seq_idx)
    *,
    sliding_window: int,
    logical_page_size: int,
    bkv_p: int,
    phys_tokens_per_page: int,
) -> jax.Array:
    """Every copy `SwaDecodeWindowRefs` issues for a decode sequence, as
    rows of the flat uint8-slab cache view: i32[(max_num_seqs +
    SWA_COPY_TABLE_PAD) * width], one row of `SwaCopyCols` per sequence.

    Built once per step from the same tables every SWA layer reads, so the
    kernels move the window with a few scalar loads per sequence instead of
    the page-table walk."""
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    lps = logical_page_size
    num_chunks = _swa_window_copy_chunks(sliding_window, lps, bkv_p)

    valid = jnp.arange(max_num_seqs, dtype=jnp.int32) < end_seq_idx
    kv_len = jnp.where(valid, kv_lens, 0)
    start_offset, num_keys = _swa_decode_window_span(
        kv_len, sliding_window=sliding_window, bkv_sz=bkv_p * lps
    )
    pages = page_indices.reshape(max_num_seqs, pages_per_seq)
    page_col = lax.broadcasted_iota(jnp.int32, pages.shape, 1)

    def cache_row(token):
        # One-hot select, not a gather: the table is a few pages wide.
        page = jnp.minimum(token // lps, pages_per_seq - 1)
        phys_page = jnp.sum(jnp.where(page_col == page[:, None], pages, 0), axis=1)
        return phys_page * phys_tokens_per_page + token % lps

    cols = []
    copied = 0
    for i in range(num_chunks):
        # Chunk i starts at the window start, then at each later page start.
        token = start_offset if i == 0 else (start_offset // lps + i) * lps
        rows = jnp.minimum(num_keys - copied, lps - token % lps)
        cols += [cache_row(token), rows]
        copied = copied + rows
    new_rows = valid.astype(jnp.int32)
    cols += [
        jnp.where(valid, cu_q_lens[:-1], 0),
        new_rows,
        cache_row(jnp.maximum(kv_len - 1, 0)),
        num_keys,
    ]
    table = jnp.pad(jnp.stack(cols, axis=1), ((0, SWA_COPY_TABLE_PAD), (0, 0)))
    return table.reshape(-1)


class SwaDecodeWindowRefs(NamedTuple):
    """One decode sequence's sliding-window kv block, as the batched decode
    kernel moves it: the window keys from the paged cache plus the new token
    into VMEM, then the new token from VMEM into the cache."""

    kv_lens_ref: Any  # [max_num_seqs]
    copy_table_ref: Any  # `swa_decode_copy_table`
    # [total_num_pages * phys_tokens_per_page, slots_per_token * kv_packing,
    #  lkv_dim] uint8 views of the cache and of the aliased updated cache.
    cache_hbm_ref: Any
    updated_cache_hbm_ref: Any
    new_kv_hbm_ref: Any  # [max_num_tokens, token_bytes // 128, 128] uint8
    sliding_window: int
    logical_page_size: int
    bkv_p: int
    token_bytes: int

    @property
    def bkv_sz(self) -> int:
        return self.bkv_p * self.logical_page_size

    @property
    def num_chunks(self) -> int:
        return _swa_window_copy_chunks(
            self.sliding_window, self.logical_page_size, self.bkv_p
        )

    def _col(self, seq_idx, col):
        return self.copy_table_ref[seq_idx * SwaCopyCols.width(self.num_chunks) + col]

    def _tail(self, seq_idx, col):
        """A `SwaCopyCols` column of the sequence's row."""
        return self._col(seq_idx, 2 * self.num_chunks + col)

    def num_tokens(self, seq_idx):
        """Rows `fetch` copies: the window's cache keys and the new token."""
        return self._tail(seq_idx, SwaCopyCols.SLOT) + self._tail(
            seq_idx, SwaCopyCols.NEW_ROWS
        )

    def wait_fetches(self, seq_idxs, bkv_s_refs, sems):
        """`fetch(..., wait=True)` for several sequences. A DMA wait orders
        the scalar work around it, so every wait is sized before the first."""
        sizes = [self.num_tokens(i) for i in seq_idxs]
        for n, ref, sem in zip(sizes, bkv_s_refs, sems, strict=True):
            _wait_rows(ref, sem, n)

    def wait_stores(self, num_rows, sem):
        """Wait for the `store_new_kv` copies of `num_rows` sequences."""
        _wait_rows(self.updated_cache_hbm_ref, sem, num_rows)

    def fetch(self, seq_idx, bkv_s_ref, sem, *, wait=False):
        """Copy (or wait for) the window keys and the new token into
        `bkv_s_ref`, [bkv_sz, slots_per_token * kv_packing, lkv_dim] uint8."""
        if wait:
            # One wait for all the copies below: the semaphore counts bytes.
            _wait_rows(bkv_s_ref, sem, self.num_tokens(seq_idx))
            return
        vmem_off = 0
        for i in range(self.num_chunks):
            rows = self._col(seq_idx, 2 * i + 1)
            _dma(
                self.cache_hbm_ref.at[pl.ds(self._col(seq_idx, 2 * i), rows)],
                bkv_s_ref.at[pl.ds(vmem_off, rows)],
                sem,
            )
            vmem_off += rows
        new_kv_vmem_ref = bkv_s_ref.bitcast(jnp.uint8).reshape(
            self.bkv_sz, self.token_bytes // 128, 128
        )
        new_rows = self._tail(seq_idx, SwaCopyCols.NEW_ROWS)
        _dma(
            self.new_kv_hbm_ref.at[
                pl.ds(self._tail(seq_idx, SwaCopyCols.NEW_SRC), new_rows)
            ],
            new_kv_vmem_ref.at[pl.ds(vmem_off, new_rows)],
            sem,
        )

    def store_new_kv(self, seq_idx, sem):
        """Copy the new token from new_kv into the cache, HBM to HBM: no
        window copy reads its row, so it needs no ordering against them."""
        new_rows = self._tail(seq_idx, SwaCopyCols.NEW_ROWS)
        _dma(
            self.new_kv_hbm_ref.at[
                pl.ds(self._tail(seq_idx, SwaCopyCols.NEW_SRC), new_rows)
            ],
            self.updated_cache_hbm_ref.at[
                pl.ds(self._tail(seq_idx, SwaCopyCols.STORE_DST), new_rows)
            ],
            sem,
        )


def _dma(src, dst, sem, *, wait=False):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
        cp.wait()
    else:
        cp.start()


def _wait_rows(ref, sem, n):
    """Wait for copies of `n` leading rows' worth of `ref` on `sem`."""
    _dma(ref.at[pl.ds(0, n)], ref.at[pl.ds(0, n)], sem, wait=True)


def _swa_window_kv(bkv_ref, keep_rows, num_slots=None):
    """One decode token's window keys/values, bf16 [slots, head_dim], zero
    where `keep_rows` [slots, lanes] is False (or the first `num_slots` rows
    as they are when `keep_rows` is None).

    Built from the 32-bit words of its uint8 slab rows `bkv_ref`
    [>= slots, slots_per_token * kv_packing, lkv_dim]: word w of word row k
    holds bf16 dims 256 k + w (low half) and 256 k + 128 + w (high half), the
    values a uint8 -> bfloat16 bitcast of the rows reads, without its
    sublane relayout.
    """
    _, rows, lanes = bkv_ref.shape
    if keep_rows is not None:
        num_slots = keep_rows.shape[0]
    word_rows = rows // 4
    words = bkv_ref.reshape(bkv_ref.shape[0] * rows, lanes).bitcast(jnp.int32)
    pieces = []
    for k in range(word_rows):
        w = words[pl.ds(k, num_slots, stride=word_rows), :]
        if keep_rows is not None:
            w = jnp.where(keep_rows, w, 0)
        for half in (w << 16, w & jnp.int32(-65536)):
            pieces.append(pltpu.bitcast(half, jnp.float32).astype(jnp.bfloat16))
    return jnp.concatenate(pieces, axis=1)


def swa_decode_window_partials(
    q_batch,  # [seq_batch_sz, num_q_heads, head_dim]
    # [seq_batch_sz, >= bkv_sz, slots_per_token * kv_packing, lkv_dim] uint8
    bkv_batch_ref,
    kv_lens,  # seq_batch_sz i32 scalars
    *,
    bkv_sz: int,
    sliding_window: int,
    sm_scale: float,
):
    """The unnormalized SWA softmax state of a batch of decode tokens over
    their single kv block: (pv [s, n, d], l [s, n], m [s, n]), all float32."""
    kv_lens_batch = jnp.stack(kv_lens)
    # A decode token's window keys and the new token fill slots
    # [0, min(kv_len, sliding_window)); later slots are always masked (p == 0,
    # kv == 0), so leave them out of both matmuls.
    num_slots = min(bkv_sz, align_to(sliding_window, 128))

    def keep_mask(kv_len, shape, axis):
        """Window slots a decode token attends to, along `axis`."""
        m_vec = lax.broadcasted_iota(jnp.int32, shape, axis)
        k_span = jnp.maximum(kv_len - sliding_window, jnp.int32(0)) + m_vec
        q_span = kv_len - 1
        return ((q_span - k_span).astype(jnp.uint32) < jnp.uint32(sliding_window)) | (
            m_vec == 0
        )

    keep_batch = keep_mask(kv_lens_batch[:, None, None], (1, 1, num_slots), 2)

    # Slots past kv_left were not written this step and still hold whatever
    # the previous kernel left in VMEM. Masking their scores is not enough,
    # since 0 * NaN is NaN in the PV matmul, so zero the values too.
    bkv_batch = jnp.stack(
        [
            _swa_window_kv(bkv_batch_ref.at[s], keep_mask(kv_len, (num_slots, 128), 0))
            for s, kv_len in enumerate(kv_lens)
        ]
    )

    s_mat = jnp.einsum(
        "snd,smd->snm", q_batch, bkv_batch, preferred_element_type=jnp.float32
    )
    s_mat *= sm_scale

    s_mat = jnp.where(keep_batch, s_mat, jnp.finfo(jnp.float32).min)
    s_max = jnp.max(s_mat, axis=2, keepdims=True)
    p_mat = jnp.exp(s_mat - s_max)
    pv_batch = jnp.einsum(
        "snm,smd->snd", p_mat, bkv_batch, preferred_element_type=jnp.float32
    )
    return pv_batch, jnp.sum(p_mat, axis=2), s_max[:, :, 0]


def _mla_batched_decode_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    copy_table_ref,  # `swa_decode_copy_table`
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2]
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    new_kv_hbm_ref,  # [max_num_tokens, token_bytes // 128, 128] uint8
    # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    cache_kv_hbm_ref,
    in_output_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    in_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    in_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    updated_cache_kv_hbm_ref,
    l_hbm_ref,  # [max_num_tokens, num_l_heads]
    m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Scratch (x2 for double buffering)
    # [2, seq_batch_sz, bkv_sz, slots_per_token * kv_packing, lkv_dim] uint8
    bkv_batch_x2_vmem,
    q_batch_x2_vmem,  # [2, seq_batch_sz, num_q_heads, head_dim]
    o_batch_x2_vmem,  # [2, seq_batch_sz, num_q_heads, head_dim]
    l_batch_x2_vmem,  # [2, seq_batch_sz, num_l_heads]
    m_batch_x2_vmem,  # [2, seq_batch_sz, num_l_heads]
    sems,  # [2, seq_batch_sz + 5]
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    unnormalized_output: bool,
    bkv_p: int,
    seq_batch_sz: int,
    # With `hca`, fold each sequence's HCA kv (one block of `hca_bkv_p` pages)
    # into the SWA softmax state and write the normalized output; no l/m.
    hca: HcaDecodeRefs | None = None,
    hca_bkv_p: int = 0,
):
    del in_output_hbm_ref, in_l_hbm_ref, in_m_hbm_ref  # Aliased with out, l, m.

    def async_copy(src, dst, sem, wait=False):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    _, num_q_heads, head_dim = q_hbm_ref.shape
    # An int32 CSA host array is viewed as the uint8 slabs it holds.
    cache_kv_hbm_ref = csa_cache_layout.as_u8_slabs(cache_kv_hbm_ref)
    updated_cache_kv_hbm_ref = csa_cache_layout.as_u8_slabs(updated_cache_kv_hbm_ref)
    total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
        cache_kv_hbm_ref.shape
    )
    q_dtype = q_hbm_ref.dtype
    token_bytes = head_dim * get_dtype_bitwidth(q_dtype) // 8
    slot_bytes = kv_packing * lkv_dim
    slots_per_token = token_bytes // slot_bytes
    phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token

    max_num_seqs = kv_lens_ref.shape[0]

    bkv_sz = bkv_p * logical_page_size

    reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
        total_num_pages * phys_tokens_per_page,
        slots_per_token * kv_packing,
        lkv_dim,
    )
    reshaped_updated_cache_hbm_ref = updated_cache_kv_hbm_ref.reshape(
        total_num_pages * phys_tokens_per_page,
        slots_per_token * kv_packing,
        lkv_dim,
    )

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]

    num_decode_seqs = end_seq_idx - start_seq_idx
    num_batches = cdiv(num_decode_seqs, seq_batch_sz)

    if hca is not None:
        hca_total_pages, hca_slots_per_page, *hca_slot_shape = hca.cache_hbm_ref.shape
        assert tuple(hca_slot_shape) == (kv_packing, lkv_dim)
        hca_page_size = hca_slots_per_page // slots_per_token
        hca_bkv_sz = hca_bkv_p * hca_page_size
        hca_pages_per_seq = hca.page_indices_ref.shape[0] // max_num_seqs
        assert hca_pages_per_seq <= hca_bkv_p, "HCA kv must fit one block"
        hca_kv_len_cap = hca_pages_per_seq * hca_page_size
        reshaped_hca_cache_hbm_ref = hca.cache_hbm_ref.reshape(
            hca_total_pages * hca_page_size,
            slots_per_token * kv_packing,
            lkv_dim,
        )

    store_sem = sems.at[0, seq_batch_sz + 4]
    window = SwaDecodeWindowRefs(
        kv_lens_ref,
        copy_table_ref,
        reshaped_cache_hbm_ref,
        reshaped_updated_cache_hbm_ref,
        new_kv_hbm_ref,
        sliding_window,
        logical_page_size,
        bkv_p,
        token_bytes,
    )

    def _fetch_bq(batch_idx, sem_idx, wait=False):
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        q_start_seq = jnp.minimum(s_start, max_num_seqs - 1)
        q_token_idx = jnp.where(
            s_start < end_seq_idx,
            cu_q_lens_ref[q_start_seq],
            0,
        )
        batch_sz = jnp.clip(end_seq_idx - s_start, 0, seq_batch_sz)
        async_copy(
            q_hbm_ref.at[pl.ds(q_token_idx, batch_sz)],
            q_batch_x2_vmem.at[sem_idx, pl.ds(0, batch_sz)],
            sems.at[sem_idx, 0],
            wait=wait,
        )

    def _fetch_bkv(batch_idx, sem_idx, wait=False):
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        seq_idxs = [s_start + s for s in range(seq_batch_sz)]
        bkv_refs = [bkv_batch_x2_vmem.at[sem_idx, s] for s in range(seq_batch_sz)]
        bkv_sems = [sems.at[sem_idx, 4 + s] for s in range(seq_batch_sz)]
        if wait:
            window.wait_fetches(seq_idxs, bkv_refs, bkv_sems)
            return
        for args in zip(seq_idxs, bkv_refs, bkv_sems):
            window.fetch(*args)
            window.store_new_kv(args[0], store_sem)

    def _fetch_hca_bkv(batch_idx, sem_idx, wait=False):
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        kv_lens = []
        for s in range(seq_batch_sz):
            seq_idx = s_start + s
            seq_idx_clamped = jnp.minimum(seq_idx, max_num_seqs - 1)
            kv_lens.append(
                jnp.where(seq_idx < end_seq_idx, hca.kv_lens_ref[seq_idx_clamped], 0)
            )
        if wait:
            # One wait per sequence for all its page copies (the semaphore
            # counts bytes), every wait sized before the first (see
            # `SwaDecodeWindowRefs.wait_fetches`).
            sizes = [jnp.clip(kv_len, 0, hca_kv_len_cap) for kv_len in kv_lens]
            for s, sz in enumerate(sizes):
                _wait_rows(hca.bkv_x2_vmem.at[sem_idx, s], hca.sems.at[sem_idx, s], sz)
            return
        for s, kv_len in enumerate(kv_lens):
            seq_idx_clamped = jnp.minimum(s_start + s, max_num_seqs - 1)
            sem = hca.sems.at[sem_idx, s]
            bkv_s_ref = hca.bkv_x2_vmem.at[sem_idx, s]
            for i in range(hca_pages_per_seq):
                sz = jnp.clip(kv_len - i * hca_page_size, 0, hca_page_size)
                page_idx = hca.page_indices_ref[seq_idx_clamped * hca_pages_per_seq + i]
                async_copy(
                    reshaped_hca_cache_hbm_ref.at[pl.ds(page_idx * hca_page_size, sz)],
                    bkv_s_ref.at[pl.ds(i * hca_page_size, sz)],
                    sem,
                )

    def fold_hca(s_start, sem_idx, q_batch, pv_batch, l_val, m_val):
        """One flash-attention step over the HCA block from the SWA state,
        then the sinks; the same ops as `mla._mla_ragged_paged_attention_kernel`
        runs on the state the SWA kernel hands it through HBM."""
        num_tokens = hca.kv_lens_to_attend_ref.shape[0]
        kv_lens_to_attend = jnp.stack(
            [
                jnp.where(
                    (s_start + s) < end_seq_idx,
                    hca.kv_lens_to_attend_ref[
                        jnp.minimum(
                            cu_q_lens_ref[jnp.minimum(s_start + s, max_num_seqs - 1)],
                            num_tokens - 1,
                        )
                    ],
                    0,
                )
                for s in range(seq_batch_sz)
            ]
        )
        kv = jnp.stack(
            [
                _swa_window_kv(hca.bkv_x2_vmem.at[sem_idx, s], None, hca_bkv_sz)
                for s in range(seq_batch_sz)
            ]
        )
        k_span = lax.broadcasted_iota(jnp.int32, (1, 1, hca_bkv_sz), 2)
        mask = kv_lens_to_attend[:, None, None] <= k_span

        # The HBM handoff stores the SWA accumulator in q_dtype.
        acc = pv_batch.astype(q_dtype).astype(jnp.float32)

        s_mat = jnp.einsum(
            "snd,smd->snm", q_batch, kv, preferred_element_type=jnp.float32
        )
        s_mat *= sm_scale
        s_mat = jnp.where(mask, jnp.finfo(s_mat.dtype).min, s_mat)
        m_curr = jnp.maximum(m_val, jnp.max(s_mat, axis=2))
        p_mat = jnp.exp(s_mat - m_curr[:, :, None])
        pv = jnp.einsum("snm,smd->snd", p_mat, kv, preferred_element_type=jnp.float32)
        exp_m_diff = jnp.exp(m_val - m_curr)
        l_curr = exp_m_diff * l_val + jnp.sum(p_mat, axis=2)
        acc = exp_m_diff[:, :, None] * acc + pv

        exp_attention_sinks = jnp.exp(attention_sinks_ref[...][None, :] - m_curr)
        l_sum = (l_curr + exp_attention_sinks)[:, :, None]
        if q_dtype == jnp.float32:
            return lax.div(acc, l_sum)
        return (acc * pl.reciprocal(l_sum, approx=True)).astype(q_dtype)

    def compute_batch(batch_idx, sem_idx):
        s_start = start_seq_idx + batch_idx * seq_batch_sz

        kv_lens_batch = [
            jnp.where(
                (s_start + s) < end_seq_idx,
                kv_lens_ref[jnp.minimum(s_start + s, max_num_seqs - 1)],
                0,
            )
            for s in range(seq_batch_sz)
        ]

        q_batch = q_batch_x2_vmem[sem_idx]
        pv_batch, l_val, m_val = swa_decode_window_partials(
            q_batch,
            bkv_batch_x2_vmem.at[sem_idx],
            kv_lens_batch,
            bkv_sz=bkv_sz,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
        )

        if hca is not None:
            o_batch_x2_vmem[sem_idx, ...] = fold_hca(
                s_start, sem_idx, q_batch, pv_batch, l_val, m_val
            )
            return
        if unnormalized_output:
            out = pv_batch.astype(q_dtype)
        else:
            attention_sinks = attention_sinks_ref[...][None, :]
            exp_attention_sinks = jnp.exp(attention_sinks - m_val)
            l_tot = l_val + exp_attention_sinks
            l_tile = l_tot[:, :, None]
            out = (
                lax.div(pv_batch, l_tile)
                if q_dtype == jnp.float32
                else (pv_batch * pl.reciprocal(l_tile, approx=True)).astype(q_dtype)
            )

        o_batch_x2_vmem[sem_idx, ...] = out
        num_l_heads = l_batch_x2_vmem.shape[2]
        if num_l_heads == num_q_heads:
            l_batch_x2_vmem[sem_idx, ...] = l_val
            m_batch_x2_vmem[sem_idx, ...] = m_val
        else:
            l_batch_x2_vmem[sem_idx, ...] = jnp.pad(
                l_val, ((0, 0), (0, num_l_heads - num_q_heads))
            )
            m_batch_x2_vmem[sem_idx, ...] = jnp.pad(
                m_val, ((0, 0), (0, num_l_heads - num_q_heads))
            )

    def _send_bo(batch_idx, sem_idx, wait=False):
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        q_start_seq = jnp.minimum(s_start, max_num_seqs - 1)
        q_token_idx = jnp.where(
            s_start < end_seq_idx,
            cu_q_lens_ref[q_start_seq],
            0,
        )
        batch_sz = jnp.clip(end_seq_idx - s_start, 0, seq_batch_sz)
        async_copy(
            o_batch_x2_vmem.at[sem_idx, pl.ds(0, batch_sz)],
            o_hbm_ref.at[pl.ds(q_token_idx, batch_sz)],
            sems.at[sem_idx, 1],
            wait=wait,
        )
        if hca is not None:
            return
        async_copy(
            l_batch_x2_vmem.at[sem_idx, pl.ds(0, batch_sz)],
            l_hbm_ref.at[pl.ds(q_token_idx, batch_sz)],
            sems.at[sem_idx, 2],
            wait=wait,
        )
        async_copy(
            m_batch_x2_vmem.at[sem_idx, pl.ds(0, batch_sz)],
            m_hbm_ref.at[pl.ds(q_token_idx, batch_sz)],
            sems.at[sem_idx, 3],
            wait=wait,
        )

    def fetch_batch(batch_idx, sem_idx):
        _fetch_bq(batch_idx, sem_idx, wait=False)
        _fetch_bkv(batch_idx, sem_idx, wait=False)
        if hca is not None:
            _fetch_hca_bkv(batch_idx, sem_idx, wait=False)

    def wait_fetch(batch_idx, sem_idx):
        _fetch_bq(batch_idx, sem_idx, wait=True)
        _fetch_bkv(batch_idx, sem_idx, wait=True)
        if hca is not None:
            _fetch_hca_bkv(batch_idx, sem_idx, wait=True)

    def wait_send_bo(batch_idx, sem_idx):
        @pl.when(batch_idx >= 0)
        def _():
            _send_bo(batch_idx, sem_idx, wait=True)

    start_batch_idx = 0

    if hca is not None:
        # Masked HCA slots are multiplied by p == 0; stale VMEM may be NaN.
        hca.bkv_x2_vmem[...] = jnp.zeros(hca.bkv_x2_vmem.shape, jnp.uint8)

    @pl.when(start_batch_idx < num_batches)
    def _():
        fetch_batch(start_batch_idx, 0)

    def loop_body(i, _):
        batch_idx = i

        sem_idx = i % 2
        next_sem_idx = (i + 1) % 2

        next_batch_idx = batch_idx + 1

        @pl.when(next_batch_idx < num_batches)
        def _():
            fetch_batch(next_batch_idx, next_sem_idx)

        wait_fetch(batch_idx, sem_idx)

        wait_send_bo(batch_idx - 2, sem_idx)

        compute_batch(batch_idx, sem_idx)

        _send_bo(batch_idx, sem_idx, wait=False)

        return ()

    loop_cnt = num_batches
    lax.fori_loop(0, loop_cnt, loop_body, ())

    @pl.when(loop_cnt > 0)
    def _():
        last_i = loop_cnt - 1
        last_batch_idx = last_i
        window.wait_stores(num_decode_seqs, store_sem)
        wait_send_bo(last_batch_idx, last_i % 2)

        @pl.when(loop_cnt > 1)
        def _():
            wait_send_bo(last_batch_idx - 1, (last_i - 1) % 2)


def run_mla_batched_decode_kernel(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    new_kv: jax.Array,  # [max_num_tokens, lkv_dim]
    # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    cache_kv: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_kv_pages_per_block: int,
    num_queries_per_block: int,
    start_seq_idx: jax.Array,  # i32
    end_seq_idx: jax.Array,  # i32
    # [max_num_tokens, actual_num_q_heads, head_dim]; None with `hca_decode`,
    # which writes the output over q.
    in_output: jax.Array | None,
    in_l: jax.Array,  # [max_num_tokens, num_l_heads]
    in_m: jax.Array,  # [max_num_tokens, num_l_heads]
    attention_sinks: jax.Array,  # float32[num_q_heads]
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    vmem_limit_bytes: int,
    unnormalized_output: bool,
    seq_batch_sz: int = 32,
    hca_decode: HcaDecodeInputs | None = None,
):
    assert (in_output is None) == (hca_decode is not None)
    num_decode_seqs = end_seq_idx - start_seq_idx
    bkv_p = num_kv_pages_per_block
    bkv_sz = bkv_p * logical_page_size

    grid = (jnp.where(num_decode_seqs > 0, 1, 0),)

    _, num_q_heads, head_dim = q.shape
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    num_l_heads = align_to(num_q_heads, 128)
    (
        total_num_pages,
        kv_packing,
        lkv_dim,
        slots_per_token,
        phys_tokens_per_page,
        pages_per_seq,
    ) = get_physical_layout(
        csa_cache_layout.u8_slab_shape(cache_kv.shape, cache_kv.dtype),
        q.dtype,
        head_dim,
        max_num_seqs,
        num_page_indices,
    )

    # The last batch of decode sequences reads up to seq_batch_sz - 1 rows
    # past them.
    assert seq_batch_sz <= SWA_COPY_TABLE_PAD
    copy_table = swa_decode_copy_table(
        kv_lens,
        page_indices,
        cu_q_lens,
        end_seq_idx,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        bkv_p=bkv_p,
        phys_tokens_per_page=phys_tokens_per_page,
    )
    scalar_prefetches = (
        kv_lens,
        copy_table,
        cu_q_lens,
        jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
    )
    if hca_decode is not None:
        scalar_prefetches += (
            hca_decode.kv_lens,
            hca_decode.kv_lens_to_attend,
            hca_decode.page_indices,
        )

    operands = (attention_sinks, q, new_kv, cache_kv)
    if in_output is not None:
        operands += (in_output,)
    operands += (in_l, in_m)
    # attention_sinks in VMEM, the rest in HBM.
    in_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)]
    in_specs += [pl.BlockSpec(memory_space=pltpu.HBM) for _ in operands[1:]]
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),  # out
        pl.BlockSpec(memory_space=pltpu.HBM),  # updated_cache_kv
        pl.BlockSpec(memory_space=pltpu.HBM),  # l
        pl.BlockSpec(memory_space=pltpu.HBM),  # m
    ]
    out_shape = [
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
        jax.ShapeDtypeStruct(shape=cache_kv.shape, dtype=cache_kv.dtype),
        jax.ShapeDtypeStruct(shape=(q.shape[0], num_l_heads), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(q.shape[0], num_l_heads), dtype=jnp.float32),
    ]
    base = len(scalar_prefetches)
    input_output_aliases = {
        base + (1 if in_output is None else 4): 0,  # q or in_output with out
        base + 3: 1,  # Alias cache_kv with updated_cache_kv
        base + len(operands) - 2: 2,  # Alias in_l with out_l
        base + len(operands) - 1: 3,  # Alias in_m with out_m
    }

    # Double buffered VMEM allocations
    bkv_batch_x2_vmem = pltpu.VMEM(
        (2, seq_batch_sz, bkv_sz, slots_per_token * kv_packing, lkv_dim),
        jnp.uint8,
    )
    q_batch_x2_vmem = pltpu.VMEM(
        (2, seq_batch_sz, num_q_heads, head_dim),
        q.dtype,
    )
    o_batch_x2_vmem = pltpu.VMEM(
        (2, seq_batch_sz, num_q_heads, head_dim),
        q.dtype,
    )
    l_batch_x2_vmem = pltpu.VMEM(
        (2, seq_batch_sz, num_l_heads),
        jnp.float32,
    )
    m_batch_x2_vmem = pltpu.VMEM(
        (2, seq_batch_sz, num_l_heads),
        jnp.float32,
    )
    # 4 for q, o, l, m; seq_batch_sz for bkv; 1 for every new-kv store.
    sems = pltpu.SemaphoreType.DMA((2, seq_batch_sz + 5))

    scratch_shapes = [
        bkv_batch_x2_vmem,
        q_batch_x2_vmem,
        o_batch_x2_vmem,
        l_batch_x2_vmem,
        m_batch_x2_vmem,
        sems,
    ]

    kernel_fn = _mla_batched_decode_kernel
    hca_kwargs = {}
    hca_operands = ()
    scope_name = f"SWA-d-bq_1-bkvp_{bkv_p}"
    if hca_decode is not None:
        hca_cache = hca_decode.cache_kv
        assert hca_cache.dtype == jnp.uint8
        hca_page_size = hca_cache.shape[1] // slots_per_token
        in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))  # hca cache_kv
        hca_operands = (hca_cache,)
        scratch_shapes += [
            pltpu.VMEM(
                (
                    2,
                    seq_batch_sz,
                    hca_decode.num_kv_pages_per_block * hca_page_size,
                    slots_per_token * kv_packing,
                    lkv_dim,
                ),
                jnp.uint8,
            ),
            pltpu.SemaphoreType.DMA((2, seq_batch_sz)),
        ]
        kernel_fn = _mla_batched_decode_hca_kernel
        hca_kwargs = {"hca_bkv_p": hca_decode.num_kv_pages_per_block}
        scope_name += f"-hca_bkvp_{hca_decode.num_kv_pages_per_block}"

    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                kernel_fn,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                logical_page_size=logical_page_size,
                unnormalized_output=unnormalized_output,
                bkv_p=bkv_p,
                seq_batch_sz=seq_batch_sz,
                **hca_kwargs,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=out_shape,
            input_output_aliases=input_output_aliases,
            name=scope_name,
        )
    )
    return kernel(*scalar_prefetches, *operands, *hca_operands)


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "logical_page_size",
        "unnormalized_output",
        "non_causal_block",
        "q_compute_block_size",
        "lane_padded_lm",
        "hca_num_kv_pages_per_block",
        "decode_by_caller",
    ),
    donate_argnames=("cache_kv",),
)
def mla_sliding_window_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    new_kv: jax.Array,  # [max_num_tokens, actual_head_dim]
    # [total_num_pages, physical_page_size_per_kv_packing, 4, 128] uint8
    cache_kv: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    *,
    sm_scale: float = 1.0,
    sliding_window: int,
    logical_page_size: int,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params for decode, prefill, and mixed cases.
    # If passsed in as int, all cases are the same.
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    q_compute_block_size: int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    unnormalized_output: bool = False,
    # Draft (DSpark/DFlash) block mode: the last q_len positions of each
    # sequence attend bidirectionally within themselves (window still bounds
    # context keys). Requires q_len <= sliding_window.
    non_causal_block: bool = False,
    decode_seq_batch_size: int = 32,
    # Return l/m at the kernel's 128-lane width instead of slicing to the heads.
    lane_padded_lm: bool = False,
    # Fuse the HCA pass into every case (see `HcaDecodeInputs`): the result is
    # then (final output, updated cache), the output written over q.
    hca_cache_kv: jax.Array | None = None,
    hca_kv_lens: jax.Array | None = None,
    hca_kv_lens_to_attend: jax.Array | None = None,
    hca_page_indices: jax.Array | None = None,
    hca_num_kv_pages_per_block: int | None = None,
    # Skip the decode rows and append their `SwaDecodeInputs` to the result:
    # a fused kernel runs them (see `sparse_mla.SwaDecodeRefs`).
    decode_by_caller: bool = False,
) -> tuple[
    jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    jax.Array,  # [total_num_pages, physical_page_size_per_kv_packing, 4, 128] uint8
    jax.Array,  # [max_num_tokens, actual_num_q_heads]
    jax.Array,  # [max_num_tokens, actual_num_q_heads]
]:
    """MLA Ragged paged attention that supports mixed prefill and decode.

    Args:
      q: concatenated all sequences' queries.
      new_kv: concatenated all sequences' kv values
      cache_kv: the current kv cache: uint8 slabs, or an int32
        [total_num_pages, rows, 128] CSA host array holding the same bytes
        (see `csa_cache_layout`).
      kv_lens: the length of each sequence in the kv cache.
      page_indices: flattened page indices look-up table by (seq_id, page_id).
      cu_q_lens: the cumulative sum of the effective query lengths. Similar to
        kv_lens, only the first num_seqs+1 values are valid.
      distribution: (i, j, k) represents that sequences[0:i] are decode-only,
        sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
        k is also the total number of sequences.
      sm_scale: the softmax scale which will be applied to the Q@K^T.
      sliding_window: the sliding window size for the attention.
      logical_page_size: the logical page size. The second and third dimensions of
        the kv cache are related to the physical page size, which is guaranteed to
        be >= logical_page_size. For model with hybrid attentions, such as
        DeepSeek V4, vLLM may use a single physical page pool shared across
        different attention types. Therefore, multiple attention types' pages will
        be padded up to the same physical page size. For the perspective of this
        kernel, it only read & write data within the logical page size.
      num_kv_pages_per_block: number of kv pages to be processed in one flash
        attention block in the pallas kernel. This is a tuple of (decode, prefill,
        mixed) cases.
      num_queries_per_block: number of queries to be processed in one flash
        attention block in the pallas kernel. This is a tuple of (decode, prefill,
        mixed) cases.
      vmem_limit_bytes: the vmem limit for the pallas kernel.

    Returns:
      The output of attention and the updated kv cache.
    """
    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be specified."
        )
    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
    else:
        num_kv_pages_per_blocks = num_kv_pages_per_block

    if isinstance(num_queries_per_block, int):
        num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
    else:
        num_queries_per_blocks = num_queries_per_block

    _, actual_num_q_heads, actual_head_dim = q.shape

    q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
    attention_sinks = jnp.pad(
        attention_sinks,
        (0, q.shape[1] - actual_num_q_heads),
        constant_values=jnp.finfo(attention_sinks.dtype).min,
    )
    assert new_kv.dtype == jnp.bfloat16
    # The SWA KV cache overlay onto the compressed kv cache of CSA, whose
    # size is (num_pages, page_size, 2, 128) uint8.
    #
    # Each token's 512 bf16 is stored as (2, 4, 128) uint8 bytes. we skip the
    # quantization and dequantization for DSV4 fp8 format here because
    # align(448 + 64 + 7, 512) = 1024 bytes, which is same as the plain 512 bf16,
    # thus no benefit from dsv4 fp8 format.
    #
    # The CSA host may be declared int32 (see `csa_cache_layout`); the kernels
    # view it as the same uint8 slabs.
    assert cache_kv.dtype in (jnp.uint8, jnp.int32)
    head_dim = q.shape[-1]
    _, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
        csa_cache_layout.u8_slab_shape(cache_kv.shape, cache_kv.dtype)
    )

    slot_bytes = kv_packing * lkv_dim
    token_bytes = head_dim * get_dtype_bitwidth(new_kv.dtype) // 8
    assert token_bytes % slot_bytes == 0
    slots_per_token = token_bytes // slot_bytes
    phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token

    new_kv = prepare_kv_inputs(new_kv)  # [max_num_tokens, token_bytes] uint8
    new_kv = new_kv.reshape(new_kv.shape[0], slots_per_token * kv_packing, lkv_dim)
    assert logical_page_size <= phys_tokens_per_page

    _, num_q_heads, _ = q.shape
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    kv_lens, page_indices = narrow_page_indices(
        kv_lens,
        page_indices,
        cu_q_lens,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        max_q_len=q.shape[0],
        max_kv_pages_per_block=max(num_kv_pages_per_blocks),
    )

    def run_mla_kernel(
        q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
        new_kv: jax.Array,  # [max_num_tokens, lkv_dim]
        # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
        cache_kv: jax.Array,
        kv_lens: jax.Array,  # i32[max_num_seqs]
        page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
        cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
        start_seq_idx: jax.Array,  # i32
        end_seq_idx: jax.Array,  # i32
        # [max_num_tokens, actual_num_q_heads, head_dim]; None with `hca`,
        # which writes the output over q.
        in_output: jax.Array | None,
        in_l: jax.Array,  # [max_num_tokens, num_l_heads]
        in_m: jax.Array,  # [max_num_tokens, num_l_heads]
        attention_sinks: jax.Array,  # float32[num_q_heads]
        static_q_len: int | None,
        unnormalized_output: bool,
        num_kv_pages_per_block: int,
        num_queries_per_block: int,
        case: MlaCase = MlaCase.MIXED,
        hca: HcaDecodeInputs | None = None,
    ):
        assert (in_output is None) == (hca is not None)
        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz = bkv_p * logical_page_size
        grid = (end_seq_idx - start_seq_idx,)

        operands = (attention_sinks, q, new_kv, cache_kv)
        if in_output is not None:
            operands += (in_output,)
        operands += (in_l, in_m)
        # attention_sinks in VMEM, the rest in HBM.
        in_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)]
        in_specs += [pl.BlockSpec(memory_space=pltpu.HBM) for _ in operands[1:]]

        out_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # o
            pl.BlockSpec(memory_space=pltpu.HBM),  # updated_cache_kv
            pl.BlockSpec(memory_space=pltpu.HBM),  # l
            pl.BlockSpec(memory_space=pltpu.HBM),  # m
        ]

        bkv_double_buf = pltpu.VMEM(
            (2, bkv_sz, slots_per_token * kv_packing, lkv_dim),
            jnp.uint8,
        )

        bq_double_bufq = pltpu.VMEM(
            (2, bq_sz, num_q_heads, head_dim),
            q.dtype,
        )

        bo_double_buf = bq_double_bufq

        num_l_heads = align_to(num_q_heads, 128)
        bl_double_buf = pltpu.VMEM(
            (2, bq_sz, num_l_heads),
            jnp.float32,
        )
        bm_double_buf = bl_double_buf

        l_scratch = pltpu.VMEM(
            (bq_sz * num_q_heads,),
            jnp.float32,
        )
        m_scratch = l_scratch

        acc_scratch = pltpu.VMEM(
            (bq_sz * num_q_heads, head_dim),
            jnp.float32,
        )

        scratch_shapes = [
            bkv_double_buf,
            bq_double_bufq,
            bo_double_buf,  # Double buffering for output block.
            bl_double_buf,  # Double buffering for l output.
            bm_double_buf,  # Double buffering for m output.
            # Semaphores for double buffering of bkv, bq, bo, bkv_update, l, m.
            pltpu.SemaphoreType.DMA((6, 2)),
            # Intermediate buffers per kv head for flash attention.
            l_scratch,
            m_scratch,
            acc_scratch,
        ]

        scalar_prefetches = (
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
            jnp.zeros((3,), jnp.int32),
            # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
            jnp.full((4,), -1, jnp.int32),
            # (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset,
            #     bkv_sem_0_sz, bkv_sem_1_sz)
            jnp.full((6,), -1, jnp.int32),
        )

        scope_name = (
            f"SWA-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}"
            f"{'-ncb' if non_causal_block else ''}"
        )
        kernel_fn = _mla_sliding_window_ragged_paged_attention_kernel
        hca_kwargs = {}
        hca_operands = ()
        if hca is not None:
            scalar_prefetches += (hca.kv_lens, hca.kv_lens_to_attend, hca.page_indices)
            hca_page_size = hca.cache_kv.shape[1] // slots_per_token
            in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))  # hca cache_kv
            hca_operands = (hca.cache_kv,)
            scratch_shapes += [
                pltpu.VMEM(
                    (
                        2,
                        hca.num_kv_pages_per_block * hca_page_size,
                        slots_per_token * kv_packing,
                        lkv_dim,
                    ),
                    jnp.uint8,
                ),
                pltpu.SemaphoreType.DMA((2,)),
            ]
            kernel_fn = _mla_sliding_window_hca_kernel
            hca_kwargs = {"hca_bkv_p": hca.num_kv_pages_per_block}
            scope_name += f"-hca_bkvp_{hca.num_kv_pages_per_block}"
        base = len(scalar_prefetches)
        input_output_aliases = {
            base + (1 if in_output is None else 4): 0,  # q or in_output with out
            base + 3: 1,  # Alias cache_kv with updated_cache_kv
            base + len(operands) - 2: 2,  # Alias in_l with out_l
            base + len(operands) - 1: 3,  # Alias in_m with out_m
        }
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    kernel_fn,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    static_q_len=static_q_len,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
                    logical_page_size=logical_page_size,
                    unnormalized_output=unnormalized_output,
                    non_causal_block=non_causal_block,
                    q_compute_block_size=q_compute_block_size,
                    **hca_kwargs,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                ),
                out_shape=[
                    jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                    jax.ShapeDtypeStruct(shape=cache_kv.shape, dtype=cache_kv.dtype),
                    jax.ShapeDtypeStruct(
                        shape=(q.shape[0], num_l_heads), dtype=jnp.float32
                    ),
                    jax.ShapeDtypeStruct(
                        shape=(q.shape[0], num_l_heads), dtype=jnp.float32
                    ),
                ],
                input_output_aliases=input_output_aliases,
                name=scope_name,
            )
        )
        return kernel(*scalar_prefetches, *operands, *hca_operands)

    num_l_heads = align_to(num_q_heads, 128)
    l = jnp.zeros((q.shape[0], num_l_heads), dtype=jnp.float32)  # noqa: E741
    m = jnp.zeros((q.shape[0], num_l_heads), dtype=jnp.float32)
    # output of swa attn is consumed by csa, hca attn, padding tokens'
    # data won't be used downstream at all, so un-initialized buffer is fine.
    # With HCA fusion every case writes its final rows over q instead.
    in_output = None
    if hca_num_kv_pages_per_block is None:
        in_output = jnp.empty_like(q) if unnormalized_output else jnp.zeros_like(q)

    hca_decode = None
    if hca_num_kv_pages_per_block is not None:
        if not uses_batched_decode(
            sliding_window, logical_page_size, num_kv_pages_per_blocks[0]
        ):
            raise ValueError("HCA fusion needs the batched decode kernel.")
        if not unnormalized_output:
            raise ValueError("HCA fusion folds the unnormalized SWA state.")
        assert hca_kv_lens.shape == (max_num_seqs,)
        assert hca_page_indices.shape[0] % max_num_seqs == 0
        hca_decode = HcaDecodeInputs(
            hca_cache_kv,
            hca_kv_lens,
            hca_kv_lens_to_attend,
            hca_page_indices,
            hca_num_kv_pages_per_block,
        )

    if decode_by_caller:
        if hca_decode is not None or not unnormalized_output:
            raise ValueError("decode_by_caller hands over the unnormalized state.")
        if not uses_batched_decode(
            sliding_window, logical_page_size, num_kv_pages_per_blocks[0]
        ):
            raise ValueError("decode_by_caller needs single-block decode.")
        output, updated_kv, out_l, out_m = in_output, cache_kv, l, m
    elif uses_batched_decode(
        sliding_window, logical_page_size, num_kv_pages_per_blocks[0]
    ):
        output, updated_kv, out_l, out_m = run_mla_batched_decode_kernel(
            q,
            new_kv,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=num_queries_per_blocks[0],
            start_seq_idx=jnp.array(0),
            end_seq_idx=distribution[0],
            in_output=in_output,
            in_l=l,
            in_m=m,
            attention_sinks=attention_sinks,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            logical_page_size=logical_page_size,
            vmem_limit_bytes=vmem_limit_bytes,
            unnormalized_output=unnormalized_output,
            seq_batch_sz=decode_seq_batch_size,
            hca_decode=hca_decode,
        )
    else:
        output, updated_kv, out_l, out_m = run_mla_kernel(
            q,
            new_kv,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            start_seq_idx=jnp.array(0),
            end_seq_idx=distribution[0],
            in_output=in_output,
            in_l=l,
            in_m=m,
            attention_sinks=attention_sinks,
            static_q_len=1,
            unnormalized_output=unnormalized_output,
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=num_queries_per_blocks[0],
            case=MlaCase.DECODE,
        )

    def run_non_decode(output, updated_kv, out_l, out_m):
        def run_case(**kwargs):
            if hca_decode is not None:
                # Finish the rows in `output`, the buffer that holds q with
                # the decode rows already final.
                return run_mla_kernel(
                    output,
                    new_kv,
                    updated_kv,
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    in_output=None,
                    in_l=out_l,
                    in_m=out_m,
                    hca=hca_decode,
                    **kwargs,
                )
            return run_mla_kernel(
                q,
                new_kv,
                updated_kv,
                kv_lens,
                page_indices,
                cu_q_lens,
                in_output=output,
                in_l=out_l,
                in_m=out_m,
                **kwargs,
            )

        if chunk_prefill_size is not None:
            # Handle prefill where the query length is fixed per sequence.
            output, updated_kv, out_l, out_m = run_case(
                num_kv_pages_per_block=num_kv_pages_per_blocks[1],
                num_queries_per_block=num_queries_per_blocks[1],
                start_seq_idx=distribution[0],
                end_seq_idx=distribution[1],
                attention_sinks=attention_sinks,
                static_q_len=chunk_prefill_size,
                unnormalized_output=unnormalized_output,
                case=MlaCase.PREFILL,
            )

        # Handle mixed case where the query length per sequence is variable.
        output, updated_kv, out_l, out_m = run_case(
            num_kv_pages_per_block=num_kv_pages_per_blocks[2],
            num_queries_per_block=num_queries_per_blocks[2],
            start_seq_idx=distribution[1],
            end_seq_idx=distribution[2],
            attention_sinks=attention_sinks,
            static_q_len=None,
            unnormalized_output=unnormalized_output,
            case=MlaCase.MIXED,
        )
        return output, updated_kv, out_l, out_m

    if hca_decode is not None:
        # Skip the prefill and mixed launches in decode-only steps.
        output, updated_kv, out_l, out_m = lax.cond(
            distribution[2] > distribution[0],
            run_non_decode,
            lambda *state: state,
            output,
            updated_kv,
            out_l,
            out_m,
        )
    else:
        output, updated_kv, out_l, out_m = run_non_decode(
            output, updated_kv, out_l, out_m
        )
    output = prepare_outputs(
        output, actual_num_q_heads, actual_head_dim
    )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    if hca_decode is not None:
        return output, updated_kv
    if not lane_padded_lm:
        out_l = out_l[:, :actual_num_q_heads]
        out_m = out_m[:, :actual_num_q_heads]
    if decode_by_caller:
        copy_table = swa_decode_copy_table(
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution[0],
            sliding_window=sliding_window,
            logical_page_size=logical_page_size,
            bkv_p=num_kv_pages_per_blocks[0],
            phys_tokens_per_page=phys_tokens_per_page,
        )
        decode = SwaDecodeInputs(
            new_kv, updated_kv, kv_lens, copy_table, distribution[0]
        )
        return output, updated_kv, out_l, out_m, decode
    return output, updated_kv, out_l, out_m
