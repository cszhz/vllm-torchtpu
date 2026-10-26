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
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather, mla_swa

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    kv_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        align_to(kv_dim, 128),
    )


def _largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of ``x`` that is <= ``cap``."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


# DSV4 FP8 nope layout: 448 e4m3 values, then one e8m0 scale per lane of a
# 64-lane period -- 64 quantization groups of 7 dims, group j holding the dims
# {j, j + 64, ... j + 384}. The 64 scale bytes fill the row's padding, so the
# record is still 448 + 64 = 512 bytes.
NOPE_DIM = 448
NOPE_SCALE_PERIOD = 64


def _dequant_dsv4_fp8(words: jax.Array):
    """Dequantize gathered NoPE records to BF16 [rows, NOPE_DIM].

    `words` is int32 [rows, 128], byte b of word w holding record byte
    128 b + w (see `csa_cache_layout`). Each byte plane is converted on its own
    and the planes are joined along lanes, which is the record in dim order; a
    uint8 [4 rows, 128] -> [rows, 512] reshape first would relayout every byte
    across sublanes.
    """
    planes = [lax.shift_right_logical(words, 8 * b).astype(jnp.uint8) for b in range(4)]
    nope_fp8 = jnp.concatenate(
        [
            pltpu.bitcast(plane, jnp.float8_e4m3fn).astype(jnp.bfloat16)
            for plane in planes
        ],
        axis=1,
    )[:, :NOPE_DIM]
    # scales[d] = scales[d % 64]; the 64 scale bytes are the upper half of plane 3.
    # e8m0 is a bare exponent with f32's bias, so moving the byte into the f32
    # exponent field is 2^(e - 127) exactly for every code the compressor
    # writes (its amax clamp keeps e off 0 and 0xFF). That skips the e8m0
    # convert's special-code selects; rows past kv_len are zeroed anyway.
    scale_bits = lax.shift_left(lax.shift_right_logical(words, 24), 23)
    tail = pltpu.bitcast(scale_bits, jnp.float32).astype(jnp.bfloat16)
    lane = lax.broadcasted_iota(jnp.int32, tail.shape, 1)
    tile = jnp.where(
        lane < NOPE_SCALE_PERIOD, pltpu.roll(tail, NOPE_SCALE_PERIOD, 1), tail
    )
    nope_scales = jnp.concatenate([tile] * 4, axis=1)[:, :NOPE_DIM]
    return (nope_fp8 * nope_scales).astype(jnp.bfloat16)


class SwaDecodeRefs(NamedTuple):
    """The decode tokens' sliding-window pass the CSA kernel runs itself,
    instead of reading the SWA kernel's state back from HBM."""

    window: mla_swa.SwaDecodeWindowRefs
    # [2, batch_size, bkv_sz, slots_per_token * kv_packing, lkv_dim] uint8
    bkv_x2_vmem: Any
    sems: Any  # DMA [2, batch_size], the window fetches
    store_sem: Any  # DMA, every new kv store


def _attention_swa_decode_kernel(
    kv_lens_ref,
    start_end_seq_idx_ref,
    sem_ids_ref,
    bo_ids_ref,
    swa_kv_lens_ref,  # [max_num_seqs]
    swa_copy_table_ref,  # `mla_swa.swa_decode_copy_table`
    attention_sinks_ref,
    q_hbm_ref,
    cache_kv_nope_hbm_ref,
    cache_kv_rope_hbm_ref,
    swa_accumution_hbm_ref,
    swa_l_hbm_ref,
    swa_m_hbm_ref,
    swa_new_kv_hbm_ref,  # `mla_swa.prepare_kv_inputs` words
    swa_cache_hbm_ref,  # SWA cache: uint8 slabs or an int32 CSA host array
    o_hbm_ref,
    updated_swa_cache_hbm_ref,
    *scratch,
    swa_sliding_window: int,
    swa_logical_page_size: int,
    swa_bkv_p: int,
    **kwargs,
):
    """`_attention_kernel` that also runs the decode tokens' sliding-window
    pass (see `SwaDecodeRefs`) and writes their new kv into the SWA cache.
    Decode token t is decode sequence t; start_end_seq_idx_ref[3] holds the
    number of decode tokens."""
    *scratch, swa_bkv_x2_vmem, swa_sems = scratch
    swa_new_kv_hbm_ref = mla_swa.as_new_kv_rows(
        swa_new_kv_hbm_ref, swa_bkv_x2_vmem.shape[3] * swa_bkv_x2_vmem.shape[4]
    )
    cache = csa_cache_layout.as_u8_slabs(swa_cache_hbm_ref)
    updated_cache = csa_cache_layout.as_u8_slabs(updated_swa_cache_hbm_ref)
    total_num_pages, phys_page_size_per_kv_packing, kv_packing, lkv_dim = cache.shape
    slots = swa_bkv_x2_vmem.shape[3]
    token_bytes = slots * lkv_dim
    phys_tokens_per_page = phys_page_size_per_kv_packing * kv_packing // slots
    flat_shape = (total_num_pages * phys_tokens_per_page, slots, lkv_dim)
    window = mla_swa.SwaDecodeWindowRefs(
        swa_kv_lens_ref,
        swa_copy_table_ref,
        cache.reshape(flat_shape),
        updated_cache.reshape(flat_shape),
        swa_new_kv_hbm_ref,
        swa_sliding_window,
        swa_logical_page_size,
        swa_bkv_p,
        token_bytes,
    )
    _attention_kernel(
        kv_lens_ref,
        start_end_seq_idx_ref,
        sem_ids_ref,
        bo_ids_ref,
        attention_sinks_ref,
        q_hbm_ref,
        cache_kv_nope_hbm_ref,
        cache_kv_rope_hbm_ref,
        swa_accumution_hbm_ref,
        swa_l_hbm_ref,
        swa_m_hbm_ref,
        o_hbm_ref,
        *scratch,
        swa=SwaDecodeRefs(
            window,
            swa_bkv_x2_vmem,
            swa_sems.at[:, : swa_bkv_x2_vmem.shape[1]],
            swa_sems.at[0, swa_bkv_x2_vmem.shape[1]],
        ),
        **kwargs,
    )


# Prefill and mixed calls of more tokens than this read the rows of a
# `DenseRefs` batch from the paged cache; decode-sized calls never do.
DENSE_MIN_TOKENS = 256
# Tokens of a `DenseRefs` batch per QK / PV matmul over its shared rows.
DENSE_TOKENS_PER_MATMUL = 1
# A `DenseRefs` batch attends over its first rows in multiples of this many,
# enough to cover its longest kv_len.
DENSE_KV_BLOCK = 256


class DenseRefs(NamedTuple):
    """Batches of one sequence's tokens that all have fewer than top_k valid
    entries. The indexer then selected every compressed token visible to each
    of them, in position order (the SparseCore top-k writes a row it keeps
    whole in column order), so a token's gathered row j is row j of its
    sequence for every j < kv_len. Such a batch copies those rows once from
    the paged cache and dequantizes them once, instead of once per token."""

    # SMEM i32[num_tokens]: token t's sequence s, or -(s + 1) when t is in
    # such a batch.
    token_seq_ref: Any
    page_indices_ref: Any  # SMEM i32[max_num_seqs * pages_per_seq]
    cache_nope_hbm_ref: Any  # int32[total_pages, page_size, 128]
    cache_rope_hbm_ref: Any  # int32[total_pages, page_size // 4, 128]
    rope_vmem_ref: Any  # int32[topk // 4, 128], the rows in csa_gather's rope layout
    kv_vmem_ref: Any  # bf16[topk, head_dim], the batch's dequantized rows
    pages_per_seq: int


def _attention_dense_kernel(
    kv_lens_ref,
    start_end_seq_idx_ref,
    sem_ids_ref,
    bo_ids_ref,
    token_seq_ref,
    page_indices_ref,
    attention_sinks_ref,
    q_hbm_ref,
    cache_kv_nope_hbm_ref,
    cache_kv_rope_hbm_ref,
    swa_accumution_hbm_ref,
    swa_l_hbm_ref,
    swa_m_hbm_ref,
    cache_nope_hbm_ref,
    cache_rope_hbm_ref,
    o_hbm_ref,
    *scratch,
    pages_per_seq: int,
    **kwargs,
):
    """`_attention_kernel` that reads `DenseRefs` batches from the paged cache."""
    *scratch, rope_vmem_ref, kv_vmem_ref = scratch
    _attention_kernel(
        kv_lens_ref,
        start_end_seq_idx_ref,
        sem_ids_ref,
        bo_ids_ref,
        attention_sinks_ref,
        q_hbm_ref,
        cache_kv_nope_hbm_ref,
        cache_kv_rope_hbm_ref,
        swa_accumution_hbm_ref,
        swa_l_hbm_ref,
        swa_m_hbm_ref,
        o_hbm_ref,
        *scratch,
        dense=DenseRefs(
            token_seq_ref,
            page_indices_ref,
            cache_nope_hbm_ref,
            cache_rope_hbm_ref,
            rope_vmem_ref,
            kv_vmem_ref,
            pages_per_seq,
        ),
        **kwargs,
    )


def _rope_rows_as_gathered(rows: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The even and odd int32 rows of csa_gather's rope block for top-k
    entries that are consecutive cache tokens.

    `rows` is int32[topk // 4, 128] in the cache's layout: row i holds tokens
    4i..4i+3, word k of a token holding its bf16 k (low) and k + 32 (high).
    csa_gather's row 2i holds tokens (4i, 4i + 1), row 2i + 1 tokens
    (4i + 2, 4i + 3), word k pairing one bf16 of both, and its lanes 64:128
    the same for the tokens topk // 2 further on.
    """
    words = csa_cache_layout.ROPE_WORDS
    lane = lax.broadcasted_iota(jnp.int32, rows.shape, 1)
    # Lanes [0, 64) pair tokens 4i and 4i + 1, lanes [64, 128) 4i + 2 and 4i + 3.
    pairs = jnp.where(
        lane % (2 * words) < words,
        jnp.bitwise_or(
            jnp.bitwise_and(rows, 0xFFFF),
            jnp.left_shift(pltpu.roll(rows, rows.shape[1] - words, 1), 16),
        ),
        jnp.bitwise_or(
            lax.shift_right_logical(pltpu.roll(rows, words, 1), 16),
            jnp.bitwise_and(rows, jnp.int32(-65536)),  # 0xFFFF0000
        ),
    )
    half = rows.shape[0] // 2
    lower = lane[:half] < 2 * words
    first, second = pairs[:half], pairs[half:]
    even = jnp.where(lower, first, pltpu.roll(second, 2 * words, 1))
    odd = jnp.where(lower, pltpu.roll(first, 2 * words, 1), second)
    return even, odd


def _attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    # [3] (start_seq_idx, end_seq_idx, num_valid_tokens), [4] with
    # `swa` (+ num_decode_tokens)
    start_end_seq_idx_ref,
    sem_ids_ref,  # [2] (bi_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [2] seq_idx_start of the in-flight bo DMA per bo sem
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    cache_kv_nope_hbm_ref,  # int32[chunk_size, topk, 128]
    cache_kv_rope_hbm_ref,  # int32[chunk_size, topk // 4, 128]
    swa_accumution_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    swa_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    swa_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # Scratch
    bkv_nope_x2_ref,  # [2, batch_size, page_size * 4, 128]
    bkv_rope_x2_ref,  # [2, batch_size, page_size, rope_dim]
    bq_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bo_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bl_x2_ref,  # [2, batch_size, num_l_heads]
    bm_x2_ref,  # [2, batch_size, num_l_heads]
    swa_acc_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    sems,  # [7, 2]
    *,
    sm_scale: float,
    batch_size: int = 1,
    swa: SwaDecodeRefs | None = None,
    dense: DenseRefs | None = None,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape

    num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
    assert kv_lens_ref.shape[0] == num_tokens
    # csa_gather emits int32 words; these are the uint8 nope slabs
    # [.., topk * 4, 128] and bf16 rope rows [.., topk // 2, 128] they encode,
    # a free bitcast of the same bytes.
    cache_kv_nope_hbm_ref = cache_kv_nope_hbm_ref.bitcast(jnp.uint8)
    cache_kv_rope_hbm_ref = cache_kv_rope_hbm_ref.bitcast(jnp.bfloat16)
    # The gathered rope buffer carries two top-k entries per row (csa_gather).
    bkv_sz = cache_kv_rope_hbm_ref.shape[1] * 2

    q_dtype = q_hbm_ref.dtype
    q_packing = get_dtype_packing(q_dtype)
    # Validate against the KV dtype.
    assert o_hbm_ref.dtype == q_dtype

    assert head_dim % 128 == 0
    assert num_q_heads % q_packing == 0

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    num_valid_tokens = start_end_seq_idx_ref[2]

    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1
    # Tokens below this run the SWA pass here; the rest read its HBM state.
    num_decode_tokens = start_end_seq_idx_ref[3] if swa is not None else 0

    def flash_attention_step1_qk_softmax(
        q,  # [bq_sz * num_q_heads, head_dim]
        kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_*_x2_ref
        kv_len,  # scalar
        swa_m,  # [bq_sz * num_q_heads],
        swa_l,  # [bq_sz * num_q_heads],
        attention_sinks,  # [num_q_heads]
        masked,
    ):
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape[1] == head_dim

        # Follow FlashAttention-2 forward pass.
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
        if masked:
            k_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
            s = jnp.where(k_span < kv_len, s, jnp.finfo(s.dtype).min)

        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = swa_m
        m_curr = jnp.maximum(m_prev, s_rowmax)
        p = jnp.exp(s - m_curr)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        l_prev = swa_l
        l_curr = exp_m_diff * l_prev + p_rowsum
        exp_attention_sinks = jnp.exp(attention_sinks - m_curr)
        l_total = l_curr + exp_attention_sinks

        return p, exp_m_diff, l_total

    def flash_attention_step2_pv(
        p,
        kv,
        exp_m_diff,
        swa_acc,
        l_total,
    ):
        pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)

        o_prev = swa_acc
        acc = exp_m_diff * o_prev + pv
        out = (
            lax.div(acc, l_total)
            if q_dtype == jnp.float32
            else (acc * pl.reciprocal(l_total, approx=True)).astype(q_dtype)
        )
        return out

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_gathered_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
        sem_nope = sems.at[0, bkv_sem_idx]
        sem_rope = sems.at[6, bkv_sem_idx]

        bkv_nope_vmem_ref = bkv_nope_x2_ref.at[bkv_sem_idx, :]
        bkv_rope_vmem_ref = bkv_rope_x2_ref.at[bkv_sem_idx, :]

        # The index into cache_kv_hbm_ref should be relative to the current
        # chunk.
        page_idx_start = seq_idx_start - start_seq_idx
        if not wait:
            _async_copy(
                cache_kv_nope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
                bkv_nope_vmem_ref,
                sem_nope,
                wait,
            )
            _async_copy(
                cache_kv_rope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
                bkv_rope_vmem_ref,
                sem_rope,
                wait,
            )
        else:
            dst_nope = bkv_nope_vmem_ref
            _async_copy(src=dst_nope, dst=dst_nope, sem=sem_nope, wait=True)
            dst_rope = bkv_rope_vmem_ref
            _async_copy(src=dst_rope, dst=dst_rope, sem=sem_rope, wait=True)

    if dense is not None:
        # A `DenseRefs` batch's rows land in its slot's first batch entry, in
        # the cache's own layouts: uint8 nope slabs and bf16-viewed rope rows.
        dense_nope_hbm_ref = dense.cache_nope_hbm_ref.bitcast(jnp.uint8)
        dense_rope_hbm_ref = dense.cache_rope_hbm_ref.bitcast(jnp.bfloat16)
        rows_per_copy = min(dense.cache_nope_hbm_ref.shape[1], bkv_sz)
        assert bkv_sz % rows_per_copy == 0
        nope_rows_per_copy = rows_per_copy * csa_cache_layout.SLAB_ROWS
        rope_rows_per_copy = rows_per_copy // 2

    def is_dense_batch(seq_idx_start):
        return dense.token_seq_ref[seq_idx_start] < 0

    def _fetch_dense_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
        """Copy the cache pages holding rows [0, max kv_len) of the batch's
        sequence; rows past a token's kv_len are masked like gathered ones."""
        seq = -dense.token_seq_ref[seq_idx_start] - 1
        max_kv_len = kv_lens_ref[seq_idx_start]
        for b in range(1, batch_size):
            max_kv_len = jnp.maximum(max_kv_len, kv_lens_ref[seq_idx_start + b])
        num_copies = cdiv(max_kv_len, rows_per_copy)
        for i in range(bkv_sz // rows_per_copy):

            @pl.when(i < num_copies)
            def _():
                nope_dst = bkv_nope_x2_ref.at[
                    bkv_sem_idx, 0, pl.ds(i * nope_rows_per_copy, nope_rows_per_copy)
                ]
                rope_dst = bkv_rope_x2_ref.at[
                    bkv_sem_idx, 0, pl.ds(i * rope_rows_per_copy, rope_rows_per_copy)
                ]
                nope_src, rope_src = nope_dst, rope_dst
                if not wait:
                    page = dense.page_indices_ref[seq * dense.pages_per_seq + i]
                    nope_src = dense_nope_hbm_ref.at[page, pl.ds(0, nope_rows_per_copy)]
                    rope_src = dense_rope_hbm_ref.at[page, pl.ds(0, rope_rows_per_copy)]
                _async_copy(nope_src, nope_dst, sems.at[0, bkv_sem_idx], wait)
                _async_copy(rope_src, rope_dst, sems.at[6, bkv_sem_idx], wait)

    def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
        if dense is None:
            _fetch_gathered_batch(seq_idx_start, bkv_sem_idx, wait=wait)
            return
        from_cache = is_dense_batch(seq_idx_start)

        @pl.when(from_cache)
        def _():
            _fetch_dense_batch(seq_idx_start, bkv_sem_idx, wait=wait)

        @pl.when(jnp.logical_not(from_cache))
        def _():
            _fetch_gathered_batch(seq_idx_start, bkv_sem_idx, wait=wait)

    def _fetch_bq_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_vmem_ref = bq_x2_ref.at[bq_sem_idx, :]

        if not wait:
            _async_copy(
                q_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bq_vmem_ref,
                sem,
                wait,
            )
        else:
            _async_copy(src=bq_vmem_ref, dst=bq_vmem_ref, sem=sem, wait=True)

    def _send_bo_batch(seq_idx_start, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        sz = jnp.clip(num_valid_tokens - seq_idx_start, 1, batch_size)
        vmem_ref = bo_x2_ref.at[bo_sem_idx, pl.ds(0, sz)]

        if not wait:
            # Remember where this DMA started so its wait can rebuild the same
            # `sz`; a mismatch here deadlocks on the DMA semaphore.
            bo_ids_ref[bo_sem_idx] = seq_idx_start
            _async_copy(
                vmem_ref,
                o_hbm_ref.at[pl.ds(seq_idx_start, sz)],
                sem,
                wait,
            )
        else:
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def _swa_window_batch(seq_idx_start, slot, *, wait):
        """Fetch the decode tokens' window kv blocks (and store their new kv
        into the cache), or wait for the fetches."""
        for b in range(batch_size):
            token_idx = seq_idx_start + b
            args = (swa.bkv_x2_vmem.at[slot, b], swa.sems.at[slot, b])

            @pl.when(token_idx < num_decode_tokens)
            def _():
                swa.window.fetch(token_idx, *args, wait=wait)
                if not wait:
                    swa.window.store_new_kv(token_idx, swa.store_sem)

    def _fetch_swa_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        if swa is None:
            _fetch_swa_state_batch(seq_idx_start, bq_sem_idx, wait=wait)
            return

        @pl.when(seq_idx_start + batch_size > num_decode_tokens)
        def _():
            _fetch_swa_state_batch(seq_idx_start, bq_sem_idx, wait=wait)

        _swa_window_batch(seq_idx_start, bq_sem_idx, wait=wait)

    def _fetch_swa_state_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        sem_acc = sems.at[3, bq_sem_idx]
        sem_l = sems.at[4, bq_sem_idx]
        sem_m = sems.at[5, bq_sem_idx]

        if not wait:
            _async_copy(
                swa_accumution_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                swa_acc_x2_ref.at[bq_sem_idx, :],
                sem_acc,
                wait=False,
            )
            _async_copy(
                swa_l_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bl_x2_ref.at[bq_sem_idx, :],
                sem_l,
                wait=False,
            )
            _async_copy(
                swa_m_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bm_x2_ref.at[bq_sem_idx, :],
                sem_m,
                wait=False,
            )

        else:
            dst_acc = swa_acc_x2_ref.at[bq_sem_idx, :]
            _async_copy(src=dst_acc, dst=dst_acc, sem=sem_acc, wait=True)

            dst_l = bl_x2_ref.at[bq_sem_idx, :]
            _async_copy(src=dst_l, dst=dst_l, sem=sem_l, wait=True)

            dst_m = bm_x2_ref.at[bq_sem_idx, :]
            _async_copy(src=dst_m, dst=dst_m, sem=sem_m, wait=True)

    def start_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx)

    def wait_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, wait=True)

    def start_fetch_bq_batch(seq_idx_start, bq_sem_idx):
        return _fetch_bq_batch(seq_idx_start, bq_sem_idx)

    def wait_fetch_bq_batch(seq_idx_start, bq_sem_idx):
        return _fetch_bq_batch(seq_idx_start, bq_sem_idx, wait=True)

    def start_fetch_swa_batch(seq_idx_start, bq_sem_idx):
        return _fetch_swa_batch(seq_idx_start, bq_sem_idx)

    def wait_fetch_swa_batch(seq_idx_start, bq_sem_idx):
        return _fetch_swa_batch(seq_idx_start, bq_sem_idx, wait=True)

    def start_send_bo_batch(seq_idx_start, bo_sem_idx):
        return _send_bo_batch(seq_idx_start, bo_sem_idx)

    def wait_send_bo_batch(bo_sem_idx):
        # `sz` must match the started DMA exactly, so recover its
        # seq_idx_start rather than using the current step's.
        old_seq_idx_start = bo_ids_ref[bo_sem_idx]

        @pl.when(old_seq_idx_start >= 0)
        def _():
            _send_bo_batch(old_seq_idx_start, bo_sem_idx, wait=True)

    def load_bq(bq_sem_idx, batch_idx):
        q = bq_x2_ref.at[bq_sem_idx, batch_idx][...]
        return q

    def join_bkv(words, bkv_rope):
        """The bf16 [bkv_sz, head_dim] rows of NoPE words and a rope block."""
        bkv_nope = _dequant_dsv4_fp8(words)

        # The gathered rope block is [bkv_sz // 2, 2 * rope_dim]: entry i in
        # lanes 0:rope_dim, entry i + bkv_sz // 2 in the upper half (see
        # csa_gather). Mosaic has no [m, 2n] -> [2m, n] shape cast, but the
        # lane slice + row concatenate below is one lane rotate per vreg.
        bkv_rope = bkv_rope.reshape(bkv_sz // 2, -1)
        rope_dim = bkv_rope.shape[-1] // 2
        bkv_rope = jnp.concatenate(
            [bkv_rope[:, :rope_dim], bkv_rope[:, rope_dim:]], axis=0
        )
        return jnp.concatenate([bkv_nope, bkv_rope[: words.shape[0]]], axis=-1)

    def load_bkv(bkv_sem_idx, batch_idx, masked):
        # The gathered NoPE rows as the int32 words csa_gather wrote.
        words = bkv_nope_x2_ref.bitcast(jnp.int32).at[bkv_sem_idx, batch_idx][...]
        bkv = join_bkv(words, bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx][...])

        # In vLLM, multiple caches may overlay on the same KV Tensor. For example,
        # compressor state cache write data in bfloat16 / float32 format, certain
        # byte pattern are interpreted as NaN in FP8, e.g. float8_e8m0fnu byte 0xFF
        # decodes to NaN.
        # We need to mask out the data by the actual kv_len to avoid NaN propagting
        # to the downstream computation.
        if masked:
            kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
            k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
            bkv = jnp.where(k_span < kv_len, bkv, 0)
        return bkv

    def build_dense_bkv(bkv_sem_idx, num_rows):
        """Dequantize a `DenseRefs` batch's first `num_rows` rows into
        `dense.kv_vmem_ref`, the same rows `load_bkv` builds from a gather."""
        nope_words = bkv_nope_x2_ref.bitcast(jnp.int32).at[bkv_sem_idx, 0]
        words = nope_words[pl.ds(0, num_rows)]
        rows = bkv_rope_x2_ref.bitcast(jnp.int32).at[bkv_sem_idx, 0][...]
        even, odd = _rope_rows_as_gathered(rows)
        half = even.shape[0]
        dense.rope_vmem_ref[pl.ds(0, half, stride=2), :] = even
        dense.rope_vmem_ref[pl.ds(1, half, stride=2), :] = odd
        bkv_rope = dense.rope_vmem_ref.bitcast(jnp.bfloat16)[...]
        dense.kv_vmem_ref[pl.ds(0, num_rows)] = join_bkv(words, bkv_rope)

    def attend_from_cache(bi_sem_idx, bo_sem_idx, attention_sinks):
        """`process`'s token loop for a `DenseRefs` batch, over only the
        DENSE_KV_BLOCK multiple of rows that covers its longest kv_len: the
        rows past it are masked for every token, which adds exact zeros to
        their softmax and PV sums."""
        kv_lens = [kv_lens_ref[batch_start_seq_idx + b] for b in range(batch_size)]
        max_kv_len = functools.reduce(jnp.maximum, kv_lens)
        num_rows = jnp.maximum(cdiv(max_kv_len, DENSE_KV_BLOCK), 1) * DENSE_KV_BLOCK
        for width in range(DENSE_KV_BLOCK, bkv_sz + 1, DENSE_KV_BLOCK):

            @pl.when(num_rows == width)
            def _(width=width):
                attend_rows_from_cache(
                    bi_sem_idx, bo_sem_idx, attention_sinks, kv_lens, max_kv_len, width
                )

    def attend_rows_from_cache(
        bi_sem_idx, bo_sem_idx, attention_sinks, kv_lens, max_kv_len, num_rows
    ):
        """The tokens share their rows, so each group of
        DENSE_TOKENS_PER_MATMUL tokens runs one QK and one PV matmul over
        them, the rows loaded into the MXU once."""
        build_dense_bkv(bi_sem_idx, num_rows)
        bkv = dense.kv_vmem_ref[pl.ds(0, num_rows)]
        # Rows past every token's kv_len may decode to NaN and are zeroed; a
        # row past only some tokens' kv_len is a finite row of the sequence,
        # which their masked scores give probability 0.
        k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
        bkv = jnp.where(k_span < max_kv_len, bkv, 0)
        group = min(DENSE_TOKENS_PER_MATMUL, batch_size)
        assert batch_size % group == 0
        for first in range(0, batch_size, group):
            tokens = range(first, first + group)
            states = [load_swa_output(bi_sem_idx, b) for b in tokens]
            swa_acc, swa_l, swa_m = (
                jnp.concatenate([state[i] for state in states], axis=0)
                for i in range(3)
            )
            p, exp_m_diff, l_total = flash_attention_step1_qk_softmax(
                jnp.concatenate([load_bq(bi_sem_idx, b) for b in tokens], axis=0),
                bkv,
                jnp.concatenate(
                    [jnp.full((num_q_heads, 1), kv_lens[b], jnp.int32) for b in tokens]
                ),
                swa_m,
                swa_l,
                jnp.concatenate([attention_sinks] * group, axis=0),
                masked=True,
            )
            out = flash_attention_step2_pv(p, bkv, exp_m_diff, swa_acc, l_total)
            for i, b in enumerate(tokens):
                bo_x2_ref.at[bo_sem_idx, b][...] = out[
                    i * num_q_heads : (i + 1) * num_q_heads
                ]

    def run_swa_decode(bq_sem_idx):
        """Write the decode tokens' unnormalized SWA state where the HBM
        handoff would have landed: acc in q_dtype, l/m in float32."""
        token_ids = [batch_start_seq_idx + b for b in range(batch_size)]
        max_num_seqs = swa.window.kv_lens_ref.shape[0]
        kv_lens_batch = [
            jnp.where(
                t < num_decode_tokens,
                swa.window.kv_lens_ref[jnp.minimum(t, max_num_seqs - 1)],
                0,
            )
            for t in token_ids
        ]
        pv, l_val, m_val = mla_swa.swa_decode_window_partials(
            bq_x2_ref[bq_sem_idx],
            swa.bkv_x2_vmem.at[bq_sem_idx],
            kv_lens_batch,
            bkv_sz=swa.window.bkv_sz,
            sliding_window=swa.window.sliding_window,
            sm_scale=sm_scale,
        )
        num_l_heads = bl_x2_ref.shape[-1]
        lane_pad = ((0, 0), (0, num_l_heads - num_q_heads))
        state = (
            (swa_acc_x2_ref, pv.astype(q_dtype)),
            (bl_x2_ref, jnp.pad(l_val, lane_pad)),
            (bm_x2_ref, jnp.pad(m_val, lane_pad)),
        )
        all_decode = batch_end_seq_idx < num_decode_tokens

        @pl.when(all_decode)
        def _():
            for ref, val in state:
                ref[bq_sem_idx] = val

        @pl.when(jnp.logical_not(all_decode))
        def _():
            for ref, val in state:
                token_ids = batch_start_seq_idx + lax.broadcasted_iota(
                    jnp.int32, val.shape, 0
                )
                is_decode = token_ids < num_decode_tokens
                ref[bq_sem_idx] = jnp.where(is_decode, val, ref[bq_sem_idx])

    def load_swa_output(bq_sem_idx, batch_idx):
        swa_acc = swa_acc_x2_ref[bq_sem_idx, batch_idx, ...]
        swa_l = bl_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
        swa_m = bm_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
        return swa_acc, swa_l, swa_m

    def process(masked, from_cache=False):
        def get_next_seq_ids(seq_idx, bi_sem_idx):
            next_seq_idx = seq_idx + batch_size
            next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bi_sem_idx

        bi_sem_idx = sem_ids_ref[0]
        next_seq_idx, next_bi_sem_idx = get_next_seq_ids(
            batch_start_seq_idx, bi_sem_idx
        )

        # Prefetch next seq
        @pl.when(next_seq_idx < end_seq_idx)
        def prefetch_next_seq():
            sem_ids_ref[0] = next_bi_sem_idx
            start_fetch_bq_batch(next_seq_idx, next_bi_sem_idx)
            start_fetch_swa_batch(next_seq_idx, next_bi_sem_idx)
            start_fetch_bkv_batch(next_seq_idx, next_bi_sem_idx)

        bo_sem_idx = sem_ids_ref[1]
        sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)
        attention_sinks = attention_sinks_ref[...][..., None]

        prev_p = None
        prev_bkv = None
        prev_exp_m_diff = None
        prev_l = None
        prev_swa_acc = None

        # Wait for cur blocks if not ready yet
        wait_fetch_bq_batch(batch_start_seq_idx, bi_sem_idx)
        wait_fetch_swa_batch(batch_start_seq_idx, bi_sem_idx)
        wait_fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx)

        if swa is not None:

            @pl.when(batch_start_seq_idx < num_decode_tokens)
            def _():
                run_swa_decode(bi_sem_idx)

        @pl.when(pl.program_id(0) >= 2)
        def _wait_send():
            wait_send_bo_batch(bo_sem_idx)

        if from_cache:
            attend_from_cache(bi_sem_idx, bo_sem_idx, attention_sinks)
            start_send_bo_batch(batch_start_seq_idx, bo_sem_idx)
            return

        for batch_idx in range(batch_size):
            bkv = load_bkv(bi_sem_idx, batch_idx, masked)
            bq = load_bq(bi_sem_idx, batch_idx)

            swa_acc, swa_l, swa_m = load_swa_output(bi_sem_idx, batch_idx)

            p, exp_m_diff, l_total = flash_attention_step1_qk_softmax(
                bq,
                bkv,
                kv_lens_ref[batch_start_seq_idx + batch_idx],
                swa_m,
                swa_l,
                attention_sinks,
                masked,
            )

            if prev_p is not None:
                assert prev_bkv is not None
                assert prev_exp_m_diff is not None
                assert prev_l is not None
                out = flash_attention_step2_pv(
                    prev_p,
                    prev_bkv,
                    prev_exp_m_diff,
                    prev_swa_acc,
                    prev_l,
                )

                # Store output from acc to bo.
                bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out

            prev_p = p
            prev_bkv = bkv
            prev_exp_m_diff = exp_m_diff
            prev_l = l_total
            prev_swa_acc = swa_acc

        # end of pipelining loop
        out = flash_attention_step2_pv(
            prev_p, prev_bkv, prev_exp_m_diff, prev_swa_acc, prev_l
        )
        bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out
        start_send_bo_batch(batch_start_seq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq_batch(batch_start_seq_idx, 0)
        start_fetch_swa_batch(batch_start_seq_idx, 0)
        start_fetch_bkv_batch(batch_start_seq_idx, 0)

    # Every row past a token's kv_len is masked out of both the keys and the
    # scores. Long contexts fill all bkv_sz top-k slots, so a batch whose tokens
    # all do skips both masks, which are identities there (~15% of the bundles).
    all_full = kv_lens_ref[batch_start_seq_idx] >= bkv_sz
    for batch_idx in range(1, batch_size):
        all_full = jnp.logical_and(
            all_full, kv_lens_ref[batch_start_seq_idx + batch_idx] >= bkv_sz
        )

    @pl.when(all_full)
    def _process_full():
        process(masked=False)

    gathered = jnp.logical_not(all_full)
    if dense is not None:
        # A `DenseRefs` batch has kv_len < bkv_sz throughout, so never all_full.
        from_cache = is_dense_batch(batch_start_seq_idx)
        gathered = jnp.logical_and(gathered, jnp.logical_not(from_cache))

        @pl.when(from_cache)
        def _process_from_cache():
            process(masked=True, from_cache=True)

    @pl.when(gathered)
    def _process_masked():
        process(masked=True)

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def epilogue():
        wait_send_bo_batch(0)

        @pl.when(pl.num_programs(0) >= 2)
        def _wait_1():
            wait_send_bo_batch(1)

        if swa is not None:
            # Every new kv store of this call's decode tokens.
            num_stores = jnp.minimum(end_seq_idx, num_decode_tokens) - start_seq_idx
            swa.window.wait_stores(jnp.maximum(num_stores, 0), swa.store_sem)

    ### ------- Kernel end ------- ###


def _dense_token_seqs(kv_lens, seq_ids, batch_size: int, topk: int):
    """`DenseRefs.token_seq_ref` for kernel batches of `batch_size` tokens."""
    seqs = seq_ids.reshape(-1, batch_size)
    short = (kv_lens < topk).reshape(-1, batch_size)
    dense = jnp.all(short, axis=1, keepdims=True) & jnp.all(
        seqs == seqs[:, :1], axis=1, keepdims=True
    )
    return jnp.where(dense, -seqs - 1, seqs).reshape(-1).astype(jnp.int32)


def prepare_q_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    _, actual_num_q_heads, actual_head_dim = q.shape
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


def prepare_swa_inputs(
    swa_accumution: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # [max_num_tokens, num_q_heads or num_l_heads]
    swa_m: jax.Array,  # [max_num_tokens, num_q_heads or num_l_heads]
):
    _, actual_num_q_heads, actual_head_dim = swa_accumution.shape
    swa_packing = get_dtype_packing(swa_accumution.dtype)
    num_q_heads = align_to(actual_num_q_heads, swa_packing)
    head_dim = align_to(actual_head_dim, 128)
    swa_accumution = jnp.pad(
        swa_accumution,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    num_l_heads = align_to(num_q_heads, 128)
    # l/m may already come at the kernel's 128 lanes (`lane_padded_lm`); the
    # kernel reads only the first num_q_heads of them.
    swa_l = jnp.pad(
        swa_l,
        (
            (0, 0),
            (0, num_l_heads - swa_l.shape[1]),
        ),
        constant_values=0,
    )
    swa_m = jnp.pad(
        swa_m,
        (
            (0, 0),
            (0, num_l_heads - swa_m.shape[1]),
        ),
        constant_values=0,
    )
    return swa_accumution, swa_l, swa_m


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
    return out[:, :actual_num_q_heads, :actual_head_dim]


# Main Attention kernel for DeepSeek V4 CSA (gather and attention)
# Note that the compressed kv tokens of current batch (current forward pass)
# have been written to the `cache_kv` by the compressor module before calling
# this function, `kv_lens` reflects the length after compressed kv cache write.
@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
        "vmem_limit_bytes",
        "swa_sliding_window",
        "swa_logical_page_size",
        "swa_num_kv_pages_per_block",
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # int32[total_num_pages, page_size, 128]
    cache_kv_rope: jax.Array,  # int32[total_num_pages, page_size // 4, 128]
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    swa_accumution: jax.Array,  # bf16[max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # float32[max_num_tokens, num_q_heads]
    swa_m: jax.Array,  # float32[max_num_tokens, num_q_heads]
    *,
    sm_scale: float = 1.0,
    # Kernel optimization params.
    gather_and_attention_chunk_size: int = 128,
    attention_kernel_batch_size: int = 16,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    # Run the decode tokens' sliding-window pass here (see `SwaDecodeRefs`),
    # from the operands `mla_swa.mla_sliding_window_ragged_paged_attention`
    # returns with `decode_by_caller=True`; swa_accumution / swa_l / swa_m
    # then only hold the other tokens. The result is (output, updated cache).
    swa_decode: mla_swa.SwaDecodeInputs | None = None,
    swa_sliding_window: int | None = None,
    swa_logical_page_size: int | None = None,
    swa_num_kv_pages_per_block: int | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """MLA Ragged paged attention that supports mixed prefill and decode.

    Args:
      q: concatenated all sequences' queries.
      cache_kv_nope: the current kv cache for nope, one row per token (see
        `csa_cache_layout`).
      cache_kv_rope: the current kv cache for rope, 32 words per token.
      topk_indices: for each query token, the indices of the top k key tokens to
        attend to.
      page_indices: flattened page indices look-up table by (seq_id, page_id).
      cu_q_lens: the cumulative sum of the effective query lengths. Similar to
        kv_lens, only the first num_seqs+1 values are valid.
      distribution: (i, j, k) represents that sequences[0:i] are decode-only,
        sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
        k is also the total number of sequences.
      sm_scale: the softmax scale which will be applied to the Q@K^T.
      vmem_limit_bytes: the vmem limit for the pallas kernel.

    Returns:
      The output of attention.
    """
    # The cache is DSV4 FP8 format.
    # nope_cache contains 448 fp8 + 7 fp8 scales,
    # rope_cache contains 64 bf16
    assert cache_kv_nope.dtype == jnp.int32
    assert cache_kv_rope.dtype == jnp.int32

    _, actual_num_q_heads, actual_head_dim = q.shape

    q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
    head_dim = q.shape[-1]
    attention_sinks = jnp.pad(
        attention_sinks,
        (0, q.shape[1] - actual_num_q_heads),
        constant_values=jnp.finfo(attention_sinks.dtype).min,
    )
    assert swa_accumution.dtype == q.dtype
    swa_accumution, swa_l, swa_m = prepare_swa_inputs(swa_accumution, swa_l, swa_m)

    _, page_size, _ = cache_kv_nope.shape

    _, num_q_heads, _ = q.shape
    max_num_seqs = cu_q_lens.shape[0] - 1
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0

    def run_mla_kernel(
        q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
        cache_kv_nope: jax.Array,  # int32[chunk_size, topk, 128]
        cache_kv_rope: jax.Array,  # int32[chunk_size, topk // 4, 128]
        kv_lens: jax.Array,  # i32[max_num_seqs]
        attention_sinks: jax.Array,  # float32[num_q_heads]
        swa_accumution: jax.Array,  # bf16[max_num_tokens, num_q_heads, head_dim]
        swa_l: jax.Array,  # float32[max_num_tokens, num_l_heads]
        swa_m: jax.Array,  # float32[max_num_tokens, num_l_heads]
        start_seq_idx: int,
        end_seq_idx: jax.Array,  # i32
        num_valid_tokens: jax.Array,  # i32
        kernel_batch_size: int,
        swa_cache: jax.Array | None = None,
        dense_inputs: tuple[jax.Array, ...] | None = None,
    ):
        batch_size = kernel_batch_size
        end_seq_idx = jnp.maximum(start_seq_idx, end_seq_idx)
        grid = (cdiv(end_seq_idx - start_seq_idx, batch_size),)
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.VMEM),  # attention_sinks
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_nope
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_rope
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_accumution
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_l
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_m
        ]

        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # o

        # One batch entry's worth of gathered top-k rows, per cache, in the
        # uint8 / bf16 views the kernel bitcasts the int32 words to.
        _, topk_rows, lanes = cache_kv_nope.shape
        bkv_nope_double_buf = pltpu.VMEM(
            (2, batch_size, topk_rows * 4, lanes),
            jnp.uint8,
        )
        _, rope_rows, lanes = cache_kv_rope.shape
        bkv_rope_double_buf = pltpu.VMEM(
            (2, batch_size, rope_rows * 2, lanes),
            jnp.bfloat16,
        )

        bq_double_bufq = pltpu.VMEM(
            (2, batch_size, num_q_heads, head_dim),
            q.dtype,
        )

        bo_double_buf = bq_double_bufq

        num_l_heads = align_to(num_q_heads, 128)
        bl_double_buf = pltpu.VMEM(
            (2, batch_size, num_l_heads),
            jnp.float32,
        )
        bm_double_buf = bl_double_buf

        swa_acc_double_buf = pltpu.VMEM(
            (2, batch_size, num_q_heads, head_dim),
            q.dtype,
        )

        scratch_shapes = [
            bkv_nope_double_buf,
            bkv_rope_double_buf,
            bq_double_bufq,
            bo_double_buf,  # Double buffering for output block.
            bl_double_buf,  # Double buffering for l output.
            bm_double_buf,  # Double buffering for m output.
            swa_acc_double_buf,  # Buffer for swa_accumution.
            # Semaphores for double buffering of
            # bkv_nope, bq, bo, swa_acc, swa_l, swa_m, bkv_rope
            pltpu.SemaphoreType.DMA((7, 2)),
        ]

        seq_bounds = [start_seq_idx, end_seq_idx, num_valid_tokens]
        operands = (
            attention_sinks,
            q,
            cache_kv_nope,
            cache_kv_rope,
            swa_accumution,
            swa_l,
            swa_m,
        )
        out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
        kernel_fn = _attention_kernel
        swa_kwargs = {}
        swa_prefetches = ()
        if swa_cache is not None:
            seq_bounds.append(swa_decode.num_decode_tokens)
            swa_prefetches = (swa_decode.kv_lens, swa_decode.copy_table)
            operands += (swa_decode.new_kv, swa_cache)
            in_specs += [pl.BlockSpec(memory_space=pltpu.HBM)] * 2
            out_specs = [out_specs, pl.BlockSpec(memory_space=pltpu.HBM)]
            out_shape = [
                out_shape,
                jax.ShapeDtypeStruct(swa_cache.shape, swa_cache.dtype),
            ]
            lkv_dim = swa_decode.new_kv.shape[1]
            slots = head_dim * jnp.dtype(q.dtype).itemsize // lkv_dim
            scratch_shapes += [
                pltpu.VMEM(
                    (
                        2,
                        batch_size,
                        swa_num_kv_pages_per_block * swa_logical_page_size,
                        slots,
                        lkv_dim,
                    ),
                    jnp.uint8,
                ),
                # The window fetches, plus one for every new kv store.
                pltpu.SemaphoreType.DMA((2, batch_size + 1)),
            ]
            kernel_fn = _attention_swa_decode_kernel
            swa_kwargs = dict(
                swa_sliding_window=swa_sliding_window,
                swa_logical_page_size=swa_logical_page_size,
                swa_bkv_p=swa_num_kv_pages_per_block,
            )
        dense_prefetches = ()
        if dense_inputs is not None:
            assert swa_cache is None
            token_seq, dense_page_indices, dense_cache_nope, dense_cache_rope = (
                dense_inputs
            )
            dense_prefetches = (token_seq, dense_page_indices)
            operands += (dense_cache_nope, dense_cache_rope)
            in_specs += [pl.BlockSpec(memory_space=pltpu.HBM)] * 2
            scratch_shapes += [
                pltpu.VMEM((topk_rows // 4, lanes), jnp.int32),
                pltpu.VMEM((topk_rows, head_dim), jnp.bfloat16),
            ]
            kernel_fn = _attention_dense_kernel
            swa_kwargs = dict(pages_per_seq=num_page_indices // max_num_seqs)

        scalar_prefetches = (
            kv_lens,
            jnp.array(seq_bounds, jnp.int32),
            # (bi_sem_idx, bo_sem_idx)
            jnp.zeros((2,), jnp.int32),
            # seq_idx_start of the in-flight bo DMA per bo sem; -1 = none.
            jnp.full((2,), -1, jnp.int32),
            *swa_prefetches,
            *dense_prefetches,
        )
        input_output_aliases = {len(scalar_prefetches) + 1: 0}  # q with output
        if swa_cache is not None:
            input_output_aliases[len(scalar_prefetches) + len(operands) - 1] = 1

        gathered_topk = cache_kv_nope.shape[1]
        scope_name = (
            f"SparseMLA-p_{gathered_topk}-bz_{batch_size}-gcz_{cache_kv_nope.shape[0]}"
            f"{'-swa_d' if swa_cache is not None else ''}"
            f"{'-dense' if dense_inputs is not None else ''}"
        )
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    kernel_fn,
                    sm_scale=sm_scale,
                    batch_size=batch_size,
                    **swa_kwargs,
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
        return kernel(*scalar_prefetches, *operands)

    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    # Whole 32-byte DMA granules for the SparseCore's copy of it.
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs),
        tokens_per_seq,
        total_repeat_length=align_to(q.shape[0], 8),
    )
    assert topk_indices is not None
    kv_lens = jnp.sum(topk_indices != -1, axis=-1)
    topk = topk_indices.shape[-1]
    num_chunks = cdiv(q.shape[0], gather_and_attention_chunk_size)
    swa_cache = None if swa_decode is None else swa_decode.cache_kv
    dense_inputs = None
    if (
        swa_cache is None
        and q.shape[0] > DENSE_MIN_TOKENS
        and q.shape[0] % gather_and_attention_chunk_size == 0
        and gather_and_attention_chunk_size % attention_kernel_batch_size == 0
    ):
        token_seq = _dense_token_seqs(
            kv_lens, seq_ids_segment[: q.shape[0]], attention_kernel_batch_size, topk
        )
        dense_inputs = (token_seq, page_indices, cache_kv_nope, cache_kv_rope)

    for i in range(num_chunks):
        start_pos = i * gather_and_attention_chunk_size
        end_pos = min(start_pos + gather_and_attention_chunk_size, q.shape[0])
        chunk_size = end_pos - start_pos

        # Batching
        kernel_batch_size = _largest_divisor(chunk_size, attention_kernel_batch_size)
        assert chunk_size % kernel_batch_size == 0
        # The kernel grid walks [start_pos, batch_end) in `kernel_batch_size`
        # steps, so `batch_end - start_pos` MUST be a multiple of
        # `kernel_batch_size`.
        batch_end = start_pos + (
            cdiv(
                jnp.maximum(
                    0,
                    jnp.minimum(
                        cu_q_lens[distribution[2]],
                        end_pos,
                    )
                    - start_pos,
                ),
                kernel_batch_size,
            )
            * kernel_batch_size
        )
        num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk

        # For prefilling of short sequences (or early in the sequence), there are
        # very few number of KVs in the sequence, so different qs' selected topk
        # would have large overlap. This causes gather read hotspotting. We've seen
        # 30%+ performance degradation compared to the no-duplicate-indices case.
        #
        # TODO: we could consider let the caller (tpu-runner) to sort the sequences
        # based on their lengths. For the sequences-segment below certain length,
        # we use a different kernel (dense attention and mask), for the rest of
        # sequences, we use this gather-and-attention kernel.
        gathered_nope_buffer, gathered_rope_buffer = csa_gather.csa_gather_topk(
            cache_kv_nope,
            cache_kv_rope,
            topk_indices,
            # The SparseMLA kernel reads a `DenseRefs` batch's rows itself.
            seq_ids_segment if dense_inputs is None else dense_inputs[0],
            page_indices,
            num_valid_indices,
            token_start=start_pos,
            num_tokens=chunk_size,
            top_k=topk,
            pages_per_seq=num_page_indices // max_num_seqs,
        )
        gathered_nope_buffer = gathered_nope_buffer.reshape(chunk_size, topk, -1)
        gathered_rope_buffer = gathered_rope_buffer.reshape(chunk_size, topk // 4, -1)

        q = run_mla_kernel(
            q,
            gathered_nope_buffer,
            gathered_rope_buffer,
            kv_lens,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
            start_seq_idx=start_pos,
            end_seq_idx=batch_end,
            num_valid_tokens=cu_q_lens[distribution[2]],
            kernel_batch_size=kernel_batch_size,
            swa_cache=swa_cache,
            dense_inputs=dense_inputs,
        )
        if swa_cache is not None:
            q, swa_cache = q
    out = prepare_outputs(
        q, actual_num_q_heads, actual_head_dim
    )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    if swa_decode is not None:
        return out, swa_cache
    return out
