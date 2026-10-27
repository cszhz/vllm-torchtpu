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
DENSE_TOKENS_PER_MATMUL = 16
# A `DenseRefs` batch attends over the blocks of this many of its sequence's
# first rows that cover every row its tokens selected,
DENSE_KV_BLOCK = 256
# this many blocks per online-softmax step: each step rescales the f32
# accumulator once.
DENSE_BLOCKS_PER_STEP = 2
# Nonzero: a batch leaves out the blocks none of its tokens selected.
DENSE_SKIP_EMPTY_BLOCKS = 0
# A single-sequence batch whose tokens can select more compressed rows than
# this keeps the per-token SparseCore gather.
DENSE_MAX_ROWS = 1 << 30


class DenseRefs(NamedTuple):
    """Batches of one sequence's tokens. Every top-k entry of such a token is
    a row of the sequence's compressed cache, so the batch copies the
    sequence's first rows once from the paged cache, dequantizes them once,
    and attends each token over all of them under its selection mask from the
    SparseCore (`csa_gather.csa_gather_topk`), which gathers nothing for it.
    The unselected rows get probability exactly 0, so only the order of the
    softmax and PV sums differs from attending over the gathered rows."""

    # SMEM i32[num_tokens]: token t's sequence s, or -(s + 1) when t is in
    # such a batch.
    token_seq_ref: Any
    # SMEM i32[num_tokens // batch_size]: per batch, rows [0, n) of its
    # sequence that cover every entry its tokens selected.
    batch_rows_ref: Any
    # SMEM i32[num_tokens // batch_size * (max_rows // DENSE_KV_BLOCK)]: per
    # batch, in order, the blocks of rows [0, n) it attends, then padding.
    block_list_ref: Any
    num_blocks_ref: Any  # SMEM i32[num_tokens // batch_size], the list's length
    page_indices_ref: Any  # SMEM i32[max_num_seqs * pages_per_seq]
    cache_nope_hbm_ref: Any  # int32[total_pages, page_size, 128]
    cache_rope_hbm_ref: Any  # int32[total_pages, page_size // 4, 128]
    # int32[num_tokens * max_rows // 128, 128], the call's selection masks.
    select_hbm_ref: Any
    nope_x2_ref: Any  # int32[2, max_rows, 128], a batch's copied nope rows
    rope_x2_ref: Any  # int32[2, max_rows // 4, 128], its copied rope rows
    select_x2_ref: Any  # int32[2, batch_size * max_rows // 128, 128]
    # int32[DENSE_KV_BLOCK // 4, 128], a block's rope in csa_gather's layout
    rope_vmem_ref: Any
    kv_vmem_ref: Any  # bf16[max_rows, head_dim], the batch's dequantized rows
    # f32 online-softmax state of a DENSE_TOKENS_PER_MATMUL token group:
    acc_ref: Any  # [group * num_q_heads, head_dim]
    m_ref: Any  # [group * num_q_heads, 128], lane-replicated
    l_ref: Any  # [group * num_q_heads, 128], lane-replicated
    sems: Any  # DMA [3, 2]: nope, rope and mask copies per slot
    pages_per_seq: int


def _attention_listed_kernel(
    kv_lens_ref,
    start_end_seq_idx_ref,
    sem_ids_ref,
    bo_ids_ref,
    batch_list_ref,
    *refs,
    **kwargs,
):
    """`_attention_kernel` over the batches of `batch_list_ref`."""
    _attention_kernel(
        kv_lens_ref,
        start_end_seq_idx_ref,
        sem_ids_ref,
        bo_ids_ref,
        *refs,
        batch_list_ref=batch_list_ref,
        **kwargs,
    )


def _attention_dense_kernel(
    kv_lens_ref,
    start_end_seq_idx_ref,
    sem_ids_ref,
    bo_ids_ref,
    batch_list_ref,
    token_seq_ref,
    batch_rows_ref,
    block_list_ref,
    num_blocks_ref,
    page_indices_ref,
    attention_sinks_ref,
    q_hbm_ref,
    swa_accumution_hbm_ref,
    swa_l_hbm_ref,
    swa_m_hbm_ref,
    cache_nope_hbm_ref,
    cache_rope_hbm_ref,
    select_hbm_ref,
    o_hbm_ref,
    *scratch,
    pages_per_seq: int,
    **kwargs,
):
    """`_attention_kernel` over the `DenseRefs` batches of `batch_list_ref`,
    read from the paged cache: no gathered rows, so no buffers for them."""
    *state, dense_sems = scratch[-4:]
    *scratch, nope_x2, rope_x2, select_x2, rope_vmem, kv_vmem = scratch[:-4]
    _attention_kernel(
        kv_lens_ref,
        start_end_seq_idx_ref,
        sem_ids_ref,
        bo_ids_ref,
        attention_sinks_ref,
        q_hbm_ref,
        None,
        None,
        swa_accumution_hbm_ref,
        swa_l_hbm_ref,
        swa_m_hbm_ref,
        o_hbm_ref,
        None,
        None,
        *scratch,
        batch_list_ref=batch_list_ref,
        dense=DenseRefs(
            token_seq_ref,
            batch_rows_ref,
            block_list_ref,
            num_blocks_ref,
            page_indices_ref,
            cache_nope_hbm_ref,
            cache_rope_hbm_ref,
            select_hbm_ref,
            nope_x2,
            rope_x2,
            select_x2,
            rope_vmem,
            kv_vmem,
            *state,
            dense_sems,
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
    # SMEM i32: token starts of the batches grid steps walk, from entry
    # start_end_seq_idx_ref[3]; None walks start_seq_idx onward.
    batch_list_ref: Any = None,
):
    """With `dense`, every batch of the call is a `DenseRefs` batch and the
    gathered-row operands are None."""
    assert q_hbm_ref.shape == o_hbm_ref.shape

    num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
    assert kv_lens_ref.shape[0] == num_tokens
    # csa_gather emits int32 words; these are the uint8 nope slabs
    # [.., topk * 4, 128] and bf16 rope rows [.., topk // 2, 128] they encode,
    # a free bitcast of the same bytes.
    if dense is None:
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

    def batch_start(step):
        if batch_list_ref is None:
            return start_seq_idx + step * batch_size
        return batch_list_ref[start_end_seq_idx_ref[3] + step]

    batch_start_seq_idx = batch_start(pl.program_id(0))
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1
    is_last_step = pl.program_id(0) == pl.num_programs(0) - 1
    # Tokens below this run the SWA pass here; the rest read its HBM state.
    assert swa is None or batch_list_ref is None
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
        page_size = dense.cache_nope_hbm_ref.shape[1]
        rope_rows_per_page = dense.cache_rope_hbm_ref.shape[1]
        max_rows = dense.kv_vmem_ref.shape[0]
        assert max_rows == dense.pages_per_seq * page_size
        select_words = max_rows // csa_cache_layout.ROW_WORDS
        assert max_rows % DENSE_KV_BLOCK == 0
        assert DENSE_KV_BLOCK % csa_cache_layout.ROW_WORDS == 0

    def dense_batch_rows(seq_idx_start):
        return dense.batch_rows_ref[seq_idx_start // batch_size]

    def starts_dense_run(seq_idx_start):
        """Whether the `DenseRefs` batch must build its rows: the batches of
        one sequence run in a call share `dense_batch_rows`, so the rest of
        the run reuses the first batch's `dense.kv_vmem_ref`."""
        prev = dense.token_seq_ref[jnp.maximum(seq_idx_start - 1, 0)]
        return jnp.logical_or(
            seq_idx_start == batch_start(0), prev != dense.token_seq_ref[seq_idx_start]
        )

    def _fetch_dense_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
        """Copy the batch's selection masks and, if it starts a run, the
        cache pages holding rows [0, `dense_batch_rows`) of its sequence."""
        sem_nope, sem_rope, sem_select = (
            dense.sems.at[i, bkv_sem_idx] for i in range(3)
        )
        local = (seq_idx_start - start_seq_idx) * select_words
        select_dst = dense.select_x2_ref.at[bkv_sem_idx]
        select_src = select_dst
        if not wait:
            select_src = dense.select_hbm_ref.at[
                pl.ds(local, batch_size * select_words)
            ]
        _async_copy(select_src, select_dst, sem_select, wait)

        seq = -dense.token_seq_ref[seq_idx_start] - 1
        num_pages = jnp.where(
            starts_dense_run(seq_idx_start),
            cdiv(dense_batch_rows(seq_idx_start), page_size),
            0,
        )
        for i in range(dense.pages_per_seq):

            @pl.when(i < num_pages)
            def _():
                nope_dst = dense.nope_x2_ref.at[
                    bkv_sem_idx, pl.ds(i * page_size, page_size)
                ]
                rope_dst = dense.rope_x2_ref.at[
                    bkv_sem_idx, pl.ds(i * rope_rows_per_page, rope_rows_per_page)
                ]
                nope_src, rope_src = nope_dst, rope_dst
                if not wait:
                    page = dense.page_indices_ref[seq * dense.pages_per_seq + i]
                    nope_src = dense.cache_nope_hbm_ref.at[page]
                    rope_src = dense.cache_rope_hbm_ref.at[page]
                _async_copy(nope_src, nope_dst, sem_nope, wait)
                _async_copy(rope_src, rope_dst, sem_rope, wait)

    def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
        if dense is None:
            _fetch_gathered_batch(seq_idx_start, bkv_sem_idx, wait=wait)
        else:
            _fetch_dense_batch(seq_idx_start, bkv_sem_idx, wait=wait)

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
        """The bf16 [rows, head_dim] rows of NoPE words and a rope block."""
        bkv_nope = _dequant_dsv4_fp8(words)

        # The gathered rope block is [n, 2 * rope_dim]: entry i in lanes
        # 0:rope_dim, entry i + n in the upper half (see csa_gather). Mosaic
        # has no [m, 2n] -> [2m, n] shape cast, but the lane slice + row
        # concatenate below is one lane rotate per vreg.
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

    def build_dense_block(bkv_sem_idx, num_rows, first_row):
        """Dequantize copied rows [first_row, first_row + DENSE_KV_BLOCK) of a
        `DenseRefs` batch into `dense.kv_vmem_ref`, the same rows `load_bkv`
        builds from a gather, zeroing the rows from `num_rows` on: they may be
        uncopied or decode to NaN, and no token selected them."""
        words = dense.nope_x2_ref[bkv_sem_idx, pl.ds(first_row, DENSE_KV_BLOCK)]
        even, odd = _rope_rows_as_gathered(
            dense.rope_x2_ref[
                bkv_sem_idx,
                pl.ds(pl.multiple_of(first_row // 4, 8), DENSE_KV_BLOCK // 4),
            ]
        )
        half = even.shape[0]
        dense.rope_vmem_ref[pl.ds(0, half, stride=2), :] = even
        dense.rope_vmem_ref[pl.ds(1, half, stride=2), :] = odd
        bkv_rope = dense.rope_vmem_ref.bitcast(jnp.bfloat16)[...]
        bkv = join_bkv(words, bkv_rope)
        k_span = first_row + lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
        dense.kv_vmem_ref[pl.ds(first_row, DENSE_KV_BLOCK)] = jnp.where(
            k_span < num_rows, bkv, 0
        )

    def load_select(bkv_sem_idx, batch_idx, first_row):
        """bool [1, DENSE_KV_BLOCK]: which rows of the block at `first_row`
        token `batch_idx` selected."""
        row_words = csa_cache_layout.ROW_WORDS
        rows = dense.select_x2_ref[
            bkv_sem_idx,
            pl.ds(
                batch_idx * select_words + first_row // row_words,
                DENSE_KV_BLOCK // row_words,
            ),
        ]
        return (
            jnp.concatenate([rows[i : i + 1] for i in range(rows.shape[0])], axis=1)
            != 0
        )

    def attend_from_cache(bi_sem_idx, bo_sem_idx, attention_sinks):
        """`process`'s token loop for a `DenseRefs` batch, over the
        DENSE_KV_BLOCK-row blocks of `dense.block_list_ref`."""
        num_rows = dense_batch_rows(batch_start_seq_idx)

        @pl.when(starts_dense_run(batch_start_seq_idx))
        def _():
            @pl.loop(0, cdiv(num_rows, DENSE_KV_BLOCK))
            def _(j):
                first_row = pl.multiple_of(j * DENSE_KV_BLOCK, DENSE_KV_BLOCK)
                build_dense_block(bi_sem_idx, num_rows, first_row)

        batch = batch_start_seq_idx // batch_size
        num_blocks = dense.num_blocks_ref[batch]
        list_start = batch * (max_rows // DENSE_KV_BLOCK)

        def block_row(i):
            block = dense.block_list_ref[list_start + i]
            return pl.multiple_of(block * DENSE_KV_BLOCK, DENSE_KV_BLOCK)

        per_step = DENSE_BLOCKS_PER_STEP
        num_steps = num_blocks // per_step
        group = min(DENSE_TOKENS_PER_MATMUL, batch_size)
        assert batch_size % group == 0
        for first in range(0, batch_size, group):
            tokens = range(first, first + group)
            init_dense_state(bi_sem_idx, tokens)

            @pl.loop(0, num_steps)
            def _(i):
                rows = [block_row(i * per_step + k) for k in range(per_step)]
                attend_blocks_from_cache(bi_sem_idx, tokens, rows)

            for tail in range(1, per_step):

                @pl.when(num_blocks - num_steps * per_step == tail)
                def _(tail=tail):
                    rows = [block_row(num_steps * per_step + k) for k in range(tail)]
                    attend_blocks_from_cache(bi_sem_idx, tokens, rows)

            out = finish_dense_state(attention_sinks, group)
            for i, b in enumerate(tokens):
                bo_x2_ref.at[bo_sem_idx, b][...] = out[
                    i * num_q_heads : (i + 1) * num_q_heads
                ]

    def init_dense_state(bi_sem_idx, tokens):
        """Start the online softmax of `tokens` from their SWA state."""
        states = [load_swa_output(bi_sem_idx, b) for b in tokens]
        swa_acc, swa_l, swa_m = (
            jnp.concatenate([state[i] for state in states], axis=0)
            for i in range(3)
        )
        dense.acc_ref[...] = swa_acc.astype(jnp.float32)
        dense.l_ref[...] = jnp.broadcast_to(swa_l, dense.l_ref.shape)
        dense.m_ref[...] = jnp.broadcast_to(swa_m, dense.m_ref.shape)

    def attend_blocks_from_cache(bi_sem_idx, tokens, first_rows):
        """One online-softmax step of `tokens` over the blocks at
        `first_rows`: the tokens share their rows, so each block runs one QK
        and one PV matmul for all of them."""
        bkvs = [dense.kv_vmem_ref[pl.ds(r, DENSE_KV_BLOCK)] for r in first_rows]
        q = jnp.concatenate([load_bq(bi_sem_idx, b) for b in tokens], axis=0)
        s = jnp.concatenate(
            [
                jnp.einsum("nd,md->nm", q, bkv, preferred_element_type=jnp.float32)
                for bkv in bkvs
            ],
            axis=1,
        )
        s *= sm_scale
        width = s.shape[1]
        select = jnp.concatenate(
            [
                jnp.broadcast_to(
                    jnp.concatenate(
                        [load_select(bi_sem_idx, b, r) for r in first_rows], axis=1
                    ),
                    (num_q_heads, width),
                )
                for b in tokens
            ],
            axis=0,
        )
        s = jnp.where(select, s, jnp.finfo(s.dtype).min)
        m_prev = dense.m_ref[...]
        m_curr = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
        lanes = m_curr.shape[1]
        p = jnp.exp(s - jnp.tile(m_curr, (1, width // lanes)))
        alpha = jnp.exp(m_prev - m_curr)
        dense.l_ref[...] = alpha * dense.l_ref[...] + jnp.sum(
            p, axis=1, keepdims=True
        )
        dense.m_ref[...] = m_curr
        pv = sum(
            jnp.einsum(
                "nm,md->nd",
                p[:, k * DENSE_KV_BLOCK : (k + 1) * DENSE_KV_BLOCK],
                bkv,
                preferred_element_type=jnp.float32,
            )
            for k, bkv in enumerate(bkvs)
        )
        dense.acc_ref[...] = (
            jnp.tile(alpha, (1, head_dim // lanes)) * dense.acc_ref[...] + pv
        )

    def finish_dense_state(attention_sinks, group):
        """The tokens' normalized output, the sinks added to the denominator."""
        m = dense.m_ref[:, :1]
        l_total = dense.l_ref[:, :1] + jnp.exp(
            jnp.concatenate([attention_sinks] * group, axis=0) - m
        )
        acc = dense.acc_ref[...]
        if q_dtype == jnp.float32:
            return lax.div(acc, l_total)
        return (acc * pl.reciprocal(l_total, approx=True)).astype(q_dtype)

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
        bi_sem_idx = sem_ids_ref[0]
        next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)

        # Prefetch next seq
        @pl.when(jnp.logical_not(is_last_step))
        def prefetch_next_seq():
            next_seq_idx = batch_start(pl.program_id(0) + 1)
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

    @pl.when(pl.program_id(0) == 0)
    def prologue():
        start_fetch_bq_batch(batch_start_seq_idx, 0)
        start_fetch_swa_batch(batch_start_seq_idx, 0)
        start_fetch_bkv_batch(batch_start_seq_idx, 0)

    if dense is not None:
        process(masked=True, from_cache=True)
    else:
        # Every row past a token's kv_len is masked out of both the keys and
        # the scores. Long contexts fill all bkv_sz top-k slots, so a batch
        # whose tokens all do skips both masks, which are identities there
        # (~15% of the bundles).
        all_full = kv_lens_ref[batch_start_seq_idx] >= bkv_sz
        for batch_idx in range(1, batch_size):
            all_full = jnp.logical_and(
                all_full, kv_lens_ref[batch_start_seq_idx + batch_idx] >= bkv_sz
            )

        @pl.when(all_full)
        def _process_full():
            process(masked=False)

        @pl.when(jnp.logical_not(all_full))
        def _process_masked():
            process(masked=True)

    @pl.when(is_last_step)
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


def _dense_batches(
    topk_indices, seq_ids, num_valid_tokens, batch_size: int, chunk_size: int
):
    """`DenseRefs.token_seq_ref` and `DenseRefs.batch_rows_ref` for kernel
    batches of `batch_size` tokens in calls of `chunk_size` tokens, and each
    batch's own rows, the ones its tokens selected an entry in.

    A batch of valid tokens of one sequence reads the cache when its rows fit
    in DENSE_MAX_ROWS. Its rows cover every entry selected by a batch of the
    same sequence and call, so consecutive ones can share one build.
    """
    num_tokens = topk_indices.shape[0]
    num_batches = num_tokens // batch_size
    seqs = seq_ids.reshape(num_batches, batch_size)
    valid = (jnp.arange(num_tokens) < num_valid_tokens).reshape(seqs.shape)
    single = jnp.all(valid, axis=1) & jnp.all(seqs == seqs[:, :1], axis=1)
    own_rows = (jnp.max(topk_indices, axis=1) + 1).reshape(seqs.shape).max(axis=1)
    batch = jnp.arange(num_batches)
    call = batch // (chunk_size // batch_size)
    same = (
        (seqs[:, :1] == seqs[:, 0]) & (call[:, None] == call) & single[:, None] & single
    )
    rows = jnp.max(jnp.where(same, own_rows, 0), axis=1)
    dense = single & (rows > 0) & (rows <= DENSE_MAX_ROWS)
    token_seq = jnp.where(dense[:, None], -seqs - 1, seqs).reshape(-1)
    return (
        token_seq.astype(jnp.int32),
        rows.astype(jnp.int32),
        own_rows.astype(jnp.int32),
    )


def _dense_blocks(topk_indices, batch_rows, batch_size: int, max_rows: int):
    """`DenseRefs.block_list_ref` and `DenseRefs.num_blocks_ref`: each batch
    attends the DENSE_KV_BLOCK-row blocks of its own `batch_rows` that one of
    its tokens selected an entry in, or all of them without
    DENSE_SKIP_EMPTY_BLOCKS. Leaving out a block nobody selected is exact: it
    would add probability 0 to every token."""
    num_blocks = max_rows // DENSE_KV_BLOCK
    block = jnp.arange(num_blocks)
    attend = block < cdiv(batch_rows, DENSE_KV_BLOCK)[:, None]
    if DENSE_SKIP_EMPTY_BLOCKS:
        # Per batch, the OR over its entries of one bit per selected block.
        assert DENSE_KV_BLOCK & (DENSE_KV_BLOCK - 1) == 0
        entry_block = lax.shift_right_logical(
            topk_indices, DENSE_KV_BLOCK.bit_length() - 1
        )
        bit = jnp.left_shift(jnp.uint32(1), (entry_block & 31).astype(jnp.uint32))
        words = []
        for w in range(cdiv(num_blocks, 32)):
            hit = (topk_indices >= 0) & (lax.shift_right_logical(entry_block, 5) == w)
            words.append(
                lax.reduce(
                    jnp.where(hit, bit, jnp.uint32(0)).reshape(
                        -1, batch_size, topk_indices.shape[1]
                    ),
                    jnp.uint32(0),
                    lax.bitwise_or,
                    (1, 2),
                )
            )
        bits = jnp.stack(words, axis=1)[:, block // 32]
        selected = (bits >> (block % 32).astype(jnp.uint32)) & 1
        attend &= selected != 0
    block_list = jnp.sort(jnp.where(attend, block, num_blocks), axis=1)
    return (
        block_list.reshape(-1).astype(jnp.int32),
        jnp.sum(attend, axis=1).astype(jnp.int32),
    )


def _batch_starts(flags, batch_size: int):
    """Token starts of the flagged kernel batches, in order, then padding."""
    (batches,) = jnp.nonzero(flags, size=flags.shape[0], fill_value=0)
    return (batches * batch_size).astype(jnp.int32)


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
        # int32[chunk_size, topk, 128] / [chunk_size, topk // 4, 128], or
        # None with `dense_inputs`.
        cache_kv_nope: jax.Array | None,
        cache_kv_rope: jax.Array | None,
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
        # (token starts, first entry, count): the grid walks those batches.
        batch_list: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    ):
        batch_size = kernel_batch_size
        end_seq_idx = jnp.maximum(start_seq_idx, end_seq_idx)
        grid = (cdiv(end_seq_idx - start_seq_idx, batch_size),)
        if batch_list is not None:
            grid = (batch_list[2],)
        gathered = dense_inputs is None
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.VMEM),  # attention_sinks
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            *[pl.BlockSpec(memory_space=pltpu.HBM)] * (2 if gathered else 0),
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_accumution
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_l
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_m
        ]

        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # o

        lanes = csa_cache_layout.ROW_WORDS
        bkv_double_bufs = []
        if gathered:
            # One batch entry's worth of gathered top-k rows, per cache, in the
            # uint8 / bf16 views the kernel bitcasts the int32 words to.
            _, topk_rows, _ = cache_kv_nope.shape
            _, rope_rows, _ = cache_kv_rope.shape
            bkv_double_bufs = [
                pltpu.VMEM((2, batch_size, topk_rows * 4, lanes), jnp.uint8),
                pltpu.VMEM((2, batch_size, rope_rows * 2, lanes), jnp.bfloat16),
            ]

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
            *bkv_double_bufs,
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
        list_prefetches = ()
        if batch_list is not None:
            seq_bounds.append(batch_list[1])
            list_prefetches = (batch_list[0],)
        operands = (
            attention_sinks,
            q,
            *((cache_kv_nope, cache_kv_rope) if gathered else ()),
            swa_accumution,
            swa_l,
            swa_m,
        )
        out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
        kernel_fn = _attention_kernel
        if batch_list is not None:
            kernel_fn = _attention_listed_kernel
        swa_kwargs = {}
        swa_prefetches = ()
        if swa_cache is not None:
            assert batch_list is None
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
            assert swa_cache is None and batch_list is not None
            (
                token_seq,
                batch_rows,
                block_list,
                num_blocks,
                dense_page_indices,
                dense_cache_nope,
                dense_cache_rope,
                select,
            ) = dense_inputs
            dense_prefetches = (
                token_seq,
                batch_rows,
                block_list,
                num_blocks,
                dense_page_indices,
            )
            operands += (dense_cache_nope, dense_cache_rope, select)
            in_specs += [pl.BlockSpec(memory_space=pltpu.HBM)] * 3
            pages_per_seq = num_page_indices // max_num_seqs
            max_rows = pages_per_seq * dense_cache_nope.shape[1]
            select_rows = batch_size * max_rows // lanes
            group_rows = min(DENSE_TOKENS_PER_MATMUL, batch_size) * num_q_heads
            scratch_shapes += [
                pltpu.VMEM((2, max_rows, lanes), jnp.int32),
                pltpu.VMEM((2, max_rows // 4, lanes), jnp.int32),
                pltpu.VMEM((2, select_rows, lanes), jnp.int32),
                pltpu.VMEM((DENSE_KV_BLOCK // 4, lanes), jnp.int32),
                pltpu.VMEM((max_rows, head_dim), jnp.bfloat16),
                pltpu.VMEM((group_rows, head_dim), jnp.float32),
                pltpu.VMEM((group_rows, lanes), jnp.float32),
                pltpu.VMEM((group_rows, lanes), jnp.float32),
                pltpu.SemaphoreType.DMA((3, 2)),
            ]
            kernel_fn = _attention_dense_kernel
            swa_kwargs = dict(pages_per_seq=pages_per_seq)

        scalar_prefetches = (
            kv_lens,
            jnp.array(seq_bounds, jnp.int32),
            # (bi_sem_idx, bo_sem_idx)
            jnp.zeros((2,), jnp.int32),
            # seq_idx_start of the in-flight bo DMA per bo sem; -1 = none.
            jnp.full((2,), -1, jnp.int32),
            *list_prefetches,
            *swa_prefetches,
            *dense_prefetches,
        )
        input_output_aliases = {len(scalar_prefetches) + 1: 0}  # q with output
        if swa_cache is not None:
            input_output_aliases[len(scalar_prefetches) + len(operands) - 1] = 1

        if gathered:
            chunk, gathered_topk = cache_kv_nope.shape[:2]
        else:
            chunk, gathered_topk = q.shape[0], topk
        scope_name = (
            f"SparseMLA-p_{gathered_topk}-bz_{batch_size}-gcz_{chunk}"
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
    pages_per_seq = num_page_indices // max_num_seqs
    dense_inputs = None
    if (
        swa_cache is None
        and q.shape[0] > DENSE_MIN_TOKENS
        and q.shape[0] % gather_and_attention_chunk_size == 0
        and gather_and_attention_chunk_size % attention_kernel_batch_size == 0
    ):
        token_seq, batch_rows, own_rows = _dense_batches(
            topk_indices,
            seq_ids_segment[: q.shape[0]],
            cu_q_lens[distribution[2]],
            attention_kernel_batch_size,
            q.shape[0],
        )
        block_list, num_blocks = _dense_blocks(
            topk_indices,
            own_rows,
            attention_kernel_batch_size,
            pages_per_seq * page_size,
        )
        dense_inputs = (
            token_seq,
            batch_rows,
            block_list,
            num_blocks,
            page_indices,
            cache_kv_nope,
            cache_kv_rope,
        )

    chunks = range(num_chunks)
    if dense_inputs is not None:
        # One SparseCore launch writes every cache-read token's selection
        # mask and one kernel launch attends all their batches, so a sequence's
        # rows are copied and dequantized once per call, not once per chunk.
        select = csa_gather.csa_gather_topk(
            cache_kv_nope,
            cache_kv_rope,
            topk_indices,
            token_seq,
            page_indices,
            0,
            token_start=0,
            num_tokens=q.shape[0],
            top_k=topk,
            pages_per_seq=pages_per_seq,
            select_rows=pages_per_seq * page_size,
            gather=False,
        )[2]
        num_valid_tokens = cu_q_lens[distribution[2]]
        batch_size = attention_kernel_batch_size
        from_cache = token_seq[::batch_size] < 0
        q = run_mla_kernel(
            q,
            None,
            None,
            kv_lens,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
            start_seq_idx=0,
            end_seq_idx=q.shape[0],
            num_valid_tokens=num_valid_tokens,
            kernel_batch_size=batch_size,
            dense_inputs=dense_inputs + (select,),
            batch_list=(
                _batch_starts(from_cache, batch_size),
                0,
                jnp.sum(from_cache, dtype=jnp.int32),
            ),
        )
        # The other batches (decode tokens, sequence boundaries, padding) get
        # one SparseCore gather and one launch: the gather runs under the
        # launch above, which a call of this size keeps long since its decode
        # tokens are bounded by the number of sequences.
        gathered_batches = jnp.logical_not(from_cache) & (
            jnp.arange(from_cache.shape[0]) * batch_size < num_valid_tokens
        )
        gathered = csa_gather.csa_gather_topk(
            cache_kv_nope,
            cache_kv_rope,
            topk_indices,
            token_seq,
            page_indices,
            align_to(num_valid_tokens, batch_size) * topk,
            token_start=0,
            num_tokens=q.shape[0],
            top_k=topk,
            pages_per_seq=pages_per_seq,
        )
        q = run_mla_kernel(
            q,
            gathered[0].reshape(q.shape[0], topk, -1),
            gathered[1].reshape(q.shape[0], topk // 4, -1),
            kv_lens,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
            start_seq_idx=0,
            end_seq_idx=q.shape[0],
            num_valid_tokens=num_valid_tokens,
            kernel_batch_size=batch_size,
            batch_list=(
                _batch_starts(gathered_batches, batch_size),
                0,
                jnp.sum(gathered_batches, dtype=jnp.int32),
            ),
        )
        chunks = ()

    for i in chunks:
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
        gathered = csa_gather.csa_gather_topk(
            cache_kv_nope,
            cache_kv_rope,
            topk_indices,
            seq_ids_segment,
            page_indices,
            num_valid_indices,
            token_start=start_pos,
            num_tokens=chunk_size,
            top_k=topk,
            pages_per_seq=pages_per_seq,
        )
        gathered_nope_buffer, gathered_rope_buffer = gathered
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
        )
        if swa_cache is not None:
            q, swa_cache = q
    out = prepare_outputs(
        q, actual_num_q_heads, actual_head_dim
    )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    if swa_decode is not None:
        return out, swa_cache
    return out
