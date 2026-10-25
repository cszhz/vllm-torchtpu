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

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
# Tokens per grid step for 128-token gather chunks. Each step's prologue and
# epilogue (the first QK and the last PV, which nothing overlaps) are paid
# once per step, so wider steps amortize them over more tokens.
PREFILL_ATTENTION_KERNEL_BATCH_SIZE = 32
# Dense-window prefill path (`_dense_window_kernel`). A gather chunk of at
# least DENSE_MIN_CHUNK tokens from one sequence whose selected rows all lie
# below DENSE_MAX_WINDOW is answered from its copied window instead of a
# per-token gather; decode-sized chunks always take the gather path.
DENSE_MIN_CHUNK = 128
DENSE_MAX_WINDOW = 2048
# Window rows per score tile; a multiple of 128 (lanes) that divides the window.
DENSE_KV_TILE = 256
# Query tokens per grid step: the MXU sees tokens x heads rows.
DENSE_TOKENS_PER_STEP = 16
# Score tiles whose P @ V the MXU accumulates in place before adding it to
# the accumulator.
DENSE_TILE_GROUP = 4
# A step's tile count is padded to a multiple of this, so each step runs one
# fully unrolled body that the scheduler can overlap end to end; padded tiles
# are zero rows that no token selects. Divides DENSE_MAX_WINDOW // DENSE_KV_TILE.
DENSE_TILE_ROUNDING = 2
# Independent token slices of a step whose MXU passes and softmax the
# scheduler interleaves; each slice is a multiple of 8 tokens.
DENSE_ROW_SPLIT = 2


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


_GATHER_PAGE_CHUNK = 128


def _gather_page_ids_kernel(windows_ref, logical_ref, out_ref, *, num_chunks):
    logical = logical_ref[...]  # i32[block_tokens, topk]
    out = jnp.zeros_like(logical)
    for c in range(num_chunks):
        window_chunk = windows_ref[
            :, c * _GATHER_PAGE_CHUNK : (c + 1) * _GATHER_PAGE_CHUNK
        ]  # i32[block_tokens, 128]
        local = logical - c * _GATHER_PAGE_CHUNK
        gathered = jnp.take_along_axis(
            window_chunk, jnp.clip(local, 0, _GATHER_PAGE_CHUNK - 1), axis=1
        )
        out = jnp.where((local >= 0) & (local < _GATHER_PAGE_CHUNK), gathered, out)
    out_ref[...] = out


def gather_page_ids(
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    seq_page_ids: jax.Array,  # i32[num_tokens, topk]  (logical page within seq)
    seq_ids_segment: jax.Array,  # i32[num_tokens]  (token -> seq id)
    max_num_seqs: int,
    *,
    block_tokens: int = 8,
) -> jax.Array:
    """Gathers physical page ids for the CSA top-k tokens."""
    num_tokens, topk = seq_page_ids.shape
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    num_chunks = cdiv(pages_per_seq, _GATHER_PAGE_CHUNK)
    padded_pps = num_chunks * _GATHER_PAGE_CHUNK

    page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
    if padded_pps != pages_per_seq:
        page_table = jnp.pad(page_table, ((0, 0), (0, padded_pps - pages_per_seq)))
    # Per-token page-table window. This is a whole-row gather.
    windows = page_table[seq_ids_segment]  # i32[num_tokens, padded_pps]
    logical = jnp.clip(seq_page_ids, 0, pages_per_seq - 1)

    padded_tokens = align_to(num_tokens, block_tokens)
    if padded_tokens != num_tokens:
        pad = padded_tokens - num_tokens
        windows = jnp.pad(windows, ((0, pad), (0, 0)))
        logical = jnp.pad(logical, ((0, pad), (0, 0)))

    out = pl.pallas_call(
        functools.partial(_gather_page_ids_kernel, num_chunks=num_chunks),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_tokens, padded_pps), lambda t: (t, 0)),
                pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
            ],
            out_specs=pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
            grid=(padded_tokens // block_tokens,),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, topk), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        name="gather_page_ids",
    )(windows, logical)
    return out[:num_tokens]


# DSV4 FP8 nope layout: 448 e4m3 values, then one e8m0 scale per lane of a
# 64-lane period -- 64 quantization groups of 7 dims, group j holding the dims
# {j, j + 64, ... j + 384}. The 64 scale bytes fill the row's padding, so the
# record is still 448 + 64 = 512 bytes.
NOPE_DIM = 448
NOPE_SCALE_PERIOD = 64


def _e8m0_high_bytes_to_bf16(pairs: jax.Array) -> jax.Array:
    """BF16 2^(e - 127) for the e8m0 byte e in the high byte of each int16.

    Exact for e in [1, 254], every scale the compressor writes for finite data
    (amax >= 1e-4); e = 0 gives 0 rather than 2^-127 and e = 255 gives +inf
    rather than NaN. Costs a shift and a mask per word, against the compare and
    select fix-ups of an e8m0 -> bf16 convert.
    """
    bits = lax.shift_right_logical(pairs, 1) & 0x7F807F80
    return pltpu.bitcast(bits, jnp.bfloat16)


def _dequant_dsv4_fp8(words: jax.Array) -> list[jax.Array]:
    """int32 [n, 128] nope words -> the record's 4 sub-rows as BF16 [n, 128].

    Byte b of word (t, l) is record byte 128 b + l of token t (csa_cache_layout).
    A truncating cast keeps the low half / low byte, so each sub-row costs a
    shift and a pack; a (4, 128) -> (1, 512) uint8 reshape relayouts sublanes.
    Sub-row 3 lanes NOPE_DIM % 128: are the scale bytes, not values; the caller
    replaces them.
    """

    def int16_pairs(halves):  # int16 [n, 128] -> int32 [n // 2, 128], same bits
        return pltpu.bitcast(halves, jnp.int32)

    def low_and_high_bytes(halves):
        high = lax.shift_right_logical(int16_pairs(halves), 8)
        return [
            halves.astype(jnp.int8),
            pltpu.bitcast(high, jnp.int16).astype(jnp.int8),
        ]

    low_halves = words.astype(jnp.int16)
    high_halves = lax.shift_right_logical(words, 16).astype(jnp.int16)
    sub_rows = low_and_high_bytes(low_halves) + low_and_high_bytes(high_halves)
    tail = _e8m0_high_bytes_to_bf16(int16_pairs(high_halves))
    lane = lax.broadcasted_iota(jnp.int32, tail.shape, 1)
    # scales[d] = scales[d % 64]
    tile = jnp.where(
        lane < NOPE_SCALE_PERIOD, pltpu.roll(tail, NOPE_SCALE_PERIOD, 1), tail
    )
    return [
        (pltpu.bitcast(r, jnp.float8_e4m3fn).astype(jnp.bfloat16) * tile).astype(
            jnp.bfloat16
        )
        for r in sub_rows
    ]


def _attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    start_end_seq_idx_ref,  # [3] (start_seq_idx, end_seq_idx, num_valid_tokens)
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
    bkv_nope_x2_ref,  # int32[2, batch_size, topk, 128]
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
):
    assert q_hbm_ref.shape == o_hbm_ref.shape

    num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
    assert kv_lens_ref.shape[0] == num_tokens
    # csa_gather emits int32 words; the rope ones are the bf16 rows
    # [.., topk // 2, 128] they encode, a free bitcast of the same bytes.
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

    def flash_attention_step1_qk_softmax(
        q,  # [bq_sz * num_q_heads, head_dim]
        kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_*_x2_ref
        kv_len,  # scalar
        swa_m,  # [bq_sz * num_q_heads],
        swa_l,  # [bq_sz * num_q_heads],
        attention_sinks,  # [num_q_heads]
    ):
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape == (bkv_sz, head_dim)

        # Follow FlashAttention-2 forward pass.
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
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

    def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
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

    def _fetch_swa_batch(seq_idx_start, bq_sem_idx, *, wait=False):
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

    def load_bkv(bkv_sem_idx, batch_idx):
        bkv_nope = _dequant_dsv4_fp8(bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx][...])

        # The gathered rope block is [bkv_sz // 2, 2 * rope_dim]: entry i in
        # lanes 0:rope_dim, entry i + bkv_sz // 2 in the upper half (see
        # csa_gather). The rope dims belong in lanes NOPE_DIM % 128: of the
        # last nope sub-row, where the upper half already is, so only the
        # first half's rows take a lane rotate before the select.
        bkv_rope = bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx][...]
        rope_dim = bkv_rope.shape[-1] // 2
        assert rope_dim == NOPE_DIM % 128
        bkv_rope = jnp.concatenate(
            [pltpu.roll(bkv_rope, rope_dim, 1), bkv_rope], axis=0
        )
        lane = lax.broadcasted_iota(jnp.int32, bkv_rope.shape, 1)
        bkv_rope = jnp.where(lane < rope_dim, bkv_nope[-1], bkv_rope)
        # Rows past kv_len were zeroed in VMEM (zero_rows_past_kv_len).
        return jnp.concatenate([*bkv_nope[:-1], bkv_rope], axis=-1)

    def zero_rows_past_kv_len(bkv_sem_idx):
        """Zeroes the gathered rows at or past each entry's kv_len, in VMEM.

        In vLLM, multiple caches may overlay on the same KV Tensor. For example,
        compressor state cache write data in bfloat16 / float32 format, certain
        byte pattern are interpreted as NaN in FP8, e.g. float8_e8m0fnu byte 0xFF
        decodes to NaN, which would reach the output through P @ V. Zero words
        dequantize to zeros, so this is the [bkv_sz, head_dim] select load_bkv
        would otherwise pay per entry, paid only by batches holding a short entry.
        """
        kv_lens = [kv_lens_ref[batch_start_seq_idx + b] for b in range(batch_size)]
        min_kv_len = functools.reduce(jnp.minimum, kv_lens)

        @pl.when(min_kv_len < bkv_sz)
        def _():
            def zero_entry(b, carry):
                kv_len = kv_lens_ref[batch_start_seq_idx + b]
                nope_ref = bkv_nope_x2_ref.at[bkv_sem_idx, b]
                row = lax.broadcasted_iota(jnp.int32, nope_ref.shape, 0)
                nope_ref[...] = jnp.where(row < kv_len, nope_ref[...], 0)
                # Rope word (t, l) holds entries e and e + 1 in its low and
                # high halves, e = 2 t + bkv_sz // 2 * (l >= rope lanes / 2).
                rope_ref = bkv_rope_x2_ref.at[bkv_sem_idx, b]
                words = pltpu.bitcast(rope_ref[...], jnp.int32)
                lane = lax.broadcasted_iota(jnp.int32, words.shape, 1)
                entry = 2 * lax.broadcasted_iota(jnp.int32, words.shape, 0) + jnp.where(
                    lane < words.shape[1] // 2, 0, bkv_sz // 2
                )
                words = jnp.where(
                    entry + 1 < kv_len,
                    words,
                    jnp.where(entry < kv_len, words & 0xFFFF, 0),
                )
                rope_ref[...] = pltpu.bitcast(words, rope_ref.dtype)
                return carry

            lax.fori_loop(0, batch_size, zero_entry, 0)

    def load_swa_output(bq_sem_idx, batch_idx):
        swa_acc = swa_acc_x2_ref[bq_sem_idx, batch_idx, ...]
        swa_l = bl_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
        swa_m = bm_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
        return swa_acc, swa_l, swa_m

    def process():
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
        zero_rows_past_kv_len(bi_sem_idx)

        @pl.when(pl.program_id(0) >= 2)
        def _wait_send():
            wait_send_bo_batch(bo_sem_idx)

        for batch_idx in range(batch_size):
            bkv = load_bkv(bi_sem_idx, batch_idx)
            bq = load_bq(bi_sem_idx, batch_idx)

            swa_acc, swa_l, swa_m = load_swa_output(bi_sem_idx, batch_idx)

            p, exp_m_diff, l_total = flash_attention_step1_qk_softmax(
                bq,
                bkv,
                kv_lens_ref[batch_start_seq_idx + batch_idx],
                swa_m,
                swa_l,
                attention_sinks,
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

    process()

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def epilogue():
        wait_send_bo_batch(0)

        @pl.when(pl.num_programs(0) >= 2)
        def _wait_1():
            wait_send_bo_batch(1)

    ### ------- Kernel end ------- ###


def _dense_window_position(c: jax.Array) -> jax.Array:
    """Window slot of compressed row ``c`` (``-1`` stays ``-1``).

    A RoPE cache row packs 4 consecutive rows in 32-word lane groups, so each
    `DENSE_KV_TILE`-row tile is stored group-major: row ``c`` goes to quarter
    ``c % 4`` of its tile, at offset ``(c % tile) // 4``. A quarter's RoPE is
    then one lane group of consecutive cache rows, with no relayout.
    """
    quarter = DENSE_KV_TILE // 4
    slot = (c // DENSE_KV_TILE) * DENSE_KV_TILE + (c % 4) * quarter
    slot = slot + (c % DENSE_KV_TILE) // 4
    return jnp.where(c < 0, -1, slot)


def _dense_window_kernel(
    # Prefetch
    blk_window_ref,  # [num_blocks] window rows of an eligible block, else -1
    step_window_ref,  # [num_steps] window rows the step's tokens select from
    step_map_ref,  # [num_steps] token block the step's q/swa/o BlockSpecs use
    blk_pages_ref,  # [num_blocks * max_pages] physical pages of the window
    num_valid_ref,  # [1] tokens past this keep q, as in the gather path
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_ref,  # [tq, num_q_heads, head_dim]
    slots_ref,  # i32[topk, block_tokens] window slot of each selected row
    cache_kv_nope_hbm_ref,  # int32[total_num_pages, page_size, 128]
    cache_kv_rope_hbm_ref,  # int32[total_num_pages, page_size // 4, 128]
    swa_acc_ref,  # [tq, num_q_heads, head_dim]
    swa_l_ref,  # [tq, 1, num_l_heads]
    swa_m_ref,  # [tq, 1, num_l_heads]
    # Output
    o_ref,  # [tq, num_q_heads, head_dim]
    # Scratch
    nope_win_x2_ref,  # int32[2, max_pages * page_size, 128] window per block parity
    rope_win_x2_ref,  # int32[2, max_pages * page_size // 4, 128]
    kv_ref,  # bf16[num_tiles, tile, head_dim] dequantized window, slot order
    bitmap_ref,  # int32[window_cap // 32, block_tokens] selection bit words
    mask_ref,  # int32[num_tiles, block_tokens, tile] 0/1 selection
    s_ref,  # f32[num_tiles, tq * num_q_heads, tile]
    acc_ref,  # f32[tq * num_q_heads, head_dim]
    m_ref,  # f32[tq * num_q_heads, 128] lane-wide running row max
    l_ref,  # f32[tq * num_q_heads, 128] lane-wide running row sum
    sems,  # DMA [2, 2] (block parity, nope/rope)
    *,
    sm_scale: float,
):
    """Attention for a block of one sequence's tokens over its whole window.

    Every query token of the block attends over the same ``W`` compressed
    rows: the block's highest selected row plus one. Instead of gathering
    each token's top-k rows, the window is copied in page-sized DMAs and
    dequantized once per block, and each token's top-k set becomes a 0/1
    score mask: the rows a token did not select get the same minimum score
    the gather path gives its padding entries, so it attends the same rows.
    """
    tq, num_q_heads, head_dim = q_ref.shape
    part_tokens = tq // DENSE_ROW_SPLIT
    part_rows = part_tokens * num_q_heads
    topk, block_tokens = slots_ref.shape
    num_tiles, tile, _ = kv_ref.shape
    quarter = tile // 4
    page_size = cache_kv_nope_hbm_ref.shape[1]
    max_pages = nope_win_x2_ref.shape[1] // page_size
    num_blocks = pl.num_programs(0)
    steps_per_block = pl.num_programs(1)

    blk = pl.program_id(0)
    step_in_blk = pl.program_id(1)
    step = blk * steps_per_block + step_in_blk
    window = blk_window_ref[blk]
    eligible = window >= 0
    nope_win_ref = nope_win_x2_ref.at[blk % 2]
    rope_win_ref = rope_win_x2_ref.at[blk % 2]

    def padded_tiles(rows_used):
        return align_to(jnp.maximum(cdiv(rows_used, tile), 1), DENSE_TILE_ROUNDING)

    def page_copies(b, k):
        page = blk_pages_ref[b * max_pages + k]
        nope = pltpu.make_async_copy(
            cache_kv_nope_hbm_ref.at[page],
            nope_win_x2_ref.at[b % 2, pl.ds(k * page_size, page_size)],
            sems.at[b % 2, 0],
        )
        rope = pltpu.make_async_copy(
            cache_kv_rope_hbm_ref.at[page],
            rope_win_x2_ref.at[b % 2, pl.ds(k * (page_size // 4), page_size // 4)],
            sems.at[b % 2, 1],
        )
        return nope, rope

    def window_copies(b, method):
        def body(k, carry):
            for cp in page_copies(b, k):
                getattr(cp, method)()
            return carry

        lax.fori_loop(0, cdiv(blk_window_ref[b], page_size), body, 0)

    def fetch_window():
        # Block blk - 1 started this block's copies if both are eligible.
        prefetched = (blk > 0) & (blk_window_ref[jnp.maximum(blk - 1, 0)] >= 0)

        @pl.when(jnp.logical_not(prefetched))
        def _():
            window_copies(blk, "start")

        window_copies(blk, "wait")
        # The other buffer's block is done; fill it for the next block.
        nxt = jnp.minimum(blk + 1, num_blocks - 1)

        @pl.when((blk + 1 < num_blocks) & (blk_window_ref[nxt] >= 0))
        def _():
            window_copies(nxt, "start")

    def dequant_window():
        """Window rows -> kv_ref tiles in slot order; rows >= window are 0."""
        lane = lax.broadcasted_iota(jnp.int32, (quarter, 128), 1)
        row = lax.broadcasted_iota(jnp.int32, (quarter, head_dim), 0)
        rope_dim = NOPE_DIM % 128
        words_per_token = csa_cache_layout.ROPE_WORDS

        def tile_body(b, carry):
            rope_words = rope_win_ref[pl.ds(b * quarter, quarter), :]
            for g in range(4):
                words = nope_win_ref[pl.ds(b * tile + g, quarter, stride=4), :]
                nope = _dequant_dsv4_fp8(words)
                # The last sub-row's lanes rope_dim + k hold rope value k, as in
                # the gather path. Word k of quarter g's token (lane 32 g + k)
                # holds value k in its low half and k + 32 in its high half.
                low_lanes = rope_dim - words_per_token * g
                low = pltpu.roll(rope_words, low_lanes % 128, 1)
                high = pltpu.roll(rope_words, (low_lanes + words_per_token) % 128, 1)
                low = pltpu.bitcast(lax.shift_left(low, 16), jnp.float32)
                high = pltpu.bitcast(high & jnp.int32(-65536), jnp.float32)
                rope = jnp.where(lane < rope_dim + words_per_token, low, high)
                last = jnp.where(lane < rope_dim, nope[-1], rope.astype(jnp.bfloat16))
                kv = jnp.concatenate([*nope[:-1], last], axis=-1)
                # NaN bytes can sit past the written rows; P @ V must not see them.
                kv = jnp.where(b * tile + 4 * row + g < window, kv, 0)
                kv_ref[b, pl.ds(g * quarter, quarter), :] = kv
            return carry

        lax.fori_loop(0, padded_tiles(window), tile_body, 0)

    def build_mask():
        """Top-k slots -> mask_ref[b][t, slot] = 1 iff token t selected it."""
        words = bitmap_ref.shape[0]
        word = lax.broadcasted_iota(jnp.int32, (words, block_tokens), 0)

        def select(k, bitmap):
            slot = slots_ref[pl.ds(k, 1), :]
            bit = lax.shift_left(jnp.int32(1), slot & 31)
            hit = word == lax.shift_right_arithmetic(slot, 5)  # -1 matches no word
            return bitmap | jnp.where(hit, bit, 0)

        bitmap_ref[...] = lax.fori_loop(
            0, topk, select, jnp.zeros((words, block_tokens), jnp.int32), unroll=8
        )
        tile_words = tile // 32
        shift = lax.broadcasted_iota(jnp.int32, (tile, block_tokens), 0) & 31

        def tile_body(b, carry):
            w = bitmap_ref[pl.ds(b * tile_words, tile_words), :]
            w = jnp.broadcast_to(w[:, None, :], (tile_words, 32, block_tokens))
            bits = lax.shift_right_logical(w.reshape(tile, block_tokens), shift) & 1
            mask_ref[b] = bits.T
            return carry

        lax.fori_loop(0, padded_tiles(window), tile_body, 0)

    def lane_folds(x):  # [rows, k * 128] -> its k 128-lane slices
        return [x[:, i : i + 128] for i in range(0, x.shape[1], 128)]

    def attend(num_step_tiles):
        """Attention for the step's tokens over `num_step_tiles` tiles.

        The step's DENSE_ROW_SPLIT token slices ("parts") are independent.
        Their phases are issued interleaved, so the scheduler runs one part's
        Q @ K or P @ V on the MXU while another part's softmax runs.
        """
        groups = [
            list(range(g, min(g + DENSE_TILE_GROUP, num_step_tiles)))
            for g in range(0, num_step_tiles, DENSE_TILE_GROUP)
        ]

        def tokens(part):
            return pl.ds(part * part_tokens, part_tokens)

        def rows(part):
            return pl.ds(part * part_rows, part_rows)

        def per_row(part, vec_of_token):  # [num_q_heads] per token -> [part_rows, 1]
            first = part * part_tokens
            vecs = [vec_of_token(first + t)[..., None] for t in range(part_tokens)]
            return jnp.concatenate(vecs, axis=0)

        def score(part):
            """s_ref <- masked scores; m_ref <- their row max, 128 lanes wide."""
            r = rows(part)
            q = q_ref[tokens(part)].reshape(part_rows, head_dim)
            first = step_in_blk * tq + part * part_tokens
            m = jnp.full((part_rows, 128), jnp.finfo(jnp.float32).min)
            for tiles in groups:
                for b in tiles:
                    s = jnp.einsum(
                        "nd,md->nm", q, kv_ref[b], preferred_element_type=jnp.float32
                    )
                    s *= sm_scale
                    mask = mask_ref[b, pl.ds(first, part_tokens), :]
                    mask = jnp.broadcast_to(
                        mask[:, None, :], (part_tokens, num_q_heads, tile)
                    )
                    s = jnp.where(
                        mask.reshape(part_rows, tile) != 0, s, jnp.finfo(s.dtype).min
                    )
                    s_ref[b, r] = s
                    m = functools.reduce(jnp.maximum, lane_folds(s), m)
            m_ref[r] = m

        def accumulate(part):
            """acc_ref/l_ref <- the SWA state plus P @ V; returns the merge terms."""
            r = rows(part)
            swa_m = per_row(part, lambda t: swa_m_ref[t, 0, :num_q_heads])
            m_curr = jnp.maximum(swa_m, jnp.max(m_ref[r], axis=1, keepdims=True))
            exp_m_diff = jnp.exp(swa_m - m_curr)
            m = jnp.broadcast_to(m_curr, (part_rows, 128))
            swa_acc = swa_acc_ref[tokens(part)].reshape(part_rows, head_dim)
            acc_ref[r] = exp_m_diff * swa_acc
            row_sum = jnp.zeros((part_rows, 128), jnp.float32)
            for tiles in groups:
                pv_sum = None
                for b in tiles:
                    s = s_ref[b, r]
                    p = jnp.concatenate([jnp.exp(x - m) for x in lane_folds(s)], axis=1)
                    row_sum = functools.reduce(jnp.add, lane_folds(p), row_sum)
                    pv = jnp.einsum(
                        "nm,md->nd", p, kv_ref[b], preferred_element_type=jnp.float32
                    )
                    pv_sum = pv if pv_sum is None else pv_sum + pv
                acc_ref[r] += pv_sum
            l_ref[r] = row_sum
            return m_curr, exp_m_diff

        def finish(part, m_curr, exp_m_diff):
            r = rows(part)
            swa_l = per_row(part, lambda t: swa_l_ref[t, 0, :num_q_heads])
            sinks = per_row(part, lambda t: attention_sinks_ref[...])
            p_rowsum = jnp.sum(l_ref[r], axis=1, keepdims=True)
            l_total = exp_m_diff * swa_l + p_rowsum + jnp.exp(sinks - m_curr)
            out = (acc_ref[r] * pl.reciprocal(l_total, approx=True)).astype(o_ref.dtype)
            out_shape = (part_tokens, *o_ref.shape[1:])
            token = step * tq + part * part_tokens
            token = token + lax.broadcasted_iota(jnp.int32, out_shape, 0)
            o_ref[tokens(part)] = jnp.where(
                token < num_valid_ref[0], out.reshape(out_shape), q_ref[tokens(part)]
            )

        parts = range(DENSE_ROW_SPLIT)
        for part in parts:
            score(part)
        merges = [accumulate(part) for part in parts]
        for part in parts:
            finish(part, *merges[part])

    @pl.when(eligible & (step_in_blk == 0))
    def _prepare():
        fetch_window()
        dequant_window()
        build_mask()

    step_tiles = padded_tiles(step_window_ref[step])
    for n in range(DENSE_TILE_ROUNDING, num_tiles + 1, DENSE_TILE_ROUNDING):

        @pl.when(eligible & (step_tiles == n))
        def _attend():
            attend(n)

    # No block is eligible: every step maps to block 0, so keep it unchanged.
    @pl.when(jnp.logical_not(eligible) & (step_map_ref[step] == step))
    def _keep():
        o_ref[...] = q_ref[...]


def _dense_window_attention(
    q: jax.Array,  # [num_tokens, num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # int32[total_num_pages, page_size, 128]
    cache_kv_rope: jax.Array,  # int32[total_num_pages, page_size // 4, 128]
    topk_indices: jax.Array,  # i32[num_tokens, topk], -1 past the valid tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    seq_ids: jax.Array,  # i32[num_tokens], -1 past the valid tokens
    attention_sinks: jax.Array,  # float32[num_q_heads]
    swa_accumution: jax.Array,  # [num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # float32[num_tokens, num_l_heads]
    swa_m: jax.Array,  # float32[num_tokens, num_l_heads]
    num_valid_tokens: jax.Array,  # i32
    *,
    max_num_seqs: int,
    block_tokens: int,
    sm_scale: float,
    vmem_limit_bytes: int,
) -> tuple[jax.Array, jax.Array]:
    """Runs the dense-window kernel on every eligible `block_tokens` block.

    A block is eligible when its valid tokens all belong to one sequence and
    select only rows below `DENSE_MAX_WINDOW`. Returns q with those blocks'
    outputs written in place, and the per-block eligibility, so the caller
    sends only the other blocks through the gather path.
    """
    num_tokens, num_q_heads, head_dim = q.shape
    topk = topk_indices.shape[1]
    page_size = cache_kv_nope.shape[1]
    tq = DENSE_TOKENS_PER_STEP
    assert tq % (8 * DENSE_ROW_SPLIT) == 0, (tq, DENSE_ROW_SPLIT)
    assert block_tokens % tq == 0, (tq, block_tokens)
    assert DENSE_MAX_WINDOW % DENSE_KV_TILE == 0 and DENSE_KV_TILE % 128 == 0
    num_tiles = DENSE_MAX_WINDOW // DENSE_KV_TILE
    assert num_tiles % DENSE_TILE_ROUNDING == 0, (num_tiles, DENSE_TILE_ROUNDING)
    num_blocks = num_tokens // block_tokens
    steps_per_block = block_tokens // tq
    num_steps = num_blocks * steps_per_block
    max_pages = cdiv(DENSE_MAX_WINDOW, page_size)

    token_window = jnp.max(topk_indices, axis=1) + 1  # 0 for no selection
    blk_window = jnp.max(token_window.reshape(num_blocks, -1), axis=1)
    blk_seqs = seq_ids.reshape(num_blocks, block_tokens)
    seq_hi = jnp.max(blk_seqs, axis=1)
    seq_lo = jnp.min(
        jnp.where(blk_seqs < 0, jnp.iinfo(jnp.int32).max, blk_seqs), axis=1
    )
    eligible = (seq_hi >= 0) & (seq_lo == seq_hi) & (blk_window <= DENSE_MAX_WINDOW)

    step_window = jnp.max(token_window.reshape(num_steps, tq), axis=1)
    step_ids = jnp.arange(num_steps, dtype=jnp.int32)
    step_eligible = jnp.repeat(eligible, steps_per_block)
    # An ineligible step points its blocks at a neighbouring eligible step's,
    # so the pipeline neither fetches nor writes back its tokens.
    before = lax.cummax(jnp.where(step_eligible, step_ids, -1))
    after = lax.cummin(jnp.where(step_eligible, step_ids, num_steps), reverse=True)
    step_map = jnp.where(
        step_eligible,
        step_ids,
        jnp.where(before >= 0, before, jnp.where(after < num_steps, after, 0)),
    )

    page_table = page_indices.reshape(max_num_seqs, -1)
    if page_table.shape[1] < max_pages:
        page_table = jnp.pad(page_table, ((0, 0), (0, max_pages - page_table.shape[1])))
    blk_pages = page_table[jnp.clip(seq_hi, 0, max_num_seqs - 1), :max_pages]

    slots = _dense_window_position(topk_indices).T  # [topk, num_tokens]
    num_l_heads = swa_l.shape[1]
    swa_l = swa_l.reshape(num_tokens, 1, num_l_heads)
    swa_m = swa_m.reshape(num_tokens, 1, num_l_heads)

    def token_block(blk, step_in_blk, blk_window, step_window, step_map, *_):
        return (step_map[blk * steps_per_block + step_in_blk], 0, 0)

    def slots_block(blk, step_in_blk, blk_window, step_window, step_map, *_):
        return (0, step_map[blk * steps_per_block + step_in_blk] // steps_per_block)

    tokens_spec = pl.BlockSpec((tq, num_q_heads, head_dim), token_block)
    lse_spec = pl.BlockSpec((tq, 1, num_l_heads), token_block)
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM),  # attention_sinks
        tokens_spec,  # q
        pl.BlockSpec((topk, block_tokens), slots_block),
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_nope
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_rope
        tokens_spec,  # swa_accumution
        lse_spec,  # swa_l
        lse_spec,  # swa_m
    ]
    rows = tq * num_q_heads
    scratch_shapes = [
        pltpu.VMEM((2, max_pages * page_size, 128), jnp.int32),
        pltpu.VMEM((2, max_pages * page_size // 4, 128), jnp.int32),
        pltpu.VMEM((num_tiles, DENSE_KV_TILE, head_dim), jnp.bfloat16),
        pltpu.VMEM((DENSE_MAX_WINDOW // 32, block_tokens), jnp.int32),
        pltpu.VMEM((num_tiles, block_tokens, DENSE_KV_TILE), jnp.int32),
        pltpu.VMEM((num_tiles, rows, DENSE_KV_TILE), jnp.float32),
        pltpu.VMEM((rows, head_dim), jnp.float32),
        pltpu.VMEM((rows, 128), jnp.float32),
        pltpu.VMEM((rows, 128), jnp.float32),
        pltpu.SemaphoreType.DMA((2, 2)),
    ]
    scalar_prefetches = (
        jnp.where(eligible, blk_window, -1).astype(jnp.int32),
        step_window.astype(jnp.int32),
        step_map,
        blk_pages.reshape(-1),
        jnp.reshape(num_valid_tokens, (1,)).astype(jnp.int32),
    )
    scope_name = f"SparseMLA-dense-w_{DENSE_MAX_WINDOW}-bt_{block_tokens}-tq_{tq}"
    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(_dense_window_kernel, sm_scale=sm_scale),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=tokens_spec,
                grid=(num_blocks, steps_per_block),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary", "arbitrary"),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
            input_output_aliases={len(scalar_prefetches) + 1: 0},
            name=scope_name,
        )
    )
    q = kernel(
        *scalar_prefetches,
        attention_sinks,
        q,
        slots,
        cache_kv_nope,
        cache_kv_rope,
        swa_accumution,
        swa_l,
        swa_m,
    )
    return q, eligible


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
    swa_l: jax.Array,  # [max_num_tokens, num_q_heads]
    swa_m: jax.Array,  # [max_num_tokens, num_q_heads]
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
    swa_l = jnp.pad(
        swa_l,
        (
            (0, 0),
            (0, num_l_heads - actual_num_q_heads),
        ),
        constant_values=0,
    )
    swa_m = jnp.pad(
        swa_m,
        (
            (0, 0),
            (0, num_l_heads - actual_num_q_heads),
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
) -> jax.Array:
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

        # One batch entry's worth of gathered top-k rows, per cache: nope as
        # its int32 words, rope in the bf16 view the kernel bitcasts it to.
        _, topk_rows, lanes = cache_kv_nope.shape
        bkv_nope_double_buf = pltpu.VMEM(
            (2, batch_size, topk_rows, lanes),
            jnp.int32,
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

        scalar_prefetches = (
            kv_lens,
            jnp.array([start_seq_idx, end_seq_idx, num_valid_tokens], jnp.int32),
            # (bi_sem_idx, bo_sem_idx)
            jnp.zeros((2,), jnp.int32),
            # seq_idx_start of the in-flight bo DMA per bo sem; -1 = none.
            jnp.full((2,), -1, jnp.int32),
        )

        gathered_topk = cache_kv_nope.shape[1]
        scope_name = (
            f"SparseMLA-p_{gathered_topk}-bz_{batch_size}-gcz_{cache_kv_nope.shape[0]}"
        )
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _attention_kernel,
                    sm_scale=sm_scale,
                    batch_size=batch_size,
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
                out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                input_output_aliases={
                    5: 0,  # Alias output activation with q
                },
                name=scope_name,
            )
        )
        return kernel(
            *scalar_prefetches,
            attention_sinks,
            q,
            cache_kv_nope,
            cache_kv_rope,
            swa_accumution,
            swa_l,
            swa_m,
        )

    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs), tokens_per_seq, total_repeat_length=q.shape[0]
    )
    assert topk_indices is not None
    kv_lens = jnp.sum(topk_indices != -1, axis=-1)

    seq_page_ids = topk_indices // page_size
    token_offset = topk_indices % page_size
    topk = topk_indices.shape[-1]
    page_ids = gather_page_ids(
        page_indices, seq_page_ids, seq_ids_segment, max_num_seqs
    )

    # For the "-1" padding elements in topk_indices, we scatter the corresponding
    # page_ids and token_offset to avoid gather memory access hotspotting.
    is_padding = topk_indices == -1
    total_num_pages = cache_kv_nope.shape[0]
    flat_element_index = jnp.arange(q.shape[0] * topk, dtype=jnp.int32).reshape(
        q.shape[0], topk
    )
    # 104729 and 15485863 are randomly chosen large prime numbers.
    scattered_page_ids = (flat_element_index * 104729) % total_num_pages
    scattered_token_offset = (flat_element_index * 15485863) % page_size
    page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
    token_offset = jnp.where(
        is_padding,
        scattered_token_offset,
        token_offset,
    )

    assert page_ids.shape == (q.shape[0], topk)
    gather_indices = (page_ids * page_size + token_offset).reshape(-1)
    num_chunks = cdiv(q.shape[0], gather_and_attention_chunk_size)

    dense_chunks = None
    if (
        gather_and_attention_chunk_size >= DENSE_MIN_CHUNK
        and q.shape[0] % gather_and_attention_chunk_size == 0
    ):
        num_valid_tokens = cu_q_lens[distribution[2]]
        valid = jnp.arange(q.shape[0]) < num_valid_tokens
        q, dense_chunks = _dense_window_attention(
            q,
            cache_kv_nope,
            cache_kv_rope,
            jnp.where(valid[:, None], topk_indices, -1),
            page_indices,
            jnp.where(valid, seq_ids_segment, -1),
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
            num_valid_tokens,
            max_num_seqs=max_num_seqs,
            block_tokens=gather_and_attention_chunk_size,
            sm_scale=sm_scale,
            vmem_limit_bytes=vmem_limit_bytes,
        )

    for i in range(num_chunks):
        start_pos = i * gather_and_attention_chunk_size
        end_pos = min(start_pos + gather_and_attention_chunk_size, q.shape[0])
        chunk_size = end_pos - start_pos
        indices = gather_indices[start_pos * topk : end_pos * topk]

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
        if dense_chunks is not None:
            # The dense-window kernel already answered this chunk: gather nothing.
            batch_end = jnp.where(dense_chunks[i], start_pos, batch_end)
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
        gathered_nope_buffer, gathered_rope_buffer = csa_gather.csa_gather(
            cache_kv_nope,
            cache_kv_rope,
            indices,
            num_valid_indices=num_valid_indices,
            top_k=topk,
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
        )
    return prepare_outputs(
        q, actual_num_q_heads, actual_head_dim
    )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
