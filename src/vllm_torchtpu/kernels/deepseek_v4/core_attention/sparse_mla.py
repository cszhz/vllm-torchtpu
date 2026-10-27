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


def _dequant_dsv4_fp8(bkv_nope: jax.Array):
    """Dequantize FP8 values to BF16."""
    nope_fp8 = pltpu.bitcast(bkv_nope[:, :NOPE_DIM], jnp.float8_e4m3fn).astype(
        jnp.bfloat16
    )
    # scales[d] = scales[d % 64]
    tail = pltpu.bitcast(bkv_nope[:, -128:], jnp.float8_e8m0fnu).astype(jnp.bfloat16)
    lane = lax.broadcasted_iota(jnp.int32, tail.shape, 1)
    tile = jnp.where(
        lane < NOPE_SCALE_PERIOD, pltpu.roll(tail, NOPE_SCALE_PERIOD, 1), tail
    )
    nope_scales = jnp.concatenate([tile] * 4, axis=1)[:, :NOPE_DIM]
    return (nope_fp8 * nope_scales).astype(jnp.bfloat16)


def _dequant_cache_page(nope_words: jax.Array, rope_words: jax.Array):
    """One cache page's int32 NoPE / RoPE words (`csa_cache_layout`) as bf16
    KV rows `[page_size, NOPE_DIM + rope_dim]`, in token order."""
    page_size = nope_words.shape[0]
    nope = pltpu.bitcast(nope_words, jnp.uint8).reshape(page_size, -1)
    nope = _dequant_dsv4_fp8(nope)

    # Row r of the RoPE words holds tokens 4r + j in lanes [32 j, 32 j + 32),
    # word k carrying bf16 values k (low half) and k + 32 (high half).
    words = csa_cache_layout.ROPE_WORDS
    lo = lax.bitcast_convert_type(jnp.left_shift(rope_words, 16), jnp.float32)
    hi = lax.bitcast_convert_type(
        jnp.bitwise_and(rope_words, jnp.int32(-65536)), jnp.float32
    )
    groups = csa_cache_layout.ROPE_TOKENS_PER_ROW
    rope = jnp.concatenate(
        [
            jnp.concatenate(
                [
                    lo[:, j * words : (j + 1) * words],
                    hi[:, j * words : (j + 1) * words],
                ],
                axis=1,
            )
            for j in range(groups)
        ],
        axis=0,
    ).astype(jnp.bfloat16)  # row j * rows + r is token groups * r + j
    # Back to token order with an exact one-hot matmul (one nonzero per row).
    rows = rope_words.shape[0]
    token = lax.broadcasted_iota(jnp.int32, (page_size, page_size), 0)
    src = lax.broadcasted_iota(jnp.int32, (page_size, page_size), 1)
    perm = (src == (token % groups) * rows + token // groups).astype(jnp.bfloat16)
    rope = jnp.dot(perm, rope, preferred_element_type=jnp.float32)
    return jnp.concatenate([nope, rope.astype(jnp.bfloat16)], axis=1)


# Smallest padded batch that compiles the shared-KV path: decode-only batches
# (at most max-num-seqs tokens) skip its per-chunk launch.
SHARED_KV_MIN_TOKENS = 512
# Query tokens stacked into one matmul on the shared-KV path: 4 x 64 heads
# fill the MXU's 256 rows.
SHARED_KV_TOKENS_PER_MATMUL = 4


def _selects_leading_rows(topk_indices: jax.Array, kv_lens: jax.Array) -> jax.Array:
    """Whether each token's top-k is exactly {0, ..., kv_len - 1}, in any
    order, then -1 padding: the top-k of a token with at most k candidates.

    Attention over those gathered rows is attention over the first kv_len rows
    of the token's cache. Duplicates fail the test, so they keep the gather.
    """
    slot = lax.broadcasted_iota(jnp.int32, topk_indices.shape, 1)
    in_prefix = slot < kv_lens[:, None]
    in_range = (topk_indices >= 0) & (topk_indices < kv_lens[:, None])
    well_formed = jnp.all(jnp.where(in_prefix, in_range, topk_indices == -1), axis=-1)
    # kv_len values in [0, kv_len) are distinct iff their bitset has kv_len
    # bits set.
    topk = topk_indices.shape[-1]
    word = lax.broadcasted_iota(jnp.int32, (1, cdiv(topk, 32), 1), 1)
    bits = jnp.where(
        in_prefix[:, None, :] & (topk_indices[:, None, :] >> 5 == word),
        jnp.left_shift(jnp.int32(1), topk_indices[:, None, :] & 31),
        0,
    )
    bitset = lax.reduce(bits, jnp.int32(0), lax.bitwise_or, (2,))
    distinct = jnp.sum(lax.population_count(bitset), axis=-1) == kv_lens
    return well_formed & distinct


def _attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    # [3] (start_seq_idx, end_seq_idx, num_valid_tokens); with `shared_kv`,
    # then (kv_rows, page_0, page_1, ...): the sequence's first pages and how
    # many of their rows any query token of the launch reads.
    start_end_seq_idx_ref,
    sem_ids_ref,  # [2] (bi_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [2] seq_idx_start of the in-flight bo DMA per bo sem
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # int32[chunk_size, topk, 128], or with `shared_kv` the whole cache
    # int32[total_num_pages, page_size, 128].
    cache_kv_nope_hbm_ref,
    # int32[chunk_size, topk // 4, 128], or with `shared_kv` the whole cache
    # int32[total_num_pages, page_size // 4, 128].
    cache_kv_rope_hbm_ref,
    swa_accumution_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    swa_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    swa_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # Scratch
    # [2, batch_size, topk * 4, 128] u8; with `shared_kv` the cache pages
    # int32[shared_pages, page_size, 128].
    bkv_nope_x2_ref,
    # [2, batch_size, topk, 128] bf16; with `shared_kv` the cache pages
    # int32[shared_pages, page_size // 4, 128].
    bkv_rope_x2_ref,
    bq_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bo_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bl_x2_ref,  # [2, batch_size, num_l_heads]
    bm_x2_ref,  # [2, batch_size, num_l_heads]
    swa_acc_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    sems,  # [7, 2]
    # bf16[shared_pages * page_size, head_dim], with `shared_kv` only.
    shared_bkv_ref=None,
    *,
    sm_scale: float,
    batch_size: int = 1,
    shared_kv: bool = False,
):
    """With `shared_kv`, every query token of the launch belongs to one
    sequence and attends over the leading rows of its cache, read and
    dequantized once per launch, instead of over its own gathered top-k rows;
    `kv_lens` then counts those leading rows per token."""
    assert q_hbm_ref.shape == o_hbm_ref.shape

    num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
    assert kv_lens_ref.shape[0] == num_tokens
    if shared_kv:
        bkv_sz = shared_bkv_ref.shape[0]
    else:
        # csa_gather emits int32 words; these are the uint8 nope slabs
        # [.., topk * 4, 128] and bf16 rope rows [.., topk // 2, 128] they
        # encode, a free bitcast of the same bytes.
        cache_kv_nope_hbm_ref = cache_kv_nope_hbm_ref.bitcast(jnp.uint8)
        cache_kv_rope_hbm_ref = cache_kv_rope_hbm_ref.bitcast(jnp.bfloat16)
        # The gathered rope buffer carries two top-k entries per row
        # (csa_gather).
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
        if shared_kv:
            return  # Loaded once, by `load_shared_kv`.
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

    def load_shared_kv():
        num_pages, page_size, _ = bkv_nope_x2_ref.shape
        kv_rows = start_end_seq_idx_ref[3]

        def page_copies(p):
            page = start_end_seq_idx_ref[4 + p]
            return (
                pltpu.make_async_copy(
                    cache_kv_nope_hbm_ref.at[page], bkv_nope_x2_ref.at[p], sems.at[0, p]
                ),
                pltpu.make_async_copy(
                    cache_kv_rope_hbm_ref.at[page], bkv_rope_x2_ref.at[p], sems.at[6, p]
                ),
            )

        # Pages past `kv_rows` may be unallocated; they are not read, and
        # their rows are zeroed below.
        for p in range(num_pages):

            @pl.when(p * page_size < kv_rows)
            def _():
                for cp in page_copies(p):
                    cp.start()

        for p in range(num_pages):

            @pl.when(p * page_size < kv_rows)
            def _():
                for cp in page_copies(p):
                    cp.wait()

            # Rows no token reads may hold other caches' bytes (see load_bkv),
            # which can decode to NaN. RoPE rows are zeroed before the
            # re-ordering matmul mixes them into the rows that are read.
            rope_words = bkv_rope_x2_ref[p]
            rope_token = (
                p * page_size
                + csa_cache_layout.ROPE_TOKENS_PER_ROW
                * lax.broadcasted_iota(jnp.int32, rope_words.shape, 0)
                + lax.broadcasted_iota(jnp.int32, rope_words.shape, 1)
                // csa_cache_layout.ROPE_WORDS
            )
            rope_words = jnp.where(rope_token < kv_rows, rope_words, 0)
            bkv = _dequant_cache_page(bkv_nope_x2_ref[p], rope_words)
            row = p * page_size + lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
            bkv = jnp.where(row < kv_rows, bkv, 0)
            shared_bkv_ref[pl.ds(p * page_size, page_size)] = bkv.astype(
                shared_bkv_ref.dtype
            )

    def load_bkv(bkv_sem_idx, batch_idx):
        bkv_nope = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx][...]
        # The gather nope is (4, 128) u8, reshape it to (1, 512) u8.
        bkv_nope = bkv_nope.reshape(bkv_sz, -1)
        bkv_nope = _dequant_dsv4_fp8(bkv_nope)

        # The gathered rope block is [bkv_sz // 2, 2 * rope_dim]: entry i in
        # lanes 0:rope_dim, entry i + bkv_sz // 2 in the upper half (see
        # csa_gather). Mosaic has no [m, 2n] -> [2m, n] shape cast, but the
        # lane slice + row concatenate below is one lane rotate per vreg.
        bkv_rope = bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx][...]
        bkv_rope = bkv_rope.reshape(bkv_sz // 2, -1)
        rope_dim = bkv_rope.shape[-1] // 2
        bkv_rope = jnp.concatenate(
            [bkv_rope[:, :rope_dim], bkv_rope[:, rope_dim:]], axis=0
        )
        bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)

        # In vLLM, multiple caches may overlay on the same KV Tensor. For example,
        # compressor state cache write data in bfloat16 / float32 format, certain
        # byte pattern are interpreted as NaN in FP8, e.g. float8_e8m0fnu byte 0xFF
        # decodes to NaN.
        # We need to mask out the data by the actual kv_len to avoid NaN propagting
        # to the downstream computation.
        kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
        k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
        bkv = jnp.where(k_span < kv_len, bkv, 0)
        return bkv

    def load_swa_output(bq_sem_idx, batch_idx):
        swa_acc = swa_acc_x2_ref[bq_sem_idx, batch_idx, ...]
        swa_l = bl_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
        swa_m = bm_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
        return swa_acc, swa_l, swa_m

    def process_shared_kv_batch(bi_sem_idx, bo_sem_idx, attention_sinks):
        # Every token reads the same KV, so a group of tokens shares each
        # matmul's weights instead of loading them once per token. Rows in
        # [kv_len, kv_rows) get exactly zero weight from the score mask, so
        # they need no per-token zeroing.
        bkv = shared_bkv_ref[...]
        group = _largest_divisor(batch_size, SHARED_KV_TOKENS_PER_MATMUL)
        pending = None  # (first, step-2 args) of the previous group

        def finish(first, args):
            out = flash_attention_step2_pv(*args)
            bo_x2_ref[bo_sem_idx, pl.ds(first, group)] = out.reshape(
                group, num_q_heads, head_dim
            )

        for first in range(0, batch_size, group):
            tokens = range(first, first + group)
            bq = bq_x2_ref[bi_sem_idx, pl.ds(first, group)].reshape(-1, head_dim)
            swa_acc = swa_acc_x2_ref[bi_sem_idx, pl.ds(first, group)].reshape(
                -1, head_dim
            )
            swa_l = jnp.concatenate(
                [bl_x2_ref[bi_sem_idx, t, :num_q_heads][..., None] for t in tokens]
            )
            swa_m = jnp.concatenate(
                [bm_x2_ref[bi_sem_idx, t, :num_q_heads][..., None] for t in tokens]
            )
            kv_len = jnp.concatenate(
                [
                    jnp.full(
                        (num_q_heads, 1),
                        kv_lens_ref[batch_start_seq_idx + t],
                        jnp.int32,
                    )
                    for t in tokens
                ]
            )
            p, exp_m_diff, l_total = flash_attention_step1_qk_softmax(
                bq,
                bkv,
                kv_len,
                swa_m,
                swa_l,
                jnp.concatenate([attention_sinks] * group),
            )
            # As in `process`, the previous group's PV goes after this group's
            # QK, so the MXU works while the VPU does the softmax.
            if pending is not None:
                finish(*pending)
            pending = (first, (p, bkv, exp_m_diff, swa_acc, l_total))
        finish(*pending)

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
        prev_out = None

        # Wait for cur blocks if not ready yet
        wait_fetch_bq_batch(batch_start_seq_idx, bi_sem_idx)
        wait_fetch_swa_batch(batch_start_seq_idx, bi_sem_idx)
        wait_fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx)

        @pl.when(pl.program_id(0) >= 2)
        def _wait_send():
            wait_send_bo_batch(bo_sem_idx)

        if shared_kv:
            process_shared_kv_batch(bi_sem_idx, bo_sem_idx, attention_sinks)
            start_send_bo_batch(batch_start_seq_idx, bo_sem_idx)
            return

        for batch_idx in range(batch_size):
            bkv = load_bkv(bi_sem_idx, batch_idx)
            bq = load_bq(bi_sem_idx, batch_idx)

            if prev_out is not None:
                # Artificial dependency to force MXU/VPU interleaving by limiting
                # LLO's QK runahead
                # We use jnp.where to prevent XLA from optimizing the dependency away.
                # prev_out won't be inf in practice.
                bq = jnp.where(prev_out == jnp.inf, prev_out, bq)

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
                prev_out = out

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
        if shared_kv:
            load_shared_kv()

    process()

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def epilogue():
        wait_send_bo_batch(0)

        @pl.when(pl.num_programs(0) >= 2)
        def _wait_1():
            wait_send_bo_batch(1)

    ### ------- Kernel end ------- ###


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
        # With these, cache_kv_* are the whole caches and every token attends
        # over the first kv_lens rows of these pages (`_attention_kernel`).
        shared_kv_pages: jax.Array | None = None,  # i32[shared_pages]
        shared_kv_rows: jax.Array | None = None,  # i32
    ):
        batch_size = kernel_batch_size
        shared_kv = shared_kv_pages is not None
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

        if shared_kv:
            (shared_pages,) = shared_kv_pages.shape
            _, page_size, lanes = cache_kv_nope.shape
            bkv_nope_double_buf = pltpu.VMEM(
                (shared_pages, page_size, lanes), cache_kv_nope.dtype
            )
            bkv_rope_double_buf = pltpu.VMEM(
                (shared_pages, *cache_kv_rope.shape[1:]), cache_kv_rope.dtype
            )
            extra_scratch = [
                pltpu.VMEM((shared_pages * page_size, head_dim), jnp.bfloat16)
            ]
        else:
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
            extra_scratch = []

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
            *extra_scratch,
        ]

        launch = [start_seq_idx, end_seq_idx, num_valid_tokens]
        if shared_kv:
            launch += [shared_kv_rows, *shared_kv_pages]
        scalar_prefetches = (
            kv_lens,
            jnp.array(launch, jnp.int32),
            # (bi_sem_idx, bo_sem_idx)
            jnp.zeros((2,), jnp.int32),
            # seq_idx_start of the in-flight bo DMA per bo sem; -1 = none.
            jnp.full((2,), -1, jnp.int32),
        )

        if shared_kv:
            scope_name = f"SparseMLA-shared_{shared_pages * page_size}-bz_{batch_size}"
        else:
            gathered_topk = cache_kv_nope.shape[1]
            scope_name = (
                f"SparseMLA-p_{gathered_topk}-bz_{batch_size}"
                f"-gcz_{cache_kv_nope.shape[0]}"
            )
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _attention_kernel,
                    sm_scale=sm_scale,
                    batch_size=batch_size,
                    shared_kv=shared_kv,
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

    # A chunk whose tokens are one sequence's and each select all their
    # candidates (early prefill: kv_len <= topk) attends over the sequence's
    # leading cache rows, dequantized once per chunk, and skips the gather.
    pages_per_seq = num_page_indices // max_num_seqs
    shared_pages = cdiv(topk, page_size)
    shared_kv_path = q.shape[0] >= SHARED_KV_MIN_TOKENS and shared_pages <= min(
        2, pages_per_seq
    )  # 2: the kernel's DMA semaphore slots per cache.
    num_valid_tokens = cu_q_lens[distribution[2]]
    if shared_kv_path:
        leading_rows_only = _selects_leading_rows(topk_indices, kv_lens)
        seq_pages = page_indices.reshape(max_num_seqs, pages_per_seq)

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
        gather_end = batch_end
        if shared_kv_path:
            chunk_seq = seq_ids_segment[start_pos]
            is_valid = jnp.arange(start_pos, end_pos) < num_valid_tokens
            use_shared = jnp.any(is_valid) & jnp.all(
                ~is_valid
                | (
                    leading_rows_only[start_pos:end_pos]
                    & (seq_ids_segment[start_pos:end_pos] == chunk_seq)
                )
            )
            num_valid_indices = jnp.where(use_shared, 0, num_valid_indices)
            gather_end = jnp.where(use_shared, start_pos, batch_end)

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
            end_seq_idx=gather_end,
            num_valid_tokens=num_valid_tokens,
            kernel_batch_size=kernel_batch_size,
        )
        if shared_kv_path:
            q = run_mla_kernel(
                q,
                cache_kv_nope,
                cache_kv_rope,
                kv_lens,
                attention_sinks,
                swa_accumution,
                swa_l,
                swa_m,
                start_seq_idx=start_pos,
                end_seq_idx=jnp.where(use_shared, batch_end, start_pos),
                num_valid_tokens=num_valid_tokens,
                kernel_batch_size=kernel_batch_size,
                shared_kv_pages=seq_pages[chunk_seq, :shared_pages],
                shared_kv_rows=jnp.max(
                    jnp.where(is_valid, kv_lens[start_pos:end_pos], 0)
                ),
            )
    return prepare_outputs(
        q, actual_num_q_heads, actual_head_dim
    )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
