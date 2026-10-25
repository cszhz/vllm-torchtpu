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
"""Masked-dense CSA attention for DeepSeek-V4 prefill.

`sparse_mla` gathers each query token's top-k compressed KV rows on SparseCore
and attends over them one token at a time, so every token streams its own
top-k rows and the MXU sees only `num_q_heads` query rows per KV block. While a
sequence's compressed KV is short, the tokens of a chunk select heavily
overlapping rows; this kernel instead streams the sequence's whole compressed
KV once per block of `bq_sz` query tokens and applies the indexer's selection
as a mask over the scores. It attends over exactly the same keys, merges the
SWA partial state and the attention sinks the same way, and differs from the
sparse kernel only in summation order.

Row order inside a streamed KV block. A CSA RoPE row holds 4 tokens
(`csa_cache_layout`), so a page's RoPE is naturally read as 4 groups of
`page_size // 4` rows, group `j` holding tokens `4 r + j`. The NoPE rows are
read in that same order with stride-4 sublane loads, which makes KV row
`c = p * page_size + j * (page_size // 4) + r` of a block hold position
`p * page_size + 4 r + j`. The mask is built in that row order (`build_mask`).

Mask encoding. One bit per (token, compressed position), `int32[tokens,
max_kv_len // 32]`. Positions are grouped in super-blocks of 32 x 128: bit `b`
of word `128 s + l` is row `(b % bits_per_block) * 128 + l` of KV block
`s * blocks_per_super + b // bits_per_block`. Expanding one block's mask is
then a few shifts of one lane-dense `[bq_sz, 128]` tile.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels import vmem_budget
from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla
from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
    _dequant_dsv4_fp8,
    prepare_outputs,
    prepare_q_inputs,
    prepare_swa_inputs,
)

DEFAULT_VMEM_LIMIT_BYTES = vmem_budget.dsv4_limit_bytes()
LANES = 128
WORD_BITS = 32
SUPER_BLOCK = WORD_BITS * LANES  # positions covered by one 128-word column
ROPE_GROUPS = 4  # tokens per RoPE row
ROPE_WORDS = 32


def cdiv(a, b):
    return (a + b - 1) // b


def _row_of_position(pos, *, page_size: int, bkv_sz: int):
    """(block, row) of a compressed position in the kernel's KV row order."""
    rows_per_group = page_size // ROPE_GROUPS
    block = pos // bkv_sz
    within = pos % bkv_sz
    page = within // page_size
    w = within % page_size
    row = page * page_size + (w % ROPE_GROUPS) * rows_per_group + w // ROPE_GROUPS
    return block, row


def mask_codes(
    topk_indices: jax.Array,  # i32[num_tokens, topk], -1 padded
    *,
    max_kv_len: int,
    page_size: int,
    bkv_sz: int,
) -> jax.Array:
    """Per selected position, `32 * word + bit` of its mask bit; -1 if dropped."""
    assert max_kv_len % SUPER_BLOCK == 0 and SUPER_BLOCK % bkv_sz == 0
    assert bkv_sz % LANES == 0
    blocks_per_super = SUPER_BLOCK // bkv_sz
    bits_per_block = bkv_sz // LANES
    valid = (topk_indices >= 0) & (topk_indices < max_kv_len)
    block, row = _row_of_position(
        jnp.where(valid, topk_indices, 0), page_size=page_size, bkv_sz=bkv_sz
    )
    word = (block // blocks_per_super) * LANES + row % LANES
    bit = (block % blocks_per_super) * bits_per_block + row // LANES
    return jnp.where(valid, word * WORD_BITS + bit, -1)


def _mask_sc_kernel(
    codes_hbm_ref, mask_hbm_ref, *, topk: int, max_kv_len: int, num_lanes: int
):
    """One token row per step: scatter ones into an unpacked row, then pack.

    `codes_hbm_ref` holds each selected position's index in the unpacked row,
    `4096 s + 128 b + l` for bit `b` of word `128 s + l`; selected positions
    are distinct, so the scatter has no conflicts.
    """
    core = lax.axis_index(("core", "subcore"))
    num_cores = lax.psum(1, ("core", "subcore"))
    rows_per_core = codes_hbm_ref.shape[0] // num_cores
    L = num_lanes

    def scoped(row_ref):
        def body(codes_ref, out_ref):
            @plsc.parallel_loop(0, max_kv_len, step=L)
            def _zero(i):
                row_ref[pl.ds(i, L)] = jnp.zeros((L,), jnp.int32)

            @plsc.parallel_loop(0, topk, step=L)
            def _scatter(k):
                c = codes_ref[0, pl.ds(k, L)]
                valid = c >= 0
                plsc.addupdate_scatter(
                    row_ref,
                    (jnp.where(valid, c, 0),),
                    jnp.ones((L,), jnp.int32),
                    mask=valid,
                )

            for sb in range(max_kv_len // SUPER_BLOCK):

                @plsc.parallel_loop(0, LANES, step=L)
                def _pack(lane, sb=sb):
                    acc = jnp.zeros((L,), jnp.int32)
                    for b in range(WORD_BITS):
                        bit = row_ref[pl.ds(sb * SUPER_BLOCK + b * LANES + lane, L)]
                        acc = jnp.bitwise_or(acc, jnp.left_shift(bit, b))
                    out_ref[0, pl.ds(sb * LANES + lane, L)] = acc

        pltpu.emit_pipeline(
            body,
            grid=(rows_per_core,),
            in_specs=pl.BlockSpec((1, topk), lambda r: (r * num_cores + core, 0)),
            out_specs=pl.BlockSpec(
                (1, max_kv_len // WORD_BITS), lambda r: (r * num_cores + core, 0)
            ),
        )(codes_hbm_ref, mask_hbm_ref)

    pl.run_scoped(scoped, pltpu.VMEM((max_kv_len,), jnp.int32))


def build_mask(
    topk_indices: jax.Array,  # i32[num_tokens, topk], -1 padded
    *,
    max_kv_len: int,
    page_size: int,
    bkv_sz: int,
) -> jax.Array:
    """Bit mask `int32[num_tokens, max_kv_len // 32]` of the top-k selection.

    Built on SparseCore, one token row per subcore step. Positions at or past
    `max_kv_len` are dropped; their tokens must not be routed to this kernel.
    """
    codes = mask_codes(
        topk_indices, max_kv_len=max_kv_len, page_size=page_size, bkv_sz=bkv_sz
    )
    word, bit = codes // WORD_BITS, codes % WORD_BITS
    unpacked = jnp.where(
        codes >= 0,
        (word // LANES) * SUPER_BLOCK + bit * LANES + word % LANES,
        -1,
    )
    num_tokens, topk = unpacked.shape
    sc = pltpu.get_tpu_info().sparse_core
    assert topk % sc.num_lanes == 0
    num_cores = sc.num_cores * sc.num_subcores
    pad = -num_tokens % num_cores
    unpacked = jnp.pad(unpacked, ((0, pad), (0, 0)), constant_values=-1)
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out = pl.kernel(
        functools.partial(
            _mask_sc_kernel,
            topk=topk,
            max_kv_len=max_kv_len,
            num_lanes=sc.num_lanes,
        ),
        out_type=jax.ShapeDtypeStruct(
            (num_tokens + pad, max_kv_len // WORD_BITS), jnp.int32
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True, disable_bounds_checks=True
        ),
        mesh=mesh,
        name="csa_masked_dense_mask",
    )(unpacked)
    return out[:num_tokens]


def build_mask_xla(
    topk_indices: jax.Array,
    *,
    max_kv_len: int,
    page_size: int,
    bkv_sz: int,
) -> jax.Array:
    """`build_mask` as an XLA scatter; the reference for tests."""
    codes = mask_codes(
        topk_indices, max_kv_len=max_kv_len, page_size=page_size, bkv_sz=bkv_sz
    )
    valid = codes >= 0
    word = jnp.where(valid, codes // WORD_BITS, 0)
    value = jnp.where(valid, jnp.left_shift(jnp.int32(1), codes % WORD_BITS), 0)
    num_tokens = codes.shape[0]
    token = jnp.broadcast_to(
        jnp.arange(num_tokens, dtype=jnp.int32)[:, None], codes.shape
    )
    # Selected positions are distinct per token, so adding the bits ORs them.
    return (
        jnp.zeros((num_tokens, max_kv_len // WORD_BITS), jnp.int32)
        .at[token.reshape(-1), word.reshape(-1)]
        .add(value.reshape(-1), mode="drop")
    )


def _kernel(
    # Scalar prefetch
    kv_lens_ref,  # i32[max_num_seqs] compressed KV length per sequence
    page_indices_ref,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # i32[max_num_seqs + 1]
    start_end_seq_idx_ref,  # i32[2]
    sem_ids_ref,  # i32[3] (bq, bkv, bo)
    bo_ids_ref,  # i32[4] (seq of bo sem 0/1, bq block of bo sem 0/1)
    # Inputs
    sinks_ref,  # VMEM f32[bq_sz * num_q_heads, 128]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    nope_hbm_ref,  # i32[total_pages, page_size, 128]
    rope_hbm_ref,  # i32[total_pages, page_size // 4, 128]
    mask_hbm_ref,  # i32[max_num_tokens, mask_words // 128, 128]
    swa_acc_hbm_ref,  # [max_num_tokens * num_q_heads, head_dim]
    swa_l_hbm_ref,  # f32[max_num_tokens * num_q_heads, 128]
    swa_m_hbm_ref,  # f32[max_num_tokens * num_q_heads, 128]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim], aliases q
    # Scratch
    nope_x2_ref,  # i32[2, bkv_sz, 128]
    rope_x2_ref,  # i32[2, bkv_sz // 4, 128]
    mask_x2_ref,  # i32[2, bq_sz, 1, 128]
    bq_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    bo_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    swa_acc_x2_ref,  # [2, bq_sz * num_q_heads, head_dim]
    swa_l_x2_ref,  # f32[2, bq_sz * num_q_heads, 128]
    swa_m_x2_ref,  # f32[2, bq_sz * num_q_heads, 128]
    sems,  # DMA[7, 2]
    l_ref,  # f32[bq_sz * num_q_heads, 128]
    m_ref,  # f32[bq_sz * num_q_heads, 128]
    acc_ref,  # f32[bq_sz * num_q_heads, head_dim]
    *,
    sm_scale: float,
    bkv_p: int,
    bq_sz: int,
    pv_bf16: bool,
):
    _, num_q_heads, head_dim = q_hbm_ref.shape
    _, page_size, _ = nope_hbm_ref.shape
    rows_per_group = page_size // ROPE_GROUPS
    bkv_sz = bkv_p * page_size
    blocks_per_super = SUPER_BLOCK // bkv_sz
    bits_per_block = bkv_sz // LANES
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    q_dtype = q_hbm_ref.dtype
    q_packing = 32 // jax.dtypes.itemsize_bits(q_dtype)
    heads_per_packing = num_q_heads // q_packing
    rows = bq_sz * num_q_heads

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    seq_idx = pl.program_id(0) + start_seq_idx
    q_len = cu_q_lens_ref[seq_idx + 1] - cu_q_lens_ref[seq_idx]
    kv_len = kv_lens_ref[seq_idx]

    def copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def q_rows(seq, bq_idx):
        start = cu_q_lens_ref[seq] + bq_idx * bq_sz
        return start, jnp.minimum(bq_sz, cu_q_lens_ref[seq + 1] - start)

    def fetch_bkv(seq, bq_idx, bkv_idx, sem_idx, *, wait=False):
        nope_vmem = nope_x2_ref.at[sem_idx]
        rope_vmem = rope_x2_ref.at[sem_idx]
        mask_vmem = mask_x2_ref.at[sem_idx]
        start, sz = q_rows(seq, bq_idx)
        if wait:
            copy(nope_vmem, nope_vmem, sems.at[0, sem_idx], True)
            copy(rope_vmem, rope_vmem, sems.at[1, sem_idx], True)
            dst = mask_vmem.at[pl.ds(0, sz), 0, pl.ds(0, LANES)]
            copy(dst, dst, sems.at[2, sem_idx], True)
            return
        for i in range(bkv_p):
            # Past the last page the copy re-reads a valid page; those rows are
            # zeroed in `load_bkv` and absent from the mask.
            idx = jnp.minimum(
                seq * pages_per_seq + bkv_idx * bkv_p + i, num_page_indices - 1
            )
            page = page_indices_ref[idx]
            copy(
                nope_hbm_ref.at[page],
                nope_vmem.at[pl.ds(i * page_size, page_size)],
                sems.at[0, sem_idx],
                False,
            )
            copy(
                rope_hbm_ref.at[page],
                rope_vmem.at[pl.ds(i * rows_per_group, rows_per_group)],
                sems.at[1, sem_idx],
                False,
            )
        copy(
            mask_hbm_ref.at[
                pl.ds(start, sz), bkv_idx // blocks_per_super, pl.ds(0, LANES)
            ],
            mask_vmem.at[pl.ds(0, sz), 0, pl.ds(0, LANES)],
            sems.at[2, sem_idx],
            False,
        )

    def fetch_bq(seq, bq_idx, sem_idx, *, wait=False):
        start, sz = q_rows(seq, bq_idx)
        hs = start * num_q_heads
        hsz = sz * num_q_heads
        copy(
            q_hbm_ref.at[pl.ds(start, sz)],
            bq_x2_ref.at[sem_idx, pl.ds(0, sz)],
            sems.at[3, sem_idx],
            wait,
        )
        copy(
            swa_acc_hbm_ref.at[pl.ds(hs, hsz)],
            swa_acc_x2_ref.at[sem_idx, pl.ds(0, hsz)],
            sems.at[4, sem_idx],
            wait,
        )
        copy(
            swa_l_hbm_ref.at[pl.ds(hs, hsz)],
            swa_l_x2_ref.at[sem_idx, pl.ds(0, hsz)],
            sems.at[5, sem_idx],
            wait,
        )
        copy(
            swa_m_hbm_ref.at[pl.ds(hs, hsz)],
            swa_m_x2_ref.at[sem_idx, pl.ds(0, hsz)],
            sems.at[5, sem_idx],
            wait,
        )

    def send_bo(seq, bq_idx, sem_idx, *, wait=False):
        start, sz = q_rows(seq, bq_idx)
        copy(
            bo_x2_ref.at[sem_idx, pl.ds(0, sz)],
            o_hbm_ref.at[pl.ds(start, sz)],
            sems.at[6, sem_idx],
            wait,
        )

    def wait_send_bo(sem_idx):
        old_seq = bo_ids_ref[sem_idx]
        old_bq = bo_ids_ref[sem_idx + 2]

        @pl.when(jnp.logical_and(old_seq >= 0, old_seq <= seq_idx))
        def _():
            send_bo(old_seq, old_bq, sem_idx, wait=True)

    def load_bq(sem_idx):
        q_ref = (
            bq_x2_ref.bitcast(jnp.uint32)
            .at[sem_idx]
            .reshape(bq_sz * heads_per_packing, head_dim)
        )
        return pltpu.bitcast(q_ref[...], q_dtype).reshape(rows, head_dim)

    def load_bkv(sem_idx, bkv_idx):
        nope_words = []
        rope_rows = []
        for p in range(bkv_p):
            for j in range(ROPE_GROUPS):
                nope_words.append(
                    nope_x2_ref[
                        sem_idx,
                        pl.ds(p * page_size + j, rows_per_group, stride=ROPE_GROUPS),
                        :,
                    ]
                )
                w = rope_x2_ref[
                    sem_idx,
                    pl.ds(p * rows_per_group, rows_per_group),
                    pl.ds(j * ROPE_WORDS, ROPE_WORDS),
                ]
                lo = pltpu.bitcast(jnp.left_shift(w, 16), jnp.float32)
                hi = pltpu.bitcast(jnp.bitwise_and(w, jnp.int32(-65536)), jnp.float32)
                rope_rows.append(jnp.concatenate([lo, hi], axis=1))
        nope = _dequant_dsv4_fp8(jnp.concatenate(nope_words, axis=0))
        rope = jnp.concatenate(rope_rows, axis=0).astype(jnp.bfloat16)
        kv = jnp.concatenate([nope, rope], axis=1)  # [bkv_sz, 512]
        # Rows past kv_len may hold another cache's bytes (the NoPE array also
        # hosts the SWA cache and compressor state), which can decode to NaN.
        c = lax.broadcasted_iota(jnp.int32, kv.shape, 0)
        w = c % page_size
        pos = (
            bkv_idx * bkv_sz
            + (c // page_size) * page_size
            + (w % rows_per_group) * ROPE_GROUPS
            + w // rows_per_group
        )
        return jnp.where(pos < kv_len, kv, jnp.zeros_like(kv))

    def load_mask(sem_idx, bkv_idx):
        words = mask_x2_ref[sem_idx][:, 0, :]  # [bq_sz, 128]
        base = (bkv_idx % blocks_per_super) * bits_per_block
        planes = [
            jnp.bitwise_and(lax.shift_right_logical(words, base + t), 1)
            for t in range(bits_per_block)
        ]
        return jnp.concatenate(planes, axis=1) != 0  # [bq_sz, bkv_sz]

    def lanes(x, n):
        return jnp.concatenate([x] * (n // LANES), axis=1) if n != LANES else x

    def flash(q, kv, keep):
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
        s3 = s.reshape(bq_sz, num_q_heads, bkv_sz)
        s3 = jnp.where(keep[:, None, :], s3, jnp.finfo(jnp.float32).min)
        s = s3.reshape(rows, bkv_sz)
        m_prev = m_ref[...]
        m_curr = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
        m_ref[...] = m_curr
        p = jnp.exp(s - lanes(m_curr, bkv_sz))
        pv = jnp.einsum(
            "nm,md->nd",
            p.astype(kv.dtype) if pv_bf16 else p,
            kv,
            preferred_element_type=jnp.float32,
        )
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_ref[...] = exp_m_diff * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
        acc_ref[...] = lanes(exp_m_diff, head_dim) * acc_ref[...] + pv

    def process():
        num_bkv = jnp.maximum(1, cdiv(kv_len, bkv_sz))
        num_bq = jnp.maximum(1, cdiv(q_len, bq_sz))

        def next_bq_ids(bq_idx, sem_idx):
            nxt = bq_idx + 1
            last = nxt == num_bq
            return (
                lax.select(last, seq_idx + 1, seq_idx),
                lax.select(last, 0, nxt),
                lax.select(sem_idx == 0, 1, 0),
            )

        def next_bkv_ids(bq_idx, bkv_idx, sem_idx):
            nxt_kv = bkv_idx + 1
            last_kv = nxt_kv == num_bkv
            nxt_kv = lax.select(last_kv, 0, nxt_kv)
            nxt_q = lax.select(last_kv, bq_idx + 1, bq_idx)
            last_q = nxt_q == num_bq
            return (
                lax.select(last_q, seq_idx + 1, seq_idx),
                lax.select(last_q, 0, nxt_q),
                nxt_kv,
                lax.select(sem_idx == 0, 1, 0),
            )

        def compute_with_bq(bq_idx, _):
            bq_sem = sem_ids_ref[0]
            nseq, nbq, nsem = next_bq_ids(bq_idx, bq_sem)

            @pl.when(nseq < end_seq_idx)
            def _():
                sem_ids_ref[0] = nsem
                fetch_bq(nseq, nbq, nsem)

            fetch_bq(seq_idx, bq_idx, bq_sem, wait=True)
            # Start from the SWA pass's partial softmax state.
            m_ref[...] = swa_m_x2_ref[bq_sem]
            l_ref[...] = swa_l_x2_ref[bq_sem]
            acc_ref[...] = swa_acc_x2_ref[bq_sem].astype(jnp.float32)

            def compute_with_bkv(bkv_idx, _):
                kv_sem = sem_ids_ref[1]
                nseq, nbq, nkv, nsem = next_bkv_ids(bq_idx, bkv_idx, kv_sem)

                @pl.when(nseq < end_seq_idx)
                def _():
                    sem_ids_ref[1] = nsem
                    fetch_bkv(nseq, nbq, nkv, nsem)

                fetch_bkv(seq_idx, bq_idx, bkv_idx, kv_sem, wait=True)
                flash(
                    load_bq(bq_sem),
                    load_bkv(kv_sem, bkv_idx),
                    load_mask(kv_sem, bkv_idx),
                )

            lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

            l_total = l_ref[...] + jnp.exp(sinks_ref[...] - m_ref[...])
            inv = lanes(pl.reciprocal(l_total, approx=True), head_dim)
            out = (acc_ref[...] * inv).astype(q_dtype)

            bo_sem = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem == 0, 1, 0)
            wait_send_bo(bo_sem)
            bo_x2_ref.at[bo_sem].bitcast(jnp.int32).reshape(
                bq_sz * heads_per_packing, head_dim
            )[...] = pltpu.bitcast(out, jnp.int32)
            bo_ids_ref[bo_sem] = seq_idx
            bo_ids_ref[bo_sem + 2] = bq_idx
            send_bo(seq_idx, bq_idx, bo_sem)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    @pl.when(seq_idx == start_seq_idx)
    def _prologue():
        fetch_bq(start_seq_idx, 0, 0)
        fetch_bkv(start_seq_idx, 0, 0, 0)

    process()

    @pl.when(seq_idx == end_seq_idx - 1)
    def _epilogue():
        for i in range(2):
            wait_send_bo(i)


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "max_kv_len",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "pv_bf16",
        "vmem_limit_bytes",
    ),
)
def masked_dense_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # i32[total_pages, page_size, 128]
    cache_kv_rope: jax.Array,  # i32[total_pages, page_size // 4, 128]
    mask: jax.Array,  # i32[max_num_tokens, max_kv_len // 32] from build_mask
    kv_lens: jax.Array,  # i32[max_num_seqs] compressed KV length per sequence
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    start_seq_idx: jax.Array,  # i32 scalar
    end_seq_idx: jax.Array,  # i32 scalar
    attention_sinks: jax.Array,  # f32[actual_num_q_heads]
    swa_accumution: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    swa_l: jax.Array,  # f32[max_num_tokens, actual_num_q_heads]
    swa_m: jax.Array,  # f32[max_num_tokens, actual_num_q_heads]
    *,
    sm_scale: float,
    max_kv_len: int,
    num_kv_pages_per_block: int = 2,
    num_queries_per_block: int = 32,
    pv_bf16: bool = False,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """CSA attention for sequences `[start_seq_idx, end_seq_idx)`.

    Same contract as `sparse_mla.sparse_ragged_paged_attention` for those
    sequences' tokens; every other row of the result is `q` unchanged (the
    output aliases `q`), so a caller can run the sparse kernel for the rest.
    Each routed sequence must have `kv_lens[seq] <= max_kv_len`.
    """
    assert cache_kv_nope.dtype == jnp.int32 and cache_kv_rope.dtype == jnp.int32
    _, actual_num_q_heads, actual_head_dim = q.shape
    q = prepare_q_inputs(q)
    num_tokens, num_q_heads, head_dim = q.shape
    _, page_size, _ = cache_kv_nope.shape
    bkv_p = num_kv_pages_per_block
    bkv_sz = bkv_p * page_size
    bq_sz = num_queries_per_block
    assert mask.shape == (num_tokens, max_kv_len // WORD_BITS), mask.shape

    swa_acc, swa_l, swa_m = prepare_swa_inputs(swa_accumution, swa_l, swa_m)
    swa_acc = swa_acc.reshape(num_tokens * num_q_heads, head_dim)

    def per_row(x):  # [tokens, >=heads] -> lane-broadcast [tokens * heads, 128]
        x = x[:, :num_q_heads].reshape(num_tokens * num_q_heads, 1)
        return jnp.broadcast_to(x, (num_tokens * num_q_heads, LANES))

    sinks = jnp.pad(
        attention_sinks,
        (0, num_q_heads - actual_num_q_heads),
        constant_values=jnp.finfo(attention_sinks.dtype).min,
    )
    sinks = jnp.broadcast_to(
        jnp.tile(sinks, bq_sz)[:, None], (bq_sz * num_q_heads, LANES)
    ).astype(jnp.float32)

    rows = bq_sz * num_q_heads
    scratch_shapes = [
        pltpu.VMEM((2, bkv_sz, LANES), jnp.int32),
        pltpu.VMEM((2, bkv_sz // ROPE_GROUPS, LANES), jnp.int32),
        pltpu.VMEM((2, bq_sz, 1, LANES), jnp.int32),
        pltpu.VMEM((2, bq_sz, num_q_heads, head_dim), q.dtype),
        pltpu.VMEM((2, bq_sz, num_q_heads, head_dim), q.dtype),
        pltpu.VMEM((2, rows, head_dim), swa_acc.dtype),
        pltpu.VMEM((2, rows, LANES), jnp.float32),
        pltpu.VMEM((2, rows, LANES), jnp.float32),
        pltpu.SemaphoreType.DMA((7, 2)),
        pltpu.VMEM((rows, LANES), jnp.float32),
        pltpu.VMEM((rows, LANES), jnp.float32),
        pltpu.VMEM((rows, head_dim), jnp.float32),
    ]
    end_seq_idx = jnp.maximum(start_seq_idx, end_seq_idx)
    scalar_prefetches = (
        kv_lens,
        page_indices,
        cu_q_lens,
        jnp.stack([start_seq_idx, end_seq_idx]).astype(jnp.int32),
        jnp.zeros((3,), jnp.int32),
        jnp.full((4,), -1, jnp.int32),
    )
    scope = f"CsaMaskedDense-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}"
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    kernel = pl.pallas_call(
        functools.partial(
            _kernel, sm_scale=sm_scale, bkv_p=bkv_p, bq_sz=bq_sz, pv_bf16=pv_bf16
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] + [hbm] * 7,
            out_specs=hbm,
            grid=(end_seq_idx - start_seq_idx,),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        input_output_aliases={len(scalar_prefetches) + 1: 0},
        name=scope,
    )
    out = jax.named_scope(scope)(kernel)(
        *scalar_prefetches,
        sinks,
        q,
        cache_kv_nope,
        cache_kv_rope,
        mask.reshape(num_tokens, -1, LANES),
        swa_acc,
        per_row(swa_l),
        per_row(swa_m),
    )
    return prepare_outputs(out, actual_num_q_heads, actual_head_dim)


# A sparse prefix of at most this many rows (verify windows) runs on a static
# slice of the step, so the sparse path's per-chunk launches over the dense
# rows are skipped.
SPARSE_PREFIX_ROWS = 64


def split_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    topk_indices: jax.Array,  # i32[max_num_tokens, topk]
    kv_lens: jax.Array,  # i32[max_num_seqs] compressed KV length per sequence
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sinks: jax.Array,
    swa_accumution: jax.Array,
    swa_l: jax.Array,
    swa_m: jax.Array,
    *,
    sm_scale: float,
    max_kv_len: int,
    num_kv_pages_per_block: int,
    num_queries_per_block: int,
    gather_no_pad: bool = False,
) -> jax.Array:
    """CSA attention: sparse gather for the leading sequences, masked dense
    for the trailing ones whose compressed KV fits `max_kv_len`."""
    seq_ids = jnp.arange(kv_lens.shape[0], dtype=jnp.int32)
    too_long = (seq_ids < distribution[2]) & (kv_lens > max_kv_len)
    split_seq = jnp.max(jnp.where(too_long, seq_ids + 1, 0)).astype(jnp.int32)
    num_sparse_tokens = cu_q_lens[split_seq]

    def sparse(rows):
        def run(q):
            out = sparse_mla.sparse_ragged_paged_attention(
                q=q[:rows],
                cache_kv_nope=cache_kv_nope,
                cache_kv_rope=cache_kv_rope,
                topk_indices=topk_indices[:rows],
                page_indices=page_indices,
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                attention_sinks=attention_sinks,
                swa_accumution=swa_accumution[:rows],
                swa_l=swa_l[:rows],
                swa_m=swa_m[:rows],
                sm_scale=sm_scale,
                gather_and_attention_chunk_size=32 if rows <= 128 else 64,
                attention_kernel_batch_size=8,
                gather_no_pad=gather_no_pad,
                num_valid_tokens=num_sparse_tokens,
            )
            return out if rows == q.shape[0] else q.at[:rows].set(out)

        return run

    num_tokens = q.shape[0]
    if num_tokens <= SPARSE_PREFIX_ROWS:
        q = sparse(num_tokens)(q)
    else:
        q = lax.cond(
            num_sparse_tokens <= SPARSE_PREFIX_ROWS,
            sparse(SPARSE_PREFIX_ROWS),
            sparse(num_tokens),
            q,
        )
    page_size = cache_kv_nope.shape[1]
    mask = build_mask(
        topk_indices,
        max_kv_len=max_kv_len,
        page_size=page_size,
        bkv_sz=num_kv_pages_per_block * page_size,
    )
    return masked_dense_attention(
        q,
        cache_kv_nope,
        cache_kv_rope,
        mask,
        kv_lens,
        page_indices,
        cu_q_lens,
        split_seq,
        distribution[2],
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m,
        sm_scale=sm_scale,
        max_kv_len=max_kv_len,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
    )
