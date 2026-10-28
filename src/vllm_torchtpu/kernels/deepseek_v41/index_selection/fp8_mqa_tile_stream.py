# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream 4K FP32 score tiles over already-gathered sparse keys.

Adapted from the user-provided pallas_kernel_lab/kernel/fp8_paged_mqa_logits_sparse.py
(source SHA256: 9c90d381979c5a796a8c289d01795f28535272144c0b13c12b00779b6e5e740b).
Keeps the existing interleaved packed-cache layout and separate visibility bits,
so every FP32 K-scale bit is preserved. Query-slot restoration is outside this
TensorCore stream, before the unchanged top-k.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .packed_fp8_cache import PAGE_RECORDS, PAGE_ROWS

FIELDS = PAGE_ROWS - 1
TILE_PAGES = 32
TILE_KEYS = TILE_PAGES * PAGE_RECORDS
TILE_ROWS = TILE_PAGES * PAGE_ROWS
UNROLL = 4
KEY_SLOTS = 2 * UNROLL
RECORD_SLOTS = 2 * UNROLL
FILL_BATCH = 8
MXU_PAGES = 2
SUBLANES = 8
DESC_FIELDS = 4


def pack_query_records(q: jax.Array, weights: jax.Array) -> jax.Array:
    """[T,H,128] FP8 Q and FP32 [T,H] weights -> int32 [T, H/4 + H, 128].

    Rows 0..H/4-1: word [a, d] byte k = Q[4a+k, d], the order pltpu.bitcast
    unpacks. Rows H/4..: weights*q_scale of head h broadcast over 128 lanes.
    20 KiB per query at H32; a pair loads two adjacent records in one DMA.
    """
    tokens, heads, dim = q.shape
    bytes_ = q.reshape(tokens, heads // 4, 4, dim).transpose(0, 1, 3, 2)
    words = lax.bitcast_convert_type(bytes_, jnp.int32)
    bits = lax.bitcast_convert_type(weights.astype(jnp.float32), jnp.int32)
    bits = jnp.broadcast_to(bits[:, :, None], (tokens, heads, dim))
    return jnp.concatenate((words, bits), axis=1)


def _combine(x, y, k, sub):
    """Butterfly stage: sublanes with bit k clear sum x over (s, s+k), others y."""
    a = x + pltpu.roll(x, SUBLANES - k, 0)  # a[s] = x[s] + x[s+k]
    b = y + pltpu.roll(y, k, 0)  # b[s] = y[s] + y[s-k]
    return jnp.where((sub & k) == 0, a, b)


def _kernel(
    meta,
    desc,
    dead,
    records,
    keys,
    visibility,
    out,
    pair_out,
    key_buf,
    visibility_buf,
    rec_buf,
    score_buf,
    fill,
    key_sem,
    visibility_sem,
    rec_sem,
    out_sem,
    pair_sem,
    fill_sem,
    *,
    heads,
):
    pair_steps, total_steps, dead_count = meta[0], meta[1], meta[2]
    pair_iters, total_iters = pair_steps // UNROLL, total_steps // UNROLL
    q_rows = heads // 4
    sub = lax.broadcasted_iota(jnp.int32, (SUBLANES, PAGE_RECORDS), 0)

    def field(x, f):
        return desc[x * DESC_FIELDS + f]

    def record_copy(row, slot, members):
        return pltpu.make_async_copy(
            records.at[pl.ds(row, members)],
            rec_buf.at[slot, pl.ds(0, members)],
            rec_sem.at[slot],
        )

    def key_copy(slot, tile=0):
        return pltpu.make_async_copy(keys.at[tile], key_buf.at[slot], key_sem.at[slot])

    def start_output(x, slot, members):
        if members == 1:
            pltpu.make_async_copy(
                score_buf.at[slot, 0], out.at[field(x, 1)], out_sem.at[slot]
            ).start()
            return
        # Member 0 keeps its own block order: union tile t is its output tile t.
        pltpu.make_async_copy(
            score_buf.at[slot, 0], out.at[field(x, 4)], pair_sem.at[slot]
        ).start()
        pltpu.make_async_copy(
            score_buf.at[slot, 1], pair_out.at[field(x, 1)], pair_sem.at[slot]
        ).start()

    def wait_output(slot, members):
        # Semaphores count bytes: one wait retires a pair's two 16 KiB copies.
        src = score_buf.at[slot] if members == 2 else score_buf.at[slot, 0]
        sem = pair_sem if members == 2 else out_sem
        pltpu.make_async_copy(src, src, sem.at[slot]).wait()

    def visibility_copy(slot, tile=0):
        return pltpu.make_async_copy(
            visibility.at[tile], visibility_buf.at[slot], visibility_sem.at[slot]
        )

    def issue(x, slot):
        key_copy(slot, field(x, 0)).start()
        visibility_copy(slot, field(x, 0)).start()
        flags, row = field(x, 3), field(x, 2)
        record_slot = lax.shift_right_logical(flags, 1)

        @pl.when((flags & 1) != 0)
        def load_record():
            pl.when(x < pair_steps)(lambda: record_copy(row, record_slot, 2).start())
            pl.when(x >= pair_steps)(lambda: record_copy(row, record_slot, 1).start())

    def compute(x, slot, members):
        record_slot = lax.shift_right_logical(field(x, 3), 1)
        q = [rec_buf[record_slot, m, :q_rows, :] for m in range(members)]
        lhs = pltpu.bitcast(
            jnp.concatenate(q) if members > 1 else q[0], jnp.float8_e4m3fn
        )
        # [H,128] per member; each 128-lane half of a logit chunk reuses it.
        head_weights = [
            lax.bitcast_convert_type(
                rec_buf[record_slot, m, q_rows : q_rows + heads, :], jnp.float32
            )
            for m in range(members)
        ]

        def head_sum(logits, m, half):
            # 4 vregs -> [8,128], sublane = head % 8 partial of page half.
            lanes = slice(half * PAGE_RECORDS, (half + 1) * PAGE_RECORDS)
            y = jnp.maximum(logits[m * heads : (m + 1) * heads, lanes], 0)
            y = (y * head_weights[m]).reshape(heads // SUBLANES, SUBLANES, -1)
            return y.sum(0)

        for group in range(TILE_PAGES // SUBLANES):
            # Butterfly stages fold as pages arrive (k=1 per chunk, k=2 per two
            # chunks, k=4 at the end), so at most two partials per member live.
            # Stage order is free: stage k maps page index bit log2(k) to
            # sublane bit k, so sublane j still holds page 8*group+j.
            pending = [[] for _ in range(members)]
            for chunk in range(SUBLANES // MXU_PAGES):
                page = group * SUBLANES + chunk * MXU_PAGES
                packed = jnp.concatenate(
                    [
                        key_buf[slot, pl.ds((page + i) * PAGE_ROWS, FIELDS), :]
                        for i in range(MXU_PAGES)
                    ],
                    axis=1,
                )
                rhs = pltpu.bitcast(packed, jnp.float8_e4m3fn)  # [128 dims, 256 keys]
                logits = lax.dot(lhs, rhs, preferred_element_type=jnp.float32)
                for m in range(members):
                    pair = _combine(
                        head_sum(logits, m, 0), head_sum(logits, m, 1), 1, sub
                    )
                    if chunk % 2:
                        pending[m].append(_combine(pending[m].pop(), pair, 2, sub))
                    else:
                        pending[m].append(pair)
            scale_words = jnp.concatenate(
                [
                    key_buf[
                        slot, pl.ds((group * SUBLANES + i) * PAGE_ROWS + FIELDS, 1), :
                    ]
                    for i in range(SUBLANES)
                ],
                axis=0,
            )
            scale = lax.bitcast_convert_type(scale_words, jnp.float32)
            bits = visibility_buf[slot, pl.ds(group * SUBLANES, SUBLANES), :]
            rows = slice(group * SUBLANES, (group + 1) * SUBLANES)
            for m in range(members):
                values = _combine(pending[m][0], pending[m][1], 4, sub)
                visible = (bits & (1 << m)) != 0
                score_buf[slot, m, rows, :] = jnp.where(
                    visible, values * scale, -jnp.inf
                )
        start_output(x, slot, members)

    def wait_outputs(k, members):
        for i in range(UNROLL):
            wait_output((k & 1) * UNROLL + i, members)

    def body(k, members, start):
        @pl.when(k + 1 < total_iters)
        def prefetch():
            base = ((k + 1) & 1) * UNROLL
            for i in range(UNROLL):
                issue((k + 1) * UNROLL + i, base + i)

        # Score slots of iteration k were last written by iteration k-2.
        pl.when(k - 2 >= start)(lambda: wait_outputs(k - 2, members))
        base = (k & 1) * UNROLL
        for i in range(UNROLL):
            key_copy(base + i).wait()
            visibility_copy(base + i).wait()
            flags = field(k * UNROLL + i, 3)
            pl.when((flags & 1) != 0)(
                lambda flags=flags: record_copy(
                    0, lax.shift_right_logical(flags, 1), members
                ).wait()
            )
        for i in range(UNROLL):
            compute(k * UNROLL + i, base + i, members)

    def stream(start, stop, members):
        lax.fori_loop(start, stop, lambda k, c: body(k, members, start) or c, None)
        for j in (2, 1):
            pl.when(stop - j >= start)(lambda j=j: wait_outputs(stop - j, members))

    @pl.when(total_iters > 0)
    def prologue():
        for i in range(UNROLL):
            issue(i, i)

    fill[...] = jnp.full(fill.shape, -jnp.inf, jnp.float32)

    def start_fill(x, carry):
        pltpu.make_async_copy(fill.at[0], out.at[dead[x]], fill_sem).start()
        return carry

    lax.fori_loop(0, dead_count, start_fill, None)
    stream(0, pair_iters, 2)
    stream(pair_iters, total_iters, 1)

    def wait_fills(x, carry):
        pltpu.make_async_copy(fill, fill, fill_sem).wait()
        return carry

    lax.fori_loop(0, dead_count // FILL_BATCH, wait_fills, None)


def score_tiles(
    meta,
    desc,
    dead,
    records,
    keys,
    visibility,
    *,
    heads,
    rows,
    pair_rows,
    interpret=False,
):
    """Run the TensorCore stream; returns ([rows,32,128], [pair_rows,32,128])."""
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    tile = (TILE_PAGES, PAGE_RECORDS)
    scratch = (
        pltpu.VMEM((KEY_SLOTS, TILE_ROWS, PAGE_RECORDS), jnp.int32),
        pltpu.VMEM((KEY_SLOTS, TILE_PAGES, PAGE_RECORDS), jnp.int32),
        pltpu.VMEM((RECORD_SLOTS, 2, *records.shape[1:]), jnp.int32),
        pltpu.VMEM((KEY_SLOTS, 2, *tile), jnp.float32),
        pltpu.VMEM((FILL_BATCH, *tile), jnp.float32),
        pltpu.SemaphoreType.DMA((KEY_SLOTS,)),
        pltpu.SemaphoreType.DMA((KEY_SLOTS,)),
        pltpu.SemaphoreType.DMA((RECORD_SLOTS,)),
        pltpu.SemaphoreType.DMA((KEY_SLOTS,)),
        pltpu.SemaphoreType.DMA((KEY_SLOTS,)),
        pltpu.SemaphoreType.DMA,
    )
    return pl.pallas_call(
        partial(_kernel, heads=heads),
        out_shape=(
            jax.ShapeDtypeStruct((rows, *tile), jnp.float32),
            jax.ShapeDtypeStruct((pair_rows, *tile), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=(hbm, hbm, hbm),
            out_specs=(hbm, hbm),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
            disable_bounds_checks=True,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="fp8_sparse_mqa_logits",
    )(meta, desc, dead, records, keys, visibility)
