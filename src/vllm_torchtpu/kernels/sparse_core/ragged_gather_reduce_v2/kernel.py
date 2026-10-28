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
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import config, memory_ref


def call_kernel_pipeline(
    row_partition_id: jax.Array,
    refs: memory_ref.KernelRefs,
    col_start: jax.Array,
    cfg: config.Config,
):
    """Reduces this row partition's compacted rows [start, end) in one flat pipeline.

    Every output pair lies inside one partition's range, so a pair's last row
    carries its final value and is the pair's only write. The range is walked in
    16-row chunks from ``start`` rounded down to a chunk boundary; rows outside
    [start, end) are computed but not written. The partition's empty pairs are
    zeroed by scatters issued first, which drain under the pipeline.
    """
    num_simd_lanes = cfg.sc_info.num_lanes
    bounds = refs.scratch.partition_bounds_vmem
    fields = [bounds[pl.ds(k * num_simd_lanes, num_simd_lanes)] for k in range(4)]
    start, end, zero_start, zero_end = [jnp.array(0, jnp.int32)] * 4
    for i in range(cfg.num_row_partitions):
        is_mine = row_partition_id == i
        start = jnp.where(is_mine, fields[0][i], start)
        end = jnp.where(is_mine, fields[1][i], end)
        zero_start = jnp.where(is_mine, fields[2][i], zero_start)
        zero_end = jnp.where(is_mine, fields[3][i], zero_end)
    first_row = start - start % num_simd_lanes
    num_chunks = jnp.where(end > start, pl.cdiv(end - first_row, num_simd_lanes), 0)

    num_zero_scatters = _start_zero_pair_scatters(
        zero_start, zero_end, refs, col_start, cfg
    )

    # The zero row stands in for a pair's missing half.
    @plsc.parallel_loop(0, cfg.col_chunk_size, step=num_simd_lanes)
    def _(col):
        refs.scratch.out_vmem[cfg.zero_row, pl.ds(col, num_simd_lanes)] = jnp.zeros(
            (num_simd_lanes,), jnp.float32
        )

    refs.scratch.pair_smem[_PENDING_WRITES] = 0

    # The resident row metadata holds one window of chunks, so the chunks are run
    # one window at a time; the window count follows the partition's row count.
    num_windows = pl.cdiv(num_chunks, cfg.max_window)
    pl.loop(0, num_windows)(
        functools.partial(
            _window_kernel,
            refs=refs,
            start=start,
            end=end,
            first_row=first_row,
            num_chunks=num_chunks,
            col_start=col_start,
            cfg=cfg,
        )
    )

    _wait_pair_writes(refs, cfg)

    @pl.loop(0, num_zero_scatters)
    def _(_):
        _zero_pair_scatter(
            jnp.zeros((num_simd_lanes,), jnp.int32), 0, refs, col_start, cfg
        ).wait()


# pair_smem layout: per step, 16-entry lists of single-token pair writes (row,
# shift, pair) and two-token pair writes (low row, high row, pair), then the
# number of pair writes in flight.
_SINGLE_ROW, _SINGLE_SHIFT, _SINGLE_PAIR, _BOTH_LOW, _BOTH_HIGH, _BOTH_PAIR = (
    16 * k for k in range(6)
)
_PENDING_WRITES = 96


def _out_words(refs: memory_ref.KernelRefs) -> jax.Ref:
    """The output as 32-bit word rows (one row per pair, see Config)."""
    return refs.data.out.bitcast(jnp.uint32)


def _zero_pair_scatter(
    pairs: jax.Array,
    piece: int | jax.Array,
    refs: memory_ref.KernelRefs,
    col_start: jax.Array,
    cfg: config.Config,
):
    cols = pl.ds(col_start + piece * cfg.zero_cols, cfg.zero_cols)
    return pltpu.make_async_copy(
        refs.scratch.zero_vmem,
        _out_words(refs).at[pairs, cols],
        refs.scratch.sem.at[4],
    )


def _start_zero_pair_scatters(
    zero_start: jax.Array,
    zero_end: jax.Array,
    refs: memory_ref.KernelRefs,
    col_start: jax.Array,
    cfg: config.Config,
) -> jax.Array:
    """Starts zero scatters over this partition's columns of its empty pairs,
    ``sort_order[zero_start:zero_end] - num_rows``, 16 pairs per scatter.

    Lanes past ``zero_end`` repeat their chunk's first pair (an idempotent zero
    write). Returns the number of scatters started.
    """
    num_simd_lanes = cfg.sc_info.num_lanes
    scratch = refs.scratch
    ids_sem = scratch.sem.at[2]
    window_chunks = config.ZERO_ID_WINDOW_CHUNKS
    num_pieces = cfg.col_size // cfg.zero_cols
    lanes = jax.lax.broadcasted_iota(jnp.int32, (num_simd_lanes,), 0)
    num_zero_chunks = jnp.where(
        zero_end > zero_start, pl.cdiv(zero_end - zero_start, num_simd_lanes), 0
    )

    @plsc.parallel_loop(0, cfg.zero_cols, step=num_simd_lanes)
    def _(col):
        for row in range(num_simd_lanes):
            scratch.zero_vmem[row, pl.ds(col, num_simd_lanes)] = jnp.zeros(
                (num_simd_lanes,), jnp.uint32
            )

    def ids_gather(chunk, window_chunk):
        chunk_start = zero_start + chunk * num_simd_lanes
        positions = chunk_start + lanes
        positions = jnp.where(positions < zero_end, positions, chunk_start)
        return pltpu.make_async_copy(
            refs.index.sort_order.at[positions],
            scratch.zero_ids_vmem.at[
                pl.ds(window_chunk * num_simd_lanes, num_simd_lanes)
            ],
            ids_sem,
        )

    @pl.loop(0, pl.cdiv(num_zero_chunks, window_chunks))
    def _(window):
        first = window * window_chunks
        count = jnp.minimum(window_chunks, num_zero_chunks - first)

        @pl.loop(0, count)
        def _(k):
            ids_gather(first + k, k).start()

        @pl.loop(0, count)
        def _(k):
            ids_gather(first + k, k).wait()

        @pl.loop(0, count)
        def _(k):
            ids = scratch.zero_ids_vmem[pl.ds(k * num_simd_lanes, num_simd_lanes)]
            pairs = ids - cfg.padded_input_size
            for piece in range(num_pieces):
                _zero_pair_scatter(pairs, piece, refs, col_start, cfg).start()

    return num_zero_chunks * num_pieces


def _window_kernel(
    window_id: jax.Array,
    *,
    refs: memory_ref.KernelRefs,
    start: jax.Array,
    end: jax.Array,
    first_row: jax.Array,
    num_chunks: jax.Array,
    col_start: jax.Array,
    cfg: config.Config,
):
    """Stages one window of row metadata, then runs its chunks' column steps."""
    num_simd_lanes = cfg.sc_info.num_lanes
    scratch = refs.scratch
    recv_sem = scratch.sem.at[0]
    meta_sem = scratch.sem.at[3]
    window_words = cfg.window_size
    window_start = first_row + window_id * window_words
    chunks_in_window = jnp.minimum(
        cfg.max_window, num_chunks - window_id * cfg.max_window
    )

    window_dmas = [
        pltpu.make_async_copy(hbm.at[pl.ds(window_start, window_words)], vmem, recv_sem)
        for hbm, vmem in (
            (refs.index.sorted_by_validity, scratch.sorted_by_validity_vmem),
            (refs.index.scatter_meta, scratch.scatter_meta_vmem),
        )
    ]
    for dma in window_dmas:
        dma.start()
    for dma in window_dmas:
        dma.wait()

    # Gather each row's source index and weight through its sort slot.
    def row_meta_dmas(chunk):
        chunk_rows = pl.ds(chunk * num_simd_lanes, num_simd_lanes)
        sort_slots = scratch.sorted_by_validity_vmem[chunk_rows]
        return [
            pltpu.make_async_copy(src.at[sort_slots], dst.at[chunk_rows], meta_sem)
            for src, dst in (
                (refs.index.indices, scratch.src_indices_vmem),
                (refs.data.topk_weights, scratch.tw_f32_vmem),
            )
        ]

    @pl.loop(0, chunks_in_window)
    def _(chunk):
        for dma in row_meta_dmas(chunk):
            dma.start()

    @pl.loop(0, chunks_in_window)
    def _(chunk):
        for dma in row_meta_dmas(chunk):
            dma.wait()

    pipeline = pltpu.emit_pipeline(
        functools.partial(
            _col_kernel,
            refs=refs,
            start=start,
            end=end,
            window_start=window_start,
            col_start=col_start,
            cfg=cfg,
        ),
        grid=(chunks_in_window, cfg.num_col_chunks),
        in_specs=_col_gather_spec(scratch.src_indices_vmem, col_start, cfg),
    )
    pipeline(refs.data.source.bitcast(jnp.uint32))


def _col_gather_spec(
    src_indices_vmem: jax.Ref,
    col_start: jax.Array,
    cfg: config.Config,
) -> pl.BlockSpec:
    """Indirect BlockSpec gathering one chunk's rows of a column chunk of x."""
    num_simd_lanes = cfg.sc_info.num_lanes

    def col_index_map(
        r: int | jax.Array, c: int | jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        row = jnp.bitwise_right_shift(
            src_indices_vmem[pl.ds(r * num_simd_lanes, num_simd_lanes)],
            cfg.row_shift,
        )
        col = col_start // cfg.col_chunk_size + c
        return (row, col)

    return pl.BlockSpec(
        (pl.Indirect(num_simd_lanes), cfg.col_chunk_size),
        col_index_map,
    )


def _col_kernel(
    gather_ref: jax.Ref,
    *,
    refs: memory_ref.KernelRefs,
    start: jax.Array,
    end: jax.Array,
    window_start: jax.Array,
    col_start: jax.Array,
    cfg: config.Config,
):
    r = pl.program_id(0)
    c = pl.program_id(1)
    num_simd_lanes = cfg.sc_info.num_lanes
    scratch = refs.scratch
    out_vmem = scratch.out_vmem

    row_slice = pl.ds(r * num_simd_lanes, num_simd_lanes)
    tw_slice = scratch.tw_f32_vmem[row_slice]
    src_idx_slice = scratch.src_indices_vmem[row_slice]
    meta_slice = scratch.scatter_meta_vmem[row_slice]
    same_as_prev = jnp.bitwise_and(meta_slice, memory_ref.META_SAME_AS_PREV)

    def reduce_rows(slot_out_vmem: jax.Ref):
        @plsc.parallel_loop(0, cfg.col_chunk_size, step=num_simd_lanes)
        def col_loop(col_compute_offset: jax.Array):
            col_slice = pl.ds(col_compute_offset, num_simd_lanes)
            # Running sum, seeded by the carry from the previous chunk.
            previous_accumulated_data = scratch.prev_iter_last_row_vmem[c, col_slice]
            for row_src in range(num_simd_lanes):
                val_u32 = gather_ref[row_src, col_slice]
                if cfg.in_dtype == jnp.bfloat16:
                    # The two bfloat16 rows packed in one uint32 word sit in the low
                    # (even row) or high (odd row) 16 bits. Shift the wanted half
                    # into the float32 sign/exponent position and clear the rest.
                    is_even_row = jnp.bitwise_and(src_idx_slice[row_src], 1) == 0
                    shift = jnp.where(is_even_row, 16, 0)
                    lower_mask = jnp.uint32(jnp.iinfo(jnp.uint16).max)
                    upper_mask = jnp.left_shift(lower_mask, 16)
                    shifted = jnp.bitwise_and(
                        jnp.left_shift(val_u32, shift), upper_mask
                    )
                    data_f32 = plsc.bitcast(shifted, jnp.float32)
                else:
                    data_f32 = plsc.bitcast(val_u32, jnp.float32)
                data_f32 *= tw_slice[row_src]

                # Reduction: accumulate while the destination group is unchanged,
                # restart otherwise. Sorting guarantees rows of one group are
                # contiguous.
                accumulated_data = jnp.where(
                    same_as_prev[row_src] != 0,
                    previous_accumulated_data + data_f32,
                    data_f32,
                )
                previous_accumulated_data = accumulated_data

                slot_out_vmem[row_src, col_slice] = accumulated_data
                if row_src == num_simd_lanes - 1:
                    scratch.prev_iter_last_row_vmem[c, col_slice] = accumulated_data

    # The chunk's rows are consumed within the step (the pair writes read
    # packed_vmem), so one set of rows suffices.
    reduce_rows(out_vmem.at[pl.ds(0, num_simd_lanes)])

    # Each pair whose last row is in this chunk and in [start, end) is packed into
    # a row of packed_vmem: a pair with one token from that token's row, from the
    # top; a pair with both from its two halves' rows, from the bottom.
    lanes = jax.lax.broadcasted_iota(jnp.int32, (num_simd_lanes,), 0)
    row_ids = window_start + r * num_simd_lanes + lanes
    in_range = jnp.logical_and(row_ids >= start, row_ids < end)
    is_pair_final = jnp.bitwise_and(meta_slice, memory_ref.META_PAIR_FINAL)
    writes = jnp.where(in_range, is_pair_final, 0)
    pair_smem = scratch.pair_smem
    num_single = jnp.array(0, jnp.int32)
    num_both = jnp.array(0, jnp.int32)
    for i in range(num_simd_lanes):
        meta = meta_slice[i]
        src = jnp.bitwise_and(
            jnp.right_shift(meta, memory_ref.META_SRC_SHIFT), memory_ref.META_SRC_MASK
        )
        is_odd = jnp.bitwise_and(meta, memory_ref.META_IS_ODD) != 0
        has_both = jnp.logical_and(is_odd, src != memory_ref.SRC_ZERO)
        write = writes[i] != 0
        pair = jnp.right_shift(meta, memory_ref.META_PAIR_SHIFT)
        # Single: the row and the shift that puts its half in place.
        pair_smem[_SINGLE_ROW + num_single] = i
        pair_smem[_SINGLE_SHIFT + num_single] = jnp.where(is_odd, 16, 0)
        pair_smem[_SINGLE_PAIR + num_single] = pair
        # Both: the low half's row (the high half is this row).
        low_row = jnp.where(src == memory_ref.SRC_CARRY, cfg.carry_row + c, src)
        pair_smem[_BOTH_LOW + num_both] = low_row
        pair_smem[_BOTH_HIGH + num_both] = i
        pair_smem[_BOTH_PAIR + num_both] = pair
        num_single += jnp.logical_and(write, jnp.logical_not(has_both)).astype(
            jnp.int32
        )
        num_both += jnp.logical_and(write, has_both).astype(jnp.int32)

    # The previous step's writes read packed_vmem; they drained under this
    # step's reduction.
    _wait_pair_writes(refs, cfg)
    pair_smem[_PENDING_WRITES] = num_single + num_both

    tiles = cfg.col_chunk_size // cfg.tpu_info.num_lanes

    @_flat_tile_loop(num_single, tiles, cfg)
    def _(k, col_slice):
        value = out_vmem[pair_smem[_SINGLE_ROW + k], col_slice]
        if cfg.in_dtype == jnp.bfloat16:
            # The convert rounds to nearest even, as XLA's f32->bf16 does.
            word = plsc.pack(
                value, jnp.zeros_like(value), format=plsc.PackFormat.INTERLEAVED
            )
            word = jnp.left_shift(
                plsc.bitcast(word, jnp.uint32), pair_smem[_SINGLE_SHIFT + k]
            )
        else:
            word = plsc.bitcast(value, jnp.uint32)
        scratch.packed_vmem[k, col_slice] = word

    if cfg.in_dtype == jnp.bfloat16:

        @_flat_tile_loop(num_both, tiles, cfg)
        def _(k, col_slice):
            word = plsc.pack(
                out_vmem[pair_smem[_BOTH_LOW + k], col_slice],
                out_vmem[pair_smem[_BOTH_HIGH + k], col_slice],
                format=plsc.PackFormat.INTERLEAVED,
            )
            scratch.packed_vmem[num_simd_lanes - 1 - k, col_slice] = plsc.bitcast(
                word, jnp.uint32
            )

    out_words = _out_words(refs)
    cols = pl.ds(col_start + c * cfg.col_chunk_size, cfg.col_chunk_size)

    @pl.loop(0, num_single)
    def _(k):
        _pair_write(k, pair_smem[_SINGLE_PAIR + k], out_words, cols, refs).start()

    @pl.loop(0, num_both)
    def _(k):
        _pair_write(
            num_simd_lanes - 1 - k, pair_smem[_BOTH_PAIR + k], out_words, cols, refs
        ).start()

    # A chunk that ends inside a pair keeps the pair's low half for the next
    # chunk (after the writes above have read the previous one).
    last = meta_slice[num_simd_lanes - 1]
    carry_src = jnp.bitwise_and(
        jnp.right_shift(last, memory_ref.META_SRC_SHIFT), memory_ref.META_SRC_MASK
    )
    keeps_carry = jnp.logical_and(
        jnp.bitwise_and(last, memory_ref.META_PAIR_FINAL) == 0,
        carry_src < num_simd_lanes,
    )

    @pl.when(keeps_carry)
    def _():
        @_tile_col_loop(cfg)
        def _(col_slice):
            out_vmem[cfg.carry_row + c, col_slice] = out_vmem[carry_src, col_slice]


def _flat_tile_loop(count: jax.Array, tiles: int, cfg: config.Config):
    """Runs ``body(k, col_slice)`` over k < count and a column chunk as one
    software-pipelined loop, 16 lanes at a time (see _tile_col_loop)."""
    num_simd_lanes = cfg.sc_info.num_lanes
    lanes_per_tile = cfg.tpu_info.num_lanes

    def decorator(body):
        @plsc.parallel_loop(0, count * tiles)
        def _(t):
            k = t // tiles
            tile_col = (t % tiles) * lanes_per_tile
            for j in range(0, lanes_per_tile, num_simd_lanes):
                body(k, pl.ds(tile_col + j, num_simd_lanes))

    return decorator


def _tile_col_loop(cfg: config.Config):
    """Runs ``body(col_slice)`` over a column chunk, 16 lanes at a time.

    One (8, 128) VMEM tile column per iteration keeps the in-tile offsets static,
    so a row's address is computed once per tile, not per 16 lanes.
    """
    num_simd_lanes = cfg.sc_info.num_lanes
    lanes_per_tile = cfg.tpu_info.num_lanes

    def decorator(body):
        @plsc.parallel_loop(0, cfg.col_chunk_size, step=lanes_per_tile)
        def _(tile_col):
            for j in range(0, lanes_per_tile, num_simd_lanes):
                body(pl.ds(tile_col + j, num_simd_lanes))

    return decorator


def _pair_write(k, pair, out_words, cols, refs: memory_ref.KernelRefs):
    return pltpu.make_async_copy(
        refs.scratch.packed_vmem.at[k], out_words.at[pair, cols], refs.scratch.sem.at[1]
    )


def _wait_pair_writes(refs: memory_ref.KernelRefs, cfg: config.Config):
    """Waits for the pair writes in flight (all one column chunk wide)."""
    cols = pl.ds(0, cfg.col_chunk_size)

    @pl.loop(0, refs.scratch.pair_smem[_PENDING_WRITES])
    def _(_):
        _pair_write(0, 0, _out_words(refs), cols, refs).wait()
