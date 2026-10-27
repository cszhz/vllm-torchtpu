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

    Every reduce group lies inside one partition's range, so a group's last row
    holds its full sum and is the group's only scatter. The range is walked in
    16-row chunks from ``start`` rounded down to a chunk boundary; rows outside
    [start, end) are computed but routed to the garbage row.
    """
    num_simd_lanes = cfg.sc_info.num_lanes
    bounds = refs.scratch.partition_bounds_vmem
    starts = bounds[pl.ds(0, num_simd_lanes)]
    ends = bounds[pl.ds(num_simd_lanes, num_simd_lanes)]
    start = jnp.array(0, jnp.int32)
    end = jnp.array(0, jnp.int32)
    for i in range(cfg.num_row_partitions):
        start = jnp.where(row_partition_id == i, starts[i], start)
        end = jnp.where(row_partition_id == i, ends[i], end)
    first_row = start - start % num_simd_lanes
    num_chunks = jnp.where(end > start, pl.cdiv(end - first_row, num_simd_lanes), 0)

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
    step = r * cfg.num_col_chunks + c
    slot = step % 2
    col_hbm_start = col_start + c * cfg.col_chunk_size
    num_simd_lanes = cfg.sc_info.num_lanes
    scratch = refs.scratch
    out_vmem = scratch.out_vmem

    row_slice = pl.ds(r * num_simd_lanes, num_simd_lanes)
    tw_slice = scratch.tw_f32_vmem[row_slice]
    src_idx_slice = scratch.src_indices_vmem[row_slice]
    meta_slice = scratch.scatter_meta_vmem[row_slice]
    same_as_prev = jnp.bitwise_and(meta_slice, 1)

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

                # The output buffer stays float32: a bfloat16 output would be
                # (16, 128)-tiled and the per-row scatter below writes a single
                # row at an arbitrary, non-tile-aligned destination, which is only
                # legal for 32-bit elements. The cast happens in the wrapper.
                slot_out_vmem[row_src, col_slice] = accumulated_data
                if row_src == num_simd_lanes - 1:
                    scratch.prev_iter_last_row_vmem[c, col_slice] = accumulated_data

    # A static slot keeps the out_vmem address out of the loop's scalar registers.
    for static_slot in range(2):
        pl.when(slot == static_slot)(
            functools.partial(reduce_rows, out_vmem.at[static_slot])
        )

    # Each row's scatter destination: a group's last row carries the group's
    # full sum, every other row (and every row outside [start, end)) goes to the
    # garbage row.
    row_ids = (
        window_start
        + r * num_simd_lanes
        + jax.lax.broadcasted_iota(jnp.int32, (num_simd_lanes,), 0)
    )
    in_range = jnp.logical_and(row_ids >= start, row_ids < end)
    garbage_dst = refs.data.out.shape[0] - 1
    dst_rows = jnp.where(
        in_range,
        jnp.bitwise_right_shift(scratch.scatter_meta_vmem[row_slice], 1),
        garbage_dst,
    )

    # One indirect scatter of the 16 rows, completed within the step.
    scatter = pltpu.make_async_copy(
        out_vmem.at[slot],
        refs.data.out.at[dst_rows, pl.ds(col_hbm_start, cfg.col_chunk_size)],
        scratch.sem.at[1 + slot],
    )
    scatter.start()
    scatter.wait()
