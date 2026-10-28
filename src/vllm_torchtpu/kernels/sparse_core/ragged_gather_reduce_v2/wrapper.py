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
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import (
    config,
    kernel,
    memory_ref,
)


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    out = jnp.sum(out, axis=1).astype(jnp.bfloat16)
    return out


def _sort_rows_then_empty_pairs(
    valid_rows_mask: jax.Array, pair_has_rows: jax.Array
) -> jax.Array:
    """One sort that lists the valid rows, then the empty pairs, each in order.

    Entries ``r < num_rows`` are source rows, entries ``num_rows + q`` are output
    pairs. Each key packs (class, entry) into one int32: valid rows, then pairs
    without a valid row, then invalid rows, then the other pairs. The keys are
    distinct, so one single-operand unstable sort gives the stable order.
    """
    (num_rows,) = valid_rows_mask.shape
    (num_pairs,) = pair_has_rows.shape
    num_entries = num_rows + num_pairs
    index_bits = max(1, (num_entries - 1).bit_length())
    assert index_bits + 2 < 31, f"{num_entries} entries do not pack into int32 keys"
    row_class = jnp.where(valid_rows_mask, 0, 2)
    pair_class = jnp.where(pair_has_rows, 3, 1)
    keys = jnp.left_shift(jnp.concatenate([row_class, pair_class]), index_bits)
    keys |= jnp.arange(num_entries, dtype=jnp.int32)
    keys = jax.lax.sort(keys, is_stable=False)
    return jnp.bitwise_and(keys, (1 << index_bits) - 1)


def _preprocess_scalar_data(
    indices: jax.Array,
    valid_rows_mask: jax.Array,
    cfg: config.Config,
) -> memory_ref.IndexRef:
    """Compacts the valid source rows and splits them evenly over row partitions.

    Args:
      indices: Indices for gather.
      valid_rows_mask: Mask indicating valid rows.
      cfg: Ragged gather reduce config.

    Returns:
      The kernel's index refs: ``sorted_by_validity``, each compacted row's
        source position (all valid rows first, in order); ``scatter_meta``, per
        compacted row its output pair above the ``memory_ref.META_*`` bits;
        ``sort_order``, the sort whose entries after the valid rows are the
        empty pairs (as ``num_rows + pair``); ``partition_bounds``, per row
        partition the [start, end) range of compacted rows it reduces and the
        [start, end) range of ``sort_order`` holding the empty pairs it zeroes,
        each field in its own ``num_simd_lanes`` words.
    """
    num_simd_lanes = cfg.sc_info.num_lanes
    num_row_partitions = cfg.num_row_partitions
    group_size = cfg.reduce_group_size
    pair_shift = cfg.row_shift
    num_rows = valid_rows_mask.shape[0]
    num_pairs = cfg.num_pairs

    pair_has_rows = jnp.any(
        valid_rows_mask.reshape(num_pairs, group_size << pair_shift), axis=-1
    )
    sort_order = _sort_rows_then_empty_pairs(valid_rows_mask, pair_has_rows)
    sorted_by_validity = sort_order[:num_rows]
    num_valid = jnp.sum(valid_rows_mask, dtype=jnp.int32)

    dst = sorted_by_validity // group_size
    pair = jnp.right_shift(dst, pair_shift)
    slot = jnp.arange(num_rows, dtype=jnp.int32)
    lane = slot % num_simd_lanes
    chunk_start = slot - lane
    no_group = jnp.full((1,), -1, jnp.int32)
    next_dst = jnp.concatenate([dst[1:], no_group])
    next_pair = jnp.concatenate([pair[1:], no_group])
    prev_dst = jnp.concatenate([no_group, dst[:-1]])
    is_valid = slot < num_valid
    is_last_valid = slot + 1 >= num_valid
    is_tok_final = is_valid & (is_last_valid | (next_dst != dst))
    is_pair_final = is_valid & (is_last_valid | (next_pair != pair))
    is_odd = dst > jnp.left_shift(pair, pair_shift)

    # The low half of an odd token's pair is the even token's last row, the row
    # just before the odd token's first. A token has at most group_size rows and
    # tokens ascend, so both follow from the group_size rows before each row.
    dst_before = jnp.concatenate([jnp.full((group_size,), -2, jnp.int32), dst])
    earlier = [
        dst_before[group_size - k : group_size - k + num_rows]
        for k in range(1, group_size + 1)
    ]
    rows_before_in_tok = sum((e == dst).astype(jnp.int32) for e in earlier[:-1])
    has_low = is_odd & functools.reduce(jnp.logical_or, [e == dst - 1 for e in earlier])
    low_row = slot - 1 - rows_before_in_tok
    low_in_chunk = low_row >= chunk_start
    odd_low_src = jnp.where(
        has_low,
        jnp.where(low_in_chunk, low_row - chunk_start, memory_ref.SRC_CARRY),
        memory_ref.SRC_ZERO,
    )
    pair_src = jnp.where(is_odd, odd_low_src, lane)
    # A chunk ending inside a pair keeps the pair's low half if it ends here:
    # the chunk's last row itself (an even token's last row), or the even
    # token's last row before an odd token's first.
    carry_src = jnp.where(
        is_odd,
        jnp.where(has_low & low_in_chunk, low_row - chunk_start, memory_ref.SRC_NONE),
        jnp.where(is_tok_final, lane, memory_ref.SRC_NONE),
    )
    src = jnp.where(is_pair_final, pair_src, carry_src)
    assert num_pairs < (1 << (31 - memory_ref.META_PAIR_SHIFT)), num_pairs
    scatter_meta = (
        jnp.left_shift(pair, memory_ref.META_PAIR_SHIFT)
        | jnp.left_shift(src, memory_ref.META_SRC_SHIFT)
        | jnp.where(prev_dst == dst, memory_ref.META_SAME_AS_PREV, 0)
        | jnp.where(is_pair_final, memory_ref.META_PAIR_FINAL, 0)
        | jnp.where(is_odd, memory_ref.META_IS_ODD, 0)
    )

    # Split the valid rows evenly, each boundary moved back to the first row of
    # the output pair it falls in, so no pair straddles two partitions.
    targets = jnp.arange(num_row_partitions + 1, dtype=jnp.int32) * num_valid
    targets = targets // num_row_partitions
    has_target = targets < num_valid
    target_pair = pair[jnp.minimum(targets, num_rows - 1)]
    first_target_row = jnp.left_shift(target_pair, pair_shift) * group_size
    rows_before_pair = jnp.sum(
        valid_rows_mask[None, :] & (slot[None, :] < first_target_row[:, None]),
        axis=1,
        dtype=jnp.int32,
    )
    bounds = jnp.where(has_target, rows_before_pair, num_valid)
    # The pairs each partition owns, the first from pair 0, the last to the end.
    pair_bounds = jnp.where(has_target, target_pair, num_pairs).at[0].set(0)
    pair_ids = jnp.arange(num_pairs, dtype=jnp.int32)
    empty_before = jnp.sum(
        ~pair_has_rows[None, :] & (pair_ids[None, :] < pair_bounds[:, None]),
        axis=1,
        dtype=jnp.int32,
    )
    zero_bounds = num_valid + empty_before

    tail = cfg.compacted_size - num_rows
    # Rows past a partition's end are gathered but never written; keep their
    # source positions inside the (unpadded) index and weight arrays.
    sorted_by_validity = jnp.pad(
        jnp.minimum(sorted_by_validity, cfg.input_size - 1), (0, tail)
    )
    scatter_meta = jnp.pad(
        scatter_meta,
        (0, tail),
        constant_values=memory_ref.SRC_NONE << memory_ref.META_SRC_SHIFT,
    )

    lane_pad = (0, num_simd_lanes - num_row_partitions)
    partition_bounds = jnp.concatenate(
        [
            jnp.pad(b, lane_pad)
            for b in (bounds[:-1], bounds[1:], zero_bounds[:-1], zero_bounds[1:])
        ]
    )
    return memory_ref.IndexRef(
        partition_bounds=partition_bounds,
        indices=indices,
        sorted_by_validity=sorted_by_validity,
        scatter_meta=scatter_meta.astype(jnp.int32),
        sort_order=sort_order,
    )


def main_kernel(
    scalar_ref: memory_ref.IndexRef,
    in_hbm_ref: jax.Ref,
    topk_weights_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    scratch_ref: memory_ref.ScratchRef,
    *,
    cfg: config.Config,
):
    # Step 1: Resolve this core's row/column partition and its column slice.
    core_id = jax.lax.axis_index((cfg.core_axis_name, cfg.subcore_axis_name))
    row_partition_id = core_id // cfg.num_column_partitions
    col_partition_id = core_id % cfg.num_column_partitions

    col_start = col_partition_id * cfg.col_size

    refs = memory_ref.KernelRefs.create(
        scalar_ref=scalar_ref,
        in_hbm_ref=in_hbm_ref,
        topk_weights_hbm_ref=topk_weights_hbm_ref,
        out_hbm_ref=out_hbm_ref,
        scratch_ref=scratch_ref,
    )

    # Step 2: Stage the partition bounds (the compacted row metadata is streamed
    # one window at a time in call_kernel_pipeline, bounding the resident scratch).
    recv_sem = refs.scratch.sem.at[0]
    bounds_dma = pltpu.make_async_copy(
        refs.index.partition_bounds,
        refs.scratch.partition_bounds_vmem,
        recv_sem,
    )
    bounds_dma.start()
    bounds_dma.wait()

    kernel.call_kernel_pipeline(
        row_partition_id=row_partition_id,
        refs=refs,
        col_start=col_start,
        cfg=cfg,
    )


@jax.jit(static_argnames=("reduce_group_size",))
def ragged_gather_reduce_v2(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Gathers ``x`` by ``indices``, weights and masks, then reduces by group.

    Args:
      x: 2-D input features, ``(num_rows, hidden_size)``.
      indices: 1-D gather indices, ``(input_size,)``.
      topk_weights: 1-D per-row weights, ``(input_size,)``.
      valid_rows_mask: 1-D bool mask of valid gathered rows, ``(input_size,)``.
      reduce_group_size: number of consecutive rows summed into one output row.

    Returns:
      Reduced output, ``(input_size // reduce_group_size, hidden_size)``.
    """
    # Step 1: Create config object.
    cfg = config.Config(
        input_size=indices.size,
        hidden_size=x.shape[-1],
        source_rows=x.shape[0],
        reduce_group_size=reduce_group_size,
        in_dtype=x.dtype,
        core_axis_name="core",
        subcore_axis_name="subcore",
        tpu_info=pltpu.get_tpu_info(),
    )

    # Step 2: Fallback to compiler version if needed.
    if cfg.should_fallback:
        return _fallback_implementation(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        )

    # Step 3: Pre-process inputs (weights, padding, sort by validity).
    # Simplify topk gather by using fp32 and ensure data is always word aligned.
    topk_weights_f32 = topk_weights.astype(jnp.float32)

    # Pad the input so each row partition holds a whole number of reduce
    # groups; no group is then split across two physical cores.
    valid_rows_mask = jnp.pad(
        valid_rows_mask,
        (0, cfg.padded_input_size - cfg.input_size),
        constant_values=False,
    )

    scalar = _preprocess_scalar_data(indices, valid_rows_mask, cfg)

    # Step 4: Launch the SparseCore kernel.
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=cfg.sc_info.num_cores,
        num_subcores=cfg.sc_info.num_subcores,
        core_axis_name=cfg.core_axis_name,
        subcore_axis_name=cfg.subcore_axis_name,
    )

    # The kernel writes every output row, in x's dtype, with zeros for groups
    # that have no valid source row.
    out = core_map_helper.kernel(
        functools.partial(main_kernel, cfg=cfg),
        out_type=jax.ShapeDtypeStruct(
            (cfg.padded_output_size, cfg.aligned_hidden_size), x.dtype
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        scratch_types=(memory_ref.ScratchRef.create_scratch_types(cfg),),
        mesh=vector_mesh,
        name="sc_ragged_gather_reduce_v2",
    )(scalar, x, topk_weights_f32)

    # Step 5: Drop the padding (a no-op when the shapes are already aligned).
    return out[: cfg.output_size, : cfg.hidden_size]
