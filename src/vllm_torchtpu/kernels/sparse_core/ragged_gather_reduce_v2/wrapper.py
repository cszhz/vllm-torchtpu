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


def _stable_partition_by_validity(valid_rows_mask: jax.Array) -> jax.Array:
    """The indices of the valid entries, then of the invalid ones, each in order.

    Equal to a stable ``argsort(~mask)``: each key packs (invalid, index) into
    one int32, the keys are distinct, so one single-operand unstable sort gives
    the stable order without the two-operand comparator.
    """
    (num_rows,) = valid_rows_mask.shape
    index_bits = max(1, (num_rows - 1).bit_length())
    assert index_bits < 31, f"{num_rows} entries do not pack into int32 keys"
    index = jnp.arange(num_rows, dtype=jnp.int32)
    keys = jnp.where(valid_rows_mask, 0, 1 << index_bits) | index
    keys = jax.lax.sort(keys, is_stable=False)
    return jnp.bitwise_and(keys, (1 << index_bits) - 1)


def _preprocess_scalar_data(
    indices: jax.Array,
    valid_rows_mask: jax.Array,
    cfg: config.Config,
) -> tuple[memory_ref.IndexRef, jax.Array]:
    """Compacts the valid source rows and splits them evenly over row partitions.

    Args:
      indices: Indices for gather.
      valid_rows_mask: Mask indicating valid rows.
      cfg: Ragged gather reduce config.

    Returns:
      The kernel's index refs: ``sorted_by_validity``, each compacted row's
        source position (all valid rows first, in order); ``scatter_meta``, per
        compacted row ``2 * destination + same_as_prev``, where the destination
        is the output row for the last row of a reduce group and the garbage
        row otherwise; ``partition_bounds``, the [start, end) range of
        compacted rows each row partition reduces (starts in the first
        ``num_simd_lanes`` words, ends in the next).
      mask: per output group, whether the group has any valid source row.
    """
    num_simd_lanes = cfg.sc_info.num_lanes
    num_row_partitions = cfg.num_row_partitions
    group_size = cfg.reduce_group_size
    num_rows = valid_rows_mask.shape[0]

    # A stable partition: valid rows keep their relative order and move ahead
    # of the invalid ones.
    sorted_by_validity = _stable_partition_by_validity(valid_rows_mask)
    num_valid = jnp.sum(valid_rows_mask, dtype=jnp.int32)

    dst = sorted_by_validity // group_size
    slot = jnp.arange(num_rows, dtype=jnp.int32)
    no_group = jnp.full((1,), -1, jnp.int32)
    next_dst = jnp.concatenate([dst[1:], no_group])
    prev_dst = jnp.concatenate([no_group, dst[:-1]])
    is_final = (slot < num_valid) & ((slot + 1 >= num_valid) | (next_dst != dst))
    garbage_dst = cfg.padded_input_size // group_size
    scatter_meta = jnp.where(is_final, dst, garbage_dst) * 2 + (prev_dst == dst)

    # Split the valid rows evenly, each boundary moved back to the first row of
    # the reduce group it falls in, so no group straddles two partitions.
    targets = jnp.arange(num_row_partitions + 1, dtype=jnp.int32) * num_valid
    targets = targets // num_row_partitions
    target_group = dst[jnp.minimum(targets, num_rows - 1)]
    rows_before_group = jnp.sum(
        valid_rows_mask[None, :] & (slot[None, :] < target_group[:, None] * group_size),
        axis=1,
        dtype=jnp.int32,
    )
    bounds = jnp.where(targets < num_valid, rows_before_group, num_valid)

    tail = cfg.compacted_size - num_rows
    # Rows past a partition's end are gathered but never written; keep their
    # source positions inside the (unpadded) index and weight arrays.
    sorted_by_validity = jnp.pad(
        jnp.minimum(sorted_by_validity, cfg.input_size - 1), (0, tail)
    )
    scatter_meta = jnp.pad(scatter_meta, (0, tail), constant_values=garbage_dst * 2)

    lane_pad = (0, num_simd_lanes - num_row_partitions)
    partition_bounds = jnp.concatenate(
        [jnp.pad(bounds[:-1], lane_pad), jnp.pad(bounds[1:], lane_pad)]
    )
    mask = jnp.any(valid_rows_mask.reshape(-1, group_size), axis=-1)
    scalar = memory_ref.IndexRef(
        partition_bounds=partition_bounds,
        indices=indices,
        sorted_by_validity=sorted_by_validity,
        scatter_meta=scatter_meta.astype(jnp.int32),
    )
    return scalar, mask


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

    scalar, mask = _preprocess_scalar_data(indices, valid_rows_mask, cfg)

    # Step 4: Launch the SparseCore kernel.
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=cfg.sc_info.num_cores,
        num_subcores=cfg.sc_info.num_subcores,
        core_axis_name=cfg.core_axis_name,
        subcore_axis_name=cfg.subcore_axis_name,
    )

    # The output gets one extra row: the kernel's garbage scatter destination.
    out = core_map_helper.kernel(
        functools.partial(main_kernel, cfg=cfg),
        out_type=jax.ShapeDtypeStruct(
            (
                cfg.padded_input_size // reduce_group_size + 1,
                cfg.aligned_hidden_size,
            ),
            jnp.float32,
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

    # Step 5: Post-process the output (drop padding, zero empty groups, cast).
    out = out[: cfg.output_size, : cfg.hidden_size]
    # TODO(kyuyeunk): Use zero initialization instead of masking.
    out = jnp.where(mask[: cfg.output_size, None], out, jnp.zeros_like(out))
    return out.astype(x.dtype)
