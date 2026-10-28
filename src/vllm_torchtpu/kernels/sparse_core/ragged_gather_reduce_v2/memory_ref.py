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
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import config

# scatter_meta fields per compacted row: flags, a 5-bit row source, and the
# row's output pair in the remaining high bits.
META_SAME_AS_PREV = 1
META_PAIR_FINAL = 2
# The row's token is the high (odd) half of its pair.
META_IS_ODD = 4
META_SRC_SHIFT = 3
META_SRC_MASK = 31
META_PAIR_SHIFT = 8
# Row sources: 0..15 a row of the chunk, else the low carry, a zero row, none.
# A pair-final row's source is its pair's low half; the last row of a chunk
# that ends inside a pair names the row to keep as that pair's low carry.
SRC_CARRY = 16
SRC_ZERO = 17
SRC_NONE = 31


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class IndexRef:
    partition_bounds: Any
    indices: Any
    sorted_by_validity: Any
    scatter_meta: Any
    sort_order: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DataRef:
    source: Any
    topk_weights: Any
    out: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ScratchRef:
    partition_bounds_vmem: Any
    prev_iter_last_row_vmem: Any
    sorted_by_validity_vmem: Any
    scatter_meta_vmem: Any
    src_indices_vmem: Any
    tw_f32_vmem: Any
    out_vmem: Any
    packed_vmem: Any
    pair_smem: Any
    zero_vmem: Any
    zero_ids_vmem: Any
    sem: Any

    @classmethod
    def create_scratch_types(cls, cfg: config.Config) -> Any:
        num_simd_lanes = cfg.sc_info.num_lanes
        window_vmem = pltpu.VMEM((cfg.window_size,), jnp.int32)
        return cls(
            partition_bounds_vmem=pltpu.VMEM((4 * num_simd_lanes,), jnp.int32),
            prev_iter_last_row_vmem=pltpu.VMEM(
                (cfg.num_col_chunks, cfg.col_chunk_size), jnp.float32
            ),
            sorted_by_validity_vmem=window_vmem,
            scatter_meta_vmem=window_vmem,
            src_indices_vmem=window_vmem,
            tw_f32_vmem=pltpu.VMEM((cfg.window_size,), jnp.float32),
            # Each chunk row's reduced value, then the low carries, a zero row.
            out_vmem=pltpu.VMEM(
                (cfg.num_reduced_rows, cfg.col_chunk_size), jnp.float32
            ),
            # The output words of a step's pair writes.
            packed_vmem=pltpu.VMEM((num_simd_lanes, cfg.col_chunk_size), jnp.uint32),
            # A step's pair writes (see kernel._SINGLE_ROW), then those in flight.
            pair_smem=pltpu.SMEM((6 * num_simd_lanes + 1,), jnp.int32),
            zero_vmem=pltpu.VMEM((num_simd_lanes, cfg.zero_cols), jnp.uint32),
            zero_ids_vmem=pltpu.VMEM(
                (config.ZERO_ID_WINDOW_CHUNKS * num_simd_lanes,), jnp.int32
            ),
            # recv, pair writes, empty-pair id gather, row-metadata gather,
            # empty-pair zero scatter.
            sem=pltpu.SemaphoreType.DMA((5,)),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class KernelRefs:
    index: IndexRef
    data: DataRef
    scratch: ScratchRef

    @classmethod
    def create(
        cls,
        scalar_ref: IndexRef,
        in_hbm_ref: Any,
        topk_weights_hbm_ref: Any,
        out_hbm_ref: Any,
        scratch_ref: ScratchRef,
    ) -> "KernelRefs":
        return cls(
            index=scalar_ref,
            data=DataRef(
                source=in_hbm_ref,
                topk_weights=topk_weights_hbm_ref,
                out=out_hbm_ref,
            ),
            scratch=scratch_ref,
        )
