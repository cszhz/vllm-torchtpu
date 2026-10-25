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
"""Sequence-parallel prefill for the DeepSeek-V4 backbone.

At prefill sizes every TP rank otherwise repeats the per-token work between
the attention blocks on all tokens: both mHC seams over the [tokens, hc,
hidden] residual streams, the norms, and the MoE gate. With
TPU_DSV4_PREFILL_SP=1 a backbone step of at least
TPU_DSV4_PREFILL_SP_MIN_TOKENS tokens keeps the streams sharded by token over
the TP group instead:

- the attention input is all-gathered before the attention block, whose
  projections and kernels are unchanged,
- the attention output (wo_b's TP partial sum) is reduce-scattered rather
  than all-reduced,
- the routed experts take this rank's rows directly (the fused EP op built
  with singleton token groups: no replica split, no restore all-gather), and
- the shared expert runs on the gathered rows and is reduce-scattered, or,
  with TPU_DSV4_PREFILL_SP_REPLICATED_SHARED=1, runs unsharded on this
  rank's rows with no collective at all.

The final hidden states are all-gathered once, after the head collapse and
the final norm.

Rank r owns rows [r * T / tp, (r + 1) * T / tp), the order the reduce-scatter
produces and the all-gather consumes. The rank enters the graph as data, so
every rank traces and compiles the same program.

The decision reads the token count while tracing, so it leaves a shape guard
and the backbone is re-traced per bucket group (compilation/shape_variants).
Every DP rank runs the same bucket, so all EP ranks take the same MoE op.
"""

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    model_parallel_is_initialized,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)

from vllm_torchtpu import envs

# Each rank's shard stays a multiple of a bf16 sublane tile.
ROW_ALIGNMENT = 16


def tp_world_size() -> int:
    if not model_parallel_is_initialized():
        return 1
    return get_tensor_model_parallel_world_size()


def eligible(num_tokens) -> bool:
    """Whether a backbone step of ``num_tokens`` tokens runs sequence-parallel.

    ``num_tokens`` must be a (possibly symbolic) tensor dimension: comparing it
    is what leaves the guard the per-bucket re-trace partitions on.
    """
    tp = tp_world_size()
    if not envs.TPU_DSV4_PREFILL_SP or tp <= 1:
        return False
    minimum = max(int(envs.TPU_DSV4_PREFILL_SP_MIN_TOKENS), tp * ROW_ALIGNMENT)
    return bool(num_tokens >= minimum and num_tokens % (tp * ROW_ALIGNMENT) == 0)


def rank_as_data(device: torch.device) -> torch.Tensor:
    """This rank's index in its TP group as an i32[1] device value."""
    # Shared with the indexer's kv-split, which needs the same property.
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_indexer import (  # noqa: E501
        tp_rank_as_data,
    )

    return tp_rank_as_data(get_tp_group(), device)


def shard(x: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    """This rank's contiguous rows of a replicated [tokens, ...] tensor."""
    blocks = x.unflatten(0, (tp_world_size(), -1))
    return blocks.index_select(0, rank).squeeze(0)


def gather(x: torch.Tensor) -> torch.Tensor:
    """All ranks' rows of a sharded tensor, in token order."""
    return tensor_model_parallel_all_gather(x, 0)


def reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """Sum TP partial outputs over the group; keep this rank's rows."""
    return tensor_model_parallel_reduce_scatter(x, 0)
