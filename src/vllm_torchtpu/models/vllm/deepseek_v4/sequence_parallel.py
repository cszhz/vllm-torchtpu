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

Rank r owns rows [r * T / tp, (r + 1) * T / tp), where r is this core's
position in its TP group as the EP mesh lists it -- the order the
reduce-scatter produces and the all-gather consumes. The rank enters the graph
as data, so every rank traces and compiles the same program. The collectives
are JAX ops over the EP mesh, grouped by TP group (as the small TP all-reduce
is): vLLM's own all-gather allocates its output at a size derived from the
symbolic token count, which torch_tpu's export cannot lower.

The sequence-parallel steps run in a compiled module of their own
(model.DeepseekV4ModelSP over the same layers), chosen per step on the padded
bucket size outside both graphs; each graph compiles only its own buckets
(compilation/bucket_filter). Every DP rank runs the same bucket, so all EP
ranks take the same MoE op.
"""

from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
)

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Each rank's shard stays a multiple of a bf16 sublane tile.
ROW_ALIGNMENT = 16


def tp_world_size() -> int:
    if not model_parallel_is_initialized():
        return 1
    return get_tensor_model_parallel_world_size()


def enabled() -> bool:
    """Whether the sequence-parallel graph is built at all."""
    return bool(envs.TPU_DSV4_PREFILL_SP) and tp_world_size() > 1


def eligible(num_tokens: int) -> bool:
    """Whether a backbone step of ``num_tokens`` (padded) tokens is routed to
    the sequence-parallel graph. The same rule filters each graph's buckets."""
    if not enabled():
        return False
    tp = tp_world_size()
    minimum = max(int(envs.TPU_DSV4_PREFILL_SP_MIN_TOKENS), tp * ROW_ALIGNMENT)
    return bool(num_tokens >= minimum and num_tokens % (tp * ROW_ALIGNMENT) == 0)


# Built by `prebuild` at model load: the ops close over the EP mesh, which a
# compiled forward may not see.
_AG: Any = None
_RS: Any = None
_RANK: torch.Tensor | None = None


def _group_ops(mesh, groups: tuple[tuple[int, ...], ...]):
    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME

    spec = PartitionSpec(EP_AXIS_NAME)
    tp = len(groups[0])
    index_groups = [list(g) for g in groups]

    def shard_mapped(body, operands):
        return lambda *xs: jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * operands,
            out_specs=spec,
            check_vma=False,
        )(*xs)

    gather = shard_mapped(
        lambda b, _like: jax.lax.all_gather(
            b, EP_AXIS_NAME, axis=0, tiled=True, axis_index_groups=index_groups
        ),
        2,
    )
    scatter = shard_mapped(
        lambda b: jax.lax.psum_scatter(
            b,
            EP_AXIS_NAME,
            scatter_dimension=0,
            tiled=True,
            axis_index_groups=index_groups,
        ),
        1,
    )

    # Every operand annotated jax.Array: torch_tpu infers static argnums from
    # the signature and refuses an unannotated one. `like` only names the
    # gathered row count for the fake (it is not read).
    def ag(x: jax.Array, like: jax.Array) -> jax.Array:
        return gather(x, like)

    def rs(x: jax.Array) -> jax.Array:
        return scatter(x)

    ag_op = jax_op(
        "pallas::dsv4_sp_tp_ag",
        ag,
        mesh=mesh,
        input_partition_specs=(spec, spec),
        output_partition_specs=spec,
    )
    rs_op = jax_op(
        "pallas::dsv4_sp_tp_rs",
        rs,
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=spec,
    )
    # The default fakes export fn at the operands' shapes, which would pin the
    # caller's dynamic token dimension. The gather's row count is taken from
    # `like` (a [tokens] tensor of the step) so it is the step's own token
    # symbol, not an expression derived from it.
    ag_op.register_fake(lambda x, like: x.new_empty((like.shape[0], *x.shape[1:])))
    rs_op.register_fake(lambda x: x.new_empty((x.shape[0] // tp, *x.shape[1:])))
    return ag_op, rs_op


def prebuild() -> None:
    """Build the TP-group collectives and this core's rank, outside any traced
    region (the groups come from a collective). Idempotent."""
    global _AG, _RS, _RANK
    if _AG is not None:
        return
    from vllm_torchtpu.distributed.ep_mesh import (
        build_ep_mesh,
        ep_mesh_index,
        ep_token_replica_groups,
    )

    mesh = build_ep_mesh()
    if mesh is None:
        raise NotImplementedError(
            "TPU_DSV4_PREFILL_SP runs its collectives over the expert-parallel "
            "mesh, and this deployment has no expert-parallel group."
        )
    groups = ep_token_replica_groups()
    assert len(groups[0]) == tp_world_size(), (groups, tp_world_size())
    me = ep_mesh_index()
    (group,) = [g for g in groups if me in g]
    _RANK = torch.tensor([group.index(me)], dtype=torch.int32, device="tpu")
    _AG, _RS = _group_ops(mesh, groups)
    logger.info("DeepSeek-V4 sequence-parallel prefill armed | groups=%s", groups)


def rank_as_data() -> torch.Tensor:
    """This core's position in its TP group as an i32[1] device value."""
    assert _RANK is not None, "sequence_parallel.prebuild() did not run"
    return _RANK


def shard(x: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    """This rank's contiguous rows of a replicated [tokens, ...] tensor."""
    blocks = x.unflatten(0, (tp_world_size(), -1))
    return blocks.index_select(0, rank).squeeze(0)


def gather(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """All ranks' rows of a sharded tensor, in token order.

    `like` is any [tokens, ...] tensor of the step (e.g. `positions`): the
    result's row count is its size, so a gathered tensor lines up with the
    step's other tensors and a returned one has the step's token count rather
    than a size the runner's static compiled regions see as symbolic.
    """
    return _AG(x, like)


def reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """Sum TP partial outputs over the group; keep this rank's rows."""
    return _RS(x)
