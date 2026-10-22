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
"""Torch bridge for the DP + EP MoE layer that routes before it dispatches.

vLLM's naive DP/EP dispatch all-gathers each rank's bf16 hidden states and
router logits ahead of the MoE, and every rank then selects the top-k of every
gathered token. An armed layer skips that dispatch: each rank routes its own
tokens, and one SPMD op over the EP mesh gathers the expert kernel's input rows
(quantized where the routed kernel serves the batch) with their expert ids and
weights (`kernels/megablox/moe_local_dispatch.py`). A second op runs this rank's
experts over the gathered rows, and the runner's reduce-scatter combines them as
before. The runner's dispatch is skipped for armed layers
(`vllm_torchtpu._patch_moe_runner_local_dispatch`).
"""

import functools
from dataclasses import dataclass
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
)
from vllm_torchtpu.kernels.megablox.moe_local_dispatch import (
    META_LANES,
    max_dispatch_topk,
)
from vllm_torchtpu.kernels.megablox.moe_routed_ffn import MAX_TOPK
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
    enable_pipelined_collective_and_compute,
)
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_dispatched_func,
    fused_moe_local_dispatch_func,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Recorded on the layer's quant method by `prebuild_fused_moe_local_dispatch`.
FUSED_MOE_LOCAL_DISPATCH_ATTR = "_fused_moe_local_dispatch"

# One pair of ops per closure, shared by every layer it serves: torch_tpu
# exports and compiles per op object, and two closures under one qualname would
# share a dispatch entry.
_OPS: dict[tuple, "ArmedLocalDispatch"] = {}


@dataclass(frozen=True)
class ArmedLocalDispatch:
    dispatch: Any
    experts: Any


def fused_moe_local_dispatch_supported(owner: Any) -> bool:
    """Whether `prebuild_fused_moe_local_dispatch` armed this quant method's layer."""
    return getattr(owner, FUSED_MOE_LOCAL_DISPATCH_ATTR, None) is not None


def _build_ops(mesh, *, topk: int, activation: str) -> ArmedLocalDispatch:
    num_shards = mesh.shape[EP_AXIS_NAME]
    use_sparse_core = envs.USE_MOE_SPARSE_CORE
    onehot_threshold = resolve_onehot_permute_threshold()
    key = (mesh, topk, activation, use_sparse_core, onehot_threshold)
    cached = _OPS.get(key)
    if cached is not None:
        return cached
    routing_kw = dict(
        topk=topk,
        activation=activation,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=onehot_threshold,
    )

    body = functools.partial(
        fused_moe_local_dispatch_func,
        **routing_kw,
        num_shards=num_shards,
        axis_name=EP_AXIS_NAME,
    )
    spec = PartitionSpec(EP_AXIS_NAME)
    specs = (spec,) * 7
    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=spec, check_vma=False
    )

    # Every operand is annotated: torch_tpu infers static argnums from the
    # signature. Each is this rank's shard: its own tokens and their routes,
    # and its experts, read for their shapes.
    def dispatch(
        hidden_states: jax.Array,
        topk_ids: jax.Array,
        topk_weights: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
    ) -> jax.Array:
        return sharded(
            hidden_states, topk_ids, topk_weights, w1, w2, w1_scale, w2_scale
        )

    suffix = len(_OPS)
    dispatch_op = jax_op(
        f"pallas::fused_moe_local_dispatch_{suffix}",
        dispatch,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=spec,
    )

    def _dispatch_fake(hidden_states, *_args):
        return hidden_states.new_empty(
            (hidden_states.shape[0] * num_shards, hidden_states.shape[1] + META_LANES)
        )

    dispatch_op.register_fake(_dispatch_fake)

    experts_op = jax_op(
        f"pallas::fused_moe_dispatched_{suffix}",
        functools.partial(fused_moe_dispatched_func, **routing_kw),
    )

    def _experts_fake(rows, *_args):
        return rows.new_empty((rows.shape[0], rows.shape[1] - META_LANES))

    experts_op.register_fake(_experts_fake)
    armed = ArmedLocalDispatch(dispatch=dispatch_op, experts=experts_op)
    _OPS[key] = armed
    return armed


def prebuild_fused_moe_local_dispatch(
    layer, *, topk: int, activation: str
) -> ArmedLocalDispatch | None:
    """Arm route-before-dispatch for one layer, or None where it cannot serve it.

    Called from `process_weights_after_loading`: resolving the mesh is a
    collective, which a compiled forward cannot run. Arms only plain DP + EP
    (no TP, PCP or sequence-parallel MoE), where the DP all-gather lays out
    tokens in mesh order, so the gathered rows and the runner's reduce-scatter
    agree, and where routing is row-local.
    """
    cfg = layer.moe_config.moe_parallel_config
    if not (
        cfg.use_ep
        and cfg.dp_size > 1
        and cfg.tp_size == 1
        and cfg.pcp_size == 1
        and not cfg.is_sequence_parallel
        and envs.USE_MOE_SPARSE_CORE
        and not envs.TPU_MOE_SKIP_PADDED_TOKENS
        and not enable_pipelined_collective_and_compute()
        and moe_routing._SIMULATION_STRATEGY is None
        and topk <= min(MAX_TOPK, max_dispatch_topk())
        and getattr(layer, "w13_bias", None) is None
        and getattr(layer, "w2_bias", None) is None
        and layer.moe_config.hidden_dim == layer.w13_weight.shape[1]
    ):
        return None
    mesh = build_ep_mesh()
    if mesh is None:
        return None
    num_shards = mesh.shape[EP_AXIS_NAME]
    shard = ep_mesh_index()
    local_experts = layer.w13_weight.shape[0]
    if (
        num_shards != cfg.dp_size
        or shard != cfg.dp_rank
        or shard != cfg.ep_rank
        or layer.global_num_experts != local_experts * num_shards
        or moe_routing.get_experts_start(layer) != shard * local_experts
    ):
        logger.info_once(
            "Route-before-dispatch MoE not engaged: DP ranks, EP mesh and expert "
            "blocks do not line up (dp=%d/%d, ep=%d, mesh=%d/%d, experts=%d x %d).",
            cfg.dp_rank,
            cfg.dp_size,
            cfg.ep_rank,
            shard,
            num_shards,
            local_experts,
            layer.global_num_experts // max(local_experts, 1),
        )
        return None
    logger.info_once(
        "Route-before-dispatch MoE armed: each of %d DP ranks routes its own "
        "tokens and one all-gather carries the expert inputs and routes.",
        num_shards,
    )
    return _build_ops(mesh, topk=topk, activation=activation)


def fused_moe_local_dispatch(
    owner: Any,
    layer,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """This rank's experts over every DP rank's tokens, rank-major.

    Takes this rank's own tokens and their routes, which the runner did not
    dispatch; returns what the naive dispatch's expert step returns, for the
    runner's reduce-scatter.
    """
    armed: ArmedLocalDispatch = getattr(owner, FUSED_MOE_LOCAL_DISPATCH_ATTR)
    w1, w2 = layer.w13_weight, layer.w2_weight
    w1_scale, w2_scale = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    rows = armed.dispatch(
        hidden_states, topk_ids, topk_weights, w1, w2, w1_scale, w2_scale
    )
    return armed.experts(rows, w1, w2, w1_scale, w2_scale, layer._experts_start)
