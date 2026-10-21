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
"""MOE_EXPERT_COUNT_TAP_DIR: per-layer routed-expert counts from a live server.

An expert placement is only as good as the routing statistics it is built
from, and the fused EP kernel selects inside its own program, so its choices
never reach the host. With the tap on, every MoE layer re-runs the same
selection in the traced forward (`moe_routing.select_experts`: hash table or
scored top-k) and accumulates, on the device, per global expert:

    row 0: routed pairs           sum over steps of c
    row 1: squared per-step count sum over steps of c * c
    row 2: [steps, valid tokens, 0, ...]

where c is this rank's per-step count of valid (non-padding) tokens that
selected the expert. Rows 0-1 give the mean and the step-to-step variance of
each expert's load from this rank. Every MOE_EXPERT_COUNT_TAP_EVERY steps the
runner copies the buffers to the host and writes one cumulative snapshot:

    <dir>/counts_ep<rank>_<seq>.npz   counts [layers, 3, experts] i32,
                                      layer_names, step, wall

Snapshots are cumulative so a reader can difference any two of them to drop
warmup and compile traffic. The tap only reads the routing inputs: the model's
outputs are unchanged, but the forward does extra work, so a tapped run is a
diagnostic, not a benchmark arm.
"""

import os
import time

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

COUNTS_ATTR = "_tpu_expert_counts"

# (layer name, layer) for every tapped layer of this process, in registration
# order. The layers own the buffers; this only lists them for the dump.
_TAPPED: list[tuple[str, torch.nn.Module]] = []
_STATE = {"steps": 0, "seq": 0}


def enabled() -> bool:
    return bool(envs.MOE_EXPERT_COUNT_TAP_DIR)


def _directory() -> str:
    # Pod specs pass env values verbatim, e.g. "$CDK_OUTPUT_DIR/moe_counts".
    directory = os.path.expandvars(envs.MOE_EXPERT_COUNT_TAP_DIR)
    if "$" in directory:
        raise ValueError(
            f"MOE_EXPERT_COUNT_TAP_DIR={envs.MOE_EXPERT_COUNT_TAP_DIR!r} references "
            "an unset variable"
        )
    return directory


def register(layer) -> None:
    """Give a RoutedExperts layer its count buffer. Call after weight load."""
    if not enabled():
        return
    _directory()  # an unresolvable directory fails at load, not hours later
    experts = layer.global_num_experts
    layer.register_buffer(
        COUNTS_ATTR,
        torch.zeros((3, experts), dtype=torch.int32, device=layer.w13_weight.device),
        persistent=False,
    )
    _TAPPED.append((layer.layer_name, layer))


def record(layer, router_logits: torch.Tensor, input_ids) -> None:
    """Accumulate this step's selections into the layer's buffer (traced)."""
    counts = getattr(layer, COUNTS_ATTR, None)
    if counts is None:
        return
    _, topk_ids = moe_routing.select_experts(
        hidden_states=router_logits,
        router_logits=router_logits,
        topk=layer.moe_config.experts_per_token,
        renormalize=layer.renormalize,
        scoring_fn=layer.scoring_func,
        layer=layer,
        input_ids=input_ids,
    )
    tokens = router_logits.shape[0]
    experts = counts.shape[1]
    device = router_logits.device
    padding = token_padding.local_padding_mask(tokens)
    valid = (
        torch.ones((tokens,), dtype=torch.bool, device=device)
        if padding is None
        else ~padding
    )
    hit = topk_ids.long().unsqueeze(-1) == torch.arange(experts, device=device)
    per_step = (hit & valid[:, None, None]).sum(dim=(0, 1), dtype=torch.int32)
    meta = torch.cat(
        [
            torch.ones((1,), dtype=torch.int32, device=device),
            valid.sum(dtype=torch.int32).reshape(1),
            torch.zeros((experts - 2,), dtype=torch.int32, device=device),
        ]
    )
    counts.copy_(counts + torch.stack([per_step, per_step * per_step, meta]))


def maybe_dump(ep_rank: int) -> None:
    """Called once per runner step on the host; writes every N steps."""
    if not _TAPPED:
        return
    _STATE["steps"] += 1
    if _STATE["steps"] % envs.MOE_EXPERT_COUNT_TAP_EVERY:
        return
    import numpy as np

    host = torch.stack([getattr(layer, COUNTS_ATTR) for _, layer in _TAPPED]).cpu()
    directory = _directory()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"counts_ep{ep_rank}_{_STATE['seq']:04d}.npz")
    np.savez(
        path,
        counts=host.numpy(),
        layer_names=np.array([name for name, _ in _TAPPED]),
        step=_STATE["steps"],
        wall=time.time(),
    )
    _STATE["seq"] += 1
    logger.info_once("Expert count tap writing snapshots to %s", directory)
