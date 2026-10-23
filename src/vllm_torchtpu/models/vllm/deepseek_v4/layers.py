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
"""DeepSeek-V4 feed-forward and Manifold-Harmonized Coupling helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    quantized_matmul_f32,
)


# Pure PyTorch head projection for MHC manifold collapse;
# upstream vLLM only provides Triton/TileLang GPU kernels for hc_head.
def mhc_collapse_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Collapse multi-stream residual manifold into a single vector before LM head."""
    hc_mult, hidden_size = hidden_states.shape[-2:]
    assert hc_mult == 4, f"stream sum order is written for 4 hc streams, got {hc_mult}"
    streams_flat = hidden_states.flatten(-2)
    residual_flat = streams_flat.float()
    residual_norm = residual_flat * torch.rsqrt(
        residual_flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps
    )
    pre_mix = torch.nn.functional.linear(residual_norm, hc_fn)
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    # Weigh lane-aligned slices of the flat bf16 rows: a sum over the stream
    # axis makes XLA relayout an f32 [T, 4, H] copy first. (0+2)+(1+3) is the
    # order XLA's sublane-tree reduce adds the 4 streams in, so the result is
    # bit-identical to torch.sum(dim=-2).
    terms = [
        pre_mix[..., j : j + 1]
        * streams_flat[..., j * hidden_size : (j + 1) * hidden_size].float()
        for j in range(hc_mult)
    ]
    return ((terms[0] + terms[2]) + (terms[1] + terms[3])).to(hidden_states.dtype)


class DeepseekV4MLP(nn.Module):
    """DeepSeek-V4 feed-forward layer for dense/shared expert projections."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

    def forward_unrounded(self, x: torch.Tensor) -> torch.Tensor:
        """``forward`` with the down projection's f32 product left unrounded.

        For a caller that adds it to another output and rounds the sum once.
        The down projection must be the per-channel FP8 linear (single TP
        rank), whose ``apply`` is ``quantized_matmul`` on these tensors.
        """
        down = self.down_proj
        assert down.tp_size == 1 and down.bias is None
        weight = down.weight
        if getattr(down, KEEP_VLLM_LAYOUT_ATTR, False):
            weight = weight.transpose(0, 1)
        gate_up, _ = self.gate_up_proj(x)
        return quantized_matmul_f32(self.act_fn(gate_up), weight, down.weight_scale)
