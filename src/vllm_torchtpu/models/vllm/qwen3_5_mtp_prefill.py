# SPDX-License-Identifier: Apache-2.0
"""Whole Qwen3.5 MoE MTP head on the last pipeline stage.

Backports the input-projection fix from vllm-project/vllm#46994 for the
pinned vLLM. This diagnostic only executes whole heads, never intermediate
MTP tensors, and currently admits TP1 only.
"""

import torch
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP
from vllm.sequence import IntermediateTensors


class Qwen3_5MoeMTPPrefill(Qwen3_5MoeMTP, SupportsPP):
    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if intermediate_tensors is not None or spec_step_idx != 0:
            raise NotImplementedError("MTP prefill accepts one complete step-0 head.")
        model = self.model
        if inputs_embeds is None:
            inputs_embeds = model.embed_input_ids(input_ids)
        hidden_states = model.fc(
            torch.cat(
                [
                    model.pre_fc_norm_embedding(inputs_embeds),
                    model.pre_fc_norm_hidden(hidden_states),
                ],
                dim=-1,
            )
        )
        layer = model.layers[0]
        if layer.use_attn_reduce_scatter_for_moe or model.num_mtp_layers != 1:
            raise NotImplementedError("MTP prefill requires one MTP layer and TP1.")
        hidden_states, residual = layer(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states, _ = model.norm(hidden_states, residual)
        return hidden_states
