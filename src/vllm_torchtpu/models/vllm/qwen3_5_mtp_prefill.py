# SPDX-License-Identifier: Apache-2.0
"""Whole Qwen3Next / Qwen3.5 MoE MTP head on the last pipeline stage.

Backports the input-projection fix from vllm-project/vllm#46994 for the
pinned vLLM. This diagnostic only executes whole heads, never intermediate
MTP tensors, and currently admits TP1 only.
"""

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP
from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMTP
from vllm.sequence import IntermediateTensors


class QwenMTPPrefillMixin:
    """Run the complete one-layer head on the target's last PP rank."""

    _prefill_only = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        from vllm_torchtpu import envs

        # The next MRO entry is the concrete upstream Qwen MTP, not object.
        # pyrefly: ignore[unexpected-keyword]
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._prefill_only = envs.TPU_MTP_PREFILL_ONLY

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
        if not self._prefill_only:
            # pyrefly: ignore[missing-attribute]
            return super().forward(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                spec_step_idx=spec_step_idx,
            )
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


class Qwen3_5MoeMTPPrefill(QwenMTPPrefillMixin, Qwen3_5MoeMTP, SupportsPP):
    """Qwen3.5 MoE head with last-stage prefill execution."""


class Qwen3NextMTPPrefill(QwenMTPPrefillMixin, Qwen3NextMTP, SupportsPP):
    """Qwen3Next head with last-stage prefill execution."""
