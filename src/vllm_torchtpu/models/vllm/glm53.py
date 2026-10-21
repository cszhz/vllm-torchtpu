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
"""Explicit GLM-5.3 FP4-expert / FP8-linear checkpoint adapter for TPU."""

import re
from collections.abc import Iterable, Iterator

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM

from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.layers.adapter.quantization.mxfp4 import (
    VllmDeepseekV4Mxfp4MoEMethod,
    VllmMxfp4MoEMethod,
)

_EXPERT_SCALE = re.compile(r"\.experts\.\d+\.(?:gate|up|down)_proj\.scale$")


class _Glm53Mxfp4MoEMethod(VllmDeepseekV4Mxfp4MoEMethod):
    def _resolve_tpu_activation(self, layer: RoutedExperts) -> str:
        # The split MXFP4 checkpoint layout is shared with DeepSeek-V4;
        # GLM uses ordinary SiLU, without DeepSeek-V4's activation clamp.
        return VllmMxfp4MoEMethod._resolve_tpu_activation(self, layer)


class _Glm53Fp8Config(VllmFp8Config):
    is_scale_e8m0 = True

    def _get_checkpoint_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            return _Glm53Mxfp4MoEMethod(self.get_moe_config(layer))
        return super()._get_checkpoint_quant_method(layer, prefix)


def _map_checkpoint_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        if name.endswith(".scale"):
            suffix = (
                ".weight_scale" if _EXPERT_SCALE.search(name) else ".weight_scale_inv"
            )
            name = name.removesuffix(".scale") + suffix
        yield name, tensor


class Glm53MixedForCausalLM(GlmMoeDsaForCausalLM):
    """Opt-in checkpoint layout; the upstream GLM architecture is unchanged."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        hf_config = vllm_config.model_config.hf_config
        checkpoint = hf_config.quantization_config
        expected = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "expert_dtype": "fp4",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
            "activation_scheme": "dynamic",
        }
        if hf_config.model_type != "glm_moe_dsa" or hf_config.hidden_act != "silu":
            raise ValueError(
                "GLM-5.3 mixed quantization requires glm_moe_dsa with SiLU"
            )
        if not isinstance(checkpoint, dict) or any(
            checkpoint.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(
                f"Unsupported GLM-5.3 checkpoint quantization: {checkpoint}"
            )
        if checkpoint.get("ignored_layers") or checkpoint.get("modules_to_not_convert"):
            raise ValueError(
                "GLM-5.3 mixed quantization does not support ignored layers"
            )
        additional_config = vllm_config.additional_config
        if not isinstance(additional_config, dict):
            raise ValueError("GLM-5.3 additional_config must be a dictionary")
        if additional_config.get("tpu_online_fp8"):
            raise ValueError(
                "GLM-5.3 mixed quantization does not support tpu_online_fp8"
            )
        if vllm_config.speculative_config is not None:
            raise ValueError("GLM-5.3 mixed quantization does not support speculation")
        _Glm53Fp8Config.set_configs(vllm_config)
        quant_config = _Glm53Fp8Config.from_config(checkpoint)
        configure_quant_config(quant_config, type(self))
        vllm_config.quant_config = quant_config
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(_map_checkpoint_weights(weights))
