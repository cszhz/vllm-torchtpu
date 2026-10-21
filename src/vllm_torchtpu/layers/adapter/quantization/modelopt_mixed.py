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
"""ModelOpt MIXED_PRECISION quantization support for TPU.

Loads NVIDIA ModelOpt MIXED_PRECISION checkpoints (e.g. GLM-5.3-NVFP4) whose
``quantized_layers`` mapping assigns a per-layer ``quant_algo``: FP8
(per-tensor static) dense linears and NVFP4 (block-16) routed experts on this
checkpoint family; attention stays unquantized.

Checkpoint parsing (``from_config``, ``_resolve_quant_algo``,
``apply_vllm_mapper``) is inherited wholesale from upstream
``ModelOptMixedPrecisionConfig``; only the per-layer method dispatch is
overridden to hand layers to the existing TPU methods:

  - FP8 dense linear -> ``VllmFp8LinearMethodTPU`` with a static-activation
    ``VllmFp8Config``, so the checkpoint's per-tensor ``weight_scale`` /
    ``input_scale`` register and load with the standard FP8 contract.
  - NVFP4 routed experts -> ``VllmNvfp4MoEMethod`` (W4A16 GMM / opt-in W4A8
    fused EP), built on the checkpoint's group_size.
  - NVFP4 dense linear -> ``VllmNvfp4LinearMethod``.
  - Excluded or unmapped layers -> the TPU unquantized methods.

Algos without a TPU method (FP8 MoE, MXFP8, W4A16_NVFP4, anything unknown)
are rejected at the config/dispatch boundary rather than silently loaded as
BF16. All NVFP4 layers must share one group_size; the TPU MoE method serves a
single scale geometry, so a mixed-group checkpoint is rejected at
construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
)
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import (
    VllmNvfp4Config,
    VllmNvfp4LinearMethod,
    VllmNvfp4MoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.layers.core.quant_methods import (
    MODEL_OPT_MIXED,
    get_tpu_quant_method,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

# Algos with a TPU dense-linear method.
_SUPPORTED_LINEAR_ALGOS = ("FP8", "NVFP4")
# Algos with a TPU routed-experts method.
_SUPPORTED_MOE_ALGOS = ("NVFP4",)


@register_quantization_config(get_tpu_quant_method(MODEL_OPT_MIXED))
class VllmModelOptMixedConfig(ModelOptMixedPrecisionConfig, VllmQuantConfig):
    """TPU dispatch adapter over the upstream ModelOpt MIXED_PRECISION parser."""

    def __init__(
        self,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        quantized_layers: dict[str, dict[str, Any]],
        fp8_config: ModelOptFp8Config,
        nvfp4_config: ModelOptNvFp4Config,
        w4a16_nvfp4_config: ModelOptNvFp4Config,
        mxfp8_config: Any,
    ) -> None:
        super().__init__(
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            quantized_layers=quantized_layers,
            fp8_config=fp8_config,
            nvfp4_config=nvfp4_config,
            w4a16_nvfp4_config=w4a16_nvfp4_config,
            mxfp8_config=mxfp8_config,
        )
        algos: set[str] = set()
        nvfp4_group_sizes: set[int] = set()
        for info in quantized_layers.values():
            algo = str(info.get("quant_algo", "")).upper()
            algos.add(algo)
            if algo == "NVFP4":
                nvfp4_group_sizes.add(int(info.get("group_size", 16)))
        unsupported = algos - set(_SUPPORTED_LINEAR_ALGOS) - set(_SUPPORTED_MOE_ALGOS)
        if unsupported:
            raise ValueError(
                f"ModelOpt MIXED_PRECISION quant_algo(s) {sorted(unsupported)} "
                "are not supported on TPU. Supported dense-linear algos: "
                f"{list(_SUPPORTED_LINEAR_ALGOS)}; MoE algos: "
                f"{list(_SUPPORTED_MOE_ALGOS)}."
            )
        if len(nvfp4_group_sizes) > 1:
            raise ValueError(
                "ModelOpt MIXED_PRECISION NVFP4 layers must share one "
                f"group_size on TPU, got {sorted(nvfp4_group_sizes)}."
            )
        # Swap the upstream GPU-oriented sub-configs for the TPU variants.
        # Per-tensor FP8 checkpoints carry a static input_scale, so the dense
        # FP8 method must see activation_scheme="static" to register it.
        self.fp8_config = VllmFp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
            weight_block_size=None,
        )
        self.nvfp4_config = VllmNvfp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=kv_cache_quant_method,
            exclude_modules=[],
            group_size=nvfp4_config.group_size,
        )
        # No TPU W4A16_NVFP4 / MXFP8 methods; the construction-time validation
        # above rejects those algos, so these sub-configs are never dispatched.
        # pyrefly: ignore[bad-assignment]
        self.w4a16_nvfp4_config = None
        # pyrefly: ignore[bad-assignment]
        self.mxfp8_config = None

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return MODEL_OPT_MIXED

    @classmethod
    def get_min_capability(cls) -> int:
        # TPU-only config; the upstream SMxx capability gates do not apply.
        return 0

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, Attention):
            # ModelOpt kv-cache quantization is not implemented on TPU.
            return None

        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return VllmUnquantizedLinearMethod()
            if isinstance(layer, RoutedExperts):
                return VllmUnquantizedFusedMoEMethod(self.get_moe_config(layer))
            return None

        quant_algo = self._resolve_quant_algo(prefix)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if quant_algo is None:
                return VllmUnquantizedLinearMethod()
            if quant_algo == "FP8":
                return VllmFp8LinearMethodTPU(
                    self.fp8_config,
                    self.get_linear_config(layer),
                    prefix=prefix,
                )
            if quant_algo == "NVFP4":
                # TPU NVFP4 runs W4A16: bf16 activations, fp4 weights.
                return VllmNvfp4LinearMethod(
                    self.nvfp4_config, self.get_linear_config(layer)
                )
            raise ValueError(
                f"ModelOpt MIXED_PRECISION quant_algo {quant_algo!r} for "
                f"dense layer {prefix!r} is not supported on TPU."
            )

        if isinstance(layer, RoutedExperts):
            if quant_algo == "NVFP4":
                return VllmNvfp4MoEMethod(self.nvfp4_config, self.get_moe_config(layer))
            if quant_algo is None:
                return VllmUnquantizedFusedMoEMethod(self.get_moe_config(layer))
            raise ValueError(
                f"ModelOpt MIXED_PRECISION quant_algo {quant_algo!r} for "
                f"routed-experts layer {prefix!r} is not supported on TPU."
            )

        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        # The mapper rewrites exclude_modules and quantized_layers in place;
        # the TPU sub-configs carry no per-checkpoint prefix state.
