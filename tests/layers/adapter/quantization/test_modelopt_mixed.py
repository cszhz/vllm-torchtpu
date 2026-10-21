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
"""Tests for TPU ModelOpt MIXED_PRECISION dispatch (GLM-5.3-NVFP4 format).

Covers the real checkpoint format end to end at config level:
``from_config`` parsing, prefix -> algo resolution (direct, fused
``gate_up_proj``, routed-experts parent prefix), per-layer method dispatch
(FP8 dense / NVFP4 MoE / unquantized attention / excluded layers), and
rejection of unsupported algos. Also verifies the dense FP8 path keeps the
checkpoint's per-tensor scales numerically intact through
``VllmFp8LinearMethodTPU.process_weights_after_loading`` and that the NVFP4
MoE method registers the packed-weight parameter contract.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
)
from vllm_torchtpu.layers.adapter.quantization.modelopt_mixed import (
    VllmModelOptMixedConfig,
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

# Mirrors nvidia/GLM-5.3-NVFP4's quantization_config shape (trimmed to the
# layers the tests exercise): FP8 dense MLP on the first decoder layers,
# NVFP4 group-16 routed experts afterwards, kv-cache FP8, layer 78 ignored.
GLM53_QUANTIZATION_CONFIG = {
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
    "ignore": ["model.layers.78*"],
    "packed_modules_mapping": {"gate_up_proj": ["gate_proj", "up_proj"]},
    "quantized_layers": {
        "model.layers.0.mlp.gate_proj": {"quant_algo": "FP8"},
        "model.layers.0.mlp.up_proj": {"quant_algo": "FP8"},
        "model.layers.0.mlp.down_proj": {"quant_algo": "FP8"},
        "model.layers.1.mlp.gate_proj": {"quant_algo": "FP8"},
        "model.layers.1.mlp.up_proj": {"quant_algo": "FP8"},
        "model.layers.1.mlp.down_proj": {"quant_algo": "FP8"},
        "model.layers.2.mlp.gate_proj": {"quant_algo": "FP8"},
        "model.layers.2.mlp.up_proj": {"quant_algo": "FP8"},
        "model.layers.2.mlp.down_proj": {"quant_algo": "FP8"},
        "model.layers.3.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
        "model.layers.4.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
    },
}


class FakeRoutedExperts(RoutedExperts):
    """Bare RoutedExperts for dispatch tests, matching test_nvfp4.py's pattern."""

    def __init__(self, num_experts: int = 4):
        torch.nn.Module.__init__(self)
        self.moe_config = MagicMock(has_bias=False, is_act_and_mul=True)
        self.moe_config.num_experts = num_experts
        self.moe_config.moe_parallel_config = MagicMock()
        self.moe_config.moe_parallel_config.use_ep = False


def make_linear_layer(input_size: int = 64, output_size: int = 32) -> LinearBase:
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.input_size = input_size
    layer.output_size = output_size
    return layer


def make_config() -> VllmModelOptMixedConfig:
    config = VllmModelOptMixedConfig.from_config(GLM53_QUANTIZATION_CONFIG)
    config.vllm_config = MagicMock()
    config.vllm_config.parallel_config.enable_expert_parallel = False
    return config


@pytest.mark.tpu_test
class TestModelOptMixedFromConfig:
    def test_get_name(self):
        assert VllmModelOptMixedConfig.get_name() == "modelopt_mixed"

    def test_from_config_parses_checkpoint_fields(self):
        config = make_config()
        assert config.kv_cache_quant_method == "FP8"
        assert config.exclude_modules == ["model.layers.78*"]
        assert len(config.quantized_layers) == 11
        # Sub-configs are the TPU variants, not the upstream GPU ones.
        assert isinstance(config.fp8_config, VllmFp8Config)
        assert isinstance(config.nvfp4_config, VllmNvfp4Config)
        # The checkpoint's dense FP8 is per-tensor static: the method must
        # register input_scale, which requires the static activation scheme.
        assert config.fp8_config.activation_scheme == "static"
        assert config.fp8_config.is_checkpoint_fp8_serialized
        # group_size is derived from the first NVFP4 quantized_layers entry.
        assert config.nvfp4_config.group_size == 16
        assert config.mxfp8_config is None
        # Method construction receives the outer model configuration explicitly.
        method = config.get_quant_method(
            make_linear_layer(), "model.layers.0.mlp.down_proj"
        )
        assert method.linear_config.vllm_config is config.vllm_config

    def test_from_config_missing_quantized_layers_rejected(self):
        bad = dict(GLM53_QUANTIZATION_CONFIG, quantized_layers={})
        with pytest.raises(ValueError, match="quantized_layers"):
            VllmModelOptMixedConfig.from_config(bad)

    def test_from_config_unsupported_algo_rejected(self):
        bad = dict(GLM53_QUANTIZATION_CONFIG)
        bad["quantized_layers"] = {
            **GLM53_QUANTIZATION_CONFIG["quantized_layers"],
            "model.layers.5.mlp.experts": {"quant_algo": "MXFP8"},
        }
        with pytest.raises(ValueError, match="MXFP8"):
            VllmModelOptMixedConfig.from_config(bad)

    def test_from_config_unknown_algo_rejected(self):
        bad = dict(GLM53_QUANTIZATION_CONFIG)
        bad["quantized_layers"] = {
            **GLM53_QUANTIZATION_CONFIG["quantized_layers"],
            "model.layers.5.mlp.gate_proj": {"quant_algo": "AWQ_W4"},
        }
        with pytest.raises(ValueError, match="AWQ_W4"):
            VllmModelOptMixedConfig.from_config(bad)

    def test_from_config_fp8_moe_rejected(self):
        """FP8 routed experts have no TPU method under MIXED_PRECISION; they
        must fail at construction, not load silently."""
        bad = dict(GLM53_QUANTIZATION_CONFIG)
        bad["quantized_layers"] = {
            **GLM53_QUANTIZATION_CONFIG["quantized_layers"],
            "model.layers.5.mlp.experts": {"quant_algo": "FP8_PB_WO"},
        }
        with pytest.raises(ValueError, match="FP8_PB_WO"):
            VllmModelOptMixedConfig.from_config(bad)

    def test_from_config_mixed_nvfp4_group_sizes_rejected(self):
        """A second group_size must not silently reuse the first group's
        scale geometry."""
        bad = dict(GLM53_QUANTIZATION_CONFIG)
        bad["quantized_layers"] = {
            **GLM53_QUANTIZATION_CONFIG["quantized_layers"],
            "model.layers.5.mlp.experts": {"quant_algo": "NVFP4", "group_size": 32},
        }
        with pytest.raises(ValueError, match="group_size"):
            VllmModelOptMixedConfig.from_config(bad)


@pytest.mark.tpu_test
class TestModelOptMixedDispatch:
    def test_fp8_dense_linear(self):
        config = make_config()
        layer = make_linear_layer()
        method = config.get_quant_method(layer, "model.layers.0.mlp.down_proj")
        assert isinstance(method, VllmFp8LinearMethodTPU)
        assert method.quant_config is config.fp8_config
        assert method.act_q_static
        assert not method.block_quant

    def test_fp8_fused_gate_up_proj_resolves_shards(self):
        """vLLM packs gate_proj/up_proj into gate_up_proj; the algo must
        resolve through the checkpoint's packed_modules_mapping."""
        config = make_config()
        config.packed_modules_mapping = GLM53_QUANTIZATION_CONFIG[
            "packed_modules_mapping"
        ]
        layer = make_linear_layer()
        method = config.get_quant_method(layer, "model.layers.2.mlp.gate_up_proj")
        assert isinstance(method, VllmFp8LinearMethodTPU)

    def test_nvfp4_dense_linear(self):
        config = make_config()
        config.quantized_layers["model.layers.9.mlp.down_proj"] = {
            "quant_algo": "NVFP4",
            "group_size": 16,
        }
        layer = make_linear_layer()
        method = config.get_quant_method(layer, "model.layers.9.mlp.down_proj")
        assert isinstance(method, VllmNvfp4LinearMethod)
        assert method.group_size == 16

    def test_nvfp4_routed_experts(self):
        config = make_config()
        layer = FakeRoutedExperts()
        method = config.get_quant_method(layer, "model.layers.3.mlp.experts")
        assert isinstance(method, VllmNvfp4MoEMethod)
        assert method.quant_config is config.nvfp4_config
        assert method.group_size == 16

    def test_excluded_layer_is_unquantized(self):
        config = make_config()
        layer = make_linear_layer()
        method = config.get_quant_method(layer, "model.layers.78.mlp.down_proj")
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_unlisted_linear_is_unquantized(self):
        """A linear absent from quantized_layers (e.g. attention projections
        on this checkpoint) stays BF16 rather than guessing an algo."""
        config = make_config()
        layer = make_linear_layer()
        method = config.get_quant_method(layer, "model.layers.0.self_attn.qkv_proj")
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_attention_returns_none(self):
        config = make_config()
        layer = Attention.__new__(Attention)
        torch.nn.Module.__init__(layer)
        assert config.get_quant_method(layer, "model.layers.0.self_attn") is None

    def test_excluded_moe_is_unquantized(self, vllm_config_context):
        config = make_config()
        config.exclude_modules = ["model.layers.3.mlp.experts"]
        layer = FakeRoutedExperts()
        method = config.get_quant_method(layer, "model.layers.3.mlp.experts")
        assert isinstance(method, VllmUnquantizedFusedMoEMethod)

    def test_other_module_returns_none(self):
        config = make_config()
        assert config.get_quant_method(torch.nn.LayerNorm(8), "model.norm") is None


@pytest.mark.tpu_test
class TestModelOptMixedFp8DenseNumerics:
    """The dense FP8 path must preserve the checkpoint's per-tensor scales:
    runtime_dequant(weight, scale) == checkpoint_weight * weight_scale."""

    def test_per_tensor_scale_numerically_equivalent(self):
        out_dim, in_dim = 64, 128
        checkpoint_scale = 2.5

        weight_fp8 = torch.ones(out_dim, in_dim, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        layer = torch.nn.Module()
        layer.logical_widths = None
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            torch.tensor([checkpoint_scale]), requires_grad=False
        )
        layer.input_scale = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)

        config = make_config()
        method = VllmFp8LinearMethodTPU(config.fp8_config, prefix="test")
        method.process_weights_after_loading(layer)

        # Runtime storage is (k, n) with a per-output-channel fp32 scale.
        assert layer.weight.dtype == torch.float8_e4m3fn
        runtime_deq = layer.weight.float() * layer.weight_scale[None, :]
        expected = weight_fp8.float() * checkpoint_scale
        assert torch.allclose(runtime_deq, expected.t(), rtol=0.06, atol=0.05)

    def test_mixed_config_builds_static_fp8_method(self):
        """Dispatch through the mixed config (not the class directly) yields a
        method whose create/process contract registers and consumes
        weight_scale/input_scale."""
        config = make_config()
        layer = make_linear_layer()
        method = config.get_quant_method(layer, "model.layers.1.mlp.down_proj")
        assert isinstance(method, VllmFp8LinearMethodTPU)
        assert method.act_q_static
        assert method.quant_config.is_checkpoint_fp8_serialized


@pytest.mark.tpu_test
class TestModelOptMixedNvfp4Moe:
    def test_create_weights_registers_nvfp4_params(self):
        """The MoE method must register the packed uint8 weight plus the
        block scale / global scale contract ModelOptNvFp4FusedMoE declares."""
        config = make_config()
        layer = FakeRoutedExperts(num_experts=2)
        method = config.get_quant_method(layer, "model.layers.3.mlp.experts")
        assert isinstance(method, VllmNvfp4MoEMethod)

        num_experts, hidden, inter = 2, 64, 32
        # ModelWeightParameter reads the TP rank at construction; the real
        # parameter creation must still run.
        with (
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
                lambda: 0,
            ),
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
                lambda: 1,
            ),
        ):
            method.create_weights(
                layer,
                num_experts,
                hidden,
                inter,
                torch.bfloat16,
            )
        # Packed fp4: 2 items per byte along the contracting dim.
        assert layer.w13_weight.shape == (num_experts, 2 * inter, hidden // 2)
        assert layer.w13_weight.dtype == torch.uint8
        assert layer.w2_weight.shape == (num_experts, hidden, inter // 2)
        # Block-16 e4m3 scales + per-tensor fp32 globals.
        assert layer.w13_weight_scale.shape == (num_experts, 2 * inter, hidden // 16)
        assert layer.w13_weight_scale.dtype == torch.float8_e4m3fn
        assert layer.w2_weight_scale.shape == (num_experts, hidden, inter // 16)
        # act_and_mul experts keep one global scale per shard (w1, w3).
        assert layer.w13_weight_scale_2.shape == (num_experts, 2)
        assert layer.w13_weight_scale_2.dtype == torch.float32
        assert layer.w2_weight_scale_2.shape == (num_experts,)
        assert layer.w2_weight_scale_2.dtype == torch.float32
