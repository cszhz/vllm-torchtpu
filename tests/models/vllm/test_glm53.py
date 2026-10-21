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
"""GLM-5.3 checkpoint semantics and opt-in model registration (CPU only)."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM

from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8LinearMethodTPU,
    _dequantize_fp8_linear,
)
from vllm_torchtpu.models.vllm.glm53 import (
    Glm53MixedForCausalLM,
    _Glm53Fp8Config,
    _Glm53Mxfp4MoEMethod,
    _map_checkpoint_weights,
)


def checkpoint_config(**changes):
    config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "expert_dtype": "fp4",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
    }
    config.update(changes)
    return config


@pytest.mark.parametrize(
    "source,target",
    [
        (f"mlp.experts.12.{proj}.scale", f"mlp.experts.12.{proj}.weight_scale")
        for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    + [
        (
            "mlp.shared_experts.gate_proj.scale",
            "mlp.shared_experts.gate_proj.weight_scale_inv",
        ),
        ("self_attn.indexer.wq_b.scale", "self_attn.indexer.wq_b.weight_scale_inv"),
        ("self_attn.q_b_proj.scale", "self_attn.q_b_proj.weight_scale_inv"),
        ("mlp.gate.weight", "mlp.gate.weight"),
    ],
)
def test_checkpoint_name_mapping_preserves_tensor(source, target):
    tensor = torch.ones(1)
    result = list(_map_checkpoint_weights([(f"model.layers.7.{source}", tensor)]))
    assert result[0][0] == f"model.layers.7.{target}"
    assert result[0][1] is tensor


def test_signed_packed_experts_and_scale_bytes():
    """All I8 byte patterns survive the U8 container before CPU dequantization."""
    moe = SimpleNamespace(w13_num_shards=2, has_bias=False)
    method = _Glm53Mxfp4MoEMethod(moe)
    layer = torch.nn.Module()
    method.create_weights(
        layer, 1, 256, 128, torch.bfloat16, weight_loader=lambda *a, **kw: None
    )
    packed_i8 = torch.arange(256, dtype=torch.int16).to(torch.int8).repeat(64)
    with torch.no_grad():
        layer.w13_weight.copy_(packed_i8.repeat(2).reshape(1, 256, 128))
        layer.w2_weight.copy_(packed_i8.reshape(1, 256, 64))
        layer.w13_weight_scale.fill_(126)  # E8M0 byte code for 0.5.
        layer.w2_weight_scale.fill_(129)  # E8M0 byte code for 4.0.
    assert layer.w13_weight.dtype == layer.w2_weight.dtype == torch.uint8
    assert torch.equal(layer.w2_weight.flatten(), packed_i8.view(torch.uint8))
    w13, w2, _, _, intermediate = method._dequantize_and_pad(layer)
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )

    def reference(packed):
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        return values[codes.long()]

    torch.testing.assert_close(
        w13[:, :256], (reference(layer.w13_weight) * 0.5).transpose(1, 2).bfloat16()
    )
    torch.testing.assert_close(
        w2[:, :, :256], (reference(layer.w2_weight) * 4).transpose(1, 2).bfloat16()
    )
    assert intermediate == 128
    layer.activation = "silu"
    assert method._resolve_tpu_activation(layer) == "silu"


def test_dense_scale_is_e8m0_value_not_byte_code(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        SimpleNamespace(world_size=1, rank_in_group=0),
    )
    quant = _Glm53Fp8Config.from_config(checkpoint_config())
    method = VllmFp8LinearMethodTPU(quant)
    layer = torch.nn.Module()
    method.create_weights(layer, 256, [256], 256, 256, torch.bfloat16)
    assert layer.weight_scale_inv.dtype == torch.float8_e8m0fnu
    scales = torch.tensor([[0.5, 1], [2, 4]], dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(torch.ones(256, 256).to(torch.float8_e4m3fn))
        layer.weight_scale_inv.copy_(scales.to(torch.float8_e8m0fnu))
    actual = _dequantize_fp8_linear(
        layer.weight,
        weight_scale=None,
        weight_scale_inv=layer.weight_scale_inv,
        block_quant=True,
        weight_block_size=(128, 128),
        out_dtype=torch.float32,
    )
    torch.testing.assert_close(
        actual, scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    )


@pytest.mark.parametrize(
    "changes", [{"fmt": "e5m2"}, {"expert_dtype": "fp8"}, {"scale_fmt": "fp32"}]
)
def test_unsupported_checkpoint_fails_at_model_boundary(changes):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="glm_moe_dsa",
                hidden_act="silu",
                quantization_config=checkpoint_config(**changes),
            )
        ),
        additional_config={},
        speculative_config=None,
    )
    with pytest.raises(ValueError, match="checkpoint quantization"):
        Glm53MixedForCausalLM(vllm_config=config)


def test_registration_keeps_upstream_glm_and_config_hook(monkeypatch):
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models import config as model_config

    from vllm_torchtpu.models.vllm import register_models
    from vllm_torchtpu.offload.raiden_store import has_multi_shapes_kv_caches

    resolve_config = SimpleNamespace(model_impl="vllm")
    old = ModelRegistry.resolve_model_cls(["GlmMoeDsaForCausalLM"], resolve_config)[0]
    monkeypatch.setattr(ModelRegistry, "models", ModelRegistry.models.copy())
    monkeypatch.setattr(
        model_config, "MODELS_CONFIG_MAP", model_config.MODELS_CONFIG_MAP.copy()
    )
    register_models()
    assert (
        ModelRegistry.resolve_model_cls(["GlmMoeDsaForCausalLM"], resolve_config)[0]
        is old
        is GlmMoeDsaForCausalLM
    )
    assert (
        ModelRegistry.resolve_model_cls(["Glm53MixedForCausalLM"], resolve_config)[0]
        is Glm53MixedForCausalLM
    )
    assert (
        model_config.MODELS_CONFIG_MAP["Glm53MixedForCausalLM"]
        is model_config.MODELS_CONFIG_MAP["GlmMoeDsaForCausalLM"]
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(architecture="Glm53MixedForCausalLM")
    )
    assert has_multi_shapes_kv_caches(config)
