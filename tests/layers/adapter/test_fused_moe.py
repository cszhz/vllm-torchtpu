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

from enum import Enum
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter import fused_moe
from vllm_torchtpu.layers.adapter.fused_moe import get_fused_moe_activation


class _Activation(Enum):
    SILU = "silu"


@pytest.mark.parametrize("activation", ["silu", _Activation.SILU])
def test_get_fused_moe_activation_preserves_standard_activations(activation):
    assert get_fused_moe_activation(activation, object()) == "silu"


@pytest.mark.parametrize(
    "linear_beta,expected",
    [
        (None, "situ:4.0:none"),
        (25.0, "situ:4.0:25.0"),
    ],
)
def test_get_fused_moe_activation_encodes_situ(linear_beta, expected):
    moe_config = SimpleNamespace(
        activation_situ_beta=4.0,
        activation_situ_linear_beta=linear_beta,
    )

    assert get_fused_moe_activation("situ", moe_config) == expected


def test_get_fused_moe_activation_requires_situ_beta():
    moe_config = SimpleNamespace(
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )

    with pytest.raises(AssertionError):
        get_fused_moe_activation("situ", moe_config)


@pytest.mark.tpu_test
def test_get_fused_moe_activation_compiles():
    @torch.compile(fullgraph=True)
    def compiled_fn(x, activation):
        act_str = get_fused_moe_activation(activation, None)
        if act_str == "silu":
            return x * 2
        return x

    res = compiled_fn(torch.ones(1).to("tpu"), _Activation.SILU)
    assert res.cpu() == 2.0


@pytest.mark.parametrize("use_sparse_core", [True, False])
def test_custom_op_takes_sparse_core_flag_from_resolver(monkeypatch, use_sparse_core):
    """The op binds resolve_use_sparse_core(), which turns the SparseCore path
    off on TPUs whose SparseCore the kernels do not support (TPU8i)."""
    built = {}

    class _Op:
        def register_fake(self, fn):
            pass

    def jax_op(name, fn):
        built["fn"] = fn
        return _Op()

    monkeypatch.setattr(fused_moe.pallas, "jax_op", jax_op)
    monkeypatch.setattr(fused_moe, "resolve_use_sparse_core", lambda: use_sparse_core)
    monkeypatch.setattr(fused_moe, "resolve_onehot_permute_threshold", lambda: 0)
    monkeypatch.setattr(fused_moe, "_fused_moe_kernel_cache", {})
    fused_moe._build_fused_moe_custom_op(
        topk=2, activation="silu", use_ep=False, skip_padded_tokens=False
    )
    assert built["fn"].keywords["use_sparse_core"] is use_sparse_core
