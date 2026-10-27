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
"""Unit tests for the roofline engine knob registry and envs.py parity contract."""

from __future__ import annotations

import re

import vllm_torchtpu.roofline as rf

QWEN35_CFG = {
    "text_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
    }
}


def test_registry_covers_envs_threshold_keys() -> None:
    live_keys = rf.env_keys()
    pat = re.compile(rf.THRESHOLD_KEY_RE)
    threshold_keys = {k for k in live_keys if pat.search(k)}
    assert len(threshold_keys) == 13

    env_knobs = [k for k in rf.KNOBS if k.env_var]
    registry_keys = {k.key for k in env_knobs}
    missing_from_registry = threshold_keys - registry_keys
    assert not missing_from_registry, (
        f"envs.py threshold keys missing from registry: {sorted(missing_from_registry)}"
    )

    allowed_keys = live_keys | set(rf.PENDING_ENV_KEYS)
    missing_from_envs = registry_keys - allowed_keys
    assert not missing_from_envs, (
        "Registry keys absent from envs.py and PENDING_ENV_KEYS: "
        f"{sorted(missing_from_envs)}"
    )

    route_names = {k.key for k in rf.KNOBS}
    assert "megakernel" in route_names
    mk_knob = next(k for k in rf.KNOBS if k.key == "megakernel")
    assert mk_knob.cost is None and mk_knob.mechanism == "measure-only"

    geom = rf.from_hf_config(QWEN35_CFG)
    par = rf.Parallel(tp=8, moe="ep", ep=8)
    step_knob = next(k for k in rf.KNOBS if k.key == "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS")
    assert step_knob.scope == "per-step"
    assert step_knob.supports(geom, par, 32, rf.Dtypes(w_expert=1.0)) is True
    assert step_knob.supports(geom, par, 32, rf.Dtypes(w_expert=2.0)) is False
    assert (
        step_knob.supports(geom, par, 32, rf.Dtypes(w_expert=1.0), sorted_gather=True)
        is False
    )
    assert (
        step_knob.supports(geom, par, 32, rf.Dtypes(w_expert=1.0), has_hash_ids=True)
        is False
    )


def test_registry_pending_keys_absent_from_envs() -> None:
    live_keys = rf.env_keys()
    already_landed = set(rf.PENDING_ENV_KEYS) & live_keys
    assert not already_landed, (
        "Pending keys already exist in envs.py; remove from PENDING_ENV_KEYS: "
        f"{sorted(already_landed)}"
    )
