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
"""Declarative `ModelDef` for Qwen3.8 MoE (`L=92, H=8192`)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from ..model_def import register_model
from ..spec import Block, ModelSpec
from .qwen35 import Qwen35, Qwen35Geometry
from .qwen38_profile import (
    DEFAULT_HF_CONFIG,
    PROFILE,
    QWEN38_MODEL_NAME,
    TP32_DCP8_PROFILE,
)


@dataclass(frozen=True, init=False, eq=False)
class Qwen38Geometry(Qwen35Geometry):
    """Architecture geometry for Qwen3.8 MoE."""

    def __init__(self, **overrides: Any) -> None:
        base = dataclasses.asdict(FULL) if "FULL" in globals() else {}
        super(Qwen35Geometry, self).__init__(**(base | overrides))

    def to_model_spec(self, R: int = 32) -> ModelSpec:
        """Convert this geometry into its declarative `ModelSpec`."""
        return qwen38_spec(self, R=R)


@register_model(
    name=QWEN38_MODEL_NAME,
    hf_architectures=(
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForCausalLM",
    ),
    hf_model_type=("qwen3_5_moe", "qwen3_5", "qwen3_5_moe_text"),
)
class Qwen38(Qwen35):
    """Declarative model definition for Qwen3.8 MoE."""

    geometry_cls = Qwen38Geometry
    layer_type_name = "qwen38_layer"
    default_hf_config = DEFAULT_HF_CONFIG
    profiles = {
        "default": PROFILE,
        "tp32-dcp8": TP32_DCP8_PROFILE,
    }


FULL = Qwen38.derive_geometry(DEFAULT_HF_CONFIG, moe_tp=True)
FULL_EP = dataclasses.replace(FULL, moe_tp=False)
TINY = dataclasses.replace(
    Qwen38.shrink(DEFAULT_HF_CONFIG, periods=1, R=8), moe_tp=True
)
TINY_EP = dataclasses.replace(TINY, moe_tp=False)


def qwen38_spec(
    g: Qwen38Geometry = FULL,
    gdn_block: Block | None = None,
    *,
    R: int = 32,
) -> ModelSpec:
    """Construct the declarative `ModelSpec` for Qwen3.8."""
    return Qwen38.to_model_spec(g, gdn_block=gdn_block, R=R)
