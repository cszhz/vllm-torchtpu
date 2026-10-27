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
"""Analytical TPU serving roofline, step floor, mesh solver, and knob registry."""

from __future__ import annotations

from .chips import CHIPS, TOPOLOGIES, ChipSpec, SliceTopology
from .collectives import (
    Collective,
    CollectiveTable,
    submesh_for_group,
    torus_collective_s,
)
from .geometry import LAYER_FAMILIES, ModelGeometry, from_hf_config
from .hf_loader import dtypes_from_hf_config, load_hf_config
from .modules import (
    Dtypes,
    ModuleCost,
    Parallel,
    StepShape,
    ep_busiest_experts,
    experts_touched,
    pass_costs,
)
from .registry import (
    KNOBS,
    PENDING_ENV_KEYS,
    THRESHOLD_KEY_RE,
    Knob,
    default_envs_path,
    env_keys,
    threshold_env_keys,
)
from .serving import ConcurrencyRow, plan
from .solver import (
    SCENARIO_PRESETS,
    SLO_TIERS,
    ParetoPoint,
    PoolMeshPlan,
    ServingPlan,
    TrafficScenario,
    build_scenario,
    hbm_footprint_per_core_bytes,
    solve_serving_plan,
)
from .step import StepFloor, spec_cycle_floor, step_floor

__all__ = [
    "CHIPS",
    "KNOBS",
    "LAYER_FAMILIES",
    "PENDING_ENV_KEYS",
    "SCENARIO_PRESETS",
    "SLO_TIERS",
    "THRESHOLD_KEY_RE",
    "TOPOLOGIES",
    "ChipSpec",
    "Collective",
    "CollectiveTable",
    "ConcurrencyRow",
    "Dtypes",
    "Knob",
    "ModelGeometry",
    "ModuleCost",
    "Parallel",
    "ParetoPoint",
    "PoolMeshPlan",
    "ServingPlan",
    "SliceTopology",
    "StepFloor",
    "StepShape",
    "TrafficScenario",
    "build_scenario",
    "default_envs_path",
    "dtypes_from_hf_config",
    "env_keys",
    "ep_busiest_experts",
    "experts_touched",
    "from_hf_config",
    "hbm_footprint_per_core_bytes",
    "load_hf_config",
    "pass_costs",
    "plan",
    "solve_serving_plan",
    "spec_cycle_floor",
    "step_floor",
    "submesh_for_group",
    "threshold_env_keys",
    "torus_collective_s",
]
