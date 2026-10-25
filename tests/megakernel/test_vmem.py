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
"""Unit tests for `vmem`."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest
from conftest import (
    TINY_TILES,
    FakeGeometry,
    FakeMixerModule,
    make_fake_model_spec,
)
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel import planner, vmem
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.ops import F32, FP8
from vllm_torchtpu.megakernel.spec import (
    LayerType,
    ModelSpec,
    Module,
    ScratchReq,
    SmallField,
    WeightDecl,
)


def test_vmem_phase_overlay_and_conflict(fake_ex: ExecSpec):
    """Verify `ScratchReq.phase` overlays, budget checks, `build_scratch`, and `fitted_plan`."""
    ms_hybrid = make_fake_model_spec(L=4, att_every=4, H=256)
    cp, scr = vmem.build_scratch(ms_hybrid, fake_ex)

    # `branch_tmp` is declared by FakeAltMixerModule and FakeMixerModule -> overlaid!
    assert "branch_tmp" in scr
    assert scr["branch_tmp"].shape == (fake_ex.MR * 2, 128)
    assert cp.phase_map["branch_tmp"] == "branch:gdn|branch:gqa"

    # Overlaying two SMEM phase buffers preserves SMEM memory space
    smem_scratch: dict[str, object] = {}
    smem_phases: dict[str, str] = {}
    vmem._merge_scratch(
        smem_scratch,
        smem_phases,
        [
            ScratchReq("smem_tmp", pltpu.SMEM((4,), jnp.int32), phase="branch:a"),
            ScratchReq("smem_tmp", pltpu.SMEM((8,), jnp.int32), phase="branch:b"),
        ],
    )
    assert smem_scratch["smem_tmp"].shape == (8,)
    assert smem_scratch["smem_tmp"].memory_space == pltpu.SMEM

    # Persistent name collision must raise MegakernelConfigError
    class CollidingModule(Module):
        name = "colliding"

        def scratch(self, g, ex, plan):
            return [ScratchReq("kbuf", pltpu.VMEM((16, 128), F32), phase="persistent")]

    bad_ms = ModelSpec(
        name="bad",
        config=FakeGeometry(L=2, H=256),
        layer_type=LayerType("bad_layer", slots=(FakeMixerModule(), CollidingModule())),
    )
    with pytest.raises(
        MegakernelConfigError, match="Conflicting scratch buffer 'kbuf'"
    ):
        vmem.build_scratch(bad_ms, fake_ex)

    # Exceeding tiles.vmem_mib raises MegakernelConfigError with per-buffer breakdown
    tiny_budget_ex = dataclasses.replace(
        fake_ex, tiles=dataclasses.replace(TINY_TILES, vmem_mib=1)
    )
    with pytest.raises(MegakernelConfigError, match="exceeds tiles.vmem_mib budget"):
        vmem.build_scratch(ms_hybrid, tiny_budget_ex)

    # Custom module scratch & static-only stream layout (no epool/esem) + fitted_plan
    class CustomMLPModule(Module):
        name = "custom_mlp"
        reduction = "allreduce"

        def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
            return [
                WeightDecl("mlp_up", K=g.H, N=512, dtype=FP8, stream_ring="static"),
                WeightDecl("mlp_down", K=512, N=g.H, dtype=FP8, stream_ring="static"),
            ]

        def smalls(self, g: FakeGeometry, ex: ExecSpec) -> list[SmallField]:
            return [SmallField("mlp_scale", 512)]

        def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
            return {"acc": 512, "xb2": 512}

        def scratch(self, g: FakeGeometry, ex: ExecSpec, plan) -> dict[str, object]:
            return {"mlp_tmp": pltpu.VMEM((ex.MR, 512), F32)}

    custom_ms = ModelSpec(
        name="custom_model",
        config=FakeGeometry(L=2, H=256),
        layer_type=LayerType(
            "custom_layer", slots=(FakeMixerModule(), CustomMLPModule())
        ),
    )
    p = planner.plan(custom_ms, fake_ex)
    cp_custom, scr_custom = vmem.build_scratch(custom_ms, fake_ex, p)
    assert "mlp_tmp" in scr_custom
    assert "epool" not in scr_custom and "esem" not in scr_custom
    assert cp_custom.NACC >= 512 and cp_custom.KX2 >= 512

    ex_unset = dataclasses.replace(fake_ex, tiles=None)
    fp = vmem.fitted_plan(custom_ms, ex_unset)
    assert fp.ex.tiles is not None
