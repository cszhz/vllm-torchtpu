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
"""Host-side checks for the routed-row accumulator readback (expert_load.py)."""

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import expert_load

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(expert_load, "_REGISTERED", [])
    monkeypatch.setattr(expert_load, "_STATE", {"steps": 0, "seq": 0, "previous": None})


def _layer(name):
    layer = torch.nn.Module()
    layer.layer_name = name
    layer.w13_weight = torch.zeros(2)
    return layer


def test_off_registers_nothing_and_leaves_the_operand_absent(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_LOAD_LOG_EVERY", 0)
    layer = _layer("model.layers.3.ffn.experts")
    expert_load.register(layer, physical=8, ep=2)
    assert expert_load.operand(layer) is None
    assert not expert_load.enabled()
    expert_load.maybe_log(0)  # nothing registered: a no-op


def test_summarize_reads_the_slowest_rank_over_the_mean():
    window = np.array([10, 0, 0, 0, 2, 2, 2, 2])  # rank 0: 10 rows, rank 1: 8
    s = expert_load.summarize(window, ep=2)
    assert s["moe_lb"] == pytest.approx(10 / 9)
    assert s["slowest_rank"] == 0 and s["hottest_slot"] == 0 and s["hottest_rows"] == 10
    assert expert_load.summarize(np.zeros(8, dtype=np.int64), ep=2)["moe_lb"] == 1.0


def test_readback_every_n_steps_differences_the_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_LOAD_LOG_EVERY", 2)
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_LOAD_DIR", str(tmp_path))
    a, b = _layer("model.layers.0.ffn.experts"), _layer("model.layers.1.ffn.experts")
    expert_load.register(a, physical=4, ep=2)
    expert_load.register(b, physical=2, ep=2)  # a drafter layer: fewer slots
    assert expert_load.operand(a).tolist() == [0, 0, 0, 0]
    assert expert_load.operand(b).tolist() == [0, 0]
    # Step 1: the op "returned" an updated accumulator; nothing is read yet.
    expert_load.commit(a, torch.tensor([3, 1, 0, 0], dtype=torch.int32))
    expert_load.maybe_log(0)
    assert not list(tmp_path.iterdir())
    expert_load.commit(a, torch.tensor([6, 2, 0, 0], dtype=torch.int32))
    expert_load.maybe_log(0)
    files = sorted(tmp_path.iterdir())
    assert [f.name for f in files] == ["expert_load_dp0_0000.npz"]
    snap = np.load(files[0])
    assert snap["counts"].tolist() == [[6, 2, 0, 0], [0, 0, 0, 0]]
    assert snap["window"].tolist() == [[6, 2, 0, 0], [0, 0, 0, 0]]
    assert snap["layer_names"].tolist() == [a.layer_name, b.layer_name]
    assert snap["physical"].tolist() == [4, 2]
    # The next window is the difference from the last readback.
    expert_load.commit(a, torch.tensor([7, 2, 5, 0], dtype=torch.int32))
    expert_load.maybe_log(0)
    expert_load.maybe_log(0)
    snap = np.load(sorted(tmp_path.iterdir())[1])
    assert snap["window"].tolist() == [[1, 0, 5, 0], [0, 0, 0, 0]]
