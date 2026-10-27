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
    monkeypatch.setattr(expert_load, "_READER", None)
    monkeypatch.setattr(expert_load, "_CADENCE", [])


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


def _fake_events(monkeypatch):
    """A torch.tpu.Event stand-in: a recorded event completes when the test
    lands it; copies are the CPU tensors themselves (already on the host)."""
    import types

    events = []

    class Event:
        def __init__(self):
            self.recorded, self.done = False, False

        def record(self):
            self.recorded = True
            events.append(self)

        def query(self):
            return self.done

        def synchronize(self):
            self.done = True

    monkeypatch.setattr(torch, "tpu", types.SimpleNamespace(Event=Event), raising=False)
    # The CPU guard keeps the suite on the synchronous path; lift it here.
    monkeypatch.setattr(expert_load, "_reader", _bind)
    return events


def _bind():
    """Production's `_reader()` without its CPU guard."""
    if expert_load._READER is not None or expert_load._STATE.get("raw_disabled"):
        return expert_load._READER
    try:
        reader = expert_load._AsyncReader()
        reader.bind()
    except Exception:  # noqa: BLE001
        expert_load._STATE["raw_disabled"] = True
        return None
    expert_load._READER = reader
    return reader


def test_async_reader_issues_after_dispatch_adopts_at_the_hook_never_waits(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_LOAD_LOG_EVERY", 0)
    monkeypatch.setattr(envs, "TPU_MOE_EPLB_DYNAMIC", True)
    events = _fake_events(monkeypatch)
    a, b = _layer("model.layers.0.ffn.experts"), _layer("model.layers.1.ffn.experts")
    expert_load.register(a, physical=4, ep=2)
    expert_load.register(b, physical=2, ep=2)
    a._tpu_layout_key = ("target", 0)
    b._tpu_layout_key = ("draft", 0)
    expert_load.schedule(8)
    a._tpu_expert_load.copy_(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    b._tpu_expert_load.copy_(torch.tensor([5, 6], dtype=torch.int32))

    def step():
        expert_load.maybe_log(0)  # the hook: adopt
        expert_load.after_dispatch()  # after the forward: issue

    for _ in range(3):
        step()
    reader = expert_load._READER
    assert reader is not None and not reader.polls_block and events == []
    assert expert_load.snapshot() == {}  # nothing landed: no wait, no counts
    step()  # step 4: issued after dispatch, four steps ahead of the boundary
    assert len(events) == 1 and reader.latest is None
    events[0].done = True
    step()  # step 5: adopted at the hook
    assert reader.latest_step == 5 and reader.stats["ready_steps"] == [1]
    assert reader.latest.tolist() == [[1, 2, 3, 4], [5, 6, 0, 0]]
    for _ in range(3):
        step()  # steps 6..8
    snap = expert_load.snapshot()  # the boundary takes the adopted read
    assert len(events) == 1 and snap[("target", 0)].tolist() == [1, 2, 3, 4]
    for _ in range(8):  # steps 9..16: the step-12 read never completes
        step()
    assert len(events) == 2 and reader.latest_step == 5
    assert expert_load.snapshot()[("target", 0)].tolist() == [
        1,
        2,
        3,
        4,
    ]  # stale, no wait


def test_async_reader_without_an_event_falls_back_to_the_synchronous_path(monkeypatch):
    import types

    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_LOAD_LOG_EVERY", 4)
    monkeypatch.setattr(envs, "TPU_MOE_EPLB_DYNAMIC", False)
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_LOAD_DIR", "")

    def no_event():
        raise RuntimeError("no device")

    monkeypatch.setattr(
        torch, "tpu", types.SimpleNamespace(Event=no_event), raising=False
    )
    monkeypatch.setattr(expert_load, "_reader", _bind)
    a = _layer("model.layers.0.ffn.experts")
    expert_load.register(a, physical=4, ep=2)
    a._tpu_layout_key = ("target", 0)
    for _ in range(4):
        expert_load.maybe_log(0)  # the log point reads synchronously
    assert expert_load._STATE.get("raw_disabled") and expert_load._READER is None
