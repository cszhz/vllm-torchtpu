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
"""CPU checks for the routed-expert count tap (layers/adapter/expert_count_tap.py)."""

from types import SimpleNamespace

import numpy as np
import torch

from vllm_torchtpu.layers.adapter import expert_count_tap, token_padding
from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState

EXPERTS, TOPK = 8, 2


class _Layer(torch.nn.Module):
    def __init__(self, hash_table=None):
        super().__init__()
        self.layer_name = "model.layers.5.ffn.experts"
        self.global_num_experts = EXPERTS
        self.w13_weight = torch.zeros(1)
        self.moe_config = SimpleNamespace(experts_per_token=TOPK)
        self.renormalize = True
        self.scoring_func = "sqrtsoftplus"
        self.use_grouped_topk = False
        self.routed_scaling_factor = 1.0
        self.custom_routing_function = None
        self.e_score_correction_bias = None
        self.hash_indices_table = hash_table


def _tapped(monkeypatch, tmp_path, layer):
    monkeypatch.setenv("MOE_EXPERT_COUNT_TAP_DIR", str(tmp_path))
    monkeypatch.setenv("MOE_EXPERT_COUNT_TAP_EVERY", "2")
    monkeypatch.setattr(expert_count_tap, "_TAPPED", [])
    monkeypatch.setattr(expert_count_tap, "_STATE", {"steps": 0, "seq": 0})
    expert_count_tap.register(layer)
    return getattr(layer, expert_count_tap.COUNTS_ATTR)


def test_counts_squares_and_meta_skip_padding(monkeypatch, tmp_path):
    layer = _Layer()
    counts = _tapped(monkeypatch, tmp_path, layer)
    logits = torch.full((4, EXPERTS), -5.0)
    logits[:, 3] = 5.0
    logits[:, 6] = 4.0
    logits[3, :] = 0.0  # the padding row selects 1 and 3; it must not count
    logits[3, 1] = 9.0
    state = TokenPaddingState(local_padding_mask=torch.zeros(4, dtype=torch.bool))
    state.update(3, 4)
    token_padding.set_padding_state(state)
    try:
        expert_count_tap.record(layer, logits, None)
        expert_count_tap.record(layer, logits, None)
    finally:
        token_padding.set_padding_state(None)
    want = np.zeros(EXPERTS, np.int32)
    want[[3, 6]] = 3
    np.testing.assert_array_equal(counts[0].numpy(), 2 * want)
    np.testing.assert_array_equal(counts[1].numpy(), 2 * want * want)
    assert counts[2, :2].tolist() == [2, 6]


def test_hash_layers_count_the_table_selection(monkeypatch, tmp_path):
    table = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
    layer = _Layer(hash_table=table)
    counts = _tapped(monkeypatch, tmp_path, layer)
    expert_count_tap.record(layer, torch.zeros((3, EXPERTS)), torch.tensor([2, 2, 0]))
    assert counts[0].tolist() == [1, 1, 0, 0, 2, 2, 0, 0]


def test_dump_writes_cumulative_snapshots_every_n_steps(monkeypatch, tmp_path):
    layer = _Layer()
    counts = _tapped(monkeypatch, tmp_path, layer)
    counts[0, 2] = 7
    for _ in range(5):
        expert_count_tap.maybe_dump(ep_rank=3)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["counts_ep3_0000.npz", "counts_ep3_0001.npz"]
    snap = np.load(tmp_path / files[-1])
    assert snap["counts"].shape == (1, 3, EXPERTS)
    assert snap["counts"][0, 0, 2] == 7
    assert snap["layer_names"].tolist() == [layer.layer_name]
    assert int(snap["step"]) == 4


def test_the_directory_expands_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("QH_TAP_ROOT", str(tmp_path))
    layer = _Layer()
    _tapped(monkeypatch, tmp_path, layer)
    monkeypatch.setenv("MOE_EXPERT_COUNT_TAP_DIR", "$QH_TAP_ROOT/sub")
    for _ in range(2):
        expert_count_tap.maybe_dump(ep_rank=0)
    assert (tmp_path / "sub" / "counts_ep0_0000.npz").exists()
