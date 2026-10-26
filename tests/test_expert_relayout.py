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
"""The dynamic re-layout controller, on the host, with fakes for the tap, the
mover and the per-step all-reduce (expert_relayout.py)."""

import numpy as np
import pytest
import torch

from vllm_torchtpu import eplb_policy
from vllm_torchtpu import expert_relayout as rl

pytestmark = pytest.mark.cpu_test

SPEC = dict(experts=8, active=4, staging=1, replica_cap=2)
LAYOUT = [[0, 1, 2, 3, rl.FREE], [4, 5, 6, 7, rl.FREE]]


def _layout():
    return rl.LayerLayout([list(r) for r in LAYOUT])


def _rows(hot=(), steps=3):
    """[steps, experts] rows with the `hot` experts at 200 and the rest at 10.
    Two hot experts on one rank is what a repacking can fix: with no
    redundant active slot a single hot expert cannot be split."""
    rows = np.full((steps, 8), 10.0)
    for e in hot:
        rows[:, e] = 200.0
    return rows


# ---------------------------------------------------------------------------
# Layouts, windows, plans


def test_layout_slot_table_and_window_rows():
    lay = _layout()
    assert lay.ranks() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    table = lay.slot_table(8, 2)
    assert table[0] == [0, -1] and table[4] == [5, -1]  # rank 1's slots start at 5
    assert lay.free_slots(0) == [4] and lay.holders(6) == [1]
    delta = np.array([7, 1, 1, 1, 0, 2, 2, 2, 2, 0])  # per physical slot
    rows = rl.window_rows(delta, lay, 8)
    assert rows.tolist() == [7, 1, 1, 1, 2, 2, 2, 2]
    with pytest.raises(ValueError, match="more than 1 copies"):
        rl.LayerLayout([[0, 1, 2, 3, 0], [4, 5, 6, 7, -1]]).slot_table(8, 1)


def test_plan_hash_is_positive_deterministic_and_sensitive():
    a = rl.Plan(1, 0, {("target", 0): _layout()})
    b = rl.Plan(1, 0, {("target", 0): _layout()})
    assert a.hash62 == b.hash62 > 0 and a.hash62 < 1 << rl.HASH_BITS
    moved = _layout()
    moved.held[0][4] = 7
    assert rl.Plan(1, 0, {("target", 0): moved}).hash62 != a.hash62
    assert rl.Plan(2, 0, {("target", 0): _layout()}).hash62 != a.hash62


def test_policy_copy_cap_bounds_the_replicas():
    rows = torch.tensor(_rows(hot=(0,)), dtype=torch.float32)[:, None, :]  # [K, 1, E]
    _, _, counts = eplb_policy.rebalance_experts_global(rows, 16, 4, max_copies=2)
    assert int(counts.max()) <= 2 and int(counts.sum()) == 16
    _, _, counts3 = eplb_policy.rebalance_experts_global(rows, 16, 4)
    assert int(counts3.max()) > 2  # the cap was what held it
    with pytest.raises(ValueError, match="max_copies"):
        eplb_policy.rebalance_experts_global(rows, 16, 4, max_copies=0)


# ---------------------------------------------------------------------------
# Solve: matching, budget, hysteresis


def test_rank_matching_removes_renaming_moves():
    lay = _layout()
    swapped = [[4, 5, 6, 7], [0, 1, 2, 3]]  # the policy's ranks, relabelled
    assert rl._match_ranks(swapped, lay) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    nxt = rl._budgeted_next(lay, rl._match_ranks(swapped, lay), np.ones(8), 1)
    assert nxt.held == lay.held


def test_budget_limits_adds_and_keeps_every_expert_held():
    lay = _layout()
    # The target swaps two experts between the ranks: 2 adds per rank, budget 1.
    targets = [[0, 1, 6, 7], [4, 5, 2, 3]]
    load = np.array([1, 1, 5, 2, 1, 1, 9, 3], dtype=float)
    nxt = rl._budgeted_next(lay, targets, load, staging=1)
    # Rank 0 gained its hottest wanted expert (6) and let one departing go.
    assert nxt.held[0].count(6) == 1 and nxt.held[0].count(rl.FREE) == 1
    assert nxt.held[1].count(2) == 1 and nxt.held[1].count(rl.FREE) == 1
    held = sorted(e for row in nxt.held for e in row if e != rl.FREE)
    assert held == list(range(8))  # complete: nobody orphaned
    moves = rl._moves_between("target", 0, lay, nxt)
    assert {(m.expert, m.src_rank, m.dst_rank) for m in moves} == {(6, 1, 0), (2, 0, 1)}
    assert all(m.src_slot == lay.held[m.src_rank].index(m.expert) for m in moves)
    assert all(nxt.held[m.dst_rank][m.dst_slot] == m.expert for m in moves)
    # A second generation toward the same target finishes the swap.
    nxt2 = rl._budgeted_next(nxt, targets, load, staging=1)
    assert sorted(e for e in nxt2.held[0] if e != rl.FREE) == [0, 1, 6, 7]
    assert sorted(e for e in nxt2.held[1] if e != rl.FREE) == [2, 3, 4, 5]


def test_removes_never_orphan_a_single_copy():
    lay = _layout()
    # Rank 0 wants 7 and would drop 3, but 3 has no other holder in the
    # target either: the add is given back rather than orphaning 3.
    targets = [[0, 1, 2, 7], [4, 5, 6, 7]]
    nxt = rl._budgeted_next(lay, targets, np.ones(8), staging=1)
    held = sorted(e for row in nxt.held for e in row if e != rl.FREE)
    assert held == list(range(8)) or held == sorted(list(range(8)) + [7])
    assert 3 in nxt.held[0] or 3 in nxt.held[1]


def test_solve_group_moves_for_a_hot_expert_and_skips_a_flat_window():
    lay = _layout()
    flat = {0: _rows()}
    layouts, moves, _ = rl.solve_group("target", {0: lay}, flat, min_gain=0.03, **SPEC)
    assert layouts == {} and moves == []
    hot = {0: _rows(hot=(0, 1))}
    layouts, moves, predicted = rl.solve_group(
        "target", {0: lay}, hot, min_gain=0.03, **SPEC
    )
    assert set(layouts) == {0} and moves
    before, after = predicted[0]
    assert after < before
    held = sorted(e for row in layouts[0].held for e in row if e != rl.FREE)
    assert set(held) == set(range(8))
    for row in layouts[0].held:
        assert row.count(rl.FREE) == 1  # the budget's spare slot survives


# ---------------------------------------------------------------------------
# The protocol over a fake all-reduce


class _Tap:
    """Replicated counts: one hot expert, identical on every rank."""

    def __init__(self):
        self.step = 0

    def advance(self, steps=1):
        self.step += steps

    def snapshot(self):
        counts = np.full(10, 10 * self.step, dtype=np.int64)
        counts[0] = counts[1] = 200 * self.step  # experts 0 and 1, both on rank 0
        counts[4] = counts[9] = 0  # staging slots
        return {("target", 0): counts.copy()}


class _Mover:
    def __init__(self, generation=0, auto=True):
        self.staged = generation
        self.auto = auto
        self.started: list[rl.Plan] = []
        self.cancelled = 0
        self._failed = False

    def start(self, plan, modules):
        self.started.append(plan)
        if self.auto:
            self.staged = plan.generation

    def staged_generation(self):
        return self.staged

    def failed(self):
        return self._failed

    def cancel(self):
        self.cancelled += 1

    def settle(self, generation):
        self.staged = generation


def _controller(rank, tap, mover, installs, **kw):
    return rl.Controller(
        rank=rank,
        groups={"target": SPEC},
        layouts={("target", 0): _layout()},
        modules={("target", 0): f"module-{rank}"},
        snapshot=tap.snapshot,
        install=lambda module, table: installs.append((module, table)),
        mover=mover,
        window_steps=4,
        sub_windows=2,
        warmup_steps=0,
        **kw,
    )


def _run(controllers, tap, steps, start=0):
    for step in range(start, start + steps):
        fields = [c.fields() for c in controllers]
        reduced = tuple(max(f[i] for f in fields) for i in range(3))
        for c in controllers:
            c.on_step(step, reduced)
        tap.advance()


def test_two_ranks_reach_the_same_plan_and_install_in_the_same_step():
    tap = _Tap()
    installs = []
    movers = [_Mover(), _Mover()]
    cs = [_controller(r, tap, movers[r], installs) for r in range(2)]
    _run(cs, tap, 4)  # step 0 takes the baseline snapshot, nothing else
    assert all(c.state == rl.IDLE for c in cs) and installs == []
    _run(cs, tap, 1, start=4)  # trigger: both solve the same plan
    assert all(c.state == rl.PENDING for c in cs)
    assert cs[0].pending.hash62 == cs[1].pending.hash62
    _run(cs, tap, 1, start=5)  # hashes agree -> staging starts (auto-staged)
    assert all(c.state == rl.STAGING for c in cs)
    assert [len(m.started) for m in movers] == [1, 1]
    _run(cs, tap, 1, start=6)  # both staged -> both install this step
    assert all(c.state == rl.IDLE and c.generation == 1 for c in cs)
    assert [m for m, _ in installs] == ["module-0", "module-1"]
    table = installs[0][1]
    assert len(table) == 8 and all(len(r) == 2 for r in table)
    assert cs[0].layouts[("target", 0)].held == cs[1].layouts[("target", 0)].held
    assert cs[0].stats["installed"] == 1 and len(cs[0].history) == 1


def test_install_waits_for_the_slowest_rank():
    tap = _Tap()
    installs = []
    movers = [_Mover(), _Mover(auto=False)]
    cs = [_controller(r, tap, movers[r], installs) for r in range(2)]
    _run(cs, tap, 7)
    assert all(c.state == rl.STAGING for c in cs) and installs == []
    _run(cs, tap, 1, start=7)
    assert installs == []  # rank 1 still staging
    movers[1].staged = 1
    _run(cs, tap, 1, start=8)
    assert all(c.generation == 1 for c in cs) and len(installs) == 2


def test_hash_disagreement_drops_the_generation_everywhere():
    tap = _Tap()
    installs = []
    cs = [_controller(r, tap, _Mover(), installs) for r in range(2)]
    _run(cs, tap, 5)
    assert all(c.state == rl.PENDING for c in cs)
    cs[1].pending.generation = 7  # a rank that solved something else
    _run(cs, tap, 1, start=5)
    assert all(c.state == rl.IDLE and c.generation == 0 for c in cs)
    assert all(c.stats["aborted"] == 1 for c in cs) and installs == []


def test_staging_timeout_and_mover_failure_abort():
    tap = _Tap()
    installs = []
    slow = _Mover(auto=False)
    cs = [_controller(0, tap, slow, installs, timeout_windows=1)]
    _run(cs, tap, 6)
    assert cs[0].state == rl.STAGING
    _run(cs, tap, 5, start=6)  # past the timeout (window 4 x 1)
    assert (
        cs[0].state == rl.IDLE and cs[0].stats["aborted"] == 1 and slow.cancelled == 1
    )
    failing = _Mover(auto=False)
    cs = [_controller(0, tap, failing, installs)]
    _run(cs, tap, 6)
    failing._failed = True
    _run(cs, tap, 1, start=6)
    assert cs[0].state == rl.IDLE and cs[0].stats["aborted"] == 1


def test_flat_traffic_never_triggers_a_move():
    class Flat(_Tap):
        def snapshot(self):
            counts = np.full(10, 10 * self.step, dtype=np.int64)
            counts[4] = counts[9] = 0
            return {("target", 0): counts}

    tap = Flat()
    installs = []
    cs = [_controller(0, tap, _Mover(), installs)]
    _run(cs, tap, 13)
    assert cs[0].stats["installed"] == 0 and cs[0].stats["skipped"] >= 2


# ---------------------------------------------------------------------------
# Arming in a server


def _arm_env(monkeypatch, **over):
    from vllm_torchtpu import envs

    values = dict(
        TPU_MOE_EPLB_DYNAMIC=True,
        TPU_MOE_EPLB_WINDOW_STEPS=16,
        TPU_MOE_EPLB_SUB_WINDOWS=2,
        TPU_MOE_EPLB_MIN_GAIN=0.05,
        TPU_MOE_EPLB_WARMUP_STEPS=0,
        TPU_MOE_EPLB_MOVER="torch",
    )
    values.update(over)
    for k, v in values.items():
        monkeypatch.setitem(vars(envs), k, v)


def _fake_placed_world(monkeypatch, staging=1):
    from types import SimpleNamespace

    from vllm_torchtpu import expert_mover, expert_placement
    from vllm_torchtpu.layers.adapter import expert_load

    group = SimpleNamespace(
        experts=8,
        active=4,
        staging=staging,
        slots=4 + staging,
        replica_cap=2,
        ranks=lambda layer: ((0, 1, 2, 3), (4, 5, 6, 7)),
    )
    placement = SimpleNamespace(group=lambda name: group if name == "target" else None)
    module = SimpleNamespace(
        layer_name="layers.0.mlp.experts",
        _tpu_expert_slot=torch.zeros((8, 2), dtype=torch.int32),
    )
    monkeypatch.setattr(expert_placement, "current", lambda vllm_config=None: placement)
    monkeypatch.setattr(
        expert_placement, "applied_layers", lambda: {("target", 0): module}
    )
    monkeypatch.setattr(
        expert_placement, "configured_value", lambda vllm_config=None: "fake"
    )
    monkeypatch.setattr(expert_load, "registered_layers", lambda: 1)
    monkeypatch.setattr(
        expert_mover, "make_gloo_group", lambda ranks: ("gloo", tuple(ranks))
    )
    return module


def test_arm_builds_a_controller_from_the_placed_layers(monkeypatch):
    _arm_env(monkeypatch)
    _fake_placed_world(monkeypatch)
    c = rl.arm(ep_rank=1, ep_world=2, ep_global_ranks=[4, 5])
    assert c is not None and c.rank == 1 and c.window_steps == 16
    assert c.groups == {"target": dict(experts=8, active=4, staging=1, replica_cap=2)}
    assert c.layouts[("target", 0)].held == [
        [0, 1, 2, 3, rl.FREE],
        [4, 5, 6, 7, rl.FREE],
    ]
    assert c.mover.p2p.group == ("gloo", (4, 5)) and c.mover.p2p._global(1) == 5


def test_arm_refuses_without_the_knob_staging_or_lockstep(monkeypatch):
    _fake_placed_world(monkeypatch)
    _arm_env(monkeypatch, TPU_MOE_EPLB_DYNAMIC=False)
    assert rl.arm(ep_rank=0, ep_world=2) is None
    _arm_env(monkeypatch)
    assert rl.arm(ep_rank=0, ep_world=2, lockstep=False) is None
    _fake_placed_world(monkeypatch, staging=0)
    assert rl.arm(ep_rank=0, ep_world=2) is None


def test_the_tap_exists_under_the_dynamic_knob(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import expert_load

    monkeypatch.setitem(vars(envs), "TPU_MOE_EXPERT_LOAD_LOG_EVERY", 0)
    monkeypatch.setitem(vars(envs), "TPU_MOE_EPLB_DYNAMIC", True)
    assert expert_load.enabled()
    monkeypatch.setitem(vars(envs), "TPU_MOE_EPLB_DYNAMIC", False)
    assert not expert_load.enabled()
