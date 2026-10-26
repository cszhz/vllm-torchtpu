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
"""Dynamic expert re-layout (EPLB round 2): solve from the live load window,
stage the moved experts into spare slots, install on every rank in the same
step.

Every rank runs one `Controller`. Its inputs are state every rank shares:
the load tap's accumulators (replicated sums, identical after the same step)
and the coordinated step counter (every rank performs the DP step all-reduce
once per step). So the trigger, the solve and the install are the same
function of the same data on every rank, and the protocol needs no
collective of its own; three int64 fields ride the existing per-step MAX
all-reduce to confirm that and to learn when every rank has staged:

    fields()  -> (plan_hash, -plan_hash, -staged_generation)
    reduced   -> (max hash, -min hash, -min staged)

A generation: trigger at step T0 (every `window_steps`): snapshot the tap,
form the window, solve (policy + rank matching + move budget + hysteresis)
-> a Plan with a 62-bit hash. Step T0 + 1: the hashes agree on every rank
(else the generation is dropped, identically everywhere); the mover starts
copying the moved experts' slabs into the destinations' staging slots in the
background. When the reduced staged generation reaches the plan's on every
rank -- the same step for all of them -- each rank writes its layers' new
slot tables between that step and the next. Rank B's rows for rank A's old
slot were processed inside A's own step, so A overwriting the slot after its
step is safe, and B routes with the new table from the next step because it
installs between the same two steps.

See DSV41_EPLB_DYNAMIC_DESIGN.md.
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from vllm_torchtpu import eplb_policy
from vllm_torchtpu import eplb_tile_model as tm
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

HASH_BITS = 62
FREE = -1


# ---------------------------------------------------------------------------
# Layouts, plans, moves


@dataclass
class LayerLayout:
    """The live physical layout of one layer: per rank, per physical slot, the
    expert held or FREE (a staging slot)."""

    held: list[list[int]]

    @property
    def shards(self) -> int:
        return len(self.held)

    @property
    def slots(self) -> int:
        return len(self.held[0])

    def ranks(self) -> list[list[int]]:
        """Per rank, the experts held, in slot order (the tile model's input)."""
        return [[e for e in row if e != FREE] for row in self.held]

    def holders(self, expert: int) -> list[int]:
        return [r for r, row in enumerate(self.held) if expert in row]

    def slot_of(self, rank: int, expert: int) -> int:
        return self.held[rank].index(expert)

    def free_slots(self, rank: int) -> list[int]:
        return [j for j, e in enumerate(self.held[rank]) if e == FREE]

    def copies(self, experts: int) -> np.ndarray:
        counts = np.zeros(experts, dtype=np.int64)
        for row in self.held:
            for e in row:
                if e != FREE:
                    counts[e] += 1
        return counts

    def slot_table(self, experts: int, cap: int) -> list[list[int]]:
        """Row e: the physical slots holding e, padded with -1 to `cap`."""
        rows: list[list[int]] = [[] for _ in range(experts)]
        for r, row in enumerate(self.held):
            for j, e in enumerate(row):
                if e != FREE:
                    rows[e].append(r * self.slots + j)
        if any(len(h) > cap for h in rows):
            raise ValueError(f"an expert has more than {cap} copies")
        return [h + [FREE] * (cap - len(h)) for h in rows]

    def expert_of_slot(self) -> np.ndarray:
        """[shards * slots] -> expert or FREE."""
        return np.asarray([e for row in self.held for e in row], dtype=np.int64)

    def copy(self) -> LayerLayout:
        return LayerLayout([list(row) for row in self.held])

    @classmethod
    def from_group(cls, group, layer: int) -> LayerLayout:
        ranks = group.ranks(layer)
        return cls([list(held) + [FREE] * group.staging for held in ranks])


@dataclass(frozen=True)
class Move:
    group: str
    layer: int
    expert: int
    src_rank: int
    src_slot: int
    dst_rank: int
    dst_slot: int


@dataclass
class Plan:
    generation: int
    trigger_step: int
    layouts: dict[tuple[str, int], LayerLayout] = field(default_factory=dict)
    moves: list[Move] = field(default_factory=list)
    predicted: dict[tuple[str, int], tuple[float, float]] = field(default_factory=dict)

    @property
    def hash62(self) -> int:
        """A positive 62-bit digest of what every rank must agree on."""
        doc = {
            "generation": self.generation,
            "layouts": [
                [g, lid, self.layouts[(g, lid)].held]
                for (g, lid) in sorted(self.layouts)
            ],
        }
        digest = hashlib.sha256(json.dumps(doc, separators=(",", ":")).encode())
        value = int.from_bytes(digest.digest()[:8], "big") & ((1 << HASH_BITS) - 1)
        return max(value, 1)

    def moves_for(self, rank: int) -> tuple[list[Move], list[Move]]:
        """(incoming, outgoing) moves of one rank."""
        inc = [m for m in self.moves if m.dst_rank == rank]
        out = [m for m in self.moves if m.src_rank == rank]
        return inc, out

    def summary(self) -> str:
        per_rank: dict[int, int] = {}
        for m in self.moves:
            per_rank[m.dst_rank] = per_rank.get(m.dst_rank, 0) + 1
        gains = [
            f"{g}/{lid}: {b:.0f}->{a:.0f}us"
            for (g, lid), (b, a) in sorted(self.predicted.items())
        ]
        return (
            f"generation {self.generation}: {len(self.layouts)} layer(s), "
            f"{len(self.moves)} slab move(s), per destination rank "
            f"{dict(sorted(per_rank.items()))}; predicted slowest device "
            f"{', '.join(gains[:6])}{' ...' if len(gains) > 6 else ''}"
        )


# ---------------------------------------------------------------------------
# Window and solve


def window_rows(delta: np.ndarray, layout: LayerLayout, experts: int) -> np.ndarray:
    """Rows per logical expert from a window's per-slot counts."""
    rows = np.zeros(experts, dtype=np.float64)
    owner = layout.expert_of_slot()
    n = min(len(owner), len(delta))
    for s in range(n):
        if owner[s] != FREE:
            rows[owner[s]] += float(delta[s])
    return rows


def _match_ranks(new_lists: list[list[int]], old: LayerLayout) -> list[list[int]]:
    """Relabel the policy's ranks so each lands on the old rank it overlaps
    most (greedy on the overlap matrix): the policy's rank labels are
    arbitrary, so this removes every move that is only a renaming."""
    shards = len(new_lists)
    old_sets = [set(r) for r in old.ranks()]
    overlap = np.zeros((shards, shards), dtype=np.int64)
    for i, lst in enumerate(new_lists):
        s = set(lst)
        for j in range(shards):
            overlap[i, j] = len(s & old_sets[j])
    assign = [-1] * shards  # new index -> old rank
    taken_new, taken_old = set(), set()
    # Largest overlap first; ties by (new, old) index for determinism.
    order = sorted(
        ((int(overlap[i, j]), -i, -j) for i in range(shards) for j in range(shards)),
        reverse=True,
    )
    for _, ni, nj in order:
        i, j = -ni, -nj
        if i in taken_new or j in taken_old:
            continue
        assign[i] = j
        taken_new.add(i)
        taken_old.add(j)
    matched: list[list[int]] = [[] for _ in range(shards)]
    for i, j in enumerate(assign):
        matched[j] = list(new_lists[i])
    return matched


def _budgeted_next(
    old: LayerLayout,
    targets: list[list[int]],
    load: np.ndarray,
    staging: int,
    cap: int | None = None,
) -> LayerLayout:
    """The next layout toward `targets` within the move budget.

    Each rank may gain at most `staging` experts (its free slots) and must
    end the generation holding its active count again, so it removes as many
    departing experts as it gained; an expert is removed only where another
    rank still holds it afterwards, so every intermediate layout is complete.
    Adds are kept by descending load, removes chosen by ascending load.

    `cap` bounds the copies of one expert in the layout this returns, which
    the slot table's width requires. The policy's target honours it, but the
    step toward the target need not: a rank that departs two experts and adds
    one removes the lighter departure and keeps the hot one, while other ranks
    add that hot expert, so a replicated hot expert gains copies net. An
    expert over the cap gives back its adds on the highest-numbered ranks
    (those ranks take their next wanted expert instead), and the loop runs to
    a fixed point; the choice is deterministic, as every rank solves alone.
    """
    shards = old.shards
    old_sets = [set(r) for r in old.ranks()]
    target_sets = [set(t) for t in targets]
    adds = {
        r: sorted(target_sets[r] - old_sets[r], key=lambda e: (-load[e], e))
        for r in range(shards)
    }
    departing = {
        r: sorted(old_sets[r] - target_sets[r], key=lambda e: (load[e], e))
        for r in range(shards)
    }
    budget = {r: min(staging, len(old.free_slots(r))) for r in range(shards)}
    blocked: dict[int, set[int]] = {r: set() for r in range(shards)}
    while True:
        kept = {
            r: [e for e in adds[r] if e not in blocked[r]][: budget[r]]
            for r in range(shards)
        }
        # Holders after this generation's adds, before its removes.
        holders: dict[int, set[int]] = {}
        for r in range(shards):
            for e in old_sets[r] | set(kept[r]):
                holders.setdefault(e, set()).add(r)
        removes: dict[int, list[int]] = {}
        trimmed = False
        for r in range(shards):
            chosen: list[int] = []
            for e in departing[r]:
                if len(chosen) == len(kept[r]):
                    break
                if len(holders[e]) > 1:
                    chosen.append(e)
                    holders[e].discard(r)
            if len(chosen) < len(kept[r]):
                # Not enough removable departures: give back the least
                # valuable adds and try again with everyone's holders.
                budget[r] = len(chosen)
                trimmed = True
            removes[r] = chosen
        if cap is not None:
            for e in sorted(holders):
                excess = len(holders[e]) - cap
                if excess <= 0:
                    continue
                adders = sorted(
                    (r for r in range(shards) if e in kept[r]), reverse=True
                )
                for r in adders[:excess]:
                    blocked[r].add(e)
                    trimmed = True
        if not trimmed:
            break
    nxt = old.copy()
    for r in range(shards):
        for e in removes[r]:
            nxt.held[r][nxt.held[r].index(e)] = FREE
        free = nxt.free_slots(r)
        for e, j in zip(kept[r], free, strict=False):
            nxt.held[r][j] = e
    return nxt


def _moves_between(
    group: str, layer: int, old: LayerLayout, nxt: LayerLayout
) -> list[Move]:
    moves = []
    for r in range(old.shards):
        for j, e in enumerate(nxt.held[r]):
            if e == FREE or e in old.held[r]:
                continue
            src = min(old.holders(e))  # deterministic: the lowest holder sends
            moves.append(Move(group, layer, int(e), src, old.slot_of(src, e), r, j))
    return moves


def solve_group(
    group: str,
    layouts: dict[int, LayerLayout],
    windows: dict[int, np.ndarray],
    *,
    experts: int,
    active: int,
    staging: int,
    replica_cap: int,
    min_gain: float,
) -> tuple[dict[int, LayerLayout], list[Move], dict[int, tuple[float, float]]]:
    """Next layouts for one group's layers from their windows [K, experts]."""
    layer_ids = sorted(windows)
    if not layer_ids:
        return {}, [], {}
    shards = next(iter(layouts.values())).shards
    rows = np.stack([windows[lid] for lid in layer_ids], axis=1)  # [K, L, E]

    def policy(rows_kle):
        phy2log, _, _ = eplb_policy.rebalance_experts_global(
            torch.as_tensor(rows_kle, dtype=torch.float32),
            shards * active,
            active,
            max_copies=replica_cap,
        )
        return eplb_policy.rank_lists(phy2log, active)

    try:
        lists = policy(rows)
        targets_of = {lid: lists[i] for i, lid in enumerate(layer_ids)}
    except ValueError:
        # The greedy packer refuses a layer whose last slots could only take
        # an expert the rank already holds, and the batch fails as a whole.
        # Pack the layers one at a time and skip the refused ones; every rank
        # sees the same windows, so every rank skips the same layers.
        targets_of = {}
        for i, lid in enumerate(layer_ids):
            try:
                targets_of[lid] = policy(rows[:, i : i + 1, :])[0]
            except ValueError:
                continue
        logger.warning(
            "expert relayout | %s: the packer refused %d of %d layers this generation",
            group,
            len(layer_ids) - len(targets_of),
            len(layer_ids),
        )
    out_layouts, moves, predicted = {}, [], {}
    for i, lid in enumerate(layer_ids):
        if lid not in targets_of:
            continue
        old = layouts[lid]
        mean_rows = rows[:, i, :].mean(axis=0)
        targets = _match_ranks(targets_of[lid], old)
        nxt = _budgeted_next(old, targets, mean_rows, staging, cap=replica_cap)
        if nxt.held == old.held:
            continue
        copies = nxt.copies(experts)
        if (
            copies.min() < 1
            or copies.max() > replica_cap
            or any(len(r) != active for r in nxt.ranks())
        ):
            # Never hand install a layout the slot table cannot hold.
            logger.warning(
                "expert relayout | %s layer %d: budgeted layout invalid "
                "(copies %d..%d, cap %d), layer skipped this generation",
                group,
                lid,
                int(copies.min()),
                int(copies.max()),
                replica_cap,
            )
            continue
        before = tm.predict(mean_rows, old.ranks(), old.copies(experts))[0]
        after = tm.predict(mean_rows, nxt.ranks(), nxt.copies(experts))[0]
        if before <= 0 or (before - after) / before < min_gain:
            continue
        out_layouts[lid] = nxt
        predicted[lid] = (float(before), float(after))
        moves.extend(_moves_between(group, lid, old, nxt))
    return out_layouts, moves, predicted


# ---------------------------------------------------------------------------
# Movers and the controller


class Mover(Protocol):
    """Copies the moved experts' slabs into the destinations' staging slots.

    `start` returns at once and works in the background; `staged_generation`
    is the highest generation whose incoming slabs are all in place on this
    rank (the active generation when nothing is in flight); `failed` says the
    current generation cannot complete.
    """

    def start(self, plan: Plan, modules: dict[tuple[str, int], object]) -> None: ...

    def pump(self) -> None:
        """One bounded slice of this rank's device work, from the step hook."""

    def staged_generation(self) -> int: ...

    def failed(self) -> bool: ...

    def cancel(self) -> None: ...

    def settle(self, generation: int) -> None: ...


class NoopMover:
    """Stages nothing and is at once ready: for layouts with no slab to move
    (tests, and the control arm that re-solves without moving)."""

    def __init__(self, generation: int = 0) -> None:
        self._staged = generation

    def start(self, plan, modules) -> None:
        self._staged = plan.generation

    def pump(self) -> None:
        pass

    def staged_generation(self) -> int:
        return self._staged

    def failed(self) -> bool:
        return False

    def cancel(self) -> None:
        pass

    def settle(self, generation: int) -> None:
        self._staged = generation


IDLE, SOLVING, PENDING, STAGING = "idle", "solving", "pending", "staging"


class Controller:
    """One rank's view of the generation protocol (module docstring)."""

    def __init__(
        self,
        *,
        rank: int,
        groups: dict[str, dict],
        layouts: dict[tuple[str, int], LayerLayout],
        modules: dict[tuple[str, int], object],
        snapshot,
        install,
        mover: Mover,
        window_steps: int,
        sub_windows: int = 4,
        min_gain: float = 0.03,
        warmup_steps: int | None = None,
        timeout_windows: int = 2,
        synchronous_solve: bool = False,
        min_gain_us: float = 0.0,
        max_bytes: int = 0,
        move_bytes: dict[str, int] | None = None,
        hang_dump_s: float = 0.0,
    ) -> None:
        """groups: name -> dict(experts, active, staging, replica_cap).
        snapshot(): (group, layer) -> cumulative counts [P]. install(module,
        table): writes a layer's table. layouts: the live layouts at arm."""
        self.rank = rank
        self.groups = groups
        self.layouts = layouts
        self.modules = modules
        self.snapshot = snapshot
        self.install = install
        self.mover = mover
        self.window_steps = max(int(window_steps), 1)
        self.sub_windows = max(int(sub_windows), 1)
        self.min_gain = float(min_gain)
        self.min_gain_us = float(min_gain_us)
        self.max_bytes = int(max_bytes)
        self.move_bytes = dict(move_bytes or {})  # group -> bytes of one slab move
        self.hang_dump_s = float(hang_dump_s)
        self._pumped_since_step = False
        self.warmup_steps = (
            self.window_steps if warmup_steps is None else int(warmup_steps)
        )
        self.timeout_steps = max(int(timeout_windows), 1) * self.window_steps
        self.generation = 0
        self.state = IDLE
        self.pending: Plan | None = None
        self.history: list[dict[tuple[str, int], np.ndarray]] = []
        self.stats = {
            "triggers": 0,
            "skipped": 0,
            "aborted": 0,
            "installed": 0,
            "moves": 0,
        }
        self._stage_started: float | None = None
        # The solve (policy, matching, tile model over every layer) takes
        # hundreds of milliseconds on the host; it runs on a thread so the
        # step hook never carries it. Tests may run it inline.
        self.synchronous_solve = bool(synchronous_solve)
        self._solve_thread: threading.Thread | None = None
        self._solve_result: Plan | None = None
        self._solve_error: BaseException | None = None
        self._solve_step = 0

    # -- the three int64 fields of the per-step all-reduce

    def fields(self) -> tuple[int, int, int]:
        h = self.pending.hash62 if (self.state == PENDING and self.pending) else 0
        return h, -h, -int(self.mover.staged_generation())

    # -- per step, after the all-reduce, before the forward

    def on_step(self, step: int, reduced: tuple[int, int, int]) -> None:
        self._watchdog(True)  # re-armed every hook: a dump means a hung step
        hash_max, neg_hash_max, neg_staged_max = (int(v) for v in reduced)
        min_hash = -neg_hash_max
        min_staged = -neg_staged_max
        if self.state == SOLVING:
            self._collect_solve(step)
        if self.state == STAGING:
            assert self.pending is not None
            if not self._pumped_since_step:
                # No forward ran since the last hook (an idle step on this
                # rank): the mover still gets its slice of work.
                self.mover.pump()
            self._pumped_since_step = False
            if self.mover.failed():
                self._abort(step, "the mover failed")
            elif min_staged >= self.pending.generation:
                self._install(step)
            elif step - self.pending.trigger_step > self.timeout_steps:
                self._abort(step, f"not staged within {self.timeout_steps} steps")
        elif self.state == PENDING:
            assert self.pending is not None
            if hash_max == min_hash == self.pending.hash62:
                self.state = STAGING
                self._stage_started = time.monotonic()
                self._pumped_since_step = True
                self._watchdog(True)
                self.mover.start(self.pending, self.modules)
                logger.info("expert relayout | confirmed %s", self.pending.summary())
            elif min_hash == 0:
                # A rank is still solving (it publishes 0 until it has a
                # plan); wait, within the generation's timeout.
                if step - self.pending.trigger_step > self.timeout_steps:
                    self._abort(step, "not every rank published a plan in time")
            else:
                self._abort(
                    step,
                    f"plan hash disagrees across ranks (max {hash_max:#x}, "
                    f"min {min_hash:#x}, mine {self.pending.hash62:#x})",
                )
        if (
            self.state == IDLE
            and step >= self.warmup_steps
            and step % self.window_steps == 0
        ):
            self._trigger(step)

    def _watchdog(self, on: bool) -> None:
        """While a generation stages, dump every thread's stack to stderr if
        the process makes no progress for `hang_dump_s` (a hung step shows
        where); off again at install or abort."""
        if self.hang_dump_s <= 0:
            return
        try:
            if on:
                faulthandler.dump_traceback_later(
                    self.hang_dump_s, repeat=False, file=sys.stderr
                )
            else:
                faulthandler.cancel_dump_traceback_later()
        except Exception:  # noqa: BLE001 - diagnostics only
            pass

    def after_dispatch(self) -> None:
        """Called once per step right after the forward's programs are
        dispatched: the tap's read and the mover's device legs are issued
        here, behind the step's launches in the device's in-order queue,
        never ahead of them."""
        from vllm_torchtpu.layers.adapter import expert_load

        expert_load.after_dispatch()
        if self.state == STAGING:
            self._pumped_since_step = True
            self.mover.pump()

    # -- pieces

    def _trigger(self, step: int) -> None:
        snap = self.snapshot()
        if not snap:
            return  # no counts have landed yet; the next boundary will
        self.history.append(snap)
        self.history = self.history[-(self.sub_windows + 1) :]
        if len(self.history) < 2:
            return  # the first snapshot is the baseline
        self.stats["triggers"] += 1
        history = list(self.history)
        layouts = {k: v.copy() for k, v in self.layouts.items()}
        generation = self.generation + 1
        self._solve_result = None
        self._solve_error = None
        self._solve_step = step
        self.state = SOLVING
        if self.synchronous_solve:
            try:
                self._solve_result = self._solve(history, layouts, generation, step)
            except Exception as e:  # noqa: BLE001 - reported by _collect_solve
                self._solve_error = e
            self._collect_solve(step)
            return

        def run():
            try:
                self._solve_result = self._solve(history, layouts, generation, step)
            except BaseException as e:  # noqa: BLE001 - reported on the main thread
                self._solve_error = e

        self._solve_thread = threading.Thread(
            target=run, name="expert-relayout-solve", daemon=True
        )
        self._solve_thread.start()

    def _collect_solve(self, step: int) -> None:
        """Adopt a finished solve: a plan goes PENDING, nothing goes IDLE."""
        if self._solve_thread is not None and self._solve_thread.is_alive():
            return
        self._solve_thread = None
        if self._solve_error is not None:
            logger.error(
                "expert relayout | step %d: the solve failed: %r",
                step,
                self._solve_error,
            )
            self._solve_error = None
            self.stats["skipped"] += 1
            self.state = IDLE
            return
        plan = self._solve_result
        self._solve_result = None
        if plan is None or not plan.layouts:
            self.stats["skipped"] += 1
            logger.info(
                "expert relayout | step %d: no layer clears the %.0f%% gain gate",
                step,
                100 * self.min_gain,
            )
            self.state = IDLE
            return
        plan.trigger_step = step  # the timeout counts from when the plan exists
        self.pending = plan
        self.state = PENDING
        logger.info("expert relayout | step %d: solved %s", step, plan.summary())

    def _solve(self, history, layouts, generation: int, step: int) -> Plan:
        """The whole solve over every group, from copies of the window and the
        layouts: safe to run off the main thread."""
        plan = Plan(generation=generation, trigger_step=step)
        # Then the floors: a layer whose predicted saving is under
        # `min_gain_us` per step is dropped, and the generation's moves are
        # capped at `max_bytes` by saving per byte (the plan says what is
        # worth moving; the mover decides how fast).
        for name, spec in self.groups.items():
            layer_ids = sorted(lid for (g, lid) in layouts if g == name)
            windows = {}
            for lid in layer_ids:
                key = (name, lid)
                deltas = []
                for a, b in zip(history[:-1], history[1:], strict=False):
                    if key in a and key in b:
                        deltas.append(
                            window_rows(b[key] - a[key], layouts[key], spec["experts"])
                        )
                if deltas and sum(d.sum() for d in deltas) > 0:
                    windows[lid] = np.stack(deltas)
            if not windows:
                continue
            new_layouts, moves, predicted = solve_group(
                name,
                {lid: layouts[(name, lid)] for lid in layer_ids},
                windows,
                experts=spec["experts"],
                active=spec["active"],
                staging=spec["staging"],
                replica_cap=spec["replica_cap"],
                min_gain=self.min_gain,
            )
            for lid, lay in new_layouts.items():
                plan.layouts[(name, lid)] = lay
                plan.predicted[(name, lid)] = predicted[lid]
            plan.moves.extend(moves)
        return self._floors(plan)

    def _floors(self, plan: Plan) -> Plan:
        keys = list(plan.layouts)
        saving = {k: plan.predicted[k][0] - plan.predicted[k][1] for k in keys}
        drop = {k for k in keys if saving[k] < self.min_gain_us}
        if self.max_bytes > 0:
            cost = {
                k: sum(1 for m in plan.moves if (m.group, m.layer) == k)
                * self.move_bytes.get(k[0], 0)
                for k in keys
            }
            spent = 0
            for k in sorted(
                (k for k in keys if k not in drop),
                key=lambda k: -(saving[k] / max(cost[k], 1)),
            ):
                if spent + cost[k] > self.max_bytes:
                    drop.add(k)
                else:
                    spent += cost[k]
        if drop:
            for k in drop:
                plan.layouts.pop(k, None)
                plan.predicted.pop(k, None)
            plan.moves = [m for m in plan.moves if (m.group, m.layer) not in drop]
            self.stats["floored"] = self.stats.get("floored", 0) + len(drop)
        return plan

    def _install(self, step: int) -> None:
        assert self.pending is not None
        plan = self.pending
        for key, layout in plan.layouts.items():
            spec = self.groups[key[0]]
            self.install(
                self.modules[key],
                layout.slot_table(spec["experts"], spec["replica_cap"]),
            )
            self.layouts[key] = layout
        self.generation = plan.generation
        self._watchdog(False)
        self.mover.settle(self.generation)
        self.stats["installed"] += 1
        self.stats["moves"] += len(plan.moves)
        staged_s = (
            time.monotonic() - self._stage_started if self._stage_started else 0.0
        )
        logger.info(
            "expert relayout | step %d: installed generation %d (%d layers, %d moves, "
            "staged in %.1f s, %d steps after the trigger)",
            step,
            self.generation,
            len(plan.layouts),
            len(plan.moves),
            staged_s,
            step - plan.trigger_step,
        )
        # The window starts over from the counts the new layout begins with.
        self.history = [self.snapshot()]
        self.pending = None
        self.state = IDLE

    def _abort(self, step: int, why: str) -> None:
        assert self.pending is not None
        self.stats["aborted"] += 1
        logger.warning(
            "expert relayout | step %d: generation %d dropped: %s",
            step,
            self.pending.generation,
            why,
        )
        self._watchdog(False)
        self.mover.cancel()
        self.mover.settle(self.generation)
        self.pending = None
        self.state = IDLE


# ---------------------------------------------------------------------------
# Arming in a server


def arm(
    *,
    ep_rank: int,
    ep_world: int,
    ep_global_ranks: list[int] | None = None,
    lockstep: bool = True,
    vllm_config=None,
) -> Controller | None:
    """Build this rank's controller, or return None with the reason logged.

    Called once per rank at the same point of startup on every rank (after
    the model is loaded and warmed up): the mover's gloo group is created
    here, collectively. Needs the dynamic knob, a layout with staging slots
    applied to at least one layer, the load tap registered, and the DP
    lockstep (every rank performs the step all-reduce every step).

    `vllm_config` is the served model's config: a layout selected through
    `--additional-config deepseek_v41.expert_layout` (the cluster recipe) is
    only visible through it once model construction is over, and the
    `TPU_EXPERT_PLACEMENT` alias the local harness sets may be absent.
    """
    from vllm_torchtpu import envs, expert_mover, expert_placement
    from vllm_torchtpu.layers.adapter import expert_load

    if not envs.TPU_MOE_EPLB_DYNAMIC:
        return None
    placement = expert_placement.current(vllm_config)
    modules = expert_placement.applied_layers()
    if placement is None or not modules:
        logger.warning(
            "expert relayout: TPU_MOE_EPLB_DYNAMIC is set but no layer is placed"
        )
        return None
    if not lockstep:
        logger.error("expert relayout: needs the DP/EP lockstep (DP > 1 with EP)")
        return None
    if not expert_load.registered_layers():
        logger.error("expert relayout: the load tap registered no layer")
        return None
    # Counts arrive by an asynchronous read issued ahead of each sub-window.
    expert_load.schedule(
        max(envs.TPU_MOE_EPLB_WINDOW_STEPS // max(envs.TPU_MOE_EPLB_SUB_WINDOWS, 1), 1)
    )
    groups: dict[str, dict] = {}
    layouts: dict[tuple[str, int], LayerLayout] = {}
    kept_modules: dict[tuple[str, int], object] = {}
    for (gname, lid), module in sorted(modules.items(), key=lambda kv: kv[0]):
        group = placement.group(gname)
        if group is None or group.staging < 1:
            continue
        table = expert_placement.expert_slot_table(module)
        if table is None:
            continue
        cap = int(table.shape[1])
        spec = groups.setdefault(
            gname,
            dict(
                experts=group.experts,
                active=group.active,
                staging=group.staging,
                replica_cap=cap,
            ),
        )
        if spec["replica_cap"] != cap:
            logger.error(
                "expert relayout: %s layers disagree on the table width", gname
            )
            return None
        layouts[(gname, lid)] = LayerLayout.from_group(group, lid)
        kept_modules[(gname, lid)] = module
    if not groups:
        logger.warning(
            "expert relayout: layout %s has no staging slots; nothing to re-lay",
            expert_placement.configured_value(),
        )
        return None
    transport = expert_mover.make_transport(
        envs.TPU_MOE_EPLB_MOVER,
        host_memory=envs.TPU_MOE_EPLB_MOVER_HOST_MEMORY,
        skip_buffer_lock=envs.TPU_MOE_EPLB_MOVER_SKIP_BUFFER_LOCK,
    )
    ranks = (
        list(ep_global_ranks) if ep_global_ranks is not None else list(range(ep_world))
    )
    p2p = expert_mover.GlooP2P(expert_mover.make_gloo_group(ranks), rank_map=ranks)
    mover = expert_mover.SlabMover(
        ep_rank,
        transport,
        p2p,
        step_budget_bytes=int(envs.TPU_MOE_EPLB_MOVER_STEP_BUDGET_MB) << 20,
    )
    move_bytes = {}
    for (gname, _lid), module in kept_modules.items():
        if gname in move_bytes:
            continue
        try:
            move_bytes[gname] = expert_mover.move_bytes(module, transport)
        except Exception:  # noqa: BLE001 - a layer without expert tensors (tests' fakes)
            move_bytes[gname] = 0
    controller = Controller(
        rank=ep_rank,
        groups=groups,
        layouts=layouts,
        modules=kept_modules,
        snapshot=expert_load.snapshot,
        install=expert_placement.install_table,
        mover=mover,
        window_steps=envs.TPU_MOE_EPLB_WINDOW_STEPS,
        sub_windows=envs.TPU_MOE_EPLB_SUB_WINDOWS,
        min_gain=envs.TPU_MOE_EPLB_MIN_GAIN,
        warmup_steps=envs.TPU_MOE_EPLB_WARMUP_STEPS or None,
        min_gain_us=envs.TPU_MOE_EPLB_MIN_GAIN_US,
        max_bytes=int(envs.TPU_MOE_EPLB_MAX_BYTES_PER_GEN_MB) << 20,
        move_bytes=move_bytes,
        hang_dump_s=envs.TPU_MOE_EPLB_HANG_DUMP_S,
    )
    first = next(iter(kept_modules.values()))
    try:
        # Everything the mover needs on the device is made here, before any
        # step: a pump must never issue a device op of its own.
        prepare = getattr(transport, "prepare_device", None)
        if prepare is not None:
            # The first layer's load accumulator (a few KB on the device) is
            # the anchor: an existing tensor, so arming issues no device op.
            prepare(expert_load.operand(first))
        parts = expert_mover.describe_parts(first, transport)
    except Exception as e:  # noqa: BLE001 - a layer without expert tensors (tests' fakes)
        parts = str(e)
    logger.info("expert relayout | first placed layer's tensors: %s", parts)
    logger.info(
        "expert relayout armed | rank %d of %d | groups %s | %d layers | "
        "window %d steps "
        "x %d sub-windows | min gain %.0f%% | mover %s",
        ep_rank,
        ep_world,
        {g: (v["active"], v["staging"], v["replica_cap"]) for g, v in groups.items()},
        len(layouts),
        controller.window_steps,
        controller.sub_windows,
        100 * controller.min_gain,
        type(transport).__name__,
    )
    return controller


def arm_for_runner(runner) -> Controller | None:
    """The server's entry point: the EP group's rank and world, never raising."""
    try:
        from vllm.distributed.parallel_state import get_ep_group

        group = get_ep_group()
        return arm(
            ep_rank=int(group.rank_in_group),
            ep_world=int(group.world_size),
            ep_global_ranks=[int(r) for r in group.ranks],
            lockstep=bool(runner._dp_lockstep_enabled()),
            vllm_config=getattr(runner, "vllm_config", None),
        )
    except Exception:  # noqa: BLE001 - serving goes on with the static layout
        logger.exception("expert relayout: arming failed; serving the static layout")
        return None
