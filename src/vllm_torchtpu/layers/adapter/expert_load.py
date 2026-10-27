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
"""TPU_MOE_EXPERT_LOAD_LOG_EVERY=N: the fused EP kernel's own per-slot routed
row counts, accumulated on the device and read back every N steps.

The fused EP layer all-gathers every rank's per-slot pair counts on every
step (`rows_by_dest`, [ep, physical slots]) to plan its dispatch, so every
rank already knows the whole step's load per physical slot with no further
collective. With this on, each armed layer carries a [physical slots] int32
accumulator that the op takes as a donated operand and returns updated; the
runner reads the accumulators back every N steps and logs, per layer,

    moe_lb = rows on the slowest rank / rows on the mean rank

over the window (DeepInfer's imbalance metric; under DP lockstep the slowest
rank is the layer's cost), the layer's hottest slot, and the window's worst
layers. TPU_MOE_EXPERT_LOAD_DIR writes the cumulative counts as
expert_load_dp<rank>_<seq>.npz as well. Off (0) the program is unchanged: the
accumulator is one more operand and one [ep, P] reduce per layer, and the
readback is 43 x P int32 per window, so a tapped start costs little, but it
is still not a timing arm.
"""

import os
import time

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LOAD_ATTR = "_tpu_expert_load"

# (layer name, layer, ep) for every registered layer of this process, in
# registration order. The layers own the buffers; this only lists them.
_REGISTERED: list[tuple[str, torch.nn.Module, int]] = []
_STATE: dict = {"steps": 0, "seq": 0, "previous": None}


def enabled() -> bool:
    """The accumulator exists when the log asks for it or the dynamic
    re-layout needs it (expert_relayout.py reads it through snapshot())."""
    return envs.TPU_MOE_EXPERT_LOAD_LOG_EVERY > 0 or bool(envs.TPU_MOE_EPLB_DYNAMIC)


def register(layer, *, physical: int, ep: int) -> None:
    """Give an armed layer its [physical] accumulator. Call after weight load."""
    if not enabled():
        return
    layer.register_buffer(
        LOAD_ATTR,
        torch.zeros((physical,), dtype=torch.int32, device=layer.w13_weight.device),
        persistent=False,
    )
    _REGISTERED.append((layer.layer_name, layer, ep))


def operand(layer):
    """The layer's accumulator, or None when the tap is off."""
    return getattr(layer, LOAD_ATTR, None)


def commit(layer, updated: torch.Tensor) -> None:
    """Write the op's returned accumulator back: the operand was donated."""
    getattr(layer, LOAD_ATTR).copy_(updated)


def registered_layers() -> int:
    """How many layers carry an accumulator in this process."""
    return len(_REGISTERED)


def _read_all_sync() -> np.ndarray:
    """Every registered layer's accumulator as one [layers, widest] int64 host
    array through torch: a stack program and a synchronous `.cpu()`. The
    fallback where the asynchronous reader is unavailable (CPU tests)."""
    buffers = [getattr(layer, LOAD_ATTR) for _, layer, _ in _REGISTERED]
    width = max(b.numel() for b in buffers)
    if all(b.numel() == width for b in buffers):
        stacked = torch.stack(buffers)
    else:
        stacked = torch.stack(
            [torch.nn.functional.pad(b, (0, width - b.numel())) for b in buffers]
        )
    return stacked.cpu().numpy().astype(np.int64)


class _AsyncReader:
    """Reads the accumulators the way the runner fetches sampled tokens: one
    non-blocking device-to-host copy per accumulator enqueued on the device
    stream right after the step's dispatch (`.to("cpu", non_blocking=True)`)
    and a recorded `torch.tpu.Event`, polled at the next hook. Nothing on the
    host waits at issue: the copies queue behind the program that produces
    the accumulators. (A tpu-sync raw transfer per accumulator worked but its
    constructor awaits that program on the host, ~170 ms per read at DP32.)"""

    def __init__(self) -> None:
        self._srcs: list[torch.Tensor] = []
        self._cpu: list[torch.Tensor] | None = None
        self._event = None
        self.issued_step: int | None = None
        self.latest: np.ndarray | None = None
        self.latest_step: int | None = None
        self.polls_block = False  # the event lacks query(): poll synchronizes
        self.stats = {
            "issues": 0,
            "issue_ms": 0.0,
            "issue_ms_max": 0.0,
            "ready_steps": [],
        }

    def bind(self) -> None:
        self._srcs = [getattr(layer, LOAD_ATTR) for _, layer, _ in _REGISTERED]
        event = torch.tpu.Event()  # raises where the device module is absent
        self.polls_block = not callable(getattr(event, "query", None))

    def issue(self, step: int) -> None:
        if self._event is not None:
            return
        t0 = time.perf_counter()
        self._cpu = [t.to("cpu", non_blocking=True) for t in self._srcs]
        event = torch.tpu.Event()
        event.record()
        self._event = event
        ms = (time.perf_counter() - t0) * 1e3
        self.issued_step = step
        st = self.stats
        st["issues"] += 1
        st["issue_ms"] += ms
        st["issue_ms_max"] = max(st["issue_ms_max"], ms)

    def _done(self) -> bool:
        if self.polls_block:
            self._event.synchronize()
            return True
        return bool(self._event.query())

    def poll(self, step: int) -> bool:
        """Adopt a completed read; True when `latest` changed."""
        if self._event is None or not self._done():
            return False
        width = max(t.numel() for t in self._cpu)
        host = np.zeros((len(self._cpu), width), dtype=np.int64)
        for i, c in enumerate(self._cpu):
            host[i, : c.numel()] = c.numpy().astype(np.int64, copy=False)
        self._event, self._cpu = None, None
        self.latest, self.latest_step = host, step
        self.stats["ready_steps"].append(step - (self.issued_step or step))
        return True


_READER: _AsyncReader | None = None
_CADENCE: list[int] = []  # step cadences that want a fresh read (controller windows)


def _reader() -> _AsyncReader | None:
    """The asynchronous reader once bound, or None (synchronous torch path)."""
    global _READER
    if _READER is not None or _STATE.get("raw_disabled"):
        return _READER
    if not _REGISTERED or getattr(_REGISTERED[0][1], LOAD_ATTR).device.type == "cpu":
        _STATE["raw_disabled"] = True
        return None
    try:
        reader = _AsyncReader()
        reader.bind()
    except Exception as e:  # noqa: BLE001 - the torch path stays
        logger.warning(
            "expert load | asynchronous readback unavailable (%s); reading "
            "synchronously",
            e,
        )
        _STATE["raw_disabled"] = True
        return None
    logger.info(
        "expert load | asynchronous readback armed (%s)",
        "poll synchronizes" if reader.polls_block else "event query",
    )
    _READER = reader
    return reader


def _compilations() -> int | None:
    try:
        stats = torch.tpu._get_cache_stats()  # type: ignore[attr-defined]
        return int(stats.get("num_compilations", stats.get("compilations", -1)))
    except Exception:  # noqa: BLE001
        return None


def _read_all(step: int) -> np.ndarray | None:
    """The counts for a consumer (a window boundary or a log point): the
    reader's latest adopted read, or None when none has landed yet. Never a
    wait: the hook runs before the step is dispatched, and a read's DMA is
    chained behind the program that produces the accumulators, which the hook
    itself has yet to dispatch (waiting there deadlocks the step). The torch
    path, where the raw reader is unavailable, reads synchronously."""
    reader = _reader()
    if reader is None:
        return _read_all_sync()
    return reader.latest


def after_dispatch() -> None:
    """Called once per step right after the forward's programs are
    dispatched: the step's read is issued here, so its copies queue behind
    the programs that produce the accumulators; nothing here blocks."""
    reader = _reader()
    if reader is None:
        return
    try:
        if _due_soon(_STATE["steps"]):
            reader.issue(_STATE["steps"])
    except Exception as e:  # noqa: BLE001 - fall back for good
        logger.warning(
            "expert load | asynchronous readback failed (%s); reading synchronously", e
        )
        _STATE["raw_disabled"] = True
        global _READER
        _READER = None


def schedule(cadence: int) -> None:
    """A consumer (the dynamic re-layout) wants fresh counts every `cadence`
    steps: a read is issued `_AHEAD_STEPS` ahead of each boundary."""
    if cadence > 0 and cadence not in _CADENCE:
        _CADENCE.append(int(cadence))


_AHEAD_STEPS = 4  # a read is issued this many steps before a consumer boundary


def _due_soon(step: int) -> bool:
    every = envs.TPU_MOE_EXPERT_LOAD_LOG_EVERY
    cadences = list(_CADENCE) + ([every] if every > 0 else [])
    return any((step + _AHEAD_STEPS) % c == 0 for c in cadences)


def snapshot() -> dict[tuple[str, int], np.ndarray]:
    """Every registered layer's cumulative counts, keyed by its layout
    (group, layer index), as int64 host arrays. The counts are replicated
    sums, identical on every rank after the same step, which is what lets
    the dynamic re-layout solve without a collective (expert_relayout.py)."""
    from vllm_torchtpu import expert_placement

    if not _REGISTERED:
        return {}
    host = _read_all(_STATE["steps"])
    if host is None:
        return {}
    out = {}
    for i, (name, layer, _) in enumerate(_REGISTERED):
        key = expert_placement.layout_key(layer)
        if key is None:
            continue
        out[key] = host[i, : getattr(layer, LOAD_ATTR).numel()].copy()
    return out


def summarize(window: np.ndarray, ep: int) -> dict:
    """Per-layer balance of a [physical] row-count window over `ep` ranks."""
    per_rank = window.reshape(ep, -1).sum(axis=1)
    mean = per_rank.mean()
    return {
        "moe_lb": float(per_rank.max() / mean) if mean > 0 else 1.0,
        "slowest_rank": int(per_rank.argmax()),
        "hottest_slot": int(window.argmax()),
        "hottest_rows": int(window.max()),
        "rows": int(window.sum()),
    }


def maybe_log(dp_rank: int) -> None:
    """Called once per runner step on the host, before the step is dispatched:
    adopts a completed raw read, issues the next one two steps ahead of a
    consumer's boundary, and logs every N steps."""
    if not _REGISTERED:
        return
    _STATE["steps"] += 1
    step = _STATE["steps"]
    reader = _reader()
    if reader is not None:
        try:
            if reader.poll(step):
                st = reader.stats
                logger.info(
                    "expert load | read %d adopted at step %d: issued %d step(s) "
                    "earlier, issue %.2f ms (max %.2f), compilations %s",
                    st["issues"],
                    step,
                    st["ready_steps"][-1],
                    st["issue_ms"] / max(st["issues"], 1),
                    st["issue_ms_max"],
                    _compilations(),
                )
        except Exception as e:  # noqa: BLE001 - fall back for good
            logger.warning(
                "expert load | asynchronous readback failed (%s); reading synchronously",
                e,
            )
            _STATE["raw_disabled"] = True
            global _READER
            _READER = None
    every = envs.TPU_MOE_EXPERT_LOAD_LOG_EVERY
    if every <= 0 or step % every:
        return
    # One readback for every layer, rows padded to the widest slot count and
    # summarized at their own width.
    widths = [getattr(layer, LOAD_ATTR).numel() for _, layer, _ in _REGISTERED]
    t0 = time.perf_counter()
    host = _read_all(step)
    if host is None:
        return  # no read has landed yet
    read_ms = (time.perf_counter() - t0) * 1e3
    previous = _STATE["previous"]
    window = host if previous is None else host - previous
    _STATE["previous"] = host
    stats = [
        summarize(window[i, : widths[i]], ep)
        for i, (_, _, ep) in enumerate(_REGISTERED)
    ]
    lbs = np.array([s["moe_lb"] for s in stats])
    worst = np.argsort(-lbs)[:3]
    if dp_rank == 0:
        logger.info(
            "expert load | step %d | window %d steps | read %.2f ms | moe_lb mean "
            "%.3f max %.3f | worst layers %s | per layer %s",
            _STATE["steps"],
            every,
            read_ms,
            lbs.mean(),
            lbs.max(),
            ", ".join(
                f"{_REGISTERED[i][0]}={lbs[i]:.2f} (slot {stats[i]['hottest_slot']}: "
                f"{stats[i]['hottest_rows']} rows)"
                for i in worst
            ),
            " ".join(f"{lb:.2f}" for lb in lbs),
        )
    directory = envs.TPU_MOE_EXPERT_LOAD_DIR
    if directory:
        directory = os.path.expandvars(directory)
        os.makedirs(directory, exist_ok=True)
        np.savez(
            os.path.join(directory, f"expert_load_dp{dp_rank}_{_STATE['seq']:04d}.npz"),
            counts=host,
            window=window,
            layer_names=np.array([name for name, _, _ in _REGISTERED]),
            ep=np.array([ep for _, _, ep in _REGISTERED]),
            physical=np.array(widths),
            step=_STATE["steps"],
            wall=time.time(),
        )
    _STATE["seq"] += 1
