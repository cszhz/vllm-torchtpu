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
    fallback where tpu-sync's raw transfer is unavailable (CPU tests) or
    refused the accumulators, and the reference the async reader checks
    itself against once."""
    buffers = [getattr(layer, LOAD_ATTR) for _, layer, _ in _REGISTERED]
    width = max(b.numel() for b in buffers)
    if all(b.numel() == width for b in buffers):
        stacked = torch.stack(buffers)
    else:
        stacked = torch.stack(
            [torch.nn.functional.pad(b, (0, width - b.numel())) for b in buffers]
        )
    return stacked.cpu().numpy().astype(np.int64)


class _RawReader:
    """Reads the accumulators by tpu-sync's whole-buffer raw D2H into pinned
    `RawHostBuffer`s: no device program, no synchronous readback. Each
    accumulator gets its own host buffer of its physical size and a
    `PreparedTorchRawTransfer` rebound per read (the operand is donated, so
    the device buffer behind the tensor is new every step); the sliced
    transfer forms cannot address a 1-D array (its device shape is
    lane-padded, so a logical-size copy counts as partial, which rank < 3
    refuses). A read is issued from the step thread (`issue`), the DMA lands
    while the step runs, and the result is adopted by `poll` a step or two
    later. At bind a synchronous read is checked against the torch path and
    the reader is refused on a mismatch (the raw bytes are the device's
    physical layout: a 1-D int32 array is linear with padding at the end)."""

    def __init__(self) -> None:
        from tpu_sync.api.torch import raw_transfer

        if not callable(raw_transfer.major_slice_bytes):
            raise ImportError("tpu-sync raw transfer lacks major_slice_bytes")
        self._rt = raw_transfer
        self._srcs: list[torch.Tensor] = []
        self._bufs: list = []
        self._views: list[np.ndarray] = []
        self._prepared: list = []
        self._futures: list | None = None
        self.issued_step: int | None = None
        self.latest: np.ndarray | None = None
        self.latest_step: int | None = None
        self.checked = False
        self._ptrs: list[int] | None = None
        self.stats = {
            "issues": 0,
            "rebound": -1,
            "issue_ms": 0.0,
            "issue_ms_max": 0.0,
            "ready_steps": [],
        }

    def bind(self) -> None:
        import ctypes

        srcs = [getattr(layer, LOAD_ATTR) for _, layer, _ in _REGISTERED]
        self._bufs, self._views = [], []
        for t in srcs:
            # The physical size is lane-padded (and may be sublane-padded);
            # the buffer is sized by the first binding's answer.
            guess = max(t.numel(), 1024) * t.element_size() * 8
            buf = self._rt.RawHostBuffer(guess)
            prepared = self._rt.PreparedTorchRawTransfer(t, buf, True)
            physical = int(prepared.physical_size_bytes)
            if physical < t.numel() * t.element_size():
                raise ValueError("accumulator physical size below its logical size")
            owner = (ctypes.c_uint8 * buf.size_bytes).from_address(buf.data_ptr)
            flat = torch.frombuffer(owner, dtype=torch.uint8)[
                : t.numel() * t.element_size()
            ]
            self._bufs.append(buf)
            self._views.append(flat.numpy().view(np.int32).reshape(-1))
        self._srcs = srcs
        # The self-check, here and synchronous: a read issued now, waited for,
        # against the torch path read at the same point (no step in between).
        self.issue(-1)
        self._rt.await_all(self._futures)
        self._futures, self._prepared = None, []
        width = max(v.size for v in self._views)
        host = np.zeros((len(self._views), width), dtype=np.int64)
        for i, v in enumerate(self._views):
            host[i, : v.size] = v
        ref = _read_all_sync()
        if ref.shape != host.shape or not np.array_equal(ref, host):
            raise ValueError(
                "raw readback disagrees with the torch readback (the 1-D "
                "accumulator's physical layout is not linear)"
            )
        self.checked = True
        self.stats["issues"] = 0
        self.stats["issue_ms"] = 0.0
        self.stats["issue_ms_max"] = 0.0

    def issue(self, step: int) -> None:
        if self._futures is not None:
            return
        t0 = time.perf_counter()
        # Do the accumulators' device buffers change between reads? (They
        # are donated operands, so presumably yes; if not, the transfers
        # could be bound once instead of per read.)
        ptrs = [t.untyped_storage().data_ptr() for t in self._srcs]
        if self._ptrs is not None:
            self.stats["rebound"] = sum(
                a != b for a, b in zip(ptrs, self._ptrs, strict=True)
            )
        self._ptrs = ptrs
        prepared, futures = [], []
        for t, buf in zip(self._srcs, self._bufs, strict=True):
            pt = self._rt.PreparedTorchRawTransfer(t, buf, True)
            prepared.append(pt)
            futures.append(pt.d2h_async())
        self._prepared, self._futures = prepared, futures
        ms = (time.perf_counter() - t0) * 1e3
        self.issued_step = step
        st = self.stats
        st["issues"] += 1
        st["issue_ms"] += ms
        st["issue_ms_max"] = max(st["issue_ms_max"], ms)

    def _adopt(self, step: int) -> None:
        width = max(v.size for v in self._views)
        host = np.zeros((len(self._views), width), dtype=np.int64)
        for i, v in enumerate(self._views):
            host[i, : v.size] = v
        self._futures, self._prepared = None, []
        self.latest, self.latest_step = host, step
        self.stats["ready_steps"].append(step - (self.issued_step or step))

    def poll(self, step: int) -> bool:
        """Adopt a completed read; True when `latest` changed."""
        if self._futures is None or not self._rt.is_ready(self._futures):
            return False
        self._adopt(step)
        return True


_READER: _RawReader | None = None
_CADENCE: list[int] = []  # step cadences that want a fresh read (controller windows)


def _reader() -> _RawReader | None:
    """The raw reader once it is bound and self-checked, or None (torch path)."""
    global _READER
    if _READER is not None or _STATE.get("raw_disabled"):
        return _READER
    if not _REGISTERED or getattr(_REGISTERED[0][1], LOAD_ATTR).device.type == "cpu":
        _STATE["raw_disabled"] = True
        return None
    try:
        reader = _RawReader()
        reader.bind()
    except Exception as e:  # noqa: BLE001 - the torch path stays
        logger.warning(
            "expert load | raw readback unavailable (%s); reading through torch", e
        )
        _STATE["raw_disabled"] = True
        return None
    _READER = reader
    return reader


def _compilations() -> int | None:
    try:
        stats = torch.tpu._get_cache_stats()  # type: ignore[attr-defined]
        return int(stats.get("num_compilations", stats.get("compilations", -1)))
    except Exception:  # noqa: BLE001
        return None


def _read_all(step: int) -> np.ndarray | None:
    """The counts for a consumer (a window boundary or a log point): the raw
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
    dispatched: the step's read is issued here, so its DMA queues behind the
    programs that produce the accumulators (already dispatched) and the
    transfer's materialization is a no-op; nothing here blocks."""
    reader = _reader()
    if reader is None:
        return
    try:
        if _due_soon(_STATE["steps"]):
            reader.issue(_STATE["steps"])
    except Exception as e:  # noqa: BLE001 - fall back for good
        logger.warning(
            "expert load | raw readback failed (%s); reading through torch", e
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
                    "earlier, issue %.2f ms (max %.2f), %s buffers changed since "
                    "the last read, compilations %s",
                    st["issues"],
                    step,
                    st["ready_steps"][-1],
                    st["issue_ms"] / max(st["issues"], 1),
                    st["issue_ms_max"],
                    st["rebound"],
                    _compilations(),
                )
        except Exception as e:  # noqa: BLE001 - fall back for good
            logger.warning(
                "expert load | raw readback failed (%s); reading through torch", e
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
