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
"""Moves expert slabs between ranks for the dynamic re-layout.

A `Plan` (expert_relayout.py) names, per layer, which expert moves from
which rank's slot into which rank's staging slot. The mover runs it in a
background thread, three legs per slab, none of them on the serving steps'
critical path:

1. device -> host on the source: tpu-sync's raw transfer of exactly the
   slot's major-dimension slice of the weight tensor into pinned host memory
   (`transfer_d2h`, paced so the copy never parks the step's kernels), with
   a plain torch copy as the fallback where tpu-sync is not importable;
2. host -> host over a gloo process group of its own (never the group the
   step's all-reduce uses), `isend` / `irecv` of the pinned buffers, tagged
   by (group, layer, destination slot, part);
3. host -> device on the destination into the staging slot (`transfer_h2d`),
   which no table references, so the write is invisible to the steps until
   the controller installs the table.

The per-contraction-block scale tables are small (about 100 KB per slot)
and flat ([slots * rows, N]); the source sends the slot's rows, the
destination writes them with a torch index copy. The bytes copied are the
tensor's own, so the destination's slot holds exactly the source's and the
layer's output after the install is the static layout's bit for bit.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

WEIGHT_PARTS = ("w13_weight", "w2_weight")
SCALE_PARTS = ("w13_weight_scale", "w2_weight_scale")
BIAS_PARTS = ("w13_bias", "w2_bias")
PARTS = WEIGHT_PARTS + SCALE_PARTS + BIAS_PARTS
_GROUP_IDS = {"target": 0, "draft": 1}


def move_tag(group: str, layer: int, dst_slot: int, part: int) -> int:
    """A unique int32 tag per (group, layer, destination slot, part)."""
    return ((_GROUP_IDS.get(group, 2) * 1024 + layer) * 256 + dst_slot) * 8 + part + 1


# ---------------------------------------------------------------------------
# Device <-> host
#
# torch_tpu dispatches device tensors from one thread only (a device buffer
# touched from another thread aborts the process: context_manager.cc, "only
# allowed in the dispatch thread"), so every device leg is ISSUED from the
# controller's step hook on the main thread, asynchronously, and polled there.
# Only host memory crosses to the mover's thread.


class _Done:
    """A future that is already complete (synchronous transports)."""

    def is_ready(self) -> bool:
        return True

    def wait(self) -> None:
        pass


class TorchTransport:
    """Slot slices through torch copies, synchronously: the fallback where
    tpu-sync is not importable, and the test path on CPU tensors."""

    def d2h_slot(self, tensor: torch.Tensor, slot: int):
        return _Done(), tensor[slot].detach().cpu().contiguous()

    def h2d_slot(self, host: torch.Tensor, tensor: torch.Tensor, slot: int):
        tensor.data[slot].copy_(host.to(tensor.device))
        return _Done()

    def d2h_full(self, tensor: torch.Tensor):
        return _Done(), tensor.detach().cpu().contiguous()

    def h2d_full(self, host: torch.Tensor, tensor: torch.Tensor):
        tensor.data.copy_(host.to(tensor.device))
        return _Done()

    @staticmethod
    def ready(future) -> bool:
        return bool(future.is_ready())


class RawTransferTransport(TorchTransport):
    """Slot slices through tpu-sync's raw PJRT copies into pinned host memory,
    asynchronously (a PjRtCopyFuture per copy), paced by the library so a copy
    never parks the step's kernel launches.

    The slot-major weight tensors (rank 3, one 4 KiB-aligned slice per slot)
    take partial copies; the flat scale tables go whole (a few hundred KB).
    """

    def __init__(self) -> None:
        from tpu_sync.api.torch import raw_transfer

        self._rt = raw_transfer

    @staticmethod
    def _pinned(shape, dtype) -> torch.Tensor:
        try:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        except (RuntimeError, TypeError):
            return torch.empty(shape, dtype=dtype)

    def d2h_slot(self, tensor: torch.Tensor, slot: int):
        if tensor.dim() < 3:
            return super().d2h_slot(tensor, slot)
        host = self._pinned(tuple(tensor.shape[1:]), tensor.dtype)
        future = self._rt.transfer_d2h_async(
            tensor,
            host,
            src_offsets_major_dim=[int(slot)],
            dst_offsets_major_dim=[0],
            copy_sizes_major_dim=[1],
        )
        return future, host

    def h2d_slot(self, host: torch.Tensor, tensor: torch.Tensor, slot: int):
        if tensor.dim() < 3:
            return super().h2d_slot(host, tensor, slot)
        return self._rt.transfer_h2d_async(
            host.contiguous(),
            tensor,
            src_offsets_major_dim=[0],
            dst_offsets_major_dim=[int(slot)],
            copy_sizes_major_dim=[1],
        )

    def d2h_full(self, tensor: torch.Tensor):
        host = self._pinned(tuple(tensor.shape), tensor.dtype)
        return self._rt.transfer_d2h_async(tensor, host), host

    def h2d_full(self, host: torch.Tensor, tensor: torch.Tensor):
        return self._rt.transfer_h2d_async(host.contiguous(), tensor)


def make_transport(kind: str):
    if kind == "torch":
        return TorchTransport()
    try:
        return RawTransferTransport()
    except ImportError:
        logger.warning(
            "expert mover: tpu_sync.api.torch.raw_transfer not importable; "
            "slot copies go through torch"
        )
        return TorchTransport()


# ---------------------------------------------------------------------------
# Host <-> host


class GlooP2P:
    """Point-to-point tensors over a dedicated gloo group. `rank_map` turns
    the plan's EP ranks into the global ranks torch.distributed addresses."""

    def __init__(self, group, rank_map: list[int] | None = None) -> None:
        self.group = group
        self.rank_map = rank_map

    def _global(self, rank: int) -> int:
        return int(self.rank_map[rank]) if self.rank_map is not None else int(rank)

    def send(self, dst: int, tag: int, tensor: torch.Tensor):
        """Non-blocking: two ranks sending to each other must not rendezvous
        on blocking sends. Returns the work handle to wait on."""
        return torch.distributed.isend(
            tensor.contiguous(), dst=self._global(dst), group=self.group, tag=tag
        )

    def recv(self, src: int, tag: int, tensor: torch.Tensor) -> None:
        torch.distributed.recv(tensor, src=self._global(src), group=self.group, tag=tag)


def make_gloo_group(global_ranks: list[int]):
    """The mover's own process group over the EP ranks (a collective: every
    rank calls it, once, at arm time)."""
    return torch.distributed.new_group(
        ranks=[int(r) for r in global_ranks], backend="gloo"
    )


class LocalExchange:
    """An in-process mailbox standing in for the gloo group in tests: ranks
    are threads, a send parks the tensor under (src, dst, tag) and a recv
    waits for it."""

    def __init__(self) -> None:
        self._box: dict[tuple[int, int, int], torch.Tensor] = {}
        self._cv = threading.Condition()

    def endpoint(self, rank: int) -> _LocalEndpoint:
        return _LocalEndpoint(self, rank)

    def _send(self, src, dst, tag, tensor):
        with self._cv:
            self._box[(src, dst, tag)] = tensor.clone()
            self._cv.notify_all()

    def _recv(self, src, dst, tag, tensor, timeout=30.0):
        deadline = time.monotonic() + timeout
        with self._cv:
            while (src, dst, tag) not in self._box:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"no message {(src, dst, tag)}")
                self._cv.wait(left)
            tensor.copy_(self._box.pop((src, dst, tag)))


class _LocalEndpoint:
    def __init__(self, exchange: LocalExchange, rank: int) -> None:
        self.exchange = exchange
        self.rank = rank

    def send(self, dst, tag, tensor):
        self.exchange._send(self.rank, dst, tag, tensor)
        return None

    def recv(self, src, tag, tensor):
        self.exchange._recv(src, self.rank, tag, tensor)


# ---------------------------------------------------------------------------
# The mover


def _parts(module) -> list[tuple[int, str, torch.Tensor, str]]:
    """(part index, name, tensor, kind) for every expert tensor the module has:
    kind "slot" for slot-major tensors, "rows" for the flat scale tables. A
    module with none is a plan error, not a module with nothing to move."""
    out = []
    for i, name in enumerate(PARTS):
        t = getattr(module, name, None)
        if t is None:
            continue
        kind = "slot" if name in WEIGHT_PARTS or name in BIAS_PARTS else "rows"
        out.append((i, name, t.data if isinstance(t, torch.nn.Parameter) else t, kind))
    if not out:
        raise RuntimeError(f"{module!r} has none of the expert tensors {PARTS}")
    return out


def _rows_per_slot(tensor: torch.Tensor, module) -> int:
    return int(tensor.shape[0]) // int(module.local_num_experts)


class SlabMover:
    """Runs a plan's moves for one rank (the `Mover` protocol of
    expert_relayout.py), in three phases driven by `pump()` from the step
    hook on the main thread:

    d2h   issue the source slabs' device -> host copies (up to
          `slabs_per_pump` slot slices per call, plus the flat scale tables
          of every layer that sends or receives), poll them;
    p2p   a thread sends every outgoing host buffer (non-blocking), receives
          every incoming one, and patches the received scale rows into the
          host copies of this rank's tables;
    h2d   issue the staging slots' host -> device copies and the patched
          scale tables', poll them; then the generation is staged.
    """

    def __init__(
        self, rank: int, transport, p2p, *, generation: int = 0, slabs_per_pump: int = 4
    ):
        self.rank = rank
        self.transport = transport
        self.p2p = p2p
        self.slabs_per_pump = max(int(slabs_per_pump), 1)
        self._staged = generation
        self._failed = False
        self._cancel = threading.Event()
        self._phase = "idle"
        self._thread: threading.Thread | None = None
        self._thread_done = threading.Event()
        self._thread_error: BaseException | None = None
        self.last_report: dict[str, Any] = {}
        self._reset_work()

    def _reset_work(self) -> None:
        self._gen = None
        self._plan = None
        self._modules = {}
        self._t0 = 0.0
        self._out_todo: list = []  # (move, part, tensor, kind)
        self._out_inflight: list = []  # (future, host, move, part)
        self._out_ready: list = []  # (move, part, host)
        self._tables: dict = {}  # (key, part) -> [future, host, tensor, rows]
        self._in_slots: list = []  # (move, part, tensor, host)
        self._in_rows: list = []  # (move, part, key, lo, hi, host)
        self._h2d_todo: list = []  # (host, tensor, slot|None)
        self._h2d_inflight: list = []

    # -- Mover protocol

    def start(self, plan, modules) -> None:
        """Set the generation up; a plan the modules cannot serve marks the
        generation failed rather than raising into the step."""
        if self._phase != "idle":
            raise RuntimeError("a generation is still staging")
        self._reset_work()
        self._cancel.clear()
        self._failed = False
        self._gen = plan.generation
        self._plan = plan
        self._modules = modules
        self._t0 = time.monotonic()
        try:
            self._prepare(plan, modules)
        except Exception:  # noqa: BLE001 - reported through failed()
            logger.exception(
                "expert mover | rank %d: generation %s cannot start",
                self.rank,
                plan.generation,
            )
            self._failed = True
            self._phase = "idle"

    def _prepare(self, plan, modules) -> None:
        incoming, outgoing = plan.moves_for(self.rank)
        for m in outgoing:
            module = modules[(m.group, m.layer)]
            for part, _, tensor, kind in _parts(module):
                if kind == "slot":
                    self._out_todo.append((m, part, tensor, kind))
                else:
                    self._tables.setdefault(
                        ((m.group, m.layer), part),
                        [None, None, tensor, _rows_per_slot(tensor, module)],
                    )
        for m in incoming:
            module = modules[(m.group, m.layer)]
            for part, _, tensor, kind in _parts(module):
                if kind == "slot":
                    host = torch.empty(tuple(tensor.shape[1:]), dtype=tensor.dtype)
                    self._in_slots.append((m, part, tensor, host))
                else:
                    rows = _rows_per_slot(tensor, module)
                    key = ((m.group, m.layer), part)
                    self._tables.setdefault(key, [None, None, tensor, rows])
                    host = torch.empty((rows, *tensor.shape[1:]), dtype=tensor.dtype)
                    self._in_rows.append(
                        (m, part, key, m.dst_slot * rows, (m.dst_slot + 1) * rows, host)
                    )
        self._out_moves = outgoing
        self._in_moves = incoming
        self._phase = "d2h"
        if not outgoing and not incoming and not self._tables:
            self._finish()

    def pump(self) -> None:
        """One bounded slice of device work, from the main thread."""
        if self._phase == "idle" or self._failed:
            return
        if self._cancel.is_set():
            self._phase = "idle"
            return
        try:
            getattr(self, f"_pump_{self._phase}")()
        except Exception:  # noqa: BLE001 - reported through failed()
            logger.exception(
                "expert mover | rank %d: generation %s failed", self.rank, self._gen
            )
            self._failed = True
            self._phase = "idle"

    def staged_generation(self) -> int:
        return self._staged

    def failed(self) -> bool:
        return self._failed

    def cancel(self) -> None:
        self._cancel.set()

    def settle(self, generation: int) -> None:
        self._staged = generation
        if self._phase != "p2p":
            self._phase = "idle"

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- phases

    def _pump_d2h(self) -> None:
        # Tables first (every layer touched), then slot slices, a few per call.
        budget = self.slabs_per_pump
        for key, entry in self._tables.items():
            if entry[0] is None and budget > 0:
                entry[0], entry[1] = self.transport.d2h_full(entry[2])
                budget -= 1
        while self._out_todo and budget > 0:
            m, part, tensor, _ = self._out_todo.pop(0)
            future, host = self.transport.d2h_slot(tensor, m.src_slot)
            self._out_inflight.append((future, host, m, part))
            budget -= 1
        still = []
        for future, host, m, part in self._out_inflight:
            if self.transport.ready(future):
                self._out_ready.append((m, part, host))
            else:
                still.append((future, host, m, part))
        self._out_inflight = still
        tables_ready = all(
            e[0] is not None and self.transport.ready(e[0])
            for e in self._tables.values()
        )
        if not self._out_todo and not self._out_inflight and tables_ready:
            self._thread_done.clear()
            self._thread_error = None
            self._thread = threading.Thread(
                target=self._exchange, name="expert-mover", daemon=True
            )
            self._thread.start()
            self._phase = "p2p"

    def _exchange(self) -> None:
        """The host-to-host leg, on the mover's thread: CPU memory only."""
        try:
            in_flight = []
            for m, part, host in self._out_ready:
                h = self.p2p.send(
                    m.dst_rank, move_tag(m.group, m.layer, m.dst_slot, part), host
                )
                in_flight.append((h, host))
            for m in self._out_moves:
                for key, entry in self._tables.items():
                    if key[0] != (m.group, m.layer):
                        continue
                    rows = entry[3]
                    chunk = entry[1][
                        m.src_slot * rows : (m.src_slot + 1) * rows
                    ].contiguous()
                    h = self.p2p.send(
                        m.dst_rank,
                        move_tag(m.group, m.layer, m.dst_slot, key[1]),
                        chunk,
                    )
                    in_flight.append((h, chunk))
            for m, part, tensor, host in self._in_slots:
                if self._cancel.is_set():
                    return
                self.p2p.recv(
                    m.src_rank, move_tag(m.group, m.layer, m.dst_slot, part), host
                )
            for m, part, key, lo, hi, host in self._in_rows:
                if self._cancel.is_set():
                    return
                self.p2p.recv(
                    m.src_rank, move_tag(m.group, m.layer, m.dst_slot, part), host
                )
                self._tables[key][1][lo:hi] = host
            for h, _ in in_flight:
                if h is not None:
                    h.wait()
        except BaseException as e:  # noqa: BLE001 - handed to the main thread
            self._thread_error = e
        finally:
            self._thread_done.set()

    def _pump_p2p(self) -> None:
        if not self._thread_done.is_set():
            return
        if self._thread_error is not None:
            raise RuntimeError(f"host exchange failed: {self._thread_error!r}")
        if self._cancel.is_set():
            self._phase = "idle"
            return
        for m, part, tensor, host in self._in_slots:
            self._h2d_todo.append((host, tensor, m.dst_slot))
        touched = {(m.group, m.layer) for m in self._in_moves}
        for key, entry in self._tables.items():
            if key[0] in touched:
                self._h2d_todo.append((entry[1], entry[2], None))
        self._phase = "h2d"
        self._pump_h2d()

    def _pump_h2d(self) -> None:
        budget = self.slabs_per_pump
        while self._h2d_todo and budget > 0:
            host, tensor, slot = self._h2d_todo.pop(0)
            if slot is None:
                self._h2d_inflight.append(self.transport.h2d_full(host, tensor))
            else:
                self._h2d_inflight.append(self.transport.h2d_slot(host, tensor, slot))
            budget -= 1
        self._h2d_inflight = [
            f for f in self._h2d_inflight if not self.transport.ready(f)
        ]
        if not self._h2d_todo and not self._h2d_inflight:
            self._finish()

    def _finish(self) -> None:
        received = sum(h.numel() * h.element_size() for *_, h in self._in_slots) + sum(
            h.numel() * h.element_size() for *_, h in self._in_rows
        )
        sent = sum(h.numel() * h.element_size() for _, _, h in self._out_ready)
        for m in self._out_moves:
            for key, entry in self._tables.items():
                if key[0] == (m.group, m.layer) and entry[1] is not None:
                    rows, host = entry[3], entry[1]
                    sent += rows * host[0].numel() * host.element_size()
        seconds = time.monotonic() - self._t0
        self.last_report = {
            "generation": self._gen,
            "incoming": len(self._in_moves),
            "outgoing": len(self._out_moves),
            "received_bytes": received,
            "sent_bytes": sent,
            "seconds": seconds,
        }
        logger.info(
            "expert mover | rank %d staged generation %s: %d in / %d out, "
            "%.1f MB received, %.1f MB sent, %.2f s",
            self.rank,
            self._gen,
            len(self._in_moves),
            len(self._out_moves),
            received / 1e6,
            sent / 1e6,
            seconds,
        )
        self._staged = self._gen
        self._phase = "idle"
