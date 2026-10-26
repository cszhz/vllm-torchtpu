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
# Device <-> host, by byte regions of a tensor's STORAGE
#
# The fused path's weights and their scales come back from the quantizer's
# jax_op as a tuple, i.e. as views into one shared output storage, and
# tpu-sync's raw transfer refuses a view ("must start at storage offset 0",
# "does not span its entire base storage buffer"). torch_tpu copies of such
# views are not safe either (a clone of an offset view returned the wrong
# bytes on the chips). So the transport never looks at a tensor's dtype or
# shape: it reinterprets the tensor's whole storage as [n, 32, 128] uint8 --
# 4 KiB per major row, the uint8 tile exactly, so physical equals logical --
# and copies slot regions as 4 KiB-aligned byte ranges with the raw transfer's
# major-dimension offsets. A slot of a slot-major tensor is one contiguous
# region of the storage at the tensor's storage offset plus slot x slot bytes.
#
# torch_tpu dispatches device tensors from one thread only, so every device
# leg is ISSUED from the controller's step hook on the main thread,
# asynchronously, and polled there. Only host memory crosses to the mover's
# thread.

BLOCK = 4096
_TILE = (32, 128)


def storage_blocks(t: torch.Tensor) -> torch.Tensor:
    """The tensor's whole storage as [n, 32, 128] uint8 (4 KiB per row)."""
    storage = t.untyped_storage()
    nbytes = storage.nbytes()
    if nbytes % BLOCK:
        raise ValueError(
            f"storage of {nbytes} bytes is not a whole number of 4 KiB blocks"
        )
    flat = torch.empty(0, dtype=torch.uint8, device=t.device)
    flat.set_(storage)
    return flat.view(nbytes // BLOCK, *_TILE)


def region_of(
    t: torch.Tensor, first: int, count: int, unit_elems: int
) -> tuple[int, int]:
    """(byte offset into the storage, byte length) of `count` units of
    `unit_elems` elements starting at unit `first` of tensor `t` (a slot of a
    slot-major tensor, or rows of a flat table). Both 4 KiB aligned."""
    elsize = t.element_size()
    off = (t.storage_offset() + first * unit_elems) * elsize
    nbytes = count * unit_elems * elsize
    if off % BLOCK or nbytes % BLOCK:
        raise ValueError(
            f"region at byte {off} of {nbytes} bytes is not 4 KiB aligned; the raw "
            "transfer copies whole 4 KiB blocks"
        )
    return off, nbytes


class _Done:
    """A future that is already complete (synchronous transports)."""

    def is_ready(self) -> bool:
        return True

    def wait(self) -> None:
        pass


class TorchTransport:
    """Byte regions through torch on the storage reinterpretation: the test
    path on CPU tensors, and the fallback where tpu-sync is not importable."""

    def d2h(self, t: torch.Tensor, off: int, nbytes: int):
        blocks = storage_blocks(t).view(-1)
        return _Done(), blocks[off : off + nbytes].detach().cpu().clone()

    def h2d(self, host: torch.Tensor, t: torch.Tensor, off: int):
        blocks = storage_blocks(t).view(-1)
        blocks[off : off + host.numel()].copy_(host.to(t.device))
        return _Done()

    @staticmethod
    def ready(future) -> bool:
        return bool(future.is_ready())


class RawTransferTransport(TorchTransport):
    """Byte regions through tpu-sync's raw PJRT copies into pinned host
    memory, asynchronously (a PjRtCopyFuture per copy), paced by the library
    so a copy never parks the step's kernel launches."""

    def __init__(self) -> None:
        from tpu_sync.api.torch import raw_transfer

        self._rt = raw_transfer

    @staticmethod
    def _pinned(nblocks: int) -> torch.Tensor:
        try:
            return torch.empty((nblocks, *_TILE), dtype=torch.uint8, pin_memory=True)
        except (RuntimeError, TypeError):
            return torch.empty((nblocks, *_TILE), dtype=torch.uint8)

    def d2h(self, t: torch.Tensor, off: int, nbytes: int):
        host = self._pinned(nbytes // BLOCK)
        future = self._rt.transfer_d2h_async(
            storage_blocks(t),
            host,
            src_offsets_major_dim=[off // BLOCK],
            dst_offsets_major_dim=[0],
            copy_sizes_major_dim=[nbytes // BLOCK],
        )
        return future, host.view(-1)

    def h2d(self, host: torch.Tensor, t: torch.Tensor, off: int):
        blocks = host.contiguous().view(-1, *_TILE)
        return self._rt.transfer_h2d_async(
            blocks,
            storage_blocks(t),
            src_offsets_major_dim=[0],
            dst_offsets_major_dim=[off // BLOCK],
            copy_sizes_major_dim=[blocks.shape[0]],
        )


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


def slot_region(t: torch.Tensor, slot: int) -> tuple[int, int]:
    unit = 1
    for d in t.shape[1:]:
        unit *= int(d)
    return region_of(t, slot, 1, unit)


def rows_region(t: torch.Tensor, module, slot: int) -> tuple[int, int]:
    rows = int(t.shape[0]) // int(module.local_num_experts)
    unit = 1
    for d in t.shape[1:]:
        unit *= int(d)
    return region_of(t, slot * rows, rows, unit)


def describe_parts(module) -> str:
    """One line per expert tensor: shape, dtype, storage offset and size --
    what the raw transfer will address (logged at arm for the first layer)."""
    lines = []
    for _, name, t, kind in _parts(module):
        lines.append(
            f"{name}: {tuple(t.shape)} {str(t.dtype).split('.')[-1]} {kind}, "
            f"storage {t.untyped_storage().nbytes()} B at offset "
            f"{t.storage_offset() * t.element_size()} B"
        )
    return "; ".join(lines)


class SlabMover:
    """Runs a plan's moves for one rank (the `Mover` protocol of
    expert_relayout.py), in three phases driven by `pump()` from the step
    hook on the main thread:

    d2h   issue the source regions' device -> host copies (up to
          `slabs_per_pump` per call), poll them;
    p2p   a thread sends every outgoing host buffer (non-blocking) and
          receives every incoming one;
    h2d   issue the staging regions' host -> device copies, poll them; then
          the generation is staged.

    Every region is a 4 KiB-aligned byte range of a tensor's storage: a slot
    of a slot-major weight, or a slot's rows of a flat scale table.
    """

    def __init__(
        self, rank: int, transport, p2p, *, generation: int = 0, slabs_per_pump: int = 6
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
        self._t0 = 0.0
        self._out_moves: list = []
        self._in_moves: list = []
        self._out_todo: list = []  # (move, part, tensor, off, nbytes)
        self._out_inflight: list = []  # (future, host, move, part)
        self._out_ready: list = []  # (move, part, host)
        self._in_regions: list = []  # (move, part, tensor, off, host)
        self._h2d_todo: list = []  # (host, tensor, off)
        self._h2d_inflight: list = []

    @staticmethod
    def _region(tensor, kind, module, slot):
        return (
            slot_region(tensor, slot)
            if kind == "slot"
            else rows_region(tensor, module, slot)
        )

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
                off, nbytes = self._region(tensor, kind, module, m.src_slot)
                self._out_todo.append((m, part, tensor, off, nbytes))
        for m in incoming:
            module = modules[(m.group, m.layer)]
            for part, _, tensor, kind in _parts(module):
                off, nbytes = self._region(tensor, kind, module, m.dst_slot)
                self._in_regions.append(
                    (m, part, tensor, off, torch.empty(nbytes, dtype=torch.uint8))
                )
        self._out_moves = outgoing
        self._in_moves = incoming
        self._phase = "d2h"
        if not outgoing and not incoming:
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
        budget = self.slabs_per_pump
        while self._out_todo and budget > 0:
            m, part, tensor, off, nbytes = self._out_todo.pop(0)
            future, host = self.transport.d2h(tensor, off, nbytes)
            self._out_inflight.append((future, host, m, part))
            budget -= 1
        still = []
        for future, host, m, part in self._out_inflight:
            if self.transport.ready(future):
                self._out_ready.append((m, part, host))
            else:
                still.append((future, host, m, part))
        self._out_inflight = still
        if not self._out_todo and not self._out_inflight:
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
            for m, part, _tensor, _off, host in self._in_regions:
                if self._cancel.is_set():
                    return
                self.p2p.recv(
                    m.src_rank, move_tag(m.group, m.layer, m.dst_slot, part), host
                )
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
        self._h2d_todo = [
            (host, tensor, off) for _m, _p, tensor, off, host in self._in_regions
        ]
        self._phase = "h2d"
        self._pump_h2d()

    def _pump_h2d(self) -> None:
        budget = self.slabs_per_pump
        while self._h2d_todo and budget > 0:
            host, tensor, off = self._h2d_todo.pop(0)
            self._h2d_inflight.append(self.transport.h2d(host, tensor, off))
            budget -= 1
        self._h2d_inflight = [
            f for f in self._h2d_inflight if not self.transport.ready(f)
        ]
        if not self._h2d_todo and not self._h2d_inflight:
            self._finish()

    def _finish(self) -> None:
        received = sum(h.numel() for *_, h in self._in_regions)
        sent = sum(h.numel() for _, _, h in self._out_ready)
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
