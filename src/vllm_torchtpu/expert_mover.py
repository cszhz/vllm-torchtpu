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


class TorchTransport:
    """Slot slices through torch copies; the fallback and the test path."""

    def d2h_slot(self, tensor: torch.Tensor, slot: int) -> torch.Tensor:
        return tensor[slot].detach().cpu().contiguous()

    def h2d_slot(self, host: torch.Tensor, tensor: torch.Tensor, slot: int) -> None:
        tensor.data[slot].copy_(host.to(tensor.device))

    def d2h_rows(self, tensor: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        return tensor[lo:hi].detach().cpu().contiguous()

    def h2d_rows(
        self, host: torch.Tensor, tensor: torch.Tensor, lo: int, hi: int
    ) -> None:
        tensor.data[lo:hi].copy_(host.to(tensor.device))


class RawTransferTransport(TorchTransport):
    """Slot slices through tpu-sync's raw PJRT copies into pinned host memory.

    Only the slot-major weight tensors (rank 3, one 4 KiB-aligned slice per
    slot) take the raw path; the flat scale tables and any bias go through
    torch, as they are a few hundred KB.
    """

    def __init__(self) -> None:
        from tpu_sync.api.torch import raw_transfer

        self._rt = raw_transfer

    def _pinned_like(self, tensor: torch.Tensor, slot: int) -> torch.Tensor:
        shape = tuple(tensor.shape[1:])
        try:
            return torch.empty(shape, dtype=tensor.dtype, pin_memory=True)
        except (RuntimeError, TypeError):
            return torch.empty(shape, dtype=tensor.dtype)

    def d2h_slot(self, tensor: torch.Tensor, slot: int) -> torch.Tensor:
        if tensor.dim() < 3:
            return super().d2h_slot(tensor, slot)
        host = self._pinned_like(tensor, slot)
        self._rt.transfer_d2h(
            tensor,
            host,
            src_offsets_major_dim=[int(slot)],
            dst_offsets_major_dim=[0],
            copy_sizes_major_dim=[1],
        )
        return host

    def h2d_slot(self, host: torch.Tensor, tensor: torch.Tensor, slot: int) -> None:
        if tensor.dim() < 3:
            super().h2d_slot(host, tensor, slot)
            return
        self._rt.transfer_h2d(
            host.contiguous(),
            tensor,
            src_offsets_major_dim=[0],
            dst_offsets_major_dim=[int(slot)],
            copy_sizes_major_dim=[1],
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
    """(part index, name, tensor, kind) for every expert tensor the module has.
    A module with none is a plan error, not a module with nothing to move."""
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


class SlabMover:
    """Runs a plan's moves for one rank in a background thread (the `Mover`
    protocol of expert_relayout.py)."""

    def __init__(self, rank: int, transport, p2p, *, generation: int = 0) -> None:
        self.rank = rank
        self.transport = transport
        self.p2p = p2p
        self._staged = generation
        self._failed = False
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_report: dict[str, Any] = {}

    # -- Mover protocol

    def start(self, plan, modules) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("a generation is still staging")
        self._cancel.clear()
        self._failed = False
        self._thread = threading.Thread(
            target=self._run, args=(plan, modules), name="expert-mover", daemon=True
        )
        self._thread.start()

    def staged_generation(self) -> int:
        return self._staged

    def failed(self) -> bool:
        return self._failed

    def cancel(self) -> None:
        self._cancel.set()

    def settle(self, generation: int) -> None:
        self._staged = generation

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- the work

    def _run(self, plan, modules) -> None:
        t0 = time.monotonic()
        incoming, outgoing = plan.moves_for(self.rank)
        received_bytes = sent_bytes = 0
        try:
            # Every send is issued (non-blocking) before any receive blocks, so
            # two ranks moving slabs to each other cannot wait on one another;
            # gloo pairs the messages by (peer, tag). The host buffers stay
            # referenced until the sends have completed.
            in_flight = []
            for m in outgoing:
                if self._cancel.is_set():
                    return
                nbytes, handles = self._send_slot(m, modules[(m.group, m.layer)])
                sent_bytes += nbytes
                in_flight.extend(handles)
            staged = []
            for m in incoming:
                if self._cancel.is_set():
                    return
                received_bytes += self._recv_slot(
                    m, modules[(m.group, m.layer)], staged
                )
            for handle, _host in in_flight:
                if handle is not None:
                    handle.wait()
            for host, tensor, slot, kind, lo, hi in staged:
                if self._cancel.is_set():
                    return
                if kind == "slot":
                    self.transport.h2d_slot(host, tensor, slot)
                else:
                    self.transport.h2d_rows(host, tensor, lo, hi)
            self._staged = plan.generation
            self.last_report = {
                "generation": plan.generation,
                "incoming": len(incoming),
                "outgoing": len(outgoing),
                "received_bytes": received_bytes,
                "sent_bytes": sent_bytes,
                "seconds": time.monotonic() - t0,
            }
            logger.info(
                "expert mover | rank %d staged generation %d: %d in / %d out, "
                "%.1f MB received, %.1f MB sent, %.2f s",
                self.rank,
                plan.generation,
                len(incoming),
                len(outgoing),
                received_bytes / 1e6,
                sent_bytes / 1e6,
                time.monotonic() - t0,
            )
        except Exception:  # noqa: BLE001 - reported through failed()
            logger.exception(
                "expert mover | rank %d: generation %d failed",
                self.rank,
                plan.generation,
            )
            self._failed = True

    def _send_slot(self, m, module):
        nbytes = 0
        handles = []
        for part, _, tensor, kind in _parts(module):
            if kind == "slot":
                host = self.transport.d2h_slot(tensor, m.src_slot)
            else:
                rows = tensor.shape[0] // module.local_num_experts
                host = self.transport.d2h_rows(
                    tensor, m.src_slot * rows, (m.src_slot + 1) * rows
                )
            handle = self.p2p.send(
                m.dst_rank, move_tag(m.group, m.layer, m.dst_slot, part), host
            )
            handles.append((handle, host))
            nbytes += host.numel() * host.element_size()
        return nbytes, handles

    def _recv_slot(self, m, module, staged) -> int:
        nbytes = 0
        for part, _, tensor, kind in _parts(module):
            if kind == "slot":
                host = torch.empty(tuple(tensor.shape[1:]), dtype=tensor.dtype)
                lo = hi = 0
            else:
                rows = tensor.shape[0] // module.local_num_experts
                lo, hi = m.dst_slot * rows, (m.dst_slot + 1) * rows
                host = torch.empty((hi - lo, *tensor.shape[1:]), dtype=tensor.dtype)
            self.p2p.recv(
                m.src_rank, move_tag(m.group, m.layer, m.dst_slot, part), host
            )
            staged.append((host, tensor, m.dst_slot, kind, lo, hi))
            nbytes += host.numel() * host.element_size()
        return nbytes
