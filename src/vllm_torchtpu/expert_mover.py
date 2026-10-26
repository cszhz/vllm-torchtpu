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

import ctypes
import queue
import threading
import time
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The expert tensors of a placed layer, by part id (what move_tag carries).
# Each part names its attribute candidates in order: the fused EP kernel reads
# the row-padded flat scale tables (`_tpu_fused_*_scale`, mxfp4.py), not the
# quantizer's [E, blocks, 1, N] scale Parameters, so those are what move.
PART_ALIASES = (
    ("w13_weight",),
    ("w2_weight",),
    ("_tpu_fused_w13_scale", "w13_weight_scale"),
    ("_tpu_fused_w2_scale", "w2_weight_scale"),
    ("w13_bias",),
    ("w2_bias",),
)
PARTS = tuple(aliases[0] for aliases in PART_ALIASES)
WEIGHT_PARTS = PARTS[0:2]
SCALE_PARTS = PARTS[2:4]
BIAS_PARTS = PARTS[4:6]
_GROUP_IDS = {"target": 0, "draft": 1}


def move_tag(group: str, layer: int, dst_slot: int, part: int) -> int:
    """A unique int32 tag per (group, layer, destination slot, part)."""
    return ((_GROUP_IDS.get(group, 2) * 1024 + layer) * 256 + dst_slot) * 8 + part + 1


# ---------------------------------------------------------------------------
# Device <-> host, by major-dimension slices of a tensor
#
# tpu-sync's raw transfer copies a tensor's major-dimension slices as the
# PHYSICAL bytes of its device buffer (tiled, packed), paced, asynchronously,
# straight into the live storage the compiled program reads (compiled mode
# looks each argument's buffer up per call, so a DMA into a staging slot is
# what the next step sees once the slot table points at it). A slot of a
# slot-major weight is one slice; its bytes mean nothing on the host but land
# byte-for-byte in a slot of a same-shaped tensor on another rank.
#
# The rows of a flat scale table ([slots * rows, N]) cannot be a partial copy
# (rank-2 partial copies are refused: one row is a fraction of a tile row), so
# a table moves whole: D2H the table, cut or patch the slot's rows in the host
# image (rows is a multiple of the f32 tile's 8 rows, so a slot's rows are one
# contiguous physical range), H2D the table back. The slice byte size comes
# from raw_transfer.major_slice_bytes: the device layout can pad the minor
# dims, and the fp4 pair dtype's storage is accounted at one byte per nibble,
# so nbytes / shape[0] is not it in general.
#
# torch_tpu dispatches device tensors from one thread only, so every device
# leg is ISSUED from the controller's step hook on the main thread,
# asynchronously, and polled there. Only host memory crosses to the mover's
# thread.


class _Done:
    """A future that is already complete (synchronous transports)."""

    def is_ready(self) -> bool:
        return True

    def wait(self) -> None:
        pass


def _pinned(shape) -> torch.Tensor:
    try:
        return torch.empty(shape, dtype=torch.uint8, pin_memory=True)
    except (RuntimeError, TypeError):
        return torch.empty(shape, dtype=torch.uint8)


# RawHostBuffer owners by the data pointer of the tensor viewing them: the
# pool keeps the tensors, this keeps their memory alive.
_RAW_HOST_OWNERS: dict[int, object] = {}


def _raw_host(rt, shape, anchor: torch.Tensor) -> torch.Tensor:
    """A uint8 tensor over tpu-sync's own pinned host memory (`RawHostBuffer`
    in PJRT's pinned_host space, the memory libtpu DMAs to without a staged
    copy), the way the Engram host transfer allocates. torch's
    `pin_memory=True` is whatever the accelerator hooks make of it. The buffer
    allocates when bound to a device, which `PreparedTorchRawTransfer` does
    against `anchor`, a tiny tensor on the weights' device."""
    nbytes = 1
    for d in shape:
        nbytes *= int(d)
    buf = rt.RawHostBuffer(nbytes)
    rt.PreparedTorchRawTransfer(anchor, buf)  # binds (allocates); the hold is dropped
    owner = (ctypes.c_uint8 * buf.size_bytes).from_address(buf.data_ptr)
    t = torch.frombuffer(owner, dtype=torch.uint8)[:nbytes].view(*shape)
    _RAW_HOST_OWNERS[t.data_ptr()] = buf
    return t


class TorchTransport:
    """Slices through torch, as logical bytes: the test path on CPU tensors,
    and the fallback where tpu-sync is not importable. Host buffers are
    [count, slice bytes] uint8; `release` hands one back for reuse (a no-op
    here, a pool in the raw transport, whose pinned allocations are slow)."""

    def slice_bytes(self, t: torch.Tensor) -> int:
        return t.nbytes // int(t.shape[0])

    def release(self, host: torch.Tensor) -> None:
        pass

    def release_pending(self) -> None:
        pass

    def d2h(self, t: torch.Tensor, first: int, count: int):
        rows = t[first : first + count].detach().contiguous()
        host = rows.view(torch.uint8).cpu().reshape(count, -1).clone()
        return _Done(), host

    def h2d(self, host: torch.Tensor, t: torch.Tensor, first: int):
        count = int(host.shape[0])
        dst = t[first : first + count].view(torch.uint8)
        dst.copy_(host.reshape(dst.shape).to(t.device))
        return _Done()

    @staticmethod
    def ready(future) -> bool:
        return bool(future.is_ready())


class RawTransferTransport(TorchTransport):
    """Slices through tpu-sync's raw PJRT copies into pinned host memory,
    asynchronously (a PjRtCopyFuture per copy), paced by the library so a copy
    never parks the step's kernel launches."""

    def __init__(
        self, host_memory: str = "torch", skip_buffer_lock: bool = False
    ) -> None:
        from tpu_sync.api.torch import raw_transfer

        if host_memory not in ("torch", "raw"):
            raise ValueError(
                f"host_memory must be 'torch' or 'raw'; got {host_memory!r}"
            )
        self.host_memory = host_memory
        # tpu-sync's buffer lock (a PJRT hold on the weight) costs a call to
        # acquire and may queue behind the device's use of the weight; the
        # weights are only ever written by this mover, into slots the running
        # program does not read, which is why tpu-sync's own WeightSynchronizer
        # skips the lock by default.
        self.skip_buffer_lock = bool(skip_buffer_lock)
        self._anchor: torch.Tensor | None = None  # binds RawHostBuffers to the device
        # Pinning host memory costs milliseconds per buffer, and the whole
        # scale-table images are tens of MB each, so buffers are pooled by
        # shape across generations rather than pinned anew on the main thread.
        self._pool: dict[tuple[int, ...], list[torch.Tensor]] = {}
        self._owned: set[int] = set()  # data pointers of buffers this pool made
        self._lock = threading.Lock()
        # Pinned copies made for unpinned h2d sources; `release_pending`
        # returns them to the pool once the mover's generation is done.
        self._pending_release: list[torch.Tensor] = []
        try:
            # The module's __getattr__ loads the extension for unknown names,
            # so an old tpu-sync answers this with an import error, not an
            # attribute error.
            usable = callable(raw_transfer.major_slice_bytes)
        except Exception as e:  # noqa: BLE001 - any failure means "not usable"
            raise ImportError(
                "tpu_sync.api.torch.raw_transfer has no usable major_slice_bytes: "
                "the installed tpu-sync predates its fp4 (float4_e2m1fn_x2) "
                f"support ({e!r})"
            ) from e
        if not usable:
            raise ImportError(
                "tpu_sync.api.torch.raw_transfer.major_slice_bytes is not callable"
            )
        self._rt = raw_transfer

    def slice_bytes(self, t: torch.Tensor) -> int:
        return int(
            self._rt.major_slice_bytes(t, unsafe_skip_buffer_lock=self.skip_buffer_lock)
        )

    def _take(self, shape: tuple[int, ...], device=None) -> torch.Tensor:
        with self._lock:
            free = self._pool.get(shape)
            if free:
                return free.pop()
        if self.host_memory == "raw":
            if self._anchor is None:
                self._anchor = torch.zeros(8, dtype=torch.uint8, device=device)
            host = _raw_host(self._rt, shape, self._anchor)
        else:
            host = _pinned(shape)
        with self._lock:
            self._owned.add(host.data_ptr())
        return host

    def _ours(self, host: torch.Tensor) -> bool:
        with self._lock:
            if host.data_ptr() in self._owned:
                return True
        try:
            return bool(host.is_pinned())
        except Exception:  # noqa: BLE001 - no accelerator hooks: not pinned
            return False

    def release(self, host: torch.Tensor) -> None:
        if not self._ours(host):
            return
        with self._lock:
            self._pool.setdefault(tuple(host.shape), []).append(host)

    def release_pending(self) -> None:
        pending, self._pending_release = self._pending_release, []
        for host in pending:
            self.release(host)

    def pooled_bytes(self) -> int:
        with self._lock:
            return sum(t.numel() for free in self._pool.values() for t in free)

    def d2h(self, t: torch.Tensor, first: int, count: int):
        host = self._take((count, self.slice_bytes(t)), t.device)
        future = self._rt.transfer_d2h_async(
            t,
            host,
            src_offsets_major_dim=[first],
            dst_offsets_major_dim=[0],
            copy_sizes_major_dim=[count],
            unsafe_skip_buffer_lock=self.skip_buffer_lock,
        )
        return future, host

    def h2d(self, host: torch.Tensor, t: torch.Tensor, first: int):
        if not self._ours(host):
            pinned = self._take(tuple(host.shape), t.device)
            pinned.copy_(host)
            host = pinned
            self._pending_release.append(pinned)
        return self._rt.transfer_h2d_async(
            host,
            t,
            src_offsets_major_dim=[0],
            dst_offsets_major_dim=[first],
            copy_sizes_major_dim=[int(host.shape[0])],
            unsafe_skip_buffer_lock=self.skip_buffer_lock,
        )


def _on_tpu() -> bool:
    try:
        return (
            torch.accelerator.is_available()
            and torch.accelerator.current_accelerator().type == "tpu"
        )
    except Exception:  # noqa: BLE001 - no accelerator stack at all
        return False


def make_transport(
    kind: str, host_memory: str = "torch", skip_buffer_lock: bool = False
):
    """`raw_transfer` (the default) or `torch`. Without a usable tpu-sync the
    raw transport falls back to torch only where there is no TPU (tests,
    chip-less hosts): on a device torch cannot move the fp4 pair dtype, so
    the error surfaces and the dynamic re-layout stays off. `host_memory`
    picks the raw transport's host buffers: `torch` (pin_memory=True) or
    `raw` (tpu-sync's RawHostBuffer); `skip_buffer_lock` drops tpu-sync's
    per-transfer buffer lock on the weights."""
    if kind == "torch":
        return TorchTransport()
    try:
        return RawTransferTransport(
            host_memory=host_memory, skip_buffer_lock=skip_buffer_lock
        )
    except ImportError as e:
        if _on_tpu():
            raise
        logger.warning(
            "expert mover: tpu-sync raw transfer unavailable (%s); slot copies go "
            "through torch on this chip-less host",
            e,
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

    def irecv(self, src: int, tag: int, tensor: torch.Tensor):
        """Non-blocking receive into `tensor`; the handle answers
        `is_completed()` (the exchange thread polls it) and `wait()`."""
        return torch.distributed.irecv(
            tensor, src=self._global(src), group=self.group, tag=tag
        )


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

    def irecv(self, src, tag, tensor):
        return _LocalRecv(self.exchange, src, self.rank, tag, tensor)


class _LocalRecv:
    """A non-blocking receive on the mailbox: completes when the message is
    there, copying it out."""

    def __init__(self, exchange, src, dst, tag, tensor) -> None:
        self.exchange, self.key, self.tensor = exchange, (src, dst, tag), tensor
        self.done = False

    def is_completed(self) -> bool:
        if not self.done:
            with self.exchange._cv:
                msg = self.exchange._box.pop(self.key, None)
            if msg is not None:
                self.tensor.copy_(msg)
                self.done = True
        return self.done

    def wait(self) -> None:
        if not self.done:
            self.exchange._recv(*self.key, self.tensor)
            self.done = True


# ---------------------------------------------------------------------------
# The mover

SLOT, ROWS = "slot", "rows"


def _parts(module) -> list[tuple[int, str, torch.Tensor, str]]:
    """(part id, attribute, tensor, kind) for every expert tensor the module
    has: SLOT for slot-major tensors (rank >= 3, one slot = one major slice),
    ROWS for the flat scale tables (rank 2, a slot = a block of rows). A module
    with none is a plan error, not a module with nothing to move."""
    out = []
    for part, aliases in enumerate(PART_ALIASES):
        for name in aliases:
            t = getattr(module, name, None)
            if t is None:
                continue
            t = t.data if isinstance(t, torch.nn.Parameter) else t
            if t.dim() >= 3:
                kind = SLOT
            elif t.dim() == 2 and aliases[0] in SCALE_PARTS:
                kind = ROWS
            else:
                raise RuntimeError(
                    f"{name} of shape {tuple(t.shape)} is not movable: slots are "
                    "major slices of a rank >= 3 tensor or row blocks of a flat "
                    "scale table"
                )
            out.append((part, name, t, kind))
            break
    if not out:
        raise RuntimeError(f"{module!r} has none of the expert tensors {PARTS}")
    return out


def _slot_count(module) -> int:
    """Physical slots on this rank: the weights' leading dimension."""
    for name in WEIGHT_PARTS:
        t = getattr(module, name, None)
        if t is not None:
            return int(t.shape[0])
    return int(module.local_num_experts)


def describe_parts(module, transport=None) -> str:
    """One line per expert tensor: shape, dtype, kind and (with a transport)
    the device bytes of one slot -- logged at arm for the first layer."""
    lines = []
    slots = _slot_count(module)
    for _, name, t, kind in _parts(module):
        line = f"{name}: {tuple(t.shape)} {str(t.dtype).split('.')[-1]} {kind}"
        if transport is not None:
            per_slice = transport.slice_bytes(t)
            per_slot = (
                per_slice if kind == SLOT else per_slice * (int(t.shape[0]) // slots)
            )
            line += f", {per_slot / 1e6:.2f} MB/slot"
        lines.append(line)
    return "; ".join(lines)


class SlabMover:
    """Runs a plan's moves for one rank (the `Mover` protocol of
    expert_relayout.py) as one streaming pipeline driven by `pump()` from the
    runner right after each step's forward is dispatched:

    * device -> host: the source slots' copies and the whole-table copies of
      every scale table a move touches are issued under `step_budget_bytes`
      per pump, tables first; a payload is handed to the exchange thread the
      pump it lands (a table's outgoing rows are cut from its image);
    * host -> host: a thread posts every receive up front, sends payloads as
      they come, and reports arrivals;
    * host -> device: arrived slots go to their staging slots and arrived
      rows into their table's image, whose whole-table copy goes back once
      every patch is in, under the same budget.

    The generation is staged when every leg has drained.
    """

    def __init__(
        self,
        rank: int,
        transport,
        p2p,
        *,
        generation: int = 0,
        step_budget_bytes: int = 16 << 20,
    ):
        self.rank = rank
        self.transport = transport
        self.p2p = p2p
        # Bytes of DMA one pump may issue (0 = unbounded): the device runs
        # DMAs and program launches through one in-order queue, so a pump's
        # copies delay whatever is launched after them by their transfer time.
        self.step_budget_bytes = max(int(step_budget_bytes), 0)
        self._staged = generation
        self._failed = False
        self._cancel = threading.Event()
        self._phase = "idle"
        self._pumps = 0
        self._pump_s = 0.0
        self._pump_max_s = 0.0
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
        # Outgoing SLOT copies: (move, part, tensor, slot) -> payload hosts.
        self._out_todo: list = []
        self._out_inflight: list = []  # (future, host, move, part)
        self._out_sent = 0  # payloads handed to the exchange thread
        self._out_expected = 0
        self._sent_total = 0
        # Incoming: (move, part, tensor, kind, slot, payload host).
        self._incoming: list = []
        self._in_received = 0
        # Incoming ROWS payloads waiting for their table's image: (index, key).
        self._rows_waiting: list = []
        # Whole-table images, keyed by (group, layer, part): {"tensor", "rows",
        # "host", "future", "ready", "out_rows": [(move, part)], "expected",
        # "patched"}; a table with incoming rows goes back whole once patched.
        self._tables: dict = {}
        self._tables_todo: list = []
        self._h2d_todo: list = []  # (host, tensor, first)
        self._h2d_inflight: list = []
        self._to_release: list = []  # transport buffers handed back at the end
        # The exchange thread: payloads in, arrivals (indices into
        # `_incoming`) out, and "nothing more to send" from the pump.
        self._send_q: queue.Queue = queue.Queue()
        self._recv_q: queue.Queue = queue.Queue()
        self._sends_closed = threading.Event()
        self._marks: dict[str, float] = {}

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

    def _table(self, m, part, tensor, module):
        key = (m.group, m.layer, part)
        if key not in self._tables:
            rows = int(tensor.shape[0]) // _slot_count(module)
            self._tables[key] = {
                "tensor": tensor,
                "rows": rows,
                "host": None,
                "future": None,
                "ready": False,
                "out_rows": [],
                "expected": 0,
                "patched": 0,
            }
            self._tables_todo.append(key)
        return self._tables[key]

    def _prepare(self, plan, modules) -> None:
        incoming, outgoing = plan.moves_for(self.rank)
        for m in outgoing:
            module = modules[(m.group, m.layer)]
            for part, _, tensor, kind in _parts(module):
                if kind == SLOT:
                    self._out_todo.append((m, part, tensor, m.src_slot))
                else:
                    self._table(m, part, tensor, module)["out_rows"].append((m, part))
                self._out_expected += 1
        for m in incoming:
            module = modules[(m.group, m.layer)]
            for part, _, tensor, kind in _parts(module):
                if kind == SLOT:
                    nbytes = self.transport.slice_bytes(tensor)
                else:
                    table = self._table(m, part, tensor, module)
                    table["expected"] += 1
                    nbytes = self.transport.slice_bytes(tensor) * table["rows"]
                self._incoming.append(
                    (
                        m,
                        part,
                        tensor,
                        kind,
                        m.dst_slot,
                        torch.empty(nbytes, dtype=torch.uint8),
                    )
                )
        self._out_moves = outgoing
        self._in_moves = incoming
        if not outgoing and not incoming:
            self._phase = "stream"
            self._finish()
            return
        # The host leg streams from the start: receives are posted now and
        # every payload is sent the step it lands on the host.
        self._thread_done.clear()
        self._thread_error = None
        self._sends_closed.clear()
        self._thread = threading.Thread(
            target=self._exchange, name="expert-mover", daemon=True
        )
        self._thread.start()
        self._phase = "stream"

    def pump(self) -> None:
        """One bounded slice of device work, from the main thread."""
        if self._phase == "idle" or self._failed:
            return
        if self._cancel.is_set():
            self._phase = "idle"
            return
        t0 = time.perf_counter()
        try:
            getattr(self, f"_pump_{self._phase}")()
        except Exception:  # noqa: BLE001 - reported through failed()
            logger.exception(
                "expert mover | rank %d: generation %s failed", self.rank, self._gen
            )
            self._failed = True
            self._phase = "idle"
        finally:
            dt = time.perf_counter() - t0
            self._pumps += 1
            self._pump_s += dt
            self._pump_max_s = max(self._pump_max_s, dt)

    def staged_generation(self) -> int:
        return self._staged

    def failed(self) -> bool:
        return self._failed

    def cancel(self) -> None:
        self._cancel.set()

    def settle(self, generation: int) -> None:
        self._staged = generation
        if self._phase == "stream":  # cancelled mid-generation
            self._sends_closed.set()
            for host in self._to_release:
                self.transport.release(host)
            self._to_release = []
        self._phase = "idle"

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- phases

    def _budget(self) -> float:
        return float("inf") if self.step_budget_bytes == 0 else self.step_budget_bytes

    def _pump_stream(self) -> None:
        """One step's slice of every leg, under one byte budget: issue D2H
        (tables first), hand landed payloads to the exchange thread, apply
        arrived payloads (slots to the H2D queue, rows into their table's
        image), issue H2D, and finish when every leg has drained."""
        if self._thread_error is not None:
            raise RuntimeError(f"host exchange failed: {self._thread_error!r}")
        budget = self._budget()
        while self._tables_todo and budget > 0:
            table = self._tables[self._tables_todo.pop(0)]
            t = table["tensor"]
            table["future"], table["host"] = self.transport.d2h(t, 0, int(t.shape[0]))
            self._to_release.append(table["host"])
            budget -= table["host"].numel()
        while self._out_todo and budget > 0:
            m, part, tensor, slot = self._out_todo.pop(0)
            future, host = self.transport.d2h(tensor, slot, 1)
            self._out_inflight.append((future, host, m, part))
            self._to_release.append(host)
            budget -= host.numel()
        still = []
        for future, host, m, part in self._out_inflight:
            if self.transport.ready(future):
                self._send((m, part, host.reshape(-1)))
            else:
                still.append((future, host, m, part))
        self._out_inflight = still
        for key, table in self._tables.items():
            if table["ready"] or table["future"] is None:
                continue
            if not self.transport.ready(table["future"]):
                continue
            table["ready"] = True
            self._marks.setdefault("first_image_s", time.monotonic() - self._t0)
            image = table["host"].reshape(-1)
            span = table["rows"] * (image.numel() // int(table["tensor"].shape[0]))
            for m, part in table["out_rows"]:
                a = m.src_slot * span
                self._send((m, part, image[a : a + span].clone()))
            self._apply_waiting_rows(key)
        if (
            not self._sends_closed.is_set()
            and self._out_sent == self._out_expected
            and all(tb["ready"] or not tb["out_rows"] for tb in self._tables.values())
        ):
            self._marks["d2h_done_s"] = time.monotonic() - self._t0
            self._sends_closed.set()
        while True:
            try:
                i = self._recv_q.get_nowait()
            except queue.Empty:
                break
            self._in_received += 1
            self._marks.setdefault("first_arrival_s", time.monotonic() - self._t0)
            m, part, tensor, kind, dst_slot, payload = self._incoming[i]
            if kind == SLOT:
                self._h2d_todo.append((payload.reshape(1, -1), tensor, dst_slot))
                continue
            key = (m.group, m.layer, part)
            self._rows_waiting.append(i)
            self._apply_waiting_rows(key)
        while self._h2d_todo and budget > 0:
            host, tensor, first = self._h2d_todo.pop(0)
            self._h2d_inflight.append(self.transport.h2d(host, tensor, first))
            budget -= host.numel()
        self._h2d_inflight = [
            f for f in self._h2d_inflight if not self.transport.ready(f)
        ]
        if (
            self._thread_done.is_set()
            and self._sends_closed.is_set()
            and self._in_received == len(self._incoming)
            and not self._rows_waiting
            and not self._h2d_todo
            and not self._h2d_inflight
        ):
            if self._thread_error is not None:
                raise RuntimeError(f"host exchange failed: {self._thread_error!r}")
            self._finish()

    def _send(self, item) -> None:
        self._marks.setdefault("first_send_s", time.monotonic() - self._t0)
        self._out_sent += 1
        self._sent_total += int(item[2].numel())
        self._send_q.put(item)

    def _sent_bytes(self) -> int:
        return self._sent_total

    def _apply_waiting_rows(self, key) -> None:
        """Patch the arrived row payloads of one table into its image once
        the image is on the host; queue the table's H2D when complete."""
        table = self._tables[key]
        if not table["ready"]:
            return
        image = table["host"].reshape(-1)
        span = table["rows"] * (image.numel() // int(table["tensor"].shape[0]))
        keep = []
        for i in self._rows_waiting:
            m, part, _tensor, _kind, dst_slot, payload = self._incoming[i]
            if (m.group, m.layer, part) != key:
                keep.append(i)
                continue
            a = dst_slot * span
            image[a : a + span].copy_(payload)
            table["patched"] += 1
        self._rows_waiting = keep
        if table["expected"] and table["patched"] == table["expected"]:
            table["expected"] = 0  # queued once
            self._h2d_todo.append((table["host"], table["tensor"], 0))

    def _exchange(self) -> None:
        """The host-to-host leg on the mover's thread, CPU memory only: every
        receive is posted up front (non-blocking), payloads are sent as the
        pump hands them over, arrivals are reported by index. Ends when the
        pump has closed the sends and every send and receive has completed."""
        try:
            recvs = []
            for i, (m, part, _t, _kind, dst_slot, payload) in enumerate(self._incoming):
                recvs.append(
                    (
                        i,
                        self.p2p.irecv(
                            m.src_rank,
                            move_tag(m.group, m.layer, dst_slot, part),
                            payload,
                        ),
                    )
                )
            sends = []
            while True:
                if self._cancel.is_set():
                    return
                while True:
                    try:
                        m, part, payload = self._send_q.get_nowait()
                    except queue.Empty:
                        break
                    h = self.p2p.send(
                        m.dst_rank,
                        move_tag(m.group, m.layer, m.dst_slot, part),
                        payload,
                    )
                    if h is not None:
                        sends.append(h)
                pending = []
                for i, h in recvs:
                    if h.is_completed():
                        self._recv_q.put(i)
                    else:
                        pending.append((i, h))
                recvs = pending
                sends = [h for h in sends if not h.is_completed()]
                if (
                    self._sends_closed.is_set()
                    and self._send_q.empty()
                    and not recvs
                    and not sends
                ):
                    return
                time.sleep(0.0005)
        except BaseException as e:  # noqa: BLE001 - handed to the main thread
            self._thread_error = e
        finally:
            self._thread_done.set()

    def _finish(self) -> None:
        received = sum(p.numel() for *_, p in self._incoming)
        sent = self._sent_bytes()
        seconds = time.monotonic() - self._t0
        marks = ", ".join(f"{k} {v:.2f}" for k, v in sorted(self._marks.items()))
        self.last_report = {
            "generation": self._gen,
            "incoming": len(self._in_moves),
            "outgoing": len(self._out_moves),
            "received_bytes": received,
            "sent_bytes": sent,
            "tables": len(self._tables),
            "seconds": seconds,
            "pumps": self._pumps,
            "pump_seconds": self._pump_s,
            "pump_max_seconds": self._pump_max_s,
        }
        logger.info(
            "expert mover | rank %d staged generation %s: %d in / %d out, "
            "%.1f MB received, %.1f MB sent, %d tables re-imaged, %.2f s; "
            "%d pumps on the step thread, %.0f ms total, %.1f ms max; marks %s",
            self.rank,
            self._gen,
            len(self._in_moves),
            len(self._out_moves),
            received / 1e6,
            sent / 1e6,
            len(self._tables),
            seconds,
            self._pumps,
            self._pump_s * 1e3,
            self._pump_max_s * 1e3,
            marks,
        )
        self.last_report["marks"] = dict(self._marks)
        self._pumps, self._pump_s, self._pump_max_s = 0, 0.0, 0.0
        for host in self._to_release:
            self.transport.release(host)
        self._to_release = []
        self.transport.release_pending()
        self._staged = self._gen
        self._phase = "idle"
