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
"""The slab mover on the host: two ranks as threads, an in-process mailbox
for the gloo group, torch copies for the device legs (expert_mover.py)."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import expert_mover as mv
from vllm_torchtpu import expert_relayout as rl

pytestmark = pytest.mark.cpu_test

SLOTS, K, N, ROWS, NS = (
    5,
    32,
    128,
    2,
    512,
)  # a slot = 4 KiB; scale rows: 2 x 512 f32 = 4 KiB


def _module(seed):
    g = torch.Generator().manual_seed(seed)
    return SimpleNamespace(
        local_num_experts=SLOTS,
        w13_weight=torch.nn.Parameter(
            torch.randint(0, 255, (SLOTS, K, N), generator=g, dtype=torch.int32).to(
                torch.uint8
            ),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.randint(0, 255, (SLOTS, 2 * K, N), generator=g, dtype=torch.int32).to(
                torch.uint8
            ),
            requires_grad=False,
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.randn(SLOTS * ROWS, NS, generator=g), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.randn(SLOTS * ROWS, NS, generator=g), requires_grad=False
        ),
    )


def _plan():
    # Rank 0's slot 1 -> rank 1's slot 4; rank 1's slot 2 -> rank 0's slot 4.
    return rl.Plan(
        generation=3,
        trigger_step=0,
        moves=[
            rl.Move("target", 7, 11, 0, 1, 1, 4),
            rl.Move("target", 7, 22, 1, 2, 0, 4),
        ],
    )


def _pump_until_settled(movers, rounds=400):
    """Drive every mover's pump from one "main thread" until all are staged,
    failed or idle, the way the controllers' step hooks would."""
    import time

    for _ in range(rounds):
        for m in movers:
            m.pump()
        if all(m.failed() or m._phase == "idle" for m in movers):
            break
        time.sleep(0.005)
    for m in movers:
        m.join(timeout=10)


def _run_both(plan, modules_by_rank, transport=None):
    exchange = mv.LocalExchange()
    movers = [
        mv.SlabMover(r, transport or mv.TorchTransport(), exchange.endpoint(r))
        for r in range(2)
    ]
    for r, m in enumerate(movers):
        m.start(plan, {("target", 7): modules_by_rank[r]})
    _pump_until_settled(movers)
    return movers


def test_slabs_land_in_the_destination_staging_slots_byte_for_byte():
    mods = [_module(1), _module(2)]
    before = [
        {
            n: getattr(m, n).data.clone()
            for aliases in mv.PART_ALIASES
            for n in aliases
            if hasattr(m, n)
        }
        for m in mods
    ]
    movers = _run_both(_plan(), mods)
    assert [m.staged_generation() for m in movers] == [3, 3]
    assert not any(m.failed() for m in movers)
    # Rank 1's slot 4 now holds rank 0's slot 1, in every part.
    for name in ("w13_weight", "w2_weight"):
        torch.testing.assert_close(getattr(mods[1], name).data[4], before[0][name][1])
        torch.testing.assert_close(getattr(mods[0], name).data[4], before[1][name][2])
        # Nothing else moved.
        torch.testing.assert_close(getattr(mods[1], name).data[:4], before[1][name][:4])
    for name in ("w13_weight_scale", "w2_weight_scale"):
        torch.testing.assert_close(
            getattr(mods[1], name).data[4 * ROWS : 5 * ROWS],
            before[0][name][1 * ROWS : 2 * ROWS],
        )
        torch.testing.assert_close(
            getattr(mods[0], name).data[4 * ROWS : 5 * ROWS],
            before[1][name][2 * ROWS : 3 * ROWS],
        )
    assert (
        movers[0].last_report["incoming"] == 1
        and movers[0].last_report["outgoing"] == 1
    )
    assert (
        movers[0].last_report["sent_bytes"] == movers[1].last_report["received_bytes"]
    )


def test_tags_are_unique_per_group_layer_slot_and_part():
    tags = {
        mv.move_tag(g, layer, slot, part)
        for g in ("target", "draft")
        for layer in (0, 39, 1023)
        for slot in (0, 255)
        for part in range(6)
    }
    assert len(tags) == 2 * 3 * 2 * 6
    assert all(0 < t < 2**31 for t in tags)
    # Per generation (mod 128) too, still int32 at the largest base.
    gens = {mv.move_tag("draft", 1023, 255, 7, g) for g in range(128)}
    assert len(gens) == 128 and max(gens) < 2**31
    assert mv.move_tag("target", 7, 4, 0, 3) != mv.move_tag("target", 7, 4, 0, 4)


class _CountingTransport(mv.TorchTransport):
    """Torch copies that count the host buffers handed back."""

    def __init__(self):
        super().__init__()
        self.released = 0

    def release(self, host):
        self.released += 1
        super().release(host)


def _snapshot(mods):
    return [
        {
            n: getattr(m, n).data.clone()
            for aliases in mv.PART_ALIASES
            for n in aliases
            if hasattr(m, n)
        }
        for m in mods
    ]


def test_a_dropped_generation_drains_and_the_next_one_gets_its_own_messages():
    """The D8 hang (j-608af38f): a generation dropped at its timeout left
    receives posted on the gloo group under tags the next generation reused,
    which stole its payloads; every later generation timed out too, then the
    server hung. Draining finishes the exchange on every rank without device
    writes, and tags carry the generation besides."""
    import threading

    mods = [_module(1), _module(2)]
    before = _snapshot(mods)
    exchange = mv.LocalExchange()
    transports = [_CountingTransport() for _ in range(2)]
    movers = [
        mv.SlabMover(r, transports[r], exchange.endpoint(r), step_budget_bytes=1)
        for r in range(2)
    ]
    for r, m in enumerate(movers):
        m.start(_plan(), {("target", 7): mods[r]})
    # One pump each: under a one-byte budget a single copy is issued, so
    # nothing has staged when the controllers drop the generation.
    for m in movers:
        m.pump()
    assert all(m.staged_generation() == 0 for m in movers)
    results = [None, None]

    def drain(r):
        results[r] = movers[r].drain(10.0)

    threads = [threading.Thread(target=drain, args=(r,)) for r in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(20)
    assert results == [True, True]
    for m in movers:
        m.cancel()
        m.settle(0)
    # Nothing was written to the device, and every host buffer went back.
    after = _snapshot(mods)
    for r in range(2):
        for name, value in before[r].items():
            torch.testing.assert_close(after[r][name], value)
    assert all(t.released > 0 for t in transports)
    # The next generation moves the same slots and gets exactly its bytes.
    plan = _plan()
    plan.generation = 4
    for r, m in enumerate(movers):
        m.start(plan, {("target", 7): mods[r]})
    _drive(movers, generation=4)
    for name in ("w13_weight", "w2_weight"):
        torch.testing.assert_close(getattr(mods[1], name).data[4], before[0][name][1])
        torch.testing.assert_close(getattr(mods[0], name).data[4], before[1][name][2])


def test_a_drain_the_peer_never_joins_is_abandoned_and_keeps_its_buffers():
    mods = [_module(1), _module(2)]
    exchange = mv.LocalExchange()
    transport = _CountingTransport()
    mover = mv.SlabMover(0, transport, exchange.endpoint(0), step_budget_bytes=1)
    mover.start(_plan(), {("target", 7): mods[0]})
    mover.pump()
    # Rank 1 never runs: its payload for rank 0 never comes.
    assert mover.drain(0.2) is False
    mover.cancel()
    mover.settle(0)
    assert transport.released == 0  # a copy might still land in them
    assert mover._phase == "idle" and mover.staged_generation() == 0


def test_a_failing_leg_marks_the_generation_failed():
    mods = [
        _module(1),
        SimpleNamespace(local_num_experts=SLOTS),
    ]  # rank 1 has no weights
    plan = rl.Plan(3, 0, moves=[rl.Move("target", 7, 11, 1, 1, 0, 4)])
    exchange = mv.LocalExchange()
    movers = [
        mv.SlabMover(r, mv.TorchTransport(), exchange.endpoint(r)) for r in range(2)
    ]
    movers[1].start(plan, {("target", 7): mods[1]})
    _pump_until_settled([movers[1]])
    assert movers[1].failed() and movers[1].staged_generation() == 0


def test_cancel_stops_before_the_writes_and_settle_moves_the_generation():
    mods = [_module(1), _module(2)]
    exchange = mv.LocalExchange()
    mover = mv.SlabMover(0, mv.TorchTransport(), exchange.endpoint(0))
    mover.cancel()  # before start: the thread sees it at its first check
    plan = rl.Plan(3, 0, moves=[rl.Move("target", 7, 22, 1, 2, 0, 4)])
    mover.start(plan, {("target", 7): mods[0]})
    _pump_until_settled([mover])
    assert mover.staged_generation() == 0 and not mover.failed()
    mover.settle(3)
    assert mover.staged_generation() == 3


def test_make_transport_falls_back_to_torch_without_tpu_sync(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_tpu_sync(name, *a, **k):
        if name.startswith("tpu_sync"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_tpu_sync)
    assert type(mv.make_transport("raw_transfer")) is mv.TorchTransport
    assert type(mv.make_transport("torch")) is mv.TorchTransport


def _drive(movers, generation=1, timeout=30.0):
    """Pump every mover from this (the main) thread until all have staged."""
    import time

    deadline = time.monotonic() + timeout
    while any(m.staged_generation() != generation for m in movers):
        assert not any(m.failed() for m in movers), "a mover failed"
        assert time.monotonic() < deadline, "the movers did not finish"
        for m in movers:
            m.pump()
        time.sleep(0.001)


def test_slot_payload_is_the_slot_bytes_and_lands_in_another_slot():
    transport = mv.TorchTransport()
    src = torch.randint(0, 255, (SLOTS, K, N), dtype=torch.int32).to(torch.uint8)
    assert transport.slice_bytes(src) == K * N
    future, host = transport.d2h(src, 2, 1)
    assert future.is_ready() and host.shape == (1, K * N)
    torch.testing.assert_close(host.view(K, N), src[2])
    dst = torch.zeros((SLOTS, K, N), dtype=torch.uint8)
    transport.h2d(host, dst, 4)
    torch.testing.assert_close(dst[4], src[2])
    assert int(dst[:4].sum()) == 0


def test_fp4_pair_slices_move_as_their_packed_bytes():
    bits = torch.randint(0, 255, (SLOTS, K, N // 2), dtype=torch.int32).to(torch.uint8)
    q = bits.view(torch.float4_e2m1fn_x2)
    transport = mv.TorchTransport()
    assert transport.slice_bytes(q) == K * N // 2
    _, host = transport.d2h(q, 1, 1)
    torch.testing.assert_close(host.view(K, N // 2), bits[1])
    other = torch.zeros_like(bits).view(torch.float4_e2m1fn_x2)
    transport.h2d(host, other, 3)
    torch.testing.assert_close(other.view(torch.uint8)[3], bits[1])


def test_scale_tables_move_whole_and_patch_a_slot_block_of_rows():
    """A flat [slots * rows, N] table is imaged whole; a slot's rows are one
    contiguous range of the image, both when cut (sender) and patched
    (receiver), and the rest of the table is untouched by the round trip."""
    a, b = _module(11), _module(12)
    before_b = b.w13_weight_scale.detach().clone()
    layouts = {("target", 0): None}
    plan = rl.Plan(
        1,
        0,
        layouts,
        [rl.Move("target", 0, 7, src_rank=0, src_slot=1, dst_rank=1, dst_slot=3)],
    )
    exchange = mv.LocalExchange()
    movers = [
        mv.SlabMover(r, mv.TorchTransport(), exchange.endpoint(r)) for r in range(2)
    ]
    mods = {0: {("target", 0): a}, 1: {("target", 0): b}}
    for r, m in enumerate(movers):
        m.start(plan, mods[r])
    _drive(movers)
    assert [m.staged_generation() for m in movers] == [1, 1]
    for name in mv.PARTS:
        src = getattr(a, name, None)
        dst = getattr(b, name, None)
        if src is None:
            continue
        if src.dim() >= 3:
            torch.testing.assert_close(dst[3], src[1])
        else:
            rows = src.shape[0] // SLOTS
            torch.testing.assert_close(
                dst[3 * rows : 4 * rows], src[1 * rows : 2 * rows]
            )
            torch.testing.assert_close(dst[: 3 * rows], before_b[: 3 * rows])
            torch.testing.assert_close(dst[4 * rows :], before_b[4 * rows :])
    assert movers[1].last_report["tables"] == 2


def test_raw_host_memory_is_viewed_bound_and_pooled_by_ownership(monkeypatch):
    """The raw transport's `raw` host memory: a RawHostBuffer is bound to the
    device (allocating) through a tiny anchor, viewed as a contiguous uint8
    tensor of the requested shape, and pooled on release by ownership rather
    than by torch's is_pinned(), which such views do not report."""
    pytest.importorskip("tpu_sync.api.torch.raw_transfer")
    import ctypes

    from tpu_sync.api.torch import raw_transfer

    bound = []

    class FakeRawHostBuffer:
        def __init__(self, size_bytes):
            self._mem = (ctypes.c_uint8 * size_bytes)()
            self.size_bytes = size_bytes

        @property
        def data_ptr(self):
            assert bound, "data_ptr read before the buffer was bound"
            return ctypes.addressof(self._mem)

    class FakePrepared:
        def __init__(self, anchor, buf, **kw):
            bound.append((tuple(anchor.shape), buf.size_bytes))

    monkeypatch.setattr(raw_transfer, "RawHostBuffer", FakeRawHostBuffer, raising=False)
    monkeypatch.setattr(
        raw_transfer, "PreparedTorchRawTransfer", FakePrepared, raising=False
    )
    monkeypatch.setattr(
        raw_transfer, "major_slice_bytes", lambda t, **kw: 64, raising=False
    )
    tr = mv.RawTransferTransport(host_memory="raw")
    with pytest.raises(RuntimeError, match="prepare_device"):
        tr._take((2, 64), torch.device("cpu"))  # no anchor adopted yet
    tr.prepare_device(torch.zeros(8, dtype=torch.uint8))  # an existing (CPU) tensor
    assert bound == []  # a CPU anchor needs no bind
    host = tr._take((2, 64), torch.device("cpu"))
    assert (
        tuple(host.shape) == (2, 64)
        and host.is_contiguous()
        and host.dtype == torch.uint8
    )
    assert bound == [((8,), 128)]  # the buffer's own bind through that anchor
    assert tr._ours(host) and not tr._ours(torch.empty(3, dtype=torch.uint8))
    tr.release(host)
    assert tr._take((2, 64), torch.device("cpu")) is host  # pooled and reused
    tr.release(torch.empty((2, 64), dtype=torch.uint8))  # foreign: not pooled
    assert tr.pooled_bytes() == 0


def test_the_raw_transport_forwards_the_buffer_lock_choice(monkeypatch):
    pytest.importorskip("tpu_sync.api.torch.raw_transfer")
    from tpu_sync.api.torch import raw_transfer

    calls = []

    def slice_bytes(t, **kw):
        calls.append(("slice", kw))
        return 64

    def d2h(*a, **kw):
        calls.append(("d2h", kw))
        return mv._Done()

    def h2d(*a, **kw):
        calls.append(("h2d", kw))
        return mv._Done()

    monkeypatch.setattr(raw_transfer, "major_slice_bytes", slice_bytes, raising=False)
    monkeypatch.setattr(raw_transfer, "transfer_d2h_async", d2h, raising=False)
    monkeypatch.setattr(raw_transfer, "transfer_h2d_async", h2d, raising=False)
    tr = mv.RawTransferTransport(skip_buffer_lock=True)
    t = torch.zeros((3, 64), dtype=torch.uint8)
    _future, host = tr.d2h(t, 0, 1)
    tr.h2d(host, t, 2)
    assert [c[0] for c in calls] == ["slice", "d2h", "h2d"]
    assert all(c[1]["unsafe_skip_buffer_lock"] is True for c in calls)
    tr2 = mv.RawTransferTransport()
    tr2.d2h(t, 0, 1)
    assert calls[-1][1]["unsafe_skip_buffer_lock"] is False
