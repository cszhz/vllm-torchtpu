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
"""The mover's device legs on a TPU through tpu-sync's raw transfer: a slot
of a uint8 and of an fp4-pair (float4_e2m1fn_x2) slot-major tensor, whose
storage XLA accounts at one byte per nibble, and a flat f32 scale table moved
whole with a slot's rows patched in the host image. Host bytes are the
device's physical layout, so values are only ever compared after a round trip
back onto the device."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import expert_mover as mv
from vllm_torchtpu import expert_relayout as rl


def _has_tpu() -> bool:
    try:
        return (
            torch.accelerator.is_available()
            and torch.accelerator.current_accelerator().type == "tpu"
        )
    except Exception:  # noqa: BLE001 - no accelerator stack at all
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="needs a TPU")

SLOTS, K, N = 6, 1024, 2560


def _bits(seed, *shape):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 255, shape, generator=g, dtype=torch.int32).to(torch.uint8)


def _raw():
    transport = mv.make_transport("raw_transfer")
    assert type(transport) is mv.RawTransferTransport, "tpu_sync not importable"
    return transport


def test_a_uint8_slot_round_trips_byte_for_byte():
    transport = _raw()
    src_bits, dst_bits = _bits(1, SLOTS, K, N), _bits(2, SLOTS, K, N)
    src, dst = src_bits.to("tpu"), dst_bits.to("tpu")
    assert transport.slice_bytes(src) == K * N
    future, host = transport.d2h(src, 1, 1)
    future.wait()
    assert host.device.type == "cpu" and tuple(host.shape) == (1, K * N)
    transport.h2d(host, dst, 4).wait()
    got = dst.cpu()
    torch.testing.assert_close(got[4], src_bits[1])
    torch.testing.assert_close(got[:4], dst_bits[:4])
    torch.testing.assert_close(got[5], dst_bits[5])


def test_an_fp4_pair_slot_moves_as_its_packed_bytes():
    """The fused path's weights: torch packs two fp4 per byte, torch_tpu backs
    the tensor with an F4 array of twice the minor dim. The slice is the packed
    size and a slot lands in another fp4 tensor's slot."""
    transport = _raw()
    src_bits, dst_bits = _bits(3, SLOTS, K, N // 2), _bits(4, SLOTS, K, N // 2)
    # torch cannot move the pair dtype host <-> device: reinterpret on device.
    src = src_bits.to("tpu").view(torch.float4_e2m1fn_x2)
    dst = dst_bits.to("tpu").view(torch.float4_e2m1fn_x2)
    assert src.dtype == torch.float4_e2m1fn_x2
    assert transport.slice_bytes(src) == K * N // 2
    future, host = transport.d2h(src, 1, 1)
    future.wait()
    transport.h2d(host, dst, 4).wait()
    got = dst.view(torch.uint8).cpu()
    torch.testing.assert_close(got[4], src_bits[1])
    torch.testing.assert_close(got[:4], dst_bits[:4])
    torch.testing.assert_close(got[5], dst_bits[5])


def test_a_scale_table_is_imaged_whole_and_patched_by_slot_rows():
    transport = _raw()
    rows = 32  # fp4_scale_rows pads to the f32 tile: a slot's rows = 4 tile rows
    a_host = torch.randn(SLOTS * rows, N)
    b_host = torch.randn(SLOTS * rows, N)
    a, b = a_host.to("tpu"), b_host.to("tpu")
    assert transport.slice_bytes(a) == N * 4
    fa, image_a = transport.d2h(a, 0, SLOTS * rows)
    fb, image_b = transport.d2h(b, 0, SLOTS * rows)
    fa.wait()
    fb.wait()
    row_bytes = image_a.shape[1]
    flat_a, flat_b = image_a.reshape(-1), image_b.reshape(-1)
    span = rows * row_bytes
    flat_b[3 * span : 4 * span].copy_(flat_a[1 * span : 2 * span])
    transport.h2d(image_b, b, 0).wait()
    got = b.cpu()
    torch.testing.assert_close(got[3 * rows : 4 * rows], a_host[1 * rows : 2 * rows])
    torch.testing.assert_close(got[: 3 * rows], b_host[: 3 * rows])
    torch.testing.assert_close(got[4 * rows :], b_host[4 * rows :])


def test_the_mover_stages_fp4_weights_and_scale_tables_between_two_fake_ranks():
    """Two modules on one chip stand in for two ranks; the mover moves expert
    slot 1 of rank 0 into staging slot 5 of rank 1 through the LocalExchange,
    with the real device legs."""
    transport = _raw()
    rows = 32

    def module(seed):
        w13 = _bits(seed, SLOTS, K, N // 2).to("tpu").view(torch.float4_e2m1fn_x2)
        w2 = (
            _bits(seed + 1, SLOTS, K // 2, N // 2)
            .to("tpu")
            .view(torch.float4_e2m1fn_x2)
        )
        return SimpleNamespace(
            local_num_experts=SLOTS,
            w13_weight=torch.nn.Parameter(w13, requires_grad=False),
            w2_weight=torch.nn.Parameter(w2, requires_grad=False),
            _tpu_fused_w13_scale=torch.randn(SLOTS * rows, N).to("tpu"),
            _tpu_fused_w2_scale=torch.randn(SLOTS * rows // 2, N).to("tpu"),
        )

    a, b = module(10), module(20)
    before = {
        name: getattr(b, name).detach().view(torch.uint8).cpu().clone()
        for name in (
            "w13_weight",
            "w2_weight",
            "_tpu_fused_w13_scale",
            "_tpu_fused_w2_scale",
        )
    }
    plan = rl.Plan(
        1,
        0,
        {("target", 0): None},
        [rl.Move("target", 0, 7, src_rank=0, src_slot=1, dst_rank=1, dst_slot=5)],
    )
    exchange = mv.LocalExchange()
    movers = [mv.SlabMover(r, transport, exchange.endpoint(r)) for r in range(2)]
    mods = {0: {("target", 0): a}, 1: {("target", 0): b}}
    for r, m in enumerate(movers):
        m.start(plan, mods[r])
    import time

    deadline = time.monotonic() + 60
    while any(m.staged_generation() != 1 for m in movers):
        assert not any(m.failed() for m in movers), "a mover failed"
        assert time.monotonic() < deadline, "the movers did not finish"
        for m in movers:
            m.pump()
        time.sleep(0.01)
    for name in ("w13_weight", "w2_weight"):
        src = getattr(a, name).detach().view(torch.uint8).cpu()
        got = getattr(b, name).detach().view(torch.uint8).cpu()
        torch.testing.assert_close(got[5], src[1])
        torch.testing.assert_close(got[:5], before[name][:5])
    for name, r in (("_tpu_fused_w13_scale", rows), ("_tpu_fused_w2_scale", rows // 2)):
        src = getattr(a, name).cpu()
        got = getattr(b, name).cpu()
        torch.testing.assert_close(got[5 * r : 6 * r], src[1 * r : 2 * r])
        torch.testing.assert_close(
            got[: 5 * r], before[name].view(torch.float32)[: 5 * r]
        )
    assert movers[1].last_report["tables"] == 2
