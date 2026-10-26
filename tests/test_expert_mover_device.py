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
"""The mover's device legs on a TPU: tpu-sync's raw transfer of 4 KiB byte
regions of a tensor's storage -- a slot of a slot-major tensor, whether the
tensor is its whole storage or a view into a larger one (the quantizer's
outputs are), and the slot's rows of a flat scale table. Nothing here goes
through torch's dtype or view machinery on the device: the chips returned
wrong bytes for a clone of an offset view, and torch cannot move the fp4-pair
dtype host<->device, which is why the mover addresses storages."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import expert_mover as mv


def _has_tpu() -> bool:
    try:
        return (
            torch.accelerator.is_available()
            and torch.accelerator.current_accelerator().type == "tpu"
        )
    except Exception:  # noqa: BLE001 - no accelerator stack at all
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="needs a TPU")

SLOTS, K, N = 6, 1024, 2560  # 2.6 MB per slot


def _bits(seed, *shape):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 255, shape, generator=g, dtype=torch.int32).to(torch.uint8)


@pytest.mark.parametrize("transport_kind", ["raw_transfer", "torch"])
def test_a_slot_region_round_trips_byte_for_byte(transport_kind):
    transport = mv.make_transport(transport_kind)
    if transport_kind == "raw_transfer":
        assert type(transport) is mv.RawTransferTransport, "tpu_sync not importable"
    src_bits, dst_bits = _bits(1, SLOTS, K, N), _bits(2, SLOTS, K, N)
    src, dst = src_bits.to("tpu"), dst_bits.to("tpu")
    off, nbytes = mv.slot_region(src, 1)
    future, host = transport.d2h(src, off, nbytes)
    future.wait()
    assert host.device.type == "cpu" and host.numel() == K * N
    transport.h2d(host, dst, mv.slot_region(dst, 4)[0]).wait()
    got = dst.cpu()
    torch.testing.assert_close(got[4], src_bits[1])
    torch.testing.assert_close(got[:4], dst_bits[:4])
    torch.testing.assert_close(got[5], dst_bits[5])


def test_a_view_into_a_larger_storage_is_addressed_through_the_storage():
    """What the quantizer hands back: the weight and its scale share one
    output storage, so each is an offset view. The region arithmetic folds
    the view's offset in and the raw transfer addresses the storage."""
    transport = mv.make_transport("raw_transfer")
    assert type(transport) is mv.RawTransferTransport, "tpu_sync not importable"
    base_bits = _bits(3, 2, SLOTS, K, N)
    base = base_bits.to("tpu")
    view = base[1]
    assert view.storage_offset() == SLOTS * K * N
    off, nbytes = mv.slot_region(view, 2)
    assert off == (SLOTS + 2) * K * N
    future, host = transport.d2h(view, off, nbytes)
    future.wait()
    torch.testing.assert_close(host.view(K, N), base_bits[1, 2])
    # And back into another view's slot, leaving its neighbours alone.
    other_bits = _bits(4, 2, SLOTS, K, N)
    other = other_bits.to("tpu")
    transport.h2d(host, other[0], mv.slot_region(other[0], 4)[0]).wait()
    got = other.cpu()
    torch.testing.assert_close(got[0, 4], base_bits[1, 2])
    torch.testing.assert_close(got[0, :4], other_bits[0, :4])
    torch.testing.assert_close(got[1], other_bits[1])


@pytest.mark.parametrize("transport_kind", ["raw_transfer", "torch"])
def test_scale_rows_move_as_one_region(transport_kind):
    transport = mv.make_transport(transport_kind)
    rows, n = 16, 2560  # 16 x 2560 x 4 B = 160 KiB per slot
    module = SimpleNamespace(local_num_experts=SLOTS)
    scale_host = torch.randn(SLOTS * rows, n)
    scale = scale_host.to("tpu")
    off, nbytes = mv.rows_region(scale, module, 1)
    future, host = transport.d2h(scale, off, nbytes)
    future.wait()
    transport.h2d(host, scale, mv.rows_region(scale, module, 4)[0]).wait()
    after = scale.cpu()
    torch.testing.assert_close(
        after[4 * rows : 5 * rows], scale_host[1 * rows : 2 * rows]
    )
    torch.testing.assert_close(after[: 4 * rows], scale_host[: 4 * rows])
