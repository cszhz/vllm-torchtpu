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
"""The mover's device legs on a TPU: tpu-sync's raw transfer of one slot's
slice out of a slot-major device tensor and back into another slot, on the
dtypes the fused EP kernel's weights use."""

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

SLOTS, K, N = 6, 1024, 2560  # 2.6 MB per slot at one byte per element


def _slot_tensor(dtype, seed):
    """(device tensor of `dtype`, its uint8 bytes on the host, the uint8 device
    tensor sharing its storage). The fp4-pair dtype never crosses host and
    device through torch here: torch_tpu refuses that (tensor_to_buffer.cc,
    "Cannot assign buffer of size N to tensor of shape [N]"), which is exactly
    why the mover's device legs are raw transfers. The device view is taken
    from uint8 bytes already on the device, and the bytes are read back
    through the uint8 storage the view shares."""
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 255, (SLOTS, K, N), generator=g, dtype=torch.int32)
    bits = bits.to(torch.uint8)
    dev_u8 = bits.to("tpu")
    dev = dev_u8 if dtype is torch.uint8 else dev_u8.view(dtype)
    return dev, bits, dev_u8


@pytest.mark.parametrize("transport_kind", ["raw_transfer", "torch"])
@pytest.mark.parametrize(
    "dtype",
    [torch.uint8]
    + ([torch.float4_e2m1fn_x2] if hasattr(torch, "float4_e2m1fn_x2") else []),
    ids=lambda d: str(d).split(".")[-1],
)
def test_a_slot_round_trips_through_the_host_byte_for_byte(transport_kind, dtype):
    if transport_kind == "torch" and dtype is not torch.uint8:
        pytest.skip(
            "torch_tpu cannot move fp4 pairs through .cpu()/.to(); raw path only"
        )
    transport = mv.make_transport(transport_kind)
    if transport_kind == "raw_transfer":
        assert type(transport) is mv.RawTransferTransport, "tpu_sync not importable"
    src, src_bits, _ = _slot_tensor(dtype, 1)
    dst, dst_bits, dst_u8 = _slot_tensor(dtype, 2)
    future, host = transport.d2h_slot(src, 1)
    future.wait()
    assert host.device.type == "cpu" and tuple(host.shape) == (K, N)
    transport.h2d_slot(host, dst, 4).wait()
    got = dst_u8.cpu()
    torch.testing.assert_close(got[4], src_bits[1])
    torch.testing.assert_close(got[:4], dst_bits[:4])
    torch.testing.assert_close(got[5], dst_bits[5])


@pytest.mark.parametrize("transport_kind", ["raw_transfer", "torch"])
def test_a_flat_scale_table_round_trips_whole_and_patched(transport_kind):
    """The scale tables move whole (a few hundred KB): D2H, patch the staging
    slot's rows on the host, H2D."""
    transport = mv.make_transport(transport_kind)
    rows, n = 16, 2560
    scale = torch.randn(SLOTS * rows, n).to("tpu")
    before = scale.cpu().clone()
    future, host = transport.d2h_full(scale)
    future.wait()
    host[4 * rows : 5 * rows] = host[1 * rows : 2 * rows]
    transport.h2d_full(host, scale).wait()
    after = scale.cpu()
    torch.testing.assert_close(after[4 * rows : 5 * rows], before[1 * rows : 2 * rows])
    torch.testing.assert_close(after[: 4 * rows], before[: 4 * rows])
