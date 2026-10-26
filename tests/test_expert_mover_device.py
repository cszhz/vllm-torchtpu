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

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "tpu") and torch.tpu.is_available()), reason="needs a TPU"
)

SLOTS, K, N = 6, 1024, 2560  # 2.6 MB per slot at one byte per element


def _slot_tensor(dtype, seed):
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 255, (SLOTS, K, N), generator=g, dtype=torch.int32).to(
        torch.uint8
    )
    host = bits if dtype is torch.uint8 else bits.view(dtype)
    return host.to("tpu"), bits


@pytest.mark.parametrize("transport_kind", ["raw_transfer", "torch"])
@pytest.mark.parametrize(
    "dtype",
    [torch.uint8]
    + ([torch.float4_e2m1fn_x2] if hasattr(torch, "float4_e2m1fn_x2") else []),
    ids=lambda d: str(d).split(".")[-1],
)
def test_a_slot_round_trips_through_the_host_byte_for_byte(transport_kind, dtype):
    transport = mv.make_transport(transport_kind)
    if transport_kind == "raw_transfer":
        assert type(transport) is mv.RawTransferTransport, "tpu_sync not importable"
    src, src_bits = _slot_tensor(dtype, 1)
    dst, dst_bits = _slot_tensor(dtype, 2)
    host = transport.d2h_slot(src, 1)
    assert host.device.type == "cpu" and tuple(host.shape) == (K, N)
    transport.h2d_slot(host, dst, 4)
    got = dst.cpu().view(torch.uint8) if dtype is not torch.uint8 else dst.cpu()
    torch.testing.assert_close(got[4], src_bits[1])
    torch.testing.assert_close(got[:4], dst_bits[:4])
    torch.testing.assert_close(got[5], dst_bits[5])


def test_scale_rows_round_trip_through_torch():
    transport = mv.make_transport("torch")
    rows, n = 16, 2560
    scale = torch.randn(SLOTS * rows, n).to("tpu")
    before = scale.cpu().clone()
    host = transport.d2h_rows(scale, 1 * rows, 2 * rows)
    transport.h2d_rows(host, scale, 4 * rows, 5 * rows)
    after = scale.cpu()
    torch.testing.assert_close(after[4 * rows : 5 * rows], before[1 * rows : 2 * rows])
    torch.testing.assert_close(after[: 4 * rows], before[: 4 * rows])
