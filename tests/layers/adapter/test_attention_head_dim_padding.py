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
"""Head-dim padding in `PallasAttentionBackendImpl.forward`.

`forward` pads q/k/v up to the head width of the KV pool it writes into, and
`sharded_ragged_paged_attention` picks the head_dim==64 kernel from the query
width. A head_dim==64 page packs K and V along its last axis, so its last axis
is twice the head width; reading it as the head width pads the query to 128
and dispatches the other kernel onto the head_dim==64 layout.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_torchtpu.layers.adapter import attention

_NUM_HEADS = 4
_NUM_KV_HEADS = 2
_NUM_TOKENS = 3


def _impl(head_size: int) -> attention.PallasAttentionBackendImpl:
    impl = object.__new__(attention.PallasAttentionBackendImpl)
    impl.num_heads = _NUM_HEADS
    impl.num_kv_heads = _NUM_KV_HEADS
    impl.head_size = head_size
    impl.kv_cache_quantized_dtype = None
    impl._pool_is_seq_along_lane = False
    impl.sinks = None
    impl.dcp_world_size = 1
    impl.kv_sharing_target_layer_name = None
    impl._bm_kernel = False
    return impl


def _kernel_query_width(
    impl: attention.PallasAttentionBackendImpl, kv_cache: torch.Tensor
) -> int:
    """Runs `forward` and returns the query width the RPA kernel receives."""
    widths: list[int] = []

    def _kernel(kv, query, *args):
        widths.append(query.shape[-1])
        return torch.zeros_like(query)

    impl._build_rpa_kernel = lambda *args, **kwargs: _kernel
    metadata = SimpleNamespace(
        seq_lens=None,
        block_tables=None,
        query_start_loc=None,
        request_distribution=None,
    )
    head_size = impl.head_size
    with (
        mock.patch.object(
            attention,
            "get_vllm_model_wrapper_context",
            return_value=SimpleNamespace(vllm_config=None, kv_cache_bundle=None),
        ),
        mock.patch.object(attention, "synchronize_tensors"),
    ):
        out = impl.forward(
            SimpleNamespace(),
            torch.ones(_NUM_TOKENS, _NUM_HEADS * head_size, dtype=torch.bfloat16),
            torch.ones(_NUM_TOKENS, _NUM_KV_HEADS * head_size, dtype=torch.bfloat16),
            torch.ones(_NUM_TOKENS, _NUM_KV_HEADS * head_size, dtype=torch.bfloat16),
            kv_cache,
            metadata,
        )
    assert out.shape == (_NUM_TOKENS, _NUM_HEADS * head_size)
    (width,) = widths
    return width


@pytest.mark.parametrize("head_size", [64, 128])
def test_native_kv_cache_reaches_the_kernel_at_the_layer_head_size(
    head_size: int,
) -> None:
    shape = attention.PallasAttentionBackend.get_kv_cache_shape(
        4, 16, _NUM_KV_HEADS, head_size, torch.bfloat16
    )
    kv_cache = torch.zeros(shape, dtype=torch.bfloat16)
    assert _kernel_query_width(_impl(head_size), kv_cache) == head_size


def test_narrow_layer_is_padded_to_a_wider_pool_head() -> None:
    # A 128-wide layer writing into a pool laid out for 256-wide heads.
    shape = attention.PallasAttentionBackend.get_kv_cache_shape(
        4, 16, _NUM_KV_HEADS, 256, torch.bfloat16
    )
    kv_cache = torch.zeros(shape, dtype=torch.bfloat16)
    assert _kernel_query_width(_impl(128), kv_cache) == 256
