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
"""q/k/v padding against the KV page a Pallas attention layer writes.

The kernel is picked from the query width (`q.shape[-1] == 64` selects the
head_dim=64 kernel), so padding a head_dim=64 query to its own packed page
width (128) silently switches to the v3 kernel and its layout check fails
(#2386). Padding is only for a page defined by another spec.
"""

import pytest
import torch

from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    kv_page_head_padding,
)

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn


def _page(num_kv_heads: int, head_size: int, dtype: torch.dtype):
    return PallasAttentionBackend.get_kv_cache_shape(
        1, 1, num_kv_heads, head_size, dtype
    )[2:]


@pytest.mark.parametrize("dtype", [BF16, FP8])
@pytest.mark.parametrize(
    "num_kv_heads,head_size",
    [(8, 64), (2, 64), (8, 128), (4, 256), (8, 80)],
)
def test_native_page_needs_no_padding(num_kv_heads, head_size, dtype):
    page = _page(num_kv_heads, head_size, dtype)

    assert kv_page_head_padding(page, page, page[-1], head_size) == 0


def test_hd64_native_page_is_wider_than_the_head():
    # The precondition for #2386: the packed page is 128 lanes wide.
    assert _page(8, 64, BF16)[-1] == 128


@pytest.mark.parametrize("dtype", [BF16, FP8])
@pytest.mark.parametrize(
    "draft_heads,draft_head_size,target_heads,target_head_size",
    [
        (8, 64, 8, 128),  # head_dim=64 draft in a v3 head_dim=128 pool
        (4, 128, 4, 256),  # narrower draft in a wider pool (#316)
    ],
)
def test_page_from_another_spec_is_padded_to_its_width(
    draft_heads, draft_head_size, target_heads, target_head_size, dtype
):
    pool = _page(target_heads, target_head_size, dtype)
    native = _page(draft_heads, draft_head_size, dtype)

    assert kv_page_head_padding(pool, native, pool[-1], draft_head_size) == (
        pool[-1] - draft_head_size
    )


def test_page_no_wider_than_the_head_is_never_padded():
    pool = _page(8, 128, BF16)
    native = _page(4, 128, BF16)

    assert kv_page_head_padding(pool, native, 128, 128) == 0
