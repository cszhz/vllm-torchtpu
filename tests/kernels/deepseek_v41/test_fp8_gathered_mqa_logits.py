# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check contiguous-union scores against the independent paged dense path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tests.kernels.deepseek_v41.test_fp8_paged_mqa_logits import _pack_fp8_cache
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8_paged_mqa_logits_dense import (  # noqa: E501
    fp8_paged_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8_paged_mqa_logits_sparse import (  # noqa: E501
    fp8_gathered_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PackedFP8Layout,
)


@pytest.mark.parametrize(
    "width,key_tile,start,count",
    [
        (128, 256, 0, 7),
        (8192, 4096, 0, 7),
        (8192, 8192, 1, 5),
        (32768, 32768, 0, 7),
        (32768, 8192, 1, 5),
    ],
)
def test_gathered_logits_match_dense_and_query_masks(width, key_tile, start, count):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires the TPU MQA kernels")
    rng = np.random.default_rng(821)
    torch.manual_seed(821)
    block = 8
    layout = PackedFP8Layout(width // 128)
    cache = _pack_fp8_cache(
        torch.randn((layout.pages, 128, 1, 128), dtype=torch.bfloat16), layout
    )
    q = jnp.asarray(rng.normal(size=(7, 32, 128))).astype(jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (7, 32)), jnp.float32)
    lengths = jnp.array([width, width - 1, width // 2, 0, 0, width - 7, width])
    offsets = jnp.array([0, 0, 2, 3, 3, 5, 7], jnp.int32)
    pages = jnp.asarray(np.stack([rng.permutation(layout.pages) for _ in range(6)]))
    gathered = cache.reshape(layout.pages, 33, 128)[pages].reshape(-1, 128)
    membership = rng.random((7, width // block)) > 0.5
    membership[6] = False
    baseline = fp8_paged_mqa_logits(
        q,
        cache,
        weights,
        lengths,
        pages,
        query_start_loc=offsets,
        max_model_len=width,
        q_scale=jnp.ones_like(weights),
        query_size=2,
        cache_layout=layout,
        query_start=jnp.int32(start),
        query_count=count,
    )
    actual = fp8_gathered_mqa_logits(
        q,
        gathered,
        weights,
        lengths,
        jnp.asarray(membership),
        query_start_loc=offsets,
        max_model_len=width,
        membership_block_size=block,
        query_start=jnp.int32(start),
        query_count=count,
        key_tile=key_tile,
    )
    allowed = np.repeat(membership[start : start + count], block, axis=1)
    expected = np.where(allowed, np.asarray(baseline), -np.inf)
    actual = np.asarray(actual)
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    live = np.isfinite(expected)
    np.testing.assert_allclose(actual[live], expected[live], rtol=0.01, atol=0.01)
