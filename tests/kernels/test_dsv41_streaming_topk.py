# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-row prefill streaming top-k: stable ties and finite padding contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.streaming_topk import (
    select_candidate_tokens,
    streaming_topk,
)


@pytest.mark.parametrize("width", [2**20, 2**20 + 3])
def test_representative_block_max_long_context(width):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.publication import (
        representative_block_max,
    )

    scores = np.random.default_rng(23).standard_normal((3, width)).astype(np.float32)
    scores[:, -16:] = -np.inf
    rows = np.array([2, 0, 2], np.int32)
    expected = np.pad(scores[rows], ((0, 0), (0, -width % 8)), constant_values=-np.inf)
    actual = jax.jit(lambda x, r: representative_block_max(x, r, block=8))(
        jnp.asarray(scores), jnp.asarray(rows)
    )
    np.testing.assert_array_equal(actual, expected.reshape(3, -1, 8).max(-1))


@pytest.mark.parametrize("unordered", [False, True])
@pytest.mark.parametrize("k", [0, 16])
def test_select_candidate_tokens_maps_logical_ids(unordered, k):
    candidates = jnp.array([[1, 4, -1, -1], [2, 7, 9, -1]], jnp.int32)
    scores = np.tile(np.arange(32, dtype=np.float32) % 3, (2, 1))
    scores[0, 10:] = -np.inf
    scores[1] = -np.inf
    actual = np.asarray(
        select_candidate_tokens(
            jnp.asarray(scores), candidates, k=k, block_size=8, unordered=unordered
        )
    )
    for row in range(2):
        eligible = np.flatnonzero(np.isfinite(scores[row]))
        slots = eligible[np.argsort(-scores[row, eligible], kind="stable")[:k]]
        logical = np.asarray(candidates)[row, slots // 8] * 8 + slots % 8
        expected = np.pad(np.sort(logical), (0, k - slots.size), constant_values=-1)
        np.testing.assert_array_equal(actual[row], expected)


@pytest.mark.parametrize("block,threshold", [(8192, 8192), (16384, 32768)])
@pytest.mark.parametrize("width", [65536, 65408])
@pytest.mark.parametrize(
    "distribution",
    [
        "random",
        "ascending",
        "descending",
        "equal",
        "cross_block_ties",
        "signed_zero",
        "masked",
    ],
)
def test_streaming_topk_contract(block, threshold, width, distribution):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(32, width)).astype(np.float32)
    if distribution == "ascending":
        scores[:] = np.arange(width, dtype=np.float32)
    elif distribution == "descending":
        scores[:] = -np.arange(width, dtype=np.float32)
    elif distribution == "equal":
        scores[:] = 1
    elif distribution == "cross_block_ties":
        # Equal Kth values survive repeated candidate compactions, with later
        # blocks also supplying larger winners that displace earlier ties.
        scores[:] = np.arange(width) // 4096 % 5
    elif distribution == "signed_zero":
        scores[:] = 0
        scores[:, ::2] = np.float32(-0.0)
    elif distribution == "masked":
        scores[:, ::3] = -np.inf
        scores[:4] = -np.inf
    lengths = np.resize(
        np.array([0, 1, 511, 512, 513, 8191, 8193, 16383, 50176, width], np.int32), 32
    )
    actual = np.asarray(
        streaming_topk(
            jnp.asarray(scores),
            valid_lengths=jnp.asarray(lengths),
            k=512,
            block_size=block,
            reconstruct_threshold=threshold,
        )
    )
    for row, length in enumerate(lengths):
        ids = actual[row]
        ids = ids[(ids >= 0) & (scores[row, np.maximum(ids, 0)] > -np.inf)]
        eligible = np.flatnonzero((np.arange(width) < length) & (scores[row] > -np.inf))
        bits = scores[row, eligible].view(np.uint32)
        keys = bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
        order = np.lexsort((eligible, ~keys))[:512]
        np.testing.assert_array_equal(np.sort(ids), np.sort(eligible[order]))
        assert np.unique(ids).size == ids.size
