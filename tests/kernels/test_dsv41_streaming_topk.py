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
@pytest.mark.parametrize("sorted_indices", [False, True])
@pytest.mark.parametrize("k", [0, 16])
def test_select_candidate_tokens_maps_logical_ids(unordered, sorted_indices, k):
    candidates = jnp.array([[1, 4, -1, -1], [2, 7, 9, -1]], jnp.int32)
    scores = np.tile(np.arange(32, dtype=np.float32) % 3, (2, 1))
    scores[0, 10:] = -np.inf
    scores[1] = -np.inf
    actual = np.asarray(
        select_candidate_tokens(
            jnp.asarray(scores),
            candidates,
            k=k,
            block_size=8,
            unordered=unordered,
            sorted_indices=sorted_indices,
        )
    )
    for row in range(2):
        eligible = np.flatnonzero(np.isfinite(scores[row]))
        slots = eligible[np.argsort(-scores[row, eligible], kind="stable")[:k]]
        logical = np.asarray(candidates)[row, slots // 8] * 8 + slots % 8
        expected = np.pad(np.sort(logical), (0, k - slots.size), constant_values=-1)
        if sorted_indices:
            np.testing.assert_array_equal(actual[row], expected)
        else:
            np.testing.assert_array_equal(
                np.sort(actual[row, : slots.size]), np.sort(logical)
            )
            np.testing.assert_array_equal(actual[row, slots.size :], -1)


@pytest.mark.parametrize(
    "block,threshold,unroll,bootstrap",
    [(8192, 8192, 1, 4), (16384, 32768, 1, 4), (None, 2048, 2, 8)],
)
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
def test_streaming_topk_contract(
    block, threshold, unroll, bootstrap, width, distribution
):
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
            loop_unroll=unroll,
            bootstrap_multiplier=bootstrap,
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


@pytest.mark.parametrize("width", [8192, 32768])
@pytest.mark.parametrize("sorted_indices", [False, True])
@pytest.mark.parametrize("requests", [16, 2])
def test_streaming_topk_maps_request_window(width, sorted_indices, requests):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    # 48 rows select per subcore; 6 rows column-partition each row.
    queries, k, records_per_block = 3, 512, 8
    rows = requests * queries
    rng = np.random.default_rng(20261026)
    scores = rng.integers(-3, 4, (rows, width)).astype(np.float32)
    scores[:, ::7] = -np.inf
    lengths = np.resize(np.array([0, 1, 511, 512, 513, width], np.int32), rows)
    blocks = np.arange(width // records_per_block, dtype=np.int32)[None, :] * 2
    blocks = blocks + np.arange(requests + 1, dtype=np.int32)[:, None] * width

    @jax.jit
    def select(values, limits, mapping, start):
        return streaming_topk(
            values,
            valid_lengths=limits,
            logical_blocks=mapping,
            start_request=start,
            k=k,
            exclude_neg_inf=True,
            reconstruct_threshold=2048,
            loop_unroll=2,
            bootstrap_multiplier=8,
            sorted_indices=sorted_indices,
        )

    for start in (0, 1):
        actual = np.asarray(
            select(
                jnp.asarray(scores),
                jnp.asarray(lengths.reshape(requests, queries)),
                jnp.asarray(blocks),
                jnp.asarray(start, jnp.int32),
            )
        )
        for row, length in enumerate(lengths):
            eligible = np.flatnonzero(
                (np.arange(width) < length) & np.isfinite(scores[row])
            )
            slots = eligible[np.argsort(-scores[row, eligible], kind="stable")[:k]]
            logical = (
                blocks[start + row // queries, slots // records_per_block]
                * records_per_block
                + slots % records_per_block
            )
            if sorted_indices:
                expected = np.pad(
                    np.sort(logical), (0, k - slots.size), constant_values=-1
                )
                np.testing.assert_array_equal(actual[row], expected)
            else:
                np.testing.assert_array_equal(
                    np.sort(actual[row][actual[row] >= 0]), np.sort(logical)
                )
                assert np.count_nonzero(actual[row] == -1) == k - slots.size


# Rows that fit fewer than 32 subcores split their columns across a group.
# Short lengths leave whole slices empty; -1 marks a full-width row.
@pytest.mark.parametrize(
    "lengths",
    [(-1,), (513,), (1,), (0,), (-1, 4097), (0, 1, 511, 512, 513, 8193, 50176, -1)],
)
@pytest.mark.parametrize("width", [65536, 65408])
@pytest.mark.parametrize(
    "distribution", ["random", "equal", "cross_block_ties", "signed_zero", "masked"]
)
@pytest.mark.parametrize("exclude_neg_inf", [False, True])
def test_streaming_topk_partitioned_rows(lengths, width, distribution, exclude_neg_inf):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    k = 512
    lengths = np.array([width if n < 0 else n for n in lengths], np.int32)
    rows = lengths.size
    rng = np.random.default_rng(29)
    scores = rng.normal(size=(rows, width)).astype(np.float32)
    if distribution == "equal":
        scores[:] = 1
    elif distribution == "cross_block_ties":
        scores[:] = np.arange(width) // 4096 % 5
    elif distribution == "signed_zero":
        scores[:] = 0
        scores[:, ::2] = np.float32(-0.0)
    elif distribution == "masked":
        scores[:, ::3] = -np.inf
    actual = np.asarray(
        streaming_topk(
            jnp.asarray(scores),
            valid_lengths=jnp.asarray(lengths),
            k=k,
            exclude_neg_inf=exclude_neg_inf,
            reconstruct_threshold=2048,
            loop_unroll=2,
            bootstrap_multiplier=8,
        )
    )
    for row, length in enumerate(lengths):
        visible = np.arange(width) < length
        if exclude_neg_inf:
            visible &= scores[row] > -np.inf
        eligible = np.flatnonzero(visible)
        bits = scores[row, eligible].view(np.uint32)
        keys = bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
        best = np.sort(eligible[np.lexsort((eligible, ~keys))[:k]])
        expected = np.pad(best, (0, k - best.size), constant_values=-1)
        np.testing.assert_array_equal(actual[row], expected)


@pytest.mark.parametrize("batch", [1, 2, 5, 16])
@pytest.mark.parametrize("width", [8192, 65408])
@pytest.mark.parametrize(
    "distribution", ["random", "equal", "cross_block_ties", "signed_zero", "masked"]
)
@pytest.mark.parametrize("exclude_neg_inf", [False, True])
def test_streaming_topk_partitioned_batches(
    batch, width, distribution, exclude_neg_inf
):
    """Small batches split rows across subcores; the merge must stay exact."""
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    rng = np.random.default_rng(batch * 1009 + width)
    scores = rng.normal(size=(batch, width)).astype(np.float32)
    if distribution == "equal":
        scores[:] = 1
    elif distribution == "cross_block_ties":
        scores[:] = np.arange(width) // 1024 % 5
    elif distribution == "signed_zero":
        scores[:] = 0
        scores[:, ::2] = np.float32(-0.0)
    elif distribution == "masked":
        scores[:, ::3] = -np.inf
        scores[:1] = -np.inf
    # Short rows leave most slices empty; others end mid-slice.
    lengths = np.resize(
        np.array([width, 0, 1, 511, 513, width // 3 + 7, width - 129], np.int32),
        batch,
    )
    actual = np.asarray(
        streaming_topk(
            jnp.asarray(scores),
            valid_lengths=jnp.asarray(lengths),
            k=512,
            exclude_neg_inf=exclude_neg_inf,
            reconstruct_threshold=2048,
            loop_unroll=2,
            bootstrap_multiplier=8,
        )
    )
    for row, length in enumerate(lengths):
        eligible = np.flatnonzero(
            (np.arange(width) < length)
            & ((scores[row] > -np.inf) if exclude_neg_inf else True)
        )
        bits = scores[row, eligible].view(np.uint32)
        keys = bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
        best = np.sort(eligible[np.lexsort((eligible, ~keys))[:512]])
        expected = np.pad(best, (0, 512 - best.size), constant_values=-1)
        np.testing.assert_array_equal(actual[row], expected)
