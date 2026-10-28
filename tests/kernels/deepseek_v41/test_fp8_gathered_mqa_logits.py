# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check native FP8 dots, NumPy FP32 reduction, and query-local score slots."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tests.kernels.deepseek_v41.test_fp8_paged_mqa_logits import _pack_fp8_cache
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8_paged_mqa_logits_sparse import (  # noqa: E501
    _query_scores,
    fp8_gathered_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PackedFP8Layout,
)


@pytest.mark.parametrize("count", [17, 2048])
def test_query_slot_projection_preserves_all_blocks(count):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires the SparseCore projection")
    rng = np.random.default_rng(829)
    single = np.arange(6 * 4096, dtype=np.float32).reshape(6, 32, 128)
    pairs = np.arange(6 * 2 * 4096, dtype=np.float32).reshape(6, 2, 32, 128)
    pairs += 100000
    # Two tiles per group; the third row is a single and the fourth is empty.
    control = np.array(
        [[0, 0, 1, 8192], [0, 1, 1, 8192], [2, 0, 0, 8192], [4, 0, 1, 0]],
        dtype=np.int32,
    )
    slots = rng.integers(0, 1024, size=(4, count), dtype=np.int32)
    slots[:, 2::5] = -1
    expected = []
    for row, (base, member, paired, length) in enumerate(control):
        token = np.maximum(slots[row], 0)[:, None] * 8 + np.arange(8)
        tile = base + token // 4096
        column = token % 4096
        values = (
            pairs.reshape(6, 2, 4096)[tile, member, column]
            if paired
            else single.reshape(6, 4096)[tile, column]
        )
        expected.append(
            np.where((slots[row, :, None] >= 0) & (token < length), values, -np.inf)
        )
    actual = jax.jit(lambda a, b, c, d: _query_scores(a, b, c, d, block=8))(
        jnp.asarray(single),
        jnp.asarray(pairs),
        jnp.asarray(slots),
        jnp.pad(jnp.asarray(control), ((0, 0), (0, 124))),
    )
    np.testing.assert_array_equal(
        np.asarray(actual), np.asarray(expected).reshape(4, -1)
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
def test_gathered_logits_match_fp32_and_query_masks(
    width, key_tile, start, count, record_property
):
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
    weights = jnp.asarray(rng.uniform(-0.1, 0.1, (7, 32)), jnp.float32)
    lengths = jnp.array([width, width - 1, width // 3, 0, 0, width - 7, width])
    offsets = jnp.array([0, 0, 2, 3, 3, 5, 7], jnp.int32)
    pages = jnp.asarray(np.stack([rng.permutation(layout.pages) for _ in range(6)]))
    gathered = cache.reshape(layout.pages, 33, 128)[pages].reshape(-1, 128)
    membership = rng.random((7, width // block)) > 0.5
    membership[6] = False
    words = np.asarray(cache).reshape(layout.pages, 33, 128)
    packed = np.ascontiguousarray(words[:, :32].transpose(0, 2, 1))
    raw_keys = (
        packed.view(np.uint8)
        .reshape(-1, 128)
        .view(jnp.float8_e4m3fn)
        .astype(np.float32)
    )
    scales = words[:, 32].view(np.float32).reshape(-1)
    host_q, host_w = np.asarray(q).astype(np.float32), np.asarray(weights)
    baseline, ideal = [], []
    native_dot = jax.jit(
        lambda query, key: jax.lax.dot(query, key.T, preferred_element_type=jnp.float32)
    )
    for row in range(start, start + count):
        group = np.searchsorted(np.asarray(offsets)[1:], row, side="right")
        records = (np.asarray(pages[group])[:, None] * 128 + np.arange(128)).reshape(-1)
        # Native FP8 MXU arithmetic differs from ideal FP32 multiplication.
        # Use an independent XLA dot, then do every remaining step in NumPy FP32.
        products = np.asarray(
            native_dot(q[row], jnp.asarray(raw_keys[records], dtype=q.dtype))
        )
        values = (np.maximum(products, 0) * host_w[row, :, None]).sum(axis=0) * scales[
            records
        ]
        baseline.append(np.where(np.arange(width) < int(lengths[row]), values, -np.inf))
        full_products = host_q[row] @ raw_keys[records].T
        ideal.append(
            (np.maximum(full_products, 0) * host_w[row, :, None]).sum(axis=0)
            * scales[records]
        )
    baseline = np.asarray(baseline)
    # Per-query candidates have different orders and half the union capacity.
    slots = np.stack(
        [rng.permutation(width // block)[: width // block // 2] for _ in range(7)]
    )
    slots[0, 0] = -1
    slots[1, 1] = slots[1, 0]
    output_slots = np.full((7, width // block), -1, np.int32)
    for row in range(7):
        for destination, source in enumerate(slots[row]):
            if source >= 0:
                output_slots[row, source] = destination
    actual = fp8_gathered_mqa_logits(
        q,
        gathered,
        weights,
        lengths,
        jnp.asarray(membership),
        query_start_loc=offsets,
        candidate_slots=jnp.asarray(slots, jnp.int32),
        output_slots=jnp.asarray(output_slots),
        max_model_len=width,
        membership_block_size=block,
        query_start=jnp.int32(start),
        query_count=count,
        key_tile=key_tile,
    )
    allowed = np.repeat(membership[start : start + count], block, axis=1)
    expected = np.where(allowed, np.asarray(baseline), -np.inf)
    selected_slots = slots[start : start + count]
    columns = (
        np.maximum(selected_slots, 0)[..., None] * block + np.arange(block)
    ).reshape(count, -1)
    expected = np.take_along_axis(expected, columns, axis=1)
    expected = np.where(
        np.repeat(selected_slots >= 0, block, axis=1), expected, -np.inf
    )
    assert actual.shape == (count, width // 2)
    assert actual.dtype == jnp.float32
    actual = np.asarray(actual)
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    live = np.isfinite(expected)
    np.testing.assert_allclose(actual[live], expected[live], rtol=2e-5, atol=2e-5)
    ideal = np.take_along_axis(np.asarray(ideal), columns, axis=1)[live]
    x, y = actual[live].astype(np.float64), ideal.astype(np.float64)
    diff = np.square(x - y).sum() / (np.square(x).sum() + np.square(y).sum())
    record_property("fp32_reference_diff", diff)
    # Same numerical acceptance criterion as the existing dense MQA test.
    assert diff < 1e-3
