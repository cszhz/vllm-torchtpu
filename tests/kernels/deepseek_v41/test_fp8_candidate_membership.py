# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-query membership, including duplicates and invalid union slots."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8_candidate_cache import (
    candidate_membership,
)


@pytest.mark.parametrize(
    "rows,count,columns,blocks",
    [
        (1, 3, 7, 17),
        (7, 129, 258, 257),
        (288, 2048, 4096, 12352),
    ],
)
def test_candidate_membership_matches_per_query_set(rows, count, columns, blocks):
    rng = np.random.default_rng(891)
    candidates = rng.integers(-3, blocks + 3, (rows, count), dtype=np.int32)
    canonical = rng.integers(-3, blocks + 3, (rows, columns), dtype=np.int32)
    candidates[0] = -1
    if rows > 1:
        candidates[1] = 0
        canonical[1, :3] = [0, -1, blocks]
    expected = np.stack(
        [
            np.isin(union, source) & (union >= 0) & (union < blocks)
            for source, union in zip(candidates, canonical, strict=True)
        ]
    )
    run = jax.jit(functools.partial(candidate_membership, blocks=blocks))
    actual = run(jnp.asarray(candidates), jnp.asarray(canonical))
    np.testing.assert_array_equal(np.asarray(actual), expected)
    # Reuse the same executable with changed data: no stale per-row histogram.
    actual = run(jnp.full_like(candidates, -1), jnp.asarray(canonical))
    assert not np.any(np.asarray(actual))
