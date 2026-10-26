# SPDX-License-Identifier: Apache-2.0
"""The SparseCore pair scatter, against numpy and against its own block size.

`scatter_pair` writes the routing plan's two slab tables in one kernel. It
stages the pair arrays a block at a time; the block is the whole array
wherever that fits, so the geometries that ran before chunking existed run
the same program and only the large ones stream. Both are checked here: the
same inputs through a one-block and a many-block run, and both against the
scatter they replace.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2.scatter import (
    can_scatter_pair,
    scatter_pair,
)

# (pairs, token slab rows, scale slab rows) -- the DP32 C768 decode shape,
# the world-8 shape the device tests run, and a small one.
CASES = [(27648, 28288, 28416), (6912, 7168, 7296), (1024, 1152, 1280)]


@pytest.fixture(scope="module", autouse=True)
def _require_a_sparsecore():
    """Skip the module off the chip.

    A fixture rather than a `skipif`, for the reason
    `test_fused_ep_moe_v2.py::_require_eight_devices` gives: a `skipif` is
    evaluated during collection, and asking jax for a backend there builds
    the PJRT client in the pytest parent, which libtpu's process-exclusive
    lock then denies to every spawned worker in the same run.
    """
    if jax.default_backend() != "tpu":
        pytest.skip("the pair scatter is a SparseCore kernel")


def _reference(rows, scales, tokens, token_size, scale_size, n, topk):
    tok = np.zeros(token_size, np.int32)
    scl = np.zeros(scale_size, np.int32)
    np.add.at(tok, rows[rows < token_size], tokens[rows < token_size])
    np.add.at(scl, rows[rows < scale_size], scales[rows < scale_size])
    return np.clip(tok, 0, n // topk - 1), scl


@pytest.mark.parametrize("n,token_size,scale_size", CASES)
def test_pair_scatter_matches_the_scatter_it_replaces(n, token_size, scale_size):
    if not can_scatter_pair(n, token_size, scale_size):
        pytest.skip("this shape does not fit the SparseCore scratch budget")
    topk = 6
    rng = np.random.default_rng(n)
    # A third of the pairs land off this shard, as the plan's sentinel sends
    # them: past both slab sizes, where the kernel must drop them.
    rows = rng.integers(0, max(token_size, scale_size), size=n).astype(np.int32)
    rows[rng.random(n) < 0.33] = np.int32(max(token_size, scale_size) + 7)
    scales = rng.integers(-(2**30), 2**30, size=n).astype(np.int32)
    tokens = rng.integers(0, n // topk, size=n).astype(np.int32)

    want = _reference(rows, scales, tokens, token_size, scale_size, n, topk)
    run = jax.jit(
        scatter_pair,
        static_argnames=("token_size", "scale_size", "topk", "pair_block"),
    )
    for block in (None, 512):
        tok, scl = run(
            jnp.asarray(rows),
            jnp.asarray(scales),
            token_size=token_size,
            scale_size=scale_size,
            topk=topk,
            tokens=jnp.asarray(tokens),
            pair_block=block,
        )
        np.testing.assert_array_equal(np.asarray(tok), want[0])
        np.testing.assert_array_equal(np.asarray(scl), want[1])
