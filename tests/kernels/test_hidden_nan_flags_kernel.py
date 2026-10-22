# SPDX-License-Identifier: Apache-2.0
"""First-hit ownership and immediate device-side suspension protect snapshots."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.hidden_nan_flags import record_hidden_nan


def buffers(dtype=jnp.float32):
    return (
        jnp.array([-1, 0, 0, 0], jnp.int32),
        jnp.zeros(8 * 128, dtype),
        jnp.zeros(8, jnp.int32),
    )


def check(current, hidden, tokens, slot, rounds, armed=True):
    return record_hidden_nan(
        *current,
        jnp.array([armed]),
        hidden,
        tokens,
        jnp.array([tokens.size]),
        jnp.array(slot),
        rounds,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("round_id", [1, 105])
@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_first_nonfinite_copies_full_hidden_and_tokens_once(dtype, round_id, bad_value):
    current = buffers(dtype)
    tokens = jnp.array([15, 27, 39], jnp.int32)
    good = jnp.ones((3, 128), dtype)
    bad = good.at[2, 127].set(bad_value)
    rounds = jnp.array([round_id, 0], jnp.int32)
    result = check(current, good, tokens, 0, rounds)
    np.testing.assert_array_equal(
        np.asarray(result[4]).view(np.uint8), np.asarray(good).view(np.uint8)
    )
    for actual, expected in zip(result[:3], current):
        np.testing.assert_array_equal(actual, expected)
    result = check(result[:3], bad, tokens, 1, result[3])
    np.testing.assert_array_equal(
        np.asarray(result[4]).view(np.uint8), np.asarray(bad).view(np.uint8)
    )
    np.testing.assert_array_equal(result[0], [1, round_id, 3, 3])
    np.testing.assert_array_equal(result[3], [round_id, round_id])
    np.testing.assert_array_equal(
        np.asarray(result[1][: bad.size].reshape(bad.shape), dtype=np.float32),
        np.asarray(bad, dtype=np.float32),
    )
    np.testing.assert_array_equal(result[2][:3], tokens)
    later = check(result[:3], jnp.full_like(bad, jnp.nan), tokens + 1, 2, result[3])
    for actual, expected in zip(later[:4], result[:4]):
        np.testing.assert_array_equal(
            np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
        )


def test_both_roles_capture_same_round_but_later_rounds_wait_for_host_release():
    bad = jnp.full((3, 128), jnp.nan)
    tokens = jnp.array([15, 27, 39], jnp.int32)
    record = jax.jit(check)
    target = record(buffers(), bad, tokens, 7, jnp.array([1, 0]))
    draft = record(buffers(), bad * 3, tokens + 1, 11, target[3])
    assert int(target[0][0]) == 7 and int(draft[0][0]) == 11
    # Even a model which did not hit in round 1 must stop inspecting. This
    # cannot wait for an asynchronous CPU consumer to learn about the NaN.
    clean = buffers()
    for round_id in (2, 21, 22, 100):
        paused = record(clean, bad, tokens, 0, jnp.array([round_id, 1]))
        assert int(paused[0][0]) == -1
        np.testing.assert_array_equal(paused[1], clean[1])
    resumed = record(clean, bad, tokens, 0, jnp.array([101, 0]))
    np.testing.assert_array_equal(resumed[0], [0, 101, 3, 3])


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_unarmed_warmup_preserves_snapshots_and_round_gate(bad_value):
    before = buffers()
    rounds = jnp.array([1, 0])
    after = check(
        before, jnp.full((1, 128), bad_value), jnp.array([42]), 0, rounds, False
    )
    for actual, expected in zip(after[:4], (*before, rounds)):
        np.testing.assert_array_equal(actual, expected)
