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
"""Correctness tests for Mamba-2 Selective State Space (SSD) scan kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.ssd_scan import chunk_scan_pallas, chunk_state_pallas
from vllm_torchtpu.kernels.ssd_scan.ssd_scan import _select_head_block_size
from vllm_torchtpu.layers.core.ssd_scan import _segsum_jax, ssd_minimal_discrete_hybrid


def ssd_minimal_discrete_jax(
    X_c: jax.Array,
    A_c: jax.Array,
    B_c: jax.Array,
    C_c: jax.Array,
    last_chunk_indices: jax.Array,
    initial_states: jax.Array | None = None,
    reset_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Pure JAX reference for SSD scan with float32 inter-chunk state carry."""
    n_chunks, n_heads, _, d_head = X_c.shape
    d_state = B_c.shape[-1]

    if initial_states is None:
        initial_states_f32 = jnp.zeros(
            (n_chunks, n_heads, d_head, d_state), dtype=jnp.float32
        )
    else:
        initial_states_f32 = initial_states.astype(jnp.float32)

    if reset_mask is None:
        reset_mask = jnp.zeros((n_chunks,), dtype=bool).at[0].set(True)

    A_c_f32 = A_c.astype(jnp.float32)
    A_cumsum = jnp.cumsum(A_c_f32, axis=-1)

    L = jnp.exp(_segsum_jax(A_c_f32))
    Y_diag = jnp.einsum(
        "chln,chsn,chls,chsp->chlp",
        C_c.astype(jnp.float32),
        B_c.astype(jnp.float32),
        L,
        X_c.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )

    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
    states_c = jnp.einsum(
        "chln,chlp,chl->chpn",
        B_c.astype(jnp.float32),
        X_c.astype(jnp.float32),
        decay_states,
        precision=jax.lax.Precision.HIGHEST,
    )

    decay_chunk = jnp.exp(A_cumsum[..., -1])

    def scan_body(carry, args):
        states_c_c, decay_chunk_c, init_state_c, reset_c = args
        incoming_state = jnp.where(reset_c, init_state_c, carry)
        decay_exp = decay_chunk_c[:, None, None]
        next_state = incoming_state * decay_exp + states_c_c
        return next_state, (incoming_state, next_state)

    scan_args = (states_c, decay_chunk, initial_states_f32, reset_mask)
    init_carry = jnp.zeros((n_heads, d_head, d_state), dtype=jnp.float32)
    _, (states_incoming, states_outgoing) = jax.lax.scan(
        scan_body, init_carry, scan_args
    )

    state_decay_out = jnp.exp(A_cumsum)
    Y_off = jnp.einsum(
        "chln,chpn,chl->chlp",
        C_c.astype(jnp.float32),
        states_incoming,
        state_decay_out,
        precision=jax.lax.Precision.HIGHEST,
    )

    Y = (Y_diag + Y_off).astype(X_c.dtype)
    final_states = states_outgoing[last_chunk_indices]
    return Y, final_states


def _make_ssd_inputs(
    batch: int,
    seqlen: int,
    block_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    *,
    dtype=jnp.float32,
):
    n_chunks_per_seq = seqlen // block_len
    n_chunks = batch * n_chunks_per_seq

    rng = jax.random.key(42 + n_heads + block_len)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

    X_c = jax.random.normal(k1, (n_chunks, n_heads, block_len, d_head), dtype=dtype)
    A_c = -jnp.exp(
        jax.random.normal(k2, (n_chunks, n_heads, block_len), dtype=jnp.float32)
    )
    B_c = jax.random.normal(k3, (n_chunks, n_heads, block_len, d_state), dtype=dtype)
    C_c = jax.random.normal(k4, (n_chunks, n_heads, block_len, d_state), dtype=dtype)
    init_states_per_batch = jax.random.normal(
        k5, (batch, n_heads, d_head, d_state), dtype=dtype
    )

    initial_states = jnp.zeros((n_chunks, n_heads, d_head, d_state), dtype=dtype)
    reset_mask = np.zeros((n_chunks,), dtype=bool)
    for b in range(batch):
        initial_states = initial_states.at[b * n_chunks_per_seq].set(
            init_states_per_batch[b]
        )
        reset_mask[b * n_chunks_per_seq] = True

    last_chunk_indices = jnp.array(
        [b * n_chunks_per_seq + n_chunks_per_seq - 1 for b in range(batch)],
        dtype=jnp.int32,
    )
    return X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, jnp.array(reset_mask)


def test_select_head_block_size():
    assert _select_head_block_size(1) == 1
    assert _select_head_block_size(3) == 3
    assert _select_head_block_size(6) == 6
    assert _select_head_block_size(8) == 8
    assert _select_head_block_size(12) == 6
    assert _select_head_block_size(16) == 8
    assert _select_head_block_size(24) == 8


@pytest.mark.parametrize(
    ("batch", "seqlen", "block_len", "n_heads", "d_head", "d_state"),
    [
        (2, 16, 4, 4, 8, 8),
        (2, 128, 32, 4, 16, 8),
        (4, 256, 64, 8, 32, 16),
        (2, 64, 16, 6, 16, 8),
        (2, 32, 16, 1, 16, 8),
    ],
)
def test_ssd_scan_correctness(
    batch: int,
    seqlen: int,
    block_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
):
    X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(
            batch, seqlen, block_len, n_heads, d_head, d_state, dtype=jnp.float32
        )
    )

    y_jax, final_state_jax = ssd_minimal_discrete_jax(
        X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask
    )
    y_hybrid, final_state_hybrid = ssd_minimal_discrete_hybrid(
        X_c,
        A_c,
        B_c,
        C_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        interpret=True,
    )

    assert final_state_hybrid.dtype == jnp.float32
    np.testing.assert_allclose(y_hybrid, y_jax, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        final_state_hybrid, final_state_jax, rtol=1e-4, atol=1e-4
    )


def test_ssd_scan_bfloat16_inputs_preserve_float32_state_carry():
    X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(2, 64, 16, 6, 16, 8, dtype=jnp.bfloat16)
    )

    A_cumsum = jnp.cumsum(A_c, axis=-1)
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)[..., None]
    states_c = chunk_state_pallas(B_c, decay_states, X_c, interpret=True)
    assert states_c.dtype == jnp.float32

    y_out_pallas = chunk_scan_pallas(
        C_c, states_c, jnp.exp(A_cumsum)[..., None], interpret=True
    )
    assert y_out_pallas.dtype == jnp.bfloat16

    y_jax, final_state_jax = ssd_minimal_discrete_jax(
        X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask
    )
    y_hybrid, final_state_hybrid = ssd_minimal_discrete_hybrid(
        X_c,
        A_c,
        B_c,
        C_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        interpret=True,
    )

    assert y_hybrid.dtype == jnp.bfloat16
    assert final_state_hybrid.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(final_state_hybrid),
        np.asarray(final_state_jax),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(y_hybrid.astype(jnp.float32)),
        np.asarray(y_jax.astype(jnp.float32)),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(
    ("batch", "seqlen", "block_len", "n_heads", "d_head", "d_state"),
    [
        (2, 128, 32, 4, 16, 8),
        (2, 128, 64, 6, 64, 128),
        (2, 256, 64, 16, 64, 128),
    ],
)
def test_ssd_scan_correctness_tpu(
    dtype,
    batch: int,
    seqlen: int,
    block_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
):
    X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(
            batch, seqlen, block_len, n_heads, d_head, d_state, dtype=dtype
        )
    )

    y_jax, final_state_jax = ssd_minimal_discrete_jax(
        X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask
    )
    y_hybrid, final_state_hybrid = ssd_minimal_discrete_hybrid(
        X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask
    )

    assert y_hybrid.dtype == dtype
    assert final_state_hybrid.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(final_state_hybrid),
        np.asarray(final_state_jax),
        rtol=1e-3,
        atol=1e-3,
    )
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-3
    np.testing.assert_allclose(
        np.asarray(y_hybrid.astype(jnp.float32)),
        np.asarray(y_jax.astype(jnp.float32)),
        rtol=tol,
        atol=tol,
    )
