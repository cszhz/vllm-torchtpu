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
"""JAX/Pallas hybrid implementation of Selective State Space (SSD) scan."""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.ssd_scan import chunk_scan_pallas, chunk_state_pallas


def _segsum_jax(x: jax.Array) -> jax.Array:
    """Computes JAX cumsum of segment sums."""
    T = x.shape[-1]
    x_expanded = jnp.expand_dims(x, axis=-1)
    x_repeated = jnp.broadcast_to(x_expanded, x.shape + (T,))

    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x_masked = jnp.where(mask, x_repeated, 0.0)

    x_segsum = jnp.cumsum(x_masked, axis=-2)

    mask_diag = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    x_segsum_masked = jnp.where(mask_diag, x_segsum, -jnp.inf)
    return x_segsum_masked


def ssd_minimal_discrete_hybrid(
    X_c: jax.Array,
    A_c: jax.Array,
    B_c: jax.Array,
    C_c: jax.Array,
    last_chunk_indices: jax.Array,
    initial_states: jax.Array | None = None,
    reset_mask: jax.Array | None = None,
    *,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Hybrid JAX/Pallas implementation of the Selective State Space (SSD) scan.

    Implements the discrete SSD scan using Pallas TPU kernels (`kernels/ssd_scan/`)
    for intra-chunk state generation and state-to-output conversion, and native
    JAX for cumulative sums and f32 sequential inter-chunk recurrence.

    Assumes packed layout: `(n_chunks, n_heads, block_len, dim)`.

    Args:
      X_c: (n_chunks, n_heads, block_len, d_head) - Discretized input tensor
      A_c: (n_chunks, n_heads, block_len) - Log decay parameter (dt * A)
      B_c: (n_chunks, n_heads, block_len, d_state) - Input projection
      C_c: (n_chunks, n_heads, block_len, d_state) - Output projection
      last_chunk_indices: (num_seqs,) - Maps each sequence to its last chunk
      initial_states: (n_chunks, n_heads, d_head, d_state) - Optional initial
        states (read only where `reset_mask` is True)
      reset_mask: (n_chunks,) - Boolean mask indicating sequence starts
      interpret: Run Pallas kernels in host interpret mode (CPU tests only)

    Returns:
      Y: (n_chunks, n_heads, block_len, d_head) in `X_c.dtype`
      final_states: (num_seqs, n_heads, d_head, d_state) in `float32`
    """
    n_chunks, n_heads, block_len, d_head = X_c.shape
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

    # 1. Intra-chunk diagonal output (Y_diag)
    L = jnp.exp(_segsum_jax(A_c_f32))
    Y_diag = jnp.einsum(
        "chln,chsn,chls,chsp->chlp",
        C_c,
        B_c,
        L,
        X_c,
        precision=jax.lax.Precision.HIGHEST,
    )

    # 2. Intra-chunk state generation in float32 (Pallas)
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)[..., None]
    states_c = chunk_state_pallas(B_c, decay_states, X_c, interpret=interpret)

    # 3. Inter-chunk recurrence in float32 (JAX Scan)
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

    # 4. State to output conversion (Pallas)
    state_decay_out = jnp.exp(A_cumsum)[..., None]
    Y_off = chunk_scan_pallas(
        C_c, states_incoming, state_decay_out, interpret=interpret
    )

    Y = (Y_diag.astype(jnp.float32) + Y_off.astype(jnp.float32)).astype(X_c.dtype)

    # Extract final states for each sequence using last_chunk_indices
    safe_last_chunk_indices = jnp.clip(last_chunk_indices, 0, n_chunks - 1)
    final_states = states_outgoing[safe_last_chunk_indices]
    return Y, final_states
