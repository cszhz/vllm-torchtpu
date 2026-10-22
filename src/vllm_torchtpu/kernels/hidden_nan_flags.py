# SPDX-License-Identifier: Apache-2.0
"""Capture the first nonfinite layer without retaining prior activations."""

import jax
import jax.numpy as jnp


def record_hidden_nan(
    state: jax.Array,
    hidden_snapshot: jax.Array,
    token_snapshot: jax.Array,
    armed: jax.Array,
    hidden: jax.Array,
    input_ids: jax.Array,
    input_rows: jax.Array,
    layer_slot: jax.Array,
    round_state: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """State: [slot, hit round, hidden rows, token rows]; rounds: [now, hit]."""
    assert state.shape == (4,) and armed.shape == (1,), (state.shape, armed.shape)
    assert round_state.shape == (2,), round_state.shape
    assert hidden.ndim == 2 and hidden.size > 0, (
        f"Layer NaN/Inf detection requires nonempty 2D hidden: {hidden.shape}"
    )
    assert (
        hidden_snapshot.ndim == 1
        and hidden.size <= hidden_snapshot.size
        and hidden.dtype == hidden_snapshot.dtype
    ), (
        "Hidden snapshot capacity/dtype mismatch: "
        f"source={hidden.shape}/{hidden.dtype}, "
        f"snapshot={hidden_snapshot.shape}/{hidden_snapshot.dtype}"
    )
    assert input_ids.ndim == 1 and input_ids.size <= token_snapshot.size, (
        f"Input snapshot capacity mismatch: {input_ids.shape}, {token_snapshot.shape}"
    )

    def inspect(buffers):
        def capture(current):
            metadata, saved_hidden, saved_tokens, rounds = current
            # Only the hit branch flattens the activation. A 1D snapshot cannot
            # inherit the model's changing 2D tiling in passthrough branches.
            saved_hidden = jax.lax.dynamic_update_slice(
                saved_hidden, hidden.reshape(-1), (0,)
            )
            saved_tokens = jax.lax.dynamic_update_slice(
                saved_tokens, input_ids.astype(saved_tokens.dtype), (0,)
            )
            metadata = jnp.stack(
                (layer_slot, rounds[0], hidden.shape[0], input_rows[0])
            ).astype(metadata.dtype)
            return metadata, saved_hidden, saved_tokens, rounds.at[1].set(rounds[0])

        return jax.lax.cond(
            (~jnp.isfinite(hidden)).any(), capture, lambda x: x, buffers
        )

    result = jax.lax.cond(
        armed[0]
        & (state[0] < 0)
        & ((round_state[1] == 0) | (round_state[0] == round_state[1])),
        inspect,
        lambda buffers: buffers,
        (state, hidden_snapshot, token_snapshot, round_state),
    )
    # The next decoder layer must consume this value, not the original hidden.
    # Otherwise XLA can postpone every check until all decoder layers finish.
    return jax.lax.optimization_barrier((*result, hidden))
