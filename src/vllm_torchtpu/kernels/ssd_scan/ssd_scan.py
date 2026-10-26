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
"""Pallas TPU kernels for intra-chunk state and output steps of Mamba-2 SSD."""

from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_MAX_HEAD_BLOCK_SIZE = 8


def _select_head_block_size(
    n_heads: int, max_head_block: int = _MAX_HEAD_BLOCK_SIZE
) -> int:
    """Selects the largest divisor of `n_heads` up to `max_head_block`."""
    for h in range(min(n_heads, max_head_block), 0, -1):
        if n_heads % h == 0:
            return h
    return 1


def chunk_state_pallas(
    B_c: jax.Array,
    decay_states: jax.Array,
    X_c: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Pallas kernel for Step 2 of SSD scan (intra-chunk state computation).

    Computes in float32:
      states = jnp.einsum("chln,chlp,chl->chpn", B_c, X_c, decay_states[..., 0])

    Several heads are blocked per grid step (`block_shape=(1, H, ...)` with `H`
    dividing `n_heads`) to amortize per-step Pallas launch overhead while
    keeping the chunk axis `parallel` for megacore.

    Args:
      B_c: (n_chunks, n_heads, block_len, d_state)
      decay_states: (n_chunks, n_heads, block_len, 1)
      X_c: (n_chunks, n_heads, block_len, d_head)
      interpret: Run Pallas in host interpret mode (for CPU unit tests only).

    Returns:
      states: (n_chunks, n_heads, d_head, d_state) in float32.
    """
    n_chunks, n_heads, block_len, d_head = X_c.shape
    d_state = B_c.shape[-1]
    head_block = _select_head_block_size(n_heads)

    def kernel(B_ref, decay_ref, X_ref, out_ref):
        B_val = jnp.squeeze(B_ref[...], axis=0).astype(jnp.float32)
        decay_val = jnp.squeeze(decay_ref[...], axis=0).astype(jnp.float32)
        X_val = jnp.squeeze(X_ref[...], axis=0).astype(jnp.float32)

        X_scaled = X_val * decay_val  # (H, block_len, d_head)
        out = jax.lax.dot_general(
            X_scaled,
            B_val,
            (((1,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        out_ref[...] = jnp.expand_dims(out, axis=0)

    grid = (n_chunks, n_heads // head_block)

    in_specs = [
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(1, head_block, block_len, d_state),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(1, head_block, block_len, 1),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(1, head_block, block_len, d_head),
        ),
    ]

    out_spec = pl.BlockSpec(
        memory_space=pltpu.VMEM,
        index_map=lambda c, h_blk: (c, h_blk, 0, 0),
        block_shape=(1, head_block, d_head, d_state),
    )

    out_shape = (n_chunks, n_heads, d_head, d_state)

    return cast(
        jax.Array,
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
            grid=grid,
            in_specs=in_specs,
            out_specs=out_spec,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel"),
                disable_bounds_checks=True,
            ),
            interpret=interpret,
            name="ssd_chunk_state_pallas",
        )(B_c, decay_states, X_c),
    )


def chunk_scan_pallas(
    C_c: jax.Array,
    states: jax.Array,
    decay_states: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Pallas kernel for Step 5 of SSD scan (state-to-output conversion).

    Computes:
      Y_off = jnp.einsum("chln,chpn,chl->chlp", C_c, states, decay_states[..., 0])

    Args:
      C_c: (n_chunks, n_heads, block_len, d_state)
      states: (n_chunks, n_heads, d_head, d_state)
      decay_states: (n_chunks, n_heads, block_len, 1)
      interpret: Run Pallas in host interpret mode (for CPU unit tests only).

    Returns:
      Y_off: (n_chunks, n_heads, block_len, d_head) in `C_c.dtype`.
    """
    n_chunks, n_heads, block_len, d_state = C_c.shape
    d_head = states.shape[-2]
    head_block = _select_head_block_size(n_heads)

    def kernel(C_ref, states_ref, decay_ref, out_ref):
        C_val = jnp.squeeze(C_ref[...], axis=0).astype(jnp.float32)
        states_val = jnp.squeeze(states_ref[...], axis=0).astype(jnp.float32)
        decay_val = jnp.squeeze(decay_ref[...], axis=0).astype(jnp.float32)

        prod = jax.lax.dot_general(
            C_val,
            states_val,
            (((2,), (2,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        out = (prod * decay_val).astype(C_c.dtype)
        out_ref[...] = jnp.expand_dims(out, axis=0)

    grid = (n_chunks, n_heads // head_block)

    in_specs = [
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(1, head_block, block_len, d_state),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(1, head_block, d_head, d_state),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(1, head_block, block_len, 1),
        ),
    ]

    out_spec = pl.BlockSpec(
        memory_space=pltpu.VMEM,
        index_map=lambda c, h_blk: (c, h_blk, 0, 0),
        block_shape=(1, head_block, block_len, d_head),
    )

    out_shape = (n_chunks, n_heads, block_len, d_head)

    return cast(
        jax.Array,
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(out_shape, C_c.dtype),
            grid=grid,
            in_specs=in_specs,
            out_specs=out_spec,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel"),
                disable_bounds_checks=True,
            ),
            interpret=interpret,
            name="ssd_chunk_scan_pallas",
        )(C_c, states, decay_states),
    )
