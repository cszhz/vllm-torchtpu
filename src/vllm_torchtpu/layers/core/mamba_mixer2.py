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
"""JAX/Pallas implementation of Mamba-2 Mixer core for TPU."""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.causal_conv1d import causal_conv1d
from vllm_torchtpu.layers.core.ssd_scan import ssd_minimal_discrete_hybrid
from vllm_torchtpu.utils import get_mesh_shape_product

_HEAD_SHARDING_AXIS = "model"


def broadcast_groups_to_heads(x: jnp.ndarray, n_heads: int) -> jnp.ndarray:
    """Broadcasts group dimension to head dimension.

    x: (..., n_groups, dim) -> (..., n_heads, dim)
    """
    n_groups = x.shape[-2]
    if n_groups == n_heads:
        return x
    group_size = n_heads // n_groups
    return jnp.repeat(x, group_size, axis=-2)


def jax_gated_rms_norm(
    x: jnp.ndarray,
    gate: jnp.ndarray,
    weight: jnp.ndarray | None,
    eps: float,
    shard_groups: bool,
    tp_size: int,
    n_groups: int = 1,
    head_axis_name: str | None = _HEAD_SHARDING_AXIS,
) -> jnp.ndarray:
    input_dtype = x.dtype

    # 1. Apply Gate (SiLU)
    x_gated = x.astype(jnp.float32) * jax.nn.silu(gate.astype(jnp.float32))
    if weight is None:
        return x_gated.astype(input_dtype)

    # 2. Compute Variance and Normalize
    if shard_groups or tp_size <= 1 or head_axis_name is None:
        orig_shape = x_gated.shape
        if n_groups > 1:
            x_gated = x_gated.reshape(
                *orig_shape[:-1], n_groups, orig_shape[-1] // n_groups
            )
        variance = jnp.mean(jnp.square(x_gated), axis=-1, keepdims=True)
        x_norm = (x_gated * jax.lax.rsqrt(variance + eps)).reshape(orig_shape)
    else:
        local_sum = jnp.sum(jnp.square(x_gated), axis=-1, keepdims=True)
        global_sum = jax.lax.psum(local_sum, axis_name=head_axis_name)
        count = tp_size * x.shape[-1]
        variance = global_sum / count
        x_norm = x_gated * jax.lax.rsqrt(variance + eps)

    # 3. Apply weight
    return (weight.astype(jnp.float32) * x_norm).astype(input_dtype)


def pack_inputs_mamba(
    X: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    decay: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    chunk_size: int,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Pads each sequence to a multiple of chunk_size and packs into a stream."""
    num_tokens = X.shape[0]
    num_seqs = len(query_start_loc) - 1

    num_valid_seqs = distribution[2]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    last_valid_loc = query_start_loc[num_valid_seqs]
    effective_query_start_loc = jnp.where(
        valid_loc_mask, query_start_loc, last_valid_loc
    )

    # Calculate sequence lengths and pad them to multiples of chunk_size.
    seq_lengths = effective_query_start_loc[1:] - effective_query_start_loc[:-1]
    num_chunks = (seq_lengths + chunk_size - 1) // chunk_size
    padded_lengths = num_chunks * chunk_size

    new_query_start_loc = jnp.cumsum(
        jnp.concatenate([jnp.array([0], dtype=jnp.int32), padded_lengths])
    )

    # Map each original token index to its sequence ID
    token_indices = jnp.arange(num_tokens)
    valid_token_mask = token_indices < last_valid_loc
    seq_id = (
        jnp.searchsorted(effective_query_start_loc, token_indices, side="right") - 1
    )
    seq_id = jnp.clip(seq_id, 0, num_seqs - 1)
    original_start = effective_query_start_loc[seq_id]
    new_start = new_query_start_loc[seq_id]

    max_packed_tokens = num_tokens + num_seqs * chunk_size
    max_packed_tokens = (max_packed_tokens + chunk_size - 1) // chunk_size * chunk_size

    padded_indices_valid = jnp.clip(
        new_start + (token_indices - original_start), 0, max_packed_tokens - 1
    )
    scatter_token_indices = jnp.where(
        valid_token_mask, padded_indices_valid, max_packed_tokens
    )

    # Flatten last dimensions to concatenate for single scatter
    n_heads = X.shape[1]
    d_head = X.shape[2]
    d_state = B.shape[2]

    X_flat = X.reshape(num_tokens, n_heads * d_head)
    B_flat = B.reshape(num_tokens, n_heads * d_state)
    C_flat = C.reshape(num_tokens, n_heads * d_state)

    combined = jnp.concatenate(
        [
            X_flat.astype(compute_dtype),
            B_flat.astype(compute_dtype),
            C_flat.astype(compute_dtype),
        ],
        axis=-1,
    )

    output_shape = (max_packed_tokens,) + combined.shape[1:]
    packed_combined = jnp.zeros(output_shape, dtype=compute_dtype)
    packed_combined = packed_combined.at[scatter_token_indices].set(
        combined, mode="drop"
    )

    # Split back
    dim_X = n_heads * d_head
    dim_B = n_heads * d_state

    packed_X_flat = packed_combined[..., :dim_X]
    packed_B_flat = packed_combined[..., dim_X : dim_X + dim_B]
    packed_C_flat = packed_combined[..., dim_X + dim_B :]

    num_chunks_total = max_packed_tokens // chunk_size

    X_c = packed_X_flat.reshape(
        num_chunks_total, chunk_size, n_heads, d_head
    ).transpose(0, 2, 1, 3)
    B_c = packed_B_flat.reshape(
        num_chunks_total, chunk_size, n_heads, d_state
    ).transpose(0, 2, 1, 3)
    C_c = packed_C_flat.reshape(
        num_chunks_total, chunk_size, n_heads, d_state
    ).transpose(0, 2, 1, 3)

    # For decay (float32, shape (num_tokens, n_heads))
    output_shape_f32 = (max_packed_tokens, n_heads)
    packed_decay = jnp.zeros(output_shape_f32, dtype=jnp.float32)
    packed_decay = packed_decay.at[scatter_token_indices].set(
        decay.astype(jnp.float32), mode="drop"
    )

    decay_c = packed_decay.reshape(num_chunks_total, chunk_size, n_heads).transpose(
        0, 2, 1
    )

    active_nonempty_seq = (jnp.arange(num_seqs) < num_valid_seqs) & (seq_lengths > 0)
    start_chunk_indices = jnp.where(
        active_nonempty_seq,
        new_query_start_loc[:-1] // chunk_size,
        num_chunks_total,
    )
    reset_mask = (
        jnp.zeros((num_chunks_total,), dtype=bool)
        .at[start_chunk_indices]
        .set(True, mode="drop")
    )

    last_chunk_indices = jnp.clip(
        (new_query_start_loc[1:] // chunk_size) - 1, 0, num_chunks_total - 1
    )

    return (
        X_c,
        B_c,
        C_c,
        decay_c,
        reset_mask,
        last_chunk_indices,
        start_chunk_indices,
        padded_indices_valid,
    )


def run_jax_mamba_mixer_local(
    j_projected_states: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: jnp.ndarray | None,
    j_A: jnp.ndarray,
    j_D: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    j_norm_weight: jnp.ndarray | None,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    *,
    tped_intermediate_size: int,
    tped_conv_size: int,
    tped_dt_size: int,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    kernel_size: int,
    chunk_size: int,
    shard_groups: bool,
    tp_size: int,
    rms_norm_eps: float,
    apply_rms_norm: bool = True,
    has_silu_activation: bool = True,
    head_axis_name: str | None = _HEAD_SHARDING_AXIS,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the local device execution of Mamba-2 Mixer."""
    if shard_groups or tp_size <= 1 or head_axis_name is None:
        tped_intermediate_size_local = tped_intermediate_size
        tped_conv_size_local = tped_conv_size

        hidden_states_B_C = j_projected_states[
            ...,
            tped_intermediate_size_local : tped_intermediate_size_local
            + tped_conv_size_local,
        ]
        dt_raw = j_projected_states[
            ..., tped_intermediate_size_local + tped_conv_size_local :
        ]

        conv_state_in = conv_state
        j_conv_weight_local = j_conv_weight
        j_conv_bias_local = j_conv_bias
        X_MLP_local = j_projected_states[..., :tped_intermediate_size_local]
    else:
        rank = jax.lax.axis_index(head_axis_name)
        local_intermediate = tped_intermediate_size
        local_dt = tped_dt_size

        global_intermediate = local_intermediate * tp_size
        global_groups_ssm = n_groups * d_state

        X_MLP_local = jax.lax.dynamic_slice_in_dim(
            j_projected_states,
            rank * local_intermediate,
            local_intermediate,
            axis=-1,
        )

        X_conv_start = global_intermediate
        X_conv_local = jax.lax.dynamic_slice_in_dim(
            j_projected_states,
            X_conv_start + rank * local_intermediate,
            local_intermediate,
            axis=-1,
        )

        B_start = X_conv_start + global_intermediate
        B_replicated = j_projected_states[..., B_start : B_start + global_groups_ssm]
        C_start = B_start + global_groups_ssm
        C_replicated = j_projected_states[..., C_start : C_start + global_groups_ssm]

        dt_start = C_start + global_groups_ssm
        dt_local = jax.lax.dynamic_slice_in_dim(
            j_projected_states, dt_start + rank * local_dt, local_dt, axis=-1
        )

        hidden_states_B_C = jnp.concatenate(
            [X_conv_local, B_replicated, C_replicated], axis=-1
        )
        dt_raw = dt_local

        X_conv_cache_local = jax.lax.dynamic_slice_in_dim(
            conv_state,
            rank * local_intermediate,
            local_intermediate,
            axis=-1,
        )
        B_cache_replicated = conv_state[
            ..., global_intermediate : global_intermediate + global_groups_ssm
        ]
        C_cache_replicated = conv_state[
            ...,
            global_intermediate + global_groups_ssm : global_intermediate
            + 2 * global_groups_ssm,
        ]

        conv_state_in = jnp.concatenate(
            [X_conv_cache_local, B_cache_replicated, C_cache_replicated],
            axis=-1,
        )

        X_conv_w_local = jax.lax.dynamic_slice_in_dim(
            j_conv_weight,
            rank * local_intermediate,
            local_intermediate,
            axis=0,
        )
        B_w_replicated = j_conv_weight[
            global_intermediate : global_intermediate + global_groups_ssm, ...
        ]
        C_w_replicated = j_conv_weight[
            global_intermediate + global_groups_ssm : global_intermediate
            + 2 * global_groups_ssm,
            ...,
        ]

        j_conv_weight_local = jnp.concatenate(
            [X_conv_w_local, B_w_replicated, C_w_replicated], axis=0
        )

        if j_conv_bias is not None:
            X_conv_b_local = jax.lax.dynamic_slice_in_dim(
                j_conv_bias,
                rank * local_intermediate,
                local_intermediate,
                axis=0,
            )
            B_b_replicated = j_conv_bias[
                global_intermediate : global_intermediate + global_groups_ssm
            ]
            C_b_replicated = j_conv_bias[
                global_intermediate + global_groups_ssm : global_intermediate
                + 2 * global_groups_ssm
            ]
            j_conv_bias_local = jnp.concatenate(
                [X_conv_b_local, B_b_replicated, C_b_replicated], axis=0
            )
        else:
            j_conv_bias_local = None

        tped_intermediate_size_local = local_intermediate
        tped_conv_size_local = local_intermediate + 2 * global_groups_ssm

    max_reqs = seq_lens.shape[0]
    query_lens = query_start_loc[1 : max_reqs + 1] - query_start_loc[:max_reqs]
    has_initial_state = (seq_lens - query_lens) > 0

    if j_conv_weight_local.ndim == 2:
        j_conv_weight_local = jnp.expand_dims(j_conv_weight_local, axis=1)

    out_conv, new_conv_state = causal_conv1d.ragged_causal_conv1d(
        hidden_states_B_C,
        conv_state_in,
        j_conv_weight_local,
        j_conv_bias_local,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )
    if has_silu_activation:
        out_conv = jax.nn.silu(out_conv)

    local_groups_ssm = n_groups * d_state

    X_raw, B_raw, C_raw = jnp.split(
        out_conv,
        [
            tped_intermediate_size_local,
            tped_intermediate_size_local + local_groups_ssm,
        ],
        axis=-1,
    )

    X = X_raw.reshape(-1, num_heads, head_dim)
    B_raw = B_raw.reshape(-1, n_groups, d_state)
    C_raw = C_raw.reshape(-1, n_groups, d_state)

    B = broadcast_groups_to_heads(B_raw, num_heads)
    C = broadcast_groups_to_heads(C_raw, num_heads)

    dt = jax.nn.softplus(dt_raw.astype(jnp.float32) + j_dt_bias.astype(jnp.float32))
    decay = dt * j_A.astype(jnp.float32)

    (
        X_c,
        B_c,
        C_c,
        decay_c,
        reset_mask,
        last_chunk_indices,
        start_chunk_indices,
        padded_indices_valid,
    ) = pack_inputs_mamba(
        X,
        B,
        C,
        decay,
        query_start_loc,
        distribution,
        chunk_size,
        compute_dtype=X.dtype,
    )

    init_states_for_seqs = recurrent_state[state_indices]

    num_chunks_total = X_c.shape[0]
    init_h_per_chunk = jnp.zeros(
        (num_chunks_total, num_heads, head_dim, d_state),
        dtype=recurrent_state.dtype,
    )

    init_states_for_seqs = jnp.where(
        has_initial_state[:, None, None, None],
        init_states_for_seqs,
        0.0,
    )
    init_h_per_chunk = init_h_per_chunk.at[start_chunk_indices].set(
        init_states_for_seqs, mode="drop"
    )

    Y_c, new_recurrent_states = ssd_minimal_discrete_hybrid(
        X_c,
        decay_c,
        B_c,
        C_c,
        last_chunk_indices,
        init_h_per_chunk,
        reset_mask,
    )

    Y_flat = Y_c.transpose(0, 2, 1, 3).reshape(-1, num_heads * head_dim)
    Y_final = (
        Y_flat[padded_indices_valid].astype(jnp.float32)
        + X.reshape(-1, num_heads * head_dim).astype(jnp.float32)
        * jnp.repeat(j_D.astype(jnp.float32), head_dim)[None, :]
    ).astype(j_projected_states.dtype)

    num_seqs = last_chunk_indices.shape[0]
    valid_seq_mask = (jnp.arange(num_seqs) < distribution[2]) & (query_lens > 0)
    scatter_state_indices = jnp.where(
        valid_seq_mask, state_indices, recurrent_state.shape[0]
    )
    updated_recurrent_state = recurrent_state.at[scatter_state_indices].set(
        new_recurrent_states.astype(recurrent_state.dtype), mode="drop"
    )

    if apply_rms_norm:
        Y_out = jax_gated_rms_norm(
            x=Y_final,
            gate=X_MLP_local,
            weight=j_norm_weight,
            eps=rms_norm_eps,
            shard_groups=shard_groups,
            tp_size=tp_size,
            n_groups=n_groups,
            head_axis_name=head_axis_name,
        )
    else:
        Y_out = Y_final.astype(j_projected_states.dtype)

    if not shard_groups and tp_size > 1 and head_axis_name is not None:
        global_groups_ssm = n_groups * d_state
        new_X_conv_local = new_conv_state[..., :tped_intermediate_size_local]
        new_B = new_conv_state[
            ...,
            tped_intermediate_size_local : tped_intermediate_size_local
            + global_groups_ssm,
        ]
        new_C = new_conv_state[..., tped_intermediate_size_local + global_groups_ssm :]

        new_X_conv_global = jax.lax.all_gather(
            new_X_conv_local, axis_name=head_axis_name, axis=2
        )
        new_X_conv_global = new_X_conv_global.reshape(
            new_X_conv_global.shape[0], new_X_conv_global.shape[1], -1
        )
        new_conv_state_out = jnp.concatenate([new_X_conv_global, new_B, new_C], axis=2)
    else:
        new_conv_state_out = new_conv_state

    return (new_conv_state_out, updated_recurrent_state), Y_out


def run_jax_mamba_mixer(
    j_projected_states: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: jnp.ndarray | None,
    j_A: jnp.ndarray,
    j_D: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    j_norm_weight: jnp.ndarray | None,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    *,
    mesh: jax.sharding.Mesh,
    tped_intermediate_size: int,
    tped_conv_size: int,
    tped_dt_size: int,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    kernel_size: int,
    chunk_size: int = 64,
    shard_groups: bool = True,
    rms_norm_eps: float = 1e-5,
    apply_rms_norm: bool = True,
    has_silu_activation: bool = True,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the sharded JAX Mamba-2 Mixer over the given TPU mesh."""
    if j_conv_weight.ndim == 2:
        j_conv_weight = jnp.expand_dims(j_conv_weight, axis=1)

    data_axis = "attn_dp" if "attn_dp" in mesh.axis_names else None
    head_axis = _HEAD_SHARDING_AXIS if _HEAD_SHARDING_AXIS in mesh.axis_names else None
    tp_size = get_mesh_shape_product(mesh, head_axis) if head_axis else 1

    if shard_groups:
        projected_states_spec = P(data_axis, head_axis)
        conv_state_spec = P(data_axis, None, head_axis)
        conv_weight_spec = P(head_axis, None, None)
        conv_bias_spec = P(head_axis) if j_conv_bias is not None else None
        new_conv_state_spec = P(data_axis, None, head_axis)
    else:
        projected_states_spec = P(data_axis, None)
        conv_state_spec = P(data_axis, None, None)
        conv_weight_spec = P(None, None, None)
        conv_bias_spec = None
        new_conv_state_spec = P(data_axis, None, None)

    norm_weight_spec = P(head_axis) if j_norm_weight is not None else None

    in_specs = (
        projected_states_spec,
        conv_state_spec,
        P(data_axis, head_axis, None, None),  # recurrent_state
        conv_weight_spec,
        conv_bias_spec,
        P(head_axis),  # j_A
        P(head_axis),  # j_D
        P(head_axis),  # j_dt_bias
        norm_weight_spec,  # j_norm_weight
        P(data_axis),  # state_indices
        P(data_axis),  # query_start_loc
        P(),  # distribution
        P(data_axis),  # seq_lens
    )

    out_specs = (
        (
            new_conv_state_spec,
            P(data_axis, head_axis, None, None),  # updated_recurrent_state
        ),
        P(data_axis, head_axis),  # Y_out
    )

    local_fn = functools.partial(
        run_jax_mamba_mixer_local,
        tped_intermediate_size=tped_intermediate_size,
        tped_conv_size=tped_conv_size,
        tped_dt_size=tped_dt_size,
        num_heads=num_heads,
        head_dim=head_dim,
        n_groups=n_groups,
        d_state=d_state,
        kernel_size=kernel_size,
        chunk_size=chunk_size,
        shard_groups=shard_groups,
        tp_size=tp_size,
        rms_norm_eps=rms_norm_eps,
        apply_rms_norm=apply_rms_norm,
        has_silu_activation=has_silu_activation,
        head_axis_name=head_axis,
    )

    mapped_fn = jax.shard_map(
        local_fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    return mapped_fn(
        j_projected_states,
        conv_state,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A,
        j_D,
        j_dt_bias,
        j_norm_weight,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
    )
