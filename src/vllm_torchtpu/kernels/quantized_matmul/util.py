# Copyright 2025 Google LLC
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
"""Utility functions for quantized matmul kernel."""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import TunedValue

# Per-row activation quantization takes the one-pass Pallas kernel from this
# many tokens up (prefill); shorter batches keep XLA's reduce + convert pair.
ROW_QUANT_MIN_TOKENS = 512
# bf16 input bytes per grid step of the one-pass kernel, and the fewest grid
# steps it may take, so the input DMA of one tile hides behind another.
ROW_QUANT_TILE_BYTES = 2 * 1024 * 1024
ROW_QUANT_MIN_STEPS = 4
# Rows reduced and quantized together: two fp8 (32, 128) tiles high.
ROW_QUANT_ROW_GROUP = 64
# Columns per amax/quantize step inside a row group.
ROW_QUANT_COL_CHUNK = 512


def unfold_args(
    conditions: tuple[jax.Array | bool, ...],
    fn_conditions: tuple[bool, ...],
    fn: Callable[..., Any],
):
    """Minimize run-time branching of fn by converting jnp.bool to python bool."""
    if conditions:
        arg = conditions[0]
        if isinstance(arg, bool):
            unfold_args(conditions[1:], fn_conditions + (arg,), fn)
        else:
            assert arg.dtype == jnp.bool and arg.size == 1
            jax.lax.cond(
                arg,
                lambda: unfold_args(conditions[1:], fn_conditions + (True,), fn),
                lambda: unfold_args(conditions[1:], fn_conditions + (False,), fn),
            )
    else:
        fn(*fn_conditions)


def quantize_tensor(
    x: jax.Array, dtype: jnp.dtype, dim: int = -1, block_size: int | None = None
):
    if block_size is not None:
        # Flatten all leading dims into a single batch dim for block
        # quantization, then restore the original shape.
        orig_shape = x.shape
        k_dim = orig_shape[-1]
        x_flat = x.reshape(-1, k_dim)
        n_dim = x_flat.shape[0]
        x_reshaped = x_flat.reshape(n_dim, -1, block_size)
        x_q, scale = quantize_block(x_reshaped, axis=-1, target_dtype=dtype)

        x_q = x_q.reshape(orig_shape)

        return x_q, scale.transpose(1, 2, 0).astype(jnp.float32)
    if _row_quant_kernel_applies(x, dim, dtype):
        return quantize_rows(x, dtype)
    data_q, scale = quantize_block(x, axis=dim, target_dtype=dtype)
    return data_q, scale.astype(jnp.float32)


def _row_quant_kernel_applies(x: jax.Array, dim: int, dtype: jnp.dtype) -> bool:
    return (
        x.ndim == 2
        and dim in (-1, 1)
        and x.dtype == jnp.bfloat16
        and jnp.issubdtype(dtype, jnp.floating)
        and x.shape[0] >= ROW_QUANT_MIN_TOKENS
        and x.shape[0] % 128 == 0
        and x.shape[1] % 128 == 0
    )


def _row_quant_tile_m(t: int, k: int) -> int:
    """Largest power-of-two row tile within the byte and step budgets that
    divides t; at least 128, the lane width of the scale's output block."""
    limit = min(ROW_QUANT_TILE_BYTES // (2 * k), t // ROW_QUANT_MIN_STEPS)
    tile_m = 128
    while tile_m * 2 <= limit and t % (tile_m * 2) == 0:
        tile_m *= 2
    return tile_m


def _row_quant_kernel(
    x_ref, x_q_ref, scale_ref, scale_col_ref, *, inv_dtype_max, dtype_max
):
    """quantize_block(axis=1) on one [tile_m, K] block, reading x once.

    Same arithmetic as the XLA lowering of quantize_block: bf16 amax, its f32
    times the rounded 1/dtype_max (XLA rewrites the divide by the constant
    into that multiply), 0 -> 1, f32 divide, clip, convert.
    """
    tile_m, k = x_ref.shape
    chunk = min(ROW_QUANT_COL_CHUNK, k)
    while k % chunk:
        chunk //= 2
    n_chunks = k // chunk

    def cols(c):
        return pl.ds(pl.multiple_of(c * chunk, chunk), chunk)

    def row_group(g, carry):
        rows = pl.ds(
            pl.multiple_of(g * ROW_QUANT_ROW_GROUP, ROW_QUANT_ROW_GROUP),
            ROW_QUANT_ROW_GROUP,
        )
        acc = jax.lax.fori_loop(
            0,
            n_chunks,
            lambda c, m: jnp.maximum(m, jnp.abs(x_ref[rows, cols(c)])),
            jnp.zeros((ROW_QUANT_ROW_GROUP, chunk), x_ref.dtype),
            unroll=True,
        )
        abs_max = jnp.max(acc, axis=1, keepdims=True)
        scale = abs_max.astype(jnp.float32) * inv_dtype_max
        scale = jnp.where(scale == 0, 1.0, scale)
        scale_col_ref[rows, :] = scale

        def quantize_chunk(c, carry):
            data = x_ref[rows, cols(c)].astype(jnp.float32) / scale
            x_q_ref[rows, cols(c)] = jnp.clip(data, -dtype_max, dtype_max).astype(
                x_q_ref.dtype
            )
            return carry

        jax.lax.fori_loop(0, n_chunks, quantize_chunk, 0, unroll=True)
        return carry

    jax.lax.fori_loop(0, tile_m // ROW_QUANT_ROW_GROUP, row_group, 0)
    # Lane-dense, so the [T] scale the matmul epilogue reads is a bitcast.
    scale_ref[...] = jnp.transpose(scale_col_ref[...])


def quantize_rows(x: jax.Array, dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
    """Per-row quantize_tensor of a [T, K] bf16 activation in one pass over x.

    XLA splits quantize_block into an amax reduce fusion and a divide/convert
    fusion that each read x. This kernel reads each [tile_m, K] tile once and
    writes x_q and the [T, 1] f32 scale.
    """
    t, k = x.shape
    dtype_max = float(jnp.finfo(dtype).max)
    kernel = functools.partial(
        _row_quant_kernel,
        inv_dtype_max=float(np.float32(1) / np.float32(dtype_max)),
        dtype_max=dtype_max,
    )
    tile_m = _row_quant_tile_m(t, k)
    x_q, scale = pl.pallas_call(
        kernel,
        grid=(t // tile_m,),
        in_specs=[pl.BlockSpec((tile_m, k), lambda i: (i, 0))],
        out_specs=[
            pl.BlockSpec((tile_m, k), lambda i: (i, 0)),
            pl.BlockSpec((1, tile_m), lambda i: (0, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((t, k), dtype),
            jax.ShapeDtypeStruct((1, t), jnp.float32),
        ],
        scratch_shapes=[pltpu.VMEM((tile_m, 1), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="act_quant_rows",
    )(x)
    return x_q, scale.reshape(t, 1)


def next_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def get_kernel_name(tuned_value: TunedValue):
    batch_block_size = tuned_value.batch_block_size
    out_block_size = tuned_value.out_block_size
    in_block_size = tuned_value.in_block_size
    return (
        f"quantized_matmul_kernel_{batch_block_size}_{out_block_size}_{in_block_size}"
    )


def xla_quantized_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quantize_activation=True,
) -> jax.Array:
    """
    Reference (pure JAX) implementation of the quantized matmul kernel.

    Weights are in the canonical (k, n) layout, so this is a plain
    `(m, k) @ (k, n)` contraction. Callers holding an N-major
    `[n_output_features, n_input_features]` weight (the Pallas kernels and
    their tests) transpose at the call site.

    Args:
        x:  Activation.
        x: Activation. [num_tokens, n_input_features]
        w_q: Weight quantized array. [n_input_features, n_output_features]
        w_scale: Weight quantization scale. [n_output_features]

    Returns:
        Output of the quantized matmul.
    """
    if quantize_activation:
        acc_dtype = jnp.float32
        if quantize_activation and jnp.issubdtype(w_q.dtype, jnp.integer):
            acc_dtype = jnp.int32

        x_q, x_scale = quantize_tensor(x, w_q.dtype)
        out = jax.lax.dot_general(
            x_q,
            w_q,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=acc_dtype,
        ).astype(jnp.float32)
        out *= x_scale
    else:
        out = jax.lax.dot_general(
            x,
            w_q,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
    out *= jnp.expand_dims(w_scale, 0)
    return out.astype(x.dtype)


def xla_quantized_batched_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    dimension_numbers: tuple,
    quantize_activation: bool = True,
) -> jax.Array:
    """Quantized matmul with batch dimensions via dot_general.

    Generalizes xla_quantized_matmul to support batch dimensions (axes
    shared between both operands that appear in the output). Uses
    jax.lax.dot_general with configurable dimension_numbers.

    Args:
        x: Activation tensor (e.g. [T, N, H]).
        w_q: Quantized weight (e.g. [A, N, H]).
        w_scale: Per-output-channel weight scale (e.g. [A]).
        dimension_numbers: ``((contracting_x, contracting_w),
            (batch_x, batch_w))`` for dot_general.
        quantize_activation: Whether to dynamically quantize activations.

    Returns:
        Output of the batched quantized matmul. Shape is determined by
        dot_general: batch dims first, then lhs free dims, then rhs free dims.
    """
    contract_dims, batch_dims = dimension_numbers

    if quantize_activation:
        acc_dtype = jnp.float32
        if jnp.issubdtype(w_q.dtype, jnp.integer):
            acc_dtype = jnp.int32

        x_q, x_scale = quantize_tensor(x, w_q.dtype)
        out = jax.lax.dot_general(
            x_q,
            w_q,
            dimension_numbers=(contract_dims, batch_dims),
            preferred_element_type=acc_dtype,
        ).astype(jnp.float32)
        # Permute x_scale to match dot_general output ordering.
        # dot_general output: batch dims, lhs free dims, rhs free dims.
        # x_scale has same shape as x but with contracting dim(s) as 1.
        contract_set = set(contract_dims[0])
        batch_set = set(batch_dims[0])
        lhs_free = [
            i for i in range(x.ndim) if i not in contract_set and i not in batch_set
        ]
        perm = list(batch_dims[0]) + lhs_free + list(contract_dims[0])
        if perm != list(range(x.ndim)):
            x_scale = jnp.transpose(x_scale, perm)
        out *= x_scale
    else:
        out = jax.lax.dot_general(
            x,
            w_q,
            dimension_numbers=(contract_dims, batch_dims),
            preferred_element_type=jnp.float32,
        )

    # Broadcast w_scale to match the output shape.
    # dot_general output order: batch dims, lhs free dims, rhs free dims.
    # w_scale is per-output-channel (rhs free dims), so expand leading dims.
    n_leading = out.ndim - w_scale.ndim
    for _ in range(n_leading):
        w_scale = jnp.expand_dims(w_scale, 0)
    out *= w_scale
    return out.astype(x.dtype)


def quantize_array(
    x: jax.Array,  # [bs_block_size, in_block_size]
    x_abs_max: jax.Array,  # [1, bs_block_size]
    quant_dtype: jnp.dtype,
):
    is_float = jnp.issubdtype(quant_dtype, jnp.floating)
    dtype_info = jnp.finfo(quant_dtype) if is_float else jnp.iinfo(quant_dtype)
    dtype_max = float(dtype_info.max)

    # TODO(kyuyeunk): Investigate performance gain from non xlu transpose.
    scale = jnp.transpose(x_abs_max / dtype_max)
    scale_inv = jnp.nan_to_num(1 / scale, dtype_max)
    return (x * scale_inv).astype(quant_dtype), scale.astype(jnp.float32)


def get_vmem_limit(
    n_batch: int,
    n_out: int,
    n_in: int,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
    x_dtype: jnp.dtype,
    x_q_dtype: jnp.dtype,
    w_q_dtype: jnp.dtype,
    scale_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    save_acc: bool,
    save_x_q: bool,
    upper_limit_bytes: int,
):
    """Calculate VMEM limit for the kernel."""

    # Calculate in/out VMEM size.
    x_size = batch_block_size * in_block_size * dtypes.itemsize_bits(x_dtype)
    x_abs_max_size = batch_block_size * dtypes.itemsize_bits(scale_dtype)
    w_q_size = out_block_size * in_block_size * dtypes.itemsize_bits(w_q_dtype)
    w_scale_size = out_block_size * dtypes.itemsize_bits(scale_dtype)
    out_size = batch_block_size * out_block_size * dtypes.itemsize_bits(out_dtype)

    vmem_in_out = x_size + x_abs_max_size + w_q_size + w_scale_size + out_size
    vmem_in_out *= 2  # Account for compute and vreg spills.

    # Account for double buffering.
    # Double buffering is used only if there are multiple blocks per in/out.
    vmem_in_out += x_size if (n_batch > 1 or n_in > 1) else 0
    vmem_in_out += x_abs_max_size if (n_batch > 1) else 0
    vmem_in_out += w_q_size if (n_out > 1 or n_in > 1) else 0
    vmem_in_out += w_scale_size if (n_out > 1) else 0
    vmem_in_out += out_size if (n_batch > 1 or n_out > 1) else 0

    # Calculate scratch VMEM size.
    acc_size = batch_block_size * out_block_size * dtypes.itemsize_bits(acc_dtype)
    x_q_size = batch_block_size * in_block_size * dtypes.itemsize_bits(x_q_dtype)
    x_scale_size = batch_block_size * dtypes.itemsize_bits(scale_dtype)

    vmem_scratch = acc_size if save_acc else 0
    vmem_scratch += x_q_size + x_scale_size if save_x_q else 0
    vmem_scratch *= 2  # Account for compute and vreg spills.

    # Add in/out and scratch VMEM size.
    vmem_used = vmem_in_out + vmem_scratch
    vmem_used_bytes = vmem_used // 8  # Convert bits to bytes.
    # Specify upper limit. Defaults to 96MB.
    vmem_limit_bytes = min(vmem_used_bytes, upper_limit_bytes)

    return vmem_limit_bytes


def validate_inputs(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    x_abs_max: jax.Array | None,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
):
    """Verify inputs invoking the kernel."""

    # Verify input shapes.
    if x.shape[1] != w_q.shape[1]:
        raise ValueError(f"{x.shape[1]=} must be equal to {w_q.shape[1]=}")
    if w_q.shape[0] != w_scale.shape[1] and (
        w_scale.ndim == 3 and w_q.shape[0] != w_scale.shape[2]
    ):
        raise ValueError(f"{w_q.shape[0]=} must be equal to {w_scale.shape[1]=}")
    if x_abs_max is not None and x_abs_max.shape != (1, x.shape[0]):
        raise ValueError(f"{x_abs_max.shape=} must be equal to (1, {x.shape[0]=})")
    if x.shape[0] % batch_block_size != 0:
        raise ValueError(f"{x.shape[0]=} must be a multiple of {batch_block_size=}")
    if w_q.shape[0] % out_block_size != 0:
        raise ValueError(f"{w_q.shape[0]=} must be a multiple of {out_block_size=}")
    if x.shape[1] % in_block_size != 0:
        raise ValueError(f"{x.shape[1]=} must be a multiple of {in_block_size=}")


def get_max_min(target_dtype):
    if jnp.issubdtype(target_dtype, jnp.floating):
        return jnp.finfo(target_dtype).max.astype(jnp.float32), jnp.finfo(
            target_dtype
        ).min.astype(jnp.float32)
    else:
        return jnp.iinfo(target_dtype).max, jnp.iinfo(target_dtype).min


def quantize_block(data, axis, target_dtype):
    """Calculates scale and quantizes a block of data."""
    abs_max = jnp.max(
        jnp.abs(data),
        axis=axis,
        keepdims=True,
    )
    dtype_max, dtype_min = get_max_min(target_dtype)
    scale = abs_max / dtype_max
    scale = jnp.where(scale == 0, 1.0, scale)

    if jnp.issubdtype(target_dtype, jnp.floating):
        data_q = (data / scale).clip(dtype_min, dtype_max).astype(target_dtype)
    else:
        data_q = jnp.round(data / scale).astype(target_dtype)
    return data_q, scale
