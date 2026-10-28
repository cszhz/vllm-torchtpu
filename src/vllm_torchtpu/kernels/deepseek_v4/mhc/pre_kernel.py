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

import functools
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils

# Scoped VMEM the pre kernels reserve: the smallest round budget whose
# _vmem_need estimate still admits the default token block, instead of
# the whole of VMEM, so XLA can keep its own buffers resident across the call.
PRE_VMEM_LIMIT_BYTES = 24 * 1024 * 1024

# Tokens per grid step of the chain-start pre kernel. Each step re-pushes the
# resident fn chunks into the MXU, so a larger block amortizes those pushes.
TOKEN_BLOCK = 128

# Tokens per grid step when the pre reads the embedding (one stream wide):
# the fn pushes are 4x cheaper per row, so the block can grow while its
# HBM bytes per step stay below a TOKEN_BLOCK block of full residual rows.
EMBED_TOKEN_BLOCK = 256


def _mixes_kernel(x_ref, fn_ref, mixes_ref, sqrsum_ref):
    """One token block: exact GEMM against the resident stacked fn chunks."""
    x = x_ref[...]
    mixes_ref[...] = utils.dot_stacked_chunks(x, fn_ref[...], mixes_ref.shape[1])
    xf = x.astype(jnp.float32)  # f32 needed only for the squared sum
    sqrsum_ref[...] = jnp.sum(xf * xf, axis=-1, keepdims=True)


def _mixes_collapse_kernel(
    x_ref,
    fn_ref,
    sc_ref,
    hb_ref,
    layer_ref,
    post_out_ref,
    comb_out_ref,
    mix_scr,
    fn_stack_ref,
    *,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_pre_eps,
    gate_blocks,
    gate_kwargs,
):
    """Mixes kernel plus the in-block collapse and the post/comb gates.

    The collapse's weights are ``pre_mix`` — the sigmoid read gates,
    which need no Sinkhorn and are per-token, so they are computable
    entirely in-block. The post/comb gates run once per gate group on
    the staged mixes (``utils.stage_mixes_and_gates``).

    A (tb, hidden_size) ``x_ref`` is a row every stream repeats (the chain
    start's embedding). The kernel then works on that one row: the mix GEMM
    against fn's stream blocks summed (K = hidden_size), the squared sum
    times hc_mult, and the collapse as (sum of pre_mix) * x. That is the same
    math reassociated, so it agrees with the repeated input to f32 rounding.
    """
    m, h = hc_mult, hidden_size
    embed = x_ref.shape[1] == h

    @pl.when(pl.program_id(0) == 0)
    def _():
        utils.stack_fn_chunks_into(fn_ref, fn_stack_ref)

    x = x_ref[...]
    acc = utils.dot_stacked_chunks(x, fn_stack_ref[...], hb_ref.shape[1])
    xf = x.astype(jnp.float32)
    sqr = jnp.sum(xf * xf, axis=-1, keepdims=True)
    if embed:
        sqr = sqr * m
    # Same op order as utils.mhc_pre_gates for bit-level agreement.
    mixes = acc * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    pre_mix = jax.nn.sigmoid(mixes[:, :m] * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
    if embed:
        w = pre_mix[:, 0:1]
        for i in range(1, m):
            w = w + pre_mix[:, i : i + 1]
        lay = w * xf
    else:
        lay = pre_mix[:, 0:1] * xf[:, :h]
        for i in range(1, m):
            lay = lay + pre_mix[:, i : i + 1] * xf[:, i * h : (i + 1) * h]
    layer_ref[...] = lay.astype(jnp.bfloat16)

    utils.stage_mixes_and_gates(
        mixes,
        mix_scr,
        post_out_ref,
        comb_out_ref,
        sc_ref,
        hb_ref,
        gate_blocks=gate_blocks,
        hc_mult=hc_mult,
        **gate_kwargs,
    )


@functools.partial(jax.jit, static_argnames=("token_block_size", "vmem_limit_bytes"))
def mhc_pre_mixes(
    x2d: jax.Array,
    fn: jax.Array,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int = PRE_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array]:
    """Pallas replacement for the mix GEMM + squared sum of mhc_pre.

    Args:
        x2d: (num_tokens, hc_mult * hidden_size), bfloat16.
        fn: (hc_mult3, hc_mult * hidden_size), float32.
        token_block_size: tokens per grid step. The default (64) keeps the
            worst-case VMEM footprint (bf16 block + f32 cast + resident fn)
            around 14 MB at DeepSeek-V4 shapes.

    Returns:
        mixes: (num_tokens, hc_mult3), float32.
        sqrsum: (num_tokens, 1), float32.
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert fn.dtype == jnp.float32, fn.dtype
    num_tokens, hc_hidden = x2d.shape
    hc_mult3 = fn.shape[0]
    assert fn.shape == (hc_mult3, hc_hidden), (fn.shape, x2d.shape)

    def _vmem_need(tb: int) -> int:
        # Double-buffered bf16 x block, transient f32 copy for the squared
        # sum, resident bf16 fn chunks; x2 headroom for spills/scratch
        # (quantized_matmul-style safety factor).
        return 2 * (tb * hc_hidden * (2 * 2 + 4) + 3 * hc_mult3 * hc_hidden * 2)

    tb, padded_tokens = utils.select_token_block(
        num_tokens,
        token_block_size,
        vmem_need=_vmem_need,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    # Zero rows are safe: they produce zero mixes/sqrsum and are sliced
    # off below.
    (x2d,) = utils.pad_to(padded_tokens, x2d)

    fn_stack = utils.stack_fn_chunks(fn)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    resident = pl.BlockSpec(fn_stack.shape, lambda i: (0, 0))
    mixes, sqrsum = pl.pallas_call(
        _mixes_kernel,
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            resident,
        ],
        out_specs=[
            pl.BlockSpec((tb, hc_mult3), lambda i: (i, 0)),
            pl.BlockSpec((tb, 1), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_mult3), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, 1), jnp.float32),
        ],
        compiler_params=compiler_params,
    )(x2d, fn_stack)

    return cast(tuple[jax.Array, jax.Array], utils.trim_to(num_tokens, mixes, sqrsum))


@functools.partial(
    jax.jit,
    static_argnames=(
        "rms_eps",
        "hc_pre_eps",
        "hc_sinkhorn_eps",
        "hc_post_mult_value",
        "sinkhorn_repeat",
        "token_block_size",
        "vmem_limit_bytes",
    ),
)
def mhc_pre_collapse_gates(
    x2d: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = TOKEN_BLOCK,
    vmem_limit_bytes: int = PRE_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mixes + in-block collapse + gates: all of ``mhc_pre`` in one kernel.

    ``x2d`` is (T, hc_mult * hidden_size), or (T, hidden_size) when every
    stream equals that row (the chain start's embedding; see
    ``_mixes_collapse_kernel``).

    Returns (post_mix (T, hc_mult) f32; comb_mix (T, hc_mult * hc_mult)
    f32, row-major (i, j); layer_input2d (T, hidden_size) bf16).
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert fn.dtype == jnp.float32, fn.dtype
    num_tokens, x_width = x2d.shape
    hc_mult3, hc_hidden = fn.shape
    # hc_mult3 = hc_mult * (hc_mult + 2) -> unique positive hc_mult.
    hc_mult = next(m for m in range(1, 64) if m * (m + 2) == hc_mult3)
    hidden_size = hc_hidden // hc_mult
    assert x_width in (hc_hidden, hidden_size), (fn.shape, x2d.shape)

    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)

    def _vmem_need(tb: int) -> int:
        # Double-buffered bf16 x block + layer output, double-buffered f32
        # fn + its stacked bf16 chunks, +25% for Mosaic's own temporaries
        # (deviceless AOT at tb=128: 15.8 MiB need vs 15.25 MiB summed).
        need = 2 * tb * (x_width + hidden_size) * 2
        need += hc_mult3 * (hc_hidden * 2 * 4 + x_width * 3 * 2)
        return need * 5 // 4

    tb, padded_tokens = utils.select_token_block(
        num_tokens,
        token_block_size,
        vmem_need=_vmem_need,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    # Zero rows are safe: they produce finite gates and zero layer rows
    # and are sliced off below.
    (x2d,) = utils.pad_to(padded_tokens, x2d)
    num_blocks = padded_tokens // tb
    gate_blocks = utils.gate_group_blocks(num_blocks, tb)
    group = gate_blocks * tb

    compiler_params = pltpu.CompilerParams(
        # A gate group's output block is revisited by its grid steps, and
        # step 0 fills the stacked-chunk scratch that later steps read.
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    resident = pl.BlockSpec((hc_mult3, hc_hidden), lambda i: (0, 0))
    layer2d, post_mix, comb_mix = pl.pallas_call(
        functools.partial(
            _mixes_collapse_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            gate_blocks=gate_blocks,
            gate_kwargs=dict(
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
        ),
        grid=(num_blocks,),
        in_specs=[
            pl.BlockSpec((tb, x_width), lambda i: (i, 0)),
            resident,
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((group, hc_mult), lambda i: (i // gate_blocks, 0)),
            pl.BlockSpec((group, hc_mult * hc_mult), lambda i: (i // gate_blocks, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
            jax.ShapeDtypeStruct((padded_tokens, hc_mult), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, hc_mult * hc_mult), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((hc_mult3, group), jnp.float32),
            pltpu.VMEM((3 * hc_mult3, x_width), jnp.bfloat16),
        ],
        compiler_params=compiler_params,
    )(x2d, fn, sc2d, hb2d)

    return cast(
        tuple[jax.Array, jax.Array, jax.Array],
        utils.trim_to(num_tokens, post_mix, comb_mix, layer2d),
    )


def mhc_pre(
    residual: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = TOKEN_BLOCK,
    vmem_limit_bytes: int = PRE_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``mhc_pre`` as one Pallas kernel: mix GEMM, collapse and gates.

    Same contract as vLLM's ``mhc_pre_torch``:
    residual (..., hc_mult, hidden_size) bf16 in; post_mix (..., hc_mult, 1)
    f32, comb_mix (..., hc_mult, hc_mult) f32, layer_input (..., hidden_size)
    bf16 out.
    """
    assert residual.dtype == jnp.bfloat16
    assert fn.dtype == jnp.float32

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    x2d = residual.reshape(-1, hc_mult * hidden_size)

    post_mix, comb_mix, layer2d = mhc_pre_collapse_gates(
        x2d,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        token_block_size=token_block_size,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer2d.reshape(*outer_shape, hidden_size),
    )


def mhc_pre_embed(
    embed: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = EMBED_TOKEN_BLOCK,
    vmem_limit_bytes: int = PRE_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``mhc_pre`` of ``embed`` repeated into every stream, on one copy of it.

    embed (..., hidden_size) bf16 in; outputs as ``mhc_pre``, to f32 rounding.
    """
    assert embed.dtype == jnp.bfloat16
    outer_shape = embed.shape[:-1]
    hidden_size = embed.shape[-1]
    hc_mult = fn.shape[1] // hidden_size

    post_mix, comb_mix, layer2d = mhc_pre_collapse_gates(
        embed.reshape(-1, hidden_size),
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        token_block_size=token_block_size,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer2d.reshape(*outer_shape, hidden_size),
    )
