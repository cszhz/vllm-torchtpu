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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils

# Scoped VMEM the post kernel reserves: the smallest round budget whose
# _vmem_need estimate still admits the default 64-token block, instead of
# the whole of VMEM, so XLA can keep its own buffers resident across the call.
POST_VMEM_LIMIT_BYTES = 24 * 1024 * 1024

# Tokens per grid step of the post + hc_head kernel, and its scoped VMEM.
POST_HEAD_TOKEN_BLOCK = 128
POST_HEAD_VMEM_LIMIT_BYTES = 32 * 1024 * 1024


def _post_kernel(x_ref, res_ref, post_ref, comb_ref, out_ref, *, hc_mult, hidden_size):
    """One token block of the unrolled stream recombine."""
    x = x_ref[...].astype(jnp.float32)  # (tb, hidden)
    post = post_ref[...]  # (tb, hc_mult) f32
    comb = comb_ref[...]  # (tb, hc_mult * hc_mult) f32, row-major (i, j)

    streams = [
        res_ref[:, i * hidden_size : (i + 1) * hidden_size].astype(jnp.float32)
        for i in range(hc_mult)
    ]
    for j in range(hc_mult):
        acc = post[:, j : j + 1] * x
        for i in range(hc_mult):
            k = i * hc_mult + j
            acc = acc + comb[:, k : k + 1] * streams[i]
        out_ref[:, j * hidden_size : (j + 1) * hidden_size] = acc.astype(out_ref.dtype)


def _post_head_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    fn_ref,
    sc_ref,
    hb_ref,
    out_ref,
    streams_scr,
    fn_stack_ref,
    *,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_eps,
):
    """One token block: the stream recombine, then the hc_head collapse of it.

    The recombined streams stay in VMEM, so the full residual is never
    written to HBM and read back by the head.
    """

    @pl.when(pl.program_id(0) == 0)
    def _():
        utils.stack_fn_chunks_into(fn_ref, fn_stack_ref)

    m, h = hc_mult, hidden_size
    post = post_ref[...]  # (tb, hc_mult) f32
    comb = comb_ref[...]  # (tb, hc_mult * hc_mult) f32, row-major (i, j)
    x_bf = x_ref[...]
    old_bf = [res_ref[:, i * h : (i + 1) * h] for i in range(m)]
    for j in range(m):
        acc = post[:, j : j + 1] * x_bf.astype(jnp.float32)
        for i in range(m):
            k = i * m + j
            acc = acc + comb[:, k : k + 1] * old_bf[i].astype(jnp.float32)
        # The head reads the bf16 residual the standalone post would store.
        streams_scr[:, j * h : (j + 1) * h] = acc.astype(jnp.bfloat16)

    g_bf = streams_scr[...]
    mix = utils.dot_stacked_chunks(g_bf, fn_stack_ref[...], m)
    sqr = utils.sqrsum_on_mxu(g_bf, streams_scr[...])
    # The per-token rsqrt scales the GEMM output rather than its input.
    mix = mix * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    pre_mix = jax.nn.sigmoid(mix * sc_ref[0, 0] + hb_ref[...]) + hc_eps
    out = pre_mix[:, 0:1] * streams_scr[:, :h].astype(jnp.float32)
    for i in range(1, m):
        out = out + pre_mix[:, i : i + 1] * streams_scr[:, i * h : (i + 1) * h].astype(
            jnp.float32
        )
    out_ref[...] = out.astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("token_block_size", "vmem_limit_bytes"))
def mhc_post_2d(
    x: jax.Array,
    res2d: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int = POST_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """2D-native entry: the kernel's own layout as public API.

    The kernel consumes and produces flat ``(num_tokens, hc_mult *
    hidden_size)`` streams. Callers that can keep the residual flat (a
    composed seam, 2D model plumbing) should call this directly.

    Args:
        x: (num_tokens, hidden_size), bfloat16 — the sublayer output.
        res2d: (num_tokens, hc_mult * hidden_size), bfloat16.
        post_layer_mix: (num_tokens, hc_mult, 1) or (num_tokens, hc_mult),
            float32.
        comb_res_mix: (num_tokens, hc_mult, hc_mult), float32.

    Returns:
        (num_tokens, hc_mult * hidden_size) in ``res2d.dtype``.
    """
    assert res2d.dtype == jnp.bfloat16, res2d.dtype
    assert post_layer_mix.dtype == jnp.float32, post_layer_mix.dtype
    assert comb_res_mix.dtype == jnp.float32, comb_res_mix.dtype

    hc_mult = comb_res_mix.shape[-1]
    hidden_size = res2d.shape[-1] // hc_mult

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)
    num_tokens = res2d.shape[0]

    def _vmem_need(tb: int) -> int:
        # Double-buffered bf16 in/out blocks (x + old streams + new
        # streams) plus f32 accumulator transients; x2 spill headroom.
        return 2 * (
            tb * hidden_size * (2 + 2 * hc_mult * 2) * 2 + tb * hidden_size * 4 * 2
        )

    tb, padded_tokens = utils.select_token_block(
        num_tokens,
        token_block_size,
        vmem_need=_vmem_need,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    out = pl.pallas_call(
        functools.partial(_post_kernel, hc_mult=hc_mult, hidden_size=hidden_size),
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tb, hc_mult * hidden_size), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(
            (padded_tokens, hc_mult * hidden_size), res2d.dtype
        ),
        compiler_params=compiler_params,
    )(x2d, res2d, post2d, comb2d)

    (out,) = utils.trim_to(num_tokens, out)
    return out


def mhc_post(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int = POST_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """Pallas replacement for vLLM's ``mhc_post_torch``.

    Thin adapter over ``mhc_post_2d`` matching vLLM's op signature.

    Args:
        x: (..., hidden_size), bfloat16 — the sublayer output.
        residual: (..., hc_mult, hidden_size), bfloat16.
        post_layer_mix: (..., hc_mult, 1), float32.
        comb_res_mix: (..., hc_mult, hc_mult), float32.
        token_block_size: tokens per grid step.

    Returns:
        (..., hc_mult, hidden_size) in ``residual.dtype``.
    """
    assert residual.dtype == jnp.bfloat16, residual.dtype

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]

    out2d = mhc_post_2d(
        x.reshape(-1, hidden_size),
        residual.reshape(-1, hc_mult * hidden_size),
        post_layer_mix.reshape(-1, hc_mult, 1),
        comb_res_mix.reshape(-1, hc_mult, hc_mult),
        token_block_size=token_block_size,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    return out2d.reshape(*outer_shape, hc_mult, hidden_size)


@functools.partial(
    jax.jit,
    static_argnames=("rms_eps", "hc_eps", "token_block_size", "vmem_limit_bytes"),
)
def mhc_post_head_2d(
    x: jax.Array,
    res2d: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_eps: float,
    *,
    token_block_size: int = POST_HEAD_TOKEN_BLOCK,
    vmem_limit_bytes: int = POST_HEAD_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """``mhc_post_2d`` followed by the model's hc_head collapse, one kernel.

    Args:
        x: (num_tokens, hidden_size), bfloat16 — the last sublayer's output.
        res2d: (num_tokens, hc_mult * hidden_size), bfloat16.
        post_layer_mix: (num_tokens, hc_mult[, 1]), float32.
        comb_res_mix: (num_tokens, hc_mult, hc_mult), float32.
        hc_fn: (hc_mult, hc_mult * hidden_size), float32 — head projection.
        hc_scale: (1,), float32.  hc_base: (hc_mult,), float32.

    Returns:
        (num_tokens, hidden_size) in ``res2d.dtype``: the collapsed stream.
    """
    assert res2d.dtype == jnp.bfloat16, res2d.dtype
    assert hc_fn.dtype == jnp.float32, hc_fn.dtype

    hc_mult = comb_res_mix.shape[-1]
    hidden_size = res2d.shape[-1] // hc_mult
    hc_hidden = hc_mult * hidden_size
    assert hc_fn.shape == (hc_mult, hc_hidden), (hc_fn.shape, res2d.shape)

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)
    sc2d = hc_scale.reshape(1, 1).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult).astype(jnp.float32)
    num_tokens = res2d.shape[0]

    tb, padded_tokens = utils.select_token_block(num_tokens, token_block_size)
    # Zero rows give finite gates and zero outputs, and are sliced off below.
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)

    compiler_params = pltpu.CompilerParams(
        # Step 0 fills the stacked-chunk scratch that later steps read.
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    out = pl.pallas_call(
        functools.partial(
            _post_head_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
        ),
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((hc_mult, hc_hidden), lambda i: (0, 0)),
            pl.BlockSpec((1, 1), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, hidden_size), res2d.dtype),
        scratch_shapes=[
            pltpu.VMEM((tb, hc_hidden), jnp.bfloat16),
            pltpu.VMEM((3 * hc_mult, hc_hidden), jnp.bfloat16),
        ],
        compiler_params=compiler_params,
    )(x2d, res2d, post2d, comb2d, hc_fn, sc2d, hb2d)

    (out,) = utils.trim_to(num_tokens, out)
    return out


def mhc_post_head(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_eps: float,
) -> jax.Array:
    """Pallas replacement for ``mhc_post`` then the model's ``mhc_collapse_head``.

    residual (..., hc_mult, hidden_size) bf16 in; (..., hidden_size) bf16 out.
    """
    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    out2d = mhc_post_head_2d(
        x.reshape(-1, hidden_size),
        residual.reshape(-1, hc_mult * hidden_size),
        post_layer_mix.reshape(-1, hc_mult),
        comb_res_mix.reshape(-1, hc_mult, hc_mult),
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
    )
    return out2d.reshape(*outer_shape, hidden_size)
