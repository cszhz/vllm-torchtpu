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

# Scoped VMEM the seam kernel reserves. XLA keeps the rest of VMEM free for
# its own buffers across the call, so this sits just above the kernel's
# need at TOKEN_BLOCK=128 (AOT: ~30 MiB at T=1024, ~32 MiB at T=8192; the
# double-buffered res/new_res blocks alone are 16 MiB) rather than at the
# whole of VMEM.
FUSED_VMEM_LIMIT_BYTES = 35651584  # 34 MiB

# Tokens per grid step. Each step re-pushes the resident fn chunks into the
# MXU, so a larger block amortizes those pushes over more rows.
TOKEN_BLOCK = 128


def _fused_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    fn_ref,
    sc_ref,
    hb_ref,
    newres_ref,
    layer_ref,
    post_out_ref,
    comb_out_ref,
    mix_scr,
    fn_stack_ref,
    weights_scr,
    *dma_scratch,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_pre_eps,
    num_blocks,
    gate_blocks,
    gate_kwargs,
):
    """One token block: recombine, store once, stacked-chunk GEMM, collapse, gates.

    With ``dma_scratch`` (res_buf, new_buf, in_sem, out_sem) the residual
    streams are HBM refs moved by hand through two VMEM slots: block i+1 is
    fetched during block i and block i is written back during block i+1.
    Without it, res_ref/newres_ref are the pipelined (tb, hc_hidden) blocks.

    A res_ref only hidden_size wide is a row every old stream repeats (the
    chain start's embedding e): stream j recombines as
    post_j * x + (sum_k comb_kj) * e, the same math reassociated.
    """
    i = pl.program_id(0)
    tb = x_ref.shape[0]
    h = hidden_size
    last = num_blocks - 1
    embed = res_ref.shape[1] == h

    if dma_scratch:
        res_buf, new_buf, in_sem, out_sem = dma_scratch
        slot = i % 2
        res_blk, new_blk = res_buf.at[slot], new_buf.at[slot]
    else:
        res_blk, new_blk = res_ref, newres_ref

    def res_copy(block):
        rows = pl.ds(pl.multiple_of(block * tb, utils.SUBLANE), tb)
        return pltpu.make_async_copy(
            res_ref.at[rows], res_buf.at[block % 2], in_sem.at[block % 2]
        )

    def new_copy(block):
        rows = pl.ds(pl.multiple_of(block * tb, utils.SUBLANE), tb)
        return pltpu.make_async_copy(
            new_buf.at[block % 2], newres_ref.at[rows], out_sem.at[block % 2]
        )

    def recombine(r):
        # Keep the streams (and x) in bf16 and upcast per use: transient f32
        # values instead of 4 held f32 copies halves the resident working
        # set. bf16 -> f32 casts are exact, so results are unchanged.
        post = post_ref[r, :]  # (rows, hc_mult) f32
        comb = comb_ref[r, :]  # (rows, hc_mult * hc_mult) f32, row-major (i, j)
        x_bf = x_ref[r, :]  # (rows, hidden) bf16
        if embed:
            e_bf = res_blk[r, :]
        else:
            old_bf = [res_blk[r, k * h : (k + 1) * h] for k in range(hc_mult)]
        streams_bf = []
        for j in range(hc_mult):
            acc = post[:, j : j + 1] * x_bf.astype(jnp.float32)
            if embed:
                csum = comb[:, j : j + 1]
                for k in range(1, hc_mult):
                    csum = csum + comb[:, k * hc_mult + j : k * hc_mult + j + 1]
                acc = acc + csum * e_bf.astype(jnp.float32)
            else:
                for k in range(hc_mult):
                    kj = k * hc_mult + j
                    acc = acc + comb[:, kj : kj + 1] * old_bf[k].astype(jnp.float32)
            # Round to bf16 BEFORE the GEMM: the unfused path's pre reads the
            # bf16 residual that post stored, and parity requires matching it.
            new_bf = acc.astype(jnp.bfloat16)
            new_blk[r, j * h : (j + 1) * h] = new_bf
            streams_bf.append(new_bf)
        return jnp.concatenate(streams_bf, axis=1)

    @pl.when(i == 0)
    def _():
        if dma_scratch:
            res_copy(0).start()
        utils.stack_fn_chunks_into(fn_ref, fn_stack_ref)

    if dma_scratch:
        res_copy(i).wait()

        # Block i+1 is fetched once block i has landed, so at step 0 the two
        # do not share the bandwidth block 0 waits on.
        @pl.when(i < last)
        def _():
            res_copy(i + 1).start()

    g_bf = recombine(slice(0, tb))
    acc = utils.dot_stacked_chunks(g_bf, fn_stack_ref[...], hb_ref.shape[1])
    # The squared sum rides on the otherwise idle MXU instead of the VPU.
    sqr = utils.sqrsum_on_mxu(g_bf, new_blk[...])

    m = hc_mult
    mixes = acc * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    pre_mix = jax.nn.sigmoid(mixes[:, :m] * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
    # The collapse re-reads the stored streams: holding the packed GEMM
    # operand live past both dots would spill ~3 MiB more scoped VMEM.
    weights_scr[...] = utils.replicate_lanes(pre_mix)
    if dma_scratch:
        # The collapse and the gates only read the slot, so block i's
        # write-back runs under them, the last block's included.
        new_copy(i).start()
    utils.collapse_rows(new_blk, weights_scr, layer_ref, hc_mult=m)

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

    if not dma_scratch:
        return

    # The previous block's write-back had this whole step to finish; the
    # last step also drains its own before the kernel ends.
    @pl.when(i > 0)
    def _():
        new_copy(i - 1).wait()

    @pl.when(i == last)
    def _():
        new_copy(i).wait()


@functools.partial(
    jax.jit,
    static_argnames=(
        "rms_eps",
        "hc_pre_eps",
        "hc_sinkhorn_eps",
        "hc_post_mult_value",
        "sinkhorn_repeat",
        "token_block_size",
        "gemm_precision",
        "vmem_limit_bytes",
    ),
)
def fused_post_pre_2d(
    x2d: jax.Array,
    res2d: jax.Array,
    post2d: jax.Array,
    comb2d: jax.Array,
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
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = FUSED_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The Pallas seam kernel: post recombine fused with pre's mix GEMM,
    in-block collapse and the post/comb gates.

    Args:
        x2d: (num_tokens, hidden_size), bfloat16 — sublayer output.
        res2d: (num_tokens, hc_mult * hidden_size), bfloat16 — old streams;
            or (num_tokens, hidden_size) when every old stream is that row.
        post2d: (num_tokens, hc_mult), float32.
        comb2d: (num_tokens, hc_mult * hc_mult), float32, row-major (i, j).
        fn: (hc_mult3, hc_mult * hidden_size), float32 — next sublayer's
            gate projection.
        hc_scale: (3,), float32 — pre/post/comb logit scales.
        hc_base: (hc_mult3,), float32 — logit biases, pre entries first.
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
            sinkhorn_repeat: static gate constants, as in
            ``utils.mhc_pre_gates``.
        token_block_size: tokens per grid step (default ``TOKEN_BLOCK``).
        gemm_precision: "highest" (exact hi+mid+lo decomposition, one
            pass against the stacked chunks; the default) or "default"
            (hi chunk only; the gates and the collapse then also see the
            hi-only mixes).

    Returns:
        new_res2d: (num_tokens, hc_mult * hidden_size), bfloat16.
        post_mix: (num_tokens, hc_mult), float32 — the next post's gates.
        comb_mix: (num_tokens, hc_mult * hc_mult), float32, row-major (i, j).
        layer2d: (num_tokens, hidden_size), bfloat16.
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert res2d.dtype == jnp.bfloat16, res2d.dtype
    assert post2d.dtype == jnp.float32, post2d.dtype
    assert comb2d.dtype == jnp.float32, comb2d.dtype
    assert fn.dtype == jnp.float32, fn.dtype

    num_tokens, hidden_size = x2d.shape
    hc_mult3, hc_hidden = fn.shape
    hc_mult = hc_hidden // hidden_size
    res_width = res2d.shape[1]
    assert res2d.shape[0] == num_tokens
    assert res_width in (hc_hidden, hidden_size), (fn.shape, res2d.shape)
    assert gemm_precision in ("highest", "default"), gemm_precision

    # This kernel keeps a fixed block: its default already fits the
    # budget at DeepSeek-V4 shapes, so there is no vmem_need to shrink by.
    tb, padded_tokens = utils.select_token_block(num_tokens, token_block_size)
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)
    num_blocks = padded_tokens // tb
    gate_blocks = utils.gate_group_blocks(num_blocks, tb)
    group = gate_blocks * tb

    n_chunks = 3 if gemm_precision == "highest" else 1
    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)

    compiler_params = pltpu.CompilerParams(
        # A gate group's output block is revisited by its grid steps, step 0
        # fills the stacked-chunk scratch that later steps read, and each
        # step's residual DMAs span its neighbours.
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    resident = pl.BlockSpec((hc_mult3, hc_hidden), lambda i: (0, 0))
    scratch_shapes = [
        pltpu.VMEM((hc_mult3, group), jnp.float32),
        pltpu.VMEM((n_chunks * hc_mult3, hc_hidden), jnp.bfloat16),
        pltpu.VMEM((tb, hc_mult * 128), jnp.float32),
    ]
    # A one-block grid has no neighbour to overlap with (decode, where XLA
    # also tends to keep the streams in VMEM), so it keeps the pipelined
    # blocks; longer grids move the streams by hand.
    if num_blocks == 1:
        res_spec = pl.BlockSpec((tb, res_width), lambda i: (i, 0))
        new_spec = pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0))
    else:
        res_spec = new_spec = pl.BlockSpec(memory_space=pl.ANY)
        scratch_shapes += [
            pltpu.VMEM((2, tb, res_width), jnp.bfloat16),
            pltpu.VMEM((2, tb, hc_hidden), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ]
    new_res, layer2d, post_mix, comb_mix = pl.pallas_call(
        functools.partial(
            _fused_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            num_blocks=num_blocks,
            gate_blocks=gate_blocks,
            gate_kwargs=dict(
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
        ),
        grid=(num_blocks,),
        in_specs=[
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
            res_spec,
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            resident,
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
        ],
        out_specs=[
            new_spec,
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((group, hc_mult), lambda i: (i // gate_blocks, 0)),
            pl.BlockSpec((group, hc_mult * hc_mult), lambda i: (i // gate_blocks, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
            jax.ShapeDtypeStruct((padded_tokens, hc_mult), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, hc_mult * hc_mult), jnp.float32),
        ],
        scratch_shapes=scratch_shapes,
        compiler_params=compiler_params,
    )(x2d, res2d, post2d, comb2d, fn, sc2d, hb2d)

    return cast(
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        utils.trim_to(num_tokens, new_res, post_mix, comb_mix, layer2d),
    )


def mhc_fused_post_pre(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
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
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = FUSED_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused seam op: one Pallas kernel for post, the next pre and its gates.
    Returns (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur).
    """
    assert residual.dtype == jnp.bfloat16

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    res2d = residual.reshape(-1, hc_mult * hidden_size)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)

    new_res2d, post_mix_cur, comb_mix_cur, layer_input_cur = fused_post_pre_2d(
        x2d,
        res2d,
        post2d,
        comb2d,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        token_block_size=token_block_size,
        gemm_precision=gemm_precision,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    return (
        new_res2d.reshape(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.reshape(*outer_shape, hc_mult, 1),
        comb_mix_cur.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.reshape(*outer_shape, hidden_size),
    )


def mhc_fused_post_pre_embed(
    x: jax.Array,
    embed: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """``mhc_fused_post_pre`` whose old residual is ``embed`` in every stream.

    embed (..., hidden_size) bf16 is read once instead of hc_mult copies;
    outputs as ``mhc_fused_post_pre``, to f32 rounding.
    """
    assert embed.dtype == jnp.bfloat16
    outer_shape = embed.shape[:-1]
    hidden_size = embed.shape[-1]
    hc_mult = fn.shape[1] // hidden_size

    new_res2d, post_mix_cur, comb_mix_cur, layer_input_cur = fused_post_pre_2d(
        x.reshape(-1, hidden_size).astype(jnp.bfloat16),
        embed.reshape(-1, hidden_size),
        post_layer_mix.reshape(-1, hc_mult),
        comb_res_mix.reshape(-1, hc_mult * hc_mult),
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )

    return (
        new_res2d.reshape(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.reshape(*outer_shape, hc_mult, 1),
        comb_mix_cur.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.reshape(*outer_shape, hidden_size),
    )
