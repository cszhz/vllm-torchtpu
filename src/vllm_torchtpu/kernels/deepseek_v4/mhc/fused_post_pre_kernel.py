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
from vllm_torchtpu.kernels.deepseek_v4.mhc.utils import DEFAULT_VMEM_LIMIT_BYTES


def _fused_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    *refs,
    n_fn,
    hc_mult,
    hidden_size,
    gemm_precision,
    rms_eps,
    hc_pre_eps,
    gates,
    norm_eps,
    fn_prefetch=False,
    per_stream_dot=False,
    ablate=(),
    dense_group=0,
    hoist_f32=False,
):
    """One token block: recombine, store once, 3-pass GEMM, collapse.

    refs: [norm_w_ref] (only with ``norm_eps``), then the outputs.
    gate_a/gate_b are (mixes, sqrsum) when ``gates`` is None, else the
    in-block (post_mix, comb_mix).
    """
    fn_refs, (sc_ref, hb_ref), refs = refs[:n_fn], refs[n_fn : n_fn + 2], refs[n_fn + 2 :]
    if dense_group > 1:
        *refs, mt_ref = refs
    if fn_prefetch:
        # fn stays in HBM; step 0 streams it into a persistent VMEM copy,
        # one chunk per residual stream, under the recombine's VPU work.
        *refs, fn_vmem, fn_sem = refs
        fn_hbm = fn_refs[0]
        h_ = hidden_size
        copies = [
            pltpu.make_async_copy(
                fn_hbm.at[:, pl.ds(j * h_, h_)],
                fn_vmem.at[:, pl.ds(j * h_, h_)],
                fn_sem.at[j],
            )
            for j in range(hc_mult)
        ]
        first = pl.program_id(0) == 0

        @pl.when(first)
        def _start():
            for c in copies:
                c.start()

        fn_refs = (fn_vmem,)
    *norm_w, newres_ref, gate_a_ref, gate_b_ref, layer_ref = refs
    post = post_ref[...]  # (tb, hc_mult) f32
    comb = comb_ref[...]  # (tb, hc_mult * hc_mult) f32, row-major (i, j)

    # Keep the streams (and x) in bf16 and upcast per use: transient f32
    # values instead of 4 held f32 copies halves the resident working set,
    # leaving VMEM headroom for the pipeline to prefetch the next block.
    # bf16 -> f32 casts are exact, so results are unchanged.
    x_bf = x_ref[...]  # (tb, hidden) bf16
    old_bf = [
        res_ref[:, i * hidden_size : (i + 1) * hidden_size] for i in range(hc_mult)
    ]

    if hoist_f32:
        # One bf16 -> f32 unpack per input stream instead of one per use.
        x_f = x_bf.astype(jnp.float32)
        old_f = [o.astype(jnp.float32) for o in old_bf]
    streams_bf = []
    for j in range(hc_mult):
        acc = post[:, j : j + 1] * (x_f if hoist_f32 else x_bf.astype(jnp.float32))
        for i in range(hc_mult if "recombine" not in ablate else 0):
            k = i * hc_mult + j
            o = old_f[i] if hoist_f32 else old_bf[i].astype(jnp.float32)
            acc = acc + comb[:, k : k + 1] * o
        # Round to bf16 BEFORE the GEMM: the unfused path's pre reads the
        # bf16 residual that post stored, and parity requires matching it.
        new_bf = acc.astype(jnp.bfloat16)
        if "recombine" in ablate:  # timing probe only: wrong results
            new_bf = old_bf[j]
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = new_bf
        streams_bf.append(new_bf)

    m3 = hc_mult * (hc_mult + 2)
    if fn_prefetch and not per_stream_dot:

        @pl.when(first)
        def _wait():
            for c in copies:
                c.wait()

    if per_stream_dot:
        # One dot per stream against its K-slice of the stacked chunks: no
        # (tb, M*H) concatenation. Changes the K accumulation order.
        assert len(fn_refs) == 1
        dn = (((1,), (1,)), ((), ()))
        out = None
        for j in range(hc_mult):
            if fn_prefetch:
                # Wait only for this stream's K-chunk: the dot on chunk j
                # runs while chunk j + 1 is still in flight.
                @pl.when(first)
                def _wait_j(j=j):
                    copies[j].wait()

            d = jax.lax.dot_general(
                streams_bf[j],
                fn_refs[0][:, j * hidden_size : (j + 1) * hidden_size],
                dn,
                preferred_element_type=jnp.float32,
            )
            out = d if out is None else out + d
        acc = out[:, :m3]
        if gemm_precision == "highest":
            acc = acc + out[:, m3 : 2 * m3]
            acc = acc + out[:, 2 * m3 :]
        gf_streams = [sb.astype(jnp.float32) for sb in streams_bf]
        sqr = None
        for g in gf_streams:
            p = jnp.sum(g * g, axis=-1, keepdims=True)
            sqr = p if sqr is None else sqr + p
        gf = None
    else:
        g_bf = jnp.concatenate(streams_bf, axis=1)  # (tb, hc_mult * hidden_size)
        if "gemm" in ablate or "fn" in ablate:  # timing probe only
            acc = jnp.zeros((g_bf.shape[0], m3), jnp.float32) + g_bf[:, :m3].astype(
                jnp.float32
            )
        else:
            acc = utils.mix_gemm(g_bf, fn_refs, m3, gemm_precision)
        gf = g_bf.astype(jnp.float32)  # f32 for the squared sum + collapse
        sqr = jnp.sum(gf * gf, axis=-1, keepdims=True)
        gf_streams = [gf[:, i * hidden_size : (i + 1) * hidden_size] for i in range(hc_mult)]

    m, h = hc_mult, hidden_size
    if "gates" in ablate:  # timing probe only
        gate_a_ref[...] = acc[:, :m]
        gate_b_ref[...] = acc[:, m : m + m * m]
        pre_mix = acc[:, :m]
    elif gates is not None and dense_group > 1:
        # Defer the post/comb gates: park this block's scaled mixes and run
        # the Sinkhorn once per dense_group blocks on full (G, tb) slabs.
        m3_ = m * (m + 2)
        tb_ = acc.shape[0]
        scaled = acc * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
        pre_mix = jax.nn.sigmoid(scaled[:, :m] * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
        step = pl.program_id(0) % dense_group
        mt_ref[step] = scaled.T

        @pl.when(step == dense_group - 1)
        def _dense_gates():
            mt = jnp.stack([mt_ref[:, k, :] for k in range(m3_)])
            post_t, comb_t = utils.gates_dense(mt, sc_ref, hb_ref, hc_mult=m, gates=gates)
            for g in range(dense_group):
                gate_a_ref[g * tb_ : (g + 1) * tb_, :] = post_t[:, g, :].T
                gate_b_ref[g * tb_ : (g + 1) * tb_, :] = comb_t[:, g, :].T

    elif gates is None:
        gate_a_ref[...] = acc
        gate_b_ref[...] = sqr
        scaled = acc[:, :m] * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
        pre_mix = jax.nn.sigmoid(scaled * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
    else:
        pre_mix, gate_a_ref[...], gate_b_ref[...] = utils.gates_in_block(
            acc,
            sqr,
            sc_ref,
            hb_ref,
            hc_mult=m,
            hidden_size=h,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            gates=gates,
        )
    lay = pre_mix[:, 0:1] * gf_streams[0]
    for i in range(1, m if "collapse" not in ablate else 1):
        lay = lay + pre_mix[:, i : i + 1] * gf_streams[i]
    if "norm" in ablate:  # timing probe only
        layer_ref[...] = utils.finish_layer_input(lay, None, None)
    else:
        layer_ref[...] = utils.finish_layer_input(lay, *norm_w or [None], norm_eps)


@functools.partial(
    jax.jit,
    static_argnames=(
        "rms_eps",
        "hc_pre_eps",
        "token_block_size",
        "gemm_precision",
        "vmem_limit_bytes",
        "gates",
        "norm_eps",
        "fn_prefetch",
        "per_stream_dot",
        "ablate",
        "dense_group",
        "hoist_f32",
    ),
)
def fused_post_pre_mixes(
    x2d: jax.Array,
    res2d: jax.Array,
    post2d: jax.Array,
    comb2d: jax.Array,
    fn: jax.Array | utils.SplitFn | utils.CatFn,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    *,
    token_block_size: int = 32,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    gates: utils.GateConsts | None = None,
    norm_weight: jax.Array | None = None,
    norm_eps: float | None = None,
    fn_prefetch: bool = False,
    per_stream_dot: bool = False,
    ablate: tuple[str, ...] = (),
    dense_group: int = 0,
    hoist_f32: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The Pallas seam kernel: post recombine fused with pre's mix GEMM
    and in-block collapse.

    Args:
        x2d: (num_tokens, hidden_size), bfloat16 — sublayer output.
        res2d: (num_tokens, hc_mult * hidden_size), bfloat16 — old streams.
        post2d: (num_tokens, hc_mult), float32.
        comb2d: (num_tokens, hc_mult * hc_mult), float32, row-major (i, j).
        fn: (hc_mult3, hc_mult * hidden_size), float32 — next sublayer's
            gate projection.
        hc_scale: (3,), float32 — pre/post/comb logit scales (the kernel
            uses only the pre entry; the rest feed the XLA gates).
        hc_base: (hc_mult3,), float32 — logit biases, pre entries first.
        rms_eps: static; RMS-norm epsilon for the in-block pre gates.
        hc_pre_eps: static; additive floor on the sigmoid pre gates.
        token_block_size: tokens per grid step. Default 32 keeps the
            worst-case VMEM footprint (f32 old streams + bf16 blocks +
            the three resident bf16 fn chunks) near 12 MB at DeepSeek-V4
            shapes; 64 also fits.
        gemm_precision: "highest" (3-pass exact decomposition, the
            default) or "default" (hi-chunk pass only; pre_mix and the
            collapse then also see the hi-only mixes).
        gates: when set, the post/comb gates (incl. Sinkhorn) are computed
            in-block and returned in place of (mixes, sqrsum).
        norm_weight, norm_eps: when set, layer2d is returned RMS-normalized
            with this (hidden_size,) weight (see utils.finish_layer_input).

    Returns:
        new_res2d: (num_tokens, hc_mult * hidden_size), bfloat16.
        mixes: (num_tokens, hc_mult3), float32 — raw, unscaled, for the
            XLA gates.
        sqrsum: (num_tokens, 1), float32.
        layer2d: (num_tokens, hidden_size), bfloat16.
        With ``gates``, mixes/sqrsum are replaced by post_mix
        (num_tokens, hc_mult) and comb_mix (num_tokens, hc_mult**2), f32.
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert res2d.dtype == jnp.bfloat16, res2d.dtype
    assert post2d.dtype == jnp.float32, post2d.dtype
    assert comb2d.dtype == jnp.float32, comb2d.dtype

    num_tokens, hidden_size = x2d.shape
    hc_hidden = res2d.shape[1]
    hc_mult = hc_hidden // hidden_size
    hc_mult3 = fn.shape[0]
    assert res2d.shape == (num_tokens, hc_mult * hidden_size)
    assert fn.shape == (hc_mult3, hc_hidden), (fn.shape, res2d.shape)
    assert gemm_precision in ("highest", "default"), gemm_precision

    # This kernel keeps a fixed block: its default (32) already fits the
    # budget at DeepSeek-V4 shapes, so there is no vmem_need to shrink by.
    tb, padded_tokens = utils.select_token_block(
        num_tokens, token_block_size, whole_small_batch=gates is not None
    )
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)

    fn_ops = utils.fn_operands(fn)
    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)

    if fn_prefetch or per_stream_dot:
        assert len(fn_ops) == 1, "fn_prefetch / per_stream_dot need a CatFn"
    n_blocks = pl.cdiv(padded_tokens, tb)
    if n_blocks > 1:
        # Multi-block grids keep the pipelined resident fn: the in-kernel
        # stream pays off only when one block has nothing else to overlap.
        fn_prefetch = per_stream_dot = False
    group = 1
    if gates is not None and dense_group > 1 and "gates" not in ablate:
        group = max(g for g in range(1, dense_group + 1) if n_blocks % g == 0)
    compiler_params = pltpu.CompilerParams(
        # A prefetched fn is filled at step 0 and dense gates run every
        # group-th step, so those steps must run in order.
        dimension_semantics=(
            ("arbitrary",) if fn_prefetch or group > 1 else ("parallel",)
        ),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )
    if fn_prefetch or "fn" in ablate:
        fn_specs = [pl.BlockSpec(memory_space=pltpu.HBM) for _ in fn_ops]
        scratch = [
            pltpu.VMEM(fn_ops[0].shape, jnp.bfloat16),
            pltpu.SemaphoreType.DMA((hc_mult,)),
        ] if fn_prefetch else []
    else:
        fn_specs = [pl.BlockSpec(f.shape, lambda i: (0, 0)) for f in fn_ops]
        scratch = []
    if group > 1:
        scratch.append(pltpu.VMEM((group, hc_mult3, tb), jnp.float32))

    assert (norm_weight is None) == (norm_eps is None)
    norm_args = [] if norm_weight is None else [norm_weight.reshape(1, hidden_size)]
    norm_specs = [pl.BlockSpec((1, hidden_size), lambda i: (0, 0)) for _ in norm_args]

    gate_widths = (hc_mult3, 1) if gates is None else (hc_mult, hc_mult * hc_mult)
    new_res, gate_a, gate_b, layer2d = pl.pallas_call(
        functools.partial(
            _fused_kernel,
            n_fn=len(fn_ops),
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            gemm_precision=gemm_precision,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            gates=gates,
            norm_eps=norm_eps,
            fn_prefetch=fn_prefetch,
            per_stream_dot=per_stream_dot,
            ablate=ablate,
            dense_group=group,
            hoist_f32=hoist_f32,
        ),
        grid=(pl.cdiv(padded_tokens, tb),),
        in_specs=[
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            *fn_specs,
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
            *norm_specs,
        ],
        out_specs=[
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            *(
                pl.BlockSpec((group * tb, w), lambda i: (i // group, 0))
                for w in gate_widths
            ),
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_hidden), jnp.bfloat16),
            *(
                jax.ShapeDtypeStruct((padded_tokens, w), jnp.float32)
                for w in gate_widths
            ),
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        ],
        scratch_shapes=scratch,
        compiler_params=compiler_params,
    )(x2d, res2d, post2d, comb2d, *fn_ops, sc2d, hb2d, *norm_args)

    return cast(
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        utils.trim_to(num_tokens, new_res, gate_a, gate_b, layer2d),
    )



def _kstream_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    fn_ref,
    sc_ref,
    hb_ref,
    *refs,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_pre_eps,
    gates,
    norm_eps,
):
    """One grid step per residual stream j (= one K-chunk of the mix GEMM).

    Step j recombines stream j, stores it, and multiplies it by fn's j-th
    K-chunk while the pipeline fetches chunk j + 1 and writes stream j - 1
    back; the last step finishes the gates and the collapse. For token
    counts that fit one block, where the token-tiled kernel has nothing to
    overlap fn's DMA with.
    """
    *norm_w, newres_ref, gate_a_ref, gate_b_ref, layer_ref, acc_ref, sqr_ref, g_ref = refs
    m, h = hc_mult, hidden_size
    m3 = m * (m + 2)
    j = pl.program_id(0)
    dn = (((1,), (1,)), ((), ()))

    for jj in range(m):

        @pl.when(j == jj)
        def _step(jj=jj):
            post = post_ref[...]
            comb = comb_ref[...]
            acc = post[:, jj : jj + 1] * x_ref[...].astype(jnp.float32)
            for i in range(m):
                k = i * m + jj
                acc = acc + comb[:, k : k + 1] * res_ref[:, i * h : (i + 1) * h].astype(
                    jnp.float32
                )
            new_bf = acc.astype(jnp.bfloat16)
            newres_ref[...] = new_bf
            g_ref[jj] = new_bf
            d = jax.lax.dot_general(
                new_bf, fn_ref[...], dn, preferred_element_type=jnp.float32
            )
            gf = new_bf.astype(jnp.float32)
            sq = jnp.sum(gf * gf, axis=-1, keepdims=True)
            if jj == 0:
                acc_ref[...] = d
                sqr_ref[...] = sq
            else:
                acc_ref[...] += d
                sqr_ref[...] += sq

            if jj == m - 1:
                out = acc_ref[...]
                mixes = (out[:, :m3] + out[:, m3 : 2 * m3]) + out[:, 2 * m3 :]
                sqr = sqr_ref[...]
                if gates is None:
                    gate_a_ref[...] = mixes
                    gate_b_ref[...] = sqr
                    scaled = mixes[:, :m] * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
                    pre_mix = (
                        jax.nn.sigmoid(scaled * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
                    )
                else:
                    pre_mix, gate_a_ref[...], gate_b_ref[...] = utils.gates_in_block(
                        mixes,
                        sqr,
                        sc_ref,
                        hb_ref,
                        hc_mult=m,
                        hidden_size=h,
                        rms_eps=rms_eps,
                        hc_pre_eps=hc_pre_eps,
                        gates=gates,
                    )
                lay = pre_mix[:, 0:1] * g_ref[0].astype(jnp.float32)
                for i in range(1, m):
                    lay = lay + pre_mix[:, i : i + 1] * g_ref[i].astype(jnp.float32)
                layer_ref[...] = utils.finish_layer_input(
                    lay, *norm_w or [None], norm_eps
                )


@functools.partial(
    jax.jit,
    static_argnames=("rms_eps", "hc_pre_eps", "vmem_limit_bytes", "gates", "norm_eps"),
)
def fused_post_pre_kstream(
    x2d: jax.Array,
    res2d: jax.Array,
    post2d: jax.Array,
    comb2d: jax.Array,
    fn: utils.CatFn,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    *,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    gates: utils.GateConsts | None = None,
    norm_weight: jax.Array | None = None,
    norm_eps: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """``fused_post_pre_mixes`` for a single token block, gridded over the
    hc_mult residual streams instead of tokens (see ``_kstream_kernel``).
    Same outputs; the mixes sum the per-stream partial products, so they
    round differently from the one-GEMM kernel (tolerance tier)."""
    assert isinstance(fn, utils.CatFn), "kstream needs the stacked fn"
    num_tokens, hidden_size = x2d.shape
    hc_hidden = res2d.shape[1]
    hc_mult = hc_hidden // hidden_size
    hc_mult3 = fn.shape[0]
    tb, padded_tokens = utils.select_token_block(
        num_tokens, 1 << 30, whole_small_batch=True
    )
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)
    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)
    assert (norm_weight is None) == (norm_eps is None)
    norm_args = [] if norm_weight is None else [norm_weight.reshape(1, hidden_size)]
    norm_specs = [pl.BlockSpec((1, hidden_size), lambda j: (0, 0)) for _ in norm_args]
    gate_widths = (hc_mult3, 1) if gates is None else (hc_mult, hc_mult * hc_mult)
    const = lambda j: (0, 0)  # noqa: E731
    new_res, gate_a, gate_b, layer2d = pl.pallas_call(
        functools.partial(
            _kstream_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            gates=gates,
            norm_eps=norm_eps,
        ),
        grid=(hc_mult,),
        in_specs=[
            pl.BlockSpec((tb, hidden_size), const),
            pl.BlockSpec((tb, hc_hidden), const),
            pl.BlockSpec((tb, hc_mult), const),
            pl.BlockSpec((tb, hc_mult * hc_mult), const),
            pl.BlockSpec((3 * hc_mult3, hidden_size), lambda j: (0, j)),
            pl.BlockSpec((1, 3), const),
            pl.BlockSpec((1, hc_mult3), const),
            *norm_specs,
        ],
        out_specs=[
            pl.BlockSpec((tb, hidden_size), lambda j: (0, j)),
            *(pl.BlockSpec((tb, w), const) for w in gate_widths),
            pl.BlockSpec((tb, hidden_size), const),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_hidden), jnp.bfloat16),
            *(
                jax.ShapeDtypeStruct((padded_tokens, w), jnp.float32)
                for w in gate_widths
            ),
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        ],
        scratch_shapes=[
            pltpu.VMEM((tb, 3 * hc_mult3), jnp.float32),
            pltpu.VMEM((tb, 1), jnp.float32),
            pltpu.VMEM((hc_mult, tb, hidden_size), jnp.bfloat16),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
    )(x2d, res2d, post2d, comb2d, fn.cat, sc2d, hb2d, *norm_args)
    return cast(
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        utils.trim_to(num_tokens, new_res, gate_a, gate_b, layer2d),
    )


def mhc_fused_post_pre(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array | utils.SplitFn | utils.CatFn,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = 32,
    gemm_precision: str = "highest",
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    gates_in_kernel: bool = False,
    gates_max_tokens: int | None = None,
    norm_weight: jax.Array | None = None,
    norm_eps: float | None = None,
    fn_prefetch: bool = False,
    per_stream_dot: bool = False,
    ablate: tuple[str, ...] = (),
    kstream_max_tokens: int = 0,
    dense_group: int = 0,
    hoist_f32: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused seam op: Pallas post+GEMM+collapse kernel, gates in an XLA
    epilogue or (``gates_in_kernel``, up to
    ``utils.GATES_IN_KERNEL_MAX_TOKENS`` tokens) in the kernel itself.
    Returns (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur);
    with ``norm_weight``/``norm_eps``, layer_input_cur is RMS-normalized.
    """
    assert residual.dtype == jnp.bfloat16

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    res2d = residual.reshape(-1, hc_mult * hidden_size)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)

    gates = utils.gate_consts(
        x2d.shape[0],
        gates_in_kernel,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        gates_max_tokens,
    )
    if (
        kstream_max_tokens
        and isinstance(fn, utils.CatFn)
        and x2d.shape[0] <= kstream_max_tokens
    ):
        new_res2d, gate_a, gate_b, layer_input_cur = fused_post_pre_kstream(
            x2d,
            res2d,
            post2d,
            comb2d,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            vmem_limit_bytes=vmem_limit_bytes,
            gates=gates,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    else:
        new_res2d, gate_a, gate_b, layer_input_cur = fused_post_pre_mixes(
            x2d,
            res2d,
            post2d,
            comb2d,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            token_block_size=token_block_size,
            gemm_precision=gemm_precision,
            vmem_limit_bytes=vmem_limit_bytes,
            gates=gates,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            fn_prefetch=fn_prefetch,
            per_stream_dot=per_stream_dot,
            ablate=ablate,
            dense_group=dense_group,
            hoist_f32=hoist_f32,
        )
    if gates is not None:
        post_mix_cur, comb_mix_cur = gate_a, gate_b
    else:
        _, post_mix_cur, comb_mix_cur = utils.mhc_pre_gates(
            gate_a,
            gate_b,
            hc_mult,
            hidden_size,
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
