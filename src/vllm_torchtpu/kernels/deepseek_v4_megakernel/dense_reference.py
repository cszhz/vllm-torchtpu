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
"""Plain-JAX DSv4 dense chain: the numerics reference and the XLA incumbent.

Reproduces the stock serving formulas op by op (attention core and routed
experts stubbed exactly as the ring kernel stubs them). Three orders:

* ``"stock_kernels"``: the stock Pallas mHC and ``wo_a`` kernels plus XLA ops,
  i.e. what serving runs (TPU only).
* ``"stock"``: the same formulas in jnp (runs on CPU).
* ``"kernel"``: the same formulas with every f32 reduction split and ordered
  the way the ring kernel orders it (K tiles, lane chunks, per-stream mHC
  dots). Only the summation order differs from ``"stock"``, so the ring
  kernel in interpret mode must match it bit for bit.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils as mhc_utils
from vllm_torchtpu.kernels.deepseek_v4_megakernel.dense_config import (
    CSA,
    HCA,
    DenseChainConfig,
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel.dense_ring import (
    EW,
    sinkhorn,
)
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_quantized_matmul,
)

F32, BF16 = jnp.float32, jnp.bfloat16
FP8_MAX = 448.0
LANE = 128


def rms_norm(x, w, eps):
    """vLLM ``ir.ops.rms_norm``: f32 statistics, bf16 weight multiply."""
    xf = x.astype(F32)
    xf = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf.astype(w.dtype) * w).astype(x.dtype)


def _chunked_sum_sq(xf, ew):
    ss = None
    for c in range(0, xf.shape[1], ew):
        v = xf[:, c : c + ew]
        p = jnp.sum(v * v, axis=1, keepdims=True)
        ss = p if ss is None else ss + p
    return ss


def rms_norm_kernel_order(x, w, eps):
    xf = x.astype(F32)
    r = jax.lax.rsqrt(_chunked_sum_sq(xf, EW) / x.shape[1] + eps)
    return (xf * r).astype(BF16) * w.astype(BF16)


def layer_input_norm(h, w, eps):
    """attn_norm / ffn_norm as the stock mHC kernels fuse them: the weight
    multiply in f32, one rounding (``mhc.utils.finish_layer_input``)."""
    return mhc_utils.finish_layer_input(h, w.reshape(1, -1), eps)


def layer_input_norm_kernel_order(h, w, eps):
    xf = h.astype(F32)
    r = jax.lax.rsqrt(_chunked_sum_sq(xf, EW) / h.shape[1] + eps)
    return (xf * r * w.astype(F32)).astype(BF16)


def ktiled_dot(x, w, tile_rows):
    """x @ w accumulated over K tiles in order (the ring's accumulation)."""
    acc = None
    for k0 in range(0, w.shape[0], tile_rows):
        d = jnp.dot(x[:, k0 : k0 + tile_rows], w[k0 : k0 + tile_rows],
                    preferred_element_type=F32)  # fmt: skip
        acc = d if acc is None else acc + d
    return acc


def fp8_linear_kernel_order(x, w, w_scale, tile_rows):
    """``xla_quantized_matmul`` with the ring's K-tile accumulation."""
    x_q, x_scale = quantize_tensor(x, w.dtype)
    out = ktiled_dot(x_q, w, tile_rows) * x_scale
    return (out * w_scale[None]).astype(x.dtype)


def mhc_fused_kernel_order(x, res, post, comb, fn, scale, base, cfg):
    """mhc_fused_jnp with the ring kernel's reduction order."""
    m, h = cfg.hc_mult, cfg.hidden
    ew = math.gcd(h, EW)
    new = [[None] * (h // ew) for _ in range(m)]
    ss = jnp.zeros((x.shape[0], 1), F32)
    for ci in range(h // ew):
        c = ci * ew
        xv = x[:, c : c + ew].astype(F32)
        old = [res[:, i * h + c : i * h + c + ew].astype(F32) for i in range(m)]
        for j in range(m):
            a = post[:, j : j + 1] * xv
            for i in range(m):
                a = a + comb[:, i * m + j : i * m + j + 1] * old[i]
            nb = a.astype(BF16)
            new[j][ci] = nb
            nf = nb.astype(F32)
            ss = ss + jnp.sum(nf * nf, axis=1, keepdims=True)
    g = jnp.concatenate([jnp.concatenate(r, axis=1) for r in new], axis=1)
    chunks = jnp.concatenate(mhc_utils.split_fn3(fn)[: cfg.hc_chunks], axis=0)
    rows = cfg.hc_rows
    chunks = jnp.pad(chunks, ((0, rows - chunks.shape[0]), (0, 0)))
    dn = (((1,), (1,)), ((), ()))
    mix = jnp.zeros((x.shape[0], rows), F32)
    for ci in range(h // ew):  # the ring kernel's chunk-by-chunk mix GEMM
        for j in range(m):
            sl = slice(j * h + ci * ew, j * h + (ci + 1) * ew)
            mix = mix + jax.lax.dot_general(new[j][ci], chunks[:, sl], dn,
                                            preferred_element_type=F32)  # fmt: skip
    nm = cfg.n_mix
    mixes = mix[:, :nm]
    for c in range(1, cfg.hc_chunks):
        mixes = mixes + mix[:, c * nm : (c + 1) * nm]
    rs = jax.lax.rsqrt(ss / (m * h) + cfg.rms_eps)
    sm = mixes * rs
    pre = jax.nn.sigmoid(sm[:, :m] * scale[0] + base[:m]) + cfg.hc_eps
    post_n = jax.nn.sigmoid(sm[:, m : 2 * m] * scale[1] + base[m : 2 * m])
    post_n = post_n * cfg.hc_post_mult
    cl = sm[:, 2 * m :] * scale[2] + base[2 * m :]
    cols = [cl[:, k : k + 1] for k in range(m * m)]
    comb_n = jnp.concatenate(sinkhorn(cols, m, cfg.hc_eps, cfg.sinkhorn_iters), 1)
    gf = g.astype(F32)
    lay = pre[:, 0:1] * gf[:, :h]
    for i in range(1, m):
        lay = lay + pre[:, i : i + 1] * gf[:, i * h : (i + 1) * h]
    return g, post_n, comb_n, lay.astype(BF16)


def mhc_post_only(x, res, post, comb, m, h):
    """The residual streams after the pending post (mhc_post_op), bf16."""
    streams = []
    for j in range(m):
        acc = post[:, j : j + 1] * x.astype(F32)
        for i in range(m):
            k = i * m + j
            acc = acc + comb[:, k : k + 1] * res[:, i * h : (i + 1) * h].astype(F32)
        streams.append(acc.astype(BF16))
    return jnp.concatenate(streams, axis=1)


def mhc_fused_jnp(x, res, post, comb, fn, scale, base, cfg: DenseChainConfig):
    """``fused_post_pre_kernel._fused_kernel`` + ``mhc_pre_gates`` in jnp.

    x [M, h] bf16, res [M, 4h] bf16, post [M, 4], comb [M, 16] (row-major
    (i, j)), fn f32 [24, 4h]. Returns (res, post, comb, layer_input).
    """
    m, h = cfg.hc_mult, cfg.hidden
    streams = []
    for j in range(m):
        acc = post[:, j : j + 1] * x.astype(F32)
        for i in range(m):
            k = i * m + j
            acc = acc + comb[:, k : k + 1] * res[:, i * h : (i + 1) * h].astype(F32)
        streams.append(acc.astype(BF16))
    g = jnp.concatenate(streams, axis=1)
    dn = (((1,), (1,)), ((), ()))
    chunks = mhc_utils.split_fn3(fn)[: cfg.hc_chunks]
    mixes = jax.lax.dot_general(g, chunks[0], dn, preferred_element_type=F32)
    for c in chunks[1:]:
        mixes = mixes + jax.lax.dot_general(g, c, dn, preferred_element_type=F32)
    gf = g.astype(F32)
    sqr = jnp.sum(gf * gf, axis=-1, keepdims=True)
    scaled = mixes[:, :m] * jax.lax.rsqrt(sqr / (m * h) + cfg.rms_eps)
    pre = jax.nn.sigmoid(scaled * scale[0] + base[:m]) + cfg.hc_eps
    lay = pre[:, 0:1] * gf[:, :h]
    for i in range(1, m):
        lay = lay + pre[:, i : i + 1] * gf[:, i * h : (i + 1) * h]
    _, post_n, comb_n = mhc_utils.mhc_pre_gates(
        mixes,
        sqr,
        m,
        h,
        scale,
        base,
        cfg.rms_eps,
        cfg.hc_eps,
        cfg.hc_eps,
        cfg.hc_post_mult,
        cfg.sinkhorn_iters,
    )
    return g, post_n, comb_n.reshape(-1, m * m), lay.astype(BF16)


def mhc_fused_stock(x, res, post, comb, fn, scale, base, cfg: DenseChainConfig):
    from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel

    m, h = cfg.hc_mult, cfg.hidden
    r, p, c, lay = fused_post_pre_kernel.mhc_fused_post_pre(
        x,
        res.reshape(-1, m, h),
        post.reshape(-1, m, 1),
        comb.reshape(-1, m, m),
        fn,
        scale,
        base,
        cfg.rms_eps,
        cfg.hc_eps,
        cfg.hc_eps,
        cfg.hc_post_mult,
        cfg.sinkhorn_iters,
        gemm_precision="highest" if cfg.hc_chunks == 3 else "default",
    )
    return r.reshape(-1, m * h), p.reshape(-1, m), c.reshape(-1, m * m), lay


def stub_attention(q, cfg: DenseChainConfig):
    """Deterministic stand-in for the attention core: per-head RMSNorm of q."""
    qh = q.reshape(q.shape[0], cfg.heads, cfg.head_dim).astype(F32)
    qh = qh * jax.lax.rsqrt(jnp.mean(qh * qh, axis=-1, keepdims=True) + cfg.rms_eps)
    return qh.astype(BF16).reshape(q.shape)


def rotate_pairs(x):
    """``[-x1, x0, -x3, x2, ...]`` over the last axis (rope.py ``_rotate_gptj``)."""
    xp = x.reshape(*x.shape[:-1], -1, 2)
    return jnp.stack([-xp[..., 1], xp[..., 0]], axis=-1).reshape(x.shape)


def wo_a_jnp(o, wo_a, scale, cos_sin, cfg: DenseChainConfig):
    """``o_projection._kernel`` math: inverse rope on each head's last 128
    lanes, per-(row, group) fp8 activation quantization in bf16, fp8 dot."""
    t, d = o.shape[0], cfg.head_dim
    x = o.reshape(t, cfg.heads, d)
    tail = x[:, :, d - LANE :].astype(F32)
    cos, sin = cos_sin[:, None, :LANE], cos_sin[:, None, LANE:]
    roped = tail * cos + rotate_pairs(tail) * sin
    x = jnp.concatenate([x[:, :, : d - LANE], roped.astype(BF16)], axis=-1)
    x = x.reshape(t, cfg.groups, cfg.heads_per_group * d)
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    inv = (FP8_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(BF16)
    lhs = (x * inv).astype(jnp.float8_e4m3fn)
    w = wo_a.reshape(cfg.heads_per_group * d, cfg.groups, cfg.o_lora)
    part = jnp.einsum("tgd,dgr->tgr", lhs, w, preferred_element_type=F32)
    part = part * (1.0 / inv.astype(F32))
    out = part * scale.reshape(1, cfg.groups, cfg.o_lora)
    return out.astype(BF16).reshape(t, cfg.groups * cfg.o_lora)


def wo_a_stock(o, wo_a, scale, cos_sin, cfg: DenseChainConfig):
    from vllm_torchtpu.kernels.deepseek_v4 import o_projection

    x = o.reshape(o.shape[0], cfg.heads, cfg.head_dim)
    return o_projection.wo_a_projection(x, wo_a, scale, cos_sin)


def swiglu_clamp(gu, limit):
    """``SiluAndMulWithClamp`` (alpha=1, beta=0), f32 math, one bf16 rounding."""
    d = gu.shape[-1] // 2
    g = jnp.minimum(gu[:, :d].astype(F32), limit)
    u = jnp.clip(gu[:, d:].astype(F32), -limit, limit)
    return (g * jax.nn.sigmoid(g) * u).astype(BF16)


def layer_weights(w, cfg: DenseChainConfig, layer: int) -> dict:
    """Per-layer slices of the stacked natural weights."""
    lt = cfg.layer_types[layer]
    ords = {
        "all": layer,
        "csa": cfg.layer_types[:layer].count(CSA),
        "hca": cfg.layer_types[:layer].count(HCA),
    }
    csa_only = ("wp", "idx_comp", "comp_csa", "idx_wq_b", "idx_wq_b_scale")
    out = {}
    for name, arr in w.items():
        if name in csa_only:
            if lt == CSA:
                out[name] = arr[ords["csa"]]
        elif name == "comp_hca":
            if lt == HCA:
                out[name] = arr[ords["hca"]]
        else:
            out[name] = arr[ords["all"]]
    return out


ORDERS = ("stock_kernels", "stock", "kernel")


def dense_layer(lw, state, cos_sin, cfg: DenseChainConfig, csa: bool, order: str):
    """One decoder layer's dense chain. Returns (state, probes)."""
    assert order in ORDERS, order
    st = cfg.streams
    if order == "kernel":
        mhc = mhc_fused_kernel_order
        rms_norm_ = rms_norm_kernel_order
        in_norm = layer_input_norm_kernel_order

        def lin(x, name, wname=None):
            wname = wname or name
            return fp8_linear_kernel_order(
                x, lw[wname], lw[wname + "_scale"], st[name].tile_rows
            )

        def bdot(x, w, name):
            return ktiled_dot(x, w, st[name].tile_rows)
    else:
        mhc = mhc_fused_stock if order == "stock_kernels" else mhc_fused_jnp
        rms_norm_ = rms_norm
        in_norm = layer_input_norm

        def lin(x, name, wname=None):
            wname = wname or name
            return xla_quantized_matmul(x, lw[wname], lw[wname + "_scale"])

        def bdot(x, w, name):
            return jnp.dot(x, w, preferred_element_type=F32)

    wo_a = wo_a_stock if order == "stock_kernels" else wo_a_jnp
    eps = cfg.rms_eps
    x, res, post, comb = state

    res, post, comb, h = mhc(
        x, res, post, comb, lw["hc_attn_fn"], lw["hc_attn_scale"],
        lw["hc_attn_base"], cfg,
    )  # fmt: skip
    xa = in_norm(h, lw["attn_norm"], eps)
    qkv = lin(xa, "wqa")
    qr = rms_norm_(qkv[:, : cfg.q_lora], lw["q_norm"], eps)
    kv = rms_norm_(qkv[:, cfg.q_lora :], lw["kv_norm"], eps)
    if csa:
        wpad = jnp.zeros((cfg.hidden, cfg.wp_cols - cfg.idx_heads), lw["wp"].dtype)
        wcat = jnp.concatenate(
            [lw["wp"], wpad, lw["idx_comp"], lw["comp_csa"]], axis=-1
        )
        xproj = bdot(xa, wcat, "xproj_csa")
        idx_q = lin(qr, "idx_wq_b")
    else:
        xproj = bdot(xa, lw["comp_hca"], "xproj_hca")
        idx_q = None
    q = lin(qr, "wq_b")
    o = stub_attention(q, cfg)
    z = wo_a(o, lw["wo_a"], lw["wo_a_scale"], cos_sin, cfg)
    x = lin(z, "wo_b")

    res, post, comb, h = mhc(
        x, res, post, comb, lw["hc_ffn_fn"], lw["hc_ffn_scale"],
        lw["hc_ffn_base"], cfg,
    )  # fmt: skip
    xf = in_norm(h, lw["ffn_norm"], eps)
    logits = bdot(xf, lw["gate"], "gate")
    gu = lin(xf, "w13")
    act = swiglu_clamp(gu, cfg.swiglu_limit)
    x = lin(act, "w2")  # + routed experts (stub: 0)
    probes = {
        "qkv": jnp.concatenate([qr, kv], axis=-1).astype(F32),
        "xproj": xproj,
        "gate": logits,
        "idx_q": idx_q,
    }
    return (x, res, post, comb), probes


def dense_chain(w, inputs, cfg: DenseChainConfig, order: str = "stock"):
    """All layers. Returns (final state, per-layer probes)."""
    state = (inputs["x"], inputs["res"], inputs["post"], inputs["comb"])
    probes = []
    for layer, lt in enumerate(cfg.layer_types):
        lw = layer_weights(w, cfg, layer)
        state, p = dense_layer(lw, state, inputs["cos_sin"], cfg, lt == CSA, order)
        probes.append(p)
    return state, probes
