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
"""Composed multi-rank reference step along a resolved `Schedule`."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .blocks.gdn import GDNBlock
from .blocks.gqa import GQABlock
from .blocks.norm import NormBlock
from .decls import BlockBinding
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .geometry import ModelGeometry
from .ops import BF16, F32, rnd, rup
from .pool.config import cache_exec
from .schedule import BlockOrBinding, ResolvedSchedule
from .spec import Block
from .validate import norm_zc_and_eps


def _make_ref_block(b: BlockOrBinding | None, g: ModelGeometry) -> Block | None:
    if b is None or isinstance(b, Block):
        return b
    if not isinstance(b, BlockBinding):
        raise MegakernelConfigError(f"Expected Block or BlockBinding, got {b!r}")
    if issubclass(b.block_cls, NormBlock):
        zc, eps_val = norm_zc_and_eps(b, None, g.eps)
        return b.block_cls(b.prefix, geometry=g, zero_centered=zc, eps=eps_val)
    opts = {k: v for k, v in b.options.items() if v is not None}
    return b.block_cls(geometry=g, prefix=b.prefix, **opts)


def composed_reference_step(
    sched: ResolvedSchedule,
    final_norm: BlockOrBinding | None,
    g: ModelGeometry,
    ex: ExecSpec,
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
    *,
    paged_kv: bool | None = None,
    pool_io: bool = False,
) -> tuple[jax.Array, dict[str, Any]]:
    """Execute hybrid GQA/GDN + MoE block `reference()` along `sched`."""
    R, B, T, NS = ex.R, ex.B, ex.T, ex.NS
    L, ngdn = g.L, g.NGDN
    use_paged_kv = cache_exec(ex).paged_kv if paged_kv is None else bool(paged_kv)
    step_state: dict[str, Any] = {
        "kc": logical_states["kc"],
        "vc": logical_states["vc"],
    }
    CR, HR = rup(3 * B, 8), rup(3 + T, 8)
    vh = KD = VD = QKV = 0
    conv_in = rec_in = None
    if ngdn > 0:
        lay_gdn = GDNBlock.lin_in_layout(g, ex)
        vh, KD, VD, QKV = lay_gdn.vh, lay_gdn.KD, lay_gdn.VD, lay_gdn.QKV
        conv_in = logical_states.get(
            "conv", jnp.zeros((R, ngdn, NS, 3, QKV), dtype=F32)
        )
        if pool_io:
            conv_in = conv_in.astype(BF16).astype(F32)
        rec_in = logical_states.get(
            "rec", jnp.zeros((R, ngdn, NS, vh, KD, VD), dtype=BF16)
        ).reshape(R, ngdn, NS, vh, KD, VD)
        step_state["conv"], step_state["rec"] = conv_in, rec_in
        step_state["conv_out"] = jnp.zeros(
            (R, ngdn, CR, QKV) if T == 1 else (R, ngdn, NS, HR, QKV), dtype=F32
        )
        step_state["rec_out"] = jnp.zeros((R, ngdn, B * vh, KD, VD), dtype=BF16)

    res = hidden[:B].astype(F32)
    for lyr in range(L):
        pre_blk = _make_ref_block(sched.pre_norm[lyr], g)
        if pre_blk is None:
            raise MegakernelConfigError(
                f"layer {lyr} slot 'pre_norm' is None (kernel requires pre_norm)"
            )
        x_mix = pre_blk.reference(logical_weights["in_norm"][lyr], res, g=g)[0]
        mix_blk = _make_ref_block(sched.mixer[lyr], g)
        if mix_blk is None:
            raise MegakernelConfigError(f"Layer {lyr} has no mixer block")
        b_idx = sched.mixer_branch_indices[lyr]
        if isinstance(mix_blk, GQABlock):
            mix_kw = {
                "pos": pos,
                "att_idx": b_idx,
                "layer_idx": lyr,
                "paged_kv": use_paged_kv,
            }
        elif isinstance(mix_blk, GDNBlock):
            mix_kw = {"gdn_idx": b_idx, "layer_idx": lyr, "multi_layer": True}
        else:
            mix_kw = {"branch_idx": b_idx, "layer_idx": lyr}
        mix_delta, step_state = mix_blk.reference(
            logical_weights, x_mix, step_state, g=g, ex=ex, **mix_kw
        )
        res = rnd(res + mix_delta)
        post_blk = _make_ref_block(sched.post_norm[lyr], g)
        if post_blk is None:
            raise MegakernelConfigError(
                f"layer {lyr} slot 'post_norm' is None (kernel requires post_norm)"
            )
        x_ffn = post_blk.reference(logical_weights["post_norm"][lyr], res, g=g)[0]
        ffn_blk = _make_ref_block(sched.ffn[lyr], g)
        if ffn_blk is None:
            raise MegakernelConfigError(f"Layer {lyr} has no ffn block")
        ffn_delta, step_state = ffn_blk.reference(
            logical_weights, x_ffn, step_state, g=g, ex=ex, layer_idx=lyr
        )
        res = rnd(res + ffn_delta)

    fn_blk = _make_ref_block(final_norm, g)
    if fn_blk is None:
        raise MegakernelConfigError("final_norm is None (kernel requires final_norm)")
    out_b = fn_blk.reference(logical_weights["final_norm"], res, g=g, out_dtype=BF16)[0]
    out_states: dict[str, Any] = {"kc": step_state["kc"], "vc": step_state["vc"]}
    if ngdn > 0 and conv_in is not None and rec_in is not None:
        conv_out, rec_out = step_state["conv_out"], step_state["rec_out"]
        if pool_io:
            conv_out = conv_out.astype(BF16).astype(F32)
        if ex.VER:
            pad_spec = ((0, 0), (0, 0), (0, 0), (0, HR - conv_in.shape[-2]), (0, 0))
            conv_pad = (
                jnp.pad(conv_in, pad_spec)
                if conv_in.ndim == 5 and conv_in.shape[-2] < HR
                else conv_in
            )
            out_states["conv"] = conv_pad
            out_states["rec"] = rec_in.reshape(R, ngdn, NS * vh, KD, VD)
            out_states["conv_ckpt"], out_states["rec_ckpt"] = conv_out, rec_out
        else:
            out_states["conv"], out_states["rec"] = conv_out, rec_out
    return out_b, out_states


__all__ = ["composed_reference_step"]
