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
"""Replication helpers for Qwen3.5 and Qwen3.8 weights and packing at R > 8."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def shard_shared_expert_weights(
    hf: dict[str, Any], g: Any, R: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Shard and replicate shared-expert projections across sh_r groups.

    Replicates projections across sh_rep replicas.
    """
    L, H, IS = g.L, g.H, g.IS
    isr = g.ISR(R)
    sh_rep = g.sh_rep(R)
    sh_r = max(1, R // sh_rep)

    sh_gate = jnp.broadcast_to(hf["sh_gate"][None, ...], (R, L, H))
    if "sh_gate_proj" in hf and "sh_up_proj" in hf:
        sh_g_in, sh_u_in = hf["sh_gate_proj"], hf["sh_up_proj"]
        sh_gs_in, sh_us_in = hf["sh_gate_proj_s"], hf["sh_up_proj_s"]
    else:
        sh_g_in, sh_u_in = hf["sh13"][:, :, :IS], hf["sh13"][:, :, IS:]
        sh_gs_in, sh_us_in = hf["sh13_s"][:, :IS], hf["sh13_s"][:, IS:]

    sh_g = jnp.repeat(
        jnp.transpose(sh_g_in.reshape(L, H, sh_r, isr), (2, 0, 1, 3)), sh_rep, axis=0
    )
    sh_u = jnp.repeat(
        jnp.transpose(sh_u_in.reshape(L, H, sh_r, isr), (2, 0, 1, 3)), sh_rep, axis=0
    )
    sh13 = jnp.concatenate([sh_g, sh_u], axis=-1)

    sh_gs = jnp.repeat(
        jnp.transpose(sh_gs_in.reshape(L, sh_r, isr), (1, 0, 2)), sh_rep, axis=0
    )
    sh_us = jnp.repeat(
        jnp.transpose(sh_us_in.reshape(L, sh_r, isr), (1, 0, 2)), sh_rep, axis=0
    )
    sh13_s = jnp.concatenate([sh_gs, sh_us], axis=-1)

    sh2 = jnp.repeat(
        jnp.transpose(hf["sh2"].reshape(L, sh_r, isr, H), (1, 0, 2, 3)), sh_rep, axis=0
    )
    sh2_s = jnp.broadcast_to(hf["sh2_s"][None, ...], (R, L, H))
    return sh_gate, sh13, sh13_s, sh2, sh2_s


def shard_gdn_weights(hf: dict[str, Any], g: Any, ex: Any) -> dict[str, jax.Array]:
    """Shard and replicate GDN linear attention and conv weights across R ranks."""
    from ..modules.gdn.module import GDNModule

    R = ex.R
    H, ngdn = g.H, g.NGDN
    lay_gdn = GDNModule.lin_in_layout(g, ex)
    kh, vh, KD, VD = lay_gdn.kh, lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
    kh_rep = g.kh_rep(R)
    kh_r = max(1, R // kh_rep)
    b_pad = lay_gdn.A_OFF - lay_gdn.BA
    a_pad = lay_gdn.NLI - lay_gdn.A_OFF
    kq_w = kh * KD
    v_w = vh * VD

    lin_q = jnp.repeat(
        jnp.transpose(hf["lin_q"].reshape(ngdn, H, kh_r, kq_w), (2, 0, 1, 3)),
        kh_rep,
        axis=0,
    )
    lin_k = jnp.repeat(
        jnp.transpose(hf["lin_k"].reshape(ngdn, H, kh_r, kq_w), (2, 0, 1, 3)),
        kh_rep,
        axis=0,
    )
    lin_v = jnp.transpose(hf["lin_v"].reshape(ngdn, H, R, v_w), (2, 0, 1, 3))
    lin_z = jnp.transpose(hf["lin_z"].reshape(ngdn, H, R, v_w), (2, 0, 1, 3))
    lin_b = jnp.transpose(hf["lin_b"].reshape(ngdn, H, R, vh), (2, 0, 1, 3))
    lin_a = jnp.transpose(hf["lin_a"].reshape(ngdn, H, R, vh), (2, 0, 1, 3))
    if b_pad > vh:
        lin_b = jnp.pad(lin_b, ((0, 0), (0, 0), (0, 0), (0, b_pad - vh)))
    if a_pad > vh:
        lin_a = jnp.pad(lin_a, ((0, 0), (0, 0), (0, 0), (0, a_pad - vh)))
    lin_in = jnp.concatenate([lin_q, lin_k, lin_v, lin_z, lin_b, lin_a], axis=-1)
    assert lin_in.shape[-1] == lay_gdn.NLI, (lin_in.shape[-1], lay_gdn.NLI)

    lin_q_s = jnp.repeat(
        jnp.transpose(hf["lin_q_s"].reshape(ngdn, kh_r, kq_w), (1, 0, 2)),
        kh_rep,
        axis=0,
    )
    lin_k_s = jnp.repeat(
        jnp.transpose(hf["lin_k_s"].reshape(ngdn, kh_r, kq_w), (1, 0, 2)),
        kh_rep,
        axis=0,
    )
    lin_v_s = jnp.transpose(hf["lin_v_s"].reshape(ngdn, R, v_w), (1, 0, 2))
    lin_z_s = jnp.transpose(hf["lin_z_s"].reshape(ngdn, R, v_w), (1, 0, 2))
    lin_b_s = jnp.transpose(hf["lin_b_s"].reshape(ngdn, R, vh), (1, 0, 2))
    lin_a_s = jnp.transpose(hf["lin_a_s"].reshape(ngdn, R, vh), (1, 0, 2))
    if b_pad > vh:
        lin_b_s = jnp.pad(lin_b_s, ((0, 0), (0, 0), (0, b_pad - vh)))
    if a_pad > vh:
        lin_a_s = jnp.pad(lin_a_s, ((0, 0), (0, 0), (0, a_pad - vh)))
    lin_in_s = jnp.concatenate(
        [lin_q_s, lin_k_s, lin_v_s, lin_z_s, lin_b_s, lin_a_s], axis=-1
    )
    assert lin_in_s.shape[-1] == lay_gdn.NLI, (lin_in_s.shape[-1], lay_gdn.NLI)

    lin_out = jnp.transpose(hf["lin_out"].reshape(ngdn, R, v_w, H), (1, 0, 2, 3))
    lin_out_s = jnp.broadcast_to(hf["lin_out_s"][None, ...], (R, ngdn, H))

    c_full = hf["conv"]
    KH_KD = g.KH * KD
    c_q = jnp.repeat(
        jnp.transpose(c_full[:, :, :KH_KD].reshape(ngdn, 4, kh_r, kq_w), (2, 0, 1, 3)),
        kh_rep,
        axis=0,
    )
    c_k = jnp.repeat(
        jnp.transpose(
            c_full[:, :, KH_KD : 2 * KH_KD].reshape(ngdn, 4, kh_r, kq_w),
            (2, 0, 1, 3),
        ),
        kh_rep,
        axis=0,
    )
    c_v = jnp.transpose(
        c_full[:, :, 2 * KH_KD :].reshape(ngdn, 4, R, v_w), (2, 0, 1, 3)
    )
    conv = jnp.concatenate([c_q, c_k, c_v], axis=-1).reshape(R, ngdn, 4 * lay_gdn.QKV)

    alog = jnp.transpose(hf["alog"].reshape(ngdn, R, vh), (1, 0, 2))
    dtb = jnp.transpose(hf["dtb"].reshape(ngdn, R, vh), (1, 0, 2))
    gnorm = jnp.broadcast_to(hf["gnorm"][None, ...], (R, ngdn, VD))

    return {
        "lin_in": lin_in,
        "lin_in_s": lin_in_s,
        "lin_out": lin_out,
        "lin_out_s": lin_out_s,
        "conv": conv,
        "alog": alog,
        "dtb": dtb,
        "gnorm": gnorm,
    }


def replicate_logical_inputs(
    g: Any,
    ex: Any,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
) -> None:
    """Replicate shared expert and GDN weights/states across rank replicas."""
    R = ex.R
    sh_rep = g.sh_rep(R)
    if sh_rep > 1:
        for k in ("sh13", "sh13_s", "sh2"):
            if k in logical_weights:
                w_k = logical_weights[k]
                if isinstance(w_k, tuple):
                    logical_weights[k] = tuple(
                        jnp.repeat(part[::sh_rep], sh_rep, axis=0) for part in w_k
                    )
                else:
                    logical_weights[k] = jnp.repeat(w_k[::sh_rep], sh_rep, axis=0)

    ngdn = g.NGDN
    if ngdn > 0:
        kh_rep = g.kh_rep(R)
        if kh_rep > 1:
            from ..modules.gdn.module import GDNModule

            lay_gdn = GDNModule.lin_in_layout(g, ex)
            qkv = lay_gdn.QKV
            qk_cols = 2 * lay_gdn.kh * lay_gdn.KD
            for k in ("lin_in", "lin_in_s"):
                if k in logical_weights:
                    arr = logical_weights[k]
                    rep_qk = jnp.repeat(arr[::kh_rep, ..., :qk_cols], kh_rep, axis=0)
                    logical_weights[k] = arr.at[..., :qk_cols].set(rep_qk)
            if "conv" in logical_weights:
                cw = logical_weights["conv"].reshape(R, ngdn, 4, qkv)
                cw_qk = jnp.repeat(cw[::kh_rep, ..., :qk_cols], kh_rep, axis=0)
                logical_weights["conv"] = (
                    cw.at[..., :qk_cols].set(cw_qk).reshape(R, ngdn, 4 * qkv)
                )
            if "conv" in logical_states:
                cs = logical_states["conv"]
                logical_states["conv"] = cs.at[..., :qk_cols].set(
                    jnp.repeat(cs[::kh_rep, ..., :qk_cols], kh_rep, axis=0)
                )
