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
"""Qwen3.5 HuggingFace weight generation, sharding, and EPLB encoding helpers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .. import planner
from ..exec_spec import ExecSpec
from ..modules.gdn import GDNModule
from ..ops import BF16, F32, FP8, I32


def default_eplb_cores(g: Any, R: int) -> list[list[list[int]]]:
    """Build a deterministic per-layer redundant expert placement `[L][R][EW]`."""
    er, ew = g.ER(R), g.EW(R)
    extra = ew - er
    layers_cores: list[list[list[int]]] = []
    for lyr in range(g.L):
        cores_l: list[list[int]] = []
        for r1 in range(R):
            primary = [r1 * er + s0 for s0 in range(er)]
            redundant: list[int] = []
            if not g.moe_tp and extra > 0 and R > 1:
                for k in range(extra):
                    r0 = (r1 + 1 + (k % (R - 1))) % R
                    s0 = (lyr + k) % er
                    redundant.append(r0 * er + s0)
            else:
                redundant = [-1] * max(0, extra)
            cores_l.append(primary + redundant)
        layers_cores.append(cores_l)
    return layers_cores


def encode_eplb_ecopy(
    g: Any,
    R: int,
    e_cores: list[list[list[int]]] | None = None,
) -> jax.Array:
    """Encode `[L][R][EW]` expert placement table into `[L, E]` `f32` `ecopy` rows."""
    if e_cores is None:
        e_cores = default_eplb_cores(g, R)
    er, ew = g.ER(R), g.EW(R)
    rows = [
        planner.encode_ecopy(g.E, er, ew, g.moe_tp, e_cores[lyr]) for lyr in range(g.L)
    ]
    return jnp.asarray(rows, dtype=F32)


def random_hf_weights(
    key: jax.Array,
    g: Any,
) -> dict[str, jax.Array]:
    """Generate unsharded HuggingFace-style Qwen3.5 weights (`[L, ...]`, full `E`)."""
    L, H, E, inter_dim, IS = g.L, g.H, g.E, g.I, g.IS
    NH, NKV, HD = g.NH, g.NKV, g.HD
    KH, VH, KD, VD = g.KH, g.VH, g.KD, g.VD
    natt, ngdn = g.NATT, g.NGDN

    keys = jax.random.split(key, 56)
    it = iter(keys)

    def rfp8(shape):
        return (jax.random.normal(next(it), shape, dtype=F32) * 0.25).astype(FP8)

    def rscale(shape):
        return jnp.abs(jax.random.normal(next(it), shape, dtype=F32)) * 0.02 + 0.01

    q_proj = rfp8((natt, H, NH * 2 * HD))
    q_proj_s = rscale((natt, NH * 2 * HD))
    k_proj = rfp8((natt, NKV, H, HD))
    k_proj_s = rscale((natt, NKV, HD))
    v_proj = rfp8((natt, NKV, H, HD))
    v_proj_s = rscale((natt, NKV, HD))
    sh_gate_proj = rfp8((L, H, IS))
    sh_gate_proj_s = rscale((L, IS))
    sh_up_proj = rfp8((L, H, IS))
    sh_up_proj_s = rscale((L, IS))
    e_gate_proj = rfp8((L, E, H, inter_dim))
    e_gate_proj_s = rscale((L, E, inter_dim))
    e_up_proj = rfp8((L, E, H, inter_dim))
    e_up_proj_s = rscale((L, E, inter_dim))

    hf: dict[str, jax.Array] = {
        "in_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "post_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "final_norm": jax.random.normal(next(it), (H,), dtype=F32) * 0.05,
        "q_proj": q_proj,
        "q_proj_s": q_proj_s,
        "k_proj": k_proj,
        "k_proj_s": k_proj_s,
        "v_proj": v_proj,
        "v_proj_s": v_proj_s,
        "att_out": rfp8((natt, NH * HD, H)),
        "att_out_s": rscale((natt, H)),
        "qn": jax.random.normal(next(it), (natt, NH, HD), dtype=F32) * 0.05,
        "kn": jax.random.normal(next(it), (natt, NKV, HD), dtype=F32) * 0.05,
        "router": jax.random.normal(next(it), (L, H, E), dtype=BF16) * 0.1,
        "sh_gate": jax.random.normal(next(it), (L, H), dtype=F32) * 0.1,
        "sh_gate_proj": sh_gate_proj,
        "sh_gate_proj_s": sh_gate_proj_s,
        "sh_up_proj": sh_up_proj,
        "sh_up_proj_s": sh_up_proj_s,
        "sh13": jnp.concatenate([sh_gate_proj, sh_up_proj], axis=-1),
        "sh13_s": jnp.concatenate([sh_gate_proj_s, sh_up_proj_s], axis=-1),
        "sh2": rfp8((L, IS, H)),
        "sh2_s": rscale((L, H)),
        "e_gate_proj": e_gate_proj,
        "e_gate_proj_s": e_gate_proj_s,
        "e_up_proj": e_up_proj,
        "e_up_proj_s": e_up_proj_s,
        "e13": jnp.concatenate([e_gate_proj, e_up_proj], axis=-1),
        "e13_s": jnp.concatenate([e_gate_proj_s, e_up_proj_s], axis=-1),
        "e2": rfp8((L, E, inter_dim, H)),
        "e2_s": rscale((L, E, H)),
    }
    if ngdn > 0:
        qkv_full = 2 * KH * KD + VH * VD
        hf.update(
            {
                "lin_q": rfp8((ngdn, H, KH * KD)),
                "lin_q_s": rscale((ngdn, KH * KD)),
                "lin_k": rfp8((ngdn, H, KH * KD)),
                "lin_k_s": rscale((ngdn, KH * KD)),
                "lin_v": rfp8((ngdn, H, VH * VD)),
                "lin_v_s": rscale((ngdn, VH * VD)),
                "lin_z": rfp8((ngdn, H, VH * VD)),
                "lin_z_s": rscale((ngdn, VH * VD)),
                "lin_b": rfp8((ngdn, H, VH)),
                "lin_b_s": rscale((ngdn, VH)),
                "lin_a": rfp8((ngdn, H, VH)),
                "lin_a_s": rscale((ngdn, VH)),
                "lin_out": rfp8((ngdn, VH * VD, H)),
                "lin_out_s": rscale((ngdn, H)),
                "conv": jax.random.normal(next(it), (ngdn, 4, qkv_full), dtype=F32)
                * 0.05,
                "alog": jax.random.normal(next(it), (ngdn, VH), dtype=F32) * 0.1 - 1.0,
                "dtb": jax.random.normal(next(it), (ngdn, VH), dtype=F32) * 0.1,
                "gnorm": jnp.ones((ngdn, VD), dtype=F32)
                + jax.random.normal(next(it), (ngdn, VD), dtype=F32) * 0.05,
            }
        )
    return hf


def split_hf_weights(
    hf: dict[str, jax.Array],
    g: Any,
    ex: ExecSpec,
    *,
    e_cores: list[list[list[int]]] | None = None,
) -> dict[str, Any]:
    """Shard unsharded HuggingFace-style Qwen3.5 weights across `R = ex.R` ranks.

    Produces `logical_weights` ready for `pack_inputs(...)` and `reference_step(...)`,
    including EPLB redundant expert replication (`ew > er`) and `ecopy` metadata.
    """
    g.check(ex)
    R = ex.R
    L, H, E, inter_dim, IS = g.L, g.H, g.E, g.I, g.IS
    HD = g.HD
    nh = g.nh(R)
    isr, ir, er, ew = g.ISR(R), g.IR(R), g.ER(R), g.EW(R)
    natt, ngdn = g.NATT, g.NGDN
    G = g.G(R)
    nkv = g.NKV
    koff = nh * 2 * HD

    # 1. Attention projections: [q_proj (NH*2*HD) | k_proj (NKV,HD) | v_proj (NKV,HD)]
    q_p = jnp.transpose(hf["q_proj"].reshape(natt, H, R, koff), (2, 0, 1, 3))
    q_s = jnp.transpose(hf["q_proj_s"].reshape(natt, R, koff), (1, 0, 2))

    k_in = (
        hf["k_proj"].reshape(natt, nkv, H, HD)
        if hf["k_proj"].ndim == 3
        else hf["k_proj"]
    )
    v_in = (
        hf["v_proj"].reshape(natt, nkv, H, HD)
        if hf["v_proj"].ndim == 3
        else hf["v_proj"]
    )
    ks_in = (
        hf["k_proj_s"].reshape(natt, nkv, HD)
        if hf["k_proj_s"].ndim == 2
        else hf["k_proj_s"]
    )
    vs_in = (
        hf["v_proj_s"].reshape(natt, nkv, HD)
        if hf["v_proj_s"].ndim == 2
        else hf["v_proj_s"]
    )
    kv_p = jnp.repeat(
        jnp.transpose(jnp.concatenate([k_in, v_in], axis=-1), (1, 0, 2, 3)),
        G,
        axis=0,
    )
    kv_s = jnp.repeat(
        jnp.transpose(jnp.concatenate([ks_in, vs_in], axis=-1), (1, 0, 2)),
        G,
        axis=0,
    )
    att_in = jnp.concatenate([q_p, kv_p], axis=-1)
    att_in_s = jnp.concatenate([q_s, kv_s], axis=-1)

    att_out = jnp.transpose(hf["att_out"].reshape(natt, R, nh * HD, H), (1, 0, 2, 3))
    att_out_s = jnp.broadcast_to(hf["att_out_s"][None, ...], (R, natt, H))

    qn_in = hf["qn"]
    if qn_in.ndim == 3:
        qn = jnp.transpose(qn_in.reshape(natt, R, nh, HD), (1, 0, 2, 3)).mean(axis=2)
    else:
        qn = jnp.broadcast_to(qn_in[None, ...], (R, natt, HD))
    kn_in = hf["kn"].reshape(natt, nkv, HD) if hf["kn"].ndim == 2 else hf["kn"]
    kn = jnp.repeat(jnp.transpose(kn_in, (1, 0, 2)), G, axis=0)

    # 2. Shared expert (always TP-sharded along IS across R)
    sh_gate = jnp.broadcast_to(hf["sh_gate"][None, ...], (R, L, H))
    if "sh_gate_proj" in hf and "sh_up_proj" in hf:
        sh_g_in, sh_u_in = hf["sh_gate_proj"], hf["sh_up_proj"]
        sh_gs_in, sh_us_in = hf["sh_gate_proj_s"], hf["sh_up_proj_s"]
    else:
        sh_g_in, sh_u_in = hf["sh13"][:, :, :IS], hf["sh13"][:, :, IS:]
        sh_gs_in, sh_us_in = hf["sh13_s"][:, :IS], hf["sh13_s"][:, IS:]
    sh_g = jnp.transpose(sh_g_in.reshape(L, H, R, isr), (2, 0, 1, 3))
    sh_u = jnp.transpose(sh_u_in.reshape(L, H, R, isr), (2, 0, 1, 3))
    sh13 = jnp.concatenate([sh_g, sh_u], axis=-1)

    sh_gs = jnp.transpose(sh_gs_in.reshape(L, R, isr), (1, 0, 2))
    sh_us = jnp.transpose(sh_us_in.reshape(L, R, isr), (1, 0, 2))
    sh13_s = jnp.concatenate([sh_gs, sh_us], axis=-1)

    sh2 = jnp.transpose(hf["sh2"].reshape(L, R, isr, H), (1, 0, 2, 3))
    sh2_s = jnp.broadcast_to(hf["sh2_s"][None, ...], (R, L, H))

    # 3. Routed experts: TP-sharded (g.moe_tp=True) or EP + EPLB (g.moe_tp=False)
    if "e_gate_proj" in hf and "e_up_proj" in hf:
        e_g_in, e_u_in = hf["e_gate_proj"], hf["e_up_proj"]
        e_gs_in, e_us_in = hf["e_gate_proj_s"], hf["e_up_proj_s"]
        e13_full = jnp.concatenate([e_g_in, e_u_in], axis=-1)
        e13_s_full = jnp.concatenate([e_gs_in, e_us_in], axis=-1)
    else:
        e13_full = hf["e13"]
        e13_s_full = hf["e13_s"]
        e_g_in, e_u_in = e13_full[:, :, :, :inter_dim], e13_full[:, :, :, inter_dim:]
        e_gs_in, e_us_in = (
            e13_s_full[:, :, :inter_dim],
            e13_s_full[:, :, inter_dim:],
        )

    if g.moe_tp:
        e13_g = jnp.transpose(e_g_in.reshape(L, E, H, R, ir), (3, 0, 1, 2, 4))
        e13_u = jnp.transpose(e_u_in.reshape(L, E, H, R, ir), (3, 0, 1, 2, 4))
        e13 = jnp.concatenate([e13_g, e13_u], axis=-1)

        e13_sg = jnp.transpose(e_gs_in.reshape(L, E, R, ir), (2, 0, 1, 3))
        e13_su = jnp.transpose(e_us_in.reshape(L, E, R, ir), (2, 0, 1, 3))
        e13_s = jnp.concatenate([e13_sg, e13_su], axis=-1)

        e2 = jnp.transpose(hf["e2"].reshape(L, E, R, ir, H), (2, 0, 1, 3, 4))
        e2_s = jnp.broadcast_to(hf["e2_s"][None, ...], (R, L, E, H))
    elif ew == er:
        e13 = jnp.transpose(
            e13_full.reshape(L, R, er, H, 2 * inter_dim), (1, 0, 2, 3, 4)
        )
        e13_s = jnp.transpose(e13_s_full.reshape(L, R, er, 2 * inter_dim), (1, 0, 2, 3))
        e2 = jnp.transpose(hf["e2"].reshape(L, R, er, inter_dim, H), (1, 0, 2, 3, 4))
        e2_s = jnp.transpose(hf["e2_s"].reshape(L, R, er, H), (1, 0, 2, 3))
    else:
        if e_cores is None:
            e_cores = default_eplb_cores(g, R)
        idx = jnp.maximum(jnp.asarray(e_cores, dtype=I32), 0)  # [L, R, EW]
        l_idx = jnp.arange(L, dtype=I32)[:, None, None]
        e13 = jnp.transpose(e13_full[l_idx, idx], (1, 0, 2, 3, 4))
        e13_s = jnp.transpose(e13_s_full[l_idx, idx], (1, 0, 2, 3))
        e2 = jnp.transpose(hf["e2"][l_idx, idx], (1, 0, 2, 3, 4))
        e2_s = jnp.transpose(hf["e2_s"][l_idx, idx], (1, 0, 2, 3))

    logical_weights: dict[str, Any] = {
        "in_norm": hf["in_norm"],
        "post_norm": hf["post_norm"],
        "final_norm": hf["final_norm"],
        "att_in": att_in,
        "att_in_s": att_in_s,
        "att_out": att_out,
        "att_out_s": att_out_s,
        "qn": qn,
        "kn": kn,
        "router": hf["router"],
        "sh_gate": sh_gate,
        "sh13": sh13,
        "sh13_s": sh13_s,
        "sh2": sh2,
        "sh2_s": sh2_s,
        "e13": e13,
        "e13_s": e13_s,
        "e2": e2,
        "e2_s": e2_s,
    }
    if not g.moe_tp and ew > er:
        assert e_cores is not None
        logical_weights["e_cores"] = e_cores
        logical_weights["ecopy"] = encode_eplb_ecopy(g, R, e_cores)

    # 4. GDN linear attention: [qkv | z | b(64) | a(64)] (NLI=BA+128, A_OFF=BA+64)
    if ngdn > 0:
        lay_gdn = GDNModule.lin_in_layout(g, ex)
        kh, vh, KD, VD = lay_gdn.kh, lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
        b_pad = lay_gdn.A_OFF - lay_gdn.BA
        a_pad = lay_gdn.NLI - lay_gdn.A_OFF
        kq_w = kh * KD
        v_w = vh * VD

        lin_q = jnp.transpose(hf["lin_q"].reshape(ngdn, H, R, kq_w), (2, 0, 1, 3))
        lin_k = jnp.transpose(hf["lin_k"].reshape(ngdn, H, R, kq_w), (2, 0, 1, 3))
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

        lin_q_s = jnp.transpose(hf["lin_q_s"].reshape(ngdn, R, kq_w), (1, 0, 2))
        lin_k_s = jnp.transpose(hf["lin_k_s"].reshape(ngdn, R, kq_w), (1, 0, 2))
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

        c_full = hf["conv"]  # [ngdn, 4, 2*KH*KD + VH*VD]
        KH_KD = g.KH * KD
        c_q = jnp.transpose(
            c_full[:, :, :KH_KD].reshape(ngdn, 4, R, kq_w), (2, 0, 1, 3)
        )
        c_k = jnp.transpose(
            c_full[:, :, KH_KD : 2 * KH_KD].reshape(ngdn, 4, R, kq_w), (2, 0, 1, 3)
        )
        c_v = jnp.transpose(
            c_full[:, :, 2 * KH_KD :].reshape(ngdn, 4, R, v_w), (2, 0, 1, 3)
        )
        conv = jnp.concatenate([c_q, c_k, c_v], axis=-1).reshape(
            R, ngdn, 4 * lay_gdn.QKV
        )

        alog = jnp.transpose(hf["alog"].reshape(ngdn, R, vh), (1, 0, 2))
        dtb = jnp.transpose(hf["dtb"].reshape(ngdn, R, vh), (1, 0, 2))
        gnorm = jnp.broadcast_to(hf["gnorm"][None, ...], (R, ngdn, VD))

        for k, v in (
            ("lin_in", lin_in),
            ("lin_in_s", lin_in_s),
            ("lin_out", lin_out),
            ("lin_out_s", lin_out_s),
            ("conv", conv),
            ("alog", alog),
            ("dtb", dtb),
            ("gnorm", gnorm),
        ):
            logical_weights[k] = v

    return logical_weights


__all__ = [
    "default_eplb_cores",
    "encode_eplb_ecopy",
    "random_hf_weights",
    "split_hf_weights",
]
