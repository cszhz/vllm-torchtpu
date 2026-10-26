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
"""Qwen3.5 logical input generation, megakernel packing, and verify accept."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .. import kernel
from ..comm.replication import replicate_groups
from ..exec_spec import ExecSpec
from ..modules.gdn import GDNModule
from ..ops import BF16, F32, FP8, I32, rup
from ..pool.config import POOL_STATE_NAME, cache_exec, resolved_max_pages_per_seq
from ..weights import default_eplb_cores, encode_eplb_ecopy, pack_logical_weights
from . import qwen35_pool


def accept_states(g: Any, ex: ExecSpec, st: dict[str, Any], acc: Any) -> dict[str, Any]:
    """Delegate verify-mode GDN state selection to `GDNModule.accept`."""
    return GDNModule.accept(st, st, acc, T=ex.T, vh=g.vh(ex.R))


def replicate_logical_inputs(
    g: Any,
    ex: ExecSpec,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
) -> None:
    """Copy shared key-head and shared-expert slices across replica ranks
    in synthetic inputs.
    """
    R = ex.R
    sh_rep = g.sh_rep(R)
    for k in ("sh13", "sh13_s", "sh2"):
        w_k = logical_weights.get(k)
        if w_k is not None and sh_rep > 1:
            logical_weights[k] = (
                tuple(replicate_groups(part, sh_rep) for part in w_k)
                if isinstance(w_k, tuple)
                else replicate_groups(w_k, sh_rep)
            )
    kh_rep = g.kh_rep(R) if g.NGDN > 0 else 1
    if kh_rep <= 1:
        return
    lay = GDNModule.lin_in_layout(g, ex)
    qk = 2 * lay.kh * lay.KD
    for k in ("lin_in", "lin_in_s"):
        if k in logical_weights:
            arr = logical_weights[k]
            rep_qk = replicate_groups(arr[..., :qk], kh_rep)
            logical_weights[k] = arr.at[..., :qk].set(rep_qk)
    if "conv" in logical_weights:
        cw = logical_weights["conv"].reshape(R, g.NGDN, 4, lay.QKV)
        cw = cw.at[..., :qk].set(replicate_groups(cw[..., :qk], kh_rep))
        logical_weights["conv"] = cw.reshape(R, g.NGDN, 4 * lay.QKV)
    if "conv" in logical_states:
        cs = logical_states["conv"]
        logical_states["conv"] = cs.at[..., :qk].set(
            replicate_groups(cs[..., :qk], kh_rep)
        )


def random_logical_inputs(
    key: jax.Array,
    g: Any,
    ex: ExecSpec,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Generate small random logical inputs for `reference_step` and `pack_inputs`."""
    g.check(ex)
    R, B, T, NS, ctx = ex.R, ex.B, ex.T, ex.NS, ex.ctx
    L, H, E, HD = g.L, g.H, g.E, g.HD
    nh = g.nh(R)
    lay_gdn = GDNModule.lin_in_layout(g, ex)
    vh, KD, VD = lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
    qkv, z, nli = lay_gdn.QKV, lay_gdn.Z, lay_gdn.NLI
    isr, ir, er, ew = g.ISR(R), g.IR(R), g.ER(R), g.EW(R)
    natt, ngdn = g.NATT, g.NGDN

    keys = jax.random.split(key, 36)
    it = iter(keys)

    base_pos = jnp.arange(1, NS + 1, dtype=I32) * min(16, ctx // 4)
    if T == 1:
        pos = base_pos
    else:
        pos = (base_pos[:, None] + jnp.arange(T, dtype=I32)[None, :]).reshape(B)

    hidden = jax.random.normal(next(it), (B, H), dtype=BF16) * 0.1

    def rfp8(shape):
        return (jax.random.normal(next(it), shape, dtype=F32) * 0.25).astype(FP8)

    def rscale(shape):
        return jnp.abs(jax.random.normal(next(it), shape, dtype=F32)) * 0.02 + 0.01

    G, nkv, koff = g.G(R), g.NKV, nh * 2 * HD
    att_q = rfp8((R, natt, H, koff))
    att_kv = jnp.repeat(rfp8((nkv, natt, H, 2 * HD)), G, axis=0)
    att_in = jnp.concatenate([att_q, att_kv], axis=-1)
    att_q_s = rscale((R, natt, koff))
    att_kv_s = jnp.repeat(rscale((nkv, natt, 2 * HD)), G, axis=0)
    att_in_s = jnp.concatenate([att_q_s, att_kv_s], axis=-1)

    kn = jnp.repeat(
        jax.random.normal(next(it), (nkv, natt, HD), dtype=F32) * 0.05, G, axis=0
    )
    sh_gate = jnp.broadcast_to(
        (jax.random.normal(next(it), (L, H), dtype=F32) * 0.1)[None, :, :],
        (R, L, H),
    )

    e13 = rfp8((R, L, ew, H, 2 * ir))
    e13_s = rscale((R, L, ew, 2 * ir))
    e2 = rfp8((R, L, ew, ir, H))
    e2_s = rscale((R, L, ew, H))

    logical_weights: dict[str, Any] = {
        "in_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "post_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "final_norm": jax.random.normal(next(it), (H,), dtype=F32) * 0.05,
        "att_in": att_in,
        "att_in_s": att_in_s,
        "att_out": rfp8((R, natt, nh * HD, H)),
        "att_out_s": rscale((R, natt, H)),
        "qn": jax.random.normal(next(it), (R, natt, HD), dtype=F32) * 0.05,
        "kn": kn,
        "router": jax.random.normal(next(it), (L, H, E), dtype=BF16) * 0.1,
        "sh_gate": sh_gate,
        "sh13": rfp8((R, L, H, 2 * isr)),
        "sh13_s": rscale((R, L, 2 * isr)),
        "sh2": rfp8((R, L, isr, H)),
        "sh2_s": rscale((R, L, H)),
        "e13": e13,
        "e13_s": e13_s,
        "e2": e2,
        "e2_s": e2_s,
    }
    if not g.moe_tp and ew > er and R > 1:
        e_cores = default_eplb_cores(g, R)
        for lyr in range(L):
            for r1 in range(R):
                for k_idx, g_id in enumerate(e_cores[lyr][r1][er:ew]):
                    r0, s0 = divmod(int(g_id), er)
                    slot = er + k_idx
                    e13 = e13.at[r1, lyr, slot].set(e13[r0, lyr, s0])
                    e13_s = e13_s.at[r1, lyr, slot].set(e13_s[r0, lyr, s0])
                    e2 = e2.at[r1, lyr, slot].set(e2[r0, lyr, s0])
                    e2_s = e2_s.at[r1, lyr, slot].set(e2_s[r0, lyr, s0])
        logical_weights["e13"] = e13
        logical_weights["e13_s"] = e13_s
        logical_weights["e2"] = e2
        logical_weights["e2_s"] = e2_s
        logical_weights["e_cores"] = e_cores
        logical_weights["ecopy"] = encode_eplb_ecopy(g, R, e_cores)

    if ngdn > 0:
        logical_weights.update(
            {
                "lin_in": rfp8((R, ngdn, H, nli)),
                "lin_in_s": rscale((R, ngdn, nli)),
                "lin_out": rfp8((R, ngdn, z, H)),
                "lin_out_s": rscale((R, ngdn, H)),
                "conv": jax.random.normal(next(it), (R, ngdn, 4 * qkv), dtype=F32)
                * 0.05,
                "alog": jax.random.normal(next(it), (R, ngdn, vh), dtype=F32) * 0.1
                - 1.0,
                "dtb": jax.random.normal(next(it), (R, ngdn, vh), dtype=F32) * 0.1,
                "gnorm": jnp.ones((R, ngdn, VD), dtype=F32)
                + jax.random.normal(next(it), (R, ngdn, VD), dtype=F32) * 0.05,
            }
        )

    kc_nkv = jax.random.normal(next(it), (nkv, natt, NS, ctx, HD), dtype=BF16) * 0.1
    vc_nkv = jax.random.normal(next(it), (nkv, natt, NS, ctx, HD), dtype=BF16) * 0.1
    logical_states: dict[str, Any] = {
        "kc": jnp.repeat(kc_nkv, G, axis=0),
        "vc": jnp.repeat(vc_nkv, G, axis=0),
    }
    if ngdn > 0:
        logical_states["conv"] = (
            jax.random.normal(next(it), (R, ngdn, NS, 3, qkv), dtype=F32) * 0.05
        )
        logical_states["rec"] = (
            jax.random.normal(next(it), (R, ngdn, NS, vh, KD, VD), dtype=BF16) * 0.05
        )
    replicate_logical_inputs(g, ex, logical_weights, logical_states)
    return pos, hidden, logical_weights, logical_states


def pack_inputs(
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
    g: Any,
    ex: ExecSpec,
    *,
    mesh: Any = None,
    build_default_pool: bool = False,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Pack logical model weights and states into the tiled/folded megakernel layout."""
    R, B, T, NS, M = ex.R, ex.B, ex.T, ex.NS, ex.M
    ngdn = g.NGDN

    hid_pad = jnp.pad(hidden[:B], ((0, M - B), (0, 0)))
    hid_mk = jnp.tile(hid_pad[None, :, :], (R, 1, 1)).reshape(R * M, g.H)
    weights_mk = pack_logical_weights(g._model_def, ex, logical_weights, g=g)

    states_mk: dict[str, Any] = {}
    if ngdn > 0 and "conv" in logical_states and "rec" in logical_states:
        conv_l = logical_states["conv"]
        rec_l = logical_states["rec"]
        qkv, vh = g.QKV(R), g.vh(R)
        if conv_l.ndim == 5 and conv_l.shape[3] == 3:
            if T == 1:
                cr = rup(3 * B, 8)
                conv_3b = jnp.swapaxes(conv_l, 2, 3).reshape(R, ngdn, 3 * B, qkv)
                conv_p = jnp.pad(conv_3b, ((0, 0), (0, 0), (0, cr - 3 * B), (0, 0)))
            else:
                hr = rup(3 + T, 8)
                conv_p = jnp.pad(conv_l, ((0, 0), (0, 0), (0, 0), (0, hr - 3), (0, 0)))
        else:
            conv_p = conv_l
        rec_p = (
            rec_l.reshape(R, ngdn, NS * vh, g.KD, g.VD) if rec_l.ndim == 6 else rec_l
        )
        states_mk["conv"] = kernel.fold(conv_p.astype(F32))
        states_mk["rec"] = kernel.fold(rec_p.astype(BF16))

    if cache_exec(ex).paged_kv:
        if POOL_STATE_NAME in logical_states and "page_table" in logical_states:
            states_mk[POOL_STATE_NAME] = tuple(
                kernel.fold(r) if r.ndim == 6 else r
                for r in logical_states[POOL_STATE_NAME]
            )
            pt_in = logical_states["page_table"]
            max_pt = resolved_max_pages_per_seq(ex)
            if pt_in.ndim == 2:
                if pt_in.shape[0] not in (NS, R * NS) or pt_in.shape[1] < max_pt:
                    raise ValueError(
                        f"page_table has shape {pt_in.shape}, expected "
                        f"({NS}, >={max_pt}) or ({R * NS}, >={max_pt})"
                    )
                pt_in = pt_in[:, :max_pt]
                states_mk["page_table"] = (
                    pt_in if pt_in.shape[0] == R * NS else jnp.tile(pt_in, (R, 1))
                )
            elif pt_in.ndim == 3:
                if pt_in.shape[:2] != (R, NS) or pt_in.shape[2] < max_pt:
                    raise ValueError(
                        f"page_table has shape {pt_in.shape}, expected "
                        f"({R}, {NS}, >={max_pt})"
                    )
                states_mk["page_table"] = kernel.fold(pt_in[:, :, :max_pt])
            else:
                raise ValueError(
                    f"page_table must be 2D or 3D, got shape {pt_in.shape}"
                )
        elif build_default_pool:
            pt, ckpt, roff, _ = qwen35_pool.default_pool_layout(g, ex)
            states_mk[POOL_STATE_NAME] = qwen35_pool.states_to_pool(
                g, ex, logical_states, pt, ckpt, roff
            )
            states_mk["page_table"] = jnp.tile(
                pt[:, : resolved_max_pages_per_seq(ex)], (R, 1)
            )
        else:
            raise ValueError(
                f"pack_inputs with paged_kv=True requires {POOL_STATE_NAME!r} "
                "and 'page_table' in logical_states (or pass build_default_pool=True)."
            )
    elif "kc" in logical_states and "vc" in logical_states:
        states_mk["kc"] = kernel.fold(logical_states["kc"])
        states_mk["vc"] = kernel.fold(logical_states["vc"])

    if mesh is not None:
        rep = NamedSharding(mesh, P())
        tp = NamedSharding(mesh, P("tp"))
        pos = jax.device_put(pos, rep)
        hid_mk = jax.device_put(hid_mk, tp)
        weights_mk = {
            k: tuple(jax.device_put(x, tp) for x in v)
            if isinstance(v, tuple)
            else jax.device_put(v, tp)
            for k, v in weights_mk.items()
        }
        states_mk = {
            k: tuple(jax.device_put(x, tp) for x in v)
            if isinstance(v, tuple)
            else jax.device_put(v, tp)
            for k, v in states_mk.items()
        }

    return pos, hid_mk, weights_mk, states_mk


def random_inputs(
    key: jax.Array, g: Any, ex: ExecSpec, *, mesh: Any = None
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Generate packed megakernel inputs `(pos, hidden, weights, states)`."""
    pos, hid_l, w_l, st_l = random_logical_inputs(key, g, ex)
    return pack_inputs(
        pos,
        hid_l,
        w_l,
        st_l,
        g,
        ex,
        mesh=mesh,
        build_default_pool=cache_exec(ex).paged_kv,
    )


__all__ = ["accept_states", "pack_inputs", "random_inputs", "random_logical_inputs"]
