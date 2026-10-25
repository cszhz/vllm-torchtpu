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
"""Plain-JAX composition of the whole-step kernel (L4e) for numerics checks.

Simulates every rank of the mesh on one host: the replicated dense chain on
the T global rows, each rank's sharded slices (wq_b heads, wo_a / wo_b o-group,
shared-expert intermediate slice), the all-reduces as allreduce's model, and
the stub phases exactly as step_kernel stubs them. `rank_weights` slices the
natural full-model weights into what rank r's packed operands hold.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4_megakernel import allreduce as AR
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_reference as DREF
from vllm_torchtpu.kernels.deepseek_v4_megakernel.dense_config import DenseChainConfig

F32, BF16 = jnp.float32, jnp.bfloat16
LANE = 128


def rank_weights(w: dict, full: DenseChainConfig, r: int, R: int) -> dict:
    """Natural weights of rank r: the model's arrays with the sharded
    projections cut to rank r's slice (the inputs of dense_weights.pack_*)."""
    d, hpg = full.head_dim, full.heads_per_group
    out = dict(w)
    hq = full.n_heads // R * d  # wq_b columns per rank
    out["wq_b"] = w["wq_b"][:, :, r * hq : (r + 1) * hq]
    out["wq_b_scale"] = w["wq_b_scale"][:, r * hq : (r + 1) * hq]
    go = full.o_lora * (full.n_heads // R // hpg)
    out["wo_a"] = w["wo_a"][:, :, r * go : (r + 1) * go]
    out["wo_a_scale"] = w["wo_a_scale"][:, r * go : (r + 1) * go]
    out["wo_b"] = w["wo_b"][:, r * go : (r + 1) * go, :]
    inter, ir = full.moe_inter, full.moe_inter // R
    cols = np.r_[r * ir : (r + 1) * ir, inter + r * ir : inter + (r + 1) * ir]
    out["w13"] = w["w13"][:, :, cols]
    out["w13_scale"] = w["w13_scale"][:, cols]
    out["w2"] = w["w2"][:, r * ir : (r + 1) * ir, :]
    return out


def global_rows(q, R, T, ids, pos, seq_lens, attn_tp=1):
    """numpy model of step_kernel.step_metadata: ids/pos [R, M], seq_lens
    [R, B] -> (tid, tpos, owner group, lrow [T], own_rows: list per rank of
    the global rows of its own rows k = r*q + t). Each attention group's batch
    is its first rank's."""
    QP = -(-q // 8) * 8
    nreq = (np.asarray(seq_lens) > 0).sum(1)
    ids, pos = np.asarray(ids)[::attn_tp], np.asarray(pos)[::attn_tp]
    nreq = nreq[::attn_tp]
    tid = np.zeros(T, np.int32)
    tpos = np.zeros(T, np.int32)
    owner = -np.ones(T, np.int32)
    lrow = -np.ones(T, np.int32)
    own_rows = []
    gr = 0
    for r in range(R // attn_tp):
        rows = []
        for j in range(nreq[r]):
            for t in range(q):
                g = gr * QP + t
                k = j * q + t
                tid[g], tpos[g], owner[g], lrow[g] = ids[r][k], pos[r][k], r, k
                rows.append(g)
            gr += 1
        own_rows += [rows] * attn_tp
    assert gr * QP <= T
    return tid, tpos, owner, lrow, own_rows


@dataclasses.dataclass(frozen=True)
class RefConfig:
    full: DenseChainConfig  # tp = 1 geometry (all heads)
    R: int
    T: int
    q: int
    aux_layers: tuple[int, ...]
    ar_mode: str
    ar_group: int
    attn_tp: int = 1  # step_kernel.StepKernelConfig.attn_tp


def _qnorm_rope(x, cs, eps):
    """deepseek_v4_qnorm_rope on [T, heads, d] with the [T, 256] table."""
    d = x.shape[-1]
    xf = x.astype(F32)
    y = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    tail = y[..., d - LANE :]
    cos, sin = cs[:, None, :LANE], cs[:, None, LANE:]
    roped = tail * cos + DREF.rotate_pairs(tail) * sin
    return jnp.concatenate([y[..., : d - LANE], roped], axis=-1).astype(BF16)


def step_reference(rc: RefConfig, w, embed, cos_sin_cache, ids, pos, seq_lens,
                   order: str = "stock", moe=None, attn=None, trace=None,
                   ctx=None):  # fmt: skip
    """Returns (res_final [R, M, 4H], aux [R, A, M, 4H]) as the kernel's
    own-row outputs (zero rows for inactive own rows).

    moe: None (MoE stub: shared expert only) or dict(cfg=MoEConfig,
    weights=[(w1, w1s, w2, w2s) logical, per layer], bias=[L, E],
    tid2eid=[num_hash x [V, 6]]) for the routed experts (moe_reference's model
    of the MoE phase arithmetic).
    attn: None (stub) or dict(sets=[per-rank cache LayerSet of
    attn_phase_harness], reference=attn_phase_harness.reference) for the real
    attention phase on every owner's request (cache updates threaded through
    the layers). Returns the final cache sets as a third value.
    ctx: the DSpark draft step's context rows, dict(main_x [T, H], positions
    [T] (-1: none)), projected per SWA layer as stock
    precompute_and_store_context_kv and inserted by its attention."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK

    full, R, T, q = rc.full, rc.R, rc.T, rc.q
    M = np.asarray(ids).shape[1]
    tid, tpos, owner, lrow, own_rows = global_rows(
        q, R, T, ids, pos, seq_lens, rc.attn_tp
    )
    live = owner >= 0
    # mesh rank r holds head block hb[r]; per-rank lists stay in mesh order
    hb = [SK.head_block(r, R, rc.attn_tp) for r in range(R)]
    if attn is not None:
        attn["attn_tp"] = rc.attn_tp  # a group's ranks hold one cache replica
    rw = [rank_weights(w, full, hb[r], R) for r in range(R)]
    cs_f = SK.rope_rows(cos_sin_cache, jnp.asarray(tpos), inverse=False)
    cs_i = SK.rope_rows(cos_sin_cache, jnp.asarray(tpos), inverse=True)
    h, hcm, eps = full.hidden, full.hc_mult, full.rms_eps
    x0 = jnp.asarray(embed)[jnp.asarray(tid)].astype(BF16)
    res = jnp.concatenate([x0] * hcm, axis=1)
    x = jnp.zeros((T, h), BF16)
    post = jnp.zeros((T, hcm), F32)
    comb = jnp.tile(jnp.eye(hcm, dtype=F32).reshape(1, -1), (T, 1))
    tp_cfg = dataclasses.replace(full, tp=R, rows=T)
    mode = rc.ar_mode if rc.ar_group <= R else "flat"
    group = min(rc.ar_group, R)
    st = tp_cfg.streams
    kernel_order = order == "kernel"
    mhc = DREF.mhc_fused_kernel_order if kernel_order else DREF.mhc_fused_jnp
    # "stock_kernels" (TPU): the stock Pallas mHC and wo_a kernels, as serving
    wo_a = DREF.wo_a_stock if order == "stock_kernels" else DREF.wo_a_jnp
    if order == "stock_kernels":
        mhc = DREF.mhc_fused_stock
    norm = DREF.rms_norm_kernel_order if kernel_order else DREF.rms_norm

    def lin(xv, lw, name, sname=None):
        sname = sname or name
        if kernel_order:
            return DREF.fp8_linear_kernel_order(
                xv, lw[sname], lw[sname + "_scale"], st[name].tile_rows
            )
        return DREF.xla_quantized_matmul(xv, lw[sname], lw[sname + "_scale"])

    def allreduce(parts):
        return AR.reference_allreduce(
            jnp.stack([p.astype(F32) for p in parts]), mode, R, group
        ).astype(BF16)

    def post_only(x, res, post, comb):
        return DREF.mhc_post_only(x, res, post, comb, hcm, h)

    o_cfg = dataclasses.replace(full, tp=R)  # one o-group per rank
    live_j = jnp.asarray(live)[:, None]
    aux = []
    for li, lt in enumerate(full.layer_types):
        lws = [DREF.layer_weights(rw_r, full, li) for rw_r in rw]
        lw = lws[0]  # replicated parts are identical on every rank
        res, post, comb, hh = mhc(
            x, res, post, comb, lw["hc_attn_fn"], lw["hc_attn_scale"],
            lw["hc_attn_base"], full,
        )  # fmt: skip
        xa = norm(hh, lw["attn_norm"], eps)
        qkv = lin(xa, lw, "wqa")
        qr = norm(qkv[:, : full.q_lora], lw["q_norm"], eps)
        ctxkv = None
        if attn is not None:
            d = full.head_dim

            def kv_rope(kv, cs):
                kvn = norm(kv, lw["kv_norm"], eps).astype(F32)
                tail = kvn[:, d - LANE :]
                roped = tail * cs[:, :LANE] + DREF.rotate_pairs(tail) * cs[:, LANE:]
                return jnp.concatenate([kvn[:, : d - LANE], roped], -1).astype(BF16)

            kvr = kv_rope(qkv[:, full.q_lora :], cs_f)
            if lt == 2 and ctx is not None:  # SWA draft layer: the context kv
                kvc = lin(ctx["main_x"], lw, "wqa_ctx", "wqa")[:, full.q_lora :]
                cpos = jnp.maximum(jnp.asarray(ctx["positions"]), 0)
                ctxkv = kv_rope(kvc, SK.rope_rows(cos_sin_cache, cpos, inverse=False))

            def bdot(w, name):
                if kernel_order:
                    return DREF.ktiled_dot(xa, w, st[name].tile_rows)
                return jnp.dot(xa, w, preferred_element_type=F32)

            if lt == 2:  # SWA: no compressor / indexer
                xp = idxq = None
            elif lt == 1:  # CSA
                pad = jnp.zeros((full.hidden, full.wp_cols - full.idx_heads), BF16)
                wcat = jnp.concatenate(
                    [lw["wp"], pad, lw["idx_comp"], lw["comp_csa"]], axis=-1
                )
                xp = bdot(wcat, "xproj_csa")
                idxq = lin(qr, lw, "idx_wq_b")
            else:
                xp = bdot(lw["comp_hca"], "xproj_hca")
                idxq = None
        heads = []
        for b in range(R):  # head block order
            qrk = lin(qr, lws[hb.index(b)], "wq_b")
            heads.append(qrk.reshape(T, -1, full.head_dim))
        qall = _qnorm_rope(jnp.concatenate(heads, axis=1), cs_f, eps)
        if attn is None:  # attention stub: per-row (head) RMSNorm of q
            qf = qall.astype(F32)
            o = (qf * jax.lax.rsqrt(jnp.mean(qf * qf, -1, keepdims=True) + eps)).astype(
                BF16
            )
        else:
            o = _attention(attn, li, lt, full, q, qall, kvr, xp, idxq, owner, lrow, T,
                           ctxkv)  # fmt: skip
        o = jnp.where(live_j[:, :, None], o, 0)
        if trace is not None and li == 0:
            trace.update(qall=qall, o=o, own_rows=own_rows)
        parts = []
        hpr = full.n_heads // R
        for r in range(R):
            og = o[:, hb[r] * hpr : (hb[r] + 1) * hpr, :].reshape(T, -1)
            z = wo_a(og, lws[r]["wo_a"], lws[r]["wo_a_scale"], cs_i, o_cfg)
            parts.append(lin(z, lws[r], "wo_b"))
        x = allreduce(parts)
        if trace is not None and li == 0:
            trace.update(wo_b_parts=parts, x_att=x)
        res, post, comb, hh = mhc(
            x, res, post, comb, lw["hc_ffn_fn"], lw["hc_ffn_scale"],
            lw["hc_ffn_base"], full,
        )  # fmt: skip
        xf = norm(hh, lw["ffn_norm"], eps)
        parts = []
        for r in range(R):
            gu = lin(xf, lws[r], "w13")
            act = DREF.swiglu_clamp(gu, full.swiglu_limit)
            parts.append(lin(act, lws[r], "w2").astype(F32))
        if moe is not None:
            parts = _routed(moe, li, xf, lw, st, kernel_order, tid, live, parts)
        x = allreduce(parts)
        if trace is not None and li == 0:
            trace.update(x_ffn=x, ffn_parts=parts)
        if li in rc.aux_layers:
            aux.append(post_only(x, res, post, comb))
        del lt
    final = post_only(x, res, post, comb)

    def own(buf):
        out = np.zeros((R, M, buf.shape[1]), np.float32)
        b = np.asarray(buf.astype(F32))
        for r in range(R):
            out[r, : len(own_rows[r])] = b[own_rows[r]]
        return out

    res_out = own(final)
    aux_out = np.stack([own(a) for a in aux], axis=1) if aux else None
    if attn is not None:
        return res_out, aux_out, attn["sets"]
    return res_out, aux_out


def _routed(moe, li, xf, lw, st, kernel_order, tid, live, parts):
    """Each rank's routed-expert partial added onto its shared partial; rank
    r holds the experts of EP block moe["expert_perm"][r] (default r)."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import moe_reference as MR

    cfg = moe["cfg"]
    T = xf.shape[0]
    if kernel_order:
        logits = DREF.ktiled_dot(xf, lw["gate"], st["gate"].tile_rows)
    else:
        logits = jnp.dot(xf, lw["gate"], preferred_element_type=F32)
    # padding rows (not a prefix any more): -inf logits, as the phase's
    # row_valid mask
    logits = jnp.where(jnp.asarray(live)[:, None], logits, -jnp.inf)
    if li < len(moe["tid2eid"]):
        hid = jnp.asarray(moe["tid2eid"][li])[jnp.asarray(tid)]
        ids, wts = MR.stock_route(logits, cfg, T, hash_ids=hid)
    else:
        ids, wts = MR.stock_route(logits, cfg, T, bias=moe["bias"][li])
    x32 = jnp.zeros((cfg.rows, xf.shape[1]), BF16).at[:T].set(xf)
    ids32 = jnp.full((cfg.rows, cfg.topk), cfg.n_experts, jnp.int32).at[:T].set(ids)
    wts32 = jnp.zeros((cfg.rows, cfg.topk), F32).at[:T].set(wts)
    perm = moe.get("expert_perm", range(len(parts)))
    held = moe.get("held")  # int32 [R, L, slots] (moe_phase.placement_tables)
    return [
        MR.model_partial(
            x32, ids32, wts32, *moe["weights"][li], cfg, b,
            None if held is None else np.asarray(held)[r, li],
        )[:T] + p
        for r, (b, p) in enumerate(zip(perm, parts))
    ]  # fmt: skip


def _attention(attn, li, lt, full, q, qall, kvr, xp, idxq, owner, lrow, T,
               ctxkv=None):  # fmt: skip
    """Real attention for every owner request (block reference), as
    step_kernel gathers it: own slot j of rank r = global rows gr*8 .. +8."""
    NH, d = full.n_heads, full.head_dim
    QP = 8
    RP = -(-q * NH // LANE) * LANE
    kind = {0: "hca", 1: "csa", 2: "swa"}[lt]
    o = jnp.zeros((T, NH, d), BF16)
    owner = np.asarray(owner)
    lrow = np.asarray(lrow)
    for gr in range(T // QP):
        g = int(owner[gr * QP])
        if g < 0:
            continue
        gs = attn.get("attn_tp", 1)
        r = g * gs  # the group's first rank: its cache set is the replica
        slot = int(lrow[gr * QP]) // q
        assert slot == 0, "the reference handles one request slot per rank"
        rows = slice(gr * QP, gr * QP + QP)
        qr_ = qall[gr * QP : gr * QP + q].reshape(q * NH, d).astype(F32)
        qbuf = jnp.zeros((1, RP, d), F32).at[0, : q * NH].set(qr_)
        c0 = full.wp_cols + 4 * full.idx_head_dim if kind == "csa" else 0
        nc = 4 * d if kind == "csa" else 2 * d
        if kind == "swa":
            inputs = dict(q=qbuf, kv=kvr[rows][None], ctx_kv=ctxkv[rows][None])
        else:
            inputs = dict(q=qbuf, kv=kvr[rows][None], comp=xp[rows, c0 : c0 + nc][None])
        if kind == "csa":
            inputs.update(idx_q=idxq[rows][None], idx_w=xp[rows, 0:LANE][None],
                          idx_comp=xp[rows, full.wp_cols : c0][None])  # fmt: skip
        ls = attn["sets"][r]
        arrays, out, _ = attn["reference"](ls, kind, li, inputs)
        for jj in range(gs):
            attn["sets"][r + jj] = dataclasses.replace(
                ls, arrays=[jnp.asarray(a) for a in arrays]
            )
        o_r = jnp.asarray(out[0])
        if attn.get("noise"):  # band probe: perturb by the phase's tolerance
            key = jax.random.fold_in(jax.random.key(17), li * 1000 + r)
            o_r = o_r + attn["noise"] * jnp.abs(o_r).mean() * jax.random.normal(
                key, o_r.shape
            )
        o = o.at[gr * QP : gr * QP + q].set(o_r.astype(BF16))
        attn.setdefault("o_ref", {})[(li, r)] = np.asarray(out[0], np.float32)
    return o
