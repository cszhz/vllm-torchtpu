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
"""CPU-interpret check of the whole-step kernel on R simulated ranks.

    XLA_FLAGS=--xla_allow_excess_precision=false JAX_PLATFORMS=cpu \\
        python -m tests.kernels.deepseek_v4_megakernel.step_interpret_check --R 4

Reduced widths (dense_config.small_config), 3 layers (HCA, HCA, CSA), stub
phases. Prints one RESULT JSON line: relative errors of the kernel's own-row
outputs vs the plain-JAX composition (kernel and stock summation orders) and
the reassociation band between the two.
"""

import argparse
import json
import os

import jax

R_DEFAULT = int(os.environ.get("STEP_R", "4"))
ON_DEVICE = os.environ.get("STEP_DEVICE") == "1"  # TPU: kernel and references on device
if not ON_DEVICE:
    jax.config.update("jax_num_cpu_devices", R_DEFAULT)

import dataclasses  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_weights as dw  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_phases as SP  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_reference as SR  # noqa: E402


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=int, default=R_DEFAULT)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--q", type=int, default=7)
    p.add_argument("--B", type=int, default=2)
    # eblk: the expert arrays are permuted (--expert-perm) but the kernel is
    # told the identity, as if it read the EP rank off the mesh index
    p.add_argument("--fault", choices=("none", "a2a_heads", "eblk", "placement"), default="none")
    # EP rank whose experts mesh index i holds, comma separated (default identity)
    p.add_argument("--expert-perm", default="")
    p.add_argument("--moe", choices=("stub", "real"), default="stub")
    p.add_argument("--debug-att", action="store_true")
    p.add_argument("--per-rank", type=int, default=1, help="requests on ranks 0, R-1")
    # attention groups of this many ranks (TP8 x DP2 serving: 8); requests on
    # the first and last group, each group's ranks given the same batch
    p.add_argument("--attn-tp", type=int, default=1)
    # each rank gets only its group's vocab-parallel embed shard
    p.add_argument("--vocab-shard", action="store_true")
    # MoE: a random static expert permutation per layer (moe_phase placement)
    p.add_argument("--placement", action="store_true")
    p.add_argument("--vmem-share", action="store_true", help="vmem_share on")
    p.add_argument("--ar-live", action="store_true", help="ar_live_rows on")
    p.add_argument("--dump", default="", help="save the kernel outputs (.npz)")
    a = p.parse_args()
    R, T, q, B = a.R, a.T, a.q, a.B
    perm = (
        [int(x) for x in a.expert_perm.split(",")] if a.expert_perm else list(range(R))
    )
    assert sorted(perm) == list(range(R)), perm
    told = list(range(R)) if a.fault == "eblk" else perm
    M = B * q
    moe_inter = 1024 if a.moe == "real" else 128 * R  # MoEConfig slice constraints
    full = dc.small_config(n_heads=8 * R, tp=1, rows=T, moe_inter=moe_inter)
    tp_cfg = dataclasses.replace(full, tp=R)
    scfg = SK.StepKernelConfig(
        dense=tp_cfg, q=q, B=B, aux_layers=(1, 2),
        fault="a2a_heads" if a.fault == "a2a_heads" else "", moe_impl=a.moe,
        debug_att=a.debug_att, attn_tp=a.attn_tp,
        moe_placement="perm" if a.placement else "", vmem_share=a.vmem_share,
        ar_live_rows=a.ar_live,
    )  # fmt: skip
    w = dw.make_weights(full)
    packs = [
        dw.pack_weights(
            SR.rank_weights(w, full, SK.head_block(r, R, a.attn_tp), R), tp_cfg
        )
        for r in range(R)
    ]
    tiles = {
        n: jnp.concatenate([pk["tiles"][n] for pk in packs])
        for n in tp_cfg.stream_names
    }
    smalls = jnp.concatenate([pk["smalls"] for pk in packs])
    V = 512
    key = jax.random.key(7)
    embed = (jax.random.normal(key, (V, full.hidden), jnp.float32)).astype(jnp.bfloat16)
    max_pos = 4096
    ang = jnp.arange(max_pos)[:, None] * 10000.0 ** (-jnp.arange(32) / 32.0)[None]
    cos_sin_cache = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=1).astype(
        jnp.float32
    )
    # ranks 0 and R-1: --per-rank requests each; the others idle
    rng = np.random.default_rng(0)
    ids = np.zeros((R, M), np.int32)
    pos = np.zeros((R, M), np.int32)
    seq = np.zeros((R, B), np.int32)
    assert a.per_rank <= B and 2 * a.per_rank * scfg.QP <= T
    GS = a.attn_tp
    for r in (0, R - GS):
        for k in range(a.per_rank):
            p0 = int(rng.integers(100, 3000))
            ids[r, k * q : (k + 1) * q] = rng.integers(0, V, q)
            pos[r, k * q : (k + 1) * q] = np.arange(p0, p0 + q)
            seq[r, k] = p0 + q
        ids[r : r + GS], pos[r : r + GS], seq[r : r + GS] = ids[r], pos[r], seq[r]
    mesh = jax.make_mesh(
        (R,), ("d",), devices=sorted(jax.devices(), key=lambda d: d.id)[:R]
    )
    step = SK.build(scfg, axis="d", interpret=not ON_DEVICE)
    L = full.n_layers
    names = tp_cfg.stream_names
    moe = None
    moe_ops = []
    if a.moe == "real":
        from vllm_torchtpu.kernels.deepseek_v4_megakernel import moe_experts as ME
        from vllm_torchtpu.kernels.deepseek_v4_megakernel import moe_phase as MP
        from vllm_torchtpu.kernels.deepseek_v4_megakernel import moe_reference as MR

        mcfg = MP.phase_config(scfg.geometry())
        mk = jax.random.split(jax.random.key(11), L + 2)
        mw = [MR.synthetic_weights(mk[i], mcfg) for i in range(L)]
        bias = jax.random.normal(mk[L], (L, full.n_experts), jnp.float32) * 0.1
        tkeys = [jax.random.fold_in(mk[L + 1], i) for i in range(3)]
        E = full.n_experts
        tid2eid = [jax.random.randint(k, (V, 6), 0, E, jnp.int32) for k in tkeys]
        moe = dict(cfg=mcfg, weights=mw, bias=bias, tid2eid=tid2eid, expert_perm=perm)
        EL = E // R

        def by_mesh(x):  # mesh index i holds EP rank perm[i]'s expert block
            return jnp.concatenate([x[p * EL : (p + 1) * EL] for p in perm])

        held = None
        if a.placement:  # held[r, l, j]: the global expert in slot j of rank r
            prng = np.random.default_rng(5)
            held = np.stack([prng.permutation(E) for _ in range(L)]).reshape(L, R, EL)
            held = held.transpose(1, 0, 2).astype(np.int32)
            moe["held"] = held

            def by_mesh(x, li):  # noqa: F811 -- mesh index r holds its slots
                return jnp.concatenate([x[held[r, li]] for r in range(R)])

        else:
            _by_mesh = by_mesh

            def by_mesh(x, li):  # noqa: F811
                return _by_mesh(x)

        for li, (w1, w1s, w2, w2s) in enumerate(mw):  # EP-sharded by the mesh
            moe_ops += [by_mesh(ME.pack_fp4_rows(w1), li),
                        by_mesh(ME.pad_scale_rows(w1s), li),
                        by_mesh(ME.pack_fp4_rows(w2), li),
                        by_mesh(ME.pad_scale_rows(w2s), li)]  # fmt: skip

    Vs = V // GS
    emb_r = (
        jnp.concatenate([embed[(r % GS) * Vs : (r % GS + 1) * Vs] for r in range(R)])
        if a.vocab_shard
        else jnp.tile(embed, (R, 1))
    )
    vstart = jnp.asarray([(r % GS) * Vs for r in range(R)], jnp.int32)

    held_r = (
        # fault: the kernel is told the next layer's placement
        jnp.asarray(np.roll(moe["held"], int(a.fault == "placement"), axis=1))
        .reshape(R * L, -1)
        if moe is not None and a.placement
        else jnp.zeros((R, 1), jnp.int32)
    )

    def per_rank(ids, pos, seq, eblk, emb, vs, hr, smalls, *rest):
        tl, mo = rest[: len(names)], rest[len(names) :]
        moe_w = None
        if moe is not None:
            moe_w = dict(tid2eid=moe["tid2eid"], bias=moe["bias"],
                         w13=list(mo[0::4]), w13_scale=list(mo[1::4]),
                         w2=list(mo[2::4]), w2_scale=list(mo[3::4]))  # fmt: skip
            if a.placement:
                moe_w["held"] = hr
        outs = step(ids[0], pos[0], seq[0], emb, SP.pack_cos_sin(cos_sin_cache), smalls,
                    dict(zip(names, tl)), moe_w, eblk=eblk[0],
                    vocab_start=vs[0] if a.vocab_shard else None)  # fmt: skip
        res, aux = outs[0], outs[1]
        if a.debug_att:
            return res[None], aux[None], outs[-2][None], outs[-1][None]
        return res[None], aux[None]

    f = jax.jit(jax.shard_map(
        per_rank, mesh=mesh, in_specs=(P("d"),) * (8 + len(tiles) + len(moe_ops)),
        out_specs=(P("d"),) * (4 if a.debug_att else 2), check_vma=False,
    ))  # fmt: skip
    shard = jax.sharding.NamedSharding(mesh, P("d"))
    args = [
        jnp.asarray(ids),
        jnp.asarray(pos),
        jnp.asarray(seq),
        jnp.asarray(told, jnp.int32),
        emb_r,
        vstart,
        held_r,
        smalls,
        *[tiles[n] for n in names],
        *moe_ops,
    ]
    outs = jax.block_until_ready(f(*[jax.device_put(x, shard) for x in args]))
    res, aux = outs[0], outs[1]
    if a.dump:
        np.savez(a.dump, *[np.asarray(o).view(np.uint8) for o in outs])
    rc = SR.RefConfig(full=full, R=R, T=T, q=q, aux_layers=(1, 2), ar_mode="hrsag",
                      ar_group=8, attn_tp=a.attn_tp)  # fmt: skip
    ref = (
        jax.jit(SR.step_reference, static_argnums=(0, 7))
        if False
        else SR.step_reference
    )
    trace = {}
    rk, ak = ref(rc, w, embed, cos_sin_cache, ids, pos, seq, "kernel", moe, trace=trace)
    att_cmp = {}
    if a.debug_att:
        att = np.asarray(outs[2], np.float32).reshape(R, B, -1, full.head_dim)
        o = np.asarray(trace["o"], np.float32)  # [T, NH, d]
        q_ref = np.asarray(trace["qall"], np.float32)
        NH = full.n_heads
        for r in range(R):
            rows = trace["own_rows"][r]
            if not rows:
                continue
            NA, j = NH // GS, r % GS  # the attended heads [j NA, (j + 1) NA)
            k_att = att[r, 0, : q * NA].reshape(q, NA, -1)
            e = o[rows[:q], j * NA : (j + 1) * NA]
            att_cmp[r] = float(np.linalg.norm(k_att - e) / np.linalg.norm(e))
            att_cmp[f"q_ref_norm{r}"] = float(np.linalg.norm(q_ref[rows[:q]]))
        dx = np.asarray(outs[3], np.float32).reshape(R, 3, T, -1)
        live = sorted(sum(trace["own_rows"], []))

        def rr(k, e):
            return float(np.linalg.norm(k[live] - np.asarray(e, np.float32)[live])
                         / np.linalg.norm(np.asarray(e, np.float32)[live]))  # fmt: skip

        att_cmp["wo_b_part_rel"] = [
            rr(dx[r, 0], trace["wo_b_parts"][r]) for r in range(R)
        ]
        att_cmp["x_att_rel"] = rr(dx[0, 1], trace["x_att"])
        att_cmp["x_ffn_rel"] = rr(dx[0, 2], trace["x_ffn"])
        att_cmp["ffn_part_rel"] = [
            rr(dx[r, 2] * 0, trace["ffn_parts"][r]) for r in range(1)
        ]
    rs, as_ = ref(rc, w, embed, cos_sin_cache, ids, pos, seq, "stock", moe)
    sk = {}
    if ON_DEVICE:  # the stock Pallas mHC / wo_a kernels (serving arithmetic)
        trace_s = {}
        rsk, ask = ref(rc, w, embed, cos_sin_cache, ids, pos, seq, "stock_kernels", moe,
                       trace=trace_s)  # fmt: skip
        sk = {"res_rel_stock_kernels": _rel(res, rsk),
              "band_stock_kernels_vs_korder": _rel(rsk, rk)}  # fmt: skip
        if a.debug_att:
            dxs = np.asarray(outs[3], np.float32).reshape(R, 3, T, -1)
            lv = sorted(sum(trace_s["own_rows"], []))
            sk["wo_b_part_rel_vs_stock_kernels"] = [
                float(np.linalg.norm(dxs[r, 0][lv] - np.asarray(trace_s["wo_b_parts"][r],
                      np.float32)[lv]) / np.linalg.norm(np.asarray(
                      trace_s["wo_b_parts"][r], np.float32)[lv])) for r in range(R)]  # fmt: skip
    out = {
        "res_rel_korder": _rel(res, rk),
        "aux_rel_korder": _rel(aux, ak),
        "res_rel_stock": _rel(res, rs),
        "band_res": _rel(rs, rk),
        "band_aux": _rel(as_, ak),
        "res_max_abs": float(np.max(np.abs(np.asarray(res, np.float32) - rk))),
        "layer0_stub_att_rel": att_cmp,
        **sk,
        "inactive_rows_zero": bool(
            np.all(np.asarray(res, np.float32)[GS : R - GS] == 0)
        ),
    }
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
