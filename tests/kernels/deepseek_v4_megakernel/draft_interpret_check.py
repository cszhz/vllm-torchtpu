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
"""DSpark draft step (B9 Stage 1) in CPU interpret vs the JAX references.

    python -m tests.kernels.deepseek_v4_megakernel.draft_interpret_check --moe real

Three SWA draft layers (ratio 1, no hash routing), K = 6 draft rows per
request, the context rows (7 per request, the target's global row layout)
projected from main_x by each layer's wqa_ctx pass. Attention is the stub
(the swa phase lands separately), so the residual checks the dense chain, the
MoE and the collectives, and layer 0's context kv (kv_norm + rope of
fused_wqa_wkv(main_x)[:, q_lora:]) is checked directly.
"""

import argparse
import dataclasses
import json
import os

import jax

# STEP_DEVICE=1: the kernel compiled for TPU and the references run on the
# device too (two-host jobs pass COORD / NPROC / PID); else CPU interpret
ON_DEVICE = os.environ.get("STEP_DEVICE") == "1"
if ON_DEVICE:
    if "COORD" in os.environ:
        jax.distributed.initialize(os.environ["COORD"], int(os.environ["NPROC"]),
                                   int(os.environ["PID"]))  # fmt: skip
    R = jax.device_count()
else:
    R = int(os.environ.get("STEP_R", "4"))
    jax.config.update("jax_num_cpu_devices", R)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import (
    dense_config as dc,  # noqa: E402
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import (
    dense_reference as DREF,  # noqa: E402
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import (
    dense_weights as dw,  # noqa: E402
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_phases as SP  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import (
    step_reference as SR,  # noqa: E402
)

K, Q_CTX, T, QP = 6, 7, 16, 8
MAX_CTX = 4096


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--moe", choices=("stub", "real"), default="stub")
    p.add_argument("--fault", choices=("none", "ctx_pos"), default="none")
    p.add_argument("--attn", choices=("stub", "real"), default="stub")
    p.add_argument("--vmem-share", action="store_true", help="vmem_share on")
    p.add_argument("--dump", default="", help="save the kernel outputs (.npz)")
    a = p.parse_args()
    moe_inter = 1024 if a.moe == "real" else 128 * R
    if a.attn == "real":  # the attention phase is DSv4-shaped: 128 heads of 512
        assert R == 16, "run with STEP_R=16 (128 heads, 8 per rank)"
        B = 1
        full = dc.small_config(
            n_heads=128, head_dim=512, q_lora=256, o_lora=256, idx_heads=64,
            hidden=512, moe_inter=moe_inter, tp=1, rows=T, bank_bytes=1 << 20,
            ratios=(1, 1, 1),
        )  # fmt: skip
    else:
        B = 2
        full = dc.small_config(n_heads=8 * R, tp=1, rows=T, moe_inter=moe_inter,
                               ratios=(1, 1, 1))  # fmt: skip
    M = B * K
    tp_cfg = dataclasses.replace(full, tp=R)
    scfg = SK.StepKernelConfig(dense=tp_cfg, q=K, B=B, moe_impl=a.moe,
                               ctx_q=Q_CTX, num_hash_layers=0, debug_att=True,
                               attn_impl=a.attn, max_ctx=MAX_CTX,
                               vmem_share=a.vmem_share)  # fmt: skip
    w = dw.make_weights(full)
    packs = [dw.pack_weights(SR.rank_weights(w, full, r, R), tp_cfg) for r in range(R)]
    tiles = {n: jnp.concatenate([pk["tiles"][n] for pk in packs])
             for n in tp_cfg.stream_names}  # fmt: skip
    smalls = jnp.concatenate([pk["smalls"] for pk in packs])
    V = 512
    embed = jax.random.normal(jax.random.key(7), (V, full.hidden)).astype(jnp.bfloat16)
    max_pos = 4096
    ang = jnp.arange(max_pos)[:, None] * 10000.0 ** (-jnp.arange(32) / 32.0)[None]
    cos_sin = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=1).astype(jnp.float32)
    # ranks 0 and R-1: one request each; draft block [anchor, mask x K-1]
    rng = np.random.default_rng(0)
    ids = np.zeros((R, M), np.int32)
    pos = np.zeros((R, M), np.int32)
    seq = np.zeros((R, B), np.int32)
    base = {}
    for r in (0, R - 1):
        b0 = int(rng.integers(200, 3000))
        base[r] = b0
        ids[r, :K] = [int(rng.integers(0, V))] + [V - 1] * (K - 1)
        pos[r, :K] = np.arange(b0, b0 + K)
        seq[r, 0] = b0 + K
    # context rows: the target step's verify rows, global row blocks of QP
    owners = [r for r in range(R) if seq[r, 0] > 0]
    main_x = jax.random.normal(jax.random.key(9), (T, full.hidden)).astype(jnp.bfloat16)
    cpos = np.full((T,), -1, np.int32)
    for g, r in enumerate(owners):
        cpos[g * QP : g * QP + Q_CTX] = np.arange(base[r] - Q_CTX, base[r])
    # positive control: the kernel ropes the context kv at shifted positions
    cpos_k = np.where(cpos >= 0, cpos + 1, cpos) if a.fault == "ctx_pos" else cpos
    sets, attn_args, n_arr = None, [], 0
    if a.attn == "real":
        from tests.kernels.deepseek_v4_megakernel import attn_swa_harness as H

        hr = np.random.default_rng(5)
        sets = [H.build(hr, 3, [base.get(r, 1030)], [(base.get(r, 1030) - Q_CTX, Q_CTX)],
                        K, MAX_CTX) for r in range(R)]  # fmt: skip
        sets = [
            dataclasses.replace(s_, sinks=sets[0].sinks) for s_ in sets
        ]  # one model
        metas = [H.meta_arrays(s_) for s_ in sets]
        n_arr = 3
        attn_args = [jnp.concatenate([jnp.asarray(s_.arrays[i]) for s_ in sets])
                     for i in range(n_arr)]  # fmt: skip
        attn_args.append(jnp.concatenate([jnp.asarray(m["bt_flat"]) for m in metas]))
    devs = sorted(jax.devices(), key=lambda dv: dv.id)[:R]
    mesh = jax.sharding.Mesh(np.array(devs), ("d",))
    step = SK.build(scfg, axis="d", interpret=not ON_DEVICE)
    L, names = full.n_layers, tp_cfg.stream_names
    moe, moe_ops = None, []
    if a.moe == "real":
        from vllm_torchtpu.kernels.deepseek_v4_megakernel import moe_experts as ME
        from vllm_torchtpu.kernels.deepseek_v4_megakernel import moe_phase as MP
        from vllm_torchtpu.kernels.deepseek_v4_megakernel import moe_reference as MR

        mcfg = MP.phase_config(scfg.geometry())
        mk = jax.random.split(jax.random.key(11), L + 1)
        mw = [MR.synthetic_weights(mk[i], mcfg) for i in range(L)]
        bias = jax.random.normal(mk[L], (L, full.n_experts), jnp.float32) * 0.1
        moe = dict(cfg=mcfg, weights=mw, bias=bias, tid2eid=[])
        for w1, w1s, w2, w2s in mw:
            moe_ops += [ME.pack_fp4_rows(w1), ME.pad_scale_rows(w1s),
                        ME.pack_fp4_rows(w2), ME.pad_scale_rows(w2s)]  # fmt: skip

    def per_rank(ids, pos, seq, smalls, *rest):
        tl, mo = rest[: len(names)], rest[len(names) : len(names) + len(moe_ops)]
        at = rest[len(names) + len(moe_ops) :]
        attn = None
        if sets is not None:
            roles = SK.META_ROLES
            z = jnp.zeros((8, 128), jnp.float32)
            wts = {k: z for k in SP.AttnWeights._fields}
            wts["attn_sink"] = jnp.asarray(sets[0].sinks)
            attn = dict(
                arrays=list(at[:n_arr]),
                cache_index={r: ([0, 1, 2] if r == "swa" else []) for r in SK.ROLES},
                cache_layers={r: ([0, 1, 2] if r == "swa" else []) for r in SK.ROLES},
                bt_flat=at[n_arr], weights=wts,
                bt_off={r: jnp.asarray(metas[0]["bt_off"][r]) for r in roles},
                bt_w={r: jnp.asarray(metas[0]["bt_w"][r]) for r in roles},
            )  # fmt: skip
        moe_w = None
        if moe is not None:
            moe_w = dict(tid2eid=[], bias=moe["bias"], w13=list(mo[0::4]),
                         w13_scale=list(mo[1::4]), w2=list(mo[2::4]),
                         w2_scale=list(mo[3::4]))  # fmt: skip
        outs = step(ids[0], pos[0], seq[0], embed, SP.pack_cos_sin(cos_sin), smalls,
                    dict(zip(names, tl)), moe_w, eblk=jax.lax.axis_index("d"),
                    ctx=dict(main_x=main_x, positions=jnp.asarray(cpos_k)),
                    attn=attn)  # fmt: skip
        arrs = list(outs[5]) if attn is not None else []
        return (outs[0][None], outs[-1][None], *arrs)

    f = jax.jit(jax.shard_map(per_rank, mesh=mesh,
                              in_specs=(P("d"),) * (4 + len(tiles) + len(moe_ops)
                                                    + len(attn_args)),
                              out_specs=(P("d"),) * (2 + n_arr), check_vma=False))  # fmt: skip
    sh = jax.sharding.NamedSharding(mesh, P("d"))
    args = [jnp.asarray(ids), jnp.asarray(pos), jnp.asarray(seq), smalls,
            *[tiles[n] for n in names], *moe_ops, *attn_args]  # fmt: skip
    from jax.experimental import multihost_utils

    def put(x):
        x = np.asarray(x)
        return jax.make_array_from_callback(x.shape, sh, lambda i: x[i])

    outs = jax.block_until_ready(f(*[put(x) for x in args]))
    outs = [multihost_utils.process_allgather(o, tiled=True) for o in outs]
    if a.dump:
        np.savez(a.dump, *[np.asarray(o).view(np.uint8) for o in outs])
    res, ctx_kv = np.asarray(outs[0], np.float32), np.asarray(outs[1], np.float32)
    arr_out = [np.asarray(x) for x in outs[2:]]
    rc = SR.RefConfig(full=full, R=R, T=T, q=K, aux_layers=(), ar_mode="hrsag",
                      ar_group=8)  # fmt: skip
    ctx = dict(main_x=main_x, positions=cpos)
    refs = {}
    orders = ("kernel", "stock", "noise") + (("stock_kernels",) if ON_DEVICE else ())
    for order in orders:
        adict = None
        if sets is not None:

            def ref_fn(ls, kind, li, inputs):
                ls = dataclasses.replace(ls, arrays=[np.array(x) for x in ls.arrays])
                caches, o = H.reference(ls, li, inputs)
                return caches, o, None

            adict = dict(sets=[dataclasses.replace(s_) for s_ in sets], reference=ref_fn,
                         noise=1e-3 if order == "noise" else 0.0)  # fmt: skip
        refs[order] = SR.step_reference(
            rc, w, embed, cos_sin, ids, pos, seq,
            "kernel" if order == "noise" else order, moe, attn=adict, ctx=ctx,
        )  # fmt: skip
    rk, rs = refs["kernel"][0], refs["stock"][0]
    # layer 0's context kv, stock arithmetic (fp8 activation quant of main_x)
    lw0 = DREF.layer_weights(SR.rank_weights(w, full, 0, R), full, 0)
    kv = DREF.xla_quantized_matmul(main_x, lw0["wqa"], lw0["wqa_scale"])
    kvn = DREF.rms_norm(kv[:, full.q_lora :], lw0["kv_norm"], full.rms_eps)
    kvn = kvn.astype(jnp.float32)
    cs = SK.rope_rows(cos_sin, jnp.maximum(jnp.asarray(cpos), 0), False)
    d, lane = full.head_dim, 128
    tail = kvn[:, d - lane :]
    roped = tail * cs[:, :lane] + DREF.rotate_pairs(tail) * cs[:, lane:]
    ref_kv = jnp.concatenate([kvn[:, : d - lane], roped], -1).astype(jnp.bfloat16)
    rows = np.flatnonzero(cpos >= 0)
    got = np.asarray(ctx_kv[0], np.float32)[rows]
    out = {
        "res_rel_korder": _rel(res, rk),
        "res_rel_stock": _rel(res, rs),
        "band_res": _rel(rs, rk),
        "ctx_kv_rel": _rel(got, np.asarray(ref_kv, np.float32)[rows]),
        "ctx_kv_same_on_ranks": bool(
            all(
                np.array_equal(np.asarray(ctx_kv[r]), np.asarray(ctx_kv[0]))
                for r in range(R)
            )
        ),  # fmt: skip
        "inactive_rows_zero": bool(np.all(np.asarray(res, np.float32)[1 : R - 1] == 0)),
    }
    if sets is not None:
        out["band_res_att_noise"] = _rel(refs["noise"][0], rk)
        ksets = refs["kernel"][2]
        out["cache_rel"] = {
            f"{i}@{r}": _rel(arr_out[i].reshape(R, *sets[0].arrays[i].shape)[r],
                             np.asarray(ksets[r].arrays[i]))
            for i in range(n_arr) for r in (0, R - 1)
        }  # fmt: skip
    if ON_DEVICE:  # the stock Pallas mHC / wo_a kernels, as serving runs them
        out["res_rel_stock_kernels"] = _rel(res, refs["stock_kernels"][0])
    if jax.process_index() == 0:
        print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
