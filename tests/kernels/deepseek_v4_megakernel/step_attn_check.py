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
"""CPU-interpret check of the whole step with the REAL attention phase.

    XLA_FLAGS=--xla_allow_excess_precision=false JAX_PLATFORMS=cpu \\
        python -m tests.kernels.deepseek_v4_megakernel.step_attn_check

R=16 simulated ranks (128 heads, head_dim 512: the attention phase's
geometry), reduced hidden/widths, 3 layers (HCA, HCA, CSA), two owner ranks
with one 7-row request each over per-rank synthetic served caches
(attn_phase_harness.build). Prints a RESULT line: kernel vs the step
reference (block references composed), and the caches.
"""

import argparse
import functools
import json
import os
import sys

import jax

jax.config.update("jax_num_cpu_devices", 16)

import dataclasses  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_phase_harness as H  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_weights as dw  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_reference as SR  # noqa: E402


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--fault", choices=("none", "a2a_heads"), default="none")
    p.add_argument("--kinds", default="hca,hca,csa")
    p.add_argument("--share-tables", action="store_true",
                   help="one block table per (role, kind): layers after the first reuse the staged tables")
    # seq_len (incl. the 7 new rows) of the two requests
    p.add_argument("--seqs", default="1030,3000")
    p.add_argument("--strict", action="store_true", help="StepKernelConfig.strict_causal")
    p.add_argument("--vmem-share", action="store_true", help="vmem_share on")
    p.add_argument("--dump", default="", help="save the kernel outputs (.npz)")
    a = p.parse_args()
    R, T, q, B = 16, 16, 7, 1
    M = B * q
    kinds = tuple(a.kinds.split(","))
    aux_layers = tuple(range(max(0, len(kinds) - 2), len(kinds)))
    full = dc.small_config(
        n_heads=128, head_dim=512, q_lora=256, o_lora=256, idx_heads=64,
        hidden=512, moe_inter=128 * R, tp=1, rows=T, bank_bytes=1 << 20,
        ratios=tuple(128 if k == "hca" else 4 for k in kinds),
    )  # fmt: skip
    tp_cfg = dataclasses.replace(full, tp=R)
    scfg = SK.StepKernelConfig(
        dense=tp_cfg, q=q, B=B, aux_layers=aux_layers, attn_impl="real",
        max_ctx=a.max_ctx, fault="" if a.fault == "none" else a.fault, debug_att=True,
        strict_causal=a.strict, vmem_share=a.vmem_share,
    )  # fmt: skip
    w = dw.make_weights(full)
    packs = [dw.pack_weights(SR.rank_weights(w, full, r, R), tp_cfg) for r in range(R)]
    names = tp_cfg.stream_names
    tiles = {n: jnp.concatenate([pk["tiles"][n] for pk in packs]) for n in names}
    smalls = jnp.concatenate([pk["smalls"] for pk in packs])
    V = 512
    embed = jax.random.normal(jax.random.key(7), (V, full.hidden)).astype(jnp.bfloat16)

    s0, s1 = (int(x) for x in a.seqs.split(","))
    owners = {0: s0, 9: s1}  # rank -> seq_len (incl. the 7 new rows)
    sets = []
    for r in range(R):
        rng = np.random.default_rng(100 + r)
        sets.append(H.build(rng, kinds, [owners.get(r, 1030)], a.max_ctx,
                            share_tables=a.share_tables))
    wts = sets[0].weights  # one model: every rank holds the same weights
    sets = [dataclasses.replace(s_, weights=wts) for s_ in sets]
    cos_sin_cache = wts["cos_sin"]
    ids = np.zeros((R, M), np.int32)
    pos = np.zeros((R, M), np.int32)
    seq = np.zeros((R, B), np.int32)
    rng = np.random.default_rng(0)
    for r, sl in owners.items():
        ids[r] = rng.integers(0, V, q)
        pos[r] = np.arange(sl - q, sl)
        seq[r, 0] = sl
    metas = [H.meta_arrays(s_, 1 if r in owners else 0) for r, s_ in enumerate(sets)]
    n_arr = len(sets[0].arrays)
    arrays = [jnp.concatenate([s_.arrays[i] for s_ in sets]) for i in range(n_arr)]
    bt_flat = jnp.concatenate([jnp.asarray(m["bt_flat"]) for m in metas])
    roles = SK.META_ROLES

    mesh = jax.make_mesh((R,), ("d",))
    step = SK.build(scfg, axis="d", interpret=True)
    wnames = list(SK.SP.AttnWeights._fields)

    def per_rank(ids, pos, seq, smalls, bt, *rest):
        tl = rest[: len(names)]
        arrs = rest[len(names) :]
        attn = dict(
            arrays=list(arrs), cache_index={r: sets[0].role_arr[r] for r in SK.ROLES},
            cache_layers={r: sets[0].role_layers[r] for r in SK.ROLES}, bt_flat=bt,
            bt_off={r: jnp.asarray(metas[0]["bt_off"][r]) for r in roles},
            bt_w={r: jnp.asarray(metas[0]["bt_w"][r]) for r in roles},
            weights={k: AP.pack_cos_sin(wts[k]) if k in AP.COS_SIN_FIELDS else wts[k]
                     for k in wnames},
        )  # fmt: skip
        res, aux, _, _, _, arr_out, att0, _ = step(
            ids[0], pos[0], seq[0], embed, AP.pack_cos_sin(cos_sin_cache), smalls,
            dict(zip(names, tl)),
            attn=attn,
        )  # fmt: skip
        return (res[None], aux[None], att0[None], *arr_out)

    nl = len(names)
    f = jax.jit(jax.shard_map(
        per_rank, mesh=mesh, in_specs=(P("d"),) * (5 + nl + n_arr),
        out_specs=(P("d"),) * (3 + n_arr), check_vma=False,
    ))  # fmt: skip
    shard = jax.sharding.NamedSharding(mesh, P("d"))
    args = [jnp.asarray(ids), jnp.asarray(pos), jnp.asarray(seq), smalls, bt_flat,
            *[tiles[n] for n in names], *arrays]  # fmt: skip
    outs = jax.block_until_ready(f(*[jax.device_put(x, shard) for x in args]))
    res, aux, att0 = outs[0], outs[1], np.asarray(outs[2])
    if a.dump:  # every output incl. the updated caches
        np.savez(a.dump, *[np.asarray(o).view(np.uint8) for o in outs])
    outs = outs[:2] + outs[3:]
    rc = SR.RefConfig(full=full, R=R, T=T, q=q, aux_layers=aux_layers, ar_mode="hrsag",
                      ar_group=8)  # fmt: skip
    ref, adict = {}, {}
    for order in ("kernel", "stock", "noise"):
        sets_o = [dataclasses.replace(s_) for s_ in sets]
        adict[order] = dict(sets=sets_o,
                            reference=functools.partial(H.reference, strict_causal=a.strict),
                            noise=1e-3 if order == "noise" else 0.0)  # fmt: skip
        ref[order] = SR.step_reference(
            rc, w, embed, cos_sin_cache, ids, pos, seq,
            "kernel" if order == "noise" else order, attn=adict[order],
        )  # fmt: skip
    att_err = {}
    for r in owners:
        o_k = att0.reshape(R, B, -1, 512)[r, 0, : q * 128].reshape(q, 128, 512)
        o_r = adict["kernel"]["o_ref"][(0, r)]
        err = np.abs(o_k - o_r)
        att_err[r] = (float(err.max() / np.abs(o_r).max()), float(err.mean()))
    rk, ak, sk = ref["kernel"]
    rs, as_, _ = ref["stock"]
    cache_rel = {
        f"{i}@{r}": _rel(np.asarray(outs[2 + i]).reshape(R, *sets[0].arrays[i].shape)[r],
                         np.asarray(sk[r].arrays[i]))
        for i in range(n_arr) for r in owners
    }  # fmt: skip
    cache_rel_max = max(cache_rel.values())
    out = {
        "res_rel_korder": _rel(res, rk), "aux_rel_korder": _rel(aux, ak),
        "res_rel_stock": _rel(res, rs), "band_res": _rel(rs, rk),
        # attention output perturbed by 1e-3 of its mean |value| (the phase's
        # mean tolerance): the chain's sensitivity to a legit attention error
        "band_res_att_noise": _rel(ref["noise"][0], rk),
        "band_aux": _rel(as_, ak), "cache_rel_max": cache_rel_max,
        "cache_rel": cache_rel, "layer0_att_maxrel_meanabs": att_err,
        "idle_rows_zero": bool(np.all(np.asarray(res, np.float32)[1:9] == 0)),
    }  # fmt: skip
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
