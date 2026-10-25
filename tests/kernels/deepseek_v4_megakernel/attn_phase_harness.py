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
"""Single-layer harness of `step_phases.owner_attention_phase`.

Builds a small multi-layer cache set in the served layouts (int32 word views),
runs the phase for one traced layer in a standalone pallas_call, and computes
the expected output and caches from the block references: attn_ref (attention,
indexer scores, top-k), kv_write.compress_store (compressor stores, bit-exact
vs the stock goldens in test_kv_write) and the SWA row encoding.
"""

import dataclasses
import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_cases as C  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_core as core  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_layout as L  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_ref  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kv_write  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_phases as SP  # noqa: E402

ROLES = ("swa", "kv", "state", "kv_rope", "indexer_k", "indexer_state")
META_ROLES = ROLES + ("swa_pages",)
Q = 7


def _f32_pages(rng, n, rows, lanes):
    return rng.standard_normal((n, rows, lanes), dtype=np.float32).view(np.int32)


def _window_table(rng, R, width, start, count, seq_lens, sbs):
    """State tables: the pages under each request's last 160 positions are
    distinct pool pages, the rest point anywhere in the pool (never read)."""
    t = rng.integers(start, start + count, (R, width)).astype(np.int32)
    perm = start + rng.permutation(count)
    k = 0
    for r in range(R):
        lo = max(int(seq_lens[r]) - 160, 0) // sbs
        for c in range(lo, min(width, -(-int(seq_lens[r]) // sbs))):
            t[r, c] = perm[k]
            k += 1
    assert k <= count
    return t


def _perm_table(rng, R, width, start, count):
    """[R, width] page ids: distinct pages from [start, start + count) per row prefix."""
    pages = start + rng.permutation(count)
    per = count // R
    t = np.zeros((R, width), np.int32)
    for r in range(R):
        own = pages[r * per:(r + 1) * per]
        t[r, :min(width, len(own))] = own[:width]
    return t


@dataclasses.dataclass
class LayerSet:
    kinds: tuple
    seq_lens: np.ndarray
    arrays: list  # unique cache arrays (int32)
    role_arr: dict  # role -> [array index per layer of the role]
    role_layers: dict  # role -> layer ids
    tables: dict  # (role, layer) -> [R, W] int32
    swa_table: dict  # layer -> logical SWA table [R, P]
    weights: dict
    max_ctx: int


def build(rng, kinds, seq_lens, max_ctx, share_tables=False):
    """share_tables: one block table per (role, kind) for all layers, as
    serving's per-group tables (keeps bt_flat within SMEM at 61 layers)."""
    seq_lens = np.asarray(seq_lens, np.int32)
    memo = {}

    def tab(key, make):
        return memo.setdefault(key, make()) if share_tables else make()
    R = len(seq_lens)
    arrays, role_arr, role_layers, tables, swa_table = [], {r: [] for r in ROLES}, {r: [] for r in ROLES}, {}, {}

    def add(a):
        arrays.append(jnp.asarray(a))
        return len(arrays) - 1

    def bind(role, lid, arr, table):
        role_arr[role].append(arr)
        role_layers[role].append(lid)
        tables[(role, lid)] = table

    for lid, kind in enumerate(kinds):
        # SWA: the three window pages of each request live, the rest arbitrary.
        swa_pp = -(-max_ctx // C.SWA_PAGE)
        n_swa = 3 * R + 2
        swa = C._bf16_token_pages(rng, n_swa, C.SWA_PAGE)

        def swa_tab():
            st = rng.integers(0, n_swa, (R, swa_pp)).astype(np.int32)
            lp = np.maximum(seq_lens - Q - 127, 0) // C.SWA_PAGE
            perm = rng.permutation(n_swa)
            for r in range(R):
                for i in range(3):
                    if lp[r] + i < swa_pp:
                        st[r, lp[r] + i] = perm[3 * r + i]
            return st

        st = tab(("swa", kind), swa_tab)
        a = add(swa)
        bind("swa", lid, a, st)
        swa_table[lid] = st
        if kind == "hca":
            kp = max(1, -(-(max_ctx // 128 + 1) // 8))
            kv = C._bf16_token_pages(rng, R * kp + 1, 8)
            sp = -(-max_ctx // AP.SBS["hca"])
            ns = R * 8 + 2
            state = _f32_pages(rng, ns, 256, 128)
            bind("kv", lid, add(kv), tab(("kv", kind), lambda: _perm_table(rng, R, kp, 0, R * kp + 1)))
            bind("state", lid, add(state), tab(("state", kind), lambda: _window_table(
                rng, R, sp, 0, ns, seq_lens, AP.SBS["hca"])))
        else:
            kp = max(1, -(-(max_ctx // 4 + 1) // 256))
            sp = -(-max_ctx // AP.SBS["csa"])
            nkv = R * kp + 1
            ns = R * 12 + 2
            arr = np.concatenate([np.asarray(C.nope_pages(rng, nkv)), _f32_pages(rng, ns, 256, 128)])
            rope = np.concatenate([np.asarray(C.rope_pages(rng, nkv)),
                                   np.zeros((ns, 64, 128), np.int32)])
            a = add(arr)
            bind("kv", lid, a, tab(("kv", kind), lambda: _perm_table(rng, R, kp, 0, nkv)))
            bind("state", lid, a, tab(("state", kind), lambda: _window_table(
                rng, R, sp, nkv, ns, seq_lens, AP.SBS["csa"])))
            bind("kv_rope", lid, add(rope), tables[("kv", lid)])
            isp = -(-max_ctx // AP.SBS["idx"])
            iarr = np.concatenate([np.asarray(C.idx_pages(rng, nkv)), _f32_pages(rng, ns, 64, 256)])
            ia = add(iarr)
            bind("indexer_k", lid, ia, tab(("ik", kind), lambda: _perm_table(rng, R, kp, 0, nkv)))
            bind("indexer_state", lid, ia, tab(("is", kind), lambda: _window_table(
                rng, R, isp, nkv, ns, seq_lens, AP.SBS["idx"])))
    nl = len(kinds)
    ncsa = sum(k == "csa" for k in kinds)
    maxpos = max_ctx + 8
    ang = np.arange(maxpos)[:, None] * (0.01 * np.arange(1, 33))[None, :]
    cs = np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    weights = dict(
        attn_sink=jnp.asarray(rng.standard_normal((nl, 128), dtype=np.float32)),
        comp_ape=jnp.asarray(0.1 * rng.standard_normal((nl, 128, 1024), dtype=np.float32)),
        comp_norm=jnp.asarray(1 + 0.1 * rng.standard_normal((nl, 512), dtype=np.float32)),
        idx_comp_ape=jnp.asarray(0.1 * rng.standard_normal((max(ncsa, 1), 4, 512), dtype=np.float32)),
        idx_comp_norm=jnp.asarray(1 + 0.1 * rng.standard_normal((max(ncsa, 1), 128), dtype=np.float32)),
        cos_sin=jnp.asarray(cs), comp_cos_sin=jnp.asarray(cs[:, ::-1].copy()),
        idx_cos_sin=jnp.asarray(cs * 0.9),
    )
    return LayerSet(tuple(kinds), seq_lens, arrays, role_arr, role_layers, tables, swa_table,
                    weights, max_ctx)


def make_inputs(rng, ls, kind, idx_corr=0.0):
    """idx_corr in [0, 1): the q rows' indexer queries share that fraction of
    a common component (consecutive verify tokens select overlapping rows)."""
    R = len(ls.seq_lens)
    RP = -(-Q * 128 // 128) * 128
    q = np.zeros((R, RP, 512), np.float32)
    q[:, :Q * 128] = rng.standard_normal((R, Q * 128, 512))
    wc = 2048 if kind == "csa" else 1024
    f = lambda *s: jnp.asarray(rng.standard_normal(s, dtype=np.float32))  # noqa: E731
    return dict(
        q=jnp.asarray(q, jnp.bfloat16).astype(jnp.float32),  # bf16 values, f32 storage
        kv=jnp.pad(f(R, Q, 512), ((0, 0), (0, 1), (0, 0))).astype(jnp.bfloat16),
        comp=jnp.pad(f(R, Q, wc), ((0, 0), (0, 1), (0, 0))),
        idx_q=jnp.pad(idx_corr * f(R, 1, 8192) + np.sqrt(1 - idx_corr ** 2) * f(R, Q, 8192),
                      ((0, 0), (0, 1), (0, 0))).astype(jnp.bfloat16),
        idx_w=jnp.pad((idx_corr * f(R, 1, 128) + np.sqrt(1 - idx_corr ** 2) * f(R, Q, 128)) * 0.3,
                      ((0, 0), (0, 1), (0, 0))),
        idx_comp=jnp.pad(f(R, Q, 512), ((0, 0), (0, 1), (0, 0))),
    )


def meta_arrays(ls, num_req):
    R = len(ls.seq_lens)
    nl = len(ls.kinds)
    flat, off, wid = [], {r: np.zeros(nl, np.int32) for r in ROLES}, {r: np.ones(nl, np.int32) for r in ROLES}
    n, at = 0, {}  # a table shared by several layers is stored once
    for (role, lid), t in sorted(ls.tables.items()):
        if id(t) not in at:
            at[id(t)] = n
            flat.append(t.reshape(-1))
            n += t.size
        off[role][lid], wid[role][lid] = at[id(t)], t.shape[1]
    off["swa_pages"], wid["swa_pages"] = np.zeros(nl, np.int32), np.full(nl, 3, np.int32)
    for lid in range(nl):
        pg = np.asarray(core.swa_window_pages(jnp.asarray(ls.seq_lens), jnp.asarray(ls.swa_table[lid]),
                                              q_len=Q, window=128, swa_page=128)).reshape(-1)
        off["swa_pages"][lid] = n
        flat.append(pg)
        n += pg.size
    pos = (ls.seq_lens[:, None] - Q + np.arange(Q)[None, :]).reshape(-1).astype(np.int32)
    seq = ls.seq_lens.copy()
    seq[num_req:] = 0
    return dict(num_req=np.array([num_req], np.int32), seq_lens=seq, positions=pos,
                bt_flat=np.concatenate(flat), bt_off=off, bt_w=wid)


def u8_roles(ls):
    """Arrays served as uint8 slabs (HCA kv, indexer K/state): their indices."""
    out = set()
    for role in ("kv", "indexer_k", "indexer_state"):
        for a, lid in zip(ls.role_arr[role], ls.role_layers[role]):
            if role != "kv" or ls.kinds[lid] == "hca":
                out.add(a)
    return out


def group_heads(inputs, weights, attn_heads, h0):
    """Inputs / weights for a phase attending heads [h0, h0 + attn_heads) of
    the 128 (group TP): q rows t * attn_heads + h, sink entries rolled to h0."""
    R = inputs["q"].shape[0]
    q = inputs["q"][:, :Q * 128].reshape(R, Q, 128, 512)[:, :, h0:h0 + attn_heads]
    rp = -(-Q * attn_heads // 128) * 128
    q = jnp.pad(q.reshape(R, Q * attn_heads, 512), ((0, 0), (0, rp - Q * attn_heads), (0, 0)))
    w = dict(weights, attn_sink=jnp.roll(weights["attn_sink"], -h0, axis=1))
    return dict(inputs, q=q), w


def build_call(ls, kind, layer, inputs, num_req=None, interpret=True, debug_topk=False,
               served_dtypes=False, reps=1, lease_bytes=0, attn_heads=128, h0=0,
               strict_causal=False, attn_tp=1, stats=False, pipeline=False, races=False):
    """(function, operands) of the phase's standalone pallas_call. With
    `served_dtypes` the uint8-served arrays are passed as uint8 slabs (device
    only: the phase bitcasts them in-kernel). `lease_bytes` (device only): the
    phase gets a PoolLease over a u8 scratch of that size, as the step lends it
    the idle weight pool. `pipeline`: StepGeometry.attn_pipeline. `races`
    (interpret): DMAs executed at their start, with the race detector."""
    R = len(ls.seq_lens)
    num_req = R if num_req is None else num_req
    geom = SP.StepGeometry(R=16, T=16, q=Q, B=R, layer_kinds=ls.kinds, max_ctx=ls.max_ctx,
                           lease_bytes=lease_bytes, attn_heads=attn_heads,
                           strict_causal=strict_causal, attn_tp=attn_tp,
                           attn_split=attn_tp > 1, attn_pipeline=pipeline)
    weights = ls.weights
    if attn_heads != 128:
        inputs, weights = group_heads(inputs, weights, attn_heads, h0)
    leased = lease_bytes > 0
    m = meta_arrays(ls, num_req)
    ci = sum(k == "csa" for k in ls.kinds[:layer])
    smem_ops = ([m["num_req"], m["seq_lens"], m["positions"], AP.pad_tables(m["bt_flat"])]
                + [m["bt_off"][r] for r in META_ROLES] + [m["bt_w"][r] for r in META_ROLES]
                + [np.array([layer, ci], np.int32)])
    in_names = ["q", "kv", "comp"] + (["idx_q", "idx_w", "idx_comp"] if kind == "csa" else [])
    vmem_ops = [inputs[k] for k in in_names]
    wnames = list(SP.AttnWeights._fields)
    w_ops = [AP.pack_cos_sin(weights[k]) if k in AP.COS_SIN_FIELDS else weights[k]
             for k in wnames]
    n_s, n_v, n_w, n_c = len(smem_ops), len(vmem_ops), len(w_ops), len(ls.arrays)
    scratch = AP.owner_attention_scratch_shapes(geom, leased)
    if leased:
        scratch = scratch + [pltpu.VMEM((lease_bytes // 128, 128), jnp.uint8)]

    def kernel(*refs):
        s = refs[:n_s]
        v = refs[n_s:n_s + n_v]
        w = refs[n_s + n_v:n_s + n_v + n_w]
        caches = refs[n_s + n_v + n_w + n_c:n_s + n_v + n_w + 2 * n_c]  # aliased outputs
        out = refs[n_s + n_v + n_w + 2 * n_c]
        scr = refs[n_s + n_v + n_w + 2 * n_c + 1 + 3 * int(debug_topk) + int(stats):]
        lease = SP.PoolLease(None, 0, 0)
        if leased:
            scr, lease = scr[:-1], SP.PoolLease(scr[-1], 0, lease_bytes // 128)
        AP.DEBUG_TOPK = AP.DEBUG_SCORES = AP.DEBUG_LENS = None  # set per trace
        if debug_topk:
            AP.DEBUG_TOPK, AP.DEBUG_SCORES, AP.DEBUG_LENS = \
                refs[n_s + n_v + n_w + 2 * n_c + 1:n_s + n_v + n_w + 2 * n_c + 4]
        meta = SP.AttnMeta(
            num_req=s[0], seq_lens=s[1], positions=s[2], bt_flat=s[3],
            bt_off=dict(zip(META_ROLES, s[4:11])), bt_w=dict(zip(META_ROLES, s[11:18])))
        vin = dict(zip(in_names, v))
        inp = SP.AttnInputs(q=vin["q"], kv=vin["kv"], comp=vin["comp"], idx_q=vin.get("idx_q"),
                            idx_w=vin.get("idx_w"), idx_comp=vin.get("idx_comp"))
        lr = {role: SP.LayerRefs([caches[i] for i in ls.role_arr[role]], ls.role_layers[role])
              for role in ROLES}
        cs = SP.AttnCaches(**lr)

        def once(i, c):  # repeated calls are idempotent (timing only)
            SP.owner_attention_phase(geom, kind, s[18][0], s[18][1], meta, inp, cs,
                                     SP.AttnWeights(*w), out, lease, scr,
                                     me=jax.lax.axis_index("d") if attn_tp > 1 else None,
                                     stats=refs[-len(scratch) - 1] if stats else None)
            return c

        AP.reset_table_stage(geom, scr, leased)
        if reps == 1:
            once(0, 0)
        else:
            jax.lax.fori_loop(0, reps, once, 0)
        AP.DEBUG_TOPK = AP.DEBUG_SCORES = AP.DEBUG_LENS = None

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    arrays = list(ls.arrays)
    if served_dtypes:
        from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
        for i in u8_roles(ls):
            arrays[i] = csa_cache_layout.words_to_slabs(arrays[i])
    fn = pl.pallas_call(
        kernel,
        out_shape=[jax.ShapeDtypeStruct(a.shape, a.dtype) for a in arrays]
        + [jax.ShapeDtypeStruct((R, geom.RP, 512), inputs["q"].dtype)]  # q's storage
        + ([jax.ShapeDtypeStruct((R, 8, AP.TOPK), jnp.int32),
            jax.ShapeDtypeStruct((R, AP.dims_of(geom).n_cols // 128 * 8, 128), jnp.int32),
            jax.ShapeDtypeStruct((R, 8, 128), jnp.int32)] if debug_topk else [])
        + ([jax.ShapeDtypeStruct((len(ls.kinds), R), jnp.int32)] if stats else []),
        in_specs=[smem] * 3 + [hbm] + [smem] * (n_s - 4) + [vmem] * n_v + [hbm] * (n_w + n_c),
        out_specs=[hbm] * n_c + [vmem] + ([vmem] * 3 if debug_topk else [])
        + ([smem] if stats else []),
        scratch_shapes=scratch,
        input_output_aliases={n_s + n_v + n_w + i: i for i in range(n_c)},
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 1024 * 1024,
                                             disable_bounds_checks=True),
        interpret=(pltpu.InterpretParams(detect_races=True, dma_execution_mode="eager") if races
                   else pltpu.InterpretParams()) if interpret else False,
    )
    return fn, [jnp.asarray(x) for x in smem_ops] + list(vmem_ops) + list(w_ops) + arrays


def group_call(ls, kind, layer, inputs, attn_tp=8, attn_heads=16, num_req=None, interpret=True,
               served_dtypes=False, reps=1, ndev=None, pipeline=False, races=False):
    """(jitted fn, stacked operands) of the phase on attn_tp devices (run_group);
    ndev > attn_tp = 1: ndev independent single-rank phases side by side."""
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as Ps
    ndev = ndev or attn_tp
    devs = jax.devices()[:ndev]
    assert len(devs) == ndev
    per_dev = [build_call(ls, kind, layer, inputs, num_req, interpret, attn_heads=attn_heads,
                          h0=jj * attn_heads, attn_tp=attn_tp, served_dtypes=served_dtypes,
                          reps=reps, pipeline=pipeline, races=races)
               for jj in range(ndev)]
    fn = per_dev[0][0]
    ops = [jnp.stack([jnp.asarray(o[i]) for _, o in per_dev]) for i in range(len(per_dev[0][1]))]
    mesh = Mesh(np.array(devs), ("d",))
    g = jax.jit(jax.shard_map(lambda *a: [x[None] for x in fn(*[y[0] for y in a])], mesh=mesh,
                              in_specs=(Ps("d"),) * len(ops), out_specs=Ps("d"), check_vma=False))
    return g, ops


def run_group(ls, kind, layer, inputs, attn_tp=8, attn_heads=16, num_req=None, interpret=True,
              served_dtypes=False, pipeline=False, races=False):
    """The phase on attn_tp devices of a 1-D mesh "d" (group TP): every device
    holds the same caches and attends heads [16 jj, 16 jj + 16) of all the
    requests; device jj scores 1 / attn_tp of each request's columns. Returns
    per device (caches, out)."""
    g, ops = group_call(ls, kind, layer, inputs, attn_tp, attn_heads, num_req, interpret,
                        served_dtypes, pipeline=pipeline, races=races)
    res = g(*ops)
    n_c = len(ls.arrays)
    return [([np.asarray(c[j]) for c in res[:n_c]], np.asarray(res[n_c][j], np.float32))
            for j in range(attn_tp)]


def run_phase(ls, kind, layer, inputs, num_req=None, interpret=True, debug_topk=False,
              attn_heads=128, h0=0, strict_causal=False, pipeline=False, races=False):
    fn, ops = build_call(ls, kind, layer, inputs, num_req, interpret, debug_topk,
                         attn_heads=attn_heads, h0=h0, strict_causal=strict_causal,
                         pipeline=pipeline, races=races)
    n_c = len(ls.arrays)
    res = fn(*ops)
    AP.DEBUG_TOPK = AP.DEBUG_SCORES = AP.DEBUG_LENS = None
    if debug_topk:
        return list(res[:n_c]), res[n_c], res[n_c + 1:n_c + 4]
    return list(res[:n_c]), res[n_c]


def reference(ls, kind, layer, inputs, strict_causal=False):
    """Expected (caches, out rows [R, Q, 128, 512] f32)."""
    R = len(ls.seq_lens)
    arrays = [np.asarray(a).copy() for a in ls.arrays]
    seq = jnp.asarray(ls.seq_lens)
    w = ls.weights
    ai = lambda role: ls.role_arr[role][ls.role_layers[role].index(layer)]  # noqa: E731
    tb = lambda role: jnp.asarray(ls.tables[(role, layer)])  # noqa: E731
    # SWA rows
    swa = arrays[ai("swa")].reshape(-1, C.SWA_PAGE, 2, 128)
    words = np.asarray(L.bf16_tokens_to_words(inputs["kv"][:, :Q]))
    for r in range(R):
        for t in range(Q):
            pos = ls.seq_lens[r] - Q + t
            swa[ls.swa_table[layer][r, pos // C.SWA_PAGE], pos % C.SWA_PAGE] = words[r, t]
    pos = seq[:, None] - Q + jnp.arange(Q)[None, :]
    ci = sum(k == "csa" for k in ls.kinds[:layer])
    # compressor stores (kv_write is bit-exact vs the stock goldens)
    if kind == "hca":
        st = inputs["comp"][:, :Q]
        st = st.at[:, :, 512:].add(w["comp_ape"][layer][pos % 128][..., :512])
        s_new, kv_new = kv_write.compress_store(
            "hca", st, seq, jnp.asarray(arrays[ai("state")]), tb("state"),
            jnp.asarray(arrays[ai("kv")]), tb("kv"), w["comp_norm"][layer], w["comp_cos_sin"],
            state_block_size=AP.SBS["hca"], interpret=True)
        arrays[ai("state")], arrays[ai("kv")] = np.asarray(s_new), np.asarray(kv_new)
    else:
        st = inputs["comp"][:, :Q]
        st = st.at[:, :, 1024:].add(w["comp_ape"][layer][pos % 4][..., :1024])
        kv_new, rope_new = kv_write.compress_store(
            "csa", st, seq, None, tb("state"), jnp.asarray(arrays[ai("kv")]), tb("kv"),
            w["comp_norm"][layer], w["comp_cos_sin"], jnp.asarray(arrays[ai("kv_rope")]),
            state_block_size=AP.SBS["csa"], shared_state=True, interpret=True)
        arrays[ai("kv")], arrays[ai("kv_rope")] = np.asarray(kv_new), np.asarray(rope_new)
        ist = inputs["idx_comp"][:, :Q]
        ist = ist.at[:, :, 256:].add(w["idx_comp_ape"][ci][pos % 4][..., :256])
        (i_new,) = kv_write.compress_store(
            "idx", ist, seq, None, tb("indexer_state"), jnp.asarray(arrays[ai("indexer_k")]),
            tb("indexer_k"), w["idx_comp_norm"][ci], w["idx_cos_sin"],
            state_block_size=AP.SBS["idx"], shared_state=True, interpret=True)
        arrays[ai("indexer_k")] = np.asarray(i_new)
    q = inputs["q"][:, :Q * 128].reshape(R, Q, 128, 512).astype(jnp.bfloat16)
    sinks = w["attn_sink"][layer]
    swa_words = jnp.asarray(ls.arrays[ai("swa")])  # window reads see the pre-write cache
    if kind == "hca":
        out = attn_ref.swa_hca_reference(
            q, inputs["kv"][:, :Q], swa_words, jnp.asarray(ls.swa_table[layer]),
            jnp.asarray(arrays[ai("kv")]), tb("kv"), seq, sinks, sm_scale=AP.SM_SCALE)
        return arrays, out, None
    qi, wi = attn_ref.indexer_q_reference(inputs["idx_q"][:, :Q], inputs["idx_w"][:, :Q, :64],
                                          w["idx_cos_sin"], pos)
    sc, rl = attn_ref.indexer_scores_reference(qi, wi, jnp.asarray(arrays[ai("indexer_k")]),
                                               tb("indexer_k"), seq, strict_causal=strict_causal)
    topk = attn_ref.topk_reference(sc, rl, AP.TOPK)
    out = attn_ref.csa_reference(
        q, inputs["kv"][:, :Q], swa_words, jnp.asarray(ls.swa_table[layer]),
        jnp.asarray(arrays[ai("kv")]), jnp.asarray(arrays[ai("kv_rope")]), tb("kv"), topk, seq,
        sinks, sm_scale=AP.SM_SCALE)
    return arrays, out, topk


def reference_with_topk(ls, layer, inputs, exp_caches, topk):
    """CSA attention reference over given top-k lists [R, Q, 1024]."""
    R = len(ls.seq_lens)
    ai = lambda role: ls.role_arr[role][ls.role_layers[role].index(layer)]  # noqa: E731
    q = inputs["q"][:, :Q * 128].reshape(R, Q, 128, 512).astype(jnp.bfloat16)
    return attn_ref.csa_reference(
        q, inputs["kv"][:, :Q], jnp.asarray(ls.arrays[ai("swa")]), jnp.asarray(ls.swa_table[layer]),
        jnp.asarray(exp_caches[ai("kv")]), jnp.asarray(exp_caches[ai("kv_rope")]),
        jnp.asarray(ls.tables[("kv", layer)]), jnp.asarray(topk), jnp.asarray(ls.seq_lens),
        ls.weights["attn_sink"][layer], sm_scale=AP.SM_SCALE)


def stock_topk_equal(ls, layer, inputs, exp_caches, topk, strict_causal=False):
    """Rows whose kernel top-k equals stock rope_quant + streamindex_topk (device)."""
    from bench_attn_device import stock_indexer
    from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
    from vllm_torchtpu.kernels.deepseek_v4.rope import rope_quant
    ai = ls.role_arr["indexer_k"][ls.role_layers["indexer_k"].index(layer)]
    u8 = csa_cache_layout.words_to_slabs(jnp.asarray(exp_caches[ai]))
    R = len(ls.seq_lens)
    seq = jnp.asarray(ls.seq_lens)
    pos = (seq[:, None] - Q + jnp.arange(Q)[None, :]).astype(jnp.int32)
    ci = sum(k == "csa" for k in ls.kinds[:layer])
    del ci
    q8, qs = rope_quant(inputs["idx_q"][:, :Q].reshape(R * Q, 64, 128), pos.reshape(-1),
                        ls.weights["idx_cos_sin"], quant_dtype=jnp.float8_e4m3fn)
    w = inputs["idx_w"][:, :Q, :64].reshape(R * Q, 64) * (128 ** -0.5) * (64 ** -0.5) * qs.reshape(R * Q, 64)
    stk = np.asarray(stock_indexer(q8, w, u8, seq, jnp.asarray(ls.tables[("indexer_k", layer)]),
                                   strict_causal=strict_causal))
    return int(np.sum(np.all(stk.reshape(R, Q, -1) == np.asarray(topk), axis=-1)))
