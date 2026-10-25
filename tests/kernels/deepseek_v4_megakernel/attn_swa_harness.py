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
"""Standalone harness of owner_attention_phase kind "swa" (the DSpark draft
layers) and a NumPy reference of the stock path: the context rows inserted
into the layer's SWA cache, then the draft block (K rows at base..base+K-1)
attending its sliding window with in-block bidirectional visibility and the
attention sink (mla_swa non_causal_block), its kv stored over the cache."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_core as core
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_layout as L
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_ref
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_phases as SP

META_ROLES = ("swa", "kv", "state", "kv_rope", "indexer_k", "indexer_state", "swa_pages")
PAGE, W, CTX_ROWS = AP.SWA_PAGE, AP.WINDOW, 8


@dataclasses.dataclass
class SwaSet:
    K: int
    bases: np.ndarray  # [B] first draft position
    ctx: list  # per request: (first context position, rows)
    arrays: list  # per layer: int32 [pages, 2 * PAGE, 128]
    tables: list  # per layer: [B, max_ctx / PAGE] logical -> physical page
    sinks: np.ndarray  # f32 [L, 128]
    max_ctx: int

    @property
    def seq_lens(self):
        return (self.bases + self.K).astype(np.int32)


def build(rng, n_layers, bases, ctx, K, max_ctx):
    """ctx[r] = (first context position, rows <= 8), rows consecutive."""
    bases = np.asarray(bases, np.int32)
    B, wp = len(bases), max_ctx // PAGE
    arrays, tables = [], []
    for _ in range(n_layers):
        # the window's pages (3 per request) are distinct pool pages, the rest page 0
        n_pages = 3 * B + 1
        toks = jnp.asarray(rng.standard_normal((n_pages, PAGE, 512)), jnp.bfloat16)
        arrays.append(np.asarray(L.bf16_tokens_to_words(toks)).reshape(n_pages, 2 * PAGE, 128))
        t = np.zeros((B, wp), np.int32)
        perm = 1 + rng.permutation(3 * B)
        for r in range(B):
            lp = max(int(bases[r]) - (W - 1), 0) // PAGE
            for i in range(3):
                if lp + i < wp:
                    t[r, lp + i] = perm[3 * r + i]
        tables.append(t)
    sinks = rng.standard_normal((n_layers, 128)).astype(np.float32)
    return SwaSet(K, bases, list(ctx), arrays, tables, sinks, max_ctx)


def make_inputs(rng, ss):
    B, K = len(ss.bases), ss.K
    q = np.zeros((B, -(-K * 128 // 128) * 128, 512), np.float32)
    q[:, :K * 128] = np.asarray(jnp.asarray(rng.standard_normal((B, K * 128, 512)), jnp.bfloat16),
                                np.float32)
    bf = lambda s: jnp.asarray(rng.standard_normal(s), jnp.bfloat16)  # noqa: E731
    kv = jnp.zeros((B, 8, 512), jnp.bfloat16).at[:, :K].set(bf((B, K, 512)))
    ctx_kv = bf((B, CTX_ROWS, 512))
    return dict(q=jnp.asarray(q), kv=kv, ctx_kv=ctx_kv)


def meta_arrays(ss, layer_of_slot=None):
    """SMEM metadata; every layer gets its own swa / swa_pages tables."""
    B, nl, K = len(ss.bases), len(ss.arrays), ss.K
    flat, n = [], 0
    off = {r: np.zeros(nl, np.int32) for r in META_ROLES}
    wid = {r: np.ones(nl, np.int32) for r in META_ROLES}
    for l_ in range(nl):
        t = ss.tables[l_]
        off["swa"][l_], wid["swa"][l_] = n, t.shape[1]
        flat.append(t.reshape(-1))
        n += t.size
        pg = np.asarray(core.swa_window_pages(jnp.asarray(ss.seq_lens), jnp.asarray(t), q_len=K,
                                              window=W, swa_page=PAGE)).reshape(-1)
        off["swa_pages"][l_], wid["swa_pages"][l_] = n, 3
        flat.append(pg)
        n += pg.size
    pos = (ss.bases[:, None] + np.arange(K)[None, :]).reshape(-1).astype(np.int32)
    cpos = np.full((B, CTX_ROWS), -1, np.int32)
    for r, (c0, nc) in enumerate(ss.ctx):
        cpos[r, :nc] = c0 + np.arange(nc)
    return dict(seq_lens=ss.seq_lens, positions=pos, ctx_positions=cpos.reshape(-1),
                bt_flat=np.concatenate(flat).astype(np.int32), bt_off=off, bt_w=wid)


def build_call(ss, layer, inputs, num_req=None, interpret=True, reps=1, attn_heads=128, h0=0):
    """attn_heads < 128: the phase attends heads [h0, h0 + attn_heads) (group TP)."""
    B, K, nl = len(ss.bases), ss.K, len(ss.arrays)
    num_req = B if num_req is None else num_req
    geom = SP.StepGeometry(R=16, T=16, q=K, B=B, layer_kinds=("swa",) * nl, max_ctx=ss.max_ctx,
                           attn_heads=attn_heads)
    sinks = jnp.asarray(ss.sinks)
    if attn_heads != 128:
        q = inputs["q"][:, :K * 128].reshape(B, K, 128, 512)[:, :, h0:h0 + attn_heads]
        rp = -(-K * attn_heads // 128) * 128
        q = jnp.pad(q.reshape(B, K * attn_heads, 512), ((0, 0), (0, rp - K * attn_heads), (0, 0)))
        inputs = dict(inputs, q=q)
        sinks = jnp.roll(sinks, -h0, axis=1)
    m = meta_arrays(ss)
    seq = m["seq_lens"].copy()
    seq[num_req:] = 0
    smem_ops = ([np.array([num_req], np.int32), seq, m["positions"], AP.pad_tables(m["bt_flat"])]
                + [m["bt_off"][r] for r in META_ROLES] + [m["bt_w"][r] for r in META_ROLES]
                + [np.array([layer], np.int32), m["ctx_positions"]])
    vmem_ops = [inputs["q"], inputs["kv"], inputs["ctx_kv"]]
    one = jnp.zeros((nl, 128), jnp.float32)
    w_ops = [sinks, one, one, one, one, one, one, one]  # only the sink is read
    n_s, n_v, n_w, n_c = len(smem_ops), 3, len(w_ops), nl
    scratch = AP.owner_attention_scratch_shapes(geom, False)

    def kernel(*refs):
        s = refs[:n_s]
        q, kv, ctx_kv = refs[n_s:n_s + n_v]
        w = refs[n_s + n_v:n_s + n_v + n_w]
        caches = refs[n_s + n_v + n_w + n_c:n_s + n_v + n_w + 2 * n_c]  # aliased outputs
        out = refs[n_s + n_v + n_w + 2 * n_c]
        scr = refs[n_s + n_v + n_w + 2 * n_c + 1:]
        meta = SP.AttnMeta(num_req=s[0], seq_lens=s[1], positions=s[2], bt_flat=s[3],
                           bt_off=dict(zip(META_ROLES, s[4:11])),
                           bt_w=dict(zip(META_ROLES, s[11:18])), ctx_positions=s[19])
        inp = SP.AttnInputs(q=q, kv=kv, comp=None, idx_q=None, idx_w=None, idx_comp=None,
                            ctx_kv=ctx_kv)
        none = SP.LayerRefs([], [])
        cs = SP.AttnCaches(swa=SP.LayerRefs(list(caches), list(range(nl))), kv=none, state=none,
                           kv_rope=none, indexer_k=none, indexer_state=none)
        AP.reset_table_stage(geom, scr, False)

        def once(i, c):
            SP.owner_attention_phase(geom, "swa", s[18][0], 0, meta, inp, cs, SP.AttnWeights(*w),
                                     out, SP.PoolLease(None, 0, 0), scr)
            return c

        jax.lax.fori_loop(0, reps, once, 0)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    arrays = [jnp.asarray(a) for a in ss.arrays]
    fn = pl.pallas_call(
        kernel,
        out_shape=[jax.ShapeDtypeStruct(a.shape, a.dtype) for a in arrays]
        + [jax.ShapeDtypeStruct(inputs["q"].shape, inputs["q"].dtype)],
        in_specs=[smem] * 3 + [hbm] + [smem] * (n_s - 4) + [vmem] * n_v + [hbm] * (n_w + n_c),
        out_specs=[hbm] * n_c + [vmem],
        scratch_shapes=scratch,
        input_output_aliases={n_s + n_v + n_w + i: i for i in range(n_c)},
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 1024 * 1024),
        interpret=pltpu.InterpretParams() if interpret else False,
    )
    return fn, [jnp.asarray(x) for x in smem_ops] + vmem_ops + w_ops + arrays


def run_phase(ss, layer, inputs, num_req=None, interpret=True, attn_heads=128, h0=0):
    fn, ops = build_call(ss, layer, inputs, num_req, interpret, attn_heads=attn_heads, h0=h0)
    res = fn(*ops)
    return [np.asarray(a) for a in res[:-1]], np.asarray(res[-1], np.float32)


def _tokens(words):  # int32 [..., 2, 128] -> f32 [..., 512]
    w = words.astype(np.int64) & 0xFFFFFFFF
    halves = [((w[..., s, :] & 0xFFFF) << 16, w[..., s, :] & 0xFFFF0000) for s in range(2)]
    parts = [h for pair in halves for h in pair]
    return np.concatenate([p.astype(np.uint32).view(np.float32) for p in parts], axis=-1)


def reference(ss, layer, inputs, num_req=None, stock=False):
    """(caches after the phase, out [B, K, 128, 512] f32 for the active requests).

    `stock`: the SWA-only stock kernel's arithmetic (`attn_ref._attend_stock`)."""
    B, K = len(ss.bases), ss.K
    num_req = B if num_req is None else num_req
    caches = [a.copy() for a in ss.arrays]
    cache = caches[layer].reshape(-1, PAGE, 2, 128)
    tab = ss.tables[layer]
    ctx_w = np.asarray(L.bf16_tokens_to_words(inputs["ctx_kv"]))
    kv_w = np.asarray(L.bf16_tokens_to_words(inputs["kv"]))
    q = np.asarray(inputs["q"], np.float32)[:, :K * 128].reshape(B, K, 128, 512)
    out = np.zeros((B, K, 128, 512), np.float32)
    for r in range(num_req):
        c0, nc = ss.ctx[r]
        for c in range(nc):  # context insert first ...
            p = c0 + c
            cache[tab[r, p // PAGE], p % PAGE] = ctx_w[r, c]
        base = int(ss.bases[r])
        for t in range(K):  # ... then the draft rows (stock: the draft forward's write)
            p = base + t
            cache[tab[r, p // PAGE], p % PAGE] = kv_w[r, t]
        lo = max(base - (W - 1), 0)
        pos = np.arange(lo, base + K)
        keys = _tokens(cache[tab[r, pos // PAGE], pos % PAGE])  # [n, 512]
        for t in range(K):
            qp = base + t
            vis = ((qp - pos >= 0) & (qp - pos < W)) | ((pos >= base) & (pos < base + K))
            if stock:
                out[r, t] = attn_ref._attend_stock(q[r, t], keys, jnp.asarray(vis), len(pos),
                                                   jnp.asarray(ss.sinks[layer]), AP.SM_SCALE,
                                                   False)
                continue
            s = (q[r, t] @ keys.T) * AP.SM_SCALE  # [128, n]
            s = np.where(vis[None, :], s, -np.inf)
            m = s.max(axis=1, keepdims=True)
            p = np.exp(s - m)
            l_ = p.sum(axis=1, keepdims=True) + np.exp(ss.sinks[layer][:, None] - m)
            out[r, t] = (p / l_) @ keys
    return caches, out
