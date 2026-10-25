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
"""CSA attention of DeepSeek-V4 verify windows, as megakernel steps.

Three stages per request, with a seam between the second and the third:

  1. `index_scores`: the indexer logits of the ``q_len`` query rows against
     every visible compressed row, streamed page by page from the served
     seq-along-lane indexer K cache, in the serving kernel's arithmetic order
     (head-major fp8 ``q @ k``, f32 relu-weighted head sum head 0 first, times
     the key scale). Invisible columns are ``-inf``.
  2. top-k selection, outside this module (`topk_union_placeholder` is the
     plain-JAX stand-in): it hands stage 3 the request's *union* of the per-row
     top-k sets, ascending, with a bit per query row saying who selected it.
  3. `csa_attend`: gathers each union row (NoPE 512 B + its RoPE row) once,
     dequantizes it as `sparse_mla` does and continues the `VerifyWindow`
     softmax over it, row t seeing exactly the entries with bit t.
"""

import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_core as core
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_layout as L

NN = core.NN


# ------------------------------------------------------------------ indexer


def visible_rows(n, pos, strict_causal, ratio=4):
    """Compressed rows a query at `pos` may select of the `n` stored: with
    `strict_causal` only the groups complete at `pos`, else also the group
    `pos` is in (`streamindex_topk._effective_row_lengths`)."""
    return jnp.minimum(n, (pos + 1) // ratio if strict_causal else pos // ratio + 1)


@dataclasses.dataclass(frozen=True)
class IndexerConfig:
    max_req: int
    pages_per_req: int  # row length of the indexer page table
    n_cols: int  # score columns per row (a multiple of blk_pages * page)
    q_len: int = 7
    n_heads: int = 64
    head_dim: int = 128
    ratio: int = 4
    page: int = 256  # compressed rows per indexer page
    blk_pages: int = 8
    strict_causal: bool = False  # visible_rows

    @property
    def groups(self) -> int:  # word rows per token record: keys + scale
        return self.head_dim // 4 + 1

    @property
    def dma_groups(self) -> int:  # rows read per page: whole (8, 128) tiles
        return core.cdiv(self.groups, 8) * 8

    @property
    def blk_rows(self) -> int:
        return self.blk_pages * self.page


class IndexerScratch(NamedTuple):
    buf: object  # int32 [2, blk_pages, dma_groups, page]
    wb: object  # f32 [n_heads, 8, blk_rows]: weights broadcast along lanes
    sems: object  # DMA [2]


def indexer_scratch_shapes(cfg: IndexerConfig) -> IndexerScratch:
    return IndexerScratch(
        buf=pltpu.VMEM((2, cfg.blk_pages, cfg.dma_groups, cfg.page), jnp.int32),
        wb=pltpu.VMEM((cfg.n_heads, 8, cfg.blk_rows), jnp.float32),
        sems=pltpu.SemaphoreType.DMA((2,)),
    )


def index_scores(
    cfg: IndexerConfig,
    *,
    num_req,
    seq_lens,  # SMEM int32 [max_req]
    idx_pt,  # SMEM int32 [max_req * pages_per_req]
    q_ref,  # VMEM fp8 [max_req, n_heads * 8, head_dim], row h * 8 + t (t < q_len)
    w_ref,  # VMEM f32 [max_req, 8, n_heads]: weights with the q scales folded in
    idx_hbm,  # int32 [N, >= groups, page] seq-along-lane words
    scores_ref,  # VMEM f32 [max_req, 8, n_cols]
    scr: IndexerScratch,
):
    H = cfg.n_heads

    def n_rows(r):
        return seq_lens[r] // cfg.ratio

    def start(r, b, slot):
        last = jnp.maximum(core.cdiv(n_rows(r), cfg.page) - 1, 0)
        for p in range(cfg.blk_pages):
            pg = jnp.minimum(b * cfg.blk_pages + p, last)
            phys = jnp.clip(idx_pt[r * cfg.pages_per_req + pg], 0, idx_hbm.shape[0] - 1)
            pltpu.make_async_copy(
                idx_hbm.at[phys, pl.ds(0, cfg.dma_groups)], scr.buf.at[slot, p], scr.sems.at[slot]
            ).start()

    def wait(slot):
        blk = scr.buf.at[slot]
        pltpu.make_async_copy(blk, blk, scr.sems.at[slot]).wait()

    def block_scores(r, slot, col0, row_len):
        """Scores of one block: its pages side by side, so the head sum (a
        dependent chain per element, kept in the serving order) has
        blk_pages * page / 1024 independent vreg chains."""
        words = [scr.buf[slot, p] for p in range(cfg.blk_pages)]  # [dma_groups, page]
        k_t = jnp.concatenate(
            [pltpu.bitcast(w[: cfg.groups - 1], jnp.float8_e4m3fn) for w in words], axis=1)
        st = lax.dot_general(q_ref[r], k_t, NN, preferred_element_type=jnp.float32)
        acc = jnp.maximum(st[0:8], 0.0) * scr.wb[0]
        for h in range(1, H):
            acc = acc + jnp.maximum(st[8 * h : 8 * h + 8], 0.0) * scr.wb[h]
        scale = jnp.concatenate([w[cfg.groups - 1 : cfg.groups] for w in words], axis=1)
        acc = acc * L.e8m0_from_byte(scale & 0xFF)
        col = col0 + lax.broadcasted_iota(jnp.int32, (1, cfg.blk_rows), 1)
        scores_ref[r, :, pl.ds(col0, cfg.blk_rows)] = jnp.where(col < row_len, acc, -jnp.inf)

    def req_body(r, g):
        w = w_ref[r]  # [8, H]
        for h in range(H):
            scr.wb[h] = jnp.broadcast_to(w[:, h : h + 1], (8, cfg.blk_rows))
        n = n_rows(r)
        t = lax.broadcasted_iota(jnp.int32, (8, 1), 0)
        row_len = visible_rows(n, seq_lens[r] - cfg.q_len + t, cfg.strict_causal, cfg.ratio)
        nb = jnp.maximum(1, core.cdiv(n, cfg.blk_rows))

        def blk_body(b, g):
            slot = g % 2
            wait(slot)
            last = b + 1 == nb
            nr = jnp.where(last, jnp.minimum(r + 1, num_req - 1), r)
            start(nr, jnp.where(last, jnp.where(r + 1 < num_req, 0, b), b + 1), 1 - slot)
            block_scores(r, slot, b * cfg.blk_rows, row_len)
            return g + 1

        return lax.fori_loop(0, nb, blk_body, g)

    @pl.when(num_req > 0)
    def _():
        start(0, 0, 0)
        g = lax.fori_loop(0, num_req, req_body, jnp.int32(0))
        wait(g % 2)


def _indexer_kernel(cfg, nreq_ref, seq_ref, pt_ref, q_ref, w_ref, idx_hbm, out_ref, *scratch):
    index_scores(cfg, num_req=nreq_ref[0], seq_lens=seq_ref, idx_pt=pt_ref, q_ref=q_ref,
                 w_ref=w_ref, idx_hbm=idx_hbm, scores_ref=out_ref,
                 scr=IndexerScratch(*scratch))


def indexer_scores(
    q_idx,  # fp8 e4m3 [R, q_len, n_heads, 128]
    weights,  # f32 [R, q_len, n_heads] (query scales and softmax scale folded in)
    idx_words,  # int32 [N, G, page] seq-along-lane words
    idx_table,  # int32 [R, P]
    seq_lens,  # int32 [R]
    *,
    max_len: int,
    blk_pages: int = 8,
    strict_causal: bool = False,
    interpret=False,
    vmem_limit_bytes: int = 60 * 1024 * 1024,
):
    """Standalone wrapper: f32 scores ``[R, q_len, n_cols]`` and the visible row
    lengths ``[R, q_len]``; columns at or past a row's length are ``-inf`` up to
    the last page it touches (beyond that, unwritten)."""
    R, T, Hi, D = q_idx.shape
    page = idx_words.shape[2]
    ratio = 4
    blk_rows = blk_pages * page
    n_cols = core.cdiv(max_len // ratio, blk_rows) * blk_rows
    cfg = IndexerConfig(max_req=R, pages_per_req=idx_table.shape[1], n_cols=n_cols, q_len=T,
                        n_heads=Hi, head_dim=D, page=page, blk_pages=blk_pages,
                        strict_causal=strict_causal)
    q_hm = jnp.pad(q_idx.transpose(0, 2, 1, 3), ((0, 0), (0, 0), (0, 8 - T), (0, 0)))
    w = jnp.pad(weights.astype(jnp.float32), ((0, 0), (0, 8 - T), (0, 0)))
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scores = pl.pallas_call(
        functools.partial(_indexer_kernel, cfg),
        out_shape=jax.ShapeDtypeStruct((R, 8, n_cols), jnp.float32),
        in_specs=[smem, smem, smem, vmem, vmem, hbm],
        out_specs=vmem,
        scratch_shapes=list(indexer_scratch_shapes(cfg)),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes,
                                             disable_bounds_checks=True),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"mk_indexer_r{R}",
    )(
        jnp.array([R], jnp.int32),
        seq_lens.astype(jnp.int32),
        idx_table.astype(jnp.int32).reshape(-1),
        q_hm.reshape(R, Hi * 8, D),
        w,
        idx_words,
    )
    pos = seq_lens[:, None] - T + jnp.arange(T)[None, :]
    row_lens = visible_rows(seq_lens[:, None] // ratio, pos, strict_causal, ratio).astype(jnp.int32)
    return scores[:, :T], row_lens


# ------------------------------------------------------------------ top-k seam

UNION_BITS = 8  # union entry = (compressed index << UNION_BITS) | row bits


def topk_union_placeholder(scores, row_lens, k):
    """Plain-JAX stand-in for the in-kernel top-k (block B1's seam).

    Per row: `attn_ref.topk_reference` (SparseCore semantics). Per request: the
    ascending union of its rows' selections, each entry packed as
    ``index << UNION_BITS | bits`` (bit t: row t selected it), and its length.
    Returns (union int32 ``[R, q_len * k]``, lengths int32 ``[R]``,
    per-row indices int32 ``[R, q_len, k]``).
    """
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_ref

    R, T, n = scores.shape
    idx = attn_ref.topk_reference(scores, row_lens, k)  # [R, T, k]
    bit = (1 << jnp.arange(T, dtype=jnp.int32))[None, :, None]
    contrib = jnp.where(idx >= 0, bit, 0)
    rows = jnp.broadcast_to(jnp.arange(R)[:, None, None], idx.shape)
    bitmap = jnp.zeros((R, n), jnp.int32).at[rows, jnp.maximum(idx, 0)].add(contrib)
    u_max = T * k

    def one(bm):
        (pos,) = jnp.nonzero(bm, size=u_max, fill_value=0)
        return (pos << UNION_BITS) | bm[pos], jnp.sum(bm != 0)

    union, lens = jax.vmap(one)(bitmap)
    return union.astype(jnp.int32), lens.astype(jnp.int32), idx


# ------------------------------------------------------------------ sparse attention


@dataclasses.dataclass(frozen=True)
class CsaConfig:
    max_req: int
    pages_per_req: int  # row length of the CSA page table
    u_max: int  # union capacity per request (a multiple of granule)
    page: int = 256  # compressed rows per NoPE page
    granule: int = 128  # union entries per gather block


class CsaScratch(NamedTuple):
    nope: object  # int32 [2, granule, 128]
    rope: object  # int32 [2, granule, 128]
    sems: object  # DMA [2, 2]: slot x (nope, rope)


def csa_scratch_shapes(cfg: CsaConfig) -> CsaScratch:
    return CsaScratch(
        nope=pltpu.VMEM((2, cfg.granule, L.LANES), jnp.int32),
        rope=pltpu.VMEM((2, cfg.granule, L.LANES), jnp.int32),
        sems=pltpu.SemaphoreType.DMA((2, 2)),
    )


def _col(row):
    """[1, n] -> [n, 1]."""
    n = row.shape[1]
    return jnp.transpose(jnp.broadcast_to(row, (8, n)))[:, 0:1]


def csa_keys(nope_w, rope_w, seg, transposed):
    """Gathered rows -> bf16 keys ``[n, 512]`` (and ``[512, n]`` if transposed).

    nope_w: int32 ``[n, 128]`` NoPE records; rope_w: int32 ``[n, 128]`` RoPE rows
    holding each token in word segment ``seg [n, 1]`` (= index % 4).
    """
    lane = lax.broadcasted_iota(jnp.int32, (1, L.LANES), 1)
    byte = [(nope_w >> (8 * b)) & 0xFF for b in range(4)]
    sc = L.e8m0_from_byte(jnp.where(lane < 64, pltpu.roll(byte[3], 64, 1), byte[3]))
    blk = [L.e4m3_from_byte(b) * sc for b in byte]
    a = jnp.where(seg == 0, rope_w, jnp.where(seg == 1, pltpu.roll(rope_w, 96, 1),
                  jnp.where(seg == 2, pltpu.roll(rope_w, 64, 1), pltpu.roll(rope_w, 32, 1))))
    lo = lax.bitcast_convert_type(a << 16, jnp.float32)  # rope dims 0..31 in lanes 0..31
    hi = lax.bitcast_convert_type(a & -65536, jnp.float32)  # dims 32..63
    blk[3] = jnp.where(lane < 64, blk[3], jnp.where(lane < 96, pltpu.roll(lo, 64, 1),
                                                     pltpu.roll(hi, 96, 1)))
    kv = jnp.concatenate(blk, axis=1).astype(jnp.bfloat16)
    if not transposed:
        return kv, None
    return kv, jnp.concatenate([b.T for b in blk], axis=0).astype(jnp.bfloat16)


def csa_attend(
    wcfg: core.WindowConfig,
    ccfg: CsaConfig,
    *,
    num_req,
    union_smem,  # SMEM int32 [max_req * u_max], from the top-k seam
    union_vmem,  # VMEM int32 [max_req, u_max // 128, 128], the same entries
    union_len,  # SMEM int32 [max_req]
    csa_pt,  # SMEM int32 [max_req * pages_per_req]
    nope_hbm,  # int32 [N, page, 128]
    rope_hbm,  # int32 [N, page // 4, 128]
    win: core.VerifyWindow,
    cscr: CsaScratch,
):
    G = ccfg.granule
    T = wcfg.q_len
    page_bits = ccfg.page.bit_length() - 1
    assert ccfg.page == 1 << page_bits

    def start(r, b, slot):
        """Gathers union entries [b G, b G + G) of request r (the tail re-gathers
        the last entry; its keys are masked)."""
        last = jnp.maximum(union_len[r] - 1, 0)
        base = r * ccfg.u_max
        for k in range(G):
            e = union_smem[base + jnp.minimum(b * G + k, last)]
            idx = e >> UNION_BITS
            lp = jnp.clip(idx >> page_bits, 0, ccfg.pages_per_req - 1)
            phys = jnp.clip(csa_pt[r * ccfg.pages_per_req + lp], 0, nope_hbm.shape[0] - 1)
            slot_in_page = idx & (ccfg.page - 1)
            pltpu.make_async_copy(nope_hbm.at[phys, pl.ds(slot_in_page, 1)],
                                  cscr.nope.at[slot, pl.ds(k, 1)], cscr.sems.at[slot, 0]).start()
            pltpu.make_async_copy(rope_hbm.at[phys, pl.ds(slot_in_page >> 2, 1)],
                                  cscr.rope.at[slot, pl.ds(k, 1)], cscr.sems.at[slot, 1]).start()

    def wait(r, b, slot):
        for i, buf in enumerate((cscr.nope, cscr.rope)):
            blk = buf.at[slot]
            pltpu.make_async_copy(blk, blk, cscr.sems.at[slot, i]).wait()

    def step(r, b, slot):
        ent = union_vmem[r, pl.ds(b, 1), :]  # [1, G]
        valid = b * G + lax.broadcasted_iota(jnp.int32, (1, G), 1) < union_len[r]
        bits = jnp.where(valid, ent & ((1 << T) - 1), 0)
        seg = _col((ent >> UNION_BITS) & 3)
        kv, kv_t = csa_keys(cscr.nope[slot], cscr.rope[slot], seg, wcfg.transposed)
        if wcfg.transposed:
            t = lax.broadcasted_iota(jnp.int32, (1, wcfg.rows_pad), 1) // wcfg.heads
            vis = ((_col(bits) >> t) & 1) == 1
        else:
            t = lax.broadcasted_iota(jnp.int32, (wcfg.rows8, 1), 0) // wcfg.heads
            vis = ((bits >> t) & 1) == 1
        win.update(r, kv, kv_t, vis, first=False)

    core.run_window_stream(
        win, num_req=num_req,
        n_blocks=lambda r: jnp.maximum(1, core.cdiv(union_len[r], G)),
        start=start, wait=wait, step=step,
    )


# ------------------------------------------------- sparse attention, paged gather


@dataclasses.dataclass(frozen=True)
class CsaPagedConfig:
    max_req: int
    pages_per_req: int
    u_max: int
    page: int = 256
    blk_pages: int = 4  # NoPE/RoPE pages streamed per block
    granule: int = 128

    @property
    def blk_rows(self) -> int:
        return self.blk_pages * self.page


class CsaPagedScratch(NamedTuple):
    nope: object  # int32 [2, blk_rows, 128]
    rope: object  # int32 [2, blk_rows // 4, 128]
    stage_n: object  # int32 [granule, 128]
    stage_r: object  # int32 [granule, 128]
    stage_e: object  # int32 [granule, 128]: each union entry splat along lanes
    sems: object  # DMA [2, 2]


def csa_paged_scratch_shapes(cfg: CsaPagedConfig) -> CsaPagedScratch:
    G = cfg.granule
    return CsaPagedScratch(
        nope=pltpu.VMEM((2, cfg.blk_rows, L.LANES), jnp.int32),
        rope=pltpu.VMEM((2, cfg.blk_rows // 4, L.LANES), jnp.int32),
        stage_n=pltpu.VMEM((G, L.LANES), jnp.int32),
        stage_r=pltpu.VMEM((G, L.LANES), jnp.int32),
        stage_e=pltpu.VMEM((G, L.LANES), jnp.int32),
        sems=pltpu.SemaphoreType.DMA((2, 2)),
    )


def csa_attend_paged(
    wcfg: core.WindowConfig,
    ccfg: CsaPagedConfig,
    *,
    num_req,
    union_smem,  # SMEM int32 [max_req * u_max]
    union_len,  # SMEM int32 [max_req]
    seq_lens,  # SMEM int32 [max_req]
    csa_pt,  # SMEM int32 [max_req * pages_per_req]
    nope_hbm,
    rope_hbm,
    win: core.VerifyWindow,
    cscr: CsaPagedScratch,
):
    """`csa_attend` with a TensorCore-friendly gather: stream every NoPE/RoPE
    page of the request (2 DMAs a page instead of 2 per row) and pick the union
    rows out of VMEM with scalar-addressed row copies."""
    G, PB, page = ccfg.granule, ccfg.blk_pages, ccfg.page
    T = wcfg.q_len
    # Union rows are in-block here (local >= 0), so shifts replace the signed
    # floor divisions, which cost ~10 scalar ops a row.
    steps = max(1, ccfg.u_max.bit_length())

    def n_pages(r):
        return core.cdiv(seq_lens[r] // 4, page)

    def start(r, b, slot):
        last = jnp.maximum(n_pages(r) - 1, 0)
        for p in range(PB):
            pg = jnp.minimum(b * PB + p, last)
            phys = jnp.clip(csa_pt[r * ccfg.pages_per_req + pg], 0, nope_hbm.shape[0] - 1)
            pltpu.make_async_copy(nope_hbm.at[phys], cscr.nope.at[slot, pl.ds(p * page, page)],
                                  cscr.sems.at[slot, 0]).start()
            pltpu.make_async_copy(rope_hbm.at[phys],
                                  cscr.rope.at[slot, pl.ds(p * page // 4, page // 4)],
                                  cscr.sems.at[slot, 1]).start()

    def wait(r, b, slot):
        for i, buf in enumerate((cscr.nope, cscr.rope)):
            blk = buf.at[slot]
            pltpu.make_async_copy(blk, blk, cscr.sems.at[slot, i]).wait()

    def lower_bound(r, key):
        """First union entry of request r whose index is >= key."""
        base = r * ccfg.u_max

        def body(_, c):
            lo, hi = c
            mid = (lo + hi) // 2
            less = (union_smem[base + jnp.minimum(mid, ccfg.u_max - 1)] >> UNION_BITS) < key
            go = lo < hi
            return jnp.where(go & less, mid + 1, lo), jnp.where(go & ~less, mid, hi)

        return lax.fori_loop(0, steps, body, (jnp.int32(0), union_len[r]))[0]

    def step(r, b, slot):
        key0 = b * ccfg.blk_rows
        lo = lower_bound(r, key0)
        hi = lower_bound(r, key0 + ccfg.blk_rows)

        def granule(g, c):
            base = lo + g * G

            def gather(k, c2):
                e = union_smem[r * ccfg.u_max + jnp.minimum(base + k, hi - 1)]
                local = (e >> UNION_BITS) - key0
                cscr.stage_n[pl.ds(k, 1), :] = cscr.nope[slot, pl.ds(local, 1), :]
                cscr.stage_r[pl.ds(k, 1), :] = cscr.rope[slot, pl.ds(local >> 2, 1), :]
                cscr.stage_e[pl.ds(k, 1), :] = jnp.full((1, L.LANES), e, jnp.int32)
                return c2

            lax.fori_loop(0, G, gather, 0, unroll=8)
            e_col = cscr.stage_e[:, 0:1]  # [G, 1]
            valid = base + lax.broadcasted_iota(jnp.int32, (G, 1), 0) < hi
            bits = jnp.where(valid, e_col & ((1 << T) - 1), 0)
            kv, kv_t = csa_keys(cscr.stage_n[...], cscr.stage_r[...],
                                (e_col >> UNION_BITS) & 3, wcfg.transposed)
            if wcfg.transposed:
                t = lax.broadcasted_iota(jnp.int32, (1, wcfg.rows_pad), 1) // wcfg.heads
                vis = ((bits >> t) & 1) == 1
            else:
                t = lax.broadcasted_iota(jnp.int32, (wcfg.rows8, 1), 0) // wcfg.heads
                bits_row = jnp.transpose(jnp.broadcast_to(bits, (G, L.LANES)))[0:1, :]
                vis = ((bits_row >> t) & 1) == 1
            win.update(r, kv, kv_t, vis, first=False)
            return c

        lax.fori_loop(0, core.cdiv(hi - lo, G), granule, 0)

    core.run_window_stream(
        win, num_req=num_req,
        n_blocks=lambda r: jnp.maximum(1, core.cdiv(n_pages(r), PB)),
        start=start, wait=wait, step=step,
    )


def _csa_paged_kernel(wcfg, ccfg, nreq_ref, seq_ref, swa_pt_ref, ulen_ref, csa_pt_ref,
                      union_smem, q_ref, nkv_ref, sink_ref, swa_hbm, nope_hbm, rope_hbm,
                      out_ref, *scratch):
    nw = len(core.WindowScratch._fields)
    win = core.VerifyWindow(
        wcfg, seq_lens=seq_ref, swa_pt=swa_pt_ref, q_ref=q_ref, new_kv_ref=nkv_ref,
        sink_ref=sink_ref, swa_hbm=swa_hbm, out_ref=out_ref,
        scr=core.WindowScratch(*scratch[:nw]),
    )
    csa_attend_paged(wcfg, ccfg, num_req=nreq_ref[0], union_smem=union_smem,
                     union_len=ulen_ref, seq_lens=seq_ref, csa_pt=csa_pt_ref,
                     nope_hbm=nope_hbm, rope_hbm=rope_hbm, win=win,
                     cscr=CsaPagedScratch(*scratch[nw:]))


# ------------------------------------------------- sparse attention, hybrid gather


def hybrid_plan(union, union_len, *, pages_per_req, page=256, blk_pages=8, k_dense=4):
    """Splits each request's union between pages worth streaming whole (at least
    `k_dense` union rows) and rows gathered one DMA each (plain JAX; the
    in-kernel top-k can emit the same lists).

    Returns a dict of int32 arrays:
      dense_pages [R, P]  logical pages streamed, ascending; n_dense [R]
      dense_ent [R, U]    entries of those pages, ``local_row << 8 | bits`` with
                          ``local_row = (rank % blk_pages) * page + slot``
      dense_off [R, NB+1] first dense entry of each block of blk_pages pages
      sparse_ent [R, U]   the other entries, ``index << 8 | bits``; n_sparse [R]
    """
    R, U = union.shape
    P = pages_per_req
    pos = jnp.arange(U)[None, :]
    valid = pos < union_len[:, None]
    idx = union >> UNION_BITS
    pg = jnp.where(valid, idx // page, P)  # P = sentinel bucket
    rows = jnp.broadcast_to(jnp.arange(R)[:, None], (R, U))
    count = jnp.zeros((R, P + 1), jnp.int32).at[rows, pg].add(1)[:, :P]
    dense = count >= k_dense
    rank = jnp.cumsum(dense, axis=1) - 1  # [R, P]
    n_dense = jnp.sum(dense, axis=1).astype(jnp.int32)
    is_d = valid & jnp.take_along_axis(jnp.pad(dense, ((0, 0), (0, 1))), pg, axis=1)
    is_s = valid & ~is_d
    local = (jnp.take_along_axis(jnp.pad(rank, ((0, 0), (0, 1))), pg, axis=1) % blk_pages
             ) * page + idx % page
    nb = -(-P // blk_pages)

    def one(d_mask, s_mask, u, loc, dense_r, cnt_r, rank_r):
        (dp,) = jnp.nonzero(dense_r, size=P, fill_value=0)
        (di,) = jnp.nonzero(d_mask, size=U, fill_value=0)
        (si,) = jnp.nonzero(s_mask, size=U, fill_value=0)
        dent = (loc[di] << UNION_BITS) | (u[di] & ((1 << UNION_BITS) - 1))
        blk_cnt = jnp.zeros((nb + 1,), jnp.int32).at[
            jnp.where(dense_r, rank_r // blk_pages, nb)].add(cnt_r)[:nb]
        off = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(blk_cnt)])
        return dp, dent, off, u[si]

    dp, dent, off, sent = jax.vmap(one)(is_d, is_s, union, local, dense, count, rank)
    return dict(dense_pages=dp.astype(jnp.int32), n_dense=n_dense,
                dense_ent=dent.astype(jnp.int32), dense_off=off.astype(jnp.int32),
                sparse_ent=sent.astype(jnp.int32),
                n_sparse=jnp.sum(is_s, axis=1).astype(jnp.int32))


class CsaHybridScratch(NamedTuple):
    nope: object  # int32 [2, blk_rows, 128]  dense page blocks
    rope: object  # int32 [2, blk_rows // 4, 128]
    rn: object  # int32 [2, granule, 128]  per-row gathered NoPE
    rr: object  # int32 [2, granule, 128]  per-row gathered RoPE rows
    stage_n: object  # int32 [granule, 128]
    stage_r: object  # int32 [granule, 128]
    stage_e: object  # int32 [granule, 128]
    sems: object  # DMA [2, 4]: slot x (page nope, page rope, row nope, row rope)


def csa_hybrid_scratch_shapes(cfg: CsaPagedConfig) -> CsaHybridScratch:
    G = cfg.granule
    v = lambda *shape: pltpu.VMEM(shape, jnp.int32)  # noqa: E731
    return CsaHybridScratch(
        nope=v(2, cfg.blk_rows, L.LANES), rope=v(2, cfg.blk_rows // 4, L.LANES),
        rn=v(2, G, L.LANES), rr=v(2, G, L.LANES),
        stage_n=v(G, L.LANES), stage_r=v(G, L.LANES), stage_e=v(G, L.LANES),
        sems=pltpu.SemaphoreType.DMA((2, 4)),
    )


def csa_attend_hybrid(
    wcfg: core.WindowConfig,
    ccfg: CsaPagedConfig,
    *,
    num_req,
    plan,  # dict of SMEM refs from `hybrid_plan` (flattened per request), + sparse_vmem
    csa_pt,
    nope_hbm,
    rope_hbm,
    win: core.VerifyWindow,
    cscr: CsaHybridScratch,
):
    """`csa_attend` streaming whole pages where at least `k_dense` union rows
    sit (rows copied out of VMEM) and gathering the rest one DMA per row."""
    G, PB, page = ccfg.granule, ccfg.blk_pages, ccfg.page
    T, U, P = wcfg.q_len, ccfg.u_max, ccfg.pages_per_req
    NB1 = -(-P // PB) + 1
    page_bits = page.bit_length() - 1
    dpages, n_dense, dent, doff = plan["dense_pages"], plan["n_dense"], plan["dense_ent"], plan["dense_off"]
    sent, n_sparse, sent_v = plan["sparse_ent"], plan["n_sparse"], plan["sparse_vmem"]

    def nbd(r):
        return core.cdiv(n_dense[r], PB)

    def phys_of(r, lp):
        lp = jnp.clip(lp, 0, P - 1)
        return jnp.clip(csa_pt[r * P + lp], 0, nope_hbm.shape[0] - 1)

    def start(r, b, slot):
        @pl.when(b < nbd(r))
        def _():
            last = jnp.maximum(n_dense[r] - 1, 0)
            for p in range(PB):
                phys = phys_of(r, dpages[r * P + jnp.minimum(b * PB + p, last)])
                pltpu.make_async_copy(nope_hbm.at[phys], cscr.nope.at[slot, pl.ds(p * page, page)],
                                      cscr.sems.at[slot, 0]).start()
                pltpu.make_async_copy(rope_hbm.at[phys],
                                      cscr.rope.at[slot, pl.ds(p * page // 4, page // 4)],
                                      cscr.sems.at[slot, 1]).start()

        @pl.when(b >= nbd(r))
        def _():
            j0 = (b - nbd(r)) * G
            last = jnp.maximum(n_sparse[r] - 1, 0)
            for k in range(G):
                idx = sent[r * U + jnp.minimum(j0 + k, last)] >> UNION_BITS
                phys = phys_of(r, idx >> page_bits)
                sl = idx & (page - 1)
                pltpu.make_async_copy(nope_hbm.at[phys, pl.ds(sl, 1)], cscr.rn.at[slot, pl.ds(k, 1)],
                                      cscr.sems.at[slot, 2]).start()
                pltpu.make_async_copy(rope_hbm.at[phys, pl.ds(sl >> 2, 1)],
                                      cscr.rr.at[slot, pl.ds(k, 1)], cscr.sems.at[slot, 3]).start()

    def wait(r, b, slot):
        dense = b < nbd(r)
        for i, buf in enumerate((cscr.nope, cscr.rope, cscr.rn, cscr.rr)):
            @pl.when(dense if i < 2 else jnp.logical_not(dense))
            def _(i=i, buf=buf):
                blk = buf.at[slot]
                pltpu.make_async_copy(blk, blk, cscr.sems.at[slot, i]).wait()

    def attend(r, nope_w, rope_w, e_col, valid):
        bits = jnp.where(valid, e_col & ((1 << T) - 1), 0)
        kv, kv_t = csa_keys(nope_w, rope_w, (e_col >> UNION_BITS) & 3, wcfg.transposed)
        if wcfg.transposed:
            t = lax.broadcasted_iota(jnp.int32, (1, wcfg.rows_pad), 1) // wcfg.heads
            vis = ((bits >> t) & 1) == 1
        else:
            t = lax.broadcasted_iota(jnp.int32, (wcfg.rows8, 1), 0) // wcfg.heads
            bits_row = jnp.transpose(jnp.broadcast_to(bits, (G, L.LANES)))[0:1, :]
            vis = ((bits_row >> t) & 1) == 1
        win.update(r, kv, kv_t, vis, first=False)

    def step(r, b, slot):
        @pl.when(b < nbd(r))
        def _():
            lo, hi = doff[r * NB1 + b], doff[r * NB1 + b + 1]

            def granule(g, c):
                base = lo + g * G

                def gather(k, c2):
                    e = dent[r * U + jnp.minimum(base + k, hi - 1)]
                    local = e >> UNION_BITS
                    cscr.stage_n[pl.ds(k, 1), :] = cscr.nope[slot, pl.ds(local, 1), :]
                    cscr.stage_r[pl.ds(k, 1), :] = cscr.rope[slot, pl.ds(local >> 2, 1), :]
                    cscr.stage_e[pl.ds(k, 1), :] = jnp.full((1, L.LANES), e, jnp.int32)
                    return c2

                lax.fori_loop(0, G, gather, 0, unroll=8)
                valid = base + lax.broadcasted_iota(jnp.int32, (G, 1), 0) < hi
                attend(r, cscr.stage_n[...], cscr.stage_r[...], cscr.stage_e[:, 0:1], valid)
                return c

            lax.fori_loop(0, core.cdiv(hi - lo, G), granule, 0)

        @pl.when(b >= nbd(r))
        def _():
            j = b - nbd(r)
            rows = G // L.LANES
            ent = jnp.concatenate([sent_v[r, pl.ds(j * rows + i, 1), :] for i in range(rows)], 1)
            valid = j * G + lax.broadcasted_iota(jnp.int32, (G, 1), 0) < n_sparse[r]
            attend(r, cscr.rn[slot], cscr.rr[slot], _col(ent), valid)

    core.run_window_stream(
        win, num_req=num_req,
        n_blocks=lambda r: jnp.maximum(1, nbd(r) + core.cdiv(n_sparse[r], G)),
        start=start, wait=wait, step=step,
    )


_PLAN_KEYS = ("dense_pages", "n_dense", "dense_ent", "dense_off", "sparse_ent", "n_sparse")


def _csa_hybrid_kernel(wcfg, ccfg, nreq_ref, seq_ref, swa_pt_ref, csa_pt_ref, *refs):
    nplan = len(_PLAN_KEYS)
    plan = dict(zip(_PLAN_KEYS, refs[:nplan]))
    q_ref, nkv_ref, sink_ref, sparse_vmem, swa_hbm, nope_hbm, rope_hbm, out_ref = \
        refs[nplan:nplan + 8]
    scratch = refs[nplan + 8:]
    plan["sparse_vmem"] = sparse_vmem
    nw = len(core.WindowScratch._fields)
    win = core.VerifyWindow(
        wcfg, seq_lens=seq_ref, swa_pt=swa_pt_ref, q_ref=q_ref, new_kv_ref=nkv_ref,
        sink_ref=sink_ref, swa_hbm=swa_hbm, out_ref=out_ref,
        scr=core.WindowScratch(*scratch[:nw]),
    )
    csa_attend_hybrid(wcfg, ccfg, num_req=nreq_ref[0], plan=plan, csa_pt=csa_pt_ref,
                      nope_hbm=nope_hbm, rope_hbm=rope_hbm, win=win,
                      cscr=CsaHybridScratch(*scratch[nw:]))


def _csa_kernel(wcfg, ccfg, nreq_ref, seq_ref, swa_pt_ref, ulen_ref, csa_pt_ref, union_smem,
                q_ref, nkv_ref, sink_ref, union_vmem, swa_hbm, nope_hbm, rope_hbm, out_ref,
                *scratch):
    nw = len(core.WindowScratch._fields)
    win = core.VerifyWindow(
        wcfg, seq_lens=seq_ref, swa_pt=swa_pt_ref, q_ref=q_ref, new_kv_ref=nkv_ref,
        sink_ref=sink_ref, swa_hbm=swa_hbm, out_ref=out_ref,
        scr=core.WindowScratch(*scratch[:nw]),
    )
    csa_attend(wcfg, ccfg, num_req=nreq_ref[0], union_smem=union_smem, union_vmem=union_vmem,
               union_len=ulen_ref, csa_pt=csa_pt_ref, nope_hbm=nope_hbm, rope_hbm=rope_hbm,
               win=win, cscr=CsaScratch(*scratch[nw:]))


def csa_attention(
    q,  # bf16 [R, q_len, H, 512]
    new_kv,  # bf16 [R, q_len, 512]
    swa_words,  # int32 [N, 2 * swa_page, 128]
    swa_table,  # int32 [R, P_swa]
    nope_words,  # int32 [N, page, 128]
    rope_words,  # int32 [N, page // 4, 128]
    csa_table,  # int32 [R, P]
    union,  # int32 [R, u_max] from `topk_union_placeholder` (or the in-kernel top-k)
    union_len,  # int32 [R]
    seq_lens,  # int32 [R]
    sinks,  # f32 [H]
    *,
    sm_scale: float,
    gather: str = "paged",  # "paged": stream pages, copy rows in VMEM; "rows": DMA per
    # row; "hybrid": stream pages with >= k_dense union rows, DMA the other rows
    blk_pages: int = 8,
    granule: int = 256,
    k_dense: int = 4,  # hybrid: union rows a page needs to be streamed whole
    pv_dtype=jnp.float32,
    interpret=False,
    vmem_limit_bytes: int = 60 * 1024 * 1024,
):
    """Standalone wrapper: SWA + sparse attention, bf16 ``[R, q_len, H, 512]``."""
    R, T, H, D = q.shape
    wcfg = core.WindowConfig(heads=H, sm_scale=sm_scale, q_len=T,
                             swa_page=swa_words.shape[1] // 2, pv_dtype=pv_dtype,
                             round_window_acc=True)
    G = granule
    u_max = core.cdiv(union.shape[1], G) * G
    union = jnp.pad(union, ((0, 0), (0, u_max - union.shape[1])))
    ccfg = CsaConfig(max_req=R, pages_per_req=csa_table.shape[1], u_max=u_max,
                     page=nope_words.shape[1], granule=128)
    swa_pt = core.swa_window_pages(seq_lens, swa_table, q_len=T, window=wcfg.window,
                                   swa_page=wcfg.swa_page)
    nkv = jnp.pad(new_kv, ((0, 0), (0, wcfg.new_rows - T), (0, 0)))
    pad = wcfg.rows_pad - wcfg.rows
    q_rows = jnp.pad(q.reshape(R, wcfg.rows, D), ((0, 0), (0, pad), (0, 0)))
    sink_row = jnp.pad(jnp.tile(sinks.astype(jnp.float32), T), (0, pad)).reshape(1, -1)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    common = (jnp.array([R], jnp.int32), seq_lens.astype(jnp.int32), swa_pt.reshape(-1),
              union_len.astype(jnp.int32), csa_table.astype(jnp.int32).reshape(-1),
              union.reshape(-1), q_rows, nkv, sink_row)
    if gather == "paged":
        pcfg = CsaPagedConfig(max_req=R, pages_per_req=csa_table.shape[1], u_max=u_max,
                              page=nope_words.shape[1], blk_pages=blk_pages, granule=G)
        out = pl.pallas_call(
            functools.partial(_csa_paged_kernel, wcfg, pcfg),
            out_shape=jax.ShapeDtypeStruct((R, wcfg.rows_pad, D), q.dtype),
            in_specs=[smem] * 6 + [vmem] * 3 + [hbm] * 3,
            out_specs=vmem,
            scratch_shapes=[*core.window_scratch_shapes(wcfg), *csa_paged_scratch_shapes(pcfg)],
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes,
                                                 disable_bounds_checks=True),
            interpret=pltpu.InterpretParams() if interpret else False,
            name=f"mk_csa_paged_h{H}_r{R}",
        )(*common, swa_words, nope_words, rope_words)
        return out[:, : wcfg.rows].reshape(R, T, H, D)
    if gather == "hybrid":
        pcfg = CsaPagedConfig(max_req=R, pages_per_req=csa_table.shape[1], u_max=u_max,
                              page=nope_words.shape[1], blk_pages=blk_pages, granule=G)
        plan = hybrid_plan(union, union_len, pages_per_req=csa_table.shape[1],
                           page=pcfg.page, blk_pages=blk_pages, k_dense=k_dense)
        out = pl.pallas_call(
            functools.partial(_csa_hybrid_kernel, wcfg, pcfg),
            out_shape=jax.ShapeDtypeStruct((R, wcfg.rows_pad, D), q.dtype),
            in_specs=[smem] * (4 + len(_PLAN_KEYS)) + [vmem] * 4 + [hbm] * 3,
            out_specs=vmem,
            scratch_shapes=[*core.window_scratch_shapes(wcfg), *csa_hybrid_scratch_shapes(pcfg)],
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes,
                                                 disable_bounds_checks=True),
            interpret=pltpu.InterpretParams() if interpret else False,
            name=f"mk_csa_hybrid_h{H}_r{R}",
        )(jnp.array([R], jnp.int32), seq_lens.astype(jnp.int32), swa_pt.reshape(-1),
          csa_table.astype(jnp.int32).reshape(-1),
          *[plan[k].reshape(-1) for k in _PLAN_KEYS],
          q_rows, nkv, sink_row, plan["sparse_ent"].reshape(R, u_max // 128, 128),
          swa_words, nope_words, rope_words)
        return out[:, : wcfg.rows].reshape(R, T, H, D)
    assert gather == "rows", gather
    out = pl.pallas_call(
        functools.partial(_csa_kernel, wcfg, ccfg),
        out_shape=jax.ShapeDtypeStruct((R, wcfg.rows_pad, D), q.dtype),
        in_specs=[smem] * 6 + [vmem] * 4 + [hbm] * 3,
        out_specs=vmem,
        scratch_shapes=[*core.window_scratch_shapes(wcfg), *csa_scratch_shapes(ccfg)],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes,
                                             disable_bounds_checks=True),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"mk_csa_h{H}_r{R}",
    )(
        jnp.array([R], jnp.int32),
        seq_lens.astype(jnp.int32),
        swa_pt.reshape(-1),
        union_len.astype(jnp.int32),
        csa_table.astype(jnp.int32).reshape(-1),
        union.reshape(-1),
        q_rows,
        nkv,
        sink_row,
        union.reshape(R, u_max // 128, 128),
        swa_words,
        nope_words,
        rope_words,
    )
    return out[:, : wcfg.rows].reshape(R, T, H, D)
