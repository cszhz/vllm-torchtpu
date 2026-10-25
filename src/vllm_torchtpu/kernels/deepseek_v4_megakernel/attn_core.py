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
"""Shared pieces of the megakernel verify-window attention.

A request's ``q_len`` query rows of ``heads`` heads sit in VMEM as one
``[rows_pad, 512]`` block (row ``t * heads + h``, zero rows up to a multiple of
128). With many rows (``rows_pad >= 256``) scores are computed transposed,
``s^T [keys, rows]`` (keys on sublanes, query rows on lanes), so the per-row
softmax statistics are lane-dense ``[1, rows]`` vectors and the accumulator is
``acc^T [512, rows]``. With few rows (8 heads) the plain ``s [rows, keys]``
orientation needs no transposes: both matmuls take the keys as MXU weights.

`VerifyWindow` runs the sliding-window part (cached tokens from the served SWA
pages plus the in-flight rows), owns the online-softmax state the
compressed-KV passes continue, and applies the attention sink at the end.
"""

import dataclasses
import functools
from typing import NamedTuple

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_layout as L

NEG = float(jnp.finfo(jnp.float32).min)
NT = (((1,), (1,)), ((), ()))
NN = (((1,), (0,)), ((), ()))
TN = (((0,), (0,)), ((), ()))


def cdiv(a, b):
    return (a + b - 1) // b


@dataclasses.dataclass(frozen=True)
class WindowConfig:
    heads: int  # query heads on this rank
    sm_scale: float
    q_len: int = 7
    window: int = 128
    swa_page: int = 128  # SWA tokens per page
    # probabilities in the row sum and P.V (the stock kernels keep them f32)
    pv_dtype: object = jnp.float32
    # a compressed-KV pass follows: round the window pass's accumulator to bf16
    # as stock does handing it from the SWA kernel to the HCA / CSA kernel
    round_window_acc: bool = False
    write_swa: bool = False  # also store the new rows into the SWA pages
    # the new rows are one block that sees itself bidirectionally (stock
    # non_causal_block, the DSpark draft); else causal
    block_bidir: bool = False
    # transposed scores even below 256 query rows (the attention phase's
    # HCA / CSA passes are written for that orientation)
    force_transposed: bool = False

    @property
    def rows(self) -> int:
        return self.q_len * self.heads

    @property
    def rows_pad(self) -> int:
        return cdiv(self.rows, 128) * 128

    @property
    def rows8(self) -> int:
        return cdiv(self.rows, 8) * 8

    @property
    def transposed(self) -> bool:
        return self.force_transposed or self.rows_pad >= 256

    @property
    def new_rows(self) -> int:
        return cdiv(self.q_len, 8) * 8


class WindowScratch(NamedTuple):
    swa_buf: object  # int32 [2, 2 * window, 128]
    m: object  # f32 [1, rows_pad] (transposed) or [rows8, 1]; scaled scores
    l: object  # f32, as m
    acc: object  # f32 [512, rows_pad] (transposed) or [rows8, 512]
    sems: object  # DMA [3]: SWA reads (2 slots), SWA row writes
    wbuf: object  # int32 [2 * new_rows, 128]: the new rows as cache words


def window_scratch_shapes(cfg: WindowConfig) -> WindowScratch:
    stat = (1, cfg.rows_pad) if cfg.transposed else (cfg.rows8, 1)
    acc = (L.HEAD_DIM, cfg.rows_pad) if cfg.transposed else (cfg.rows8, L.HEAD_DIM)
    return WindowScratch(
        swa_buf=pltpu.VMEM((2, 2 * cfg.window, L.LANES), jnp.int32),
        m=pltpu.VMEM(stat, jnp.float32),
        l=pltpu.VMEM(stat, jnp.float32),
        acc=pltpu.VMEM(acc, jnp.float32),
        sems=pltpu.SemaphoreType.DMA((3,)),
        wbuf=pltpu.VMEM((2 * cfg.new_rows, L.LANES), jnp.int32),
    )


SWA_PAGES = 3  # window pages per request in `swa_window_pages`


def swa_window_pages(seq_lens, swa_table, *, q_len, window, swa_page):
    """Physical SWA pages of each request's window: [R, 3].

    The cached tokens ``[s0, p0)`` (``s0 = max(p0 - window + 1, 0)``) and the
    new rows ``[p0, p0 + q_len)`` span logical pages ``s0 // swa_page`` and the
    two after it (``q_len <= swa_page - window + 1 + swa_page``).
    """
    p0 = seq_lens - q_len
    lp = jnp.maximum(p0 - (window - 1), 0) // swa_page
    cols = jnp.minimum(lp[:, None] + jnp.arange(SWA_PAGES)[None, :], swa_table.shape[1] - 1)
    return jnp.take_along_axis(swa_table, cols, axis=1).astype(jnp.int32)


def decode_bf16_tokens(buf, n_tok, transposed):
    """int32 ``[2 n, 128]`` ref of bf16-latent tokens -> bf16 ``[n, 512]`` and,
    if `transposed`, its transpose ``[512, n]`` (else None)."""
    even = buf[pl.ds(0, n_tok, stride=2), :]
    odd = buf[pl.ds(1, n_tok, stride=2), :]
    kv = L.bf16_tokens_from_words(even, odd)
    if not transposed:
        return kv, None
    return kv, L.bf16_tokens_from_words(even.T, odd.T, axis=0)


class VerifyWindow:
    """SWA pass and online-softmax state for the requests of one layer."""

    def __init__(self, cfg: WindowConfig, *, seq_lens, swa_pt, q_ref, new_kv_ref,
                 sink_ref, swa_hbm, out_ref, scr: WindowScratch):
        self.cfg = cfg
        self.seq_lens = seq_lens  # SMEM int32 [max_req], incl. the new rows
        self.swa_pt = swa_pt  # SMEM int32 [max_req * 3], `swa_window_pages`
        self.q_ref = q_ref  # VMEM [max_req, rows_pad, 512], bf16 values (bf16 or f32 storage)
        self.new_kv_ref = new_kv_ref  # VMEM bf16 [max_req, new_rows, 512]
        self.sink_ref = sink_ref  # VMEM f32 [1, rows_pad]: sink of each row's head
        self.swa_hbm = swa_hbm  # int32 [N, 2 * swa_page, 128]
        self.out_ref = out_ref  # VMEM [max_req, rows_pad, 512]: bf16-rounded output
        self.scr = scr

    def p0(self, r):
        return self.seq_lens[r] - self.cfg.q_len

    def _s0(self, r):
        return jnp.maximum(self.p0(r) - (self.cfg.window - 1), 0)

    def swa_dma(self, r, slot, *, wait):
        cfg = self.cfg
        s0 = self._s0(r)
        n = self.p0(r) - s0
        off = s0 % cfg.swa_page
        na = jnp.minimum(n, cfg.swa_page - off)
        nb = n - na
        buf, sem = self.scr.swa_buf.at[slot], self.scr.sems.at[slot]
        copies = (
            (self.swa_pt[SWA_PAGES * r], 2 * off, 0, na),
            (self.swa_pt[SWA_PAGES * r + 1], 0, 2 * na, nb),
        )
        for page, src, dst, cnt in copies:
            @pl.when(cnt > 0)
            def _(page=page, src=src, dst=dst, cnt=cnt):
                self.swa_copy(lambda ref: ref.at[page, pl.ds(src, 2 * cnt)],
                              buf.at[pl.ds(dst, 2 * cnt)], sem, wait=wait, write=False)

    def swa_copy(self, hbm, vmem, sem, *, wait, write):
        """One SWA-cache DMA; ``hbm(ref)`` picks the window of the cache ref.
        Overridden where the cache is a per-layer operand list."""
        src, dst = (vmem, hbm(self.swa_hbm)) if write else (hbm(self.swa_hbm), vmem)
        cp = pltpu.make_async_copy(src, dst, sem)
        cp.wait() if wait else cp.start()

    def _swa_write_copies(self, r, p0, n):
        """Positions [p0, p0 + n) (inside this request's window pages) as <= 2
        page-contiguous runs."""
        cfg = self.cfg
        lp0 = self._s0(r) // cfg.swa_page
        page_a = p0 // cfg.swa_page
        off = p0 % cfg.swa_page
        na = jnp.minimum(n, cfg.swa_page - off)
        nb = n - na
        pa = self.swa_pt[SWA_PAGES * r + jnp.clip(page_a - lp0, 0, SWA_PAGES - 1)]
        pb = self.swa_pt[SWA_PAGES * r + jnp.clip(page_a + 1 - lp0, 0, SWA_PAGES - 1)]
        return ((pa, 2 * off, 0, na), (pb, 0, 2 * na, nb))

    def swa_write(self, r, *, wait, rows=None, p0=None, n=None):
        """Stores request r's new rows into its SWA pages (bytes as
        `mla_swa.prepare_kv_inputs` lays them out). Issued after the window is
        read; waited before the request finishes. ``rows`` (bf16 [B, new_rows,
        512]), ``p0`` and ``n`` store other rows at [p0, p0 + n) instead."""
        rows = self.new_kv_ref if rows is None else rows
        p0 = self.p0(r) if p0 is None else p0
        n = self.cfg.q_len if n is None else n
        if not wait:
            x = rows[r].astype(jnp.float32)  # [new_rows, 512], exact
            bits = lax.bitcast_convert_type(x, jnp.int32)
            words = []
            for s_ in range(2):  # row s: dims 256 s + w (low half), 256 s + 128 + w (high)
                lo = bits[:, 256 * s_: 256 * s_ + 128]
                hi = bits[:, 256 * s_ + 128: 256 * s_ + 256]
                words.append(((lo >> 16) & 0xFFFF) | (hi & -65536))
            self.scr.wbuf[...] = jnp.stack(words, axis=1).reshape(
                2 * self.cfg.new_rows, L.LANES)
        for page, dst, src, cnt in self._swa_write_copies(r, p0, n):
            @pl.when(cnt > 0)
            def _(page=page, dst=dst, src=src, cnt=cnt):
                self.swa_copy(lambda ref: ref.at[page, pl.ds(dst, 2 * cnt)],
                              self.scr.wbuf.at[pl.ds(src, 2 * cnt)], self.scr.sems.at[2],
                              wait=wait, write=True)

    def key_index(self, n):
        """Key index ``0..n-1`` broadcastable against `qpos`."""
        if self.cfg.transposed:
            return lax.broadcasted_iota(jnp.int32, (n, 1), 0)
        return lax.broadcasted_iota(jnp.int32, (1, n), 1)

    def qpos(self, r):
        """Position of each query row: ``[1, rows_pad]`` or ``[rows8, 1]``."""
        cfg = self.cfg
        if cfg.transposed:
            row = lax.broadcasted_iota(jnp.int32, (1, cfg.rows_pad), 1)
        else:
            row = lax.broadcasted_iota(jnp.int32, (cfg.rows8, 1), 0)
        return self.p0(r) + row // cfg.heads

    def update(self, r, kv, kv_t, vis, *, first, lanes=None):
        """Online-softmax step over keys ``kv [n, 512]`` (``kv_t`` its transpose
        when transposed, or None to contract kv directly); ``vis`` is their
        visibility, shaped like the scores.
        `first` starts the state. Transposed only: ``lanes = (lane0, width)``
        restricts the step to query rows [lane0, lane0 + width)."""
        cfg, scr = self.cfg, self.scr
        axis = 0 if cfg.transposed else 1
        if lanes is not None:
            assert cfg.transposed and not first
            l0, w = lanes
            cols = pl.ds(pl.multiple_of(l0, 128), w)
            q = self.q_ref[r, cols, :].astype(jnp.bfloat16)
            s = lax.dot_general(kv, q, NT, preferred_element_type=jnp.float32)
            s = jnp.where(vis, s * cfg.sm_scale, NEG)
            m_prev = scr.m[:, cols]
            m_cur = jnp.maximum(m_prev, jnp.max(s, axis=0, keepdims=True))
            p = jnp.exp(s - m_cur).astype(cfg.pv_dtype)
            pv = lax.dot_general(kv_t.astype(cfg.pv_dtype), p, NN,
                                 preferred_element_type=jnp.float32)
            alpha = jnp.exp(m_prev - m_cur)
            scr.l[:, cols] = alpha * scr.l[:, cols] + jnp.sum(p.astype(jnp.float32), 0, keepdims=True)
            scr.acc[:, cols] = alpha * scr.acc[:, cols] + pv
            scr.m[:, cols] = m_cur
            return
        if cfg.transposed:
            s = lax.dot_general(kv, self.q_ref[r].astype(jnp.bfloat16), NT,
                                preferred_element_type=jnp.float32)
        else:
            q = self.q_ref[r, : cfg.rows8, :].astype(jnp.bfloat16)
            s = lax.dot_general(q, kv, NT, preferred_element_type=jnp.float32)
        # The stock kernels' order: scale, mask, then exp(s - m) on scaled scores.
        s = jnp.where(vis, s * cfg.sm_scale, NEG)
        m_blk = jnp.max(s, axis=axis, keepdims=True)
        m_prev = None if first else scr.m[...]
        m_cur = m_blk if first else jnp.maximum(m_prev, m_blk)
        p = jnp.exp(s - m_cur).astype(cfg.pv_dtype)
        l_blk = jnp.sum(p.astype(jnp.float32), axis=axis, keepdims=True)
        if cfg.transposed and kv_t is None:  # contract the keys of kv itself: no transpose
            pv = lax.dot_general(kv.astype(cfg.pv_dtype), p, TN,
                                 preferred_element_type=jnp.float32)  # [512, rows_pad]
        elif cfg.transposed:
            pv = lax.dot_general(kv_t.astype(cfg.pv_dtype), p, NN,
                                 preferred_element_type=jnp.float32)  # [512, rows_pad]
        else:
            pv = lax.dot_general(p, kv.astype(cfg.pv_dtype), NN,
                                 preferred_element_type=jnp.float32)  # [rows8, 512]
        if first:  # the window pass (`swa_step`)
            scr.l[...] = l_blk
            if cfg.round_window_acc:
                pv = pv.astype(jnp.bfloat16).astype(jnp.float32)
            scr.acc[...] = pv
        else:
            alpha = jnp.exp(m_prev - m_cur)
            scr.l[...] = alpha * scr.l[...] + l_blk
            scr.acc[...] = alpha * scr.acc[...] + pv
        scr.m[...] = m_cur

    def swa_step(self, r, slot):
        """Starts the request's softmax state with its sliding window."""
        cfg = self.cfg
        W, T = cfg.window, cfg.q_len
        p0, s0 = self.p0(r), self._s0(r)
        n = p0 - s0
        # Keys [0, W): cached tokens s0 + j (j < n); [W, W + new_rows): new rows.
        nk = W + cfg.new_rows
        kc, kc_t = decode_bf16_tokens(self.scr.swa_buf.at[slot], W, cfg.transposed)
        new = self.new_kv_ref[r].astype(jnp.bfloat16)
        jr = lax.broadcasted_iota(jnp.int32, (nk, 1), 0)
        live = (jr < n) | ((jr >= W) & (jr < W + T))
        kv = jnp.where(live, jnp.concatenate([kc, new], axis=0), 0)
        kv_t = None
        if cfg.transposed:
            kv_t = jnp.where(live.T, jnp.concatenate([kc_t, new.T], axis=1), 0)
        j, qpos = self.key_index(nk), self.qpos(r)
        cached = (j < n) & (qpos - (s0 + j) < W)
        fresh = (j >= W) & ((j < W + T) if cfg.block_bidir else (p0 + j - W <= qpos))
        self.update(r, kv, kv_t, cached | fresh, first=True)
        if cfg.write_swa:
            self.swa_write(r, wait=False)

    def finalize(self, r):
        cfg, scr = self.cfg, self.scr
        if cfg.write_swa:
            self.swa_write(r, wait=True)
        sink = self.sink_ref[...]
        if not cfg.transposed:
            sink = sink.T[: cfg.rows8]
        l_total = scr.l[...] + jnp.exp(sink - scr.m[...])
        out = scr.acc[...] * pltpu.reciprocal(l_total, approx=True)
        if cfg.transposed:
            self.out_ref[r] = out.T.astype(jnp.bfloat16).astype(self.out_ref.dtype)
        else:
            self.out_ref[r, : cfg.rows8, :] = out.astype(jnp.bfloat16).astype(self.out_ref.dtype)


def run_window_stream(win: VerifyWindow, *, num_req, n_blocks, start, wait, step):
    """Per request: the sliding window (its DMAs one request ahead), then blocks
    ``b < n_blocks(r)`` (>= 1) of compressed keys, then the sink.

    Blocks flow through a 2-slot ring: ``start(r, b, slot)`` issues block b of
    request r, ``wait(r, b, slot)`` waits for it, ``step(r, b, slot)`` consumes
    it. The prefetch runs ahead across requests; past the last block of
    the last request it re-issues that block, drained at the end, so the issue
    code stays branch-free and interleaves with the compute around it.
    """

    def req_body(r, g):
        swa_slot = r % 2
        win.swa_dma(r, swa_slot, wait=True)

        @pl.when(r + 1 < num_req)
        def _():
            win.swa_dma(r + 1, 1 - swa_slot, wait=False)

        win.swa_step(r, swa_slot)
        nb = n_blocks(r)

        def blk_body(b, g):
            slot = g % 2
            wait(r, b, slot)
            last = b + 1 == nb
            more = r + 1 < num_req
            nr = jnp.where(last & more, r + 1, r)
            nxt = jnp.where(last, jnp.where(more, 0, b), b + 1)
            start(nr, nxt, 1 - slot)
            step(r, b, slot)
            return g + 1

        g = lax.fori_loop(0, nb, blk_body, g)
        win.finalize(r)
        return g

    @pl.when(num_req > 0)
    def _():
        win.swa_dma(0, 0, wait=False)
        start(0, 0, 0)
        g = lax.fori_loop(0, num_req, req_body, jnp.int32(0))
        last = num_req - 1
        wait(last, n_blocks(last) - 1, g % 2)
