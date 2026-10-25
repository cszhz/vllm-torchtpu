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
"""DSv4 dense chain of every decoder layer in one grid-less pallas_call.

All dense weights stay in HBM and stream through ONE VMEM tile ring in the
order the chain consumes them, across ops and across layers: tile g lives in
bank g % n_banks; consuming it refills the bank with tile g + n_banks, so the
next op's (and the next layer's) weights are in flight while this op computes.
The (stream, layer, tile) of every g comes from an SMEM table.

Attention core and routed experts are stubbed (``dense_reference`` documents
the stubs); everything else reproduces the stock serving formulas.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import types

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4_megakernel import vmem_alias
from vllm_torchtpu.kernels.deepseek_v4_megakernel.dense_config import (
    CSA,
    HCA,
    DenseChainConfig,
    storage_bytes,
)

F32, BF16, FP8 = jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn
FP8_MAX = 448.0
LANE = 128
CW = 1024  # output columns per MXU dot (an f32 [16, 1024] result is 16 vregs)
EW = 1024  # lanes per elementwise chunk
UNROLL = 4  # streams with at most this many tiles are unrolled
TILE_UNROLL = 2  # tiles per loop iteration of longer streams
POST0, COMB0 = 0, 128  # lanes of the pending post / comb mixes in `mixst`


@dataclasses.dataclass(frozen=True)
class Faults:
    """Deliberate deviations: positive controls of the numerics check and
    time attribution."""

    early_refill: bool = False  # refill a bank before its tile is read
    swap_tiles: tuple[int, int] | None = None  # swap two schedule entries
    no_dma: bool = False  # compute on stale banks: the compute-bound time
    # parts left out, for time attribution only: "sinkhorn", "mhc_pass",
    # "quant", "epilogue", "norm", "dot", "hc_dot", "comb", "stub", "swiglu"; "wo_a"
    # (whole op, only with no_dma: its tiles are not consumed)
    skip: frozenset[str] = frozenset()


def _dot(a, b):
    return lax.dot_general(a, b, (((1,), (0,)), ((), ())), preferred_element_type=F32)


def _dot_nt(a, b):
    return lax.dot_general(a, b, (((1,), (1,)), ((), ())), preferred_element_type=F32)


def _mo(x, m=LANE):
    """pl.multiple_of for traced offsets; static offsets pass through."""
    return x if isinstance(x, int) else pl.multiple_of(x, m)


def _div(x, d: int):
    """x // d for x >= 0 (lax.div truncates; floor division would lower a
    sign op)."""
    return x // d if isinstance(x, int) else lax.div(x, jnp.int32(d))


def _rem(x, d: int):
    return x % d if isinstance(x, int) else lax.rem(x, jnp.int32(d))


def _chunks(n: int, w: int):
    return [(c, min(w, n - c)) for c in range(0, n, w)]


class _Ring:
    """The global dense tile ring (Qwen3.5 megakernel ``DenseWeightRing``
    scheme, with a table-driven decode of g)."""

    def __init__(self, cfg, sched, tiles, pool, sem, typed, faults, fence=None):
        self.cfg, self.sched, self.tiles = cfg, sched, tiles
        self.pool, self.sem, self.typed, self.faults = pool, sem, typed, faults
        # fence: SMEM int32 [1] or None. Tiles >= fence[0] are not fetched,
        # so the banks can be lent out once the tiles before it are consumed.
        self.fence = fence
        self.names = cfg.stream_names
        self.nb = cfg.n_banks
        self.brows = cfg.max_tile_bytes // 128

    def view(self, name, bank):
        if self.typed:
            return self.pool[name].at[bank]
        s = self.cfg.streams[name]
        dt = jnp.dtype(s.dtype)
        rows = storage_bytes(s.tile_rows, s.tile_cols, s.dtype) // 128
        region = self.pool.at[pl.ds(bank * self.brows, rows)]
        return vmem_alias.view(region, (s.tile_rows, s.tile_cols), dt)

    def _copy(self, name, src, bank):
        return pltpu.make_async_copy(src, self.view(name, bank), self.sem.at[bank])

    def fetch(self, g):
        if self.faults.no_dma:
            return

        ok = g < self.cfg.n_tiles_total
        if self.fence is not None:
            ok = jnp.logical_and(ok, g < self.fence[0])

        @pl.when(ok)
        def _():
            e = self.sched[g]
            sid, ordv, t = e & 15, (e >> 4) & 255, e >> 12
            bank = lax.rem(g, self.nb)
            for i, name in enumerate(self.names):
                pl.when(sid == i)(
                    functools.partial(
                        lambda n: self._copy(
                            n, self.tiles[n].at[ordv, t], bank
                        ).start(),
                        name,
                    )
                )

    def take(self, name, g):
        bank = lax.rem(g, self.nb)
        if not self.faults.no_dma:
            self._copy(name, self.tiles[name].at[0, 0], bank).wait()
        if self.faults.early_refill:
            self.fetch(g + self.nb)
        return self.view(name, bank)

    def release(self, g):
        """Called once tile g has been read."""
        if not self.faults.early_refill:
            self.fetch(g + self.nb)

    def release_range(self, g0, n: int):
        """release(g0 .. g0 + n - 1) from one fetch site: every site carries a
        DMA descriptor per stream, and kernel code size is a cost of its own."""
        if n == 1:
            self.release(g0)
        elif not self.faults.early_refill:
            lax.fori_loop(0, n, lambda i, c: (self.fetch(g0 + i + self.nb), c)[1], 0)

    def restart(self, g0, fence):
        """Move the fence and fetch tiles g0 .. g0 + n_banks - 1 (all banks are
        idle: every tile before g0 was consumed and none after was fetched)."""
        self.fence[0] = fence
        lax.fori_loop(0, self.nb, lambda i, c: (self.fetch(g0 + i), c)[1], 0)


def make_dense_ops(
    cfg: DenseChainConfig,
    ring,
    faults: Faults,
    *,
    sm_v,
    res,
    mixst,
    tsc,
    xin,
    hin,
    xa,
    xq,
    acc,
    q,
    xg,
    z,
    cos_sin,
    head_src=None,
):
    """The dense-chain building blocks on the given scratch refs (shared by the
    B2 chain kernel and the whole-step kernel). Returns a namespace of them.

    head_src(gi, hh, lo, n): bf16 [rows, n] lanes [lo, lo + n) of attention
    head hh of o-group gi (default: the `q` buffer, heads on lanes)."""
    M, h, hcm = cfg.rows, cfg.hidden, cfg.hc_mult
    lay = cfg.smalls_layout
    eps = cfg.rms_eps
    # a group of tiles is taken before any is released: it must fit the ring
    assert cfg.n_banks >= max(UNROLL, TILE_UNROLL, cfg.streams["hc_attn"].n_tiles)
    if head_src is None:
        d_ = cfg.head_dim
        kg_ = cfg.heads_per_group * d_

        def head_src(gi, hh, lo, n):
            return q[:, pl.ds(_mo(gi * kg_ + hh * d_ + lo), n)]

    # ---- small vectors of the current layer -------------------------------
    def vec(slot, name, c0=0, w=None):
        """[1, w] f32 slice of small vector ``name`` from lane c0 (128-aligned,
        static or traced)."""
        row, n = lay[name]
        w = n if w is None else w
        if w < LANE:
            assert isinstance(c0, int) and c0 == 0
            return sm_v[slot, pl.ds(row, 1), :][:, :w]
        nr = w // LANE
        assert w % LANE == 0
        r = row + _div(c0, LANE)
        blk = sm_v[slot, pl.ds(r, nr), :]
        return jnp.concatenate([blk[i : i + 1, :] for i in range(nr)], axis=1)

    def mul_vec(x, slot, name, c0, dtype=F32):
        """x [M, w] times small vector ``name``[c0:c0+w] (cast to dtype), one
        128-lane block at a time: each block's factor is one sublane row, so
        nothing is relaid out from sublanes to lanes."""
        row, _ = lay[name]
        w = x.shape[1]
        assert w % LANE == 0
        blk = sm_v[slot, pl.ds(row + _div(c0, LANE), w // LANE), :].astype(dtype)
        return jnp.concatenate(
            [x[:, i * LANE : (i + 1) * LANE] * blk[i : i + 1, :]
             for i in range(w // LANE)],
            axis=1,
        )  # fmt: skip

    # ---- elementwise helpers ----------------------------------------------
    skip = faults.skip

    def rms_norm_to(src, src_c0, n, dst, slot, wname, ss=None, *, weight_dtype):
        """dst[:, :n] = rms_norm(src[:, c0:c0+n]), f32 statistics.

        weight_dtype: F32 multiplies the weight in f32 and rounds once, as
        the stock mHC kernels' fused attn_norm / ffn_norm do
        (``mhc.utils.finish_layer_input``); BF16 rounds x * rsqrt to bf16
        first and multiplies the weight in bf16 (vLLM ``rms_norm`` as
        written in torch). ss: the row sums of squares if the producer
        already has them (in this function's chunk order). Returns max |dst|
        per row (f32), which the next fp8 activation quantization needs.
        """
        if "norm" in skip:
            return jnp.ones((M, 1), F32)
        if ss is None:
            ss = jnp.zeros((M, 1), F32)
            for c, w in _chunks(n, EW):
                v = src[:, src_c0 + c : src_c0 + c + w].astype(F32)
                ss = ss + jnp.sum(v * v, axis=1, keepdims=True)
        r = lax.rsqrt(ss / n + eps)
        amax = jnp.zeros((M, 1), F32)
        for c, w in _chunks(n, EW):
            v = src[:, src_c0 + c : src_c0 + c + w].astype(F32) * r
            out = mul_vec(v.astype(weight_dtype), slot, wname, c, weight_dtype)
            out = out.astype(BF16)
            dst[:, c : c + w] = out.astype(dst.dtype)
            amax = jnp.maximum(
                amax, jnp.max(jnp.abs(out), axis=1, keepdims=True).astype(F32)
            )
        return amax

    def tile_loop(name, g0, lhs, on_dot, bg=None):
        """For every tile of stream ``name``: on_dot(kb, nb, c0, dot).

        bg = (state, step, n): n applications of ``step`` to ``state`` are
        spread over the tiles, in the same basic blocks as the dots, so a
        latency-bound chain (the Sinkhorn) co-issues with MXU work. Returns
        the final state.
        """
        s = cfg.streams[name]
        nk, nt, tr, tc = s.n_k, s.n_tiles, s.tile_rows, s.tile_cols
        state, step, n_bg = bg if bg is not None else (None, None, 0)
        per = -(-n_bg // nt)

        def bg_steps(t, st):
            for i in range(per):
                if isinstance(t, int):
                    if t * per + i < n_bg:
                        st = step(st)
                else:
                    take_step = t * per + i < n_bg
                    st = jax.tree.map(
                        lambda a, b, p=take_step: jnp.where(p, a, b), step(st), st
                    )
            return st

        def group(t0, u, st):
            """Tiles t0 .. t0+u-1 in one basic block: all waits first, then
            the dots back to back, so one tile's MXU drain overlaps the next
            tile's pushes (a loop boundary per tile would serialize them)."""
            ts = [t0 + i for i in range(u)]
            ws = [ring.take(name, g0 + t) for t in ts]
            for t, w in zip(ts, ws):
                kb = _rem(t, nk) if nk > 1 else 0
                nb = _div(t, nk) if s.n_n > 1 else 0
                xs = lhs(_mo(kb * tr), tr)
                for c0, cw in _chunks(tc, CW):
                    if "dot" not in skip:
                        on_dot(kb, _mo(nb * tc + c0), cw, _dot(xs, w[:, c0 : c0 + cw]))
                st = bg_steps(t, st)
            ring.release_range(g0 + t0, u)
            return st

        if nt <= UNROLL:
            state = group(0, nt, state)
        else:
            n_pairs = nt // TILE_UNROLL
            state = lax.fori_loop(
                0,
                n_pairs,
                lambda i, st: group(i * TILE_UNROLL, TILE_UNROLL, st),
                state,
            )
            if nt % TILE_UNROLL:
                state = group(n_pairs * TILE_UNROLL, nt % TILE_UNROLL, state)
        return state

    def acc_into(kb, col, cw, d, nk):
        if nk == 1:
            acc[:, pl.ds(col, cw)] = d
        elif isinstance(kb, int):
            if kb == 0:
                acc[:, pl.ds(col, cw)] = d
            else:
                acc[:, pl.ds(col, cw)] += d
        else:

            @pl.when(kb == 0)
            def _():
                acc[:, pl.ds(col, cw)] = d

            @pl.when(kb > 0)
            def _():
                acc[:, pl.ds(col, cw)] += d

    def matmul(name, g0, lhs, bg=None):
        nk = cfg.streams[name].n_k
        return tile_loop(
            name, g0, lhs, lambda kb, col, cw, d: acc_into(kb, col, cw, d, nk), bg
        )

    def quantize_rows(src, k, amax=None):
        """Stock ``xla_quantized_matmul`` activation quantization of src[:, :k]
        into xq; returns the per-row f32 scale [M, 1]. amax: max |src| per
        row if already known."""
        if "quant" in skip:
            return jnp.ones((M, 1), F32)
        if amax is None:
            amax = jnp.zeros((M, 1), F32)
            for c, w in _chunks(k, EW):
                amax = jnp.maximum(
                    amax,
                    jnp.max(jnp.abs(src[:, c : c + w]), axis=1, keepdims=True).astype(
                        F32
                    ),
                )
        scale = amax / FP8_MAX
        scale = jnp.where(scale == 0, 1.0, scale)
        for c, w in _chunks(k, EW):
            v = src[:, c : c + w].astype(F32) / scale
            xq[:, c : c + w] = jnp.clip(v, -FP8_MAX, FP8_MAX).astype(FP8)
        return scale

    def fp8_linear(name, g0, src, slot, emit, bg=None, amax=None):
        """emit(c0, cw, out_bf16) over the [M, n] output of the stock fp8
        linear (per-row dynamic activation quant, per-channel weight scale).
        Returns the background state (see tile_loop)."""
        s = cfg.streams[name]
        xs = quantize_rows(src, s.k, amax)
        bg_out = matmul(name, g0, lambda k0, w: xq[:, pl.ds(k0, w)], bg)
        sname = name + "_scale"

        def chunk(c0, cw):
            if "epilogue" in skip:
                return
            out = acc[:, pl.ds(c0, cw)] * xs
            out = mul_vec(out, slot, sname, c0)
            emit(c0, cw, out.astype(BF16))

        n_ch = s.n // CW
        if s.n % CW or n_ch <= 8:
            for c0, cw in _chunks(s.n, CW):
                chunk(c0, cw)
        else:

            def body(i, c):
                chunk(pl.multiple_of(i * CW, CW), CW)
                return c

            lax.fori_loop(0, n_ch, body, 0)
        return bg_out

    # ---- mHC fused post + pre ---------------------------------------------
    def mhc(stream, g0, slot, base_name, scale_name):
        """residual <- post(x) + comb(residual); hin <- pre(residual); new
        post/comb from the Sinkhorn gates (fused_post_pre_kernel formulas)."""
        post = [mixst[:, POST0 + j : POST0 + j + 1] for j in range(hcm)]
        comb = [mixst[:, COMB0 + k : COMB0 + k + 1] for k in range(hcm * hcm)]

        ew = math.gcd(h, EW)

        s = cfg.streams[stream]
        assert s.tile_cols == h  # one hc tile per residual stream
        with jax.named_scope("mk_mhc_wait"):
            ws = [ring.take(stream, g0 + t) for t in range(s.n_tiles)]

        def post_body(ci, carry):
            """One lane chunk of every stream: recombine, store, and the
            chunk's share of the mix GEMM (MXU work beside the VPU work)."""
            ss, mix = carry
            c = pl.multiple_of(ci * ew, ew)
            xv = xin[:, pl.ds(c, ew)].astype(F32)
            old = [res[:, pl.ds(i * h + c, ew)].astype(F32) for i in range(hcm)]
            for j in range(hcm):
                a = post[j] * xv
                if "comb" not in skip:
                    for i in range(hcm):
                        a = a + comb[i * hcm + j] * old[i]
                nb = a.astype(BF16)
                res[:, pl.ds(j * h + c, ew)] = nb
                nf = nb.astype(F32)
                ss = ss + jnp.sum(nf * nf, axis=1, keepdims=True)
                if "hc_dot" not in skip:
                    mix = mix + _dot_nt(nb, ws[j][:, pl.ds(c, ew)])
            return ss, mix

        carry = (jnp.zeros((M, 1), F32), jnp.zeros((M, s.tile_rows), F32))
        if "mhc_pass" in skip:
            sqr, mix = jnp.ones((M, 1), F32), carry[1]
        else:
            with jax.named_scope("mk_mhc_post"):
                sqr, mix = lax.fori_loop(0, h // ew, post_body, carry)
        ring.release_range(g0, s.n_tiles)
        nm = cfg.n_mix
        mixes = mix[:, :nm]
        for c in range(1, cfg.hc_chunks):
            mixes = mixes + mix[:, c * nm : (c + 1) * nm]

        sc = vec(slot, scale_name)  # [1, 3]
        bs = vec(slot, base_name)  # [1, 24]
        pre, post_n, comb_logits = mhc_gates(mixes, sqr, sc, bs, cfg)
        # Tokens onto lanes: one [1, 128] row per (i, j) entry.
        tsc[...] = jnp.zeros(tsc.shape, F32)
        tsc[0:M, 0 : hcm * hcm] = comb_logits
        tt = tsc[...].T
        sk = sinkhorn_init(
            [tt[k : k + 1, :] for k in range(hcm * hcm)], hcm, cfg.hc_eps
        )
        for j in range(hcm):
            mixst[:, POST0 + j : POST0 + j + 1] = post_n[:, j : j + 1]

        def pre_body(ci, ss):
            """Layer input chunk, plus its share of the next RMSNorm's sum of
            squares (same chunks as rms_norm_to)."""
            c = pl.multiple_of(ci * ew, ew)
            a = pre[:, 0:1] * res[:, pl.ds(c, ew)].astype(F32)
            for i in range(1, hcm):
                a = a + pre[:, i : i + 1] * res[:, pl.ds(i * h + c, ew)].astype(F32)
            hb = a.astype(BF16)
            hin[:, pl.ds(c, ew)] = hb
            hf = hb.astype(F32)
            return ss + jnp.sum(hf * hf, axis=1, keepdims=True)

        assert ew == min(EW, h)  # pre_body chunks == rms_norm_to chunks
        ss_hin = None
        if "mhc_pass" not in skip:
            with jax.named_scope("mk_mhc_pre"):
                ss_hin = lax.fori_loop(0, h // ew, pre_body, jnp.zeros((M, 1), F32))
        iters = 1 if "sinkhorn" in skip else cfg.sinkhorn_iters
        step = functools.partial(sinkhorn_step, eps=cfg.hc_eps)
        return (sk, step, iters - 1), ss_hin

    def finish_sinkhorn(bg):
        """Store the comb mix (tokens back onto sublanes)."""
        tsc[...] = jnp.zeros(tsc.shape, F32)
        for k, row in enumerate(bg):
            tsc[k : k + 1, :] = row
        mixst[:, COMB0 : COMB0 + LANE] = tsc[...].T[0:M, :]

    # ---- attention-side ops -------------------------------------------------
    def stub_attention():
        if "stub" in skip:
            return
        d = cfg.head_dim

        def body(hh, c):
            c0 = pl.multiple_of(hh * d, LANE)
            v = q[:, pl.ds(c0, d)].astype(F32)
            r = lax.rsqrt(jnp.mean(v * v, axis=1, keepdims=True) + eps)
            q[:, pl.ds(c0, d)] = (v * r).astype(BF16)
            return c

        lax.fori_loop(0, cfg.heads, body, 0)

    def wo_a(g0, slot):
        """o_projection._kernel per group: inverse rope, bf16 fp8 quant, dot."""
        if "wo_a" in skip:
            assert faults.no_dma
            return
        s = cfg.streams["wo_a"]
        d, hpg = cfg.head_dim, cfg.heads_per_group
        kg = hpg * d
        per_g = s.n_tiles // cfg.groups
        cos, sin = cos_sin[:, :LANE], cos_sin[:, LANE:]
        lane = lax.broadcasted_iota(jnp.int32, (M, LANE), 1)
        even = lane % 2 == 0

        def group(gi, c):
            for hh in range(hpg):
                if d > LANE:
                    xg[:, hh * d : hh * d + d - LANE] = head_src(gi, hh, 0, d - LANE)
                tail = head_src(gi, hh, d - LANE, LANE).astype(F32)
                # rope.py _rotate_gptj: lane i reads lane i ^ 1, negated if even
                rot = jnp.where(
                    even, -pltpu.roll(tail, LANE - 1, 1), pltpu.roll(tail, 1, 1)
                )
                xg[:, hh * d + d - LANE : hh * d + d] = (tail * cos + rot * sin).astype(
                    BF16
                )
            amax = jnp.zeros((M, 1), BF16)
            for cc, w in _chunks(kg, EW):
                amax = jnp.maximum(
                    amax, jnp.max(jnp.abs(xg[:, cc : cc + w]), axis=1, keepdims=True)
                )
            inv = (FP8_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(BF16)
            for cc, w in _chunks(kg, EW):
                xq[:, cc : cc + w] = (xg[:, cc : cc + w] * inv).astype(FP8)
            inv_f = 1.0 / inv.astype(F32)
            nk = s.n_k
            for tt in range(per_g):
                t = gi * per_g + tt
                kb, nbl = tt % nk, tt // nk
                w = ring.take("wo_a", g0 + t)
                col = gi * cfg.o_lora + nbl * s.tile_cols
                for c0, cw in _chunks(s.tile_cols, CW):
                    dd = _dot(xq[:, kb * s.tile_rows : (kb + 1) * s.tile_rows],
                              w[:, c0 : c0 + cw])  # fmt: skip
                    cc = _mo(col + c0)
                    if kb == 0:
                        acc[:, pl.ds(cc, cw)] = dd
                    else:
                        acc[:, pl.ds(cc, cw)] += dd
                ring.release(g0 + t)
            for c0, cw in _chunks(cfg.o_lora, CW):
                cc = _mo(gi * cfg.o_lora + c0)
                part = acc[:, pl.ds(cc, cw)] * inv_f
                z[:, pl.ds(cc, cw)] = mul_vec(part, slot, "wo_a_scale", cc).astype(BF16)
            return c

        if cfg.groups == 1:
            group(0, 0)
        else:
            lax.fori_loop(0, cfg.groups, group, 0)

    return types.SimpleNamespace(
        vec=vec,
        mul_vec=mul_vec,
        rms_norm_to=rms_norm_to,
        tile_loop=tile_loop,
        acc_into=acc_into,
        matmul=matmul,
        quantize_rows=quantize_rows,
        fp8_linear=fp8_linear,
        mhc=mhc,
        finish_sinkhorn=finish_sinkhorn,
        stub_attention=stub_attention,
        wo_a=wo_a,
    )


def _kernel(cfg: DenseChainConfig, typed: bool, faults: Faults, probes: bool, *refs):
    names = cfg.stream_names
    n_in = 3 + 5 + 1 + len(names)
    sched, base, ltype = refs[0:3]
    x0, res0, post0, comb0, cos_sin = refs[3:8]
    smalls_hbm = refs[8]
    tiles = dict(zip(names, refs[9:n_in]))
    x_out, res_out, post_out, comb_out, pf_hbm, pidx_hbm = refs[n_in : n_in + 6]
    scr = refs[n_in + 6 :]
    if typed:
        pool = dict(zip(names, scr[: len(names)]))
        scr = scr[len(names) :]
    else:
        pool, scr = scr[0], scr[1:]
    (
        dsem, sm_v, ssem, res, mixst, tsc, xin, hin, xa, xq, acc, qkv, qr, q, xg, z,
        act, pf, pidx, psem,
    ) = scr  # fmt: skip

    h, hcm = cfg.hidden, cfg.hc_mult
    L = cfg.n_layers
    ring = _Ring(cfg, sched, tiles, pool, dsem, typed, faults)
    ops = make_dense_ops(
        cfg, ring, faults, sm_v=sm_v, res=res, mixst=mixst, tsc=tsc, xin=xin,
        hin=hin, xa=xa, xq=xq, acc=acc, q=q, xg=xg, z=z, cos_sin=cos_sin,
    )  # fmt: skip
    rms_norm_to = ops.rms_norm_to
    matmul, fp8_linear, mhc = ops.matmul, ops.fp8_linear, ops.mhc
    finish_sinkhorn, stub_attention, wo_a = (
        ops.finish_sinkhorn,
        ops.stub_attention,
        ops.wo_a,
    )
    skip = faults.skip

    # ---- one layer -----------------------------------------------------------
    pl_ = cfg.probe_layout

    def layer(lt, li, slot):
        off = cfg.stream_offsets[lt]
        g_l = base[li]

        def g(name):
            return g_l + off[name]

        sk, ss = mhc("hc_attn", g("hc_attn"), slot, "hc_attn_base", "hc_attn_scale")
        amax = rms_norm_to(hin, 0, h, xa, slot, "attn_norm", ss, weight_dtype=F32)

        def emit_qkv(c0, cw, out):
            qkv[:, pl.ds(c0, cw)] = out

        finish_sinkhorn(
            fp8_linear("wqa", g("wqa"), xa, slot, emit_qkv, bg=sk, amax=amax)
        )
        amax_q = rms_norm_to(qkv, 0, cfg.q_lora, qr, slot, "q_norm", weight_dtype=BF16)
        q0 = pl_["qkv"][0]
        pf[slot, :, q0 : q0 + cfg.q_lora] = qr[...].astype(F32)
        rms_norm_to(
            qkv, cfg.q_lora, cfg.head_dim, xg, slot, "kv_norm", weight_dtype=BF16
        )
        pf[slot, :, q0 + cfg.q_lora : q0 + cfg.qkv_cols] = xg[:, : cfg.head_dim].astype(
            F32
        )

        def emit_q(c0, cw, out):
            q[:, pl.ds(c0, cw)] = out

        fp8_linear("wq_b", g("wq_b"), qr, slot, emit_q, amax=amax_q)
        xname = "xproj_csa" if lt == CSA else "xproj_hca"
        matmul(xname, g(xname), lambda k0, w: xa[:, pl.ds(k0, w)])
        x0_ = pl_["xproj"][0]
        nx = cfg.xproj_cols[lt]
        pf[slot, :, x0_ : x0_ + nx] = acc[:, :nx]
        if lt == CSA:

            def emit_idx(c0, cw, out):
                pidx[slot, :, pl.ds(c0, cw)] = out

            fp8_linear("idx_wq_b", g("idx_wq_b"), qr, slot, emit_idx, amax=amax_q)
        stub_attention()
        wo_a(g("wo_a"), slot)

        def emit_x(c0, cw, out):
            xin[:, pl.ds(c0, cw)] = out

        fp8_linear("wo_b", g("wo_b"), z, slot, emit_x)

        sk, ss = mhc("hc_ffn", g("hc_ffn"), slot, "hc_ffn_base", "hc_ffn_scale")
        amax_f = rms_norm_to(hin, 0, h, xa, slot, "ffn_norm", ss, weight_dtype=F32)
        finish_sinkhorn(
            matmul("gate", g("gate"), lambda k0, w: xa[:, pl.ds(k0, w)], sk)
        )
        g0_ = pl_["gate"][0]
        pf[slot, :, g0_ : g0_ + cfg.n_experts] = acc[:, : cfg.n_experts]

        inter = cfg.inter
        assert inter <= CW or inter % CW == 0

        def emit_gu(c0, cw, out):
            qkv[:, pl.ds(c0, cw)] = out  # gate|up staging (qkv is free here)

        fp8_linear("w13", g("w13"), xa, slot, emit_gu, amax=amax_f)
        for c, w in _chunks(inter, EW) if "swiglu" not in skip else ():
            gt = jnp.minimum(qkv[:, c : c + w].astype(F32), cfg.swiglu_limit)
            up = jnp.clip(
                qkv[:, inter + c : inter + c + w].astype(F32),
                -cfg.swiglu_limit,
                cfg.swiglu_limit,
            )
            act[:, c : c + w] = (gt * jax.nn.sigmoid(gt) * up).astype(BF16)
        fp8_linear("w2", g("w2"), act, slot, emit_x)

    # ---- probes ------------------------------------------------------------
    def probe_copies(li, slot):
        return (
            pltpu.make_async_copy(pf.at[slot], pf_hbm.at[li], psem.at[0, slot]),
            pltpu.make_async_copy(pidx.at[slot], pidx_hbm.at[li], psem.at[1, slot]),
        )

    # ---- prologue --------------------------------------------------------------
    def smalls_copy(li, slot):
        return pltpu.make_async_copy(smalls_hbm.at[li], sm_v.at[slot], ssem.at[slot])

    smalls_copy(0, 0).start()

    def prime(gi, c):
        ring.fetch(gi)
        return c

    lax.fori_loop(0, cfg.n_banks, prime, 0)
    xin[...] = x0[...]
    res[...] = res0[...]
    mixst[...] = jnp.zeros(mixst.shape, F32)
    for j in range(hcm):
        mixst[:, POST0 + j : POST0 + j + 1] = post0[:, j : j + 1]
    for k in range(hcm * hcm):
        mixst[:, COMB0 + k : COMB0 + k + 1] = comb0[:, k : k + 1]

    def layer_body(li, c):
        slot = lax.rem(li, 2)
        smalls_copy(0, slot).wait()

        @pl.when(li + 1 < L)
        def _():
            smalls_copy(li + 1, 1 - slot).start()

        if probes:

            @pl.when(li >= 2)
            def _():
                for cp in probe_copies(li - 2, slot):
                    cp.wait()

        is_csa = ltype[li] == CSA
        pl.when(is_csa)(lambda: layer(CSA, li, slot))
        pl.when(jnp.logical_not(is_csa))(lambda: layer(HCA, li, slot))
        if probes:
            for cp in probe_copies(li, slot):
                cp.start()
        return c

    lax.fori_loop(0, L, layer_body, 0)
    if probes:
        for li in range(max(0, L - 2), L):
            for cp in probe_copies(li, li % 2):
                cp.wait()

    x_out[...] = xin[...]
    res_out[...] = res[...]
    post_out[...] = mixst[:, POST0 : POST0 + LANE]
    comb_out[...] = mixst[:, COMB0 : COMB0 + LANE]


def mhc_gates(mixes, sqr, sc, bs, cfg: DenseChainConfig):
    """``mhc.utils.gates_in_block`` up to the Sinkhorn: (pre, post, comb
    logits) from the raw mixes [M, n_mix] and squared sums [M, 1]; sc [1, 3]
    and bs [1, n_mix] f32."""
    m = cfg.hc_mult
    sm = mixes * lax.rsqrt(sqr / (m * cfg.hidden) + cfg.rms_eps)
    pre = jax.nn.sigmoid(sm[:, :m] * sc[:, 0:1] + bs[:, :m]) + cfg.hc_eps
    post_logits = sm[:, m : 2 * m] * sc[:, 1:2] + bs[:, m : 2 * m]
    # Spelled out as stock does: Mosaic's native logistic is less accurate.
    post = (1.0 / (1.0 + jnp.exp(-post_logits))) * cfg.hc_post_mult
    return pre, post, sm[:, 2 * m :] * sc[:, 2:3] + bs[:, 2 * m :]


def _sum4(a, b, c, d):
    return (a + b) + (c + d)


def sinkhorn_init(c, m, eps):
    """``mhc_pre_gates`` comb path up to the first column normalization.

    c: the m*m row-major (i, j) logits as m*m separate same-shaped arrays
    (the kernel uses [1, 128] rows with tokens on lanes, so every group sum is
    plain elementwise VPU work). Returns the same structure.
    """
    assert m == 4, "pairwise sums are written out for hc_mult = 4"
    out = []
    for i in range(m):
        row = c[m * i : m * i + m]
        mx = jnp.maximum(jnp.maximum(row[0], row[1]), jnp.maximum(row[2], row[3]))
        ex = [jnp.exp(v - mx) for v in row]
        den = _sum4(*ex)
        out += [v / den + eps for v in ex]
    return _normalize(out, _COLS, eps)


_ROWS = tuple(tuple(4 * i + j for j in range(4)) for i in range(4))
_COLS = tuple(tuple(4 * i + j for i in range(4)) for j in range(4))


def _normalize(c, groups, eps):
    out = list(c)
    for g in groups:
        den = _sum4(*(c[k] for k in g)) + eps
        for k in g:
            out[k] = c[k] / den
    return out


def sinkhorn_step(c, *, eps):
    """One row then column normalization (``mhc_pre_gates`` loop body)."""
    return _normalize(_normalize(c, _ROWS, eps), _COLS, eps)


def sinkhorn(c, m, eps, iters):
    """The whole comb path on a list of m*m arrays."""
    c = sinkhorn_init(c, m, eps)
    for _ in range(iters - 1):
        c = sinkhorn_step(c, eps=eps)
    return c


def build(
    cfg: DenseChainConfig,
    *,
    interpret: bool | pltpu.InterpretParams = False,
    probes: bool = True,
    faults: Faults | None = None,
):
    """Returns f(tiles, smalls, inputs) -> (x, res, post, comb, probe_f32,
    probe_idx)."""
    M, h = cfg.rows, cfg.hidden
    L = cfg.n_layers
    names = cfg.stream_names
    faults = faults or Faults()
    sched, base = cfg.schedule
    if faults.swap_tiles is not None:
        a, b = faults.swap_tiles
        sched = sched.copy()
        sched[[a, b]] = sched[[b, a]]
    ltype = jax.numpy.asarray(cfg.layer_types, jnp.int32)
    typed = bool(interpret)
    st = cfg.streams
    kmax = max(cfg.hidden, cfg.groups * cfg.o_lora, cfg.inter,
               cfg.heads_per_group * cfg.head_dim)  # fmt: skip
    nmax = max(s.n for s in st.values() if not s.nt)
    pf_w = cfg.probe_layout["_total"][0]
    nidx = cfg.idx_heads * cfg.idx_head_dim

    if typed:
        pool = [
            pltpu.VMEM(
                (cfg.n_banks, st[n].tile_rows, st[n].tile_cols), jnp.dtype(st[n].dtype)
            )  # fmt: skip
            for n in names
        ]
    else:
        pool = [pltpu.VMEM((cfg.n_banks * cfg.max_tile_bytes // 128, 128), jnp.uint8)]
    scratch = pool + [
        pltpu.SemaphoreType.DMA((cfg.n_banks,)),
        pltpu.VMEM((2, cfg.smalls_rows, LANE), F32),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.VMEM((M, cfg.hc_hidden), BF16),  # res
        pltpu.VMEM((M, 2 * LANE), F32),  # pending post / comb mixes
        pltpu.VMEM((LANE, LANE), F32),  # Sinkhorn transpose staging
        pltpu.VMEM((M, h), BF16),  # xin: sublayer output
        pltpu.VMEM((M, h), BF16),  # hin: mHC layer input
        pltpu.VMEM((M, h), BF16),  # xa: normed input
        pltpu.VMEM((M, kmax), FP8),  # xq
        pltpu.VMEM((M, nmax), F32),  # acc
        pltpu.VMEM((M, max(cfg.qkv_cols, 2 * cfg.inter)), BF16),  # qkv / gate-up
        pltpu.VMEM((M, cfg.q_lora), BF16),  # qr
        pltpu.VMEM((M, cfg.heads * cfg.head_dim), BF16),  # q / o
        pltpu.VMEM((M, max(cfg.heads_per_group * cfg.head_dim, cfg.head_dim)), BF16),
        pltpu.VMEM((M, cfg.groups * cfg.o_lora), BF16),  # z
        pltpu.VMEM((M, cfg.inter), BF16),  # act
        pltpu.VMEM((2, M, pf_w), F32),
        pltpu.VMEM((2, M, nidx), BF16),
        pltpu.SemaphoreType.DMA((2, 2)),
    ]
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    anyspace = pl.BlockSpec(memory_space=pl.ANY)
    vmem_need = (
        sum(cfg.n_banks * st[n].tile_bytes for n in names)
        if typed
        else cfg.n_banks * cfg.max_tile_bytes  # fmt: skip
    )
    call = pl.pallas_call(
        functools.partial(_kernel, cfg, typed, faults, probes),
        in_specs=[smem] * 3 + [vmem] * 5 + [anyspace] * (1 + len(names)),
        out_specs=[vmem] * 4 + [anyspace] * 2,
        out_shape=[
            jax.ShapeDtypeStruct((M, h), BF16),
            jax.ShapeDtypeStruct((M, cfg.hc_hidden), BF16),
            jax.ShapeDtypeStruct((M, LANE), F32),
            jax.ShapeDtypeStruct((M, LANE), F32),
            jax.ShapeDtypeStruct((L, M, pf_w), F32),
            jax.ShapeDtypeStruct((L, M, nidx), BF16),
        ],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=min(vmem_need + 40 * 2**20, 64 * 2**20)
            if not typed
            else None,
        ),
        interpret=(
            interpret
            if isinstance(interpret, pltpu.InterpretParams)
            else (pltpu.InterpretParams() if interpret else False)
        ),
        name="dsv4_dense_ring",
    )
    sched_j = jnp.asarray(sched)
    base_j = jnp.asarray(base)

    def run(tiles, smalls, inputs):
        hcm = cfg.hc_mult
        post = jnp.zeros((M, LANE), F32).at[:, :hcm].set(inputs["post"])
        comb = jnp.zeros((M, LANE), F32).at[:, : hcm * hcm].set(inputs["comb"])
        x, res, p, c, pf_o, pidx_o = call(
            sched_j, base_j, ltype, inputs["x"], inputs["res"], post, comb,
            inputs["cos_sin"], smalls, *[tiles[n] for n in names],
        )  # fmt: skip
        return x, res, p[:, :hcm], c[:, : hcm * hcm], pf_o, pidx_o

    return run
