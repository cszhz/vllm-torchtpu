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
"""Exact TensorCore top-k over i32 score-bit rows, usable inside a megakernel.

Same contract as ``sparsecore_topk`` as ``streamindex_topk`` calls it: per row,
the ``k`` largest scores by the monotone int key
``bits ^ ((bits >> 31) & 0x7FFFFFFF)`` (so -0.0 < +0.0, raw bits, no float
compare), positions at or beyond the row length and keys <= key(-inf) never
selected, boundary ties to the lowest column, column indices ascending, ``-1``
suffix when fewer than ``k`` are eligible.

Layout: a *group* is 8 rows on sublanes. Its scores live in VMEM as
``(S * 8, 128)`` i32, vreg ``v`` = rows 0..7 x columns ``128 v .. 128 v + 127``
(``S = n / 128`` segments). The buffer is consumed.

Per group:
1. keys: monotone transform, ineligible -> INT_MIN, in place; the maxima of 4096
   interleaved chunks per row.
2. threshold ("stack", default): L = k-th largest chunk maximum, so >= k keys are
   >= L and, for unstructured rows, few more. Keys > L are pushed onto per-(row,
   lane) stacks; the k-th largest stacked key (or L itself when fewer than k keys
   exceed it) is the threshold. If any stack overflows, or with "radix", an
   MSB-first 2-bit radix over all keys (offset by a lower bound so it spans fewer
   bits, stopping once a splitter has exactly k keys above it) is used instead.
   Every radix keeps each row's prefix in a sublane vector: 8 rows share each op.
3. ranks: per 64-vreg block, the within-segment inclusive prefix of the selection
   (and, only when some row must split a tie run, of the ties first) by one bf16
   matmul against an upper-triangular constant; per-row carries give each segment's
   first output slot (CE) and one-past-last (CI); the buffer becomes global ranks.
4. extraction: per 256-slot chunk and row, a one-hot (slot x segment) matmul over
   the 128-segment blocks the chunk spans gathers each slot's segment rank row (hi
   / lo planes) and (NT matmul) its segment id; the lane is a count.
Every value through the MXU is an integer < 256 (exact in bf16, f32 sums).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

I32, F32, BF16 = jnp.int32, jnp.float32, jnp.bfloat16
INT_MIN = np.int32(-(2**31))
INT_MAX = np.int32(2**31 - 1)
MONO_MASK = np.int32(0x7FFFFFFF)
KEY_NEG_INF = int(np.int32(np.array(-np.inf, np.float32).view(np.int32)) ^ MONO_MASK)
NN = (((1,), (0,)), ((), ()))
NT = (((1,), (1,)), ((), ()))
LANES = 128
SUB = 8
SEGS = 128  # segments per extraction block
SCAN_ALIGN = LANES * 64  # scan granule (columns), 2 ** 13; = LANES * RANK_BLK
N_ALIGN = LANES * SEGS  # n must be a multiple of this
BIG = 1.0e9  # CE / CI of segments past the scanned range


def monotone_key(bits):
    return bits ^ (lax.shift_right_arithmetic(bits, I32(31)) & MONO_MASK)


RANK_BLK = 64  # vregs per rank-pass block: one (512, 128) x (128, 128) matmul
NCHAINS = 2  # independent accumulators per count predicate
CM_VREGS = 32  # chunk-maximum vregs per group: 4096 chunks per row
STACK_D = 24  # stack depth per (row, lane); multiple of 8, <= CM_VREGS


def scratch_shapes(n: int, k: int = 1024, kc: int = 256):
    """VMEM / SMEM scratch ``topk_group`` needs for rows of ``n`` columns."""
    s = n // LANES
    return [
        pltpu.VMEM((SUB, s), F32),  # CE per (row, segment)
        pltpu.VMEM((SUB, s), F32),  # CI per (row, segment)
        pltpu.VMEM((2, RANK_BLK * SUB, LANES), F32),  # rank-pass matmul results, double-buffered
        pltpu.VMEM((SUB, kc, 2 * LANES), F32),  # per-row gathered rank rows (hi, lo)
        pltpu.VMEM((SUB, SUB, kc), F32),  # per-row gathered segment ids
        pltpu.VMEM((CM_VREGS * SUB, LANES), I32),  # chunk maxima, then the key stacks
        pltpu.VMEM((SUB, LANES), I32),  # chunk block ranges + counts
        pltpu.SMEM((SUB, LANES), I32),  # same, readable as scalars
        pltpu.SemaphoreType.DMA(()),
    ]


def _cnt(m):
    return m.astype(I32)


def _i32(x: int) -> np.int32:
    return np.array(x & 0xFFFFFFFF, np.uint32).view(np.int32)


def _lane_sum(x):
    return jnp.broadcast_to(jnp.sum(x, axis=1, keepdims=True), x.shape)


def topk_group(buf, lens, nblk, out, ce_ref, ci_ref, mbuf, acc_ref, inf_ref, aux, tab_v, tab_s, sem,
               *, k, kc=256, threshold="stack", stages=9, union=None, lists=True, values=None,
               thresh=None):
    """Top-k of one 8-row group.

    buf   (S * 8, 128) i32 VMEM: score bits, rows on sublanes (consumed).
    lens  (8, 128) i32 value: each row's eligible length, lane-broadcast.
    nblk  i32 scalar: 8192-column blocks to scan (covers max(lens)).
    out   (8, k) i32 VMEM ref: ascending column indices, -1 suffix.
    stages < 9 stops early and writes a stage result to out (timing breakdown only).
    union  None, or the refs from ``union_scratch_shapes`` preceded by out_union
           ((8k / 128, 128) i32) and out_len ((8, 128) i32): see ``_union``.
    lists  False skips the per-row lists (out is not written); needs ``union``.
    values None, or (out_vals (8, k) i32, *values_scratch_shapes(n, kc)): the monotone key
           of each listed entry, aligned with out, INT_MIN padding. Needs lists.
    thresh None, or (G, allow) (8, 128) i32 lane-broadcast values replacing the group's own
           threshold: select keys > G (monotone-key order) and the first allow keys == G
           in column order (all of them if allow < 0); G = INT_MIN selects every eligible
           key. Union output only (lists=False).
    """
    s_tot = buf.shape[0] // SUB
    assert s_tot % SEGS == 0 and k % kc == 0 and kc % LANES == 0
    nch = k // kc
    assert 4 * nch + 1 <= LANES
    lane = lax.broadcasted_iota(I32, (SUB, LANES), 1)
    rows_per_rblk = RANK_BLK * SUB
    nit8 = nblk * (RANK_BLK // 8)  # 8-vreg scan iterations

    def vreg_ref(v):
        return buf.at[pl.ds(pl.multiple_of(v * SUB, SUB), SUB), :]

    def stage_out(x):
        out[:, 0:LANES] = x.astype(I32)

    def count_pass(preds, load, niter8):
        """Per-row counts (lane-broadcast) of each predicate over vregs load(0 .. 8 * niter8 - 1)."""
        z = jnp.zeros((SUB, LANES), I32)

        def it(j, acc):
            acc = list(acc)
            for u in range(8):
                x = load(j * 8 + u)
                for q, f in enumerate(preds):
                    i = NCHAINS * q + u % NCHAINS
                    acc[i] = acc[i] + _cnt(f(x))
            return tuple(acc)

        acc = lax.fori_loop(0, niter8, it, (z,) * (NCHAINS * len(preds)))
        return [_lane_sum(sum(acc[NCHAINS * q:NCHAINS * (q + 1)])) for q in range(len(preds))]

    def radix_select(load, niter8, npass, done0):
        """Per-row k-th largest i32 value over the loaded vregs (INT_MIN if fewer than k are
        > INT_MIN), all values assumed < INT_MIN + 4 ** npass in biased order. MSB-first 2-bit
        digits; a row is done once some splitter has exactly k values at or above it (the
        threshold returned is then that splitter - 1, which selects the same k)."""
        def cond(c):
            i, _, done, _ = c
            return (i < npass) & (jnp.min(done) == 0)

        def body(c):
            i, p, done, tdone = c
            sh = 2 * (npass - 1 - i)
            splits = [(p | lax.shift_left(I32(d), sh)) ^ INT_MIN for d in (1, 2, 3)]
            cnts = count_pass([lambda x, s=s_: x >= s for s_ in splits], load, niter8)
            dig = sum(_cnt(n >= k) for n in cnts)
            n_at = jnp.where(dig == 1, cnts[0], jnp.where(dig == 2, cnts[1], cnts[2]))
            split = jnp.where(dig == 1, splits[0], jnp.where(dig == 2, splits[1], splits[2]))
            hit = (done == 0) & (dig > 0) & (n_at == k)
            tdone = jnp.where(hit, split - 1, tdone)
            p = jnp.where(done == 0, p | lax.shift_left(dig, sh), p)
            return i + 1, p, done | _cnt(hit), tdone

        z = jnp.zeros((SUB, LANES), I32)
        _, p, done, tdone = lax.while_loop(
            cond, body, (I32(0), z, done0, jnp.full((SUB, LANES), INT_MIN, I32)))
        return jnp.where(done > 0, tdone, p ^ INT_MIN)

    def load_buf(v):
        return vreg_ref(v)[...]

    def load_aux(v):
        return aux.at[pl.ds(pl.multiple_of(v * SUB, SUB), SUB), :][...]

    def npass_for(span):
        """Radix passes covering biased-unsigned spans: smallest m with span < 4 ** m, group max."""
        npass_v = jnp.ones((SUB, LANES), I32)
        for m in range(1, 16):
            npass_v = npass_v + _cnt(span >= _i32((1 << (2 * m)) ^ (1 << 31)))
        return jnp.max(npass_v)

    def aux_select(nv, base, hi):
        """k-th largest per row of the nv aux vregs (consumed), counting only values >= base;
        INT_MIN if fewer than k. Offsetting by base shrinks the radix to bitlen(hi - base + 1)."""
        bm1 = base - 1
        nvalid = jnp.zeros((SUB, LANES), I32)
        for i in range(nv):
            x = aux[i * SUB:(i + 1) * SUB, :]
            ok = x >= base
            aux[i * SUB:(i + 1) * SUB, :] = jnp.where(ok, (x - bm1) ^ INT_MIN, INT_MIN)
            nvalid = nvalid + _cnt(ok)
        nvalid = _lane_sum(nvalid)
        span = jnp.where(hi >= base, hi - bm1, 0) ^ INT_MIN
        t_kb = radix_select(load_aux, nv // 8, npass_for(span), _cnt(nvalid < k))
        return jnp.where(nvalid < k, INT_MIN, (t_kb ^ INT_MIN) + bm1)

    # 1. monotone keys (ineligible -> INT_MIN) in place, and the maxima of 4096 interleaved
    # chunks per row (chunk = lane x vreg mod 32) into aux.
    for i in range(CM_VREGS):
        aux[i * SUB:(i + 1) * SUB, :] = jnp.full((SUB, LANES), INT_MIN, I32)

    def keys_it(j, c):
        base = (j & (CM_VREGS // 8 - 1)) * 8
        for u in range(8):
            v = j * 8 + u
            r = vreg_ref(v)
            key = monotone_key(r[...])
            ok = (v * LANES + lane < lens) & (key > KEY_NEG_INF)
            key = jnp.where(ok, key, INT_MIN)
            r[...] = key
            if values is not None:
                values[1][pl.ds(pl.multiple_of(v * SUB, SUB), SUB), :] = key
            a_ = aux.at[pl.ds(pl.multiple_of((base + u) * SUB, SUB), SUB), :]
            a_[...] = jnp.maximum(a_[...], key)
        return c

    lax.fori_loop(0, nit8, keys_it, 0)
    cms = [aux[i * SUB:(i + 1) * SUB, :] for i in range(CM_VREGS)]
    lo = jnp.broadcast_to(jnp.min(functools.reduce(jnp.minimum, cms), axis=1, keepdims=True), (SUB, LANES))
    hi = jnp.broadcast_to(jnp.max(functools.reduce(jnp.maximum, cms), axis=1, keepdims=True), (SUB, LANES))
    lo = jnp.maximum(lo, KEY_NEG_INF + 1)  # >= 4096 >= k keys are >= lo
    if stages == 1:
        return stage_out(lo + hi)

    if thresh is None:
        def radix_threshold():
            """Exact threshold by radix over all keys (offset by lo so it spans fewer bits)."""
            lom1 = lo - 1

            def off_it(j, acc):
                acc = list(acc)
                for u in range(8):
                    r = vreg_ref(j * 8 + u)
                    key = r[...]
                    ok = key >= lo
                    r[...] = jnp.where(ok, (key - lom1) ^ INT_MIN, INT_MIN)
                    acc[u % NCHAINS] = acc[u % NCHAINS] + _cnt(ok)
                return tuple(acc)

            n_ok = _lane_sum(sum(lax.fori_loop(0, nit8, off_it, (jnp.zeros((SUB, LANES), I32),) * NCHAINS)))
            span = jnp.where(hi >= lo, hi - lom1, 0) ^ INT_MIN  # biased unsigned max kb per row
            t = radix_select(load_buf, nit8, npass_for(span), _cnt(n_ok < k))
            n_gt, n_ge = count_pass([lambda x: x > t, lambda x: x >= t], load_buf, nit8)
            return t, n_gt, n_ge

        if threshold == "radix":
            t, n_gt, n_ge = radix_threshold()
        else:
            # L = k-th largest chunk maximum: >= k keys are >= L, and few more than k for
            # unstructured rows. Keys > L are pushed onto per-(row, lane) stacks of depth
            # STACK_D; the k-th largest stacked key is the threshold unless some stack
            # overflowed, in which case the radix above runs instead.
            big_l = aux_select(CM_VREGS, lo, hi)
            if stages == 2:
                return stage_out(big_l)

            def stack_it(j, c):
                st, cnt, ceq = list(c[:STACK_D]), c[STACK_D], list(c[STACK_D + 1:])
                for u in range(8):
                    x = vreg_ref(j * 8 + u)[...]
                    push = x > big_l
                    for d in range(STACK_D - 1, 0, -1):
                        st[d] = jnp.where(push, st[d - 1], st[d])
                    st[0] = jnp.where(push, x, st[0])
                    cnt = cnt + _cnt(push)
                    ceq[u % NCHAINS] = ceq[u % NCHAINS] + _cnt(x == big_l)
                return (*st, cnt, *ceq)

            zi = jnp.zeros((SUB, LANES), I32)
            res = lax.fori_loop(0, nit8, stack_it,
                                (jnp.full((SUB, LANES), INT_MIN, I32),) * STACK_D + (zi,) * (1 + NCHAINS))
            cnt = res[STACK_D]
            c_gt = _lane_sum(cnt)
            c_eq = _lane_sum(sum(res[STACK_D + 1:]))
            overflow = jnp.max(cnt) > STACK_D

            def stack_threshold():
                for d in range(STACK_D):
                    aux[d * SUB:(d + 1) * SUB, :] = res[d]
                base = jnp.maximum(jnp.where(big_l == INT_MAX, big_l, big_l + 1), KEY_NEG_INF + 1)
                ts = aux_select(STACK_D, base, hi)
                # unstacked keys are <= L, so the full-row threshold is max(ts, L)
                ts = jnp.maximum(ts, big_l)
                s_gt = _lane_sum(sum(_cnt(res[d] > ts) for d in range(STACK_D)))
                s_ge = _lane_sum(sum(_cnt(res[d] >= ts) for d in range(STACK_D)))
                s_ge = jnp.where(ts == big_l, s_gt + c_eq, s_ge)
                at_l = c_gt < k  # then the threshold is L itself
                return (jnp.where(at_l, big_l, ts), jnp.where(at_l, c_gt, s_gt),
                        jnp.where(at_l, c_gt + c_eq, s_ge))

            t, n_gt, n_ge = lax.cond(overflow, radix_threshold, stack_threshold)
        if stages == 3:
            return stage_out(t)
        real_t = t != INT_MIN
        n_sel = jnp.where(real_t, k, n_gt)
        need = jnp.where(real_t, k - n_gt, 0)
        n_eq = n_ge - n_gt
        take_all = real_t & (need >= n_eq)
    else:
        # external cut: keys > G, plus the first `allow` keys == G in column order (all if < 0)
        assert not lists, "lists need the group's own threshold"
        t, allow = thresh
        real_t = t != INT_MIN
        need = allow
        take_all = real_t & (allow < 0)
        rank_ties = jnp.max(_cnt(real_t & (allow >= 0))) > 0

    # 4. ranks: global inclusive rank of the selection (f32, in place) and per-segment CE / CI.
    ri = lax.broadcasted_iota(I32, (LANES, LANES), 0)
    li = lax.broadcasted_iota(I32, (LANES, LANES), 1)
    upper = (ri <= li).astype(BF16)
    if thresh is None:
        rank_ties = jnp.max(_cnt(real_t & (need > 0) & (need < n_eq))) > 0

    def init_tab(i, c):
        o = pl.multiple_of(i * LANES, LANES)
        ce_ref[:, pl.ds(o, LANES)] = jnp.full((SUB, LANES), BIG, F32)
        ci_ref[:, pl.ds(o, LANES)] = jnp.full((SUB, LANES), BIG, F32)
        return c

    lax.fori_loop(0, s_tot // LANES, init_tab, 0)
    t_b = jnp.tile(t, (RANK_BLK, 1))
    take_b = jnp.tile(take_all, (RANK_BLK, 1))
    need_b = jnp.tile(need.astype(F32), (RANK_BLK, 1))
    real_b = jnp.tile(real_t, (RANK_BLK, 1))

    def blk(b):
        return buf.at[pl.ds(pl.multiple_of(b * rows_per_rblk, rows_per_rblk), rows_per_rblk), :]

    def excl_prefix(cnt, carry):
        """per-vreg counts (rows_per_rblk, 128) -> carry + exclusive prefix over the block's
        vregs, and the new carry. Two-level (groups of 8) so the add chains stay short."""
        parts = [cnt[i * SUB:(i + 1) * SUB] for i in range(RANK_BLK)]
        groups = [parts[g:g + 8] for g in range(0, RANK_BLK, 8)]
        inner = []
        for grp in groups:
            run, ex = None, []
            for x in grp:
                ex.append(run)
                run = x if run is None else run + x
            inner.append((ex, run))
        outs, off = [], carry
        for ex, tot in inner:
            outs.extend(off if e is None else off + e for e in ex)
            off = off + tot
        return jnp.concatenate(outs, axis=0), off

    def _tree_sum(xs):
        while len(xs) > 1:
            xs = [xs[i] + xs[i + 1] if i + 1 < len(xs) else xs[i] for i in range(0, len(xs), 2)]
        return xs[0]

    def sel_plain(kb):
        return (kb > t_b) | ((kb == t_b) & take_b)

    def issue(b, slot):
        mbuf[slot] = lax.dot_general(sel_plain(blk(b)[...]).astype(BF16), upper, NN,
                                     preferred_element_type=F32)

    def finish(b, m, scar, sel=None):
        if values is not None:
            vb = values[1].at[pl.ds(pl.multiple_of(b * rows_per_rblk, rows_per_rblk), rows_per_rblk), :]
            vb[...] = jnp.where(sel_plain(blk(b)[...]) if sel is None else sel, vb[...], 0)
        cnt = jnp.broadcast_to(m[:, LANES - 1:], m.shape)
        ce, scar2 = excl_prefix(cnt, scar)
        blk(b)[...] = pltpu.bitcast(ce + m, I32)  # global inclusive rank
        ci = ce + cnt
        # segment 64 b + i -> lane (b % 2) * 64 + i of table tile b // 2
        o = pl.multiple_of(lax.shift_right_logical(b, 1) * LANES, LANES)
        base = (b & 1) * RANK_BLK
        rel = lane - base
        hits = [rel == i for i in range(RANK_BLK)]
        mine = (rel >= 0) & (rel < RANK_BLK)
        for tab_ref, vals in ((ce_ref, ce), (ci_ref, ci)):
            part = _tree_sum([jnp.where(hits[i], vals[i * SUB:(i + 1) * SUB], 0.0) for i in range(RANK_BLK)])
            tab_ref[:, pl.ds(o, LANES)] = jnp.where(mine, part, tab_ref[:, pl.ds(o, LANES)])
        return scar2

    def rank_plain():
        issue(0, 0)

        def body(b, scar):
            @pl.when(b + 1 < nblk)
            def _():
                issue(b + 1, (b + 1) & 1)
            return finish(b, mbuf[b & 1], scar)

        lax.fori_loop(0, nblk, body, jnp.zeros((SUB, LANES), F32))

    def rank_tied():
        def body(b, carry):
            tcar, scar = carry
            kb = blk(b)[...]
            gt = kb > t_b
            eq = (kb == t_b) & real_b
            e = lax.dot_general(eq.astype(BF16), upper, NN, preferred_element_type=F32)
            e_off, tcar2 = excl_prefix(jnp.broadcast_to(e[:, LANES - 1:], e.shape), tcar)
            sel = gt | (eq & ((e_off + e <= need_b) | take_b))
            m = lax.dot_general(sel.astype(BF16), upper, NN, preferred_element_type=F32)
            return tcar2, finish(b, m, scar, sel)

        zf = jnp.zeros((SUB, LANES), F32)
        lax.fori_loop(0, nblk, body, (zf, zf))

    if union is not None:
        _union(union, blk=blk, nblk=nblk, s_tot=s_tot, k=k, upper=upper, rank_ties=rank_ties,
               sel_plain=sel_plain, t_b=t_b, take_b=take_b, need_b=need_b, real_b=real_b,
               excl_prefix=excl_prefix)
        if not lists:
            return

    lax.cond(rank_ties, rank_tied, rank_plain)

    # The extraction reads whole 128-segment blocks; zero the unscanned half of the last one,
    # whose bits are whatever the caller left past the row lengths (one-hot 0 x NaN is NaN).
    @pl.when((nblk & 1) == 1)
    def _():
        blk(nblk)[...] = jnp.zeros((rows_per_rblk, LANES), I32)
    if stages == 4:
        return stage_out(ce_ref[:, 0:LANES])

    # 5a. per chunk: the union over rows of the 128-segment blocks holding its slots
    ntile = s_tot // LANES
    n_self = n_sel.astype(F32)
    tab = jnp.where(lane == LANES - 1, n_sel, 0)
    for c in range(nch):
        lo_c = F32(c * kc)
        hi_c = jnp.minimum(F32((c + 1) * kc), n_self)
        rf = jnp.zeros((SUB, LANES), I32)
        re = jnp.zeros((SUB, LANES), I32)
        for i in range(ntile):
            rf = rf + _cnt(ci_ref[:, i * LANES:(i + 1) * LANES] <= lo_c)
            re = re + _cnt(ce_ref[:, i * LANES:(i + 1) * LANES] < hi_c)
        b_lo = lax.shift_right_logical(_lane_sum(rf), 7)
        b_hi = jnp.maximum(lax.shift_right_logical(_lane_sum(re) + SEGS - 1, 7), b_lo + 1)
        live = n_sel > c * kc
        b_lo = jnp.where(live, jnp.minimum(b_lo, ntile - 1), ntile - 1)
        b_hi = jnp.where(live, jnp.minimum(b_hi, ntile), 0)
        u_lo = jnp.min(b_lo, axis=0, keepdims=True)
        u_hi = jnp.maximum(jnp.max(b_hi, axis=0, keepdims=True), u_lo + 1)
        tab = jnp.where(lane == c, u_lo, tab)
        tab = jnp.where(lane == nch + c, u_hi, tab)
    tab_v[...] = tab
    cp = pltpu.make_async_copy(tab_v, tab_s, sem)
    cp.start()
    cp.wait()

    # 5b. extraction: slot t of row r lives in the segment with CE <= t < CI, at the lane
    # where the segment's global inclusive rank first exceeds t.
    t_col = lax.broadcasted_iota(I32, (kc, 1), 0).astype(F32)
    seg_lane = lax.broadcasted_iota(I32, (SUB, LANES), 1).astype(F32)
    info_row = lax.broadcasted_iota(I32, (SUB, LANES), 0)
    ones8 = jnp.ones((SUB, LANES), BF16)
    t_row = lax.broadcasted_iota(I32, (1, kc), 1).astype(F32)

    def chunk_body(c, c0):
        tc = t_col + (c * kc).astype(F32)
        b_lo = tab_s[0, c]
        b_hi = tab_s[0, nch + c]

        def gather(b, first):
            o = pl.multiple_of(b * LANES, LANES)
            for r in range(SUB):
                ce_row = ce_ref[r:r + 1, pl.ds(o, LANES)]
                ci_row = ci_ref[r:r + 1, pl.ds(o, LANES)]
                oht = ((ce_row <= tc) & (tc < ci_row)).astype(BF16)  # (kc, 128 segments)
                blk_rows = buf.at[pl.ds(pl.multiple_of(b * (SEGS * SUB), SEGS * SUB), SEGS * SUB)]
                rank = pltpu.bitcast(blk_rows[pl.ds(r, SEGS, stride=SUB), :], F32)
                rh = jnp.floor(rank * (1.0 / LANES))
                rhs = jnp.concatenate([rh, rank - LANES * rh], axis=1).astype(BF16)
                g = lax.dot_general(oht, rhs, NN, preferred_element_type=F32)
                info = jnp.where(info_row == 0, seg_lane, jnp.where(info_row == 1, b.astype(F32), 0.0))
                gi = lax.dot_general(info.astype(BF16), oht, NT, preferred_element_type=F32)  # (8, kc)
                if first:
                    acc_ref[r] = g
                    inf_ref[r] = gi
                else:
                    acc_ref[r] += g
                    inf_ref[r] += gi
                if values is not None:  # selected keys (0 elsewhere) as 4 exact byte planes
                    kv = values[1][pl.ds(b * (SEGS * SUB) + r, SEGS, stride=SUB), :]
                    planes = jnp.concatenate(
                        [lax.shift_right_logical(kv, 8 * j) & 255 for j in range(4)], axis=1)
                    gv = lax.dot_general(oht, planes.astype(F32).astype(BF16), NN,
                                         preferred_element_type=F32)
                    if first:
                        values[2][r] = gv
                    else:
                        values[2][r] += gv

        gather(b_lo, True)

        def blk_body(b, c1):
            gather(b, False)
            return c1

        lax.fori_loop(b_lo + 1, b_hi, blk_body, 0)
        rows, vrows = [], []
        for r in range(SUB):
            a = acc_ref[r]
            rank_g = LANES * a[:, :LANES] + a[:, LANES:]
            ln = lax.dot_general(ones8, (rank_g <= tc).astype(BF16), NT, preferred_element_type=F32)[0:1]
            gi = inf_ref[r]
            pos = LANES * (SEGS * gi[1:2] + gi[0:1]) + ln
            nsel_r = tab_s[r, LANES - 1].astype(F32)
            live = t_row + (c * kc).astype(F32) < nsel_r
            rows.append(jnp.where(live, pos, -1.0).astype(I32))
            if values is not None:
                # the selected lane is the only one with rank t + 1 and a nonzero plane entry
                at = rank_g == tc + 1
                vg = values[2][r]
                key = functools.reduce(jnp.bitwise_or, [lax.shift_left(lax.dot_general(
                    ones8, jnp.where(at, vg[:, j * LANES:(j + 1) * LANES], 0.0).astype(BF16), NT,
                    preferred_element_type=F32)[0:1].astype(I32), 8 * j) for j in range(4)])
                vrows.append(jnp.where(live, key, INT_MIN))
        out[:, pl.ds(pl.multiple_of(c * kc, kc), kc)] = jnp.concatenate(rows, axis=0)
        if values is not None:
            values[0][:, pl.ds(pl.multiple_of(c * kc, kc), kc)] = jnp.concatenate(vrows, axis=0)
        return c0

    lax.fori_loop(0, nch, chunk_body, 0)


U_KC = 1024  # union slots per extraction chunk (8 output rows of 128)


def values_scratch_shapes(n: int, kc: int = 256):
    """Scratch for ``topk_group(values=(out_vals, *this))``."""
    return [
        pltpu.VMEM((n // LANES * SUB, LANES), I32),  # keys, then selected keys (0 elsewhere)
        pltpu.VMEM((SUB, kc, 4 * LANES), F32),  # per-row gathered key byte planes
    ]


def union_scratch_shapes(n: int, k: int = 1024):
    """Scratch for the union output, after out_union / out_len in ``topk_group(union=...)``."""
    s = n // LANES
    nch = 8 * k // U_KC
    assert 2 * nch + 1 <= LANES
    return [
        pltpu.VMEM((s, LANES), F32),  # member bits per column, one segment per row
        pltpu.VMEM((s, LANES), F32),  # union inclusive rank per column
        pltpu.VMEM((s // LANES * SUB, LANES), F32),  # CE per segment, lanes = segments
        pltpu.VMEM((s // LANES * SUB, LANES), F32),  # CI per segment, same
        pltpu.VMEM((U_KC, 3 * LANES), F32),  # gathered (rank hi, rank lo, bits) rows
        pltpu.VMEM((SUB, U_KC), F32),  # gathered segment ids
        pltpu.VMEM((SUB, LANES), I32),  # chunk block ranges + union size
        pltpu.SMEM((SUB, LANES), I32),  # same, readable as scalars
        pltpu.SemaphoreType.DMA(()),
    ]


def _union(refs, *, blk, nblk, s_tot, k, upper, rank_ties, sel_plain, t_b, take_b, need_b, real_b,
           excl_prefix):
    """Ascending union of the 8 rows' selections. out_union entry i at [i // 128, i % 128] is
    ``column << 8 | bits`` (bit r set iff row r selected the column), -1 after the last;
    out_len (8, 128) holds the union size in every element.

    Member bits per column are one matmul per block (P[i, 8 i + r] = 2 ** r against the
    selection), which turns the union into a single row of n columns, one 128-column segment
    per sublane row; it is then ranked and extracted like one row of the lists path.
    """
    out_u, out_len, ub, urank, uce, uci, uacc, uinf, utab_v, utab_s, usem = refs
    rows = RANK_BLK * SUB
    ri = lax.broadcasted_iota(I32, (LANES, LANES), 0)
    li = lax.broadcasted_iota(I32, (LANES, LANES), 1)
    pi = lax.broadcasted_iota(I32, (RANK_BLK, rows), 0)
    pj = lax.broadcasted_iota(I32, (RANK_BLK, rows), 1)
    pack = jnp.where(lax.shift_right_logical(pj, 3) == pi,
                     lax.shift_left(1, pj & 7), 0).astype(F32).astype(BF16)
    ntile = s_tot // LANES
    nub = lax.shift_right_logical(nblk + 1, 1)  # 128-segment union blocks covering the scan

    def put_bits(b, sel):
        ub[pl.ds(pl.multiple_of(b * RANK_BLK, RANK_BLK), RANK_BLK), :] = lax.dot_general(
            pack, sel.astype(BF16), NN, preferred_element_type=F32)

    def bits_plain():
        def body(b, c):
            put_bits(b, sel_plain(blk(b)[...]))
            return c

        lax.fori_loop(0, nblk, body, 0)

    def bits_tied():
        def body(b, tcar):
            kb = blk(b)[...]
            eq = (kb == t_b) & real_b
            e = lax.dot_general(eq.astype(BF16), upper, NN, preferred_element_type=F32)
            e_off, tcar2 = excl_prefix(jnp.broadcast_to(e[:, LANES - 1:], e.shape), tcar)
            put_bits(b, (kb > t_b) | (eq & ((e_off + e <= need_b) | take_b)))
            return tcar2

        lax.fori_loop(0, nblk, body, jnp.zeros((SUB, LANES), F32))

    lax.cond(rank_ties, bits_tied, bits_plain)

    @pl.when((nblk & 1) == 1)
    def _():
        ub[pl.ds(pl.multiple_of(nblk * RANK_BLK, RANK_BLK), RANK_BLK), :] = jnp.zeros((RANK_BLK, LANES), F32)

    for i in range(ntile):
        uce[i * SUB:(i + 1) * SUB, :] = jnp.full((SUB, LANES), BIG, F32)
        uci[i * SUB:(i + 1) * SUB, :] = jnp.full((SUB, LANES), BIG, F32)
    lower_x = (ri > li).astype(BF16)  # [s, s'] = s' < s
    ones8 = jnp.ones((SUB, LANES), BF16)

    def rank_body(j, carry):
        o = pl.multiple_of(j * LANES, LANES)
        pre = lax.dot_general((ub[pl.ds(o, LANES), :] != 0).astype(BF16), upper, NN,
                              preferred_element_type=F32)
        cnt = jnp.broadcast_to(pre[:, LANES - 1:], pre.shape)  # per segment (row), <= 128
        ce_col = lax.dot_general(lower_x, cnt.astype(BF16), NN, preferred_element_type=F32)
        urank[pl.ds(o, LANES), :] = jnp.tile(carry, (LANES // SUB, 1)) + ce_col + pre
        cnt_b = cnt.astype(BF16)
        ce_row = carry + lax.dot_general(ones8, jnp.where(ri < li, cnt_b, 0), NN, preferred_element_type=F32)
        ci_row = carry + lax.dot_general(ones8, jnp.where(ri <= li, cnt_b, 0), NN, preferred_element_type=F32)
        uce[pl.ds(pl.multiple_of(j * SUB, SUB), SUB), :] = ce_row
        uci[pl.ds(pl.multiple_of(j * SUB, SUB), SUB), :] = ci_row
        return jnp.broadcast_to(ci_row[:, LANES - 1:], ci_row.shape)

    n_u = lax.fori_loop(0, nub, rank_body, jnp.zeros((SUB, LANES), F32))
    out_len[...] = n_u.astype(I32)

    lane = lax.broadcasted_iota(I32, (SUB, LANES), 1)
    nch = out_u.shape[0] * LANES // U_KC
    tab = jnp.where(lane == LANES - 1, n_u.astype(I32), 0)
    for c in range(nch):
        rf = jnp.zeros((SUB, LANES), I32)
        re = jnp.zeros((SUB, LANES), I32)
        hi_c = jnp.minimum(F32((c + 1) * U_KC), n_u)
        for i in range(ntile):
            rf = rf + _cnt(uci[i * SUB:i * SUB + 1, :] <= F32(c * U_KC))
            re = re + _cnt(uce[i * SUB:i * SUB + 1, :] < hi_c)
        b_lo = lax.shift_right_logical(_lane_sum(rf), 7)
        b_hi = jnp.maximum(lax.shift_right_logical(_lane_sum(re) + SEGS - 1, 7), b_lo + 1)
        tab = jnp.where(lane == c, jnp.minimum(b_lo, ntile - 1), tab)
        tab = jnp.where(lane == nch + c, jnp.minimum(b_hi, ntile), tab)
    utab_v[...] = tab
    cp = pltpu.make_async_copy(utab_v, utab_s, usem)
    cp.start()
    cp.wait()

    def fill(i, c):
        out_u[pl.ds(pl.multiple_of(i * SUB, SUB), SUB), :] = jnp.full((SUB, LANES), -1, I32)
        return c

    lax.fori_loop(0, out_u.shape[0] // SUB, fill, 0)
    t_col = lax.broadcasted_iota(I32, (U_KC, 1), 0).astype(F32)
    t_row = lax.broadcasted_iota(I32, (1, U_KC), 1).astype(F32)
    seg_lane = lane.astype(F32)
    info_row = lax.broadcasted_iota(I32, (SUB, LANES), 0)
    n_u_s = utab_s[0, LANES - 1]

    def chunk_body(c, c0):
        tc = t_col + (c * U_KC).astype(F32)

        def gather(b, first):
            o = pl.multiple_of(b * LANES, LANES)
            ce_row = uce[pl.ds(pl.multiple_of(b * SUB, SUB), SUB), :][0:1]
            ci_row = uci[pl.ds(pl.multiple_of(b * SUB, SUB), SUB), :][0:1]
            oht = ((ce_row <= tc) & (tc < ci_row)).astype(BF16)  # (U_KC, 128 segments)
            rank = urank[pl.ds(o, LANES), :]
            rh = jnp.floor(rank * (1.0 / LANES))
            rhs = jnp.concatenate([rh, rank - LANES * rh, ub[pl.ds(o, LANES), :]], axis=1).astype(BF16)
            g = lax.dot_general(oht, rhs, NN, preferred_element_type=F32)
            info = jnp.where(info_row == 0, seg_lane, jnp.where(info_row == 1, b.astype(F32), 0.0))
            gi = lax.dot_general(info.astype(BF16), oht, NT, preferred_element_type=F32)
            if first:
                uacc[...] = g
                uinf[...] = gi
            else:
                uacc[...] += g
                uinf[...] += gi

        b_lo = utab_s[0, c]
        gather(b_lo, True)

        def blk_body(b, c1):
            gather(b, False)
            return c1

        lax.fori_loop(b_lo + 1, utab_s[0, nch + c], blk_body, 0)
        a = uacc[...]
        rank_g = LANES * a[:, :LANES] + a[:, LANES:2 * LANES]
        ln = lax.dot_general(ones8, (rank_g <= tc).astype(BF16), NT, preferred_element_type=F32)[0:1]
        bsel = jnp.where(rank_g == tc + 1, a[:, 2 * LANES:], 0.0).astype(BF16)
        bits = lax.dot_general(ones8, bsel, NT, preferred_element_type=F32)[0:1]
        gi = uinf[...]
        col = LANES * (SEGS * gi[1:2] + gi[0:1]) + ln
        ent = lax.shift_left(col.astype(I32), 8) | bits.astype(I32)
        ent = jnp.where(t_row + (c * U_KC).astype(F32) < n_u_s.astype(F32), ent, -1)
        tile = jnp.concatenate([ent[:, q * LANES:(q + 1) * LANES] for q in range(U_KC // LANES)], axis=0)
        out_u[pl.ds(pl.multiple_of(c * (U_KC // LANES), SUB), U_KC // LANES), :] = tile
        return c0

    lax.fori_loop(0, lax.shift_right_logical(n_u_s + U_KC - 1, 10), chunk_body, 0)


def to_group_layout(scores):
    """(R, n) i32 -> (G, n / 128 * 8, 128): 8-row groups, rows on sublanes."""
    r, n = scores.shape
    g = -(-r // SUB)
    x = jnp.pad(scores, ((0, g * SUB - r), (0, 0)))
    return x.reshape(g, SUB, n // LANES, LANES).transpose(0, 2, 1, 3).reshape(g, n // LANES * SUB, LANES)


@functools.partial(jax.jit, static_argnames=("k", "interpret", "stages", "threshold", "kc"))
def tc_topk_grouped(groups, row_lengths, *, k=1024, interpret=False, stages=9, threshold="stack",
                    kc=256):
    """Standalone wrapper: ``groups`` from ``to_group_layout``; returns i32[G * 8, k]."""
    g, rows, _ = groups.shape
    n = rows // SUB * LANES
    assert n % N_ALIGN == 0, n
    lens = jnp.pad(row_lengths.astype(I32), (0, g * SUB - row_lengths.shape[0]))
    lens = jnp.clip(lens, 0, n)

    def kernel(lens_s, x_ref, o_ref, *scr):
        gi = pl.program_id(0)
        sub = lax.broadcasted_iota(I32, (SUB, LANES), 0)
        lv = jnp.zeros((SUB, LANES), I32)
        mx = I32(0)
        for i in range(SUB):
            li = lens_s[gi * SUB + i]
            lv = jnp.where(sub == i, li, lv)
            mx = jnp.maximum(mx, li)
        nblk = lax.shift_right_logical(mx + SCAN_ALIGN - 1, 13)
        topk_group(x_ref.at[0], lv, nblk, o_ref.at[0], *scr, k=k, kc=kc, stages=stages, threshold=threshold)

    out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(g,),
            in_specs=[pl.BlockSpec((1, rows, LANES), lambda i, l: (i, 0, 0))],
            out_specs=pl.BlockSpec((1, SUB, k), lambda i, l: (i, 0, 0)),
            scratch_shapes=scratch_shapes(n, k, kc),
        ),
        out_shape=jax.ShapeDtypeStruct((g, SUB, k), I32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=56 * 1024 * 1024
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(lens, groups)
    return out.reshape(g * SUB, k)


@functools.partial(jax.jit, static_argnames=("k", "interpret", "lists"))
def tc_topk_union_grouped(groups, row_lengths, *, k=1024, interpret=False, lists=True):
    """Standalone wrapper for the union output: (lists i32[G * 8, k] or None,
    union i32[G, 8 k / 128, 128], union size i32[G])."""
    g, rows, _ = groups.shape
    n = rows // SUB * LANES
    assert n % N_ALIGN == 0, n
    lens = jnp.clip(jnp.pad(row_lengths.astype(I32), (0, g * SUB - row_lengths.shape[0])), 0, n)
    nscr = len(scratch_shapes(n))
    u_rows = 8 * k // LANES

    def kernel(lens_s, x_ref, o_ref, u_ref, ul_ref, *scr):
        gi = pl.program_id(0)
        sub = lax.broadcasted_iota(I32, (SUB, LANES), 0)
        lv = jnp.zeros((SUB, LANES), I32)
        mx = I32(0)
        for i in range(SUB):
            li = lens_s[gi * SUB + i]
            lv = jnp.where(sub == i, li, lv)
            mx = jnp.maximum(mx, li)
        nblk = lax.shift_right_logical(mx + SCAN_ALIGN - 1, 13)
        topk_group(x_ref.at[0], lv, nblk, o_ref.at[0], *scr[:nscr], k=k,
                   union=(u_ref.at[0], ul_ref.at[0], *scr[nscr:]), lists=lists)

    lists_o, union, ulen = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(g,),
            in_specs=[pl.BlockSpec((1, rows, LANES), lambda i, l: (i, 0, 0))],
            out_specs=[pl.BlockSpec((1, SUB, k), lambda i, l: (i, 0, 0)),
                       pl.BlockSpec((1, u_rows, LANES), lambda i, l: (i, 0, 0)),
                       pl.BlockSpec((1, SUB, LANES), lambda i, l: (i, 0, 0))],
            scratch_shapes=scratch_shapes(n) + union_scratch_shapes(n, k),
        ),
        out_shape=[jax.ShapeDtypeStruct((g, SUB, k), I32),
                   jax.ShapeDtypeStruct((g, u_rows, LANES), I32),
                   jax.ShapeDtypeStruct((g, SUB, LANES), I32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=56 * 1024 * 1024
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(lens, groups)
    return (lists_o.reshape(g * SUB, k) if lists else None), union, ulen[:, 0, 0]


@functools.partial(jax.jit, static_argnames=("k", "interpret", "lists", "union", "values"))
def tc_topk_ext(groups, row_lengths, thresh_key=None, thresh_allow=None, *, k=1024,
                interpret=False, lists=True, union=False, values=False):
    """Standalone wrapper for every topk_group output. thresh_key / thresh_allow: optional
    i32[G * 8] external cut (see topk_group). Returns a dict with "lists" i32[G * 8, k],
    "values" i32[G * 8, k], "union" i32[G, 8 k / 128, 128], "union_len" i32[G] as requested."""
    g, rows, _ = groups.shape
    n = rows // SUB * LANES
    assert n % N_ALIGN == 0, n
    lens = jnp.clip(jnp.pad(row_lengths.astype(I32), (0, g * SUB - row_lengths.shape[0])), 0, n)
    ext = thresh_key is not None
    u_rows = 8 * k // LANES
    scr = scratch_shapes(n)
    n_main = len(scr)
    n_val = len(values_scratch_shapes(n)) if values else 0
    if values:
        scr = scr + values_scratch_shapes(n)
    if union:
        scr = scr + union_scratch_shapes(n, k)

    def bcast(x):
        x = jnp.pad(x.astype(I32), (0, g * SUB - x.shape[0])).reshape(g, SUB, 1)
        return jnp.broadcast_to(x, (g, SUB, LANES))

    def kernel(lens_s, x_ref, *refs):
        if ext:
            tk_ref, ta_ref, *refs = refs
        o_ref, v_ref, u_ref, ul_ref, *sc = refs
        gi = pl.program_id(0)
        sub = lax.broadcasted_iota(I32, (SUB, LANES), 0)
        lv = jnp.zeros((SUB, LANES), I32)
        mx = I32(0)
        for i in range(SUB):
            li = lens_s[gi * SUB + i]
            lv = jnp.where(sub == i, li, lv)
            mx = jnp.maximum(mx, li)
        nblk = lax.shift_right_logical(mx + SCAN_ALIGN - 1, 13)
        topk_group(
            x_ref.at[0], lv, nblk, o_ref.at[0], *sc[:n_main], k=k, lists=lists,
            values=(v_ref.at[0], *sc[n_main:n_main + n_val]) if values else None,
            union=(u_ref.at[0], ul_ref.at[0], *sc[n_main + n_val:]) if union else None,
            thresh=(tk_ref[0], ta_ref[0]) if ext else None)

    blk3 = lambda shape: pl.BlockSpec((1, *shape), lambda i, l: (i, 0, 0))  # noqa: E731
    in_specs = [blk3((rows, LANES))] + ([blk3((SUB, LANES))] * 2 if ext else [])
    outs = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(g,),
            in_specs=in_specs,
            out_specs=[blk3((SUB, k)), blk3((SUB, k)), blk3((u_rows, LANES)), blk3((SUB, LANES))],
            scratch_shapes=scr,
        ),
        out_shape=[jax.ShapeDtypeStruct((g, SUB, k), I32), jax.ShapeDtypeStruct((g, SUB, k), I32),
                   jax.ShapeDtypeStruct((g, u_rows, LANES), I32),
                   jax.ShapeDtypeStruct((g, SUB, LANES), I32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=56 * 1024 * 1024
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(lens, groups, *((bcast(thresh_key), bcast(thresh_allow)) if ext else ()))
    res = {}
    if lists:
        res["lists"] = outs[0].reshape(g * SUB, k)
    if values:
        res["values"] = outs[1].reshape(g * SUB, k)
    if union:
        res["union"], res["union_len"] = outs[2], outs[3][:, 0, 0]
    return res


def tc_topk(scores, k, row_lengths, *, interpret=False, threshold="stack"):
    """(R, n) i32 score bits -> i32[R, k]; same contract as ``sparsecore_topk``."""
    r = scores.shape[0]
    return tc_topk_grouped(
        to_group_layout(scores), row_lengths, k=k, interpret=interpret, threshold=threshold)[:r]
