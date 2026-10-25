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
"""DSv4 routed MoE for the decode megakernel: router, expert weight ring, and
the in-kernel all-reduce, in TP or EP mode.

Every rank holds all T (<= 64) token rows. Each expert's FFN is cut into R
slices of the intermediate dim (S = inter / R columns of gate and of up, the
matching S rows of w2). A work item is one (expert, slice):

  tp  rank r runs slice r of every expert any token hit;
  ep  rank r runs all R slices of the experts it owns (E / R of them).

Both modes run the same item body and end with the same all-reduce of the f32
partial [T, hidden], so they differ only in who streams which bytes. Weights
stay W4A8 as in the stock fused EP kernel: fp4 e2m1 widened to fp8 in VMEM,
f32 scales per 512-row contraction block. A slice's intermediate columns
are chosen (slice_columns) as at most two pieces, each inside one scale
block, with the same widths on every slice; w2 is contracted piece by piece,
each piece taking its own block's scale, so nothing is requantized.

Numerics differ from the stock kernel in three places, all reassociation or
finer rounding: the intermediate's fp8 row scale is the max over this slice
(stock: over all 3072 columns), expert outputs are summed in f32 without the
stock's fp8 wire rounding, and partial sums cross ranks as bf16.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from . import moe_router as rt
from .allreduce import COLLECTIVE_ID, AllReduce, AllReduceSpec, barrier

BF16, F32, I32, U32 = jnp.bfloat16, jnp.float32, jnp.int32, jnp.uint32
FP4, FP8 = jnp.float4_e2m1fn, jnp.float8_e4m3fn
PACK4 = 8  # fp4 values per u32 word, consecutive contraction rows
DN = (((1,), (0,)), ((), ()))
MODES = ("tp", "ep")
CT = 32  # compact tile rows (the fp8 sublane tile)
# compact: rows of a down item added inside fused(), the rest after it; each
# slot costs every item a tile read-modify-write (step_device2 T=64: 2 best)
KS = 2
# Positive control of the compact checks, read at trace time: "row" adds the
# first gathered row of every item to the next accumulator row.
COMPACT_FAULT = ""
# Time attribution only (wrong numerics), read at trace time: "gather" skips
# the row gathers, "scatter" the accumulator adds.
COMPACT_ABLATE: frozenset[str] = frozenset()


@dataclasses.dataclass(frozen=True)
class MoEConfig:
    R: int = 8
    mode: str = "tp"
    rows: int = 32  # compute rows M (>= tokens, multiple of 32: fp8 sublane tile)
    ar_rows: int = 16  # all-reduced rows (>= tokens, multiple of 16)
    hidden: int = 7168
    inter: int = 3072
    n_experts: int = 384
    topk: int = 6
    qb: int = 512  # fp4 scale block along the contraction
    renormalize: bool = True
    routed_scaling_factor: float = 2.5
    swiglu_limit: float = 10.0
    hash_routing: bool = False
    nbuf: int = 4  # expert-slice ring depth
    w1_chunks: int = 7  # DMAs per w1 slice (waited block by block)
    w1_cols: int = 256  # gate|up output columns per accumulator pass
    w2_cols: int = 256  # output columns per w2 dot
    item_groups: int = 1  # rolled iterations per item body (code ~1/G; each
    # iteration boundary exposes an MXU drain, measured +1 us/item at G=7)
    ar_mode: str = "flat"
    ar_slots: int = 1  # 2 when consecutive calls alternate slots (per layer)
    # "sliced": slice_expert_weights tiles. "stock" (ep only): the served fp4
    # operands in place -- w1 [E, H, 2I], w2 [E, I, H] -- with the scale
    # tables w1s [E, H/qb, 2I], w2s [E, I/qb, H] row-padded once by
    # pad_scale_rows; each item DMAs its slice's column / row ranges.
    layout: str = "sliced"
    # the shared expert as this rank's 1/R intermediate slice (fp8, per
    # channel scales), its partial added before the all-reduce
    shared: bool = False
    ep_slices: int = 8  # ep: slices per local expert (work-item size)
    # Mosaic's per-DMA bounds checks cost ~300 bundles per item (12% of it);
    # every DMA index here comes from the validated hit list
    bounds_checks: bool = False
    shared_inter: int = 3072
    # the item body runs only the rows routed to its expert, gathered CT at a
    # time (MoEBlock.build_rows); per row the arithmetic is unchanged
    compact: bool = False

    def __post_init__(self):
        assert self.mode in MODES, self.mode
        assert self.layout in ("sliced", "stock") and (
            self.layout == "sliced" or self.mode == "ep"
        )
        assert self.inter % self.n_slices == 0 and self.n_experts % self.R == 0
        assert self.S % 64 == 0, f"slice width {self.S} must be a 64-lane multiple"
        assert self.qb >= self.S, "a slice spans at most two scale blocks"
        assert self.hidden % self.qb == 0 and self.inter % self.qb == 0
        assert (self.hidden // self.qb) % self.w1_chunks == 0
        assert self.hidden % self.w2_cols == 0 and (2 * self.S) % self.w1_cols == 0
        assert (
            self.rows % 32 == 0 and self.ar_rows % 16 == 0 and self.ar_rows <= self.rows
        )
        assert self.n_experts % 128 == 0
        assert not self.compact or self.rows <= 128  # row ids pack 4 per word

    @property
    def n_slices(
        self,
    ):  # tp: one slice per rank; ep: an internal tiling of a local expert
        return self.R if self.mode == "tp" else self.ep_slices

    @property
    def S(self):
        return self.inter // self.n_slices

    @property
    def nb1(self):  # w1 scale rows per slice, padded to the f32 sublane tile
        return -(-(self.hidden // self.qb) // 8) * 8

    @property
    def pieces(self):  # widths of a slice's scale-block pieces (slice_columns)
        return slice_columns(self.n_slices, self.inter, self.qb)[1]

    @property
    def Ssh(self):  # shared-expert intermediate columns per rank
        return self.shared_inter // self.R

    @property
    def EL(self):  # experts held per rank
        return self.n_experts if self.mode == "tp" else self.n_experts // self.R

    @property
    def SL(self):  # slices held per expert per rank
        return 1 if self.mode == "tp" else self.n_slices

    @property
    def item_rows(self):  # rows the item body computes
        return CT if self.compact else self.rows

    @property
    def n_chunks(self):  # compact: CT-row chunks an expert's rows may need
        return self.rows // CT

    @property
    def ar(self):
        return AllReduceSpec(
            self.R,
            self.ar_rows,
            self.hidden,
            self.ar_mode,
            slots=self.ar_slots,
            group=min(8, self.R),
        )


# ----------------------------------------------------------------------------
# Host-side weight layout (once, at load): no values change.


def pack_fp4_rows(w):
    """fp4 [..., K, N] -> u32 [..., K/8, N]: word (i, n) holds rows 8i..8i+7 of
    column n, lowest row in the lowest nibble (pltpu.bitcast's unpack order)."""
    *lead, k, n = w.shape
    w = w.reshape(*lead, k // PACK4, PACK4, n)
    return lax.bitcast_convert_type(jnp.swapaxes(w, -1, -2), U32)


def unpack_fp4_rows(packed):
    """Inverse of pack_fp4_rows, decoded arithmetically to exact f32 values
    (no sub-byte dtype needed): u32 [..., K/8, N] -> f32 [..., K, N]."""
    *lead, k8, n = packed.shape
    nib = jnp.stack(
        [(packed >> (4 * j)) & 0xF for j in range(PACK4)], axis=-2
    )  # [..., K/8, 8, N]
    mag = jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], F32)[nib & 7]
    val = jnp.where(nib >= 8, -mag, mag)
    return val.reshape(*lead, k8 * PACK4, n)


def slice_columns(R, inter, qb):
    """Intermediate columns of each slice, int [R, S], and the slice's piece
    widths. A slice is at most two pieces, each inside one scale block, and
    every slice has the same piece widths, so the w2 contraction split is
    static: with S = 384, qb = 512 each slice is 256 columns of one block
    plus 128 of another (8 x 256 fill four blocks, 8 x 128 the other two)."""
    S = inter // R
    if qb % S == 0:
        return np.arange(inter).reshape(R, S), (S,)
    p0 = 1 << (S.bit_length() - 1)
    p1 = S - p0
    if qb % p0 or qb % p1 or (R * p0) % qb:
        raise ValueError(f"no two-piece slicing of {inter} into {R} for block {qb}")
    first = np.arange(R * p0).reshape(R, p0)
    second = R * p0 + np.arange(R * p1).reshape(R, p1)
    return np.concatenate([first, second], axis=1), (p0, p1)


def split_slices(x, R, pieces):
    """The slice_columns permutation on the last axis, by reshapes only (no
    gather; XLA's TPU gather refuses fp4): [..., R * S] -> [..., R, S]."""
    *lead, _ = x.shape
    parts, k0 = [], 0
    for p in pieces:
        parts.append(x[..., k0 : k0 + R * p].reshape(*lead, R, p))
        k0 += R * p
    return jnp.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]


def merge_slices(x, pieces):
    """Inverse of the slice_columns permutation on the last two axes:
    [..., R, S] (slice r's columns in slice order) -> [..., R * S] logical."""
    *lead, R, S = x.shape
    parts, k0 = [], 0
    for p in pieces:
        parts.append(x[..., k0 : k0 + p].reshape(*lead, R * p))
        k0 += p
    return jnp.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]


def slice_expert_weights(w1, w1s, w2, w2s, *, R, qb=512):
    """Stock operand layout -> per-(expert, slice) tiles (a column/row
    permutation; no value changes).

    w1 fp4 [E, H, 2I] (gate | up), w1s f32 [E, H/qb, 2I], w2 fp4 [E, I, H],
    w2s f32 [E, I/qb, H]. Returns w1p u32 [E, R, H/8, 2S] (slice s's gate
    columns then its up columns), w1sp f32 [E, R, H/qb padded to 8, 2S], w2p
    u32 [E, R, S/8, H], w2sp f32 [E, R, 2, H] (the scale block of each of the
    slice's pieces). Shard axis 1 for tp, axis 0 for ep."""
    E, H, inter2 = w1.shape
    inter = inter2 // 2
    idx, pieces = slice_columns(R, inter, qb)

    def cols(a):  # [..., 2I] -> [..., R, 2S] with the slice axis moved to 1
        g = split_slices(a[..., :inter], R, pieces)
        u = split_slices(a[..., inter:], R, pieces)
        return jnp.moveaxis(jnp.concatenate([g, u], axis=-1), -2, 1)

    w1p = pack_fp4_rows(cols(w1))
    w1sp = cols(w1s)
    pad = -(-w1sp.shape[2] // 8) * 8 - w1sp.shape[2]  # natural (8, 128) tiling
    w1sp = jnp.pad(w1sp, ((0, 0), (0, 0), (0, pad), (0, 0)))
    w2r = split_slices(jnp.swapaxes(w2, 1, 2), R, pieces)  # [E, H, R, S]
    w2p = pack_fp4_rows(jnp.transpose(w2r, (0, 2, 3, 1)))
    blocks = idx[:, [0, pieces[0] % idx.shape[1]]] // qb  # [R, 2]
    w2sp = jnp.stack([w2s[:, blocks[:, 0]], w2s[:, blocks[:, 1]]], axis=2)
    return w1p, w1sp, w2p, w2sp


def slice_shared_weights(w13, s13, w2, s2, *, R):
    """Shared expert fp8 w13 [H, 2I] (gate | up), s13 f32 [2I], w2 [I, H],
    s2 f32 [H] -> per-rank slices sw13 [R, H, 2I/R], s13 [R, 1, 2I/R],
    sw2 [R, I/R, H], s2 [R, 1, H] (s2 replicated; the sum adds it R times
    over disjoint partials, which is the RowParallel convention)."""
    H, inter2 = w13.shape
    inter = inter2 // 2
    S = inter // R

    def cols(a):  # [..., 2I] -> [R, ..., 2S]
        g = a[..., :inter].reshape(*a.shape[:-1], R, S)
        u = a[..., inter:].reshape(*a.shape[:-1], R, S)
        return jnp.moveaxis(jnp.concatenate([g, u], axis=-1), -2, 0)

    return (
        cols(w13),
        cols(s13[None, :]),
        w2.reshape(R, S, H),
        jnp.broadcast_to(s2[None, None, :], (R, 1, H)),
    )


def pad_scale_rows(scale):
    """[E, blocks, N] -> [E, blocks padded to 8, N]. Done once at load: an
    unpadded table's default TPU layout is not the row-major tiling a kernel
    reads, so XLA would otherwise relayout it on every call."""
    pad = -(-scale.shape[1] // 8) * 8 - scale.shape[1]
    return jnp.pad(scale, ((0, 0), (0, pad), (0, 0)))


def weight_spec(mode, ax):
    return P(None, ax) if mode == "tp" else P(ax)


# ----------------------------------------------------------------------------
# In-kernel pieces.


class MoEBlock:
    """The routed-MoE body over scratch `s` for one rank `me`, whose local
    experts are block `expert_block` of the global ids (default `me`)."""

    def __init__(self, cfg: MoEConfig, s, me, w_hbm, expert_block=None, placement=None):
        self.cfg, self.s, self.me = cfg, s, me
        self.eblk = me if expert_block is None else expert_block
        # placement: None (local experts = contiguous block `eblk`) or
        # (held, copy, layer): SMEM int32 [L, slots] global expert per local
        # slot (-1 = empty) and None or [L, slots] copy_idx << 8 | n_copies.
        # A copy runs the slices s with s mod n_copies == copy_idx on all rows:
        # here an expert's cost is its weight stream, so splitting rows (the
        # stock kernel's rule) would save nothing.
        self.placement = placement
        self.w1_hbm, self.w1s_hbm, self.w2_hbm, self.w2s_hbm = w_hbm

    @staticmethod
    def scratch(cfg: MoEConfig):
        H, E, M, S, NB = cfg.hidden, cfg.n_experts, cfg.rows, cfg.S, cfg.nbuf
        return {
            "xq": pltpu.VMEM((M, H), FP8),
            "xs": pltpu.VMEM((M, 128), F32),
            "dense": pltpu.VMEM((M, E), F32),
            "hitv": pltpu.VMEM((8, 128), I32),
            "hits": pltpu.SMEM((8, 128), I32),
            "elist": pltpu.SMEM((E,), I32),
            "jlist": pltpu.SMEM((E,), I32),  # local slot of each elist entry
            # with expert copies: item -> entry << 9 | first << 8 | slice
            "ilist": pltpu.SMEM((2 * cfg.EL * cfg.SL + 64,), I32),
            "w1_vm": pltpu.VMEM((NB, H // PACK4, 2 * S), U32),
            "w2_vm": pltpu.VMEM((NB, S // PACK4, H), U32),
            **(
                {  # stock layout: whole-expert scale tables, double-buffered by expert
                    "es1": pltpu.VMEM((2, cfg.nb1, 2 * cfg.inter), F32),
                    "es2": pltpu.VMEM((2, 8, H), F32),
                    "esem": pltpu.SemaphoreType.DMA((2, 2)),
                }
                if cfg.layout == "stock"
                else {
                    "w1s_vm": pltpu.VMEM((NB, cfg.nb1, 2 * S), F32),
                    # 2 rows used; 8 is the f32 tile, so a raw-VMEM view is exact
                    "w2s_vm": pltpu.VMEM((NB, 8, H), F32),
                }
            ),
            # compact: 8 more rows, the target of fused()'s unused scatter slots
            "acc": pltpu.VMEM((M + 8 * cfg.compact, H), F32),
            "mid": pltpu.VMEM((cfg.item_rows, S), FP8),
            "coef": pltpu.VMEM((cfg.item_rows, 128), F32),
            "msem": pltpu.SemaphoreType.DMA((1,)),
            "wsem": pltpu.SemaphoreType.DMA((NB, max(cfg.w1_chunks + 3, 12))),
            **(MoEBlock._compact_scratch(cfg) if cfg.compact else {}),
        }

    @staticmethod
    def _compact_scratch(cfg: MoEConfig):
        M, H = cfg.rows, cfg.hidden
        return {
            # rtab[p, j]: the p-th row routed to local slot j (ascending)
            "rtab": pltpu.VMEM((M, 128), I32),
            # rtab packed 4 rows per word (rows [0, M/4)), then the per-slot
            # row counts (row M/4), for the scalar side
            "rtv": pltpu.VMEM((M // 4 + 8, 128), I32),
            "rts": pltpu.SMEM((M // 4 + 8, 128), I32),
            "xsl": pltpu.VMEM((8, 128), F32),  # row 0: x's row scales by lane
            "nsub": pltpu.SMEM((1,), I32),  # (item, chunk) steps of the layer
            # xq as u32 words of 4 rows (pltpu.bitcast), for row gathers
            "xq4": pltpu.VMEM((M // 4, H), U32),
            "xg4": pltpu.VMEM((CT // 4, H), U32),  # one chunk being gathered
            # the current entry's gathered rows, chunk c at rows [c*CT, c*CT+CT)
            "xg": pltpu.VMEM((M, H), FP8),
            "gsc": pltpu.VMEM((M, 128), F32),  # their x row scales
            "gw": pltpu.VMEM((M, 128), F32),  # their routing weights
            "ybuf": pltpu.VMEM((CT, H), F32),  # the down item's output rows
        }

    # -- routing ---------------------------------------------------------------
    def quantize_rows(self, x):
        q, scale = rt.rowquant_fp8(x)
        self.s["xq"][...] = q
        self.s["xs"][...] = jnp.broadcast_to(scale, self.s["xs"].shape)

    def route(self, logits, live_rows, bias=None, hash_ids=None):
        cfg = self.cfg
        ids, ws = rt.route(
            logits,
            bias=bias,
            hash_ids=hash_ids,
            live_rows=live_rows,
            topk=cfg.topk,
            n_experts=cfg.n_experts,
            renormalize=cfg.renormalize,
            routed_scaling_factor=cfg.routed_scaling_factor,
        )
        dense, hit = rt.combine_matrix(ids, ws, cfg.n_experts)
        self.s["dense"][...] = dense
        return ids, ws, hit

    def build_worklist(self, hit):
        """[1, E] hit mask -> this rank's ascending hit experts in SMEM `elist`;
        returns the number of work items."""
        cfg, s = self.cfg, self.s
        for j in range(cfg.n_experts // 128):
            s["hitv"][j : j + 1, :] = hit[:, j * 128 : (j + 1) * 128]
        cp = pltpu.make_async_copy(s["hitv"], s["hits"], s["msem"].at[0])
        cp.start()
        cp.wait()
        base = 0 if cfg.mode == "tp" else self.eblk * cfg.EL
        if self.placement is None:
            n_slots = cfg.EL

            def expert_of(j):
                return base + j

        else:
            held, _, layer = self.placement
            n_slots = held.shape[1]

            def expert_of(j):
                return held[layer, j]

        def add(j, n):
            e = expert_of(j)
            ec = jnp.maximum(e, 0)
            s["elist"][n] = ec
            s["jlist"][n] = j
            return n + jnp.where(e >= 0, s["hits"][ec >> 7, ec & 127], 0)

        n_hit = lax.fori_loop(0, n_slots, add, jnp.int32(0), unroll=8)
        self._n_hit = n_hit  # compact: run() counts the (item, chunk) steps
        if not self._copies():
            return n_hit * cfg.SL
        _, copy, layer = self.placement

        def entry_items(m, n):  # copy k of nc runs slices k, k + nc, ...
            info = copy[layer, s["jlist"][m]]
            k, nc = info >> 8, info & 255

            def one(t, n):
                s["ilist"][n] = (m << 9) | (jnp.where(t == 0, 1, 0) << 8) | (k + t * nc)
                return n + 1

            return lax.fori_loop(0, (cfg.SL - k + nc - 1) // nc, one, n)

        return lax.fori_loop(0, n_hit, entry_items, jnp.int32(0))

    def _slot_experts(self):
        """(slot count, f(j) -> global expert of local slot j, -1 = empty)."""
        cfg = self.cfg
        if self.placement is None:
            base = 0 if cfg.mode == "tp" else self.eblk * cfg.EL
            return cfg.EL, lambda j: base + j
        held, _, layer = self.placement
        return held.shape[1], lambda j: held[layer, j]

    # -- compact rows ------------------------------------------------------------
    def build_rows(self, ids):
        """Compact mode (run() calls it, after build_worklist; `ids`: topk
        [M, 1] id columns): per local slot j the rows routed to it in ascending
        order (rtab[p, j] for p < count) and the counts, packed into SMEM
        `rts` for the scatter; x's row scales as a lane vector. Vector work:
        a row's position in its expert's list is a prefix count over rows."""
        cfg, s = self.cfg, self.s
        M = cfg.rows
        n_slots, expert_of = self._slot_experts()
        assert n_slots <= 128, n_slots
        lane = lax.broadcasted_iota(I32, (1, 128), 1)
        if self.placement is None:
            g = jnp.where(lane < n_slots, expert_of(lane), -1)
        else:
            g = lax.fori_loop(
                0,
                n_slots,
                lambda j, g: jnp.where(lane == j, expert_of(j), g),
                jnp.full((1, 128), -1, I32),
            )
        hit = jnp.zeros((M, 128), F32)
        for i in ids:  # masked rows carry id E, empty slots -1: no match
            hit = jnp.where(i == g, 1.0, hit)
        sub = lax.broadcasted_iota(I32, (M, M), 0)
        before = jnp.where(lax.broadcasted_iota(I32, (M, M), 1) < sub, 1.0, 0.0)
        # inclusive count of hits up to each row (small integers: exact)
        incl = (
            lax.dot_general(
                before.astype(BF16), hit.astype(BF16), DN, preferred_element_type=F32
            )
            + hit
        )
        count = jnp.sum(hit, axis=0, keepdims=True)

        def rows_at(p):  # the p-th hit row = the number of rows before it
            n = jnp.sum(
                jnp.where(incl <= p.astype(F32), 1.0, 0.0), axis=0, keepdims=True
            )
            return jnp.where(p.astype(F32) < count, n, 0.0).astype(I32)

        def word(w, c):
            packed = jnp.zeros((1, 128), I32)
            for q in range(4):
                p = 4 * w + q
                r = rows_at(p)
                s["rtab"][pl.ds(p, 1), :] = r
                packed = packed | (r << (8 * q))
            s["rtv"][pl.ds(w, 1), :] = packed
            return c

        lax.fori_loop(0, M // 4, word, 0)
        crow = M // 4
        s["rtv"][crow : crow + 8, :] = jnp.where(
            lax.broadcasted_iota(I32, (8, 128), 0) == 0, count.astype(I32), 0
        )
        cp = pltpu.make_async_copy(s["rtv"], s["rts"], s["msem"].at[0])
        cp.start()
        s["xq4"][...] = pltpu.bitcast(s["xq"][...], U32)
        xs = s["xs"][:, 0:1]
        diag = lax.broadcasted_iota(I32, (M, 128), 0) == lax.broadcasted_iota(
            I32, (M, 128), 1
        )
        s["xsl"][...] = jnp.broadcast_to(
            jnp.sum(jnp.where(diag, xs, 0.0), axis=0, keepdims=True), (8, 128)
        )
        cp.wait()

    def _count(self, j):
        return self.s["rts"][self.cfg.rows // 4, j]

    def _row(self, p, j):
        """The p-th row routed to local slot j."""
        return (self.s["rts"][p >> 2, j] >> ((p & 3) * 8)) & 255

    def _n_chunks(self, m):
        return (self._count(self.s["jlist"][m]) + CT - 1) // CT

    def _count_steps(self, n_hit):
        """SMEM nsub: the (item, chunk) steps of the layer's items."""
        cfg, s = self.cfg, self.s

        def items_of(m):
            if not self._copies():
                return cfg.SL
            _, copy, layer = self.placement
            info = copy[layer, s["jlist"][m]]
            k, nc = info >> 8, info & 255
            return (cfg.SL - k + nc - 1) // nc

        s["nsub"][0] = lax.fori_loop(
            0, n_hit, lambda m, n: n + items_of(m) * self._n_chunks(m), jnp.int32(0)
        )

    def gather(self, m):
        """Entry m's routed rows of xq into xg, chunk by chunk (byte copies
        between the u32 views of 4-row groups; rows past the count are
        zero), with their x scales and routing weights (lane selects)."""
        cfg, s = self.cfg, self.s
        if "gather" in COMPACT_ABLATE:
            return
        M, E = cfg.rows, cfg.n_experts
        j, e = s["jlist"][m], s["elist"][m]
        count = self._count(j)
        iota_e = lax.broadcasted_iota(I32, (M, E), 1)
        wcol = jnp.sum(
            jnp.where(iota_e == e, s["dense"][...], 0.0), axis=1, keepdims=True
        )
        diag = lax.broadcasted_iota(I32, (M, 128), 0) == lax.broadcasted_iota(
            I32, (M, 128), 1
        )
        wl = jnp.sum(jnp.where(diag, wcol, 0.0), axis=0, keepdims=True)
        xl = s["xsl"][0:1, :]
        lane = lax.broadcasted_iota(I32, (CT, 128), 1)
        row = lax.broadcasted_iota(I32, (CT, 1), 0)
        for c in range(cfg.n_chunks):
            rs = slice(c * CT, (c + 1) * CT)

            def one(c=c, rs=rs):
                tab = s["rtab"][rs, :].astype(F32)
                rows = jnp.sum(jnp.where(lane == j, tab, 0.0), axis=1, keepdims=True)
                rows = jnp.where(c * CT + row < count, rows.astype(I32), -1)
                sel = rows == lane
                s["gsc"][rs, :] = jnp.broadcast_to(
                    jnp.sum(jnp.where(sel, xl, 0.0), axis=1, keepdims=True), (CT, 128)
                )
                s["gw"][rs, :] = jnp.broadcast_to(
                    jnp.sum(jnp.where(sel, wl, 0.0), axis=1, keepdims=True), (CT, 128)
                )
                s["xg4"][...] = jnp.zeros(s["xg4"].shape, U32)

                def copy(t, cc):
                    r = self._row(c * CT + t, j)
                    w = s["xq4"][pl.ds(r >> 2, 1), :]
                    b = (w >> ((r & 3) * 8).astype(U32)) & jnp.uint32(255)
                    q = pl.ds(t >> 2, 1)
                    s["xg4"][q, :] = s["xg4"][q, :] | (b << ((t & 3) * 8).astype(U32))
                    return cc

                lax.fori_loop(0, jnp.minimum(count - c * CT, CT), copy, 0)
                s["xg"][rs, :] = pltpu.bitcast(s["xg4"][...], FP8)

            if c == 0:
                one()
            else:
                pl.when(count > c * CT)(one)

    def scatter_rows(self, i, c, valid):
        """(row count, acc rows of the first KS) of item i's chunk c; unused
        slots point at a spare row past M."""
        cfg = self.cfg
        j = self.s["jlist"][self._entry(i)[0]]
        n = jnp.where(valid, jnp.clip(self._count(j) - c * CT, 0, CT), 0)
        if "scatter" in COMPACT_ABLATE:
            n = jnp.int32(0)
        rows = []
        for t in range(KS):
            r = self._row(c * CT + t, j)
            if COMPACT_FAULT == "row" and t == 0:
                r = lax.rem(r + 1, cfg.rows)
            rows.append(jnp.where(t < n, r, cfg.rows))
        return n, rows

    def scatter(self, i, c, valid, n):
        """acc[row] += ybuf[t] * coef[t] for rows KS.. of item i's chunk c (fused()
        added the first KS), in order, with coef still the item's stash; the
        expression is fused()'s."""
        s = self.s
        j = s["jlist"][self._entry(i)[0]]

        def put(t, carry):
            r = self._row(c * CT + t, j)
            contrib = s["ybuf"][pl.ds(t, 1), :] * s["coef"][pl.ds(t, 1), 0:1]
            contrib = jnp.where(valid, contrib, 0.0)
            s["acc"][pl.ds(r, 1), :] = s["acc"][pl.ds(r, 1), :] + contrib
            return carry

        lax.fori_loop(KS, n, put, 0)

    def _copies(self):
        return self.placement is not None and self.placement[1] is not None

    def _entry(self, i):
        """(worklist entry, slice, first item of the entry) of item i."""
        if self._copies():
            v = self.s["ilist"][i]
            return v >> 9, v & 255, ((v >> 8) & 1) == 1
        sl = lax.rem(i, self.cfg.n_slices)
        return i // self.cfg.n_slices, sl, sl == 0

    # -- weight ring -----------------------------------------------------------
    def item(self, i):
        """(global expert, local expert, local slice) of item i."""
        cfg, s = self.cfg, self.s
        if cfg.mode == "tp":
            e = s["elist"][i]
            return e, e, 0
        m, sl, _ = self._entry(i)
        return s["elist"][m], s["jlist"][m], sl

    def copies(self, e_loc, s_loc, slot, refs=None):
        if self.cfg.layout == "stock":
            return self._stock_copies(e_loc, s_loc, slot, refs)
        cfg, s = self.cfg, self.s
        sem = s["wsem"]
        rc = cfg.hidden // PACK4 // cfg.w1_chunks
        out = [
            pltpu.make_async_copy(
                self.w1_hbm.at[e_loc, s_loc, pl.ds(c * rc, rc)],
                s["w1_vm"].at[slot, pl.ds(c * rc, rc)],
                sem.at[slot, c],
            )
            for c in range(cfg.w1_chunks)
        ]
        n = cfg.w1_chunks
        out.append(
            pltpu.make_async_copy(
                self.w1s_hbm.at[e_loc, s_loc], s["w1s_vm"].at[slot], sem.at[slot, n]
            )
        )
        out.append(
            pltpu.make_async_copy(
                self.w2_hbm.at[e_loc, s_loc], s["w2_vm"].at[slot], sem.at[slot, n + 1]
            )
        )
        out.append(
            pltpu.make_async_copy(
                self.w2s_hbm.at[e_loc, s_loc],
                s["w2s_vm"].at[slot, pl.ds(0, 2)],
                sem.at[slot, n + 2],
            )
        )
        return out

    def _stock_copies(self, e, sl, slot, refs=None, starts=None):
        """Slice `sl` of local expert `e` straight from the stock operands
        (w1/w2 viewed as packed u32): per piece one column range of w1 for
        gate and for up, and one row range of w2 (scales: _scale_copies).
        `starts` (piece column starts) may be precomputed by the caller."""
        cfg, s = self.cfg, self.s
        sem, inter = s["wsem"], cfg.inter
        w1_hbm, _, w2_hbm, _ = refs or (self.w1_hbm, None, self.w2_hbm, None)
        starts = starts if starts is not None else self.piece_starts(sl)
        out = []
        k0 = 0
        for j, p in enumerate(cfg.pieces):
            start = starts[j]
            for h, base in enumerate((0, inter)):
                src = pl.ds(pl.multiple_of(base + start, math.gcd(p, 128)), p)
                dst = pl.ds(h * cfg.S + k0, p)
                out.append(
                    pltpu.make_async_copy(
                        w1_hbm.at[e, :, src],
                        s["w1_vm"].at[slot, :, dst],
                        sem.at[slot, len(out)],
                    )
                )
            rows = pl.ds(pl.multiple_of(start // PACK4, 8), p // PACK4)
            out.append(
                pltpu.make_async_copy(
                    w2_hbm.at[e, rows],
                    s["w2_vm"].at[slot, pl.ds(k0 // PACK4, p // PACK4)],
                    sem.at[slot, len(out)],
                )
            )
            k0 += p
        return out

    def piece_starts(self, sl):
        """Intermediate column where each piece of slice `sl` starts."""
        cfg = self.cfg
        return [
            sl * p + cfg.n_slices * sum(cfg.pieces[:j])
            for j, p in enumerate(cfg.pieces)
        ]

    def _scale_copies(self, e, buf, refs=None):
        """Stock layout: expert e's whole scale tables into buffer `buf`
        (once per expert; fewer bytes and descriptors than per slice)."""
        s = self.s
        _, w1s_hbm, _, w2s_hbm = refs or (None, self.w1s_hbm, None, self.w2s_hbm)
        return [
            pltpu.make_async_copy(
                w1s_hbm.at[e], s["es1"].at[buf], s["esem"].at[buf, 0]
            ),
            pltpu.make_async_copy(
                w2s_hbm.at[e], s["es2"].at[buf], s["esem"].at[buf, 1]
            ),
        ]

    def _expert_buf(self, i):
        return lax.rem(self._entry(i)[0], 2)

    def issue_item(self, i, slot, refs=None, pre=None):
        """Start item i's DMAs (and its expert's scales on the expert's first
        slice) from `refs` (default: this block's operands). `pre` is
        item_dma_args(i), computed once outside a per-layer dispatch."""
        e_loc, s_loc, starts, buf = pre if pre is not None else self.item_dma_args(i)
        if self.cfg.layout != "stock":
            for d in self.copies(e_loc, s_loc, slot, refs):
                d.start()
            return
        for d in self._stock_copies(e_loc, s_loc, slot, refs, starts):
            d.start()

        @pl.when(self._entry(i)[2])
        def _():
            for d in self._scale_copies(e_loc, buf, refs):
                d.start()

    def item_dma_args(self, i):
        _, e_loc, s_loc = self.item(i)
        return e_loc, s_loc, self.piece_starts(s_loc), self._expert_buf(i)

    def start_item(self, i, slot):
        self.issue_item(i, slot)

    # -- item body -------------------------------------------------------------
    # Every wait of an item is issued before its first dot: a DMA wait is a
    # scheduling barrier, and one inside the dot chain exposes the MXU drain.
    def _wait(self, desc):
        desc.wait()

    def _groups(self):
        """(G, K-blocks, w2 column chunks) per rolled iteration: the item body
        is a fori over G groups (code ~1/G of the unrolled body)."""
        cfg = self.cfg
        nb, nc = cfg.hidden // cfg.qb, cfg.hidden // cfg.w2_cols
        G = max(g for g in range(1, cfg.item_groups + 1) if nb % g == 0 and nc % g == 0)
        return G, nb // G, nc // G

    def fused(
        self,
        front_slot,
        back_slot,
        mid_q=None,
        coef=None,
        back_valid=None,
        front_item=None,
        back_item=None,
        chunk_row=None,
        back_rows=None,
    ):
        """One rolled loop running the gate-up of the slice in `front_slot`
        (returns its f32 [M, 2S] accumulator) and/or the down projection of
        the slice in `back_slot` (acc += (sum over pieces of mid_q piece @ w2
        piece x its block scale) * coef), interleaved group by group so one
        item's MXU drain hides under the other's weight pushes. Per output
        column the f32 summation order is that of the unrolled body.
        Compact: the gate-up reads xg rows [chunk_row, chunk_row + CT); the
        down projection's rows t < KS are added into acc rows back_rows[t]
        here (as below, per row) and all go to ybuf for scatter()."""
        cfg, s = self.cfg, self.s
        S, qb, M, wc, w2c = cfg.S, cfg.qb, cfg.item_rows, cfg.w1_cols, cfg.w2_cols
        G, bpg, cpg = self._groups()
        bounds = np.cumsum((0,) + cfg.pieces)
        n_acc = 2 * S // wc

        def group(g, accs):
            accs = list(accs)
            if front_slot is not None:
                for bb in range(bpg):
                    b = g * bpg + bb
                    cols = pl.ds(pl.multiple_of(b * qb, qb), qb)
                    if cfg.compact:
                        xb = s["xg"][pl.ds(chunk_row, CT), cols]
                    else:
                        xb = s["xq"][:, cols]
                    rows = pl.ds(pl.multiple_of(b * (qb // PACK4), 8), qb // PACK4)
                    for c in range(n_acc):
                        cs = pl.ds(c * wc, wc)
                        wb = pltpu.bitcast(
                            s["w1_vm"][front_slot, rows, cs], FP4
                        ).astype(FP8)
                        part = lax.dot_general(xb, wb, DN, preferred_element_type=F32)
                        accs[c] = accs[c] + part * self.w1_scale(
                            front_slot, front_item, b, c
                        )
            if back_slot is not None:
                for cc in range(cpg):
                    cs = pl.ds(pl.multiple_of((g * cpg + cc) * w2c, 128), w2c)
                    wb = pltpu.bitcast(s["w2_vm"][back_slot, :, cs], FP4).astype(FP8)
                    y = None
                    for j in range(len(cfg.pieces)):
                        k0, k1 = int(bounds[j]), int(bounds[j + 1])
                        t = lax.dot_general(
                            mid_q[:, k0:k1], wb[k0:k1], DN, preferred_element_type=F32
                        )
                        t = t * self.w2_scale(back_slot, back_item, j, cs)
                        y = t if y is None else y + t
                    if cfg.compact:
                        s["ybuf"][:, cs] = y
                        # row r through its aligned 8-row tile (a dynamic row
                        # offset costs more than a sublane select)
                        sub = lax.broadcasted_iota(I32, (8, w2c), 0)
                        for t, r in enumerate(back_rows):
                            add = y[t : t + 1] * coef[t : t + 1]
                            add = jnp.where(back_valid, add, 0.0)
                            rt = pl.ds(pl.multiple_of((r >> 3) * 8, 8), 8)
                            tile = s["acc"][rt, cs]
                            s["acc"][rt, cs] = jnp.where(
                                sub == (r & 7), tile + add, tile
                            )
                        continue
                    contrib = y * coef
                    if back_valid is not None:  # stale-slot operands may be non-finite
                        contrib = jnp.where(back_valid, contrib, 0.0)
                    s["acc"][:, cs] = s["acc"][:, cs] + contrib
            return tuple(accs)

        init = tuple(
            jnp.zeros((M, wc), F32)
            for _ in range(n_acc if front_slot is not None else 0)
        )
        accs = lax.fori_loop(0, G, group, init) if G > 1 else group(0, init)
        if front_slot is None:
            return None
        return jnp.concatenate(accs, axis=1) if n_acc > 1 else accs[0]

    def gate_up(self, slot):
        return self.fused(slot, None)

    def down(self, slot, mid_q, coef):
        self.fused(None, slot, mid_q, coef)

    def wait_item(self, i, slot):
        for d in self.copies(0, 0, slot):
            self._wait(d)
        if self.cfg.layout == "stock":
            buf = self._expert_buf(i)

            @pl.when(self._entry(i)[2])
            def _():
                for d in self._scale_copies(0, buf):
                    self._wait(d)

    def w1_scale(self, slot, i, b, c):
        """f32 [1, w1_cols]: block b's scales for tile columns of chunk c."""
        cfg, s = self.cfg, self.s
        wc = cfg.w1_cols
        if cfg.layout != "stock":
            return s["w1s_vm"][slot, pl.ds(b, 1), pl.ds(c * wc, wc)]
        sl, buf = self.item(i)[2], self._expert_buf(i)
        parts = []
        for h in range(2):
            for j, p in enumerate(cfg.pieces):
                t0 = h * cfg.S + sum(cfg.pieces[:j])
                lo, hi = max(t0, c * wc), min(t0 + p, (c + 1) * wc)
                if lo < hi:
                    e0 = h * cfg.inter + cfg.n_slices * sum(cfg.pieces[:j]) + sl * p
                    col = pl.ds(
                        pl.multiple_of(e0 + (lo - t0), math.gcd(p, 128)), hi - lo
                    )
                    parts.append(s["es1"][buf, pl.ds(b, 1), col])
        return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)

    def w2_scale(self, slot, i, j, cs):
        """f32 [1, w2_cols]: the block scale of piece j for columns cs."""
        cfg, s = self.cfg, self.s
        if cfg.layout != "stock":
            return s["w2s_vm"][slot, j : j + 1, cs]
        sl, buf = self.item(i)[2], self._expert_buf(i)
        blk = (cfg.n_slices * sum(cfg.pieces[:j]) + sl * cfg.pieces[j]) // cfg.qb
        return s["es2"][buf, pl.ds(blk, 1), cs]

    def front(self, i, slot, back=None):
        """Item i's gate-up and intermediate -> (fp8 mid, coef [M, 128]); with
        back = (slot, mid_q, coef) the previous item's down projection runs in
        the same rolled loop."""
        cfg, s = self.cfg, self.s
        e = self.item(i)[0]
        iota_e = lax.broadcasted_iota(I32, (cfg.rows, cfg.n_experts), 1)
        wcol = jnp.sum(
            jnp.where(iota_e == e, s["dense"][...], 0.0), axis=1, keepdims=True
        )
        if back is None:
            acc = self.fused(slot, None, front_item=i)
        else:
            b_slot, mid_q, coef, valid, b_item = back
            acc = self.fused(
                slot, b_slot, mid_q, coef, valid, front_item=i, back_item=b_item
            )
        mid_q, mid_scale = intermediate(cfg, acc, s["xs"][:, 0:1])
        return mid_q, jnp.broadcast_to(mid_scale * wcol, s["coef"].shape)

    def front_compact(self, i, c, slot, back, back_rows):
        """front() on chunk c of item i's gathered rows (gather())."""
        cfg, s = self.cfg, self.s
        r0 = pl.multiple_of(c * CT, CT)
        b_slot, mid_q, coef, valid, b_item = back
        acc = self.fused(
            slot,
            b_slot,
            mid_q,
            coef,
            valid,
            front_item=i,
            back_item=b_item,
            chunk_row=r0,
            back_rows=back_rows,
        )
        mid_q, mid_scale = intermediate(cfg, acc, s["gsc"][pl.ds(r0, CT), 0:1])
        wcol = s["gw"][pl.ds(r0, CT), 0:1]
        return mid_q, jnp.broadcast_to(mid_scale * wcol, s["coef"].shape)

    def stash(self, mid_coef):
        self.s["mid"][...], self.s["coef"][...] = mid_coef

    def back(self, slot, mid_q, coef):
        self.down(slot, mid_q, coef)

    def run(self, n_items, acc_init=None, ids=None):
        """Software-pipelined over items with ONE copy of each code site (the
        weight-DMA issue is replicated per layer where weights are
        layer-indexed, and code size is time inside the step kernel): step i
        issues the items whose slots are free, then runs item i's gate-up and
        item i-1's down projection in one rolled loop. Step 0's down and step
        n's gate-up run on stale operands and contribute nothing."""
        cfg, s = self.cfg, self.s
        NB = cfg.nbuf
        if acc_init is None:
            s["acc"][...] = jnp.zeros(s["acc"].shape, F32)
        elif callable(acc_init) and cfg.compact:
            s["acc"][0 : cfg.rows] = acc_init()
        elif callable(acc_init):
            s["acc"][...] = acc_init()
        # else: the caller has written s["acc"] already
        s["mid"][...] = jnp.zeros(s["mid"].shape, s["mid"].dtype)
        s["coef"][...] = jnp.zeros(s["coef"].shape, F32)
        if cfg.compact:
            self._run_compact(n_items, ids)
            return

        def step(i, issued):
            # slots of items i-1 (down) and i (gate-up) are busy: issue up to i+NB-2
            target = jnp.minimum(n_items, i + NB - 1)

            def issue(k, c):
                self.start_item(k, lax.rem(k, NB))
                return c

            lax.fori_loop(issued, target, issue, 0)
            cur = jnp.minimum(i, n_items - 1)
            slot, prev = lax.rem(i, NB), lax.rem(i + NB - 1, NB)

            @pl.when(i < n_items)
            def _():
                self.wait_item(i, slot)

            mid_q, coef = s["mid"][...], s["coef"][:, 0:1]
            back = (prev, mid_q, coef, i > 0, jnp.maximum(i - 1, 0))
            self.stash(self.front(cur, slot, back=back))
            return jnp.maximum(issued, target)

        @pl.when(n_items > 0)
        def _():
            lax.fori_loop(0, n_items + 1, step, jnp.int32(0))

    def _run_compact(self, n_items, ids):
        """run() over (item, chunk) steps: an item's chunks run back to back on
        its slot (one weight load), the first waits its DMAs and, on an
        entry's first item, gathers the entry's rows; step k's down
        projection is step k-1's, scattered into acc after the loop body.
        The first items' weights stream in while the row lists are built."""
        cfg, s = self.cfg, self.s
        NB = cfg.nbuf
        pre = jnp.minimum(n_items, NB - 1)

        def issue(q, cc):
            self.start_item(q, lax.rem(q, NB))
            return cc

        lax.fori_loop(0, pre, issue, 0)
        self.build_rows(ids)
        self._count_steps(self._n_hit)

        def step(k, carry):
            issued, i, c, pi, pc = carry
            target = jnp.minimum(n_items, i + NB - 1)
            lax.fori_loop(issued, target, issue, 0)
            cur = jnp.minimum(i, n_items - 1)
            slot, prev = lax.rem(i, NB), lax.rem(pi, NB)
            live = k < s["nsub"][0]
            m, _, first = self._entry(cur)

            @pl.when(live & (c == 0))
            def _():
                self.wait_item(i, slot)

            @pl.when(live & (c == 0) & first)
            def _():
                self.gather(m)

            mid_q, coef = s["mid"][...], s["coef"][:, 0:1]
            back = (prev, mid_q, coef, k > 0, pi)
            n_back, back_rows = self.scatter_rows(pi, pc, k > 0)
            mid_coef = self.front_compact(cur, c, slot, back, back_rows)
            self.scatter(pi, pc, k > 0, n_back)
            self.stash(mid_coef)
            more = c + 1 < self._n_chunks(m)
            nxt_i, nxt_c = jnp.where(more, i, i + 1), jnp.where(more, c + 1, 0)
            return jnp.maximum(issued, target), nxt_i, nxt_c, i, c

        zero = jnp.int32(0)

        @pl.when(n_items > 0)
        def _():
            lax.fori_loop(0, s["nsub"][0] + 1, step, (pre,) + (zero,) * 4)


def intermediate(cfg, acc, xs):
    """gate|up accumulator of one slice -> (fp8 mid, f32 row scale). Mirrors
    kernel._intermediate_rows with the clamped activation; the fp8 row scale
    is the max over this slice."""
    S, lim = cfg.S, cfg.swiglu_limit
    gate = jnp.minimum(acc[:, :S] * xs, lim)
    up = jnp.clip(acc[:, S:] * xs, -lim, lim)
    act = (jax.nn.silu(gate) * up).astype(BF16)
    amax = jnp.max(jnp.abs(act), axis=-1, keepdims=True)
    scale = amax.astype(F32) / rt.FP8_MAX
    usable = (scale > 0) & (scale < jnp.inf)
    scale = jnp.where(usable, scale, 0.0)
    sinv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    mid = jnp.where(usable, act * sinv.astype(BF16), jnp.zeros_like(act))
    return mid.astype(FP8), scale


def quantize_fp8_rows_f32(x):
    """Per-row fp8 quantization in f32 (scale = amax / 448, zero -> 1), the
    kernels/quantized_matmul/util.quantize_block rule without bf16
    intermediates: a bf16 round trip is one XLA may elide and Mosaic does not,
    so keeping it would make the kernel and any XLA reference disagree."""
    x = x.astype(F32)
    scale = jnp.max(jnp.abs(x), axis=-1, keepdims=True) / rt.FP8_MAX
    scale = jnp.where(scale == 0, 1.0, scale)
    return jnp.clip(x / scale, -rt.FP8_MAX, rt.FP8_MAX).astype(FP8), scale


def shared_expert(cfg, x, w13, s13, w2, s2):
    """This rank's shared-expert partial, f32 [M, H]: fp8 w13 and w2 with
    per-channel scales and per-row fp8 activations (the xla_quantized_matmul
    scheme, kept in f32 between the dots), clamped SiLU between; w2's input
    row scale is over this rank's slice, as under stock TP."""
    S, lim = cfg.Ssh, cfg.swiglu_limit
    xq, xs = quantize_fp8_rows_f32(x)
    h = lax.dot_general(xq, w13, DN, preferred_element_type=F32) * xs * s13
    a = jax.nn.silu(jnp.minimum(h[:, :S], lim)) * jnp.clip(h[:, S:], -lim, lim)
    aq, as_ = quantize_fp8_rows_f32(a)
    return lax.dot_general(aq, w2, DN, preferred_element_type=F32) * as_ * s2


# ----------------------------------------------------------------------------
# Standalone layer: router + experts + all-reduce in one pallas_call per rank.


def build_moe_layer(cfg: MoEConfig, *, interpret=None):
    """One rank's pallas_call (use inside shard_map):
    (meta i32 [1, 2] = (rank, live rows), x bf16 [M, H], gate_w bf16 [H, E],
     bias f32 [1, E] or hash ids i32 [M, 128], w1p, w1sp, w2p, w2sp local blocks)
    -> (out bf16 [M, H], ids i32 [M, 128], weights f32 [M, 128])."""
    M, H, E = cfg.rows, cfg.hidden, cfg.n_experts
    blk_names = list(MoEBlock.scratch(cfg))
    ar_names = list(cfg.ar.scratch())
    extra = {"gate_vm": pltpu.VMEM((H, E), BF16), "gsem": pltpu.SemaphoreType.DMA((3,))}
    if cfg.shared:
        extra["sw13_vm"] = pltpu.VMEM((H, 2 * cfg.Ssh), FP8)
        extra["sw2_vm"] = pltpu.VMEM((cfg.Ssh, H), FP8)
    extra_names = list(extra)
    scratch = {**MoEBlock.scratch(cfg), **cfg.ar.scratch(), **extra}
    names = blk_names + ar_names + extra_names

    def kernel(
        meta,
        x_ref,
        gate_hbm,
        route_ref,
        w1,
        w1s,
        w2,
        w2s,
        *refs,
    ):
        shared_refs, (out_ref, ids_ref, wts_ref), scr_refs = (
            refs[: 4 * cfg.shared],
            refs[4 * cfg.shared : 4 * cfg.shared + 3],
            refs[4 * cfg.shared + 3 :],
        )
        s = dict(zip(names, scr_refs))
        me, live = meta[0, 0], meta[0, 1]
        gcp = pltpu.make_async_copy(gate_hbm, s["gate_vm"], s["gsem"].at[0])
        gcp.start()
        acc_init = None
        if cfg.shared:  # weights stream in under the barrier, gate and routing
            sw13_hbm, s13_ref, sw2_hbm, s2_ref = shared_refs
            scp = [
                pltpu.make_async_copy(sw13_hbm.at[0], s["sw13_vm"], s["gsem"].at[1]),
                pltpu.make_async_copy(sw2_hbm.at[0], s["sw2_vm"], s["gsem"].at[2]),
            ]
            for c in scp:
                c.start()

            def acc_init():
                for c in scp:
                    c.wait()
                return shared_expert(
                    cfg, x, s["sw13_vm"][...], s13_ref[0], s["sw2_vm"][...], s2_ref[0]
                )

        barrier(cfg.R, me)
        if cfg.layout == "stock" and w1.dtype == FP4:
            # the served fp4 refs viewed as packed u32 words along K (private
            # ref bitcast, as the stock kernel; Pallas interpret lacks it, so
            # tests pass pack_fp4_rows of the same arrays, the same bits)
            w1, w2 = w1.bitcast(U32), w2.bitcast(U32)
        blk = MoEBlock(cfg, s, me, (w1, w1s, w2, w2s))
        x = x_ref[...]
        blk.quantize_rows(x)
        gcp.wait()
        logits = lax.dot_general(x, s["gate_vm"][...], DN, preferred_element_type=F32)
        if cfg.hash_routing:
            hids = [route_ref[:, k : k + 1] for k in range(cfg.topk)]
            ids, ws, hit = blk.route(logits, live, hash_ids=hids)
        else:
            ids, ws, hit = blk.route(logits, live, bias=route_ref[...])
        lane = lax.broadcasted_iota(I32, (M, 128), 1)
        ids_out = jnp.zeros((M, 128), I32)
        w_out = jnp.zeros((M, 128), F32)
        for k in range(cfg.topk):
            ids_out = jnp.where(lane == k, ids[k], ids_out)
            w_out = jnp.where(lane == k, ws[k], w_out)
        ids_ref[...] = ids_out
        wts_ref[...] = w_out
        blk.run(blk.build_worklist(hit), acc_init, ids=ids if cfg.compact else None)
        ar = AllReduce(cfg.ar, s, me)
        tot = ar(s["acc"][0 : cfg.ar_rows], 0)
        out_ref[0 : cfg.ar_rows] = tot.astype(BF16)
        if cfg.ar_rows < M:
            out_ref[cfg.ar_rows :] = jnp.zeros((M - cfg.ar_rows, H), BF16)
        ar.wait_sends(0)

    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    sm = pl.BlockSpec(memory_space=pltpu.SMEM)
    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((M, H), BF16),
            jax.ShapeDtypeStruct((M, 128), I32),
            jax.ShapeDtypeStruct((M, 128), F32),
        ),
        in_specs=[sm, vm, hbm, vm, hbm, hbm, hbm, hbm]
        + [hbm, vm, hbm, vm] * cfg.shared,
        out_specs=(vm, vm, vm),
        scratch_shapes=[scratch[k] for k in names],
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            vmem_limit_bytes=62 * 2**20,
            disable_bounds_checks=not cfg.bounds_checks,
        ),
        interpret=interpret if interpret is not None else False,
        name=f"dsv4_moe_{cfg.mode}",
    )
    return call


def moe_layer_call(mesh, cfg: MoEConfig, *, interpret=None):
    """jit(shard_map(build_moe_layer)): meta i32 [R, 2] sharded, x / gate_w /
    bias (or hash ids) replicated, weights sharded by weight_spec, then with
    cfg.shared the shared-expert slices sw13 fp8 [R, H, 2Ssh], s13 f32
    [R, 1, 2Ssh], sw2 fp8 [R, Ssh, H], s2 f32 [R, 1, H] sharded on axis 0 ->
    (out, ids, weights) stacked per shard ([R * M, ...]); every shard's block
    is that rank's copy (identical across ranks)."""
    (ax,) = mesh.axis_names
    ws = weight_spec(cfg.mode, ax)
    return jax.jit(
        jax.shard_map(
            build_moe_layer(cfg, interpret=interpret),
            mesh=mesh,
            in_specs=(P(ax), P(), P(), P(), ws, ws, ws, ws)
            + (P(ax),) * (4 * cfg.shared),
            out_specs=(P(ax), P(ax), P(ax)),
            check_vma=False,
        )
    )
