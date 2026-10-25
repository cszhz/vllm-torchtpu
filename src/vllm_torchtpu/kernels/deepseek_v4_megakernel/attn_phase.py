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
"""`step_phases.owner_attention_phase`: one layer of attention for the rank's
own requests inside the whole-step kernel.

Per active request (sequentially; an owner holds one or two at C <= 8):
  1. KV writes. SWA rows; compressor states (projection + ape) and the window
     reads of every compressed row that closes in this window (states of the
     window's in-flight rows come from VMEM, so reads and writes are issued
     together); CSA also the indexer compressor. Records are computed and
     written after one wait; all writes are waited before step 2.
  2. CSA: indexer q RoPE + per-row e4m3 quantization (as `rope_quant`), score
     scan over the request's seq-along-lane key pages, `tc_topk.topk_group`.
  3. SWA window pass, then HCA compressed blocks or the CSA top-k rows: every
     NoPE / RoPE page streamed, each query row's selected rows copied out of
     VMEM into a per-row staging granule, attended when full (no union).
  4. `out[r]`.

Per-layer caches are `LayerRefs`; every DMA start site on them dispatches on
the traced layer with a balanced tree over the layers of the static kind.

StepGeometry.attn_pipeline (HCA / CSA layers): request r + 1's SWA window, its
first HCA block and its first indexer scan block are fetched while request r
computes (`_Phase.prefetch`), into buffer slots and onto DMA semaphores of
r + 1's parity; request r + 1 re-reads the pages of that prefetch that its own
compressed-row writes changed (`_Phase.refetch`).
"""

import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_core as core
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_csa
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_group
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_layout as L
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kv_write as kw
from vllm_torchtpu.kernels.deepseek_v4_megakernel import tc_topk
from vllm_torchtpu.kernels.deepseek_v4_megakernel import vmem_alias

SM_SCALE = 512 ** -0.5
IDX_SOFTMAX_SCALE = 128 ** -0.5
IDX_HEAD_SCALE = 64 ** -0.5
WINDOW, SWA_PAGE = 128, 128
HCA_RATIO, HCA_PAGE = 128, 8  # compressed rows per HCA page
CSA_RATIO, CSA_PAGE = 4, 256
SBS = {"hca": 32, "csa": 16, "idx": 16}  # state block sizes (config.state_block_size)
TOPK = 1024
HCA_BLK = 512  # compressed rows per HCA block
IDX_BLK = 8  # indexer pages per scan block (16 independent head-sum chains)
CSA_BLK = 16  # NoPE / RoPE pages per stream block (1M: 16 beats 8 by 16%, 128k equal)
CSA_SLOTS = 2  # union stream ring depth: CSA_SLOTS - 1 blocks in flight ahead of the one consumed
assert -(-(1 << 20) // CSA_RATIO // (CSA_BLK * CSA_PAGE)) <= 128  # bend_s lanes at 1M
GRAN = 128  # staged rows per attention granule (per query row)
MAX_BOUNDARIES = 2  # compressed rows a 7-row window can close (ratio 4)
DEBUG_SCORES = DEBUG_LENS = None
CSA_MODE = "union"  # "union": the rows' top-k union, once; "rows": per-row staging
UGRAN = 512  # union rows per attention granule (union mode)
U_MAX = 8 * TOPK
ROUND_MODE = "rows"  # "rows": 7 per-row updates a round; "blockdiag": one full-width update
# Benchmark attribution only: stages to skip ("all", "kvw", "scan", "topk", "rows", "hca",
# "attend", "copy", "stream").
ABLATE = frozenset()
TRANSPOSE_KEYS = False  # explicit key transposes for PV in the HCA / union passes (A/B)
TABLE_REUSE = True  # tests only: False restages the block tables at every layer
DEBUG_TOPK = None  # test harness only: a VMEM ref [B, 8, TOPK] receiving the top-k lists
# Positive control of the store tests only: "two_runs" writes both boundaries'
# copies of a shared RoPE row / SAL lane tile, neither merged (lost records).
STORE_FAULT = ""
# Positive controls of the pipeline tests only: "same_slot" prefetches request
# r + 1 into the slots (and onto the semaphores) of request r's parity;
# "no_refetch" keeps the prefetched pages a request's own stores changed.
PF_FAULT = ""


def round_up(a, b):
    return -(-a // b) * b


@dataclasses.dataclass(frozen=True)
class Dims:
    q: int
    B: int
    RP: int
    max_ctx: int
    kinds: tuple[str, ...] = ("csa", "hca")  # layer kinds present
    heads: int = 128  # attended query heads per query row (StepGeometry.attn_heads)
    strict_causal: bool = False  # StepGeometry.strict_causal
    tp: int = 1  # ranks splitting each request's scan + top-k (attn_tp when attn_split)
    pipeline: bool = False  # StepGeometry.attn_pipeline

    @property
    def pf(self):  # prefetch of the next request (HCA / CSA layers)
        return self.pipeline and self.compressed

    @property
    def compressed(self):  # any HCA / CSA layer: their buffers are needed
        return "csa" in self.kinds or "hca" in self.kinds

    @property
    def n_blocks(self):  # NoPE page blocks of the longest request
        return -(-self.max_ctx // CSA_RATIO // (CSA_BLK * CSA_PAGE))

    @property
    def n_cols(self):  # this rank's indexer score columns (tc_topk needs multiples of 16384)
        if self.tp == 1:
            return round_up(self.max_ctx // CSA_RATIO, tc_topk.N_ALIGN)
        blocks = -(-(-(-self.max_ctx // CSA_RATIO) // CSA_PAGE) // IDX_BLK)
        return round_up(-(-blocks // self.tp) * IDX_BLK * CSA_PAGE, tc_topk.N_ALIGN)


assert 2 * HCA_BLK <= CSA_BLK * CSA_PAGE  # HCA blocks reuse the NoPE ring
# Pipeline, HCA layers: nope_buf[p, PF_HCA_ROW:] (unused by the HCA ring) holds
# the first HCA block of the next request of parity p.
PF_HCA_ROW = 2 * HCA_BLK
assert PF_HCA_ROW + 2 * HCA_BLK <= CSA_BLK * CSA_PAGE
PF_IBUF = 2  # pipeline: ibuf slots 2 + p hold the first scan block of parity p


def compress_config(d: Dims, mode: str) -> kw.CompressConfig:
    return kw.CompressConfig(
        mode=mode, max_req=d.B, state_block_size=SBS[mode], state_pages_per_req=1,
        kv_pages_per_req=1, kv_page_tokens={"hca": HCA_PAGE, "csa": CSA_PAGE, "idx": CSA_PAGE}[mode],
        state_lanes=256 if mode == "idx" else 128, q_len=d.q)


# Block tables are read from HBM (bt_flat grows with the context: the state
# tables have max_ctx / 16 columns a request) into SMEM, per layer kind and
# role, only as far as the phase reads them: KV tables up to max_ctx, state
# tables over the compressor window of this step's positions.
BT_ALIGN = 128  # an HBM int32 vector is tiled by 128: slices start on a tile


COS_SIN_FIELDS = ("cos_sin", "comp_cos_sin", "idx_cos_sin")


def pack_cos_sin(cs):
    """f32 [max_pos, 64] rotary table -> [ceil(max_pos / 2), 128], positions 2i and
    2i + 1 on row i. A 64-lane table gets a column-major device layout from XLA, so
    the kernel operand would be relaid out (copied) on every call."""
    cs = jnp.pad(cs, ((0, cs.shape[0] % 2), (0, 0)))
    return cs.reshape(cs.shape[0] // 2, 2 * cs.shape[1])


def pad_tables(bt_flat):
    """bt_flat padded to whole BT_ALIGN tiles (XLA side, before the kernel)."""
    return jnp.pad(bt_flat, (0, -bt_flat.shape[0] % BT_ALIGN))
TABLE_ROLES = {"swa": ("swa_pages",), "hca": ("kv", "state", "swa_pages"),
               "csa": ("kv", "kv_rope", "indexer_k", "state", "indexer_state", "swa_pages")}


def table_layout(d: Dims):
    """(kind, role) -> (region, base word, words per request, window mode | None);
    total words."""
    out, base, reg = {}, 0, 0
    for kind, roles in TABLE_ROLES.items():
        if kind not in d.kinds:
            continue
        for role in roles:
            mode = {"state": kind, "indexer_state": "idx"}.get(role)
            if role == "swa_pages":
                n = 3
            elif mode is not None:
                cfg = compress_config(d, mode)
                n = (cfg.window + d.q - 2) // cfg.state_block_size + 2
            elif kind == "hca":
                n = -(-(d.max_ctx // HCA_RATIO + 1) // HCA_PAGE)
            else:
                n = -(-(d.max_ctx // CSA_RATIO + 1) // CSA_PAGE)
            n = round_up(n + BT_ALIGN - 1, BT_ALIGN)  # copies start at a tile-aligned index
            out[(kind, role)] = (reg, base, n, mode)
            base += d.B * n
            reg += 1
    return out, base


def dims_of(geom) -> Dims:
    return Dims(q=geom.q, B=geom.B, RP=geom.RP, max_ctx=geom.max_ctx,
                kinds=tuple(sorted(set(geom.layer_kinds))), heads=geom.attn_heads,
                strict_causal=geom.strict_causal,
                tp=geom.attn_tp if geom.attn_split else 1, pipeline=geom.attn_pipeline)


def window_config(d: Dims, kind: str = "") -> core.WindowConfig:
    return core.WindowConfig(heads=d.heads, sm_scale=SM_SCALE, q_len=d.q, write_swa=True,
                             block_bidir=kind == "swa", force_transposed=True)


class Scratch(NamedTuple):
    # window pass (core.WindowScratch fields first)
    swa_buf: Any
    m: Any
    l: Any
    acc: Any
    wsems: Any
    wbuf: Any
    sink_raw: Any  # f32 [8, 128]
    sink_row: Any  # f32 [1, RP]
    # HCA blocks
    # compressor
    cstate: Any  # i32 [8, 16, 128]  staged main states (spt <= 16)
    istate: Any  # i32 [8, 2, 256]   staged indexer states
    ape: Any  # f32 [8, 2048]
    iape: Any  # f32 [8, 512]
    norm: Any  # f32 [8, 512]: row 0 main RMSNorm weight, row 1 lanes < 128 indexer
    cwin: Any  # i32 [1024, 128]: main windows (HCA 128 x 8 rows, CSA 2 x 8 x 16 rows)
    iwin: Any  # i32 [32, 256]: indexer windows (2 x 8 x 2 rows)
    cs: Any  # f32 [8, 128]: packed cos|sin rows (main boundaries 0, 1; indexer 2, 3)
    rec: Any  # i32 [8, 128]: records
    rope_rmw: Any  # i32 [8, 128]: RoPE rows of the CSA boundaries
    sal_rmw: Any  # i32 [2, 40, 128]: SAL lane tiles of the indexer boundaries
    # indexer + top-k
    qhm: Any  # f32 [512, 128]: e4m3-valued indexer q, row h * 8 + t
    wrow: Any  # f32 [8, 128]: folded indexer weights, row t, lane h
    wb: Any  # f32 [64, 8, 128]: weight of head h for row t, along lanes
    ibuf: Any  # i32 [2, IDX_BLK, 40, 256]
    scores: Any  # i32 [n_cols / 128 * 8, 128]: tc_topk group layout
    tk_v: Any  # i32 [8, 1024]
    tk_s: Any  # SMEM i32 [8, 1024]
    tk_scr: tuple  # tc_topk.scratch_shapes(n_cols)
    # CSA rows
    nope_buf: Any  # i32 [CSA_SLOTS, CSA_BLK * 256, 128]; also the HCA block ring (2 * HCA_BLK rows)
    rope_buf: Any  # i32 [CSA_SLOTS, CSA_BLK * 64, 128]
    stage_n: Any  # i32 [8, 2, GRAN, 128]: per query row, two halves of staged rows
    stage_r: Any  # i32 [8, 2, GRAN, 128]
    tk_t: Any  # i32 [TOPK + GRAN, 128]: the top-k lists transposed (row t in lane t)
    bend_v: Any  # i32 [8, 128]: per query row, list end of each page block
    bend_s: Any  # SMEM i32 [8, 128]
    un_v: Any  # i32 [U_MAX / 128, 128]: union entries (col << 8 | row bits), ascending, -1 pad
    un_s: Any  # SMEM i32 [U_MAX / 128, 128]
    un_len_v: Any  # i32 [8, 128]: union size (every element)
    un_len_s: Any  # SMEM i32 [8, 128]
    un_scr: tuple  # tc_topk.union_scratch_shapes(n_cols)
    un_n: Any  # i32 [UGRAN, 128] / [UGRAN, 128]: union staging (NoPE, RoPE rows)
    un_r: Any
    cnt: Any  # SMEM i32 [48]: per query row pending rows, A-half id, list pointer, half starts
    bts: Any  # SMEM i32 [table words]: the staged block tables (table_layout)
    bt_start: Any  # SMEM i32 [regions * B]: bt_flat index of each staged table's word 0
    bt_tag: Any  # SMEM i32 [regions]: bt_off of the staged tables (-1: none)
    # group split (attn_tp > 1; one vreg each otherwise), see attn_group
    g_scv: Any  # i32 [n_cols / 128 * 8, 128]: this rank's scores, kept for the second top-k
    g_vals: Any  # i32 [8, TOPK]: monotone keys of this rank's local top-k lists
    g_rv: Any  # i32 [tp, 8, TOPK]: every rank's g_vals
    g_vscr: tuple  # tc_topk.values_scratch_shapes(n_cols) (empty unless group split)
    g_ru: Any  # i32 [tp, U_MAX / 128 + 8, 128]: every rank's local union, then its size
    g_ru_s: Any  # SMEM i32 [tp, 8, 128]: the sizes of the tp local unions
    g_sem: Any  # REGULAR [2]: receive buffers free, per all_gather (keys, unions)
    g_dma: Any  # DMA [4]: sends, receives, per all_gather
    pf_sems: tuple  # (DMA [PF_NSEM],) with the pipeline, else ()
    sems: Any  # DMA [S_RING + CSA_SLOTS]


def scratch_shapes(d: Dims) -> list:
    wc = window_config(d)
    v = pltpu.VMEM
    i32, f32 = jnp.int32, jnp.float32
    tk = tc_topk.scratch_shapes(d.n_cols, TOPK)
    rows = CSA_MODE == "rows"  # per-row staging buffers (a vreg placeholder otherwise)
    stage = v((8, 2, GRAN, 128), i32) if rows else v((8, 128), i32)
    shapes = [
        *core.window_scratch_shapes(wc),
        v((8, 128), f32), v((1, d.RP), f32),
        v((8, 16, 128), i32), v((8, 2, 256), i32), v((8, 2048), f32), v((8, 512), f32),
        v((8, 512), f32), v((1024, 128), i32), v((32, 256), i32), v((8, 128), f32),
        v((8, 128), i32), v((8, 128), i32), v((2, 40, 128), i32),
        v((512, 128), f32), v((8, 128), f32), v((64, 8, 128), f32),
        v((PF_IBUF + 2 if d.pf else 2, IDX_BLK, 40, 256), i32), v((d.n_cols // 128 * 8, 128), i32),
        v((8, TOPK), i32), pltpu.SMEM((8, TOPK), i32), tuple(tk),
        v((CSA_SLOTS, CSA_BLK * 256, 128), i32), v((CSA_SLOTS, CSA_BLK * 64, 128), i32),
        stage, stage, v((TOPK + GRAN, 128), i32) if rows else v((8, 128), i32),
        v((8, 128), i32), pltpu.SMEM((8, 128), i32),
        v((U_MAX // 128, 128), i32), pltpu.SMEM((U_MAX // 128, 128), i32),
        v((8, 128), i32), pltpu.SMEM((8, 128), i32), tuple(tc_topk.union_scratch_shapes(d.n_cols, TOPK)),
        v((UGRAN, 128), i32), v((UGRAN, 128), i32),
        pltpu.SMEM((48,), i32),
        pltpu.SMEM((table_layout(d)[1],), i32), pltpu.SMEM((len(table_layout(d)[0]) * d.B,), i32),
        pltpu.SMEM((len(table_layout(d)[0]),), i32),
        *group_scratch_shapes(d),
        (pltpu.SemaphoreType.DMA((PF_NSEM,)),) if d.pf else (),
        pltpu.SemaphoreType.DMA((S_RING + CSA_SLOTS,)),
    ]
    if d.compressed:
        return shapes

    def small(sh):  # HCA / CSA buffer of an SWA-only geometry: one vreg
        return v((8, 128), sh.dtype) if str(getattr(sh, "memory_space", "")) == "vmem" else sh

    return [sh if f in SWA_FIELDS else (tuple(map(small, sh)) if isinstance(sh, tuple)
                                        else small(sh))
            for f, sh in zip(Scratch._fields, shapes, strict=True)]


def group_scratch_shapes(d: Dims) -> list:
    """The group split's buffers; empty tuples without it."""
    if d.tp == 1:
        return [()] * 8
    v, i32 = pltpu.VMEM, jnp.int32
    ur = U_MAX // 128 + 8
    return [v((d.n_cols // 128 * 8, 128), i32), v((8, TOPK), i32), v((d.tp, 8, TOPK), i32),
            tuple(tc_topk.values_scratch_shapes(d.n_cols)), v((d.tp, ur, 128), i32),
            pltpu.SMEM((d.tp, 8, 128), i32),
            pltpu.SemaphoreType.REGULAR((2,)), pltpu.SemaphoreType.DMA((4,))]


# Scratch an SWA-only geometry (the DSpark draft) uses.
SWA_FIELDS = ("swa_buf", "m", "l", "acc", "wsems", "wbuf", "sink_raw", "sink_row", "cnt", "bts",
              "bt_start", "bt_tag", "pf_sems", "sems")


def scratch_spec(d: Dims) -> list:
    """[(key, shape)]: key a Scratch field, or (field, i) for a member of a tuple field."""
    out = []
    for f, sh in zip(Scratch._fields, scratch_shapes(d), strict=True):
        out.extend([((f, i), x) for i, x in enumerate(sh)] if isinstance(sh, tuple) else [(f, sh)])
    return out


# Per-call buffers that may live in the skeleton's pool lease (StepGeometry.lease_bytes),
# largest first, taken greedily while they fit; each validated on device on a lease view
# (j-c1b81591, j-90c634fa). Excluded: the 40-row indexer page ring (wrong scores on a view)
# and the rows-mode staging (wrong results). The group split's score copy (same layout as
# "scores") comes last: it exists only with the split, and the unsplit plan stays as it was.
LEASE_ORDER = ("scores", "nope_buf", ("tk_scr", 3), "acc", ("un_scr", 4), ("un_scr", 0), "rope_buf",
               "g_scv")


def lease_plan(d: Dims, lease_bytes: int) -> dict:
    """key -> (first 128-byte row in the lease, rows) of the buffers placed in it."""
    spec = dict(scratch_spec(d))
    plan, row, cap = {}, 0, lease_bytes // 128
    for key in LEASE_ORDER:
        if key not in spec:  # a buffer of a mode this geometry does not have
            continue
        sh = spec[key]
        assert sh.shape[-1] % 128 == 0 and sh.shape[-2] % 8 == 0, key  # whole tiles
        n = vmem_alias.storage_size(sh.shape, sh.dtype) // 128
        if row + n <= cap:
            plan[key] = (row, n)
            row += n
    return plan


def flat_scratch_shapes(d: Dims, lease_bytes: int = 0) -> list:
    """Dedicated scratch: every buffer not placed in the lease."""
    plan = lease_plan(d, lease_bytes) if lease_bytes else {}
    return [sh for key, sh in scratch_spec(d) if key not in plan]


def unflatten_scratch(refs, d: Dims, lease=None, lease_bytes: int = 0) -> Scratch:
    """Scratch from the dedicated refs plus, when `lease` has a pool, views of
    it for the buffers lease_plan(d, lease_bytes) places there."""
    leased = lease is not None and lease.pool is not None
    plan = lease_plan(d, lease_bytes) if leased else {}
    if leased:
        assert sum(n for _, n in plan.values()) <= lease.rows, "lease smaller than planned"
    refs, vals = iter(refs), {}
    for key, sh in scratch_spec(d):
        if key in plan:
            r0, n = plan[key]
            vals[key] = vmem_alias.view(lease.pool.at[pl.ds(lease.row0 + r0, n)], sh.shape,
                                        sh.dtype)
        else:
            vals[key] = next(refs)
    assert next(refs, None) is None
    out = []
    for f in Scratch._fields:
        if f in vals:
            out.append(vals[f])
        else:
            out.append(tuple(vals[(f, i)] for i in range(sum(k[0] == f for k in vals
                                                            if isinstance(k, tuple)))))
    return Scratch(*out)


# sems
(S_KVW, S_WIN, S_REC, S_RMW, S_APE, S_SINK, S_IDX0, S_IDX1, S_CSA0, S_CSA1, S_HCA0, S_HCA1,
 S_BT, S_RING) = range(14)  # S_RING + slot: the union stream ring
# pf_sems: the HCA and indexer prefetches of parity p, the refetches
PF_HCA0, PF_IDX0, PF_FIX, PF_NSEM = 0, 2, 4, 5


def words(ref):
    """A served cache ref as int32 words: uint8 [N, R, 4, L] -> int32 [N, R, L]."""
    if ref.dtype == jnp.int32:
        return ref
    n, r, _, lanes = ref.shape
    return ref.bitcast(jnp.int32).reshape(n, r, lanes)


class Dispatch:
    """DMA starts on one role's per-layer arrays for a traced layer: a balanced
    tree of branches over the layers of the static kind."""

    def __init__(self, lrefs, layer, kinds, kind):
        self.items = [(lid, words(ref)) for ref, lid in zip(lrefs.refs, lrefs.layers)
                      if kinds[lid] == kind]
        assert self.items, kind
        self.layer = layer

    @property
    def first(self):
        return self.items[0][1]

    def run(self, fn):
        def rec(items):
            if len(items) == 1:
                fn(items[0][1])
                return
            mid = len(items) // 2
            split = items[mid][0]
            pl.when(self.layer < split)(lambda: rec(items[:mid]))
            pl.when(self.layer >= split)(lambda: rec(items[mid:]))

        rec(self.items)

    def read(self, src, dst, sem):  # src(ref) -> dst
        self.run(lambda ref: pltpu.make_async_copy(src(ref), dst, sem).start())

    def write(self, src, dst, sem):  # src -> dst(ref)
        self.run(lambda ref: pltpu.make_async_copy(src, dst(ref), sem).start())

    def loop(self, n, body):
        """body(ref, i) for i < n: one branch-tree walk for the whole loop."""
        self.run(lambda ref: lax.fori_loop(0, n, lambda i, c: (body(ref, i), c)[1], 0))


def wait_bytes(buf, sem):
    """Wait for copies into / out of `buf` totalling its size."""
    pltpu.make_async_copy(buf, buf, sem).wait()


class _Window(core.VerifyWindow):
    """VerifyWindow whose SWA cache is a per-layer operand list."""

    def __init__(self, *a, swa_dispatch, **k):
        super().__init__(*a, swa_hbm=None, **k)
        self.disp = swa_dispatch

    def swa_copy(self, hbm, vmem, sem, *, wait, write):
        if wait:  # a wait only needs the transfer shape
            ref = self.disp.first
            src, dst = (vmem, hbm(ref)) if write else (hbm(ref), vmem)
            pltpu.make_async_copy(src, dst, sem).wait()
        elif write:
            self.disp.write(vmem, hbm, sem)
        else:
            self.disp.read(hbm, vmem, sem)


class _Pages:
    """This layer's swa_window_pages table ([B * 3], role "swa_pages"), staged."""

    phase = None  # bound once the phase exists

    def __getitem__(self, i):
        return self.phase.bt("swa_pages", i // 3, i % 3)


def reset_table_stage(geom, scratch, leased: bool):
    """Mark no block tables staged: once per kernel, before the first phase
    call (staged tables are reused while a layer's bt_off matches)."""
    d = dims_of(geom)
    plan = lease_plan(d, geom.lease_bytes) if leased else {}
    tag = list(scratch)[[k for k, _ in scratch_spec(d) if k not in plan].index("bt_tag")]
    for i in range(tag.shape[0]):
        tag[i] = -1


def owner_attention_phase(geom, kind, layer, csa_ordinal, meta, inputs, caches, weights, out,
                          lease, scratch, me=None, stats=None):
    d = dims_of(geom)
    assert d.tp == 1 or me is not None, "attn_split needs the mesh index"
    S = unflatten_scratch(scratch, d, lease, geom.lease_bytes)
    kinds = geom.layer_kinds
    roles = {"swa": ("swa",), "hca": ("swa", "kv", "state"),
             "csa": ("swa", "kv", "state", "kv_rope", "indexer_k", "indexer_state")}[kind]
    disp = {role: Dispatch(getattr(caches, role), layer, kinds, kind) for role in roles}
    win = _Window(window_config(d, kind), seq_lens=meta.seq_lens,
                  swa_pt=(pages := _Pages()),
                  q_ref=inputs.q, new_kv_ref=inputs.kv, sink_ref=S.sink_row, out_ref=out,
                  scr=core.WindowScratch(S.swa_buf, S.m, S.l, S.acc, S.wsems, S.wbuf),
                  swa_dispatch=disp["swa"])
    ph = _Phase(geom, d, kind, layer, csa_ordinal, meta, inputs, weights, disp, win, S)
    ph.group = attn_group.Group(me, d.tp) if d.tp > 1 else None
    ph.stats = stats
    pages.phase = ph

    if "all" in ABLATE:  # attribution only: the phase does nothing
        return
    assert not (d.pf and ABLATE), "the pipeline waits every prefetch: no ablations"

    @pl.when(meta.num_req[0] > 0)
    def _():
        ph.prologue()
        lax.fori_loop(0, meta.num_req[0], lambda r, c: (ph.request(r), c)[1], 0)


class _Phase:
    def __init__(self, geom, d, kind, layer, csa_ordinal, meta, inputs, weights, disp, win, S):
        self.geom, self.d, self.kind, self.layer, self.ci = geom, d, kind, layer, csa_ordinal
        self.meta, self.inp, self.w, self.disp, self.win, self.S = meta, inputs, weights, disp, win, S
        self.ccfg = None if kind == "swa" else self._kcfg(kind)
        self.tables = table_layout(d)[0]

    # ------------------------------------------------------------------ helpers
    def bt(self, role, r, col):
        """Physical page of logical page `col` of request r in this layer's table."""
        m, S = self.meta, self.S
        w = m.bt_w[role][self.layer]
        reg, base, n, _ = self.tables[(self.kind, role)]
        g = m.bt_off[role][self.layer] + r * w + jnp.clip(col, 0, w - 1)
        return S.bts[base + r * n + jnp.clip(g - S.bt_start[reg * self.d.B + r], 0, n - 1)]

    def bt_base(self, role, r):
        """k with bts[k + col] = bt(role, r, col) for valid columns (hot loops)."""
        m, S = self.meta, self.S
        reg, base, n, _ = self.tables[(self.kind, role)]
        return (base + r * n + m.bt_off[role][self.layer] + r * m.bt_w[role][self.layer]
                - S.bt_start[reg * self.d.B + r])

    def stage_tables(self):
        """This layer's tables HBM -> SMEM, unless the same tables are staged
        (layers of a kind share their group's tables)."""
        m, S, B, lay = self.meta, self.S, self.d.B, self.layer
        total = m.bt_flat.shape[0]
        assert total % BT_ALIGN == 0, "pad bt_flat to whole tiles (pad_tables)"
        regs = [(role,) + self.tables[(self.kind, role)] for role in TABLE_ROLES[self.kind]]
        stale = not TABLE_REUSE
        for role, reg, *_ in regs:
            stale = stale | (S.bt_tag[reg] != m.bt_off[role][lay])

        @pl.when(stale)
        def _():
            copies = []
            for role, reg, base, n, mode in regs:
                off, w = m.bt_off[role][lay], m.bt_w[role][lay]
                nn = min(n, total)
                for r in range(B):
                    lo = 0
                    if mode is not None:  # the compressor window's first state block
                        cfg = compress_config(self.d, mode)
                        lo = jnp.clip(jnp.maximum(self.pos(r, 0) - cfg.window + 1, 0)
                                      // cfg.state_block_size, 0, jnp.maximum(w - 1, 0))
                    start = jnp.clip(off + r * w + lo, 0, total - nn) // BT_ALIGN * BT_ALIGN
                    S.bt_start[reg * B + r] = start
                    cp = pltpu.make_async_copy(m.bt_flat.at[pl.ds(pl.multiple_of(start, BT_ALIGN), nn)],
                                               S.bts.at[pl.ds(base + r * n, nn)], S.sems.at[S_BT])
                    cp.start()
                    copies.append(cp)
            for cp in copies:
                cp.wait()
            for role, reg, *_ in regs:
                S.bt_tag[reg] = m.bt_off[role][lay]

    def _kcfg(self, mode):
        return compress_config(self.d, mode)

    def copy_in(self, src, dst, sem):
        pltpu.make_async_copy(src, dst, self.S.sems.at[sem]).start()

    def prologue(self):
        S, w, lay = self.S, self.w, self.layer
        self.stage_tables()
        if self.d.pf and self.kind != "swa":
            self.prefetch(0, 0)
        self.copy_in(w.attn_sink.at[pl.ds(lay, 1)], S.sink_raw.at[pl.ds(0, 1)], S_SINK)
        if self.kind != "swa":
            self.copy_in(w.comp_norm.at[pl.ds(lay, 1)], S.norm.at[pl.ds(0, 1)], S_SINK)
        if self.kind == "csa":
            self.copy_in(w.idx_comp_norm.at[pl.ds(self.ci, 1)],
                         S.norm.at[pl.ds(1, 1), pl.ds(0, 128)], S_SINK)
        sem = S.sems.at[S_SINK]
        pltpu.make_async_copy(w.attn_sink.at[pl.ds(0, 1)], S.sink_raw.at[pl.ds(0, 1)], sem).wait()
        if self.kind != "swa":
            pltpu.make_async_copy(w.comp_norm.at[pl.ds(0, 1)], S.norm.at[pl.ds(0, 1)], sem).wait()
        if self.kind == "csa":
            pltpu.make_async_copy(w.idx_comp_norm.at[pl.ds(0, 1)],
                                  S.norm.at[pl.ds(1, 1), pl.ds(0, 128)], sem).wait()
        sink = S.sink_raw[0:1, :]
        H = self.d.heads  # lane t * H + h: head h's sink
        S.sink_row[...] = jnp.concatenate([sink[:, :H]] * (self.d.RP // H), axis=1)

    def pos(self, r, t):
        return self.meta.positions[r * self.d.q + t]

    # ------------------------------------------------------------------ request
    def request(self, r):
        if self.kind == "swa":
            self.draft_request(r)
            return
        if self.d.pf:
            self.pipelined_request(r)
            return
        if "kvw" not in ABLATE:
            self.kv_writes(r)
        if self.kind == "csa":
            self.indexer(r)
        win = self.win
        win.swa_dma(r, 0, wait=False)
        win.swa_dma(r, 0, wait=True)
        win.swa_step(r, 0)  # also starts the SWA row writes
        if self.kind == "hca":
            if "hca" not in ABLATE:
                self.hca(r)
            win.finalize(r)  # waits the SWA row writes
            return
        if self.stats is not None and CSA_MODE == "union":  # the union size, for serving logs
            self.stats[self.layer, r] = self.S.un_len_s[0, 0]
        if "rows" not in ABLATE:
            self.csa_union(r) if CSA_MODE == "union" else self.csa(r)
        win.finalize(r)

    def pipelined_request(self, r):
        """request() with request r + 1's first pages in flight during r."""
        par = self.par(r)

        @pl.when(r + 1 < self.meta.num_req[0])
        def _():
            self.prefetch(r + 1, self.par(r + 1))

        self.kv_writes(r)
        if self.kind == "csa":
            self.indexer(r)
        win = self.win
        win.swa_dma(r, par, wait=True)
        win.swa_step(r, par)  # also starts the SWA row writes
        if self.kind == "hca":
            self.hca(r)
        else:
            self.csa_union(r) if CSA_MODE == "union" else self.csa(r)
        win.finalize(r)  # waits the SWA row writes

    # ------------------------------------------------------------------ pipeline
    @staticmethod
    def par(r):
        """The slots and semaphores of request r's prefetch."""
        return 0 if PF_FAULT == "same_slot" else r % 2

    def prefetch(self, r, par, wait=False):
        """Request r's SWA window, first HCA block / first indexer scan block
        (this rank's share), into the slots of parity `par`."""
        self.win.swa_dma(r, par, wait=wait)
        if self.kind == "hca":
            self._hca_block_copies(r, 0, PF_IBUF + par, self.S.pf_sems[0].at[PF_HCA0 + par], wait)
        else:
            _, _, nblk, b_lo = self._idx_share(r)
            n_pg = jnp.where(nblk > 0, self._idx_pages(r, b_lo), 0)
            self._idx_block_copies(r, b_lo, n_pg, PF_IBUF + par,
                                   self.S.pf_sems[0].at[PF_IDX0 + par], wait)

    def refetch(self, r, mode, first_page, n_pg, dst):
        """Re-read the logical `mode` pages in [first_page, first_page + n_pg)
        that request r's compressed-row stores of this step wrote (waited by
        kv_writes) after the prefetch read them: dst(page, i) -> DMA (ref, i =
        page - first_page) on the PF_FIX semaphore, started and waited."""
        if PF_FAULT == "no_refetch":
            return
        cfg = self._kcfg(mode)
        per_page = HCA_PAGE if mode == "hca" else CSA_PAGE
        bl = self._boundaries(r, cfg)
        pages = [self.pos(r, jnp.minimum(t, self.d.q - 1)) // cfg.ratio // per_page
                 for t, _, _ in bl]
        role = "kv" if mode == "hca" else "indexer_k"
        dsp = self.disp[role]
        sem = self.S.pf_sems[0].at[PF_FIX]
        for j, (_, _, live) in enumerate(bl):
            i = pages[j] - first_page
            hit = live & (i >= 0) & (i < n_pg)
            if j == 1:  # both boundaries in one page: one re-read
                hit = hit & jnp.logical_not(bl[0][2] & (pages[0] == pages[1]))

            @pl.when(hit)
            def _(i=i, lp=pages[j]):
                page = self.bt(role, r, lp)
                dsp.run(lambda ref: dst(ref, page, i, sem).start())
                dst(dsp.first, page, i, sem).wait()

    def _idx_share(self, r):
        """(compressed rows, pages, this rank's scan blocks, its first block)."""
        n = self.meta.seq_lens[r] // CSA_RATIO
        npages = core.cdiv(n, CSA_PAGE)
        nblk = jnp.maximum(core.cdiv(npages, IDX_BLK), 1)
        b_lo = 0
        if self.group is not None:
            per = core.cdiv(nblk, self.d.tp)
            b_lo = jnp.minimum(self.group.jj * per, nblk)
            nblk = jnp.minimum(b_lo + per, nblk) - b_lo
        return n, npages, nblk, b_lo

    def _idx_pages(self, r, b):
        """Pages of scan block b of request r."""
        npages = core.cdiv(self.meta.seq_lens[r] // CSA_RATIO, CSA_PAGE)
        return jnp.clip(npages - b * IDX_BLK, 0, IDX_BLK)

    def _idx_block_copies(self, r, b, n_pg, slot, sem, wait):
        """The n_pg pages of scan block b -> ibuf[slot] (started, or waited)."""
        S = self.S
        if wait:
            lax.fori_loop(0, n_pg, lambda p, c: (wait_bytes(S.ibuf.at[slot, 0], sem), c)[1], 0)
            return

        def pg(ref, p):
            page = self.bt("indexer_k", r, b * IDX_BLK + p)
            pltpu.make_async_copy(ref.at[page, pl.ds(0, 40)], S.ibuf.at[slot, p], sem).start()
        self.disp["indexer_k"].loop(n_pg, pg)

    def _hca_view(self, v):
        """HCA block buffer v: the ring slots 0, 1 or the prefetch areas PF_IBUF + p."""
        row = pl.multiple_of((v // PF_IBUF) * PF_HCA_ROW, PF_HCA_ROW)
        return self.S.nope_buf.at[v % 2, pl.ds(row, 2 * HCA_BLK)]

    def _hca_block_copies(self, r, b, v, sem, wait):
        """The pages of HCA block b of request r -> buffer v (started, or waited)."""
        per_blk = HCA_BLK // HCA_PAGE
        npages = core.cdiv(self.meta.seq_lens[r] // HCA_RATIO, HCA_PAGE)
        n_pg = jnp.clip(npages - b * per_blk, 0, per_blk)
        buf = self._hca_view(v)
        if wait:
            lax.fori_loop(0, n_pg, lambda p, c: (wait_bytes(buf.at[pl.ds(0, 2 * HCA_PAGE)], sem),
                                                 c)[1], 0)
            return

        def pg(ref, p):
            page = self.bt("kv", r, b * per_blk + p)
            pltpu.make_async_copy(
                ref.at[page], buf.at[pl.ds(pl.multiple_of(p * 2 * HCA_PAGE, 16), 2 * HCA_PAGE)],
                sem).start()
        self.disp["kv"].loop(n_pg, pg)

    def draft_request(self, r):
        """Kind "swa" (DSpark draft): the context rows into the SWA pages, then
        the draft rows attend the window (bidirectional within their block)
        and are stored last, overwriting context rows at the same positions
        (stock: context insert, then the draft forward)."""
        win, cp = self.win, self.meta.ctx_positions
        c0 = cp[r * 8]
        n_ctx = sum((cp[r * 8 + c] >= 0).astype(jnp.int32) for c in range(8))
        for wait in (False, True):
            win.swa_write(r, wait=wait, rows=self.inp.ctx_kv, p0=c0, n=n_ctx)
        win.swa_dma(r, 0, wait=False)
        win.swa_dma(r, 0, wait=True)
        win.swa_step(r, 0)  # also starts the draft rows' writes
        win.finalize(r)  # waits them

    # ------------------------------------------------------------------ KV writes
    def kv_writes(self, r):
        S, d = self.S, self.d
        T = d.q
        jobs = [("main", self.ccfg, self.inp.comp, self.w.comp_ape, self.layer,
                 S.cstate.at[:, pl.ds(0, self.ccfg.spt)], S.ape, S.cwin, "state")]
        if self.kind == "csa":
            jobs.append(("idx", self._kcfg("idx"), self.inp.idx_comp, self.w.idx_comp_ape, self.ci,
                         S.istate, S.iape, S.iwin, "indexer_state"))
        # ape rows of the q positions, then the staged states
        # (one semaphore per job: a wait counts bytes, not copies)
        ape_sem = {"main": S_APE, "idx": S_SINK}
        for name, cfg, src, ape_hbm, lidx, stg, ape, cwin, srole in jobs:
            sw = cfg.state_width
            for t in range(T):
                self.copy_in(ape_hbm.at[lidx, pl.ds(self.pos(r, t) % cfg.ratio, 1), pl.ds(0, sw)],
                             ape.at[pl.ds(t, 1), pl.ds(0, sw)], ape_sem[name])
        for name, cfg, src, ape_hbm, lidx, stg, ape, cwin, srole in jobs:
            sw = cfg.state_width
            for t in range(T):
                pltpu.make_async_copy(ape_hbm.at[0, pl.ds(0, 1), pl.ds(0, sw)],
                                      ape.at[pl.ds(t, 1), pl.ds(0, sw)],
                                      S.sems.at[ape_sem[name]]).wait()
            x = src[r]  # f32 [8, 2 sw]: kv | gate
            st = jnp.concatenate([x[:, :sw], x[:, sw:2 * sw] + ape[:, 0:sw]], axis=1)
            stg[...] = lax.bitcast_convert_type(st, jnp.int32).reshape(8, cfg.spt, cfg.state_lanes)
        # starts: state rows, window reads, cos|sin rows, RMW reads
        nb = {}
        for name, cfg, src, ape_hbm, lidx, stg, ape, cwin, srole in jobs:
            nb[name] = self._starts(r, name, cfg, stg, cwin, srole)
        # waits (same structure)
        for name, cfg, src, ape_hbm, lidx, stg, ape, cwin, srole in jobs:
            self._starts(r, name, cfg, stg, cwin, srole, wait=True)
        for name, cfg, src, ape_hbm, lidx, stg, ape, cwin, srole in jobs:
            self._records(r, name, cfg, stg, cwin)
        for name, cfg, src, ape_hbm, lidx, stg, ape, cwin, srole in jobs:
            self._records(r, name, cfg, stg, cwin, wait=True)

    def _boundaries(self, r, cfg):
        """(t, slot) for each window row closing a compressed token, as a
        static list of candidates with traced predicates."""
        p0 = self.pos(r, 0)
        first = (cfg.ratio - 1 - p0 % cfg.ratio) % cfg.ratio  # first t with (pos+1) % ratio == 0
        out = []
        for slot in range(MAX_BOUNDARIES if cfg.ratio < self.d.q + 1 else 1):
            t = first + slot * cfg.ratio
            out.append((t, slot, t < self.d.q))
        return out

    def _starts(self, r, name, cfg, stg, cwin, srole, wait=False):
        S, T = self.S, self.d.q
        L_, spt, W = cfg.state_lanes, cfg.spt, cfg.window
        dsp = self.disp[srole]

        def state_dst(pos):
            page = self.bt(srole, r, pos // cfg.state_block_size)
            return lambda ref: ref.at[page, pl.ds((pos % cfg.state_block_size) * spt, spt)]

        def go_loop(n, src_of, dst_fn_of, sem, write):
            """n copies src_of(i) <-> dst_fn_of(i)(ref); a wait needs only the shape."""
            def one(ref, i):
                s_, t_ = src_of(i), dst_fn_of(i)(ref)
                s_, t_ = (s_, t_) if write else (t_, s_)
                c = pltpu.make_async_copy(s_, t_, S.sems.at[sem])
                c.wait() if wait else c.start()

            if wait:
                lax.fori_loop(0, n, lambda i, c: (one(dsp.first, i), c)[1], 0)
            else:
                dsp.loop(n, one)

        go_loop(T, lambda t: stg.at[t], lambda t: state_dst(self.pos(r, t)), S_KVW, True)
        cs_row = 0 if name == "main" else 2
        cs_hbm = self.w.comp_cos_sin if name == "main" else self.w.idx_cos_sin
        for t, slot, live in self._boundaries(r, cfg):
            @pl.when(live)
            def _(t=t, slot=slot):
                pos = self.pos(r, jnp.minimum(t, T - 1))
                start = pos - W + 1

                go_loop(W, lambda w: cwin.at[pl.ds((slot * W + w) * spt, spt)],
                        lambda w: state_dst(jnp.maximum(start + w, 0)), S_WIN, False)
                cpos = (pos // cfg.ratio) * cfg.ratio
                assert cfg.ratio % 2 == 0  # compressed positions are even: packed half 0
                csrc, cdst = cs_hbm.at[pl.ds(cpos // 2, 1)], S.cs.at[pl.ds(cs_row + slot, 1)]
                c = pltpu.make_async_copy(csrc, cdst, S.sems.at[S_WIN])
                c.wait() if wait else c.start()
                k = pos // cfg.ratio
                if cfg.mode == "csa":
                    page = self.bt("kv_rope", r, k // CSA_PAGE)
                    rsrc = lambda ref: ref.at[page, pl.ds((k % CSA_PAGE) // 4, 1)]
                    rd = S.rope_rmw.at[pl.ds(slot, 1)]
                    if wait:
                        pltpu.make_async_copy(rsrc(self.disp["kv_rope"].first), rd,
                                              S.sems.at[S_RMW]).wait()
                    else:
                        self.disp["kv_rope"].read(rsrc, rd, S.sems.at[S_RMW])
                if cfg.mode == "idx":
                    page = self.bt("indexer_k", r, k // CSA_PAGE)
                    tile = ((k % CSA_PAGE) // 128) * 128
                    ssrc = lambda ref: ref.at[page, pl.ds(0, 40), pl.ds(tile, 128)]
                    if wait:
                        pltpu.make_async_copy(ssrc(self.disp["indexer_k"].first),
                                              S.sal_rmw.at[slot], S.sems.at[S_RMW]).wait()
                    else:
                        self.disp["indexer_k"].read(ssrc, S.sal_rmw.at[slot], S.sems.at[S_RMW])

    def _records(self, r, name, cfg, stg, cwin, wait=False):
        S, T = self.S, self.d.q
        L_, spt, W, R = cfg.state_lanes, cfg.spt, cfg.window, cfg.ratio
        p0 = self.pos(r, 0)
        cs_row = 0 if name == "main" else 2
        bl = self._boundaries(r, cfg)
        # Two boundaries (ratio 4) close compressed rows k and k + 1; when they
        # share a RoPE row / SAL lane tile, slot 1's copy carries both and slot
        # 0 does not write it.
        share = None
        if len(bl) == 2:
            k0 = self.pos(r, jnp.minimum(bl[0][0], T - 1)) // R
            unit = 4 if cfg.mode == "csa" else 128
            share = bl[1][2] & (((k0 % CSA_PAGE) // unit) == (((k0 + 1) % CSA_PAGE) // unit))
        for t, slot, live in bl:
            @pl.when(live)
            def _(t=t, slot=slot):
                pos = self.pos(r, jnp.minimum(t, T - 1))
                k = pos // R
                skip_rmw = share if (share is not None and slot == 0) else None
                if STORE_FAULT == "two_runs":
                    skip_rmw = None
                if not wait:
                    start = pos - W + 1
                    f = lax.bitcast_convert_type(cwin[pl.ds(slot * W * spt, W * spt)], jnp.float32)
                    f = f.reshape(W, spt, L_)
                    # in-flight rows of the window: from the staged states
                    staged = lax.bitcast_convert_type(stg[...], jnp.float32)  # [8, spt, L]
                    for tt in range(T):
                        widx = self.pos(r, tt) - start  # window index of row tt
                        hit = (lax.broadcasted_iota(jnp.int32, (W, 1, 1), 0) == widx) & (tt <= t)
                        f = jnp.where(hit, staged[tt:tt + 1], f)
                    valid = start + lax.broadcasted_iota(jnp.int32, (W, 1), 0) >= 0
                    cs = S.cs[cs_row + slot: cs_row + slot + 1, 0:64]
                    norm = (S.norm[0:1, :] if name == "main" else S.norm[1:2, 0:128])
                    y = kw.compress_rows(cfg, f, valid, norm, cs[:, :32], cs[:, 32:])
                    self._store(r, cfg, k, y, slot, skip_rmw, wait=False)
                else:
                    self._store(r, cfg, k, None, slot, skip_rmw, wait=True)

    def _rmw_write(self, skip, fn):
        if skip is None:
            fn()
        else:
            pl.when(jnp.logical_not(skip))(fn)

    def _store(self, r, cfg, k, y, slot, skip_rmw, *, wait):
        S = self.S
        sem = S.sems.at[S_REC]
        lane = lax.broadcasted_iota(jnp.int32, (1, 128), 1)
        if cfg.mode == "hca":
            page = self.bt("kv", r, k // HCA_PAGE)
            dst = lambda ref: ref.at[page, pl.ds(2 * (k % HCA_PAGE), 2)]
            src = S.rec.at[pl.ds(0, 2)]
            if wait:
                pltpu.make_async_copy(src, dst(self.disp["kv"].first), sem).wait()
                return
            b = kw._bf16_bits(y)
            S.rec[0:2, :] = jnp.concatenate(
                [b[:, 256 * s: 256 * s + 128] | (b[:, 256 * s + 128: 256 * s + 256] << 16)
                 for s in range(2)], axis=0)
            self.disp["kv"].write(src, dst, sem)
            return
        if cfg.mode == "csa":
            page = self.bt("kv", r, k // CSA_PAGE)
            slot_in = k % CSA_PAGE
            ndst = lambda ref: ref.at[page, pl.ds(slot_in, 1)]
            nsrc = S.rec.at[pl.ds(2 + slot, 1)]
            rpage = self.bt("kv_rope", r, k // CSA_PAGE)
            rdst = lambda ref: ref.at[rpage, pl.ds(slot_in // 4, 1)]
            rsrc = S.rope_rmw.at[pl.ds(slot, 1)]
            if wait:
                pltpu.make_async_copy(nsrc, ndst(self.disp["kv"].first), sem).wait()
                self._rmw_write(skip_rmw, lambda: pltpu.make_async_copy(
                    rsrc, rdst(self.disp["kv_rope"].first), sem).wait())
                return
            nope, period, D = 448, 64, 512
            a = jnp.where(lax.broadcasted_iota(jnp.int32, (1, D), 1) < nope, jnp.abs(y), 0.0)
            folded = a[:, 0:period]
            for j in range(1, D // period):
                folded = jnp.maximum(folded, a[:, j * period:(j + 1) * period])
            scale = jnp.exp2(jnp.ceil(jnp.log2(jnp.maximum(folded, 1e-4) / kw.FP8_MAX)))
            q = kw._fp8_codes(y * (1.0 / jnp.concatenate([scale] * (D // period), axis=1)))
            sb = (lax.bitcast_convert_type(scale, jnp.int32) >> 23) & 0xFF
            b3 = jnp.where(lane < 64, q[:, 384:512], pltpu.roll(jnp.concatenate([sb, sb], 1), 64, 1))
            S.rec[2 + slot: 3 + slot, :] = (q[:, 0:128] | (q[:, 128:256] << 8)
                                            | (q[:, 256:384] << 16) | (b3 << 24))
            rb = kw._bf16_bits(y[:, nope:])
            wds = jnp.concatenate([rb[:, 0:32] | (rb[:, 32:64] << 16)] * 4, axis=1)
            # A second boundary in the same RoPE row merges into the first's copy.
            row = S.rope_rmw[slot: slot + 1, :]
            row = jnp.where(lane // 32 == slot_in % 4, wds, row)
            S.rope_rmw[slot: slot + 1, :] = row
            if slot == 1 and STORE_FAULT != "two_runs":
                k0 = k - 1
                same = (k0 % CSA_PAGE) // 4 == slot_in // 4
                @pl.when(same)
                def _():
                    S.rope_rmw[1:2, :] = jnp.where(lane // 32 == (k0 % CSA_PAGE) % 4,
                                                   S.rope_rmw[0:1, :], S.rope_rmw[1:2, :])
            self.disp["kv"].write(nsrc, ndst, sem)
            self._rmw_write(skip_rmw, lambda: self.disp["kv_rope"].write(rsrc, rdst, sem))
            return
        # idx: column slot of the seq-along-lane page, lane tile RMW
        page = self.bt("indexer_k", r, k // CSA_PAGE)
        tile = ((k % CSA_PAGE) // 128) * 128
        dst = lambda ref: ref.at[page, pl.ds(0, 40), pl.ds(tile, 128)]
        src = S.sal_rmw.at[slot]
        if wait:
            self._rmw_write(skip_rmw, lambda: pltpu.make_async_copy(
                src, dst(self.disp["indexer_k"].first), sem).wait())
            return
        amax = jnp.maximum(jnp.max(jnp.abs(y), axis=1, keepdims=True), 1e-4)
        scale = jnp.exp2(jnp.ceil(jnp.log2(amax / kw.FP8_MAX)))
        q = kw._fp8_codes(y * (1.0 / scale))
        sb = (lax.bitcast_convert_type(scale, jnp.int32) >> 23) & 0xFF
        qt = jnp.transpose(jnp.broadcast_to(q, (8, 128)))[:, 0:1]
        col = jnp.concatenate(
            [qt[4 * g:4 * g + 1] | (qt[4 * g + 1:4 * g + 2] << 8) | (qt[4 * g + 2:4 * g + 3] << 16)
             | (qt[4 * g + 3:4 * g + 4] << 24) for g in range(32)]
            + [sb] + [jnp.zeros((1, 1), jnp.int32)] * 7, axis=0)
        rowi = lax.broadcasted_iota(jnp.int32, (40, 1), 0)
        keep = (lane != (k % CSA_PAGE) % 128) | (rowi >= 33)
        S.sal_rmw[slot] = jnp.where(keep, S.sal_rmw[slot], col)
        if slot == 1 and STORE_FAULT != "two_runs":
            k0 = k - 1
            same = ((k0 % CSA_PAGE) // 128) == ((k % CSA_PAGE) // 128)
            @pl.when(same)
            def _():
                keep0 = (lane != (k0 % CSA_PAGE) % 128) | (rowi >= 33)
                S.sal_rmw[1] = jnp.where(keep0, S.sal_rmw[1], S.sal_rmw[0])
        self._rmw_write(skip_rmw, lambda: self.disp["indexer_k"].write(src, dst, sem))

    # ------------------------------------------------------------------ indexer
    def indexer(self, r):
        S, d = self.S, self.d
        T = d.q
        seq = self.meta.seq_lens[r]
        n = seq // CSA_RATIO
        # q: RoPE on lanes 64..127 at each row's position, then per (row, head) e4m3.
        for t in range(T):
            self.copy_in(self.w.idx_cos_sin.at[pl.ds(self.pos(r, t) // 2, 1)], S.cs.at[pl.ds(t, 1)],
                         S_APE)
        for t in range(T):
            pltpu.make_async_copy(self.w.idx_cos_sin.at[pl.ds(0, 1)], S.cs.at[pl.ds(t, 1)],
                                  S.sems.at[S_APE]).wait()
        lane = lax.broadcasted_iota(jnp.int32, (1, 128), 1)
        ch = lane - 64
        odd = (lane % 2) == 1
        S.qhm[...] = jnp.zeros((512, 128), jnp.float32)
        wrows = []
        for t in range(T):
            x = jnp.concatenate([self.inp.idx_q[r, pl.ds(t, 1), pl.ds(h * 128, 128)]
                                 for h in range(64)], axis=0).astype(jnp.float32)  # [64, 128]
            cs = S.cs[t:t + 1, :]
            cs = jnp.where(self.pos(r, t) % 2 == 1, pltpu.roll(cs, 64, 1), cs)  # this position's half
            cos = jnp.ones((1, 128), jnp.float32)
            sin = jnp.zeros((1, 128), jnp.float32)
            for i in range(32):
                hit = (ch >= 0) & (ch // 2 == i)
                cos = jnp.where(hit, cs[:, i:i + 1], cos)
                sin = jnp.where(hit, cs[:, 32 + i:33 + i], sin)
            partner = jnp.where(odd, pltpu.roll(x, 1, 1), pltpu.roll(x, 127, 1))
            rot = jnp.where(odd, partner, -partner)
            y = x * cos + rot * sin
            amax = jnp.max(jnp.abs(y), axis=1, keepdims=True)  # [64, 1]
            scale = amax / kw.FP8_MAX
            inv = jnp.where(amax == 0.0, 0.0, 1.0 / scale)
            q8 = (y * inv).astype(jnp.float8_e4m3fn).astype(jnp.float32)
            S.qhm[pl.ds(t, 64, stride=8), :] = q8
            srow = jnp.transpose(jnp.broadcast_to(scale, (64, 128)))[0:1, :]  # [1, 64]
            w = self.inp.idx_w[r, t:t + 1, 0:64]
            wrows.append(((w * IDX_SOFTMAX_SCALE) * IDX_HEAD_SCALE) * srow)
        wmat = jnp.concatenate(wrows + [jnp.zeros((8 - T, 64), jnp.float32)], axis=0)  # [8, 64]
        for h in range(64):
            S.wb[h] = jnp.broadcast_to(wmat[:, h:h + 1], (8, 128))
        # score scan: this rank's scan blocks [b_lo, b_lo + nblk) (all, unless group split)
        npages = core.cdiv(n, CSA_PAGE)
        nblk = jnp.maximum(core.cdiv(npages, IDX_BLK), 1)
        b_lo = 0
        if self.group is not None:
            per = core.cdiv(nblk, self.d.tp)
            b_lo = jnp.minimum(self.group.jj * per, nblk)
            nblk = jnp.minimum(b_lo + per, nblk) - b_lo
        dsp = self.disp["indexer_k"]
        t8 = lax.broadcasted_iota(jnp.int32, (8, 1), 0)
        row_len = jnp.where(t8 < T, attn_csa.visible_rows(n, self.pos(r, 0) + t8, d.strict_causal,
                                                          CSA_RATIO), 0)

        def start(i, slot):
            b = i if self.group is None else b_lo + i

            def pg(ref, p):
                page = self.bt("indexer_k", r, b * IDX_BLK + p)
                pltpu.make_async_copy(ref.at[page, pl.ds(0, 40)], S.ibuf.at[slot, p],
                                      S.sems.at[S_IDX0 + slot]).start()
            dsp.loop(jnp.clip(npages - b * IDX_BLK, 0, IDX_BLK), pg)

        def wait(i, slot):
            b = i if self.group is None else b_lo + i

            def pg(p, c):
                wait_bytes(S.ibuf.at[slot, 0], S.sems.at[S_IDX0 + slot])
                return c
            lax.fori_loop(0, jnp.clip(npages - b * IDX_BLK, 0, IDX_BLK), pg, 0)

        pf = self.d.pf
        if pf:  # block 0 was prefetched into slot PF_IBUF + par during request r - 1
            par = self.par(r)
            n_pg0 = jnp.where(nblk > 0, self._idx_pages(r, b_lo), 0)
            self._idx_block_copies(r, b_lo, n_pg0, PF_IBUF + par, S.pf_sems[0].at[PF_IDX0 + par],
                                   wait=True)
            self.refetch(r, "idx", b_lo * IDX_BLK, n_pg0,
                         lambda ref, page, i, sem: pltpu.make_async_copy(
                             ref.at[page, pl.ds(0, 40)], S.ibuf.at[PF_IBUF + par, i], sem))
        elif self.group is None:
            start(0, 0)
        else:
            @pl.when(nblk > 0)
            def _():
                start(0, 0)

        def blk(i, c):
            b = i if self.group is None else b_lo + i
            if pf:
                slot = jnp.where(i == 0, PF_IBUF + par, i % 2)
                pl.when(i > 0)(lambda: wait(i, i % 2))

                @pl.when(i + 1 < nblk)
                def _():
                    start(i + 1, (i + 1) % 2)
            else:
                slot = i % 2
                wait(i, slot)

                @pl.when(i + 1 < nblk)
                def _():
                    start(i + 1, 1 - slot)

            ws = [S.ibuf[slot, p] for p in range(IDX_BLK)]
            k_t = jnp.concatenate([pltpu.bitcast(x[0:32], jnp.float8_e4m3fn) for x in ws], axis=1)
            st = lax.dot_general(S.qhm[...].astype(jnp.float8_e4m3fn), k_t, core.NN,
                                 preferred_element_type=jnp.float32)
            nt = IDX_BLK * 2  # lane tiles: one weight vreg per head serves them all

            def wrow(h):
                return jnp.concatenate([S.wb[h]] * nt, axis=1)

            acc = jnp.maximum(st[0:8], 0.0) * wrow(0)
            for h in range(1, 64):
                acc = acc + jnp.maximum(st[8 * h:8 * h + 8], 0.0) * wrow(h)
            sc = jnp.concatenate([x[32:33] for x in ws], axis=1)
            acc = acc * L.e8m0_from_byte(sc & 0xFF)
            col0 = b * IDX_BLK * 256
            colv = col0 + lax.broadcasted_iota(jnp.int32, (1, IDX_BLK * 256), 1)
            acc = jnp.where(colv < row_len, acc, 0.0)  # see the fill below
            bits = lax.bitcast_convert_type(acc, jnp.int32)
            for j in range(IDX_BLK * 2):
                v = pl.multiple_of((i * IDX_BLK * 2 + j) * 8, 8)  # this rank's column order
                S.scores[pl.ds(v, 8), :] = bits[:, j * 128:(j + 1) * 128]
            return c

        if "scan" not in ABLATE:
            lax.fori_loop(0, nblk, blk, 0)
        elif self.group is None:
            wait(0, 0)
        else:
            @pl.when(nblk > 0)
            def _():
                wait(0, 0)
        # top-k
        lens = jnp.broadcast_to(row_len, (8, 128))
        # row lengths grow with t
        max_len = attn_csa.visible_rows(n, self.pos(r, T - 1), d.strict_causal, CSA_RATIO)
        if self.group is not None:  # this rank's columns [c_lo, c_lo + nblk * blk_cols)
            c_lo, c_n = b_lo * IDX_BLK * CSA_PAGE, nblk * IDX_BLK * CSA_PAGE
            row_len = jnp.clip(row_len - c_lo, 0, c_n)
            max_len = jnp.clip(max_len - c_lo, 0, c_n)
            lens = jnp.broadcast_to(row_len, (8, 128))
        nscan = jnp.maximum((max_len + tc_topk.SCAN_ALIGN - 1) // tc_topk.SCAN_ALIGN, 1)
        if self.group is not None:
            assert DEBUG_TOPK is None and DEBUG_SCORES is None and CSA_MODE == "union"
            if "topk" not in ABLATE:
                self.group_topk(lens, nscan, c_lo)
            return
        # columns past the row lengths (stale VMEM included) are never read by topk_group
        if DEBUG_SCORES is not None:
            DEBUG_SCORES[r] = S.scores[...]
            DEBUG_LENS[r] = jnp.broadcast_to(nscan, (8, 128))
        if "topk" not in ABLATE and CSA_MODE == "union":
            # per-row lists only for the test hooks (they cost ~7 us)
            tc_topk.topk_group(S.scores, lens, nscan, S.tk_v, *S.tk_scr, k=TOPK,
                               union=(S.un_v, S.un_len_v, *S.un_scr),
                               lists=DEBUG_TOPK is not None)
            for src, dst in ((S.un_v, S.un_s), (S.un_len_v, S.un_len_s)):
                cp = pltpu.make_async_copy(src, dst, S.sems.at[S_APE])
                cp.start()
                cp.wait()
        elif "topk" not in ABLATE:
            tc_topk.topk_group(S.scores, lens, nscan, S.tk_v, *S.tk_scr, k=TOPK)
        else:  # same amount of row work: 1024 spread rows (or all when shorter)
            col = lax.broadcasted_iota(jnp.int32, (8, TOPK), 1)
            stride = jnp.maximum(max_len // TOPK, 1)
            S.tk_v[...] = jnp.where(col * stride < row_len, col * stride, -1)
        if DEBUG_TOPK is not None:
            DEBUG_TOPK[r] = S.tk_v[...]
        if CSA_MODE == "rows":
            cp = pltpu.make_async_copy(S.tk_v, S.tk_s, S.sems.at[S_APE])
            cp.start()
            cp.wait()

    def group_topk(self, lens, nscan, c_lo):
        """The union of the request's top-k rows, this rank scoring columns
        [c_lo, ...) (attn_group)."""
        S, g, P = self.S, self.group, self.d.tp
        sync = lambda src, dst: (lambda c: (c.start(), c.wait()))(  # noqa: E731
            pltpu.make_async_copy(src, dst, S.sems.at[S_APE]))
        sync(S.scores, S.g_scv)  # the top-k consumes its buffer
        # 1. local top-k keys, all-gathered
        tc_topk.topk_group(S.scores, lens, nscan, S.tk_v, *S.tk_scr, k=TOPK,
                           values=(S.g_vals, *S.g_vscr))
        if "g_ag1" not in ABLATE:
            attn_group.all_gather(g, S.g_vals, S.g_rv, S.g_sem.at[0], S.g_dma.at[0],
                                  S.g_dma.at[1])  # semaphores of its own: waits count bytes
        # 2. the global cut; keys [8, P * TOPK], rank-major
        keys = jnp.concatenate([S.g_rv[p] for p in range(P)], axis=1)
        cut, n_gt, take_all = attn_group.global_cut(keys, TOPK)
        eq = [jnp.sum((keys[:, p * TOPK:(p + 1) * TOPK] == cut).astype(jnp.int32), axis=1,
                      keepdims=True) for p in range(P)]
        before = sum(jnp.where(p < g.jj, eq[p], 0) for p in range(P))
        mine = sum(jnp.where(p == g.jj, eq[p], 0) for p in range(P))
        allow = jnp.clip(TOPK - n_gt - before, 0, mine)  # ties this rank keeps, column order
        cut = jnp.where(take_all == 1, attn_group.INT_MIN, cut)
        # 3. this rank's members of the global top-k, as a union
        sync(S.g_scv, S.scores)
        tc_topk.topk_group(S.scores, lens, nscan, S.tk_v, *S.tk_scr, k=TOPK,
                           union=(S.un_v, S.un_len_v, *S.un_scr), lists=False,
                           thresh=(jnp.broadcast_to(cut, (8, 128)),
                                   jnp.broadcast_to(allow, (8, 128))))
        # 4. all-gather the local unions (global columns), concatenate in rank order
        uv = S.un_v[...]
        mine_ru = S.g_ru.at[g.jj]
        mine_ru[pl.ds(0, U_MAX // 128), :] = jnp.where(uv >= 0, uv + (c_lo << attn_csa.UNION_BITS),
                                                       -1)
        mine_ru[pl.ds(U_MAX // 128, 8), :] = S.un_len_v[...]
        if "g_ag2" not in ABLATE:
            attn_group.all_gather(g, mine_ru, S.g_ru, S.g_sem.at[1], S.g_dma.at[2],
                                  S.g_dma.at[3], local=False)
        UR = U_MAX // 128
        sync(S.g_ru.at[:, pl.ds(UR, 8)], S.g_ru_s.at[:, pl.ds(0, 8)])  # the P sizes
        lane = lax.broadcasted_iota(jnp.int32, (1, 128), 1)
        off = jnp.int32(0)
        for p in range(P):  # segment p -> union entries [off, off + n_p): rows rotated by off % 128
            n_p = S.g_ru_s[p, 0, 0]
            rot, row0 = off % 128, off // 128

            def cat(kk, c, p=p, n_p=n_p, rot=rot, row0=row0):
                x = pltpu.roll(S.g_ru[p, pl.ds(kk, 1), :], rot, 1)
                src = kk * 128 + (lane - rot) % 128  # entry of segment p in each lane of x
                keep = src < n_p
                a_, b_ = pl.ds(row0 + kk, 1), pl.ds(row0 + kk + 1, 1)
                S.un_v[a_, :] = jnp.where(keep & (lane >= rot), x, S.un_v[a_, :])
                S.un_v[b_, :] = jnp.where(keep & (lane < rot), x, S.un_v[b_, :])
                return c
            lax.fori_loop(0, 0 if "g_cat" in ABLATE else (n_p + 127) // 128, cat, 0)
            off = off + n_p
        S.un_len_s[0, 0] = off
        sync(S.un_v, S.un_s)

    # ------------------------------------------------------------------ HCA
    def hca(self, r):
        S, win = self.S, self.win
        n_c = self.meta.seq_lens[r] // HCA_RATIO
        npages = core.cdiv(n_c, HCA_PAGE)
        per_blk = HCA_BLK // HCA_PAGE
        nblk = core.cdiv(npages, per_blk)
        dsp = self.disp["kv"]

        def start(b, slot):
            def pg(ref, p):
                page = self.bt("kv", r, b * per_blk + p)
                pltpu.make_async_copy(
                    ref.at[page], S.nope_buf.at[slot, pl.ds(pl.multiple_of(p * 2 * HCA_PAGE, 16), 2 * HCA_PAGE)],
                    S.sems.at[S_HCA0 + slot]).start()
            dsp.loop(jnp.clip(npages - b * per_blk, 0, per_blk), pg)

        def wait(b, slot):
            def pg(p, c):
                wait_bytes(S.nope_buf.at[slot, pl.ds(0, 2 * HCA_PAGE)], S.sems.at[S_HCA0 + slot])
                return c
            lax.fori_loop(0, jnp.clip(npages - b * per_blk, 0, per_blk), pg, 0)

        pf = self.d.pf
        if pf:  # block 0 was prefetched into buffer PF_IBUF + par during request r - 1
            par = self.par(r)
            v0 = PF_IBUF + par
            self._hca_block_copies(r, 0, v0, S.pf_sems[0].at[PF_HCA0 + par], wait=True)
            self.refetch(r, "hca", 0, jnp.clip(npages, 0, per_blk),
                         lambda ref, page, i, sem: pltpu.make_async_copy(
                             ref.at[page], self._hca_view(v0).at[
                                 pl.ds(pl.multiple_of(i * 2 * HCA_PAGE, 16), 2 * HCA_PAGE)], sem))
        else:
            @pl.when(nblk > 0)
            def _():
                start(0, 0)

        def blk(b, c):
            if pf:
                pl.when(b > 0)(lambda: wait(b, b % 2))

                @pl.when(b + 1 < nblk)
                def _():
                    start(b + 1, (b + 1) % 2)

                buf = self._hca_view(jnp.where(b == 0, PF_IBUF + par, b % 2))
            else:
                slot = b % 2
                wait(b, slot)

                @pl.when(b + 1 < nblk)
                def _():
                    start(b + 1, 1 - slot)

                buf = S.nope_buf.at[slot]
            kv, kv_t = core.decode_bf16_tokens(buf, HCA_BLK, TRANSPOSE_KEYS)
            jg = b * HCA_BLK + lax.broadcasted_iota(jnp.int32, (HCA_BLK, 1), 0)
            live = jg < n_c
            kv = jnp.where(live, kv, 0)
            kv_t = None if kv_t is None else jnp.where(live.T, kv_t, 0)
            vis = jg < (win.qpos(r) + 1) // HCA_RATIO
            win.update(r, kv, kv_t, vis, first=False)
            return c

        lax.fori_loop(0, nblk, blk, 0)

    # ------------------------------------------------------------------ CSA
    def csa_union(self, r):
        """Attention over the union of the rows' top-k rows: every NoPE / RoPE
        page streamed once; each union row copied once into a granule of UGRAN
        rows, attended by all query rows at full width, row t seeing the
        entries carrying its bit."""
        S, win, T = self.S, self.win, self.d.q
        n = self.meta.seq_lens[r] // CSA_RATIO
        npages = core.cdiv(n, CSA_PAGE)
        nblk = core.cdiv(npages, CSA_BLK)
        blk_rows = CSA_BLK * CSA_PAGE
        G = UGRAN
        dn, dr = self.disp["kv"], self.disp["kv_rope"]
        U = S.un_len_s[0, 0]
        # union position where each page block ends (vector count over the entries)
        ent = S.un_v[...]
        idx = ent >> attn_csa.UNION_BITS
        pos = (lax.broadcasted_iota(jnp.int32, ent.shape, 0) * 128
               + lax.broadcasted_iota(jnp.int32, ent.shape, 1))
        live = pos < U
        lane = lax.broadcasted_iota(jnp.int32, (8, 128), 1)

        def block_end(bb, ends):
            cnt = jnp.sum((live & (idx < (bb + 1) * blk_rows)).astype(jnp.int32))
            return jnp.where(lane == bb, cnt, ends)

        S.bend_v[...] = lax.fori_loop(0, nblk, block_end, jnp.zeros((8, 128), jnp.int32))
        cp = pltpu.make_async_copy(S.bend_v, S.bend_s, S.sems.at[S_APE])
        cp.start()
        cp.wait()

        kn, kr = self.bt_base("kv", r), self.bt_base("kv_rope", r)
        sem = lambda slot: S.sems.at[S_RING + slot]  # noqa: E731

        def start(b, slot):
            if "stream" in ABLATE:
                return
            n_pg = jnp.clip(npages - b * CSA_BLK, 0, CSA_BLK)

            def nope(ref, p):
                page = S.bts[kn + b * CSA_BLK + p]
                pltpu.make_async_copy(ref.at[page], S.nope_buf.at[slot, pl.ds(
                    pl.multiple_of(p * CSA_PAGE, CSA_PAGE), CSA_PAGE)], sem(slot)).start()

            def rope(ref, p):
                rpage = S.bts[kr + b * CSA_BLK + p]
                pltpu.make_async_copy(ref.at[rpage], S.rope_buf.at[slot, pl.ds(
                    pl.multiple_of(p * (CSA_PAGE // 4), CSA_PAGE // 4), CSA_PAGE // 4)],
                    sem(slot)).start()
            dn.loop(n_pg, nope)
            dr.loop(n_pg, rope)

        def wait(b, slot):
            if "stream" in ABLATE:
                return
            n_pg = jnp.clip(npages - b * CSA_BLK, 0, CSA_BLK)

            @pl.when(n_pg == CSA_BLK)  # a wait counts bytes: one per buffer for a full block
            def _():
                wait_bytes(S.nope_buf.at[slot], sem(slot))
                wait_bytes(S.rope_buf.at[slot], sem(slot))

            @pl.when(n_pg < CSA_BLK)
            def _():
                def pg(p, c):
                    wait_bytes(S.nope_buf.at[slot, pl.ds(0, CSA_PAGE)], sem(slot))
                    wait_bytes(S.rope_buf.at[slot, pl.ds(0, CSA_PAGE // 4)], sem(slot))
                    return c
                lax.fori_loop(0, n_pg, pg, 0)

        lane_bit = 1 << (lax.broadcasted_iota(jnp.int32, (1, self.d.RP), 1) // self.d.heads)  # row's bit

        def attend(g, nvalid):
            if "attend" in ABLATE:
                return
            e = jnp.concatenate([S.un_v[pl.ds(g * (G // 128) + i, 1), :] for i in range(G // 128)],
                                axis=1)  # [1, G]
            ecol = jnp.transpose(jnp.broadcast_to(e, (8, G)))[:, 0:1]  # [G, 1]
            kv, kv_t = attn_csa.csa_keys(S.un_n[...], S.un_r[...], (ecol >> attn_csa.UNION_BITS) & 3,
                                         TRANSPOSE_KEYS)
            if not (isinstance(nvalid, int) and nvalid == G):  # a partial granule: mask the tail
                valid = lax.broadcasted_iota(jnp.int32, (G, 1), 0) < nvalid
                kv = jnp.where(valid, kv, 0)
                kv_t = None if kv_t is None else jnp.where(valid.T, kv_t, 0)
                ecol = jnp.where(valid, ecol, 0)
            win.update(r, kv, kv_t, (ecol & lane_bit) != 0, first=False)

        def copy_rows(slot, base, p0, cnt):
            """Union entries [p0, p0 + cnt) -> staging rows p0 % G .."""
            nb, rb = S.nope_buf.at[slot], S.rope_buf.at[slot]
            row0 = p0 % G

            def copy_row(i):
                p = p0 + i
                local = (S.un_s[p // 128, p % 128] >> attn_csa.UNION_BITS) - base
                row = pl.ds(row0 + i, 1)
                S.un_n[row, :] = nb[pl.ds(local, 1), :]
                S.un_r[row, :] = rb[pl.ds(local >> 2, 1), :]

            def cp8(j, c):
                for u in range(8):
                    copy_row(j * 8 + u)
                return c

            def cp1(i, c):
                copy_row(i)
                return c

            if "copy" not in ABLATE:
                lax.fori_loop(0, cnt // 8, cp8, 0)
                lax.fori_loop(cnt // 8 * 8, cnt, cp1, 0)

        for b0 in range(CSA_SLOTS - 1):  # the ring's first blocks
            @pl.when(b0 < nblk)
            def _(b0=b0):
                start(b0, b0)

        def blk(b, p):
            slot = b % CSA_SLOTS
            wait(b, slot)

            @pl.when(b + CSA_SLOTS - 1 < nblk)
            def _():
                start(b + CSA_SLOTS - 1, (b + CSA_SLOTS - 1) % CSA_SLOTS)

            hi = S.bend_s[0, b]

            def chunk(p):
                take = jnp.minimum(hi - p, G - p % G)
                copy_rows(slot, b * blk_rows, p, take)

                @pl.when((p + take) % G == 0)
                def _():
                    attend((p + take) // G - 1, G)

                return p + take

            return lax.while_loop(lambda p_: p_ < hi, chunk, p)

        lax.fori_loop(0, nblk, blk, jnp.int32(0))

        @pl.when(U % G != 0)
        def _():
            attend(U // G, U % G)

    def csa(self, r):
        """Selected rows of every query row, page block by page block: rows are
        copied out of the streamed pages into per-row staging (two halves of
        GRAN rows each); a round attends one half of every query row, the seven
        independent updates in one basic block so their latencies overlap."""
        assert self.d.heads == 128, "rows mode stages 128-lane query rows"
        S, win, T = self.S, self.win, self.d.q
        n = self.meta.seq_lens[r] // CSA_RATIO
        npages = core.cdiv(n, CSA_PAGE)
        nblk = core.cdiv(npages, CSA_BLK)
        blk_rows = CSA_BLK * CSA_PAGE
        G = GRAN
        dn, dr = self.disp["kv"], self.disp["kv_rope"]
        F, H, P, ST = 0, 8, 16, 24  # cnt: pending rows / A-half id / list pointer /
        # list position of each half's first staged row (ST + 2 t + half)
        for t in range(T):
            S.cnt[F + t] = 0
            S.cnt[H + t] = 0
            S.cnt[P + t] = 0
            S.cnt[ST + 2 * t] = 0
            S.cnt[ST + 2 * t + 1] = 0
        # Per-row list ends of every page block: bend[t, b] = #entries < (b + 1) * blk_rows.
        tk = S.tk_v[...]
        lane = lax.broadcasted_iota(jnp.int32, (8, 128), 1)

        def block_end(bb, ends):
            c = jnp.sum(((tk >= 0) & (tk < (bb + 1) * blk_rows)).astype(jnp.int32), axis=1,
                        keepdims=True)
            return jnp.where(lane == bb, c, ends)

        S.bend_v[...] = lax.fori_loop(0, nblk, block_end, jnp.zeros((8, 128), jnp.int32))
        S.tk_t[pl.ds(0, TOPK), :] = jnp.concatenate([tk.T, jnp.zeros((TOPK, 120), jnp.int32)], 1)
        S.tk_t[pl.ds(TOPK, G), :] = jnp.zeros((G, 128), jnp.int32)
        cp = pltpu.make_async_copy(S.bend_v, S.bend_s, S.sems.at[S_APE])
        cp.start()
        cp.wait()

        def start(b, slot):
            n_pg = jnp.clip(npages - b * CSA_BLK, 0, CSA_BLK)

            def nope(ref, p):
                page = self.bt("kv", r, b * CSA_BLK + p)
                pltpu.make_async_copy(ref.at[page], S.nope_buf.at[slot, pl.ds(pl.multiple_of(p * CSA_PAGE, CSA_PAGE), CSA_PAGE)],
                                      S.sems.at[S_CSA0 + slot]).start()

            def rope(ref, p):
                rpage = self.bt("kv_rope", r, b * CSA_BLK + p)
                pltpu.make_async_copy(
                    ref.at[rpage], S.rope_buf.at[slot, pl.ds(pl.multiple_of(p * (CSA_PAGE // 4), CSA_PAGE // 4), CSA_PAGE // 4)],
                    S.sems.at[S_CSA0 + slot]).start()
            dn.loop(n_pg, nope)
            dr.loop(n_pg, rope)

        def wait(b, slot):
            def pg(p, c):
                wait_bytes(S.nope_buf.at[slot, pl.ds(0, CSA_PAGE)], S.sems.at[S_CSA0 + slot])
                wait_bytes(S.rope_buf.at[slot, pl.ds(0, CSA_PAGE // 4)], S.sems.at[S_CSA0 + slot])
                return c
            lax.fori_loop(0, jnp.clip(npages - b * CSA_BLK, 0, CSA_BLK), pg, 0)

        def attend(t, half, nvalid):
            if "attend" in ABLATE:
                return
            rows = pl.ds(0, G)
            seg = S.tk_t[pl.ds(S.cnt[ST + 2 * t + half], G), :][:, t:t + 1] & 3
            kv, kv_t = attn_csa.csa_keys(S.stage_n[t, half, rows, :], S.stage_r[t, half, rows, :],
                                         seg, True)
            valid = lax.broadcasted_iota(jnp.int32, (G, 1), 0) < nvalid
            kv = jnp.where(valid, kv, 0)
            kv_t = jnp.where(valid.T, kv_t, 0)
            win.update(r, kv, kv_t, jnp.broadcast_to(valid, (G, 128)), first=False,
                       lanes=(t * 128, 128))

        def round_blockdiag():
            """All rows' halves as one key block, each query row seeing only its own."""
            kvs, kvts, oks = [], [], []
            for t in range(T):
                half, nv = S.cnt[H + t], jnp.minimum(S.cnt[F + t], G)
                rows = pl.ds(0, G)
                seg = S.tk_t[pl.ds(S.cnt[ST + 2 * t + half], G), :][:, t:t + 1] & 3
                kv, kv_t = attn_csa.csa_keys(S.stage_n[t, half, rows, :],
                                             S.stage_r[t, half, rows, :], seg, True)
                valid = lax.broadcasted_iota(jnp.int32, (G, 1), 0) < nv
                kvs.append(jnp.where(valid, kv, 0))
                kvts.append(jnp.where(valid.T, kv_t, 0))
                oks.append(valid)
            kv, kv_t, ok = (jnp.concatenate(kvs, 0), jnp.concatenate(kvts, 1),
                            jnp.concatenate(oks, 0))
            krow = lax.broadcasted_iota(jnp.int32, (T * G, 1), 0) // G
            qtok = lax.broadcasted_iota(jnp.int32, (1, self.d.RP), 1) // 128
            win.update(r, kv, kv_t, ok & (krow == qtok), first=False)

        def round_():
            if ROUND_MODE == "blockdiag" and "attend" not in ABLATE:
                round_blockdiag()
            for t in range(T if ROUND_MODE == "rows" else 0):  # one basic block of 7 updates
                attend(t, S.cnt[H + t], jnp.minimum(S.cnt[F + t], G))
            for t in range(T):
                full = S.cnt[F + t] >= G
                S.cnt[F + t] = jnp.where(full, S.cnt[F + t] - G, 0)
                S.cnt[H + t] = jnp.where(full, 1 - S.cnt[H + t], S.cnt[H + t])

        def min_fill():
            m = S.cnt[F]
            for t in range(1, T):
                m = jnp.minimum(m, S.cnt[F + t])
            return m

        def max_fill():
            m = S.cnt[F]
            for t in range(1, T):
                m = jnp.maximum(m, S.cnt[F + t])
            return m

        def copy_rows(t, slot, base, src0, half, row0, cnt):
            """List entries [src0, src0 + cnt) of row t -> rows row0.. of `half`."""
            sn, sr = S.stage_n.at[t, half], S.stage_r.at[t, half]
            nb, rb = S.nope_buf.at[slot], S.rope_buf.at[slot]

            def copy_row(i):
                idx = S.tk_s[t, src0 + i]
                local = idx - base
                row = pl.ds(row0 + i, 1)
                sn[row, :] = nb[pl.ds(local, 1), :]
                sr[row, :] = rb[pl.ds(local >> 2, 1), :]

            def cp8(j, c):
                for u in range(8):
                    copy_row(j * 8 + u)
                return c

            def cp1(i, c):
                copy_row(i)
                return c

            if "copy" not in ABLATE:
                lax.fori_loop(0, cnt // 8, cp8, 0)
                lax.fori_loop(cnt // 8 * 8, cnt, cp1, 0)

        @pl.when(nblk > 0)
        def _():
            start(0, 0)

        def blk(b, c):
            slot = b % 2
            wait(b, slot)

            @pl.when(b + 1 < nblk)
            def _():
                start(b + 1, 1 - slot)

            base = b * blk_rows

            def per_row(t, c2):
                hi = S.bend_s[t, b]

                def chunk(ptr):
                    @pl.when(S.cnt[F + t] == 2 * G)  # this row is a round ahead
                    def _():
                        round_()

                    fill, h = S.cnt[F + t], S.cnt[H + t]
                    take = jnp.minimum(hi - ptr, 2 * G - fill)
                    in_a = jnp.clip(G - fill, 0, take)

                    @pl.when(fill == 0)
                    def _():
                        S.cnt[ST + 2 * t + h] = ptr

                    @pl.when((fill + take > G) & (fill <= G))
                    def _():
                        S.cnt[ST + 2 * t + 1 - h] = ptr + in_a
                    copy_rows(t, slot, base, ptr, h, fill, in_a)
                    copy_rows(t, slot, base, ptr + in_a, 1 - h, jnp.maximum(fill - G, 0),
                              take - in_a)
                    S.cnt[F + t] = fill + take
                    return ptr + take

                S.cnt[P + t] = lax.while_loop(lambda p_: p_ < hi, chunk, S.cnt[P + t])
                return c2

            lax.fori_loop(0, T, per_row, 0)
            lax.while_loop(lambda c_: min_fill() >= G, lambda c_: (round_(), c_ + 1)[1], 0)
            return c

        lax.fori_loop(0, nblk, blk, 0)
        lax.while_loop(lambda c_: max_fill() > 0, lambda c_: (round_(), c_ + 1)[1], 0)


def owner_attention_scratch_shapes(geom, leased: bool) -> list:
    return flat_scratch_shapes(dims_of(geom), geom.lease_bytes if leased else 0)
