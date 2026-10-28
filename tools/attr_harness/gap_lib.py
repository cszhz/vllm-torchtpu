# ruff: noqa: E501, E731, E741, SIM105  (diagnostic branch, not for merge)
"""GAP shared library: input recipes, config enumeration (pruned with the modules' own check_config),
compile / bitwise-check / XProf timing for kernels A (qa_rope), B (qb_rope, + projection_rope.chain) and C (compress_rope).

Pod side (TPU7x, one device) and CPU side (enumeration only: `python3 gap_sweep.py enum`).
No file under the worktree src/ or tests/ is modified; this module only imports the staged copies.

Timing method (same as KA1/WC1): per candidate one jitted function with a unique module name; XProf kernel event =
XLA Ops event whose name contains the pallas kernel name; BLOCKS blocks x CALLS profiled calls (+WARM warmup per block),
blocks run in forward then reversed candidate order (ABBA) so drift cancels. Kernel-event median is the headline number.
"""

import dataclasses
import functools
import json
import os
import re
import shutil
import time
import traceback
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from vllm_torchtpu.kernels.attention_preprocessing import (
    compress_rope,
    qa_rope,
    qb_rope,
    rope_gather,
)
from vllm_torchtpu.kernels.attention_preprocessing import tuned_block_sizes as TBS

try:  # decode incumbent (T=288 specialization)
    from vllm_torchtpu.kernels.quantized_matmul import projection_rope

    PROJ_IMPORT_ERROR = None
except Exception:  # noqa: BLE001
    projection_rope = None
    PROJ_IMPORT_ERROR = traceback.format_exc()[-2000:]

BF16, FP8 = ml_dtypes.bfloat16, ml_dtypes.float8_e4m3fn
D, R, KV, H, HD = 5120, 1280, 512, 64, 512
CACHE_ROWS = 98816
BLOCKS, CALLS, WARM = 2, 12, 3  # 24 kernel samples per candidate, ABBA over 2 blocks
OUT = Path(os.environ.get("AOT_OUT", "/tmp/res"))

MARK = {
    "A": "qa_rope_t",
    "B": "qb_rope_t",
    "C": "compress_rope_ratio",
    "chain": "fused_quant_projection_rope_output48_all",
}

TBS_A = (48, 96, 128, 144, 240, 256, 288, 512)
CH_A = (128, 256, 896, 1792)
TBS_C = (48, 96, 144, 240, 256, 288, 512)
CH_C = {
    2: (256, 512, 1024),
    1: (256, 1024),
}  # 512 is ratio-2 only per the task; 1024 is invalid for ratio 1 (pruned)
HB_B, SLOTS_B = (4, 8, 16), (2, 4, 8)
EXTRA_TB_B = (96, 128, 192)  # not in the task list; only when the tile count stays <= 4


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def nrm(rng, shape, dt):
    return rng.normal(size=shape).astype(np.float32).astype(dt)


# ----------------------------------------------------------------------------------------- default configs
def default_A(T):
    return TBS.QA_ROPE_TUNING.get((7, T, D, R + KV)) or qa_rope.default_config(T)


def pad32(T):
    return -(-T // 32) * 32


def default_B(T):
    p = pad32(T)
    return TBS.QB_ROPE_TUNING.get((7, p, R, H * HD)) or qb_rope.default_config(p, R, H)


def default_C(T, ratio):
    return TBS.COMPRESS_ROPE_TUNING.get(
        (7, T, D, ratio)
    ) or compress_rope.default_config(T)


# ----------------------------------------------------------------------------------------- enumeration
def enum_A(T, emit_qr):
    width = R + KV if emit_qr else KV
    ok, pruned = [], []
    for tb in TBS_A:
        for ch in CH_A:
            cfg = TBS.QARopeConfig(token_block=tb, weight_chunk=ch)
            try:
                qa_rope.check_config(cfg, T, width)
            except ValueError as exc:
                pruned.append({"cfg": dataclasses.asdict(cfg), "why": str(exc)[:160]})
                continue
            vb = qa_rope.vmem_bytes(cfg, D, width, 64)
            if vb > cfg.vmem_limit_bytes:
                pruned.append(
                    {
                        "cfg": dataclasses.asdict(cfg),
                        "why": f"vmem estimate {vb} > limit",
                    }
                )
                continue
            ok.append(cfg)
    return ok, pruned


def enum_B(T):
    p = pad32(T)
    tbs = []
    if p <= 512:
        tbs.append(p)
    for tb in (256, 512):
        if p % tb == 0 and tb not in tbs:
            tbs.append(tb)
    extras = [
        tb for tb in EXTRA_TB_B if p <= 512 and p % tb == 0 and tb < p and p // tb <= 4
    ]
    ok, pruned = [], []
    for tb in tbs + extras:
        for hb in HB_B:
            for sl in SLOTS_B:
                cfg = TBS.QBRopeConfig(
                    token_block=tb,
                    head_block=hb,
                    output_slots=sl,
                    quant_fragment=qb_rope._largest_divisor(tb, 64, 32),
                    post_fragment=32,
                )
                try:
                    seg = qb_rope._segment_tokens(cfg, p, R, H, HD)
                    qb_rope.check_config(cfg, seg, H)
                except ValueError as exc:
                    pruned.append(
                        {"cfg": dataclasses.asdict(cfg), "why": str(exc)[:160]}
                    )
                    continue
                ok.append(cfg)
    return ok, pruned, extras


def enum_C(T, ratio):
    width = ratio * KV
    ok, pruned = [], []
    for tb in TBS_C:
        for ch in CH_C[ratio]:
            cfg = TBS.CompressConfig(token_block=tb, weight_chunk=ch)
            try:
                compress_rope.check_config(cfg, T, width)
            except ValueError as exc:
                pruned.append({"cfg": dataclasses.asdict(cfg), "why": str(exc)[:160]})
                continue
            vb = compress_rope.vmem_bytes(cfg, D, ratio)
            if vb > cfg.vmem_limit_bytes:
                pruned.append(
                    {
                        "cfg": dataclasses.asdict(cfg),
                        "why": f"vmem estimate {vb} > limit",
                    }
                )
                continue
            ok.append(cfg)
    return ok, pruned


# ----------------------------------------------------------------------------------------- input recipes
_W = {}


def cache_table():
    """[cos | sin] f32[98816, 64], KA1 recipe."""
    if "cache" not in _W:
        rng = np.random.default_rng(11)
        ang = rng.uniform(-3, 3, size=(CACHE_ROWS, 32))
        _W["cache"] = jax.device_put(
            np.concatenate((np.cos(ang), np.sin(ang)), 1).astype(np.float32)
        )
    return _W["cache"]


def rows_for(pos):
    return jax.jit(rope_gather.gather_rows)(cache_table(), pos)


def a_inputs(T):
    """tests/test_qa_rope.py::_inputs recipe (KA1 case), contiguous positions."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(T, D)).astype(np.float32)
    x[0] = 0
    x[1] *= 1e-6
    x = x.astype(BF16)
    w = nrm(rng, (D, R + KV), FP8)
    s = rng.uniform(0.01, 0.1, R + KV).astype(np.float32)
    gq, gk = nrm(rng, (R,), BF16), nrm(rng, (KV,), BF16)
    pos = (np.arange(T) + 17).astype(np.int32)
    x, w, s, gq, gk, pos = jax.device_put((x, w, s, gq, gk, pos))
    return [x, w, s, gq, gk, rows_for(pos)]


def b_weights():
    if "qb" not in _W:
        rng = np.random.default_rng(31)
        _W["qb"] = jax.device_put(
            (
                nrm(rng, (R, H * HD), FP8),
                rng.uniform(0.01, 0.1, H * HD).astype(np.float32),
            )
        )
    return _W["qb"]


def b_inputs(T):
    """W2 recipe: random scattered positions (decode-like), qr normal bf16."""
    w, s = b_weights()
    qr = jax.device_put(nrm(np.random.default_rng(200 + T), (T, R), BF16))
    pos = jax.device_put(
        np.random.default_rng(100 + T).integers(0, CACHE_ROWS, T).astype(np.int32)
    )
    return qr, w, s, pos


CAP, ROW, REC_ROWS = 8, 128, 8
ROPE_N = 16384


def make_case(T, ratio, structure, parity):
    """WC1 make_case (verbatim logic, minus the scatter helper): structure 'single' | 'manyL'."""
    rng = np.random.default_rng(seed_of("case", T, ratio, structure, parity))
    lens = (
        [T]
        if structure == "single"
        else [int(structure[4:])] * (T // int(structure[4:]))
    )
    nreq = len(lens)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
    positions = np.zeros(T, np.int64)
    has_prev = np.ones((T, 1), np.int32)
    valid = np.zeros((T, 1), np.int32)
    ring_write = np.full(T, -1, np.int32)
    ring_read = np.full(T, -1, np.int32)
    nblocks = nreq + 2
    blocks = rng.permutation(nblocks)[:nreq]
    for i, (s0, n) in enumerate(zip(starts, lens)):
        par = {"odd": 1, "even": 0, "mixed": i % 2}[parity]
        p0 = 8 * int(rng.integers(1, 1500)) + par
        pos = p0 + np.arange(n)
        positions[s0 : s0 + n] = pos
        has_prev[s0] = 0
        blk = int(blocks[i])
        ring_read[s0] = blk * CAP + (p0 - 1) % CAP
        for j in range(max(0, n - CAP), n):
            ring_write[s0 + j] = blk * CAP + (p0 + j) % CAP
        valid[s0 : s0 + n, 0] = (pos % 2 == 1) if ratio == 2 else 1
    x = jnp.asarray(rng.normal(size=(T, D)).astype(np.float32) * 0.5, jnp.bfloat16)
    w = jnp.asarray(
        rng.normal(size=(D, ratio * KV)).astype(np.float32) * 0.02, jnp.bfloat16
    )
    gamma = jnp.asarray(rng.normal(size=(KV,)).astype(np.float32), jnp.bfloat16)
    angles = rng.uniform(-3, 3, size=(ROPE_N, 32))
    table = jnp.asarray(
        np.concatenate((np.cos(angles), np.sin(angles)), 1), jnp.float32
    )
    group_starts = (positions // ratio * ratio).astype(np.int32)
    rows = rope_gather.gather_rows(table, jnp.asarray(group_starts))
    case = dict(
        T=T,
        ratio=ratio,
        x=x,
        w=w,
        gamma=gamma,
        rows=rows,
        has_prev=jnp.asarray(has_prev),
        valid=jnp.asarray(valid),
        n_writes=int((ring_write >= 0).sum()),
        n_reads=int((ring_read >= 0).sum()),
        ring_write=jnp.asarray(ring_write) if ratio == 2 else None,
        ring_read=jnp.asarray(ring_read) if ratio == 2 else None,
        pool=None,
    )
    if ratio == 2:
        total = nblocks * CAP
        bits = rng.normal(size=(total, 2 * KV)).astype(np.float32).view(np.int32)
        case["pool"] = bits.reshape(-1, ROW)
    return case


# ----------------------------------------------------------------------------------------- candidates
class POOLTYPE:
    pass


POOL = POOLTYPE()
STATE = {}


@dataclasses.dataclass
class Cand:
    cid: str
    label: str
    cfg: dict
    fn: object
    args: list
    marker: str
    donate: bool = False
    is_default: bool = False
    is_baseline: bool = False
    jname: str = ""
    compiled: object = None
    rec: dict = dataclasses.field(default_factory=dict)
    in_shardings: object = None


def _a_call(cfg, emit_qr, x, w, s, gq, gk, rows):
    return qa_rope.fused_qa_kv_norm_rope(
        x, w, s, gq, gk, rows, emit_qr=emit_qr, config=cfg
    )


def _b_call(cfg, qr, w, s, rows):
    return qb_rope.fused_qb_rope(qr, w, s, rows, config=cfg)


def _c2_call(cfg, x, w, g, rows, hp, valid, pool, rw, rr):
    return compress_rope.fused_compress_rope(
        x, w, g, rows, hp, valid, pool, rw, rr, ratio=2, config=cfg
    )


def _c1_call(cfg, x, w, g, rows, hp, valid):
    return compress_rope.fused_compress_rope(
        x, w, g, rows, hp, valid, ratio=1, config=cfg
    )[:3]


def cfg_label(cfg):
    d = dataclasses.asdict(cfg)
    d.pop("vmem_limit_bytes", None)
    return ",".join(f"{k}={v}" for k, v in d.items() if v is not None)


def build_group(gid):
    """-> dict(cands, meta, pruned). gid: A_<T>, A_<T>_noqr, B_<T>, C2_<T>, C1_<T>."""
    parts = gid.split("_")
    kind, T = parts[0], int(parts[1])
    meta = {"gid": gid, "kind": kind, "T": T}
    cands = []
    if kind == "A":
        emit_qr = "noqr" not in parts
        meta["emit_qr"] = emit_qr
        cfgs, pruned = enum_A(T, emit_qr)
        dflt = default_A(T)
        if dflt not in cfgs:
            cfgs.insert(0, dflt)
        args = a_inputs(T)
        for cfg in cfgs:
            cands.append(
                Cand(
                    f"{gid}|{cfg_label(cfg)}",
                    cfg_label(cfg),
                    dataclasses.asdict(cfg),
                    functools.partial(_a_call, cfg, emit_qr),
                    args,
                    MARK["A"],
                    is_default=(cfg == dflt),
                )
            )
    elif kind == "B":
        cfgs, pruned, extras = enum_B(T)
        meta["extra_token_blocks"] = extras
        dflt = default_B(T)
        if dflt not in cfgs:
            cfgs.insert(0, dflt)
        qr, w, s, pos = b_inputs(T)
        rows = rows_for(pos)
        for cfg in cfgs:
            cands.append(
                Cand(
                    f"{gid}|{cfg_label(cfg)}",
                    cfg_label(cfg),
                    dataclasses.asdict(cfg),
                    functools.partial(_b_call, cfg),
                    [qr, w, s, rows],
                    MARK["B"],
                    is_default=(cfg == dflt),
                )
            )
        if T == 288 and projection_rope is not None:
            cache = cache_table()
            sh = None
            try:
                from jax.experimental.layout import Format, Layout

                fmt = Format(
                    Layout(major_to_minor=(0, 1)),
                    jax.sharding.SingleDeviceSharding(jax.devices("tpu")[0]),
                )
                cache = jax.device_put(np.asarray(cache), fmt)
                sh = tuple(fmt if j == 4 else None for j in range(5))
                meta["chain_cache_layout"] = str(
                    getattr(cache, "format", None) or getattr(cache, "layout", None)
                )
            except Exception:  # noqa: BLE001
                meta["chain_cache_layout"] = (
                    "default (Format API failed): " + traceback.format_exc()[-300:]
                )
            cands.append(
                Cand(
                    f"{gid}|chain",
                    "chain",
                    {},
                    projection_rope.chain,
                    [qr, w, s, pos, cache],
                    MARK["chain"],
                    is_baseline=True,
                    in_shardings=sh,
                )
            )
        elif T == 288:
            meta["chain_import_error"] = PROJ_IMPORT_ERROR
    elif kind in ("C2", "C1"):
        ratio = int(kind[1])
        meta["ratio"] = ratio
        cfgs, pruned = enum_C(T, ratio)
        dflt = default_C(T, ratio)
        if dflt not in cfgs:
            cfgs.insert(0, dflt)
        structure, parity = ("single", "odd") if T == 2048 else ("many6", "mixed")
        case = make_case(T, ratio, structure, parity)
        meta.update(
            structure=structure,
            parity=parity,
            n_writes=case["n_writes"],
            n_reads=case["n_reads"],
            pool_rows=None if case["pool"] is None else int(case["pool"].shape[0]),
        )
        meta["_case"] = case
        base = [
            case["x"],
            case["w"],
            case["gamma"],
            case["rows"],
            case["has_prev"],
            case["valid"],
        ]
        for cfg in cfgs:
            if ratio == 2:
                cands.append(
                    Cand(
                        f"{gid}|{cfg_label(cfg)}",
                        cfg_label(cfg),
                        dataclasses.asdict(cfg),
                        functools.partial(_c2_call, cfg),
                        base + [POOL, case["ring_write"], case["ring_read"]],
                        MARK["C"],
                        donate=True,
                        is_default=(cfg == dflt),
                    )
                )
            else:
                cands.append(
                    Cand(
                        f"{gid}|{cfg_label(cfg)}",
                        cfg_label(cfg),
                        dataclasses.asdict(cfg),
                        functools.partial(_c1_call, cfg),
                        base,
                        MARK["C"],
                        is_default=(cfg == dflt),
                    )
                )
    else:
        raise ValueError(gid)
    for i, c in enumerate(cands):
        c.jname = f"gap_{kind.lower()}{T}_{i:03d}"
    return dict(cands=cands, meta=meta, pruned=pruned)


# ----------------------------------------------------------------------------------------- run
def named(name, fn):
    def wrapper(*a):
        return fn(*a)

    wrapper.__name__ = name
    return wrapper


def real_args(c, pool=None):
    return [pool if a is POOL else a for a in c.args]


LLO_SRC = os.environ.get(
    "GAP_LLO_DUMP"
)  # = the --xla_jf_dump_to dir (unset: no LLO handling)
LLO_KEEP = os.environ.get("GAP_LLO_KEEP")  # per-candidate kept LLO files
_KEEP_SUFFIX = (
    "final_bundles.txt",
    "packed-bundles-post-ra.txt",
    "final_hlo-static-per-bundle-utilization.txt",
)


def _dump_files():
    if not LLO_SRC or not Path(LLO_SRC).exists():
        return set()
    return {p for p in Path(LLO_SRC).rglob("*") if p.is_file()}


def _llo_dir(c):
    return Path(LLO_KEEP) / re.sub(r"[^A-Za-z0-9_.=,-]", "_", c.cid)


def keep_llo(c, before):
    """Copy the pallas kernel's final bundles + post-RA packed bundles (replay companion) + static utilization of
    the compile that was just done, then delete every new dump file (bounds disk)."""
    if not LLO_SRC or not LLO_KEEP:
        return
    new = list(_dump_files() - before)
    if not new:
        time.sleep(1.0)
        new = list(_dump_files() - before)
    kept = []
    for p in new:
        n = p.name
        if c.marker in n and "schedule-analysis" not in n and n.endswith(_KEEP_SUFFIX):
            d = _llo_dir(c)
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, d / n)
            kept.append(n)
    for p in new:
        p.unlink(missing_ok=True)
    c.rec["llo_files"] = kept
    if not kept:
        c.rec["llo_missing"] = (
            f"{len(new)} new dump files, none matched marker {c.marker}"
        )


def prune_llo(cands, keep_labels):
    """Drop kept LLO of every candidate not in keep_labels (default / best / best-bitwise / chain)."""
    if not LLO_KEEP:
        return
    for c in cands:
        d = _llo_dir(c)
        if d.exists():
            if c.label in keep_labels:
                c.rec["llo_dir"] = d.name
            else:
                shutil.rmtree(d, ignore_errors=True)
                c.rec.pop("llo_files", None)


def compile_cand(c, case=None):
    t = time.perf_counter()
    before = _dump_files()
    try:
        jf = jax.jit(
            named(c.jname, c.fn),
            donate_argnums=(6,) if c.donate else (),
            **({"in_shardings": c.in_shardings} if c.in_shardings else {}),
        )
        pool = jnp.asarray(case["pool"]) if c.donate else None
        c.compiled = jf.lower(*real_args(c, pool)).compile()
        c.rec["compile_s"] = time.perf_counter() - t
    finally:
        keep_llo(c, before)


_UINT = {1: jnp.uint8, 2: jnp.uint16, 4: jnp.uint32}


@jax.jit
def _nbad(a, b):
    ua = jax.lax.bitcast_convert_type(a, _UINT[a.dtype.itemsize])
    ub = jax.lax.bitcast_convert_type(b, _UINT[b.dtype.itemsize])
    return jnp.sum(ua != ub, dtype=jnp.int32)


def run_once(c, case=None):
    """Fresh-input run (fresh pool copy when donated); returns the output leaves."""
    pool = jnp.asarray(case["pool"]) if c.donate else None
    out = jax.block_until_ready(c.compiled(*real_args(c, pool)))
    return jax.tree.leaves(out)


def compare_leaves(a, b):
    if len(a) != len(b):
        return {"leaf_count_mismatch": [len(a), len(b)], "bitwise": False}
    bad, size = [], []
    for x, y in zip(a, b):
        if x.shape != y.shape or x.dtype != y.dtype:
            return {
                "shape_or_dtype_mismatch": [
                    str(x.shape),
                    str(y.shape),
                    str(x.dtype),
                    str(y.dtype),
                ],
                "bitwise": False,
            }
        bad.append(int(_nbad(x, y)))
        size.append(int(np.prod(x.shape)))
    return {"mismatch": bad, "size": size, "bitwise": not any(bad)}


def read_events(directory, jname, marker):
    kern, mods, states = [], [], []
    for path in directory.rglob("*.xplane.pb"):
        for plane in jax.profiler.ProfileData.from_file(str(path)).planes:
            if not plane.name.startswith("/device:TPU:") or "SparseCore" in plane.name:
                continue
            for line in plane.lines:
                if line.name == "_counters_":
                    for e in line.events:
                        if e.name == "P state":
                            states.extend(
                                float(v) for n, v in e.stats if n == "P State"
                            )
                elif line.name == "XLA Modules":
                    mods.extend(
                        float(e.duration_ns) / 1000
                        for e in line.events
                        if f"jit_{jname}" in e.name
                    )
                elif line.name == "XLA Ops":
                    kern.extend(
                        float(e.duration_ns) / 1000
                        for e in line.events
                        if marker in e.name
                    )
    return kern, mods, states


def stats(v):
    if not len(v):
        return None
    return {
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
        "min": float(np.min(v)),
        "n": len(v),
    }


def step(c):
    args = real_args(c, STATE.get("pool"))
    out = c.compiled(*args)
    if c.donate:
        STATE["pool"] = out[3]
    jax.block_until_ready(out)


def time_cands(gid, cands, blocks=BLOCKS, calls=CALLS, warm=WARM):
    """ABBA over `blocks` blocks; fills c.rec['kernel'], ['module'], ['block_medians']. Returns sorted P states."""
    live = [c for c in cands if c.compiled is not None]
    states = set()
    for c in live:
        c.rec["_k"], c.rec["_m"], c.rec["block_medians"] = [], [], []
    for b in range(blocks):
        order = live if b % 2 == 0 else live[::-1]
        for c in order:
            try:
                for _ in range(warm):
                    step(c)
                d = OUT / "xprof" / f"{gid}-{c.jname}-{b}"
                with jax.profiler.trace(str(d), create_perfetto_link=False):
                    for _ in range(calls):
                        step(c)
                k, m, st = read_events(d, c.jname, c.marker)
                shutil.rmtree(d, ignore_errors=True)
                c.rec["_k"] += k
                c.rec["_m"] += m
                c.rec["block_medians"].append(float(np.median(k)) if k else None)
                states.update(st)
            except Exception as exc:  # noqa: BLE001
                c.rec.setdefault("timing_errors", []).append(
                    f"{type(exc).__name__}: {exc}"[:600]
                )
    for c in live:
        k, m = c.rec.pop("_k"), c.rec.pop("_m")
        c.rec["kernel"] = stats(k)
        c.rec["module"] = stats(m)
        c.rec["kernel_samples_us"] = k
    return sorted(states)


def summarize_group(g):
    cs = [c for c in g["cands"] if c.rec.get("kernel")]
    ok = [c for c in cs if not c.is_baseline]
    dflt = next((c for c in cs if c.is_default), None)
    s = {"default": None, "best_bitwise": None, "best_any": None, "baseline": None}
    if dflt:
        s["default"] = {
            "label": dflt.label,
            "cfg": dflt.cfg,
            "median_us": dflt.rec["kernel"]["median"],
            "module_us": dflt.rec["module"] and dflt.rec["module"]["median"],
        }
    bit = [c for c in ok if c.rec.get("vs_default", {}).get("bitwise")]
    if bit:
        b = min(bit, key=lambda c: c.rec["kernel"]["median"])
        s["best_bitwise"] = {
            "label": b.label,
            "cfg": b.cfg,
            "median_us": b.rec["kernel"]["median"],
            "module_us": b.rec["module"] and b.rec["module"]["median"],
        }
    if ok:
        b = min(ok, key=lambda c: c.rec["kernel"]["median"])
        s["best_any"] = {
            "label": b.label,
            "cfg": b.cfg,
            "median_us": b.rec["kernel"]["median"],
            "bitwise_vs_default": b.rec.get("vs_default", {}).get("bitwise"),
            "module_us": b.rec["module"] and b.rec["module"]["median"],
        }
    base = next((c for c in cs if c.is_baseline), None)
    if base:
        s["baseline"] = {
            "label": base.label,
            "median_us": base.rec["kernel"]["median"],
            "module_us": base.rec["module"] and base.rec["module"]["median"],
        }
    return s


def run_group(gid, result, save):
    t0 = time.time()
    G = {"cands": {}, "summary": None, "error": None}
    result["groups"][gid] = G
    try:
        g = build_group(gid)
        cands, meta = g["cands"], g["meta"]
        case = meta.pop("_case", None)
        G["meta"] = meta
        G["pruned"] = g["pruned"]
        G["n_candidates"] = len(cands)
        print(
            f"GROUP {gid}: {len(cands)} candidates, {len(g['pruned'])} pruned on CPU-side check",
            flush=True,
        )
        # sanity: default equals the module's own select_config on this device
        try:
            if meta["kind"] == "A":
                sel = qa_rope.select_config(meta["T"], D, R + KV)
                want = default_A(meta["T"])
            elif meta["kind"] == "B":
                sel = qb_rope.select_config(pad32(meta["T"]), R, H * HD, H)
                want = default_B(meta["T"])
            else:
                sel = compress_rope.select_config(meta["T"], D, meta["ratio"])
                want = default_C(meta["T"], meta["ratio"])
            G["default_equals_select_config"] = bool(sel == want)
            if sel != want:
                print(
                    "WARN default differs from select_config",
                    gid,
                    sel,
                    want,
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            G["default_equals_select_config"] = f"error {exc}"[:200]
        for c in cands:
            G["cands"][c.label] = c.rec
            c.rec.update(
                label=c.label,
                cfg=c.cfg,
                is_default=c.is_default,
                is_baseline=c.is_baseline,
                jname=c.jname,
            )
            try:
                compile_cand(c, case)
            except Exception as exc:  # noqa: BLE001
                c.rec["compile_error"] = f"{type(exc).__name__}: {exc}"[:1500]
                c.compiled = None
                print("COMPILE FAILED", gid, c.label, str(exc)[:300], flush=True)
        save()
        # bitwise vs default (and vs chain for B_288)
        dflt = next((c for c in cands if c.is_default and c.compiled), None)
        chain = next((c for c in cands if c.is_baseline and c.compiled), None)
        ref = run_once(dflt, case) if dflt else None
        ref_chain = run_once(chain, case) if chain else None
        for c in cands:
            if c.compiled is None:
                continue
            try:
                out = run_once(c, case)
                c.rec["finite"] = bool(
                    all(
                        np.isfinite(np.asarray(a, np.float32)).all()
                        for a in out
                        if a.dtype in (jnp.bfloat16, jnp.float32)
                    )
                )
                if ref is not None and not c.is_baseline:
                    c.rec["vs_default"] = compare_leaves(out, ref)
                if ref_chain is not None and not c.is_baseline:
                    c.rec["vs_chain"] = compare_leaves(out[:1], ref_chain[:1])
            except Exception as exc:  # noqa: BLE001
                c.rec["run_error"] = f"{type(exc).__name__}: {exc}"[:1500]
                c.compiled = None
        save()
        if meta["kind"] == "C" or meta["kind"] == "C2" or meta.get("ratio") == 2:
            STATE["pool"] = jax.block_until_ready(jnp.asarray(case["pool"]))
        G["p_states"] = time_cands(gid, cands)
        G["summary"] = summarize_group(g)
        prune_llo(
            cands,
            {
                s["label"]
                for k, s in G["summary"].items()
                if s and k in ("default", "best_bitwise", "best_any", "baseline")
            },
        )
        top = sorted(
            [c for c in cands if c.rec.get("kernel")],
            key=lambda c: c.rec["kernel"]["median"],
        )[:6]
        for c in top:
            print(
                "TOP",
                gid,
                c.label,
                round(c.rec["kernel"]["median"], 2),
                "bitwise_vs_default",
                c.rec.get("vs_default", {}).get("bitwise"),
                "vs_chain_mismatch",
                c.rec.get("vs_chain", {}).get("mismatch"),
                flush=True,
            )
        print("SUMMARY", gid, json.dumps(G["summary"]), "P", G["p_states"], flush=True)
    except Exception as exc:  # noqa: BLE001
        G["error"] = f"{type(exc).__name__}: {exc}"[:3000]
        G["traceback"] = traceback.format_exc()[-5000:]
        print("GROUP FAILED", gid, str(exc)[:500], flush=True)
        try:  # group died: keep only the default's LLO so the artifact stays small
            prune_llo(cands, {c.label for c in cands if c.is_default})
        except Exception:  # noqa: BLE001
            pass
    G["wall_s"] = time.time() - t0
    save()


ALL_GROUPS_P1 = [
    "A_2048",
    "A_288",
    "A_288_noqr",
    "B_2048",
    "B_288",
    "C2_2048",
    "C2_288",
    "C1_2048",
    "C1_288",
]
ALL_GROUPS_P2 = [
    "A_240",
    "A_96",
    "B_384",
    "B_240",
    "B_192",
    "B_96",
    "C2_240",
    "C2_96",
    "C1_240",
    "C1_96",
]
