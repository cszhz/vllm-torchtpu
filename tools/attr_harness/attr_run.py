# ruff: noqa: E501, E731, E741, SIM105  (diagnostic branch, not for merge)
"""ATTR pod script: time + dump LLO + (diag) device-trace events for the 8 contract cases, current select_config.

usage: attr_run.py MODE        MODE in {formal, diag}
env:   AOT_OUT (result dir), GAP_LLO_DUMP / GAP_LLO_KEEP (LLO dump dir / kept files; gap_lib.keep_llo)
formal: ordinary profile, no bundle instrumentation, no custom_call_tracing (flags set by the caller).
diag:   same inputs, kernels patched with jax.named_scope stages only (make_diag.py), bundle-instrumented trace flags.
Per case (one jitted function, unique module name): 40 kernel samples (4 blocks x 10 calls, 3 warmup per block), module
samples, P state from _counters_, output crc32 of every leaf (formal vs diag identity), kept LLO of THIS compile
(original / post-RA / final bundles / utilization), and one extra 2-call capture whose xplane is kept and whose device
events are dumped to events_<case>.json.gz (+ line/DMA coverage summary).
"""

import functools
import gzip
import json
import os
import shutil
import sys
import time
import traceback
import zlib
from collections import Counter

import gap_lib as L
import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.attention_preprocessing import (
    compress_rope,
    qa_rope,
    qb_rope,
)

MODE = sys.argv[1]
OUT = L.OUT
OUT.mkdir(parents=True, exist_ok=True)
# keep original + every post-RA pass in addition to gap_lib's final bundles / packed / utilization
L._KEEP_SUFFIX = tuple(L._KEEP_SUFFIX) + ("original.txt", "post-ra.txt")
BLOCKS, CALLS, WARM = 4, 10, 3
D, R, KV, H, HD = L.D, L.R, L.KV, L.H, L.HD

QAC, QBC, CC = L.TBS.QARopeConfig, L.TBS.QBRopeConfig, L.TBS.CompressConfig
CASES = [
    dict(id="A_2048", kind="A", T=2048, exp=QAC(token_block=512, weight_chunk=896)),
    dict(id="A_288", kind="A", T=288, exp=QAC(token_block=288, weight_chunk=896)),
    dict(id="B_2048", kind="B", T=2048, exp=QBC(token_block=512)),
    dict(
        id="B_288",
        kind="B",
        T=288,
        exp=QBC(token_block=288, head_block=8, output_slots=4, quant_fragment=32),
    ),
    dict(
        id="C2_2048",
        kind="C2",
        T=2048,
        exp=CC(token_block=512, weight_chunk=512),
        structure=("single", "odd"),
    ),
    dict(
        id="C2_288",
        kind="C2",
        T=288,
        exp=CC(token_block=288, weight_chunk=512),
        structure=("many6", "mixed"),
    ),
    dict(
        id="C1_2048",
        kind="C1",
        T=2048,
        exp=CC(token_block=256),
        structure=("single", "odd"),
    ),
    dict(
        id="C1_288",
        kind="C1",
        T=288,
        exp=CC(token_block=144),
        structure=("many6", "mixed"),
    ),
]


def selected(c):
    k, T = c["kind"], c["T"]
    if k == "A":
        return qa_rope.select_config(T, D, R + KV)
    if k == "B":
        return qb_rope.select_config(L.pad32(T), R, H * HD, H)
    return compress_rope.select_config(T, D, int(k[1]))


def build(c):
    k, T = c["kind"], c["T"]
    if k == "A":
        args = L.a_inputs(T)
        return L.Cand(
            c["id"],
            c["id"],
            {},
            functools.partial(L._a_call, None, True),
            args,
            L.MARK["A"],
        ), None
    if k == "B":
        qr, w, s, pos = L.b_inputs(T)
        rows = L.rows_for(pos)
        return L.Cand(
            c["id"],
            c["id"],
            {},
            functools.partial(L._b_call, None),
            [qr, w, s, rows],
            L.MARK["B"],
        ), None
    ratio = int(k[1])
    case = L.make_case(T, ratio, *c["structure"])
    base = [
        case["x"],
        case["w"],
        case["gamma"],
        case["rows"],
        case["has_prev"],
        case["valid"],
    ]
    if ratio == 2:
        cand = L.Cand(
            c["id"],
            c["id"],
            {},
            functools.partial(L._c2_call, None),
            base + [L.POOL, case["ring_write"], case["ring_read"]],
            L.MARK["C"],
            donate=True,
        )
    else:
        cand = L.Cand(
            c["id"], c["id"], {}, functools.partial(L._c1_call, None), base, L.MARK["C"]
        )
    return cand, case


def crcs(leaves):
    return [
        zlib.crc32(np.ascontiguousarray(np.asarray(a)).view(np.uint8).tobytes())
        for a in leaves
    ]


def stat_val(v):
    try:
        return v if isinstance(v, (int, float, str, bool)) else str(v)[:120]
    except Exception:  # noqa: BLE001
        return "?"


def extract_events(d, name, max_events=400000):
    """All device events of the capture in `d`: per (plane, line) counts + names, and the events themselves (gz json)."""
    summary, events = {}, {}
    total = 0
    for path in sorted(d.rglob("*.xplane.pb")):
        for plane in jax.profiler.ProfileData.from_file(str(path)).planes:
            if not plane.name.startswith("/device:TPU:"):
                continue
            for line in plane.lines:
                key = f"{plane.name}|{line.name}"
                evs = list(line.events)
                names = Counter(e.name for e in evs)
                dma_like = sum(1 for e in evs if "dma" in e.name.lower())
                summary[key] = {
                    "n": len(evs),
                    "distinct_names": len(names),
                    "dma_named_events": dma_like,
                    "top_names": names.most_common(25),
                }
                if (
                    not evs
                    or plane.name != "/device:TPU:0"
                    and "SparseCore" not in plane.name
                ):
                    continue
                if line.name in ("_counters_",) or "SparseCore" not in plane.name:
                    keep = []
                    for e in evs:
                        if total >= max_events:
                            break
                        total += 1
                        keep.append(
                            [
                                e.name,
                                getattr(e, "start_ns", None),
                                float(e.duration_ns),
                                {str(n): stat_val(v) for n, v in list(e.stats)[:12]},
                            ]
                        )
                    events[key] = keep
    (OUT / f"events_{name}.json.gz").write_bytes(
        gzip.compress(json.dumps(events).encode())
    )
    (OUT / f"lines_{name}.json").write_text(json.dumps(summary, indent=1))
    return summary


def time_case(c, label):
    ks, ms, states = [], [], set()
    for b in range(BLOCKS):
        for _ in range(WARM):
            L.step(c)
        d = OUT / "xprof" / f"{label}-{b}"
        with jax.profiler.trace(str(d), create_perfetto_link=False):
            for _ in range(CALLS):
                L.step(c)
        k, m, st = L.read_events(d, c.jname, c.marker)
        ks += k
        ms += m
        states.update(st)
        shutil.rmtree(d, ignore_errors=True)
    return ks, ms, sorted(states)


def main():
    dev = jax.devices("tpu")[0]
    result = {
        "mode": MODE,
        "versions": {"jax": jax.__version__},
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS"),
        "device_kind": dev.device_kind,
        "blocks": BLOCKS,
        "calls": CALLS,
        "warm": WARM,
        "cases": {},
    }
    save = lambda: (OUT / "result.json").write_text(
        json.dumps(result, indent=1, default=str)
    )  # noqa: E731
    save()
    print(
        "ENV",
        MODE,
        jax.__version__,
        dev.device_kind,
        os.environ.get("LIBTPU_INIT_ARGS"),
        flush=True,
    )
    only = os.environ.get("ATTR_CASES")
    with jax.default_device(dev):
        for c in CASES:
            if only and c["id"] not in only.split(","):
                continue
            rec = result["cases"][c["id"]] = {"T": c["T"], "kind": c["kind"]}
            try:
                sel = selected(c)
                rec["select_config"] = L.cfg_label(sel)
                rec["expected_config"] = L.cfg_label(c["exp"])
                rec["select_equals_expected"] = sel == c["exp"]
                cand, case = build(c)
                cand.jname = "attr_" + c["id"].lower()
                cand.rec = rec
                t0 = time.time()
                L.compile_cand(cand, case)
                rec["compile_wall_s"] = time.time() - t0
                if case is not None and case.get("pool") is not None:
                    L.STATE["pool"] = jax.block_until_ready(jnp.asarray(case["pool"]))
                leaves = L.run_once(cand, case)
                rec["output_crc32"] = crcs(leaves)
                if case is not None and case.get("pool") is not None:
                    L.STATE["pool"] = jax.block_until_ready(jnp.asarray(case["pool"]))
                ks, ms, states = time_case(cand, c["id"])
                rec["kernel"] = L.stats(ks)
                rec["module"] = L.stats(ms)
                rec["p_states"] = states
                rec["kernel_samples_us"] = ks
                rec["module_samples_us"] = ms
                # extra capture kept for offline trace analysis
                d = OUT / "xplane" / c["id"]
                with jax.profiler.trace(str(d), create_perfetto_link=False):
                    for _ in range(2):
                        L.step(cand)
                rec["trace_lines"] = {
                    k: {kk: vv for kk, vv in v.items() if kk != "top_names"}
                    for k, v in extract_events(d, c["id"]).items()
                }
                print(
                    "CASE",
                    MODE,
                    c["id"],
                    "sel_eq",
                    rec["select_equals_expected"],
                    "kernel",
                    round(rec["kernel"]["median"], 3),
                    "module",
                    round(rec["module"]["median"], 3),
                    "n",
                    rec["kernel"]["n"],
                    "P",
                    states,
                    "llo",
                    len(rec.get("llo_files", [])),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"[:3000]
                rec["traceback"] = traceback.format_exc()[-4000:]
                print("CASE FAILED", MODE, c["id"], str(exc)[:400], flush=True)
            save()
    print("ATTR DONE", MODE, flush=True)


if __name__ == "__main__":
    main()
