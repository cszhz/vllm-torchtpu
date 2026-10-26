# SPDX-License-Identifier: Apache-2.0
"""Accuracy gate for the mixed-step routing plan (and combine).

Three parts (write-tpu-kernels 1.3):
  oracle    -- spec.oracle_plan, first-principles numpy int64 (the plan is integer
               bookkeeping, so the oracle is exact, not float64-approximate);
               spec.oracle_combine, float64.
  native    -- spec.incumbent_plan / incumbent_combine, the production code path.
  faults    -- spec.FAULTS injected into the oracle; each must turn the comparison
               red. Catch rate is reported per cell class; a fault blind on every
               production class (served, served_pad) fails the gate.
Candidates register in IMPLS (after the freeze gate) and must match the oracle
exactly on every cell. Runs on CPU (Pallas in interpret mode) and on TPU.
"""

import collections
import os
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec  # noqa: E402



def _on_tpu():
    # Not at import: touching the backend during collection is refused by conftest.
    return jax.default_backend() == "tpu"

# name -> fn(cell, exchange, local_blocks, block_hist, me, interpret) -> dict
IMPLS = {
    "incumbent": lambda c, e, l, b, m, interpret: spec.incumbent_plan(c, e, l, b, m),
    "a1": lambda c, e, l, b, m, interpret: spec.candidate_plan(c, e, l, b, m, interpret),
}

EDGE = [
    ("mix32", "one_shard"),
    ("mix32", "unrouted"),
    ("mix64", "hot"),
    ("mix64", "served_pad"),
    ("mix256", "dup"),
    ("mix256", "one_shard"),
    ("mix128", "unrouted"),
    ("tw16", "uniform"),
    ("tw16", "unrouted"),
    ("tw4", "uniform"),
    ("tw4", "hot"),
]
PROD = [(c, k) for c in spec.PROD_CELLS for k in ("served", "served_pad", "uniform")]
FAST = os.environ.get("KPLAN_FAST") == "1"


def _case(cn, kind, seed=1):
    cell = spec.CELLS[cn]
    pairs = spec.make_pairs(cell, kind, seed)
    sb = spec.make_scale_bits(cell, seed)
    ex = spec.make_exchange(cell, pairs, sb)
    mes = sorted({0, cell.ep - 1, spec.busiest_shard(cell, pairs)})
    return cell, pairs, sb, ex, mes


def _run(impl, cell, pairs, ex, me):
    lb, bh = spec.local_inputs(cell, pairs, me)
    f = jax.jit(lambda e, l, b, m: IMPLS[impl](cell, e, l, b, m, not _on_tpu()))
    return jax.device_get(f(ex, lb, bh, jnp.int32(me)))


@pytest.mark.parametrize("impl", list(IMPLS))
@pytest.mark.parametrize("cn,kind", PROD + EDGE)
def test_plan_matches_oracle(impl, cn, kind):
    cell, pairs, sb, ex, mes = _case(cn, kind)
    for me in mes[:1] if FAST else mes:
        got = _run(impl, cell, pairs, ex, me)
        want = spec.oracle_plan(cell, pairs, sb, me)
        bad = spec.compare(want, got)
        assert not bad, f"{impl} {cn} {kind} me={me}: {bad}"


CLASSES = ("served", "served_pad", "uniform", "hot", "one_shard", "unrouted")


def test_fault_control_catch_rate():
    """Positive control: every fault must be caught on a production class."""
    rate = collections.defaultdict(dict)
    for kind in CLASSES:
        cn = "mix128" if kind in ("served", "served_pad", "uniform") else "mix32"
        cell, pairs, sb, ex, mes = _case(cn, kind)
        # Every shard the plan cells use: a fault invisible from one shard (no
        # routed pairs of its own, no experts with rows) is not blind.
        refs = {me: _run("incumbent", cell, pairs, ex, me) for me in mes}
        for f in spec.FAULTS:
            rate[f][kind] = any(
                bool(spec.compare(spec.oracle_plan(cell, pairs, sb, me, fault=f), refs[me]))
                for me in mes
            )
    lines = []
    blind_prod = []
    for f in spec.FAULTS:
        caught = [k for k in CLASSES if rate[f][k]]
        blind = [k for k in CLASSES if not rate[f][k]]
        lines.append(f"{f:13s} caught {len(caught)}/{len(CLASSES)} blind={blind}")
        if not (rate[f]["served"] or rate[f]["served_pad"]):
            blind_prod.append(f)
    print("\nFAULT CONTROL (mix128 served/served_pad/uniform, mix32 others)\n" + "\n".join(lines))
    assert not blind_prod, f"faults blind on every production class: {blind_prod}"


@pytest.mark.parametrize("cn", spec.PROD_CELLS)
def test_combine_native_vs_f64_and_faults(cn):
    if not _on_tpu():
        pytest.skip("combine reference runs the TPU path")
    cell = spec.CELLS[cn]
    pairs = spec.make_pairs(cell, "served_pad", 3)
    sb = spec.make_scale_bits(cell, 3)
    ex = spec.make_exchange(cell, pairs, sb)
    # Source 0: bucket padding sits in the tail sources, so source 0 routes.
    me = 0
    plan = _run("incumbent", cell, pairs, ex, me)
    arr, scl, w, routes = spec.make_combine_inputs(cell, pairs, me, 3)
    assert bool(np.asarray(routes).any()), "degenerate combine cell: no routed rows"
    pos, mir = jnp.asarray(plan["pos"]), jnp.asarray(plan["mirror_pos"])
    native = np.asarray(jax.jit(spec.incumbent_combine)(arr, scl, pos, mir, w, routes), np.float64)
    want = spec.oracle_combine(arr, scl, pos, mir, w, routes)
    assert np.abs(want).max() > 0, "degenerate combine cell: oracle output is zero"
    scale = np.maximum(np.abs(want).max(1, keepdims=True), 1e-30)
    err = np.abs(native - want) / scale
    tol = max(4 * float(err.max()), 2.0 ** -8)  # bf16 output rounding, calibrated on native
    print(f"\nCOMBINE {cn}: native max rel err {err.max():.3e}, tol {tol:.3e}")
    faults = {
        "pos+1": (pos + (pos < cell.recv_rows - 1), mir, w),
        "mirror_lane+1": (pos, mir + 1, w),
        "drop_k0": (pos, mir, w.at[:, 0].set(0.0)),
        "weight_missing": (pos, mir, jnp.ones_like(w)),
    }
    for name, (p2, m2, w2) in faults.items():
        bad = spec.oracle_combine(arr, scl, p2, m2, w2, routes)
        e2 = np.abs(bad - want) / scale
        assert e2.max() > tol, f"combine fault {name} not caught ({e2.max():.3e} <= {tol:.3e})"


@pytest.mark.parametrize("fault", list(spec.CANDIDATE_FAULTS))
def test_candidate_fault_caught(fault):
    """Positive control for the candidate: each kernel fault turns the oracle
    comparison red on a production cell (dup_nohi: on the duplicate-id edge
    cell, the only class where it can act)."""
    kw = spec.CANDIDATE_FAULTS[fault]
    cells = [("mix256", "dup")] if fault == "dup_nohi" else [
        ("mix128", "served"), ("mix128", "served_pad"), ("mix256", "served")]
    caught = []
    for cn, kind in cells:
        cell, pairs, sb, ex, mes = _case(cn, kind)
        for me in mes:
            lb, bh = spec.local_inputs(cell, pairs, me)
            f = jax.jit(lambda e, m: spec.candidate_plan(cell, e, None, None, m, not _on_tpu(), **kw))
            got = jax.device_get(f(ex, jnp.int32(me)))
            bad = spec.compare(spec.oracle_plan(cell, pairs, sb, me), got)
            if bad:
                caught.append((cn, kind, me, bad))
    print(f"\nCANDIDATE FAULT {fault}: caught on {len(caught)} (cell, shard) -> {caught[:3]}")
    assert caught, f"candidate fault {fault} not caught"
