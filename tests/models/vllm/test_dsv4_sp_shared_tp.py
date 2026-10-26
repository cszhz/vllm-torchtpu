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
"""TPU_DSV4_SP_SHARED_TP: the sequence-parallel shared expert on TP slices.

Each core of the local mesh is one rank holding its own sequence-parallel rows
and its TP slice of the DSv4-Pro shared expert (hidden 7168, intermediate 3072,
fp8 per-channel). The op (all-gather rows over the TP group, sliced MLP,
reduce-scatter) is compared against
  * ``nosp``: the replicated layout the stack runs without sequence
    parallelism -- every TP rank runs its slice on the whole TP group's rows,
    the bf16 partials are summed (psum), each rank keeps its own rows. The op
    must match it (same matmuls on the same rows, same bf16 partial sum).
  * ``full``: what TPU_DSV4_SEQ_PARALLEL computes today -- the whole
    replicated weight on this rank's rows -- and a float64 oracle; the op's
    error against the oracle must stay in the class of ``full``'s and
    ``nosp``'s.
A positive control hands the op the slices in the wrong TP order and must fail.

    python3 tests/models/vllm/test_dsv4_sp_shared_tp.py [--time]
"""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from vllm_torchtpu.models.vllm.deepseek_v4.layers import (
    shared_mlp_fn,
    sp_tp_shared_fn,
)

pytestmark = [pytest.mark.tpu_test]

AX = "ep"
H, N = 7168, 3072
LIMIT = 10.0


def _mesh():
    devs = np.array(jax.devices())
    if devs.size < 4:
        pytest.skip("needs >= 4 cores")
    return Mesh(devs[:8] if devs.size >= 8 else devs[:4], (AX,))


def _weights(seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    w = (jax.random.normal(k[0], (H, 2 * N), jnp.float32) * 0.5).astype(
        jnp.float8_e4m3fn
    )
    ws = jax.random.uniform(k[1], (2 * N,), jnp.float32, 0.01, 0.03)
    d = (jax.random.normal(k[2], (N, H), jnp.float32) * 0.5).astype(jnp.float8_e4m3fn)
    ds = jax.random.uniform(k[3], (H,), jnp.float32, 0.01, 0.03)
    return w, ws, d, ds


def _slice(w, ws, d, ds, r, tp):
    """arm_tp_slice's slicing: gate and up columns, down rows; per-channel
    down scale whole."""
    s = N // tp
    cols = np.r_[r * s : (r + 1) * s, N + r * s : N + (r + 1) * s]
    return w[:, cols], ws[cols], d[r * s : (r + 1) * s], ds


def _stack(mesh, parts):
    """One per-core block per device -> a global array sharded on axis 0."""
    arr = jnp.concatenate(parts, 0)
    spec = PartitionSpec(AX, *([None] * (arr.ndim - 1)))
    return jax.device_put(arr, NamedSharding(mesh, spec))


def _oracle(x, w, ws, d, ds):
    x = np.asarray(x, np.float64)
    h = x @ (np.asarray(w.astype(jnp.float32), np.float64) * np.asarray(ws, np.float64))
    g, u = h[:, :N], h[:, N:]
    g = np.minimum(g, LIMIT)
    u = np.clip(u, -LIMIT, LIMIT)
    a = g / (1 + np.exp(-g)) * u
    return a @ (np.asarray(d.astype(jnp.float32), np.float64) * np.asarray(ds, np.float64))


def _rel(a, ref):
    a = np.asarray(a, np.float64)
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


def _run(tp, n_local, mode, pin, wrong_order=False, timing=False):
    mesh = _mesh()
    ndev = mesh.devices.size
    groups = [list(range(g * tp, (g + 1) * tp)) for g in range(ndev // tp)]
    w, ws, d, ds = _weights()
    x = jax.random.normal(jax.random.PRNGKey(7), (ndev * n_local, H), jnp.float32)
    x = (x * 2.0).astype(jnp.bfloat16)

    def rank_of(dev):
        r = dev % tp
        return (tp - 1 - r) if wrong_order else r

    sl = [_slice(w, ws, d, ds, rank_of(dv), tp) for dv in range(ndev)]
    args = [_stack(mesh, [x[dv * n_local : (dv + 1) * n_local] for dv in range(ndev)])]
    for i in range(4):
        args.append(_stack(mesh, [s[i] for s in sl]))
    fn = jax.jit(sp_tp_shared_fn(mesh, AX, groups, mode, pin, pin, LIMIT))
    out = np.asarray(fn(*args).astype(jnp.float32))

    mlp = jax.jit(shared_mlp_fn(pin, pin, LIMIT))
    full = np.concatenate(
        [
            np.asarray(mlp(x[i * n_local : (i + 1) * n_local], w, ws, d, ds), np.float32)
            for i in range(ndev)
        ]
    )
    # nosp: per TP group, each rank's slice on the group's rows, bf16 partials
    # summed in rank order as psum's reduction (f32 accumulate, bf16 result).
    nosp = np.zeros_like(full)
    for g in groups:
        rows = np.r_[g[0] * n_local : (g[-1] + 1) * n_local]
        xg = x[rows]
        parts = [
            mlp(xg, *_slice(w, ws, d, ds, r, tp)).astype(jnp.float32) for r in range(tp)
        ]
        nosp[rows] = np.asarray(sum(parts).astype(jnp.bfloat16).astype(jnp.float32))
    ref = _oracle(np.asarray(x.astype(jnp.float32)), w, ws, d, ds)
    res = dict(
        err_op=_rel(out, ref),
        err_full=_rel(full, ref),
        err_nosp=_rel(nosp, ref),
        op_vs_nosp=_rel(out, np.asarray(nosp, np.float64)),
        op_vs_nosp_maxabs=float(np.max(np.abs(out - nosp))),
        ref_absmax=float(np.max(np.abs(ref))),
    )
    if timing:
        jfull = jax.jit(
            jax.shard_map(
                lambda xl, *_: mlp(xl, w, ws, d, ds),
                mesh=mesh,
                in_specs=(PartitionSpec(AX, None),),
                out_specs=PartitionSpec(AX, None),
                check_vma=False,
            )
        )
        for name, f, a in (("op", fn, args), ("full", jfull, args[:1])):
            f(*a).block_until_ready()
            t0 = time.perf_counter()
            for _ in range(20):
                o = f(*a)
            o.block_until_ready()
            res[f"t_{name}_us"] = (time.perf_counter() - t0) / 20 * 1e6
    return res


CASES = [
    # (tp, rows per rank): c4/c8 DP2xTP8 T128/T256/T2048 and c16 DP4xTP4-ish
    (8, 16),
    (8, 32),
    (8, 256),
    (4, 128),
]


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("pin", [None, True])
@pytest.mark.parametrize("tp,n_local", CASES)
def test_sp_shared_tp_matches_nosp(tp, n_local, mode, pin):
    if jax.device_count() < tp:
        pytest.skip("needs a TP group of cores")
    r = _run(tp, n_local, mode, pin)
    print(tp, n_local, mode, pin, r)
    # Same rows, same slices, same bf16 partial sum as the replicated layout.
    assert r["op_vs_nosp"] < 2e-3, r
    # Error class: no worse than today's SP (full weight) or the no-SP layout.
    assert r["err_op"] <= 1.10 * max(r["err_full"], r["err_nosp"]) + 1e-4, r


def test_positive_control_wrong_tp_order_fails():
    r = _run(8 if jax.device_count() >= 8 else 4, 32, 1, None, wrong_order=True)
    print("wrong order", r)
    assert r["op_vs_nosp"] > 0.1, r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--time", action="store_true")
    a = p.parse_args()
    bad = 0
    for tp, n in CASES:
        for mode in (1, 2):
            for pin in (None, True):
                if jax.device_count() < tp:
                    continue
                r = _run(tp, n, mode, pin, timing=a.time)
                ok = r["op_vs_nosp"] < 2e-3 and r["err_op"] <= 1.10 * max(
                    r["err_full"], r["err_nosp"]
                ) + 1e-4
                bad += not ok
                print(f"tp{tp} rows{n} mode{mode} pin{pin} {'PASS' if ok else 'FAIL'} {r}")
    r = _run(8 if jax.device_count() >= 8 else 4, 32, 1, None, wrong_order=True)
    ok = r["op_vs_nosp"] > 0.1
    bad += not ok
    print(f"positive control {'PASS' if ok else 'FAIL'} {r}")
    sys.exit(1 if bad else 0)
