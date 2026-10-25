# SPDX-License-Identifier: Apache-2.0
"""Device timing harness: every TensorCore runs the same chain of kernel calls at once.

Each device holds `n_w` distinct weights; one loop iteration calls the kernel once per
weight on the loop-carried activation, so nothing is CSE'd or hoisted and the HBM sees
all cores streaming concurrently (as in serving). Per-call time = (loop - empty loop) /
(reps * n_w), wall clock, best of `trials`.
"""

from __future__ import annotations

import re
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

MESH = Mesh(np.array(jax.devices()), ("d",))
ND = len(jax.devices())
SH = NamedSharding(MESH, P("d"))


def rand(key, shape, dtype, scale=1.0):
    """Per-device random array: global shape (ND*shape[0], ...) sharded on axis 0."""
    gshape = (ND * shape[0], *shape[1:])

    @jax.jit
    def mk(k):
        v = jax.random.normal(k, gshape, jnp.float32) * scale
        return v.astype(dtype)

    return jax.jit(mk, out_shardings=SH)(key)


def _smap(f, n_in):
    return jax.shard_map(
        f, mesh=MESH, in_specs=(P("d"),) * n_in, out_specs=P("d"), check_vma=False
    )


_EMPTY: dict = {}


def _chain(kernel, per_w_args, reps, call_kernel):
    n_w = len(per_w_args)
    k = len(per_w_args[0])
    n_flat = n_w * k

    def dev(x, *flat_args):
        def body(_, x):
            acc = jnp.zeros((), jnp.float32)
            for i in range(n_w):
                if call_kernel:
                    y = jax.tree.leaves(kernel(x, *flat_args[i * k : (i + 1) * k]))[0]
                else:
                    y = x
                acc = acc + y[(0,) * y.ndim].astype(jnp.float32)
            return x + (acc * 1e-30).astype(x.dtype)

        return jax.lax.fori_loop(0, reps, body, x)

    return jax.jit(_smap(dev, 1 + n_flat))


def time_chain(kernel, x0, per_w_args, reps=40, trials=3, empty=True):
    """kernel(x, *args_i) -> y (any shape); per_w_args: list over weights of tuples of
    global sharded arrays. Returns microseconds per kernel call."""
    n_w = len(per_w_args)
    flat = [a for t in per_w_args for a in t]

    def run(f):
        jax.block_until_ready(f(x0, *flat))
        best = 1e30
        for _ in range(trials):
            t = time.perf_counter()
            jax.block_until_ready(f(x0, *flat))
            best = min(best, time.perf_counter() - t)
        return best

    t_full = run(_chain(kernel, per_w_args, reps, True))
    ekey = (x0.shape, x0.dtype, tuple((a.shape, a.dtype) for a in flat), reps)
    if empty and ekey not in _EMPTY:
        _EMPTY[ekey] = run(_chain(kernel, per_w_args, reps, False))
    t_empty = _EMPTY[ekey] if empty else 0.0
    return (t_full - t_empty) / (reps * n_w) * 1e6


def profile_chain(kernel, x0, per_w_args, reps=8, top=6):
    """Device-0 XLA-op times of the chain: [(op, calls per kernel call, us per kernel call)]."""
    calls = reps * len(per_w_args)
    flat = [a for t in per_w_args for a in t]
    rows = device_op_times(_chain(kernel, per_w_args, reps, True), x0, *flat, top=top)
    return [(name, c / calls, t / calls) for name, c, t in rows]


def first_device(a):
    """Device 0's shard of a sharded global array, as numpy."""
    return np.asarray(a.addressable_shards[0].data)


def run_on_devices(f, *args):
    """Run f per device (shard_map) once; returns the global output."""
    return jax.jit(_smap(f, len(args)))(*args)


def device_op_times(f, *args, trace_dir="/tmp/dense_prof", top=12):
    """Run f(*args) under the profiler; return [(op name, count, total us)] for device 0's
    XLA-op events, largest first. Exploratory: prints the plane/line names it walked."""
    import glob
    import shutil

    from jax.profiler import ProfileData

    shutil.rmtree(trace_dir, ignore_errors=True)
    jax.block_until_ready(f(*args))
    with jax.profiler.trace(trace_dir):
        jax.block_until_ready(f(*args))
    path = sorted(glob.glob(f"{trace_dir}/**/*.xplane.pb", recursive=True))[-1]
    pd = ProfileData.from_file(path)
    agg: dict = {}
    for plane in pd.planes:
        if plane.name != "/device:TPU:0":
            continue
        for line in plane.lines:
            if line.name != "XLA Ops":
                continue
            for ev in line.events:
                # "%fp8_skinny_x.12 = bf16[...] ..." -> "fp8_skinny_x": one row per kernel
                name = re.sub(r"\.\d+$", "", ev.name.split(" = ")[0].lstrip("%"))
                c, t = agg.get(name, (0, 0.0))
                agg[name] = (c + 1, t + ev.duration_ns / 1e3)
    if not agg:
        print("device_op_times: no events; planes:", [
            (p.name, [l.name for l in p.lines]) for p in pd.planes][:6])
    return sorted(((k, c, t) for k, (c, t) in agg.items()), key=lambda r: -r[2])[:top]
