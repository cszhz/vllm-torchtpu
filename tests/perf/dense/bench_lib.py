# SPDX-License-Identifier: Apache-2.0
"""Device timing harness: every TensorCore runs the same chain of kernel calls at once.

Each device holds `n_w` distinct weights; one loop iteration calls the kernel once per
weight on the loop-carried activation, so nothing is CSE'd or hoisted and the HBM sees
all cores streaming concurrently (as in serving). Per-call time = (loop - empty loop) /
(reps * n_w), wall clock, best of `trials`.
"""

from __future__ import annotations

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


def time_chain(kernel, x0, per_w_args, reps=40, trials=3, empty=True):
    """kernel(x, *args_i) -> y (any shape); per_w_args: list over weights of tuples of
    global sharded arrays. Returns microseconds per kernel call."""
    n_w = len(per_w_args)
    flat = [a for t in per_w_args for a in t]
    k = len(per_w_args[0])

    def make(call_kernel):
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

        return jax.jit(_smap(dev, 1 + len(flat)))

    def run(f):
        jax.block_until_ready(f(x0, *flat))
        best = 1e30
        for _ in range(trials):
            t = time.perf_counter()
            jax.block_until_ready(f(x0, *flat))
            best = min(best, time.perf_counter() - t)
        return best

    t_full = run(make(True))
    ekey = (x0.shape, x0.dtype, tuple((a.shape, a.dtype) for a in flat), reps)
    if empty and ekey not in _EMPTY:
        _EMPTY[ekey] = run(make(False))
    t_empty = _EMPTY[ekey] if empty else 0.0
    return (t_full - t_empty) / (reps * n_w) * 1e6


def first_device(a):
    """Device 0's shard of a sharded global array, as numpy."""
    return np.asarray(a.addressable_shards[0].data)


def run_on_devices(f, *args):
    """Run f per device (shard_map) once; returns the global output."""
    return jax.jit(_smap(f, len(args)))(*args)
