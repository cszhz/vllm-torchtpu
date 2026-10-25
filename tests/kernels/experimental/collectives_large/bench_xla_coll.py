# SPDX-License-Identifier: Apache-2.0
"""XLA all-reduce / reduce-scatter / all-gather of bf16 [T, 7168] per core over
TP groups of one host, at the prefill shapes. LIBTPU_INIT_ARGS decides the
lowering. Prints us per call (loop overhead subtracted)."""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

AX = "d"
H = 7168
IT = 10
SHAPES = [(8, 2048), (8, 1024), (4, 512), (4, 256), (4, 128)]


def tm(f, *a):
    jax.block_until_ready(f(*a))
    b = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*a))
        b = min(b, time.perf_counter() - t0)
    return b / IT * 1e6


def main():
    devs = jax.devices()
    nd = len(devs)
    mesh = Mesh(np.array(devs), (AX,))
    sh = NamedSharding(mesh, P(AX))
    shapes = SHAPES
    if len(sys.argv) > 1:
        shapes = [tuple(int(v) for v in s.split("x")) for s in sys.argv[1:]]
    for n, T in shapes:
        groups = [list(range(g, g + n)) for g in range(0, nd, n)]
        kw = {} if n == nd else {"axis_index_groups": groups}
        x = jax.device_put(
            jax.random.normal(jax.random.key(0), (nd * T, H), jnp.bfloat16), sh
        )
        xs = jax.device_put(
            jax.random.normal(jax.random.key(1), (nd * T // n, H), jnp.bfloat16), sh
        )
        s = jnp.bfloat16(1.0 / n)

        def loop(body):
            def f(b):
                return lax.fori_loop(0, IT, lambda i, c: body(c), b)

            return jax.jit(
                jax.shard_map(
                    f, mesh=mesh, in_specs=P(AX), out_specs=P(AX), check_vma=False
                )
            )

        base = tm(loop(lambda c: c * s), x)
        base_s = tm(loop(lambda c: c * s), xs)
        ar = tm(loop(lambda c: lax.psum(c, AX, **kw) * s), x) - base
        rs = tm(
            loop(
                lambda c: jnp.concatenate(
                    [lax.psum_scatter(c, AX, scatter_dimension=0, tiled=True, **kw) * s]
                    * n,
                    0,
                )
            ),
            x,
        )
        tile = tm(loop(lambda c: jnp.concatenate([c[: T // n] * s] * n, 0)), x)
        ag = (
            tm(
                loop(
                    lambda c: (
                        lax.all_gather(c, AX, axis=0, tiled=True, **kw)[: T // n] * s
                    )
                ),
                xs,
            )
            - base_s
        )
        print(
            f"TP{n} T{T} ({T * H * 2 / 2**20:.1f} MiB): AR {ar:7.1f}  RS {rs - tile:7.1f}  AG {ag:7.1f}  us",
            flush=True,
        )


if __name__ == "__main__":
    main()
