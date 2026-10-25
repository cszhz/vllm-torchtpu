# SPDX-License-Identifier: Apache-2.0
"""Does XLA overlap a large collective with independent TensorCore work?
Per iteration: an all-reduce (or RS / AG) of bf16 [T, 7168] and an
independent bf16 matmul chain of about the same length; prints each alone and
both together (us per iteration)."""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

AX = "d"
IT = 10


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
    for n, T, K in [(8, 2048, 24), (4, 512, 24)]:
        groups = [list(range(g, g + n)) for g in range(0, nd, n)]
        kw = {} if n == nd else {"axis_index_groups": groups}
        x = jax.device_put(
            jax.random.normal(jax.random.key(0), (nd * T, 7168), jnp.bfloat16), sh
        )
        z = jax.device_put(
            jax.random.normal(jax.random.key(1), (nd * T, 2048), jnp.bfloat16), sh
        )
        w = jax.device_put(
            jax.random.normal(jax.random.key(2), (nd * 2048, 2048), jnp.bfloat16)
            * 0.02,
            sh,
        )
        s = jnp.bfloat16(1.0 / n)
        colls = {
            "AR": lambda c: lax.psum(c, AX, **kw) * s,
            "RS+AG": lambda c: (
                lax.all_gather(
                    lax.psum_scatter(c, AX, scatter_dimension=0, tiled=True, **kw),
                    AX,
                    axis=0,
                    tiled=True,
                    **kw,
                )
                * s
            ),
        }

        def mm(zc, wl):
            for _ in range(K):
                zc = jnp.tanh(zc @ wl)
            return zc

        for cname, coll in colls.items():

            def both(xb, zb, wb, mode):
                def body(i, carry):
                    c, zc = carry
                    if mode in ("coll", "both"):
                        c = coll(c)
                    if mode in ("mm", "both"):
                        zc = mm(zc, wb)
                    return c, zc

                return lax.fori_loop(0, IT, body, (xb, zb))

            res = {}
            for mode in ("coll", "mm", "both"):
                f = jax.jit(
                    jax.shard_map(
                        lambda a, b, c, mode=mode: both(a, b, c, mode),
                        mesh=mesh,
                        in_specs=(P(AX),) * 3,
                        out_specs=(P(AX), P(AX)),
                        check_vma=False,
                    )
                )
                res[mode] = tm(f, x, z, w)
            hidden = res["coll"] + res["mm"] - res["both"]
            print(
                f"TP{n} T{T} {cname}: coll {res['coll']:7.1f} mm {res['mm']:7.1f} both {res['both']:7.1f} -> hidden {hidden:7.1f} us",
                flush=True,
            )


if __name__ == "__main__":
    main()
