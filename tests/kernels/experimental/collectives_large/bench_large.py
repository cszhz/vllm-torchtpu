# SPDX-License-Identifier: Apache-2.0
"""Pallas large reduce-scatter / all-gather / all-reduce vs XLA's at the
prefill shapes (bf16 [T, 7168] per core, TP groups of one host), with
bit-exactness checks: RS against the float32 group-order sum, AG against XLA's
all-gather, AR members agreeing."""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives import large

AX = "d"
H = 7168
IT = 10
SHAPES = [(8, 2048), (4, 512), (4, 128)]
CHUNKS = (2, 4, 8, 16)


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
        rank = jax.device_put(np.array([i % n for i in range(nd)], np.int32), sh)
        peers = jax.device_put(
            np.concatenate([groups[i // n] for i in range(nd)]).astype(np.int32), sh
        )
        x = jax.device_put(
            jax.random.normal(jax.random.key(0), (nd * T, H), jnp.bfloat16), sh
        )
        xs = jax.device_put(
            jax.random.normal(jax.random.key(1), (nd * T // n, H), jnp.bfloat16), sh
        )
        s = jnp.bfloat16(1.0 / n)
        pk = dict(num_cores=n, axis=AX)

        def sm(body, nin=3):
            return jax.jit(
                jax.shard_map(
                    body,
                    mesh=mesh,
                    in_specs=(P(AX),) * nin,
                    out_specs=P(AX),
                    check_vma=False,
                )
            )

        def loop(step):
            def f(b, r, p):
                return lax.fori_loop(0, IT, lambda i, c: step(c, r, p), b)

            return sm(f)

        base = tm(loop(lambda c, r, p: c * s), x, rank, peers)
        base_s = tm(loop(lambda c, r, p: c * s), xs, rank, peers)
        tile = tm(
            loop(lambda c, r, p: jnp.concatenate([c[: T // n] * s] * n, 0)),
            x,
            rank,
            peers,
        )
        res = {}
        res["xla_RS"] = (
            tm(
                loop(
                    lambda c, r, p: jnp.concatenate(
                        [
                            lax.psum_scatter(
                                c, AX, scatter_dimension=0, tiled=True, **kw
                            )
                            * s
                        ]
                        * n,
                        0,
                    )
                ),
                x,
                rank,
                peers,
            )
            - tile
        )
        res["xla_AG"] = (
            tm(
                loop(
                    lambda c, r, p: (
                        lax.all_gather(c, AX, axis=0, tiled=True, **kw)[: T // n] * s
                    )
                ),
                xs,
                rank,
                peers,
            )
            - base_s
        )
        for ch in CHUNKS:
            res[f"pl_RS_c{ch}"] = (
                tm(
                    loop(
                        lambda c, r, p, ch=ch: jnp.concatenate(
                            [
                                large.reduce_scatter(
                                    c, rank=r, peers=p, chunks=ch, **pk
                                )
                                * s
                            ]
                            * n,
                            0,
                        )
                    ),
                    x,
                    rank,
                    peers,
                )
                - tile
            )
            res[f"pl_AG_c{ch}"] = (
                tm(
                    loop(
                        lambda c, r, p, ch=ch: (
                            large.all_gather(c, rank=r, peers=p, chunks=ch, **pk)[
                                : T // n
                            ]
                            * s
                        )
                    ),
                    xs,
                    rank,
                    peers,
                )
                - base_s
            )
        # numerics
        rs = sm(lambda c, r, p: large.reduce_scatter(c, rank=r, peers=p, **pk))(
            x, rank, peers
        )
        xn = np.asarray(x.astype(jnp.float32)).reshape(nd, T, H)
        ok_rs = True
        rsn = np.asarray(rs.astype(jnp.float32)).reshape(nd, T // n, H)
        for g in groups:
            acc = np.zeros((T, H), np.float32)
            for m in g:
                acc = acc + xn[m]
            ref = np.asarray(jnp.asarray(acc).astype(jnp.bfloat16).astype(jnp.float32))
            for slot, m in enumerate(g):
                ok_rs &= np.array_equal(
                    rsn[m], ref[slot * (T // n) : (slot + 1) * (T // n)]
                )
        ag_p = sm(lambda c, r, p: large.all_gather(c, rank=r, peers=p, **pk))(
            xs, rank, peers
        )
        ag_x = sm(lambda c, r, p: lax.all_gather(c, AX, axis=0, tiled=True, **kw))(
            xs, rank, peers
        )
        ok_ag = bool(
            np.array_equal(
                np.asarray(ag_p).view(np.uint16), np.asarray(ag_x).view(np.uint16)
            )
        )
        ar = (
            np.asarray(
                sm(lambda c, r, p: large.all_reduce(c, rank=r, peers=p, **pk))(
                    x, rank, peers
                )
            )
            .view(np.uint16)
            .reshape(nd, T, H)
        )
        ok_ar = all(np.array_equal(ar[g[0]], ar[m]) for g in groups for m in g)
        print(
            f"TP{n} T{T}: "
            + " ".join(f"{k}={v:7.1f}" for k, v in res.items())
            + f" | RS==f32 group-order sum {ok_rs}, AG==XLA {ok_ag}, AR members agree {ok_ar}",
            flush=True,
        )


if __name__ == "__main__":
    main()
