# SPDX-License-Identifier: Apache-2.0
"""Does XLA overlap a large all-reduce with an independent Pallas kernel (the
prefill shape: the shared expert's AR next to the fused EP MoE kernel)?
K1 = a compute-only Pallas kernel, K2 = a Pallas kernel doing remote DMAs
(large all-gather over all cores). Per iteration: AR of bf16 [T, 7168] and the
kernel on independent data; prints each alone and both (us per iteration)."""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives import large

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


def compute_kernel(z, reps):
    def kernel(z_ref, o_ref):
        def body(i, acc):
            return jnp.tanh(acc * 1.0001 + 0.5)

        o_ref[...] = lax.fori_loop(
            0, reps, body, z_ref[...].astype(jnp.float32)
        ).astype(o_ref.dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(z.shape, z.dtype),
        grid=(z.shape[0] // 256,),
        in_specs=[pl.BlockSpec((256, z.shape[1]), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((256, z.shape[1]), lambda i: (i, 0)),
        name="compute_only",
    )(z)


def main():
    devs = jax.devices()
    nd = len(devs)
    mesh = Mesh(np.array(devs), (AX,))
    sh = NamedSharding(mesh, P(AX))
    rank = jax.device_put(np.arange(nd, dtype=np.int32), sh)
    peers = jax.device_put(np.tile(np.arange(nd, dtype=np.int32), nd), sh)
    for n, T in [(8, 2048), (4, 512)]:
        groups = [list(range(g, g + n)) for g in range(0, nd, n)]
        kw = {} if n == nd else {"axis_index_groups": groups}
        x = jax.device_put(
            jax.random.normal(jax.random.key(0), (nd * T, 7168), jnp.bfloat16), sh
        )
        z = jax.device_put(
            jax.random.normal(jax.random.key(1), (nd * 1024, 1024), jnp.bfloat16), sh
        )
        g = jax.device_put(
            jax.random.normal(jax.random.key(2), (nd * 256, 7168), jnp.bfloat16), sh
        )
        s = jnp.bfloat16(1.0 / n)
        kernels = {
            "K1_compute": lambda zc, gc, r, p: (compute_kernel(zc, 400), gc),
            "K2_remote": lambda zc, gc, r, p: (
                zc,
                large.all_gather(gc, num_cores=nd, axis=AX, rank=r, peers=p)[:256]
                * jnp.bfloat16(0.5),
            ),
        }
        pk = dict(num_cores=n, axis=AX)
        rank_g = jax.device_put(np.array([i % n for i in range(nd)], np.int32), sh)
        peers_g = jax.device_put(
            np.concatenate([groups[i // n] for i in range(nd)]).astype(np.int32), sh
        )

        def ag(part):
            return lax.all_gather(part, AX, axis=0, tiled=True, **kw)

        for kname, kern in kernels.items():
            res = {}
            for mode in ("ar", "k", "both", "async_ar", "async_both", "async_rs_both"):

                def f(xb, zb, gb, r, p, rg, pg, mode=mode):
                    def body(i, carry):
                        c, zc, gc = carry
                        if mode in ("ar", "both"):
                            c = lax.psum(c, AX, **kw) * s
                        if mode in ("k", "both"):
                            zc, gc = kern(zc, gc, r, p)
                        if mode == "async_ar":
                            h = large.reduce_scatter_start(c, rank=rg, peers=pg, **pk)
                            c = ag(large.reduce_scatter_done(h)) * s
                        if mode in ("async_both", "async_rs_both"):
                            h = large.reduce_scatter_start(c, rank=rg, peers=pg, **pk)
                            zc, _ = lax.optimization_barrier((zc, h[2]))
                            zc, gc = kern(zc, gc, r, p)
                            part = large.reduce_scatter_done(h, after=zc)
                            if mode == "async_both":
                                c = ag(part) * s
                            else:
                                c = jnp.concatenate([part * s] * n, 0)
                        return c, zc, gc

                    return lax.fori_loop(0, IT, body, (xb, zb, gb))

                ff = jax.jit(
                    jax.shard_map(
                        f,
                        mesh=mesh,
                        in_specs=(P(AX),) * 7,
                        out_specs=(P(AX),) * 3,
                        check_vma=False,
                    )
                )
                res[mode] = tm(ff, x, z, g, rank, peers, rank_g, peers_g)
            print(
                f"TP{n} T{T} + {kname}: "
                + " ".join(f"{k}={v:7.1f}" for k, v in res.items())
                + f" | AR exposed sync {res['both'] - res['k']:7.1f} async {res['async_both'] - res['k']:7.1f} (RS only {res['async_rs_both'] - res['k']:7.1f})",
                flush=True,
            )

        # numerics: async RS == sync RS
        def cmp(xb, rg, pg):
            a1 = large.reduce_scatter(xb, rank=rg, peers=pg, **pk)
            a2 = large.reduce_scatter_done(
                large.reduce_scatter_start(xb, rank=rg, peers=pg, **pk)
            )
            return a1, a2

        a1, a2 = jax.jit(
            jax.shard_map(
                cmp,
                mesh=mesh,
                in_specs=(P(AX),) * 3,
                out_specs=(P(AX), P(AX)),
                check_vma=False,
            )
        )(x, rank_g, peers_g)
        print(
            "async RS == sync RS bitexact",
            bool(
                np.array_equal(
                    np.asarray(a1).view(np.uint16), np.asarray(a2).view(np.uint16)
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
