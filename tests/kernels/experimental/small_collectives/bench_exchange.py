# SPDX-License-Identifier: Apache-2.0
"""Latency of the MoE token exchange all-gather (fp8 [rows, 7168] per core, all
cores): XLA's all-gather (SparseCore-offloaded or not, per LIBTPU_INIT_ARGS) vs
the Pallas small all-gather over the whole axis. Chains of dependent calls."""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives.all_gather import (
    small_all_gather,
)

AX = "d"
CHAIN = 32


def main():
    devs = jax.devices()
    n = len(devs)
    mesh = Mesh(np.array(devs), (AX,))
    sh = NamedSharding(mesh, P(AX))
    print("devices", n, "processes", jax.process_count(), flush=True)
    rank = jax.device_put(np.arange(n, dtype=np.int32), sh)
    peers = jax.device_put(np.tile(np.arange(n, dtype=np.int32), n), sh)
    for rows in [int(r) for r in (sys.argv[1:] or ["2", "4"])]:
        x = jax.device_put(jnp.ones((n * rows, 7168), jnp.float8_e4m3fn), sh)

        def xla_step(xb, r, p):
            g = lax.all_gather(xb, AX, axis=0, tiled=True)
            return g[:rows] if True else g

        def pallas_step(xb, r, p):
            u = lax.bitcast_convert_type(xb.reshape(rows, 1792, 4), jnp.uint32)
            g = small_all_gather(
                u, num_cores=n, axis=AX, rank=r, peers=p, collective_id=7
            )
            return lax.bitcast_convert_type(g[:rows], jnp.float8_e4m3fn).reshape(
                rows, 7168
            )

        def none_step(xb, r, p):
            return xb

        times = {}
        for name, st in [
            ("none", none_step),
            ("xla", xla_step),
            ("pallas", pallas_step),
        ]:

            def body(xb, r, p, st=st):
                for _ in range(CHAIN):
                    xb = (st(xb, r, p).astype(jnp.float32) * 0.5).astype(
                        jnp.float8_e4m3fn
                    )
                return xb

            f = jax.jit(
                jax.shard_map(
                    body,
                    mesh=mesh,
                    in_specs=(P(AX),) * 3,
                    out_specs=P(AX),
                    check_vma=False,
                )
            )
            jax.block_until_ready(f(x, rank, peers))
            best = 1e9
            for _ in range(7):
                t0 = time.perf_counter()
                jax.block_until_ready(f(x, rank, peers))
                best = min(best, time.perf_counter() - t0)
            times[name] = best / CHAIN * 1e6
        b = times.pop("none")
        print(
            f"n{n} rows{rows} us/call (chain op {b:.2f}): "
            + " ".join(f"{k}={v - b:.2f}" for k, v in times.items()),
            flush=True,
        )
        # correctness: pallas == xla
        fx = jax.jit(
            jax.shard_map(
                lambda xb, r, p: lax.all_gather(xb, AX, axis=0, tiled=True),
                mesh=mesh,
                in_specs=(P(AX),) * 3,
                out_specs=P(AX),
                check_vma=False,
            )
        )
        xr = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(rows), (n * rows, 7168)).astype(
                jnp.float8_e4m3fn
            ),
            sh,
        )

        def pg(xb, r, p):
            u = lax.bitcast_convert_type(xb.reshape(rows, 1792, 4), jnp.uint32)
            g = small_all_gather(
                u, num_cores=n, axis=AX, rank=r, peers=p, collective_id=7
            )
            return lax.bitcast_convert_type(g, jnp.float8_e4m3fn).reshape(
                n * rows, 7168
            )

        fp = jax.jit(
            jax.shard_map(
                pg, mesh=mesh, in_specs=(P(AX),) * 3, out_specs=P(AX), check_vma=False
            )
        )
        a = np.asarray(fx(xr, rank, peers).view(jnp.uint8))
        bb = np.asarray(fp(xr, rank, peers).view(jnp.uint8))
        print("bitexact", bool(np.array_equal(a, bb)), flush=True)


if __name__ == "__main__":
    main()
