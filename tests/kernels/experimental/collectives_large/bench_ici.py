# SPDX-License-Identifier: Apache-2.0
"""ICI calibration on one host (8 TensorCores): pairwise HBM->remote-HBM DMA
throughput for partner = me XOR m, with k DMAs in flight per core, and the
device coordinates. Prints GB/s per core (one direction)."""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

AX = "d"


def pair_kernel_fn(nbytes, mask, k, nc, reps):
    rows = nbytes // (2 * 7168)
    chunk = rows // k // 16 * 16

    def kernel(x_ref, o_ref, send, recv):
        me = lax.axis_index(AX)
        peer = jnp.bitwise_xor(me, mask)
        sem = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(
            sem, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
        )
        pl.semaphore_wait(sem, 1)

        def body(r, c):
            cps = []
            for j in range(k):
                cp = pltpu.make_async_remote_copy(
                    src_ref=x_ref.at[pl.ds(j * chunk, chunk)],
                    dst_ref=o_ref.at[pl.ds(j * chunk, chunk)],
                    send_sem=send,
                    recv_sem=recv,
                    device_id=(peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                cp.start()
                cps.append(cp)
            for cp in cps:
                cp.wait()
            return c

        lax.fori_loop(0, reps, body, 0)

    def f(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA],
            compiler_params=pltpu.CompilerParams(
                collective_id=3, has_side_effects=True
            ),
        )(x)

    return f, rows


def a2a_kernel_fn(nbytes, nc, reps, k=1):
    rows = nbytes // (2 * 7168)
    per = rows // nc

    def kernel(x_ref, o_ref, send, recv):
        me = lax.axis_index(AX)
        sem = pltpu.get_barrier_semaphore()
        for j in range(nc):
            pl.semaphore_signal(
                sem, 1, device_id=(jnp.int32(j),), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(sem, nc)

        def body(r, c):
            cps = []
            step = per // k // 16 * 16
            for j in range(1, nc):
                peer = lax.rem(me + j, nc)
                for q in range(k):
                    cp = pltpu.make_async_remote_copy(
                        src_ref=x_ref.at[pl.ds(peer * per + q * step, step)],
                        dst_ref=o_ref.at[pl.ds(me * per + q * step, step)],
                        send_sem=send,
                        recv_sem=recv,
                        device_id=(peer,),
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    cp.start()
                    cps.append(cp)
            for cp in cps:
                cp.wait()
            return c

        lax.fori_loop(0, reps, body, 0)

    def f(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA],
            compiler_params=pltpu.CompilerParams(
                collective_id=3, has_side_effects=True
            ),
        )(x)

    return f, rows


def main():
    devs = jax.devices()
    nc = len(devs)
    for d in devs:
        print(
            "device",
            d.id,
            getattr(d, "coords", None),
            getattr(d, "core_on_chip", None),
            flush=True,
        )
    mesh = Mesh(np.array(devs), (AX,))
    sh = NamedSharding(mesh, P(AX))
    for nbytes in (4 << 20, 16 << 20):
        for mask in (1, 2, 4, 3, 6, 7):
            for k in (1, 4):
                reps = 10
                f, rows = pair_kernel_fn(nbytes, mask, k, nc, reps)
                x = jax.device_put(jnp.ones((nc * rows, 7168), jnp.bfloat16), sh)
                g = jax.jit(
                    jax.shard_map(
                        f, mesh=mesh, in_specs=P(AX), out_specs=P(AX), check_vma=False
                    )
                )
                jax.block_until_ready(g(x))
                best = 1e9
                for _ in range(5):
                    t0 = time.perf_counter()
                    jax.block_until_ready(g(x))
                    best = min(best, time.perf_counter() - t0)
                gbs = nbytes * reps / best / 1e9
                print(
                    f"pair xor{mask} {nbytes >> 20}MiB k={k}: {best / reps * 1e6:8.1f} us/transfer {gbs:7.1f} GB/s per core",
                    flush=True,
                )


def main_a2a():
    devs = jax.devices()
    nc = len(devs)
    mesh = Mesh(np.array(devs), (AX,))
    sh = NamedSharding(mesh, P(AX))
    for nbytes, k in [
        (7 << 20, 1),
        (7 << 20, 4),
        (7 << 20, 16),
        (28 << 20, 1),
        (28 << 20, 4),
        (28 << 20, 16),
    ]:
        reps = 10
        f, rows = a2a_kernel_fn(nbytes, nc, reps, k)
        x = jax.device_put(jnp.ones((nc * rows, 7168), jnp.bfloat16), sh)
        g = jax.jit(
            jax.shard_map(
                f, mesh=mesh, in_specs=P(AX), out_specs=P(AX), check_vma=False
            )
        )
        jax.block_until_ready(g(x))
        best = 1e9
        for _ in range(5):
            t0 = time.perf_counter()
            jax.block_until_ready(g(x))
            best = min(best, time.perf_counter() - t0)
        sent = nbytes * (nc - 1) / nc
        print(
            f"a2a {nbytes >> 20}MiB block: {best / reps * 1e6:8.1f} us, egress {sent * reps / best / 1e9:7.1f} GB/s per core",
            flush=True,
        )


if __name__ == "__main__":
    main_a2a()
    main()
