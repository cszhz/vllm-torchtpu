# SPDX-License-Identifier: Apache-2.0
"""Where a small two-shot all-reduce's time goes: chains of kernels that stop
after the barrier, after round one, or run in full (8 TPU cores)."""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives.all_reduce import (
    small_all_reduce,
)
from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    barrier,
    call,
    push,
    split_refs,
    sum_slots,
    wait_bytes_of,
)

AX = "d"
CHAIN = 64


def phased(x, n, rank, peers, stop, chunks=1):
    rows, hidden = x.shape
    own = hidden // n
    lanes = own // 128
    bounds = [
        (
            lanes * k // chunks * 128,
            (lanes * (k + 1) // chunks - lanes * k // chunks) * 128,
        )
        for k in range(chunks)
    ]

    def pushc(src, dst, ss, rs, j, group):
        for a, w in bounds:
            push(src.at[:, pl.ds(a, w)], dst.at[:, pl.ds(a, w)], ss, rs, j, group)

    def kernel(*refs):
        me, group, refs = split_refs(refs, n, AX, True, True)
        x_ref, o_ref, mine, slices, gathered, send1, recv1, send2, recv2 = refs
        barrier(group)
        if stop == "barrier":
            o_ref[...] = x_ref[...]
            return
        for j in range(n):
            mine[j] = x_ref[:, pl.ds(j * own, own)]
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                pushc(mine.at[j], slices.at[me], send1, recv1, j, group)

        slices[pl.ds(me, 1)] = mine[pl.ds(me, 1)]
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv1)
        s = sum_slots(slices, n).astype(x.dtype)
        if stop == "round1":
            for j in range(n):
                o_ref[:, pl.ds(j * own, own)] = s
            wait_bytes_of(slices.at[pl.ds(0, n - 1)], send1)
            return
        gathered[pl.ds(me, 1)] = s[None]
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                pushc(gathered.at[me], gathered.at[me], send2, recv2, j, group)

        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], recv2)
        for j in range(n):
            o_ref[:, pl.ds(j * own, own)] = gathered[j]
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], send1)
        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], send2)

    scratch = [pltpu.VMEM((n, rows, own), x.dtype)] * 3 + [pltpu.SemaphoreType.DMA] * 4
    return call(
        kernel, x, rows, rank, peers, scratch, 5, f"phase_{stop}_c{chunks}_n{n}_r{rows}"
    )


def main():
    devs = jax.devices()
    nc = len(devs)
    mesh = Mesh(np.array(devs), (AX,))
    sh = NamedSharding(mesh, P(AX))
    for width, rows in [(8, 8), (8, 16), (8, 32), (4, 16), (4, 32)]:
        rank = jax.device_put(np.array([i % width for i in range(nc)], np.int32), sh)
        peers = jax.device_put(
            np.concatenate(
                [
                    np.arange(i // width * width, i // width * width + width)
                    for i in range(nc)
                ]
            ).astype(np.int32),
            sh,
        )
        x = jax.device_put(jnp.ones((nc * rows, 7168), jnp.bfloat16), sh)
        half = jnp.bfloat16(1.0 / width)
        steps = {
            "none": lambda xb, r, p: xb,
            "barrier": lambda xb, r, p: phased(xb, width, r, p, "barrier"),
            "round1": lambda xb, r, p: phased(xb, width, r, p, "round1"),
            "full": lambda xb, r, p: phased(xb, width, r, p, "full"),
            "full_c2": lambda xb, r, p: phased(xb, width, r, p, "full", 2),
            "full_c4": lambda xb, r, p: phased(xb, width, r, p, "full", 4),
            "full_c7": lambda xb, r, p: phased(xb, width, r, p, "full", 7),
            "lib_twoshot": lambda xb, r, p: small_all_reduce(
                xb, num_cores=width, axis=AX, rank=r, peers=p, form="twoshot"
            ),
            "lib_oneshot": lambda xb, r, p: small_all_reduce(
                xb, num_cores=width, axis=AX, rank=r, peers=p, form="oneshot"
            ),
        }
        times = {}
        for name, st in steps.items():

            def body(xb, r, p, st=st):
                for _ in range(CHAIN):
                    xb = st(xb, r, p) * half
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
            f"n{width} r{rows} us/call (chain op {b:.2f}): "
            + " ".join(f"{k}={v - b:.2f}" for k, v in times.items()),
            flush=True,
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
