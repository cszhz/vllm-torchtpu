# SPDX-License-Identifier: Apache-2.0
"""Decode-side XLA collectives under the current LIBTPU_INIT_ARGS: the MoE
merged exchange all-gather (fp8 [2, 7168] per core over all cores), the
drafter's vocab-parallel argmax gathers (f32 [rows, 2] over TP8), a small XLA
all-reduce and all-to-all, chained; us per call."""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

AX = "d"
IT = 32


def tm(f, *a):
    jax.block_until_ready(f(*a))
    b = 1e9
    for _ in range(7):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*a))
        b = min(b, time.perf_counter() - t0)
    return b / IT * 1e6


def main():
    devs = jax.devices()
    n = len(devs)
    mesh = Mesh(np.array(devs), (AX,))
    sh = NamedSharding(mesh, P(AX))

    def loop(step):
        def f(b):
            return lax.fori_loop(0, IT, lambda i, c: step(c), b)

        return jax.jit(
            jax.shard_map(
                f, mesh=mesh, in_specs=P(AX), out_specs=P(AX), check_vma=False
            )
        )

    cases = {
        "exchange_ag_fp8_2x7168": (
            jnp.float8_e4m3fn,
            (2, 7168),
            lambda c: (
                lax.all_gather(c, AX, axis=0, tiled=True)[:2].astype(jnp.float32) * 0.5
            ).astype(c.dtype),
        ),
        "argmax_ag_f32_4x2": (
            jnp.float32,
            (4, 2),
            lambda c: lax.all_gather(c, AX, axis=1, tiled=True)[:, :2] * 0.5,
        ),
        "argmax_ag_f32_16x2": (
            jnp.float32,
            (16, 2),
            lambda c: lax.all_gather(c, AX, axis=1, tiled=True)[:, :2] * 0.5,
        ),
        "psum_bf16_8x7168": (
            jnp.bfloat16,
            (8, 7168),
            lambda c: lax.psum(c, AX) * jnp.bfloat16(0.125),
        ),
        "psum_bf16_128x7168": (
            jnp.bfloat16,
            (128, 7168),
            lambda c: lax.psum(c, AX) * jnp.bfloat16(0.125),
        ),
        "psum_f32_8x128": (jnp.float32, (8, 128), lambda c: lax.psum(c, AX) * 0.125),
        "a2a_s32_8x256x16": (
            jnp.int32,
            (8, 256, 16),
            lambda c: lax.all_to_all(c, AX, 0, 0, tiled=True),
        ),
    }
    for name, (dt, shape, step) in cases.items():
        x = jax.device_put(jnp.ones((n * shape[0], *shape[1:]), dt), sh)
        base = tm(loop(lambda c: c), x)
        t = tm(loop(step), x)
        print(f"{name:26s} {t - base:8.2f} us", flush=True)


if __name__ == "__main__":
    main()
