# SPDX-License-Identifier: Apache-2.0
"""Does a neighbouring kernel's scoped-VMEM request decide whether XLA hides its
operand staging copies?  Chains of [busy kernel (vmem_limit_bytes = V) -> HCA
compressor projection] for V in {100 MiB, 16 MiB}, profiled.

    python tests/glue/vmem_pressure.py XPROF_DIR
"""

import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    proj_and_save_state as proj,
)

L, m = 6, 16


def _busy_kernel(x_ref, o_ref):
    def body(i, acc):
        return acc * 1.0001 + 1e-6

    o_ref[...] = jax.lax.fori_loop(0, 40, body, x_ref[...])


@functools.cache
def busy(vmem):
    return pl.pallas_call(
        _busy_kernel,
        out_shape=jax.ShapeDtypeStruct((m, 7168), jnp.float32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem),
        name=f"busy_v{vmem >> 20}",
    )


def chain(vmem, h, ws, apes, caches, slot, pos):
    outs = []
    for w, a, c in zip(ws, apes, caches):
        t = busy(vmem)(h.astype(jnp.float32))
        h = t.astype(jnp.bfloat16)
        c = proj.proj_and_save_state(h, w, a, pos, slot, c, compress_ratio=128)
        outs.append(c)
        h = (
            h.astype(jnp.float32)
            + 1e-3 * c.reshape(c.shape[0], -1)[:m, :1].astype(jnp.float32)
        ).astype(jnp.bfloat16)
    return outs


def main(xprof):
    key = iter(jax.random.split(jax.random.PRNGKey(0), 100))
    ws = [
        (0.02 * jax.random.normal(next(key), (7168, 1024))).astype(jnp.bfloat16)
        for _ in range(L)
    ]
    apes = [jax.random.normal(next(key), (128, 512)) for _ in range(L)]
    tok = np.arange(m)
    slot = jnp.asarray(np.where(tok < 7, (tok * 5 % 64) * 256 + tok * 8, -1), jnp.int32)
    pos = jnp.asarray(1000 + 37 * tok, jnp.int32)
    h0 = jax.random.normal(next(key), (m, 7168)).astype(jnp.bfloat16)
    fns = {}
    for v in (100, 16):
        f = jax.jit(functools.partial(chain, v << 20), donate_argnums=(3,))
        caches = [jnp.zeros((64, 256, 128), jnp.int32) for _ in range(L)]
        fns[v] = [f, caches]

    def run(v):
        f, caches = fns[v]
        fns[v][1] = jax.block_until_ready(f(h0, ws, apes, caches, slot, pos))

    for v in fns:
        for _ in range(3):
            run(v)
        print("ok", v, flush=True)
    with jax.profiler.trace(xprof):
        for v in fns:
            for _ in range(5):
                run(v)


if __name__ == "__main__":
    main(sys.argv[1])
