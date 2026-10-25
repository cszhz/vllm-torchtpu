# SPDX-License-Identifier: Apache-2.0
"""fp8_skinny per-shape sweep at DeepSeek-V4 decode shapes (TP8 and TP4 shards).

Run: python tests/perf/dense/skinny_bench.py [--sweep] [--m=8] [--shape=tp8.wq_b]
Prints, per (K, N, M): HBM floor, XLA path, the tuned config, a config sweep, and two
diagnostic kernels with the tuned config's ring: DMA only (no dot) and dot only (no DMA).
"""

from __future__ import annotations

import functools
import itertools
import sys

import jax
import numpy as np
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_lib as bl  # noqa: E402

from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny  # noqa: E402
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul  # noqa: E402

HBM = 3.10e12
N_W = 16

SHAPES = {
    # name: (K, N)
    "tp8.wqa_wkv": (7168, 2048),
    "tp8.wq_b": (1536, 8192),
    "tp8.wo_b": (2048, 7168),
    "tp8.gate_up": (7168, 768),
    "tp8.down": (384, 7168),
    "tp4.wq_b": (1536, 16384),
    "tp4.wo_b": (4096, 7168),
    "tp4.gate_up": (7168, 1536),
    "tp4.down": (768, 7168),
}


def diag_kernel(x_ref, w_hbm, o_ref, wbuf, sem, *, t, n_tiles, nbuf, mode):
    """'dma': stream the K-split ring, no dot. 'dot': the dots, no DMA."""

    def cp(j):
        return pltpu.make_async_copy(
            w_hbm.at[pl.ds(j * t, t), :], wbuf.at[j % nbuf], sem.at[j % nbuf]
        )

    if mode == "dma":
        for j in range(min(nbuf, n_tiles)):
            cp(j).start()
        for j in range(n_tiles):
            cp(j).wait()
            if j + nbuf < n_tiles:
                cp(j + nbuf).start()
        o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)
        return
    xb = x_ref[...]
    if mode == "dot8":
        xb = xb.astype(jnp.float8_e4m3fn)
    acc = None
    dn = (((1,), (0,)), ((), ()))
    for j in range(n_tiles):
        y = jax.lax.dot_general(
            xb[:, j * t : (j + 1) * t], wbuf[j % nbuf], dn,
            preferred_element_type=jnp.float32,
        )
        acc = y if acc is None else acc + y
    o_ref[...] = acc.astype(o_ref.dtype)


@functools.cache
def build_diag(m, k, n, t, nbuf, mode):
    n_tiles = k // t
    nbuf = min(nbuf, n_tiles)
    return pl.pallas_call(
        functools.partial(diag_kernel, t=t, n_tiles=n_tiles, nbuf=nbuf, mode=mode),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        grid=(1,),
        in_specs=[
            pl.BlockSpec((m, k), lambda g: (0, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec((m, n), lambda g: (0, 0)),
        scratch_shapes=[
            pltpu.VMEM((nbuf, t, n), jnp.float8_e4m3fn),
            pltpu.SemaphoreType.DMA((nbuf,)),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 << 20),
        name=f"diag_{mode}_t{t}_b{nbuf}_m{m}_k{k}_n{n}",
    )


# Fastest config per shape from the sweep (M=8 at TP8, M=16 at TP4).
SWEEP_BEST = {
    (7168, 2048): ("k", 512, 8, 1),
    (1536, 8192): ("k", 128, 8, 1),
    (2048, 7168): ("k", 128, 8, 4),
    (7168, 768): ("k", 896, 8, 4),
    (384, 7168): ("n", 1792, 4, 1),
    (1536, 16384): ("k", 128, 4, 4),
    (4096, 7168): ("k", 256, 4, 4),
    (7168, 1536): ("k", 512, 8, 4),
    (768, 7168): ("n", 1792, 4, 1),
}


def sweep_configs(k, n):
    out = []
    for t in range(128, k + 1, 128):
        if k % t:
            continue
        tile_bytes = t * n
        if not (96 << 10) <= tile_bytes <= (6 << 20):
            continue
        for nbuf, ndma in itertools.product((2, 4, 8), (1, 4)):
            if nbuf * tile_bytes > (40 << 20) or (t // ndma) % 32:
                continue
            if nbuf > k // t:
                continue
            out.append(("k", t, nbuf, ndma))
    if k <= 2048:
        for t in (256, 512, 896, 1024, 1792, 2048):
            if n % t or t == n:
                continue
            for nbuf in (2, 4):
                if nbuf <= n // t and k % 32 == 0:
                    out.append(("n", t, nbuf, 1))
    return out


def main():
    sweep = "--sweep" in sys.argv
    ms = [int(a.split("=")[1]) for a in sys.argv if a.startswith("--m=")]
    only = [a.split("=")[1] for a in sys.argv if a.startswith("--shape=")]
    key = jax.random.PRNGKey(0)
    for name, (k, n) in SHAPES.items():
        if only and name not in only:
            continue
        keys = jax.random.split(key, 2 * N_W + 1)
        ws = [bl.rand(keys[i], (k, n), jnp.float8_e4m3fn, 0.05) for i in range(N_W)]
        ss = [bl.rand(keys[N_W + i], (n,), jnp.float32, 0.01) for i in range(N_W)]
        args = list(zip(ws, ss))
        floor = k * n / HBM * 1e6
        for m in ms or ([8, 16, 32] if name.startswith("tp8") else [16, 32]):
            x0 = bl.rand(keys[-1], (m, k), jnp.bfloat16)
            res = {}

            def rec(label, us):
                res[label] = us
                print(
                    f"{name} m={m} K={k} N={n} {label:28s} {us:7.2f} us "
                    f"({floor / us * 100:5.1f}% of HBM floor {floor:.2f})",
                    flush=True,
                )

            rec("xla", bl.time_chain(lambda x, w, s: xla_quantized_matmul(x, w, s), x0, args))
            cfgs = sweep_configs(k, n) if sweep else [SWEEP_BEST[(k, n)]]
            if "--f8sweep" in sys.argv:
                cfgs = [c + ("f8dot",) for c in sweep_configs(k, n) if c[1] % 256 == 0 or c[0] == "n"]
            tuned = fp8_skinny.TUNED_HBM.get((k, n))
            if tuned and tuned not in cfgs:
                cfgs.insert(0, tuned)
            xla_ref = bl.first_device(bl.run_on_devices(lambda x, w, s: xla_quantized_matmul(x, w, s), x0, *args[0]))
            for cfg in cfgs:
                try:
                    us = bl.time_chain(
                        lambda x, w, s, cfg=cfg: fp8_skinny.fp8_skinny_matmul(
                            x, w, s, config=cfg, hbm_pin=True
                        ),
                        x0, args,
                    )
                except Exception as e:  # report and continue the sweep
                    print(f"{name} m={m} cfg={cfg} FAILED {type(e).__name__}: {str(e)[:200]}")
                    continue
                rec("skinny" + ("*" if cfg == tuned else "") + str(cfg), us)
                if "f8dot" in cfg:
                    o = bl.first_device(bl.run_on_devices(
                        lambda x, w, s, cfg=cfg: fp8_skinny.fp8_skinny_matmul(x, w, s, config=cfg, hbm_pin=True),
                        x0, *args[0]))
                    print(f"F8EXACT {name} m={m} {cfg} bitexact_vs_xla={np.array_equal(np.asarray(o, np.float32), np.asarray(xla_ref, np.float32))}", flush=True)
            ring = tuned or ("k", 128 * max(1, (1 << 20) // (128 * n)), 4, 1)
            t = ring[1] if k % ring[1] == 0 else 128
            for mode in ("dma", "dot", "dot8"):
                call = build_diag(m, k, n, t, ring[2], mode)
                rec(f"diag_{mode}_t{t}_b{ring[2]}",
                    bl.time_chain(lambda x, w, s, call=call: call(x, pltpu.with_memory_space_constraint(w, pltpu.HBM)), x0, args))
            sk = {l: v for l, v in res.items() if l.startswith("skinny")}
            if not sk:
                continue
            best = min(sk, key=sk.get)
            print(f"BEST {name} m={m}: {best} {sk[best]:.2f} us; xla {res['xla']:.2f}; floor {floor:.2f}", flush=True)
            if "--f8sweep" in sys.argv:
                continue
            bcfg = eval(best.lstrip("skinny*"))
            ref_cfg = tuned or bcfg
            outs = {}

            def out_of(cfg):
                f = lambda x, w, s: fp8_skinny.fp8_skinny_matmul(x, w, s, config=cfg, hbm_pin=True)
                return bl.first_device(bl.run_on_devices(f, x0, *args[0])).astype("float32")

            xla_out = bl.first_device(bl.run_on_devices(lambda x, w, s: xla_quantized_matmul(x, w, s), x0, *args[0])).astype("float32")
            ref_out = out_of(ref_cfg)
            for opts in ((), ("f8dot",)):
                cfg = tuple(bcfg) + opts
                if opts:
                    us = bl.time_chain(
                        lambda x, w, s, cfg=cfg: fp8_skinny.fp8_skinny_matmul(x, w, s, config=cfg, hbm_pin=True),
                        x0, args)
                else:
                    us = sk[best]
                o = out_of(cfg)
                vs_ref = np.array_equal(o, ref_out)
                vs_xla = np.array_equal(o, xla_out)
                rel = float(np.max(np.abs(o - xla_out)) / (np.max(np.abs(xla_out)) + 1e-30))
                print(f"VARIANT {name} m={m} {str(cfg):40s} {us:7.2f} us bitexact_vs_tuned={vs_ref} "
                      f"bitexact_vs_xla={vs_xla} maxrel_vs_xla={rel:.2e}", flush=True)


if __name__ == "__main__":
    main()
