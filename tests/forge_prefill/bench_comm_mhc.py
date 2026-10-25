"""Microbench: TP8 all-reduce vs reduce-scatter / all-gather of a prefill
hidden block, and the fused mHC seam kernel at the full vs per-rank token
count. Run on one tpu7x 2x2x1 host (8 TensorCores)."""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel, utils

H = 7168
HC = 4
ITERS = 20


def timeit(fn, *args, reps=10):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bench_comm(tokens, tp=8):
    devs = jax.devices()
    n = tp
    nd = len(devs)
    groups = [list(range(g, g + tp)) for g in range(0, nd, tp)]
    kw = {} if tp == nd else {"axis_index_groups": groups}
    mesh = Mesh(np.array(devs), ("x",))
    spec = P("x")
    x = jax.device_put(
        jax.random.normal(jax.random.key(0), (nd * tokens, H), jnp.bfloat16),
        NamedSharding(mesh, spec),
    )
    xs = jax.device_put(
        jax.random.normal(jax.random.key(1), (nd * tokens // n, H), jnp.bfloat16),
        NamedSharding(mesh, spec),
    )

    def loop(body):
        def f(blk):
            return lax.fori_loop(0, ITERS, lambda i, c: body(c), blk)

        return jax.jit(
            jax.shard_map(f, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False)
        )

    eighth = jnp.bfloat16(0.125)
    ar = loop(lambda c: lax.psum(c, "x", **kw) * eighth)
    # RS then AG keeps the carry shape: a round trip equal in bytes to one AR.
    rsag = loop(
        lambda c: lax.all_gather(
            lax.psum_scatter(c, "x", scatter_dimension=0, tiled=True, **kw) * eighth,
            "x",
            axis=0,
            tiled=True,
            **kw,
        )
    )
    ag = loop(
        lambda c: lax.all_gather(c, "x", axis=0, tiled=True, **kw)[: c.shape[0]] + c
    )
    rs = loop(
        lambda c: jnp.tile(
            lax.psum_scatter(c, "x", scatter_dimension=0, tiled=True, **kw) * eighth,
            (n, 1),
        )
    )
    tile = loop(lambda c: jnp.tile(c[: c.shape[0] // n] * eighth, (n, 1)))
    t_tile = timeit(tile, x) / ITERS
    t_ar = timeit(ar, x) / ITERS
    t_rsag = timeit(rsag, x) / ITERS
    t_ag = timeit(ag, xs) / ITERS
    t_rs = timeit(rs, x) / ITERS
    mb = tokens * H * 2 / 1e6
    print(
        f"COMM TP{tp} T={tokens} ({mb:.1f} MB/rank): AR {t_ar * 1e6:.1f} us | RS+AG "
        f"{t_rsag * 1e6:.1f} us | AG(T/{n}->T)+add {t_ag * 1e6:.1f} us | "
        f"RS+tile {t_rs * 1e6:.1f} us | tile {t_tile * 1e6:.1f} us",
        flush=True,
    )


def bench_mhc(tokens, gates_in_kernel):
    ks = jax.random.split(jax.random.key(2), 6)
    x = jax.random.normal(ks[0], (tokens, H), jnp.bfloat16)
    res = jax.random.normal(ks[1], (tokens, HC * H), jnp.bfloat16)
    post = jax.random.uniform(ks[2], (tokens, HC), jnp.float32)
    comb = jax.random.uniform(ks[3], (tokens, HC * HC), jnp.float32) * 0.25
    fn = jax.random.normal(ks[4], ((2 + HC) * HC, HC * H), jnp.float32) * 0.01
    sc = jnp.ones((3,), jnp.float32)
    hb = jnp.zeros(((2 + HC) * HC,), jnp.float32)
    nw = jnp.ones((H,), jnp.float32)
    split = utils.SplitFn(*utils.split_fn3(fn))

    def f(x, res):
        def body(i, c):
            x, res = c
            r, pm, cm, lay = fused_post_pre_kernel.mhc_fused_post_pre(
                x,
                res.reshape(tokens, HC, H),
                post.reshape(tokens, HC, 1),
                comb.reshape(tokens, HC, HC),
                split,
                sc,
                hb,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                20,
                gates_in_kernel=gates_in_kernel,
                norm_weight=nw,
                norm_eps=1e-6,
            )
            return lay, r.reshape(tokens, HC * H)

        return lax.fori_loop(0, ITERS, body, (x, res))

    t = timeit(jax.jit(f), x, res) / ITERS
    gb = tokens * H * 2 * (1 + 2 * HC + 1) / 1e9
    print(
        f"MHC T={tokens} gates_in_kernel={gates_in_kernel}: {t * 1e6:.1f} us "
        f"({gb / t:.0f} GB/s effective over {gb * 1e3:.0f} MB)",
        flush=True,
    )


if __name__ == "__main__":
    print(jax.devices(), os.environ.get("LIBTPU_INIT_ARGS"), flush=True)
    what = sys.argv[1:] or ["comm", "mhc"]
    if "comm" in what:
        for t in (512, 2048):
            bench_comm(t)
        for t in (128, 512):
            bench_comm(t, tp=4)
    if "mhc" in what:
        for t in (128, 256, 512, 1024, 2048):
            for g in (False, True):
                if g and t > utils.GATES_IN_KERNEL_MAX_TOKENS:
                    continue
                bench_mhc(t, g)
