# SPDX-License-Identifier: Apache-2.0
"""Device check and timing of the fused small all-reduce + all-gather.

Run on a TPU host with 8 cores: python <this file>. For every (group size,
rows) it checks the fused kernel bit-for-bit against the two separate kernels
(and the add form against the separate kernels followed by an XLA bf16 add),
then times chains of calls: AR alone, AG alone, AR then AG, fused, fused-add,
and an AR of twice the rows (a concatenated payload).
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives.all_gather import (
    small_all_gather,
)
from vllm_torchtpu.kernels.experimental.small_collectives.all_reduce import (
    small_all_reduce,
)
from vllm_torchtpu.kernels.experimental.small_collectives.all_reduce_gather import (
    small_all_reduce_gather,
)

AX = "d"
HIDDEN = 7168
CHAIN = 64


def local_rows(rows, width, alignment=8):
    # TokenReplicaLayout.create's slice size
    lr = (rows + width - 1) // width
    a = min(alignment, 1 << (lr - 1).bit_length())
    return (lr + a - 1) // a * a


def tables(ncores, width):
    groups = [list(range(g, g + width)) for g in range(0, ncores, width)]
    rank = np.array([i % width for i in range(ncores)], np.int32)
    peers = np.concatenate(
        [np.array(groups[i // width], np.int32) for i in range(ncores)]
    )
    return groups, rank, peers


def main():
    devs = jax.devices()
    ncores = len(devs)
    print("devices", ncores, devs[0].device_kind, jax.__version__, flush=True)
    mesh = Mesh(np.array(devs), (AX,))
    shard = NamedSharding(mesh, P(AX))
    failures = 0
    configs = [(8, 8), (8, 16), (8, 24), (8, 32), (4, 8), (4, 16), (4, 32)]
    if len(sys.argv) > 1:
        configs = [tuple(int(v) for v in c.split("x")) for c in sys.argv[1:]]
    for width, rows in configs:
        if ncores % width:
            continue
        yrows = local_rows(rows, width)
        groups, rank_np, peers_np = tables(ncores, width)
        key = jax.random.PRNGKey(rows * 10 + width)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (ncores * rows, HIDDEN), jnp.float32).astype(
            jnp.bfloat16
        )
        y = jax.random.normal(k2, (ncores * yrows, HIDDEN), jnp.float32).astype(
            jnp.bfloat16
        )
        x, y = jax.device_put(x, shard), jax.device_put(y, shard)
        rank = jax.device_put(jnp.asarray(rank_np), shard)
        peers = jax.device_put(jnp.asarray(peers_np), shard)
        kw = dict(num_cores=width, axis=AX)

        def sep(xb, yb, r, p):
            s = small_all_reduce(xb, rank=r, peers=p, form="twoshot", **kw)
            g = small_all_gather(yb, rank=r, peers=p, **kw)
            return s, g

        def fused(xb, yb, r, p):
            return small_all_reduce_gather(xb, yb, rank=r, peers=p, **kw)

        def fused_add(xb, yb, r, p):
            return small_all_reduce_gather(xb, yb, rank=r, peers=p, add=True, **kw)

        spec = P(AX)

        def sm(body, nout):
            return jax.jit(
                jax.shard_map(
                    body,
                    mesh=mesh,
                    in_specs=(spec,) * 4,
                    out_specs=(spec,) * nout if nout > 1 else spec,
                    check_vma=False,
                )
            )

        s0, g0 = sm(sep, 2)(x, y, rank, peers)
        s1, g1 = sm(fused, 2)(x, y, rank, peers)
        a1 = sm(fused_add, 1)(x, y, rank, peers)

        def xla_add(s, g):
            sl = s.reshape(ncores, rows, HIDDEN)
            gl = g.reshape(ncores, width * yrows, HIDDEN)[:, :rows]
            return (sl + gl).reshape(-1, HIDDEN)

        a0 = jax.jit(xla_add)(s0, g0)
        # numpy reference of the group sum (f32 in group order, rounded once)
        xn = np.asarray(x.astype(jnp.float32)).reshape(ncores, rows, HIDDEN)
        ok_ref = True
        for g in groups:
            acc = np.zeros((rows, HIDDEN), np.float32)
            for m in g:
                acc = acc + xn[m]
            ref = np.asarray(jnp.asarray(acc).astype(jnp.bfloat16).astype(jnp.float32))
            for m in g:
                got = np.asarray(s0.astype(jnp.float32)).reshape(ncores, rows, HIDDEN)[
                    m
                ]
                ok_ref &= np.array_equal(got, ref)
        eq = lambda a, b: bool(jnp.array_equal(a.view(jnp.uint16), b.view(jnp.uint16)))
        res = dict(sum=eq(s0, s1), gather=eq(g0, g1), add=eq(a0, a1), ref=ok_ref)
        failures += not all(res.values())
        if not res["add"]:
            d = np.abs(
                np.asarray(a0.astype(jnp.float32)) - np.asarray(a1.astype(jnp.float32))
            )
            print("   add mismatch max abs", d.max(), "count", int((d > 0).sum()))
        print(f"n{width} rows{rows} yrows{yrows} bitexact {res}", flush=True)

        # timing: CHAIN dependent calls in one program
        def chain(step):
            def body(xb, yb, r, p):
                for _ in range(CHAIN):
                    xb, yb = step(xb, yb, r, p)
                return xb, yb

            f = sm(body, 2)
            return f

        half = jnp.bfloat16(1.0 / width)

        def st_ar(xb, yb, r, p):
            return small_all_reduce(
                xb, rank=r, peers=p, form="twoshot", **kw
            ) * half, yb

        def st_ag(xb, yb, r, p):
            return xb, small_all_gather(yb, rank=r, peers=p, **kw)[:yrows]

        def st_sep(xb, yb, r, p):
            s, g = sep(xb, yb, r, p)
            return s * half, g[:yrows]

        def st_fused(xb, yb, r, p):
            s, g = fused(xb, yb, r, p)
            return s * half, g[:yrows]

        def st_add(xb, yb, r, p):
            s = fused_add(xb, yb, r, p)
            return s * half, yb + s[:yrows]

        def st_ar2(xb, yb, r, p):
            z = jnp.concatenate([xb, xb], axis=0)
            return small_all_reduce(z, rank=r, peers=p, form="twoshot", **kw)[
                :rows
            ] * half, yb

        def st_ar1(xb, yb, r, p):
            return small_all_reduce(
                xb, rank=r, peers=p, form="oneshot", **kw
            ) * half, yb

        def st_none(xb, yb, r, p):
            return xb * half, yb

        times = {}
        for name, step in [
            ("none", st_none),
            ("ar", st_ar),
            ("ag", st_ag),
            ("ar+ag", st_sep),
            ("fused", st_fused),
            ("fused_add", st_add),
            ("ar_2x", st_ar2),
            ("ar_1shot", st_ar1),
        ]:
            f = chain(step)
            out = f(x, y, rank, peers)
            jax.block_until_ready(out)
            best = 1e9
            for _ in range(7):
                t0 = time.perf_counter()
                out = f(x, y, rank, peers)
                jax.block_until_ready(out)
                best = min(best, time.perf_counter() - t0)
            times[name] = best / CHAIN * 1e6
        base = times["none"]
        print(
            f"n{width} rows{rows} us/call (minus chain op {base:.2f}): "
            + " ".join(f"{k}={v - base:.2f}" for k, v in times.items() if k != "none"),
            flush=True,
        )
    print("FAILURES", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
