"""On-device numerics of the sequence-parallel pieces, on one 8-core host.

1. fused_ep_moe_v2(restore_rows=False) == this slot's rows of the restored
   output (bit-exact expected: same kernel, one gather fewer).
2. psum_scatter over a TP group == that slot's rows of psum (reduction order).
3. mHC fused seam on a 1/TP slice of the rows == the same rows of the
   full-batch call (row-independent arithmetic; the gates path differs by
   token count, so this says whether the switch moves bits).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

sys.path.insert(0, "tests/kernels/fused_moe")
from test_fused_ep_moe_v2_tp_tokens import _make_inputs  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.mhc import (  # noqa: E402
    fused_post_pre_kernel,
    utils,
)
from vllm_torchtpu.kernels.fused_moe.v2 import (  # noqa: E402
    AXIS,
    WeightFormat,
    fused_ep_moe_v2,
)

FAILS = []


def report(name, got, want):
    got = np.asarray(got).astype(np.float32)
    want = np.asarray(want).astype(np.float32)
    exact = np.array_equal(got, want)
    d = got - want
    rel = float(np.linalg.norm(d) / max(np.linalg.norm(want), 1e-20))
    mx = float(np.max(np.abs(d)))
    frac = float(np.mean(got != want))
    print(
        f"{'EXACT' if exact else 'DIFF '} {name}: rel_l2={rel:.3e} max_abs={mx:.3e} "
        f"frac_diff={frac:.4f} shape={got.shape}",
        flush=True,
    )
    return exact, rel


# The served small-batch options (each applies only at <= 16 local rows).
SERVED = dict(
    merge_exchange=True,
    pallas_plan=True,
    decode_combine=True,
    pallas_restore=True,
    fused_select=True,
)


def moe_check(mesh, groups, rows, fmt, served=False):
    ep = mesh.shape[AXIS]
    rank, inputs = _make_inputs(mesh, groups, rows, fmt)

    def run(restore):
        def f(rank_arg, *args):
            return fused_ep_moe_v2(
                *args,
                topk=4,
                renormalize=True,
                mesh=mesh,
                capacity=128,
                weight_format=fmt,
                rhs_qb=512 if fmt == WeightFormat.FP4 else None,
                rank=rank_arg,
                sharded_plan=True,
                token_replica_groups=groups,
                scoring_fn="sqrtsoftplus",
                routed_scaling_factor=1.0,
                restore_rows=restore,
                **(SERVED if served else {}),
            )

        return jax.block_until_ready(jax.jit(f)(rank, *inputs))

    full = np.asarray(run(True)).reshape(ep, rows, -1)
    local = np.asarray(run(False))
    width = len(groups[0])
    local = local.reshape(ep, rows // width, -1)
    slot = {r: groups[g].index(r) for g in range(len(groups)) for r in groups[g]}
    want = np.stack(
        [
            full[r, slot[r] * (rows // width) : (slot[r] + 1) * (rows // width)]
            for r in range(ep)
        ]
    )
    exact, _ = report(
        f"moe {fmt.name} groups={groups} rows={rows} served={served}", local, want
    )
    if not exact:
        FAILS.append("moe")


def rs_check(mesh, groups, rows, hidden=7168):
    ep = mesh.shape[AXIS]
    x = jax.device_put(
        jax.random.normal(jax.random.key(7), (ep * rows, hidden), jnp.bfloat16),
        NamedSharding(mesh, P(AXIS)),
    )
    ig = [list(g) for g in groups]

    def sm(body):
        return jax.jit(
            jax.shard_map(
                body, mesh=mesh, in_specs=P(AXIS), out_specs=P(AXIS), check_vma=False
            )
        )

    ar = np.asarray(sm(lambda b: lax.psum(b, AXIS, axis_index_groups=ig))(x))
    rs = np.asarray(
        sm(
            lambda b: lax.psum_scatter(
                b, AXIS, scatter_dimension=0, tiled=True, axis_index_groups=ig
            )
        )(x)
    )
    width = len(groups[0])
    ar = ar.reshape(ep, rows, hidden)
    rs = rs.reshape(ep, rows // width, hidden)
    slot = {r: groups[g].index(r) for g in range(len(groups)) for r in groups[g]}
    want = np.stack(
        [
            ar[r, slot[r] * (rows // width) : (slot[r] + 1) * (rows // width)]
            for r in range(ep)
        ]
    )
    exact, rel = report(f"rs-vs-ar groups={groups} rows={rows}", rs, want)
    # A full-precision reference says which side carries the rounding.
    xf = np.asarray(x).astype(np.float32).reshape(ep, rows, hidden)
    ref = np.stack(
        [
            sum(xf[m] for m in groups[[i for i, g in enumerate(groups) if r in g][0]])
            for r in range(ep)
        ]
    )
    report("  ar-vs-f32sum", ar, ref)
    ref_l = np.stack(
        [
            ref[r, slot[r] * (rows // width) : (slot[r] + 1) * (rows // width)]
            for r in range(ep)
        ]
    )
    report("  rs-vs-f32sum", rs, ref_l)


def mhc_check(tokens, width, gates_in_kernel=True):
    H, HC = 7168, 4
    ks = jax.random.split(jax.random.key(3), 6)
    x = jax.random.normal(ks[0], (tokens, H), jnp.bfloat16)
    res = jax.random.normal(ks[1], (tokens, HC, H), jnp.bfloat16)
    post = jax.random.uniform(ks[2], (tokens, HC, 1), jnp.float32) * 2
    comb = jax.nn.softmax(jax.random.normal(ks[3], (tokens, HC, HC)), -1)
    fn = jax.random.normal(ks[4], ((2 + HC) * HC, HC * H), jnp.float32) * 0.01
    sc = jnp.asarray([0.5, 0.7, 0.9], jnp.float32)
    hb = jax.random.normal(ks[5], ((2 + HC) * HC,), jnp.float32) * 0.1
    nw = 1 + 0.1 * jax.random.normal(ks[5], (H,), jnp.float32)
    split = utils.SplitFn(*utils.split_fn3(fn))

    def call(x, res, post, comb):
        return fused_post_pre_kernel.mhc_fused_post_pre(
            x,
            res,
            post,
            comb,
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

    full = jax.jit(call)(x, res, post, comb)
    n = tokens // width
    parts = [
        jax.jit(call)(
            x[i * n : (i + 1) * n],
            res[i * n : (i + 1) * n],
            post[i * n : (i + 1) * n],
            comb[i * n : (i + 1) * n],
        )
        for i in range(width)
    ]
    for k, name in enumerate(("residual", "post_mix", "comb_mix", "layer_input")):
        got = np.concatenate([np.asarray(p[k]) for p in parts])
        report(f"mhc T={tokens}->{n} gik={gates_in_kernel} {name}", got, full[k])


if __name__ == "__main__":
    devs = jax.local_devices()
    print(devs, flush=True)
    order = (3, 0, 6, 1, 7, 4, 2, 5)
    mesh = Mesh(np.asarray([devs[i] for i in order]), (AXIS,))
    tp8 = ((0, 1, 2, 3, 4, 5, 6, 7),)
    tp8p = ((5, 2, 7, 0, 3, 6, 1, 4),)
    tp4 = ((7, 1, 4, 2), (6, 0, 5, 3))
    what = sys.argv[1:] or ["moe", "rs", "mhc"]
    if "rs" in what:
        for g in (tp8, tp4):
            rs_check(mesh, g, 2048 if len(g[0]) == 8 else 512)
    if "mhc" in what:
        mhc_check(2048, 8)
        mhc_check(512, 4)
        mhc_check(512, 4, gates_in_kernel=False)
    if "moe" in what:
        for g in (tp8, tp8p, tp4):
            moe_check(mesh, g, 256, WeightFormat.FP4)
            moe_check(mesh, g, 256, WeightFormat.FP4, served=True)
            moe_check(mesh, g, 64, WeightFormat.FP4, served=True)
    print("RESULT", "FAIL " + ",".join(FAILS) if FAILS else "moe-local-rows exact")
