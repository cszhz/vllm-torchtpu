# SPDX-License-Identifier: Apache-2.0
"""mHC seam kernel (fused post+pre) at decode token counts, all cores at once.

Run: python tests/perf/dense/mhc_bench.py
Variants decompose the per-call time: GEMM passes, in-kernel gates, fused norm.
"""

from __future__ import annotations

import sys

import jax
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_lib as bl  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel as fpp  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.mhc import utils  # noqa: E402

H, M = 7168, 4
M3 = (2 + M) * M
N_W = 16
GATES = utils.GateConsts(1e-6, 2.0, 20)


def variants():
    yield "prod(highest,gates,norm)", dict(gemm_precision="highest", gates=GATES, norm=True)
    yield "1pass", dict(gemm_precision="default", gates=GATES, norm=True)
    yield "xla_gates", dict(gemm_precision="highest", gates=None, norm=True)
    yield "no_norm", dict(gemm_precision="highest", gates=GATES, norm=False)


def main():
    key = jax.random.PRNGKey(1)
    ks = iter(jax.random.split(key, 20 * N_W + 8))
    fns = []
    for _ in range(N_W):
        fn = [bl.rand(next(ks), (M3, M * H), jnp.bfloat16, 1e-3) for _ in range(3)]
        sc = bl.rand(next(ks), (3,), jnp.float32, 0.1)
        hb = bl.rand(next(ks), (M3,), jnp.float32, 0.1)
        nw = bl.rand(next(ks), (H,), jnp.bfloat16, 1.0)
        fns.append((*fn, sc, hb, nw))
    for t in (8, 16, 32):
        x0 = bl.rand(next(ks), (t, H), jnp.bfloat16)
        sets = [
            (
                bl.rand(next(ks), (t, M * H), jnp.bfloat16),
                *f,
                bl.rand(next(ks), (t, M), jnp.float32, 0.5),
                bl.rand(next(ks), (t, M * M), jnp.float32, 0.5),
            )
            for f in fns
        ]
        for label, v in variants():

            def kern(x, res, fh, fm, fl, sc, hb, nw, post, comb, v=v, t=t):
                return fpp.fused_post_pre_mixes(
                    x,
                    res,
                    post,
                    comb,
                    utils.SplitFn(fh, fm, fl),
                    sc,
                    hb,
                    1e-6,
                    1e-6,
                    gemm_precision=v["gemm_precision"],
                    gates=v["gates"],
                    norm_weight=nw if v["norm"] else None,
                    norm_eps=1e-6 if v["norm"] else None,
                )

            us = bl.time_chain(kern, x0, sets)
            outs = [bl.first_device(o) for o in bl.run_on_devices(kern, x0, *sets[0])]
            if label.startswith("prod"):
                ref = outs
            exact = all(
                np.array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8)) for a, b in zip(outs, ref)
            )
            print(f"mhc T={t} {label:28s} {us:7.2f} us bitexact_vs_prod={exact}", flush=True)


if __name__ == "__main__":
    main()
