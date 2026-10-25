# SPDX-License-Identifier: Apache-2.0
"""wo_a projection at decode token counts: column tile (tile_r) sweep + bit-exactness.

Run: python tests/perf/dense/woa_bench.py
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_lib as bl  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.o_projection import wo_a_projection  # noqa: E402

N_W = 16
HBM = 3.10e12


def main():
    ks = iter(jax.random.split(jax.random.PRNGKey(2), 200))
    for tp, groups in ((8, 2), (4, 4)):
        d, r = 4096, 1024
        ws = [bl.rand(next(ks), (d, groups * r), jnp.float8_e4m3fn, 0.05) for _ in range(N_W)]
        ss = [bl.rand(next(ks), (groups * r,), jnp.float32, 0.01) for _ in range(N_W)]
        floor = d * groups * r / HBM * 1e6
        for t in (8, 16, 32):
            x0 = bl.rand(next(ks), (t, groups * 8, 512), jnp.bfloat16)
            cs = [bl.rand(next(ks), (t, 256), jnp.float32, 1.0) for _ in range(N_W)]
            args = list(zip(ws, ss, cs))
            ref = None
            variants = [("prod", None, None)] + [("grid", tr, 1) for tr in (1024, 512, 256)]
            for kind, tr, nb in variants:

                def kern(x, w, s, c, tr=tr, nb=nb):
                    return wo_a_projection(x, w, s, c, tile_r=tr, groups_per_step=nb)

                try:
                    us = bl.time_chain(kern, x0, args)
                except Exception as e:  # report and continue the sweep
                    print(f"wo_a tp{tp} T={t} {kind} r{tr} b{nb} FAILED {type(e).__name__}: {str(e)[:300]}")
                    continue
                out = np.ascontiguousarray(bl.first_device(bl.run_on_devices(kern, x0, *args[0])))
                if ref is None:
                    ref = out
                exact = np.array_equal(out.view(np.uint16), ref.view(np.uint16))
                print(
                    f"wo_a tp{tp} T={t} {kind} tile_r={tr} nbuf={nb}: {us:6.2f} us "
                    f"({floor / us * 100:5.1f}% of floor {floor:.2f}) bitexact_vs_prod={exact}",
                    flush=True,
                )

if __name__ == "__main__":
    main()
