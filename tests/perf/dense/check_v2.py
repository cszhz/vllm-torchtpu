# SPDX-License-Identifier: Apache-2.0
"""TPU_FP8_SKINNY_TABLE=v2 through the serving entry point vs the XLA path it replaces.

Run: python tests/perf/dense/check_v2.py   (prints one line per shape and M; exit 1 on a mismatch)
"""

import os
import sys

os.environ["TPU_FP8_SKINNY_TABLE"] = "v2"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny  # noqa: E402
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul  # noqa: E402
from vllm_torchtpu.layers.adapter import linear_common  # noqa: E402

V2_ONLY = sorted(set(fp8_skinny.TUNED_V2_HBM) - set(fp8_skinny.TUNED_HBM))


def main():
    bad = 0
    key = jax.random.PRNGKey(3)
    for k, n in V2_ONLY:
        kw, ks, kx = jax.random.split(jax.random.fold_in(key, k * n), 3)
        w = (jax.random.normal(kw, (k, n), jnp.float32) * 0.05).astype(jnp.float8_e4m3fn)
        s = jax.random.uniform(ks, (n,), jnp.float32, 0.001, 0.02)
        for m in (8, 16, 24, 32):
            x = jax.random.normal(jax.random.fold_in(kx, m), (m, k), jnp.bfloat16)
            got = jax.jit(lambda x, w, s: linear_common._fp8_linear_pallas(x, w, s, hbm_pin=True))(x, w, s)
            ref = jax.jit(xla_quantized_matmul)(x, w, s)
            hlo = jax.jit(lambda x, w, s: linear_common._fp8_linear_pallas(x, w, s, hbm_pin=True)).lower(x, w, s).as_text()
            exact = np.array_equal(np.asarray(got, np.float32), np.asarray(ref, np.float32))
            uses = "fp8_skinny" in hlo
            bad += not (exact and uses)
            print(f"v2 K={k} N={n} M={m}: kernel_in_hlo={uses} bitexact_vs_xla={exact}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
