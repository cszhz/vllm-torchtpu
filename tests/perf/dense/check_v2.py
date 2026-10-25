# SPDX-License-Identifier: Apache-2.0
"""TPU_FP8_SKINNY_TABLE=v2 through the serving entry point, with and without
TPU_PIN_OPERANDS_HBM: the shapes v2 adds must equal the XLA path they replace, and the
TP8 shapes must be identical across all four settings.

Run: python tests/perf/dense/check_v2.py   (one line per shape and M; exit 1 on a mismatch)
"""

import os
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.layers.adapter import linear_common

PIN = "skinny:proj:combine:select:plan"
MODES = [("", ""), ("v2", ""), ("", PIN), ("v2", PIN)]
V2_ONLY = set(fp8_skinny.TUNED_V2_HBM) - set(fp8_skinny.TUNED_HBM)


def run(mode, x, w, s):
    os.environ["TPU_FP8_SKINNY_TABLE"], os.environ["TPU_PIN_OPERANDS_HBM"] = mode
    f = jax.jit(lambda x, w, s: linear_common._fp8_linear_pallas(x, w, s, hbm_pin=True))
    kernels = sorted(set(re.findall(r"fp8_skinny_[A-Za-z0-9_]+", f.lower(x, w, s).as_text())))
    return np.asarray(f(x, w, s), np.float32), kernels


def main():
    bad = 0
    key = jax.random.PRNGKey(3)
    for k, n in sorted(fp8_skinny.TUNED_V2_HBM):
        kw, ks, kx = jax.random.split(jax.random.fold_in(key, k * n), 3)
        w = (jax.random.normal(kw, (k, n), jnp.float32) * 0.05).astype(jnp.float8_e4m3fn)
        s = jax.random.uniform(ks, (n,), jnp.float32, 0.001, 0.02)
        for m in (8, 16, 24, 32):
            x = jax.random.normal(jax.random.fold_in(kx, m), (m, k), jnp.bfloat16)
            ref = np.asarray(jax.jit(xla_quantized_matmul)(x, w, s), np.float32)
            outs = {mode: run(mode, x, w, s) for mode in MODES}
            if (k, n) in V2_ONLY:
                ok = all(np.array_equal(outs[md][0], ref) and outs[md][1] for md in MODES if md[0] == "v2")
                ok = ok and all(not outs[md][1] for md in MODES if md[0] == "")
                what = "v2 == XLA, kernel only under v2"
            else:
                base = outs[("", "")][0]
                ok = all(np.array_equal(o, base) for o, _ in outs.values())
                what = "all four settings identical"
            bad += not ok
            names = " | ".join(f"{md[0] or '-'}/{'pin' if md[1] else '-'}:{','.join(kn) or 'xla'}"
                               for md, (_, kn) in outs.items())
            print(f"K={k} N={n} M={m}: {what}: {ok} || {names}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
