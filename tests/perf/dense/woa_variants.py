# SPDX-License-Identifier: Apache-2.0
"""wo_a projection: where the per-call time goes, via source-patched `_group` variants
(diag: removals), device kernel time from a profile.

Run: python tests/perf/dense/woa_variants.py
"""

from __future__ import annotations

import inspect
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_lib as bl  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4 import o_projection as op  # noqa: E402

N_W = 8
SRC = textwrap.dedent(inspect.getsource(op._group))
ORIG = op._group
ROPE = "x = _rope_heads(x_ref[rows], cos_sin[rows], head_dim=head_dim)"
RESHAPE = "x = x.reshape(sub_t, -1)"
DOT = """partial = jax.lax.dot_general(
            lhs,
            rhs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )"""
PATCHES = {
    "prod": [],
    "diag:no_rope": [(ROPE, "x = x_ref[rows]")],
    "diag:no_rope_no_reshape": [(ROPE, "x = x_ref[rows]"), (RESHAPE, "x = jnp.zeros((sub_t, x.shape[1] * x.shape[2]), x.dtype)")],
    "diag:no_dot": [(DOT, "partial = jnp.zeros((lhs.shape[0], rhs.shape[1]), jnp.float32) + lhs[:, :1].astype(jnp.float32)")],
    "diag:only_dma": [("    rhs = w_ref[...]\n", "    out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)\n    return\n")],
}


def patched(name):
    src = SRC
    for a, b in PATCHES[name]:
        assert a in src, (name, a)
        src = src.replace(a, b)
    ns = dict(op.__dict__)
    exec(compile(src, f"<{name}>", "exec"), ns)
    return ns["_group"]


def main():
    ks = iter(jax.random.split(jax.random.PRNGKey(2), 200))
    raw = op.wo_a_projection.__wrapped__
    for tp, groups in ((8, 2), (4, 4)):
        d, r = 4096, 1024
        ws = [bl.rand(next(ks), (d, groups * r), jnp.float8_e4m3fn, 0.05) for _ in range(N_W)]
        ss = [bl.rand(next(ks), (groups * r,), jnp.float32, 0.01) for _ in range(N_W)]
        for t in (8, 32):
            x0 = bl.rand(next(ks), (t, groups * 8, 512), jnp.bfloat16)
            cs = [bl.rand(next(ks), (t, 256), jnp.float32, 1.0) for _ in range(N_W)]
            args = list(zip(ws, ss, cs))
            for name in PATCHES:
                op._group = patched(name)
                try:
                    prof = bl.profile_chain(lambda x, w, s, c: raw(x, w, s, c), x0, args)
                except Exception as e:  # report and continue
                    print(f"wo_a tp{tp} T={t} {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                    continue
                finally:
                    op._group = ORIG
                ops = "; ".join(f"{n[:30]} x{c:.2f} {us:.2f}us" for n, c, us in prof[:3])
                print(f"wo_a tp{tp} T={t} {name:24s} | {ops}", flush=True)


if __name__ == "__main__":
    main()
