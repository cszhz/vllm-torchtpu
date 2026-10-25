# SPDX-License-Identifier: Apache-2.0
"""mHC seam kernel: where the per-call time goes, via source-patched kernel variants.

Run: python tests/perf/dense/mhc_variants.py
Each variant patches `_fused_kernel`'s source (exact rewrites, and diagnostic removals
marked diag:), reports device kernel time from a profile and bit-exactness vs prod.
"""

from __future__ import annotations

import inspect
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_lib as bl  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel as fpp  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.mhc import utils  # noqa: E402

H, M = 7168, 4
M3 = (2 + M) * M
N_W = 8
GATES = utils.GateConsts(1e-6, 2.0, 20)
SRC = textwrap.dedent(inspect.getsource(fpp._fused_kernel))
ORIG = fpp._fused_kernel

DOT = "jax.lax.dot_general(\n        g_bf, fn_{c}_ref[...], dn, preferred_element_type=jnp.float32\n    )"
DOT2 = "jax.lax.dot_general(\n            g_bf, fn_{c}_ref[...], dn, preferred_element_type=jnp.float32\n        )"


def _sub_dots(src, fmt):
    src = src.replace(DOT.format(c="hi"), fmt.format(c="hi"))
    for c in ("mid", "lo"):
        src = src.replace(DOT2.format(c=c), fmt.format(c=c))
    return src


PATCHES = {
    "prod": [],
    "diag:no_gemm": "DOTS:jnp.zeros((g_bf.shape[0], fn_{c}_ref.shape[0]), jnp.float32)",
    "diag:no_gates": [
        ("        pre_mix, gate_a_ref[...], gate_b_ref[...] = utils.gates_in_block(",
         "        gate_a_ref[...] = acc[:, :m]\n        gate_b_ref[...] = acc[:, 8:24]\n        pre_mix = acc[:, :m]\n        _unused = (lambda *a, **k: None)(\n"),
    ],
    "diag:only_dma": [
        ("    post = post_ref[...]", "    return _only_dma(refs)\n    post = post_ref[...]"),
    ],
    "diag:no_recomb": [
        ("        new_bf = acc.astype(jnp.bfloat16)\n", "        new_bf = old_bf[j]\n"),
    ],
}


VARIANTS = ["prod", "diag:no_gates", "diag:only_dma", "diag:no_recomb", "diag:no_gemm"]


def patched(name):
    p = PATCHES[name]
    src = SRC
    if isinstance(p, str):
        src = _sub_dots(src, p[len("DOTS:"):])
        assert src != SRC, name
    else:
        for a, b in p:
            assert a in src, (name, a)
            src = src.replace(a, b)
    ns = dict(fpp.__dict__)

    def _only_dma(refs):
        for r in refs[-4:]:
            r[...] = jnp.zeros(r.shape, r.dtype)

    ns["_only_dma"] = _only_dma
    exec(compile(src, f"<{name}>", "exec"), ns)
    return ns["_fused_kernel"]


def skinny_check():
    """Profiler vs wall clock for one skinny GEMM (validates the harness)."""
    from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

    k, n = 7168, 2048
    keys = jax.random.split(jax.random.PRNGKey(5), 2 * N_W + 1)
    args = [(bl.rand(keys[i], (k, n), jnp.float8_e4m3fn, 0.05), bl.rand(keys[N_W + i], (n,), jnp.float32, 0.01))
            for i in range(N_W)]
    x0 = bl.rand(keys[-1], (8, k), jnp.bfloat16)
    f = lambda x, w, s: fp8_skinny.fp8_skinny_matmul(x, w, s, config=fp8_skinny.TUNED_HBM[(k, n)], hbm_pin=True)
    wall = bl.time_chain(f, x0, args)
    prof = bl.profile_chain(f, x0, args)
    print(f"skinny wqa m=8 wall {wall:.2f} us | " + "; ".join(f"{n[:50]} x{c:.2f} {us:.2f}us" for n, c, us in prof), flush=True)


def main():
    ks = iter(jax.random.split(jax.random.PRNGKey(1), 20 * N_W + 8))
    fns = []
    for _ in range(N_W):
        fn = [bl.rand(next(ks), (M3, M * H), jnp.bfloat16, 1e-3) for _ in range(3)]
        sc = bl.rand(next(ks), (3,), jnp.float32, 0.1)
        hb = bl.rand(next(ks), (M3,), jnp.float32, 0.1)
        nw = bl.rand(next(ks), (H,), jnp.bfloat16, 1.0)
        fns.append((*fn, sc, hb, nw))
    raw = fpp.fused_post_pre_mixes.__wrapped__
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

        def make_kern(place):
            def kern(x, res, fh, fm, fl, sc, hb, nw, post, comb):
                if "fn" in place:
                    fh, fm, fl = (pltpu.with_memory_space_constraint(a, pltpu.VMEM) for a in (fh, fm, fl))
                if "res" in place:
                    res = pltpu.with_memory_space_constraint(res, pltpu.VMEM)
                return raw(x, res, post, comb, utils.SplitFn(fh, fm, fl), sc, hb, 1e-6, 1e-6,
                           gates=GATES, norm_weight=nw, norm_eps=1e-6)
            return kern

        ref = None
        for name in VARIANTS:
            place = name.split("@")[1] if "@" in name else ""
            kern = make_kern(place)
            fpp._fused_kernel = patched(name.split("@")[0])
            try:
                prof = bl.profile_chain(kern, x0, sets)
                outs = [np.ascontiguousarray(bl.first_device(o)) for o in bl.run_on_devices(kern, x0, *sets[0])]
            except Exception as e:  # report and continue
                print(f"mhc T={t} {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            finally:
                fpp._fused_kernel = ORIG
            if ref is None:
                ref = outs
            exact = all(np.array_equal(a.view(np.uint8), b.view(np.uint8)) for a, b in zip(outs, ref))
            ops = "; ".join(f"{n[:30]} x{c:.2f} {us:.2f}us" for n, c, us in prof[:6])
            print(f"mhc T={t} {name:16s} bitexact_vs_prod={exact} | {ops}", flush=True)


if __name__ == "__main__":
    main()
