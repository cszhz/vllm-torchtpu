# SPDX-License-Identifier: Apache-2.0
"""mHC fused post/pre seam at prefill row counts: device kernel time (profile) for the
token block sweep and source-patched diagnostic variants, bit-exactness vs prod.

Run: python tests/perf/dense/mhc_prefill_bench.py [--t=512,2048]
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

from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel as fpp  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.mhc import utils  # noqa: E402

H, M = 7168, 4
M3 = (2 + M) * M
N_W = 3
GATES = utils.GateConsts(1e-6, 2.0, 20)
SRC = textwrap.dedent(inspect.getsource(fpp._fused_kernel))
ORIG = fpp._fused_kernel
DOT = "jax.lax.dot_general(\n        g_bf, fn_{c}_ref[...], dn, preferred_element_type=jnp.float32\n    )"
DOT2 = "jax.lax.dot_general(\n            g_bf, fn_{c}_ref[...], dn, preferred_element_type=jnp.float32\n        )"

PATCHES = {
    "diag:no_gemm": "DOTS:jnp.zeros((g_bf.shape[0], fn_{c}_ref.shape[0]), jnp.float32)",
    "diag:no_recomb": [("        new_bf = acc.astype(jnp.bfloat16)\n", "        new_bf = old_bf[j]\n")],
    "diag:no_norm": [("utils.finish_layer_input(lay, *norm_w or [None], norm_eps)", "lay.astype(jnp.bfloat16)")],
    "diag:no_collapse": [("    lay = pre_mix[:, 0:1] * gf[:, :h]\n    for i in range(1, m):\n"
                          "        lay = lay + pre_mix[:, i : i + 1] * gf[:, i * h : (i + 1) * h]\n",
                          "    lay = gf[:, :h] + pre_mix[:, 0:1]\n")],
    "diag:only_dma": [("    post = post_ref[...]", "    return _only_dma(refs)\n    post = post_ref[...]")],
}


def patched(name):
    if name not in PATCHES:
        return ORIG
    p = PATCHES[name]
    src = SRC
    if isinstance(p, str):
        for c, d in (("hi", DOT), ("mid", DOT2), ("lo", DOT2)):
            src = src.replace(d.format(c=c), p[5:].format(c=c))
    else:
        for a, b in p:
            assert a in src, (name, a)
            src = src.replace(a, b)
    assert src != SRC, name
    ns = dict(fpp.__dict__)

    def _only_dma(refs):
        for r in refs[-4:]:
            r[...] = jnp.zeros(r.shape, r.dtype)

    ns["_only_dma"] = _only_dma
    exec(compile(src, f"<{name}>", "exec"), ns)
    return ns["_fused_kernel"]


def main():
    ts = [int(v) for a in sys.argv if a.startswith("--t=") for v in a[4:].split(",")] or [256, 512, 1024, 2048]
    ks = iter(jax.random.split(jax.random.PRNGKey(1), 400))
    fns = [([bl.rand(next(ks), (M3, M * H), jnp.bfloat16, 1e-3) for _ in range(3)],
            bl.rand(next(ks), (3,), jnp.float32, 0.1), bl.rand(next(ks), (M3,), jnp.float32, 0.1),
            bl.rand(next(ks), (H,), jnp.bfloat16, 1.0)) for _ in range(N_W)]
    raw = fpp.fused_post_pre_mixes.__wrapped__
    for t in ts:
        gates = GATES if t <= utils.GATES_IN_KERNEL_MAX_TOKENS else None
        floor = t * (H * 2 * 2 + M * H * 2 * 2) / 3.1e12 * 1e6
        x0 = bl.rand(next(ks), (t, H), jnp.bfloat16)
        sets = [(bl.rand(next(ks), (t, M * H), jnp.bfloat16), *fn, sc, hb, nw,
                 bl.rand(next(ks), (t, M), jnp.float32, 0.5), bl.rand(next(ks), (t, M * M), jnp.float32, 0.5))
                for fn, sc, hb, nw in fns]
        variants = [("prod", tb, 0) for tb in (32, 16, 64, 128)] + [(d, 32, 0) for d in PATCHES]
        ref = None
        for name, tb, lc in variants:

            def kern(x, res, fh, fm, fl, sc, hb, nw, post, comb, tb=tb, lc=lc):
                return raw(x, res, post, comb, utils.SplitFn(fh, fm, fl), sc, hb, 1e-6, 1e-6,
                           token_block_size=tb, gates=gates, norm_weight=nw, norm_eps=1e-6)

            fpp._fused_kernel = patched(name)
            try:
                prof = bl.profile_chain(kern, x0, sets, reps=4, top=6)
                outs = [np.ascontiguousarray(bl.first_device(o)) for o in bl.run_on_devices(kern, x0, *sets[0])]
            except Exception as e:  # report and continue
                print(f"mhcp T={t} {name} tb={tb} lc={lc} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            finally:
                fpp._fused_kernel = ORIG
            if ref is None:
                ref = outs
            exact = [bool(np.array_equal(a.view(np.uint8), b.view(np.uint8))) for a, b in zip(outs, ref)]
            kus = sum(u for n, c, u in prof if not n.startswith("while") and c > 0.5)
            ops = "; ".join(f"{n[:24]} x{c:.2f} {u:.1f}" for n, c, u in prof[:5])
            print(f"mhcp T={t} {name:16s} tb={tb:3d} lc={lc:3d} kernel+ops {kus:7.1f} us (floor {floor:.1f}) "
                  f"bitexact={exact} | {ops}", flush=True)


if __name__ == "__main__":
    main()
