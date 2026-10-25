# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""On device (8 local devices): the CSA phase with each request's scan + top-k
split across an 8-rank group (attn_tp=8, attn_heads=16) vs the single-rank
phase on the same heads (bitwise, caches too), and its time per layer
(slope between 1 and 11 in-kernel repetitions)."""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_phase_harness as H  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout  # noqa: E402


NREP = 41  # in-kernel repetitions (launch skew across devices / NREP)


def timed(f, ops, n=15):
    jax.block_until_ready(f(*ops))
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(f(*ops))
        ts.append(time.perf_counter() - t)
    return ts


def words(c):
    return np.asarray(csa_cache_layout.slabs_to_words(c) if c.dtype == jnp.uint8 else c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="131072,358400")
    ap.add_argument("--req", default="1")
    ap.add_argument("--idx_corr", default="0.0")
    ap.add_argument("--kinds", default="csa")
    ap.add_argument("--ablate", default="", help="comma list of ablation sets joined by +")
    a = ap.parse_args()
    print(jax.devices(), flush=True)
    for ctx in map(int, a.ctx.split(",")):
        for R in map(int, a.req.split(",")):
            rng = np.random.default_rng(0)
            seq = [ctx - 3 - 17 * i for i in range(R)]
            ls = H.build(rng, ("hca", "csa"), seq, ctx)
            for kind, corr in [(k, float(c)) for k in a.kinds.split(",")
                               for c in (a.idx_corr.split(",") if k == "csa" else ["0"])]:
                layer = 0 if kind == "hca" else 1
                inp = H.make_inputs(rng, ls, kind, idx_corr=corr)
                g, ops = H.group_call(ls, kind, layer, inp, interpret=False, served_dtypes=True)
                res = g(*ops)
                n_c = len(ls.arrays)
                ok = True
                for jj in range(8):
                    f1, o1 = H.build_call(ls, kind, layer, inp, interpret=False, attn_heads=16,
                                          h0=16 * jj, served_dtypes=True)
                    r1 = jax.jit(f1)(*o1)
                    same_c = all(np.array_equal(words(c[jj]), words(c1))
                                 for c, c1 in zip(res[:n_c], r1[:n_c]))
                    same_o = np.array_equal(np.asarray(res[n_c][jj]), np.asarray(r1[n_c]))
                    ok &= same_c and same_o
                    if not (same_c and same_o):
                        print(f"MISMATCH rank {jj} caches={same_c} out={same_o}", flush=True)
                g11, ops11 = H.group_call(ls, kind, layer, inp, interpret=False, served_dtypes=True,
                                          reps=NREP)
                t1, t11 = np.median(timed(g, ops)), np.median(timed(g11, ops11))
                f1, o1 = H.build_call(ls, kind, layer, inp, interpret=False, attn_heads=16,
                                      served_dtypes=True)
                f11, o11 = H.build_call(ls, kind, layer, inp, interpret=False, attn_heads=16,
                                        served_dtypes=True, reps=NREP)
                s1, s11 = np.median(timed(jax.jit(f1), o1)), np.median(timed(jax.jit(f11), o11))
                from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP
                for ab in [x for x in a.ablate.split(",") if x]:
                    AP.ABLATE = frozenset(ab.split("+"))
                    ga, oa = H.group_call(ls, kind, layer, inp, interpret=False, served_dtypes=True)
                    ga11, oa11 = H.group_call(ls, kind, layer, inp, interpret=False,
                                              served_dtypes=True, reps=NREP)
                    ta = (np.median(timed(ga11, oa11)) - np.median(timed(ga, oa))) / (NREP - 1)
                    print(f"RESULT kind=attn_group_ablate skip={ab} R={R} ctx={ctx} corr={corr} "
                          f"tp8_us_per_layer={ta * 1e6:.1f}", flush=True)
                    AP.ABLATE = frozenset()
                c1_, o1_ = H.group_call(ls, kind, layer, inp, attn_tp=1, ndev=8, interpret=False,
                                        served_dtypes=True)
                c11_, o11_ = H.group_call(ls, kind, layer, inp, attn_tp=1, ndev=8,
                                          interpret=False, served_dtypes=True, reps=NREP)
                s8 = (np.median(timed(c11_, o11_)) - np.median(timed(c1_, o1_))) / (NREP - 1)
                print(f"RESULT kind=attn_group_single8 R={R} ctx={ctx} corr={corr} "
                      f"us_per_layer={s8 * 1e6:.1f}", flush=True)
                print(f"RESULT kind=attn_group layer={kind} R={R} ctx={ctx} corr={corr} "
                      f"tp8_us_per_layer={(t11 - t1) / (NREP - 1) * 1e6:.1f} "
                      f"single_us_per_layer={(s11 - s1) / (NREP - 1) * 1e6:.1f} bitexact_vs_single={ok}",
                      flush=True)


if __name__ == "__main__":
    main()
