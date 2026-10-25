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
"""CSA attention with each request's indexer scan + top-k split across an
8-rank attention group (attn_tp=8, attn_heads=16), CPU interpret on 8 devices:
every rank's output equals the single-rank phase on the same 16 heads
bit for bit, caches too."""

import os
import sys

import jax

jax.config.update("jax_num_cpu_devices", 8)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import attn_phase_harness as H  # noqa: E402

KINDS = ("hca", "csa", "hca", "csa")


def _group_vs_single(ls, layer, inp):
    group = H.run_group(ls, "csa", layer, inp)
    for jj, (caches, out) in enumerate(group):
        c1, o1 = H.run_phase(ls, "csa", layer, inp, attn_heads=16, h0=16 * jj)
        for a, b in zip(caches, c1):
            assert np.array_equal(a, np.asarray(b)), f"rank {jj}: caches differ"
        o1 = np.asarray(o1, np.float32)
        assert np.array_equal(out, o1), (jj, np.abs(out - o1).max())


@pytest.mark.parametrize("seq", [[70000, 3000], [9000, 300]])
def test_split_topk_bitexact(seq):
    """70000: every rank scores columns; 9000: ranks 2..7 have none."""
    rng = np.random.default_rng(31)
    ls = H.build(rng, KINDS, seq, max(seq) + 1024)
    inp = H.make_inputs(rng, ls, "csa")
    _group_vs_single(ls, 3, inp)


def test_split_topk_ties_across_ranks():
    """Every indexer page identical: each score repeats once a page, so the
    top-1024 cut falls inside tie runs spread over all 8 ranks."""
    rng = np.random.default_rng(32)
    ls = H.build(rng, KINDS, [70000], 71680)
    a = ls.role_arr["indexer_k"][ls.role_layers["indexer_k"].index(3)]
    arr = np.asarray(ls.arrays[a]).copy()
    nkv = ls.tables[("indexer_k", 3)].max() + 1
    arr[1:nkv] = arr[0]
    ls.arrays[a] = arr
    inp = H.make_inputs(rng, ls, "csa")
    _group_vs_single(ls, 3, inp)
