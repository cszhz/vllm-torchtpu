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
"""TensorCore top-k vs a NumPy model of the SparseCore contract, index for index.

Runs the kernel in Pallas interpret mode on CPU. On device the same cases are
compared against `sparsecore_topk` itself (see the module docstring of
`tc_topk`).
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from topk_reference import (  # noqa: E402
    KINDS,
    make_case,
    monotone_key,
    reference_topk,
    reference_union,
    split_cuts,
)

from vllm_torchtpu.kernels.deepseek_v4_megakernel.tc_topk import (  # noqa: E402
    tc_topk,
    tc_topk_ext,
    tc_topk_union_grouped,
    to_group_layout,
)

K = 1024
N = 16384  # smallest supported row width; 128k context is n = 32768


def _run(x, lens, threshold="stack"):
    return np.asarray(
        tc_topk(jnp.asarray(x), K, jnp.asarray(lens), interpret=True, threshold=threshold)
    )


@pytest.mark.parametrize("threshold", ["stack", "radix"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rows", [7, 14])
def test_matches_reference(kind, rows, threshold):
    x, lens = make_case(kind, rows, N, np.random.default_rng(rows))
    np.testing.assert_array_equal(_run(x, lens, threshold), reference_topk(x, K, lens))


def test_56_rows_mixed_lengths():
    rng = np.random.default_rng(56)
    x, _ = make_case("random", 56, N, rng)
    lens = rng.integers(0, N + 1, size=56).astype(np.int32)
    np.testing.assert_array_equal(_run(x, lens), reference_topk(x, K, lens))


def test_check_bites():
    """Positive control: the comparison rejects a plausible wrong answer.

    Ties broken to the highest column instead of the lowest is the error a
    kernel that gets everything else right would make.
    """
    x, lens = make_case("ties", 7, N, np.random.default_rng(0))
    got = _run(x, lens)
    assert (reference_topk(x, K, lens, ties_to_lowest=False) != got).any()
    np.testing.assert_array_equal(got, reference_topk(x, K, lens))


def _check_union(x, lens, lists_too):
    rows = x.shape[0]
    lists, union, ulen = tc_topk_union_grouped(
        to_group_layout(jnp.asarray(x)), jnp.asarray(lens), k=K, interpret=True, lists=lists_too
    )
    ref = reference_topk(x, K, lens)
    if lists_too:
        np.testing.assert_array_equal(np.asarray(lists)[:rows], ref)
    union, ulen = np.asarray(union), np.asarray(ulen)
    for g in range(union.shape[0]):
        want = reference_union(ref[g * 8:(g + 1) * 8])
        got = union[g].reshape(-1)
        assert ulen[g] == want.size
        np.testing.assert_array_equal(got[: want.size], want)
        assert (got[want.size:] == -1).all()


@pytest.mark.parametrize("kind", KINDS)
def test_union_matches_reference(kind):
    """14 rows: one full 8-row group (member bits up to 255) and one with 6 rows."""
    x, lens = make_case(kind, 14, N, np.random.default_rng(14))
    _check_union(x, lens, lists_too=True)


@pytest.mark.parametrize("kind", ["random", "ties", "garbage_tail"])
def test_union_without_lists(kind):
    x, lens = make_case(kind, 7, N, np.random.default_rng(7))
    lens[3] = 0
    _check_union(x, lens, lists_too=False)


@pytest.mark.parametrize("kind", ["random", "ties", "short", "nan", "garbage_tail"])
def test_values_aligned_with_lists(kind):
    x, lens = make_case(kind, 7, N, np.random.default_rng(21))
    res = tc_topk_ext(to_group_layout(jnp.asarray(x)), jnp.asarray(lens), k=K, values=True,
                      interpret=True)
    ref = reference_topk(x, K, lens)
    keys = np.take_along_axis(monotone_key(x), np.maximum(ref, 0), axis=1)
    np.testing.assert_array_equal(np.asarray(res["lists"])[:7], ref)
    np.testing.assert_array_equal(np.asarray(res["values"])[:7],
                                  np.where(ref >= 0, keys, np.iinfo(np.int32).min))


SLICES = 8


def _sliced_union(x, lens, g, allow):
    """Union over SLICES contiguous column slices, each cut externally, in slice order."""
    n_loc = x.shape[1] // SLICES
    ents = []
    for q in range(SLICES):
        res = tc_topk_ext(
            to_group_layout(jnp.asarray(x[:, q * n_loc:(q + 1) * n_loc])),
            jnp.asarray(np.clip(lens - q * n_loc, 0, n_loc).astype(np.int32)),
            jnp.asarray(g), jnp.asarray(allow[q]), k=K, lists=False, union=True, interpret=True)
        u = np.asarray(res["union"])[0].reshape(-1)[: int(res["union_len"][0])].astype(np.int64)
        ents.append(((u >> 8) + q * n_loc) * 256 + (u & 255))
    return np.concatenate(ents)


@pytest.mark.parametrize("kind", KINDS)
def test_sliced_union_with_external_cut(kind):
    """8 slices of N columns, each cut by the global threshold and tie quota, reproduce the
    single-slice union exactly, including tie runs across slice edges."""
    x, lens = make_case(kind, 7, SLICES * N, np.random.default_rng(31))
    g, allow = split_cuts(x, K, lens, N, SLICES)
    want = reference_union(reference_topk(x, K, lens)).astype(np.int64)
    np.testing.assert_array_equal(_sliced_union(x, lens, g, allow), want)


def test_sliced_union_check_bites():
    """Positive control: taking every tie in every slice (ignoring the quota) is rejected."""
    x, lens = make_case("straddle", 7, SLICES * N, np.random.default_rng(31))
    g, allow = split_cuts(x, K, lens, N, SLICES)
    want = reference_union(reference_topk(x, K, lens)).astype(np.int64)
    got = _sliced_union(x, lens, g, np.full_like(allow, -1))
    assert got.size != want.size or (got != want).any()
