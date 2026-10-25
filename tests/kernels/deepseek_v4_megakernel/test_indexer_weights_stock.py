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
"""CSA indexer score weights: the megakernel against the stock arithmetic.

Stock: `weights_proj` is a bf16 linear, `rope_quant` gives the f32 per-head query
scales, and `_indexer_local` folds them (its statement is evaluated from the
stock source, `H.stock_indexer_weights`). Weights and query scales: bit-exact.
Scores: 1e-6 relative (f32 sums of exact fp8 products in a different order).
Controls: the f32 fold without the bf16 roundings must fail.
"""

import contextlib
import functools
import os
import subprocess
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_phase_harness as H  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.rope import rope_quant  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_ref  # noqa: E402

Q = H.Q


@contextlib.contextmanager
def _cpu_as_tpu7():
    """`rope_quant`'s pipeline reads the TPU generation; interpret it as tpu7x."""
    tpu7 = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7, 1)  # noqa: E731
    tpu_info.get_tpu_info.cache_clear()
    try:
        with (
            patch.dict(tpu_info.registry, {"cpu": tpu7}),
            pltpu.force_tpu_interpret_mode(),
        ):
            yield
    finally:
        tpu_info.get_tpu_info.cache_clear()


def _stock_q(idx_q, pos, cos_sin):
    """Stock `rope_quant` of [R, Q, 64 * 128] bf16 queries: (e4m3 values as f32
    [R, Q, 64, 128], f32 scales [R, Q, 64])."""
    R = idx_q.shape[0]
    with _cpu_as_tpu7():
        q8, qs = rope_quant(
            idx_q.reshape(R * Q, 64, 128),
            pos.reshape(-1).astype(jnp.int32),
            cos_sin,
            quant_dtype=jnp.float8_e4m3fn,
        )
    return q8.astype(jnp.float32).reshape(R, Q, 64, 128), qs.reshape(R, Q, 64)


def _f32_fold(w, qs):
    """The fold without the bf16 roundings (negative control)."""
    return ((w * AP.IDX_SOFTMAX_SCALE) * AP.IDX_HEAD_SCALE) * qs


def test_fold_matches_stock():
    rng = np.random.default_rng(0)
    R = 4
    idx_q = jnp.asarray(rng.standard_normal((R, Q, 64 * 128)), jnp.bfloat16)
    pos = jnp.asarray(rng.integers(0, 4000, (R, Q)))
    cos_sin = jnp.asarray(rng.uniform(-1, 1, (4096, 64)), jnp.float32)
    _, qs = _stock_q(idx_q, pos, cos_sin)
    # magnitudes from 1e-3 to 1e1: the f32 accumulator of the projection
    w = jnp.asarray(
        rng.standard_normal((R, Q, 64)) * 10.0 ** rng.uniform(-3, 1, (R, Q, 64)),
        jnp.float32,
    )
    want = np.asarray(H.stock_indexer_weights(w, qs))
    assert want.dtype == np.float32
    for got in (
        AP.fold_indexer_weights(w, qs),
        jax.jit(AP.fold_indexer_weights)(w, qs),
    ):
        assert np.asarray(got).dtype == np.float32
        np.testing.assert_array_equal(np.asarray(got), want)
    assert np.mean(np.asarray(_f32_fold(w, qs)) != want) > 0.5  # control


@functools.cache
def _csa_phase():
    """The CSA phase (interpret) with its debug outputs: (layer set, inputs,
    caches, scores, folded weights [R, Q, 64], query scales [R, Q, 64])."""
    rng = np.random.default_rng(3)
    ls = H.build(rng, ("hca", "csa"), [1030, 3000], 4096)
    inp = H.make_inputs(rng, ls, "csa")
    caches, _, (_, scores, _, dbg) = H.run_phase(ls, "csa", 1, inp, debug_topk=True)
    dbg = np.asarray(dbg)[:, :Q]
    return ls, inp, caches, scores, dbg[..., :64], dbg[..., 64:]


@pytest.fixture(scope="module")
def csa_phase():
    return _csa_phase()


def _positions(ls):
    return jnp.asarray(ls.seq_lens)[:, None] - Q + jnp.arange(Q)[None, :]


def _stock_inputs(ls, inp):
    q8, qs = _stock_q(inp["idx_q"][:, :Q], _positions(ls), ls.weights["idx_cos_sin"])
    return q8, qs, inp["idx_w"][:, :Q, :64]


def test_phase_weights_match_stock(csa_phase):
    """The fold as the kernel runs it, on its own query scales."""
    ls, inp, _, _, got, scales = csa_phase
    w = inp["idx_w"][:, :Q, :64]
    np.testing.assert_array_equal(got, np.asarray(H.stock_indexer_weights(w, scales)))
    assert np.mean(got != np.asarray(_f32_fold(w, scales))) > 0.5  # control


O0 = "--xla_backend_optimization_level=0"


def _check_q_scales():
    ls, inp, _, _, _, scales = _csa_phase()
    _, qs, _ = _stock_inputs(ls, inp)
    np.testing.assert_array_equal(scales, np.asarray(qs))
    _, qs1 = _stock_q(
        inp["idx_q"][:, :Q], _positions(ls) + 1, ls.weights["idx_cos_sin"]
    )
    # control: the next position's rotation (heads whose max is a NoPE lane keep theirs)
    assert np.mean(scales != np.asarray(qs1)) > 0.2


def test_phase_q_scales_match_stock():
    """Kernel query scales vs stock `rope_quant`, with the CPU backend at -O0:
    optimized, its code generation rounds the two interpreted RoPEs differently
    (1 ulp on a few % of the scales); at -O0 both are op-by-op f32 arithmetic."""
    if O0 in os.environ.get("XLA_FLAGS", ""):
        _check_q_scales()
        return
    env = dict(os.environ, XLA_FLAGS=f"{os.environ.get('XLA_FLAGS', '')} {O0}")
    test = f"{os.path.abspath(__file__)}::test_phase_q_scales_match_stock"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test],
        env=env,
        check=True,
    )


def _group_scores(debug_scores, r):
    """tc_topk group layout [S * 8, 128] of request r -> f32 [8, S * 128]."""
    g = np.asarray(debug_scores)[r]
    return g.reshape(-1, 8, 128).transpose(1, 0, 2).reshape(8, -1).view(np.float32)


def test_phase_scores_match_stock_weights(csa_phase):
    """Kernel scores vs scores of the stock queries and weights over the
    kernel-written indexer cache."""
    ls, inp, caches, scores, _, _ = csa_phase
    q8, qs, w = _stock_inputs(ls, inp)
    ai = ls.role_arr["indexer_k"][ls.role_layers["indexer_k"].index(1)]
    table = jnp.asarray(ls.tables[("indexer_k", 1)])
    seq = jnp.asarray(ls.seq_lens)

    def rel_err(weights):
        ref, _ = attn_ref.indexer_scores_reference(
            q8, weights, jnp.asarray(caches[ai]), table, seq
        )
        err = 0.0
        for r, s in enumerate(ls.seq_lens):
            a, b = (
                _group_scores(scores, r)[:Q, : s // 4],
                np.asarray(ref)[r, :, : s // 4],
            )
            fin = np.isfinite(b)
            err = max(err, np.abs(a[fin] - b[fin]).max() / np.abs(b[fin]).max())
        return err

    assert rel_err(H.stock_indexer_weights(w, qs)) <= 1e-6
    assert rel_err(_f32_fold(w, qs)) > 1e-5  # control
