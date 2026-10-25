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
"""Megakernel pieces against the served stock code they replace (CPU interpret).

The stock side is imported and called, never re-derived. A negative control
shows each assertion can fail.
"""

import os

# As the other checks here: keep XLA from folding bf16 round trips away (the
# negative controls would then fail rather than pass silently).
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"
).strip()

import functools  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils as mhc_utils  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_ring as DR  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel.dense_config import (  # noqa: E402
    DenseChainConfig,
)

F32, BF16 = jnp.float32, jnp.bfloat16
CFG = DenseChainConfig()  # hidden 7168, 16 rows


def _bf16_bits(x):
    return np.asarray(jax.lax.bitcast_convert_type(x.astype(BF16), jnp.int16), np.int32)


def _ulp_diff(a, b):
    """Per-element distance in bf16 ulps (sign-magnitude bit patterns)."""

    def ordered(bits):
        return np.where(bits < 0, -(bits & 0x7FFF), bits)

    return np.abs(ordered(_bf16_bits(a)) - ordered(_bf16_bits(b)))


# ---- RMSNorm of the layer input (attn_norm / ffn_norm) ---------------------


def _norm_weight(rng):
    """A norm weight near 1 with spread and outlier channels."""
    w = 1.0 + 0.3 * rng.standard_normal(CFG.hidden)
    w[rng.choice(CFG.hidden, 32, replace=False)] *= 8.0
    return jnp.asarray(w, BF16)


def _norm_inputs(seed):
    """An mHC layer input: bf16, per-row scales over 3 decades, a few
    outlier channels."""
    rng = np.random.default_rng(seed)
    m, h = CFG.rows, CFG.hidden
    x = rng.standard_normal((m, h)) * np.exp(rng.uniform(-4.6, 2.3, (m, 1)))
    x[:, rng.choice(h, 16, replace=False)] *= 30.0
    return jnp.asarray(x, BF16), _norm_weight(rng)


def _exact_sum_inputs(seed):
    """x = k * 2**e_row with |k| <= 15: every partial sum of squares is an
    integer below 2**24 times 2**(2 e_row), exact in f32 in any order, so the
    kernel and stock see the same rsqrt and only their rounding can differ."""
    rng = np.random.default_rng(seed)
    m, h = CFG.rows, CFG.hidden
    x = rng.integers(-15, 16, (m, h)) * np.exp2(rng.integers(-10, 4, (m, 1)))
    return jnp.asarray(x, BF16), _norm_weight(rng)


def _smalls(name, w):
    """One layer's f32 small-vector block with ``name`` set to w (as
    ``dense_weights.pack_smalls`` lays it out)."""
    row, n = CFG.smalls_layout[name]
    flat = jnp.zeros((CFG.smalls_rows * 128,), F32)
    flat = flat.at[row * 128 : row * 128 + n].set(w.astype(F32))
    return flat.reshape(1, CFG.smalls_rows, 128)


@functools.partial(jax.jit, static_argnames=("weight_dtype",))
def _kernel_norm(x, smalls, *, weight_dtype):
    """The kernel's own ``rms_norm_to`` on attn_norm, in a pallas_call."""
    m, h = x.shape

    def body(x_ref, sm_ref, o_ref, amax_ref):
        ops = DR.make_dense_ops(
            CFG, None, DR.Faults(), sm_v=sm_ref, res=None, mixst=None,
            tsc=None, xin=None, hin=None, xa=None, xq=None, acc=None, q=None,
            xg=None, z=None, cos_sin=None,
        )  # fmt: skip
        amax_ref[...] = ops.rms_norm_to(
            x_ref, 0, h, o_ref, 0, "attn_norm", weight_dtype=weight_dtype
        )

    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((m, h), BF16),
            jax.ShapeDtypeStruct((m, 1), F32),
        ),
        interpret=True,
    )(x, smalls)


@jax.jit
def _stock_norm(x, w):
    return mhc_utils.finish_layer_input(x, w.reshape(1, -1), CFG.rms_eps)


def _norm_case(inputs, weight_dtype):
    x, w = inputs
    out, amax = _kernel_norm(x, _smalls("attn_norm", w), weight_dtype=weight_dtype)
    return out, amax, _stock_norm(x, w)


SEEDS = (0, 1, 2, 3)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_input_norm_rounding_matches_stock(seed):
    """With an order-independent sum of squares the kernel's attn_norm is
    bit-identical to stock finish_layer_input, and its amax is that of the
    bf16 output (what the next fp8 quantization must see)."""
    out, amax, ref = _norm_case(_exact_sum_inputs(seed), F32)
    assert int(_ulp_diff(out, ref).max()) == 0
    np.testing.assert_array_equal(
        np.asarray(amax),
        np.max(np.abs(np.asarray(ref, np.float32)), axis=1, keepdims=True),
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_input_norm_matches_stock(seed):
    """On general data the kernel's chunked f32 sum of squares is summed in
    a different order from stock's: a rare 1-ulp flip is all that is left."""
    out, _, ref = _norm_case(_norm_inputs(seed), F32)
    ulp = _ulp_diff(out, ref)
    print(
        f"seed {seed}: {np.mean(ulp > 0):.2e} of elements differ, max {ulp.max()} ulp"
    )
    assert ulp.max() <= 1 and np.mean(ulp > 0) < 1e-3


@pytest.mark.parametrize("make_inputs", [_exact_sum_inputs, _norm_inputs])
def test_negative_control_two_roundings(make_inputs):
    """Rounding x * rsqrt to bf16 before a bf16 weight multiply (vLLM
    rms_norm as written in torch, and this kernel before) misses stock on a
    large fraction of elements."""
    out, _, ref = _norm_case(make_inputs(SEEDS[0]), BF16)
    frac = float(np.mean(_ulp_diff(out, ref) > 0))
    print(f"two roundings: {frac:.3f} of elements differ from stock")
    assert frac > 0.1
