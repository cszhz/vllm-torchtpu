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
"""The weight-stream decode projection (`kernel_stream`) against the emit_pipeline one.

With the stream's k tile equal to the pipeline kernel's (3584) and bf16 weights
the arithmetic is the same -- per-half dots, `0 + dot` then `+=` over k, the
same APE matmul -- so every cache array must match bit for bit. Other k tiles
and fp8 weights change rounding: those are checked against an f64 oracle of the
state the projection stores.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    compressor_v1,
    proj_and_save_state as pk,
)

from .compressor_prefill_paths_test import VARIANTS, _build, _run

pytestmark = pytest.mark.tpu_test

BASE = dict(proj_runs_min_tokens=128)


def _fp8(w):
    wf = w.astype(jnp.float32)
    s = jnp.max(jnp.abs(wf), axis=0) / 448.0
    s = jnp.where(s == 0, 1.0, s)
    return jnp.clip(wf / s[None, :], -448.0, 448.0).astype(jnp.float8_e4m3fn), s


def _states(args, **kw):
    """Run only the projection; return (stored f32 state per real token, f64 oracle)."""
    head, ratio, overlap, _ = None, args["compress_ratio"], args["overlap"], None
    t2r, slots, _ = compressor_v1.derive_metadata(
        positions=args["positions"],
        block_table=args["block_table"],
        query_start_loc=args["query_start_loc"],
        kv_block_table=args["kv_block_table"],
        cache=args["cache"],
        compress_ratio=ratio,
        state_block_size=args["state_block_size"],
        head_dim=args["head_dim"],
        overlap=overlap,
        cos_sin_cache=args["cos_sin_cache"],
        state_cache=args["state_cache"],
        seq_along_lane=args["seq_along_lane"],
    )
    n_real = int(args["query_start_loc"][args["distribution"][2]])
    slots = jnp.where(jnp.arange(slots.shape[0]) < n_real, slots, -1)
    tgt = args["cache"] if args["state_cache"] is None else args["state_cache"]
    w, ws = args["wkv_wgate"], None
    if kw.pop("fp8", False):
        w, ws = _fp8(w)
    out = jax.jit(
        lambda x, w, ws, a, p, s, c: pk.proj_and_save_state(
            x, w, a, p, s, c, compress_ratio=ratio, wkv_scale=ws, **kw
        )
    )(args["hidden_states"], w, ws, args["ape"], args["positions"], slots, tgt)
    out = np.asarray(out)
    lanes, page_rows = out.shape[-1], out.shape[1]
    sw = args["wkv_wgate"].shape[1] // 2
    rows = 2 * sw // lanes
    flat = out.reshape(out.shape[0] * page_rows, -1).view(np.uint8)
    x = np.asarray(args["hidden_states"].astype(jnp.float32)).astype(np.float64)
    wf = np.asarray(args["wkv_wgate"].astype(jnp.float32)).astype(np.float64)
    ref = x @ wf
    pos = np.asarray(args["positions"])
    ref[:, sw:] += np.asarray(args["ape"]).astype(np.float64)[pos % ratio]
    got, want = [], []
    for t, s0 in enumerate(np.asarray(slots)):
        if s0 < 0:
            continue
        g = flat[s0 : s0 + rows]
        if out.dtype == np.uint8:  # byte-transposed rows
            g = g.reshape(rows, 4, lanes).transpose(0, 2, 1)
        got.append(g.copy().view(np.float32).reshape(-1))
        want.append(ref[t])
    return np.stack(got), np.stack(want)


class CompressorProjStreamTest(jtu.JaxTestCase):
    def setUp(self):
        super().setUp()
        if not jtu.test_device_matches(["tpu"]):
            self.skipTest("TPU only")

    @parameterized.product(
        variant=tuple(VARIANTS),
        decode=((1, 7), (2, 7), (4, 7), (8, 7), (3, 2)),
        nbuf=(2, 1),
    )
    def test_bit_exact_at_pipeline_tile(self, variant, decode, nbuf):
        args = _build(variant, decode, ())
        ref = _run(args, **BASE)
        new = _run(
            args, **BASE, proj_stream_max_tokens=64, proj_stream_cfg=(3584, nbuf, 1)
        )
        for r, n in zip(ref, new):
            if r is not None:
                self.assertArraysEqual(r, n)
        inputs = (args["cache"], args["rope_cache"], args["state_cache"])
        self.assertTrue(
            any(
                r is not None and not np.array_equal(r, np.asarray(i))
                for r, i in zip(ref, inputs)
            )
        )

    @parameterized.product(
        variant=tuple(VARIANTS),
        decode=((2, 7), (8, 7)),
        cfg=((512, 6, 1), (1024, 4, 2), (256, 8, 1)),
        fp8=(False, True),
    )
    def test_oracle(self, variant, decode, cfg, fp8):
        args = _build(variant, decode, ())
        got, want = _states(
            args, stream_max_tokens=64, stream_cfg=cfg, fp8=fp8
        )
        base, _ = _states(args)
        scale = np.abs(want).max(axis=1, keepdims=True)
        err = np.abs(got - want) / scale
        err_base = np.abs(base - want) / scale
        # bf16 weights: same distance to f64 as the pipeline kernel.
        # fp8 (per-channel e4m3, 3 mantissa bits): a few % of the row max.
        tol = 5e-2 if fp8 else max(4 * err_base.max(), 1e-5)
        self.assertLess(err.max(), tol)
        if fp8:
            # Positive control: the weight really is quantized.
            self.assertGreater(err.max(), 10 * err_base.max())


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
