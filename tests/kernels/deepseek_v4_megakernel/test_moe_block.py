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
"""CPU-interpret checks of the DSv4 megakernel MoE block on 8 simulated ranks.

Run with JAX_PLATFORMS=cpu. The kernel is checked against moe_reference's
model of its own arithmetic (reassociation only, so a tight tolerance), its
routing against the stock selector bit for bit, and a positive control shows
the tolerance catches a wrong scale block.
"""

import contextlib

import jax
import pytest

with contextlib.suppress(RuntimeError):  # backends already up: skipif below decides
    jax.config.update("jax_num_cpu_devices", 8)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import (
    moe_reference as ref,  # noqa: E402
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import moe_router as rt  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel.allreduce import (  # noqa: E402
    AR_MODES,
    AllReduceSpec,
    allreduce_call,
    alltoall_call,
    reference_allreduce,
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel.moe_experts import (  # noqa: E402
    BF16,
    F32,
    MoEConfig,
    moe_layer_call,
    pack_fp4_rows,
    pad_scale_rows,
    slice_expert_weights,
    slice_shared_weights,
    weight_spec,
)

R = 8
pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu" or len(jax.devices()) < R,
    reason="needs 8 CPU devices (JAX_PLATFORMS=cpu)",
)
INTERPRET = pltpu.InterpretParams()
# Kernel vs model of its own arithmetic: reassociated f32 sums and the odd
# fp8 rounding flip they cause -> ~1e-4 measured; 1e-3 leaves margin while the
# positive control (a swapped w2 scale block) lands near 3e-1.
MODEL_RTOL = 1e-3


def _mesh():
    return jax.make_mesh((R,), ("d",))


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("mode", AR_MODES)
def test_allreduce_matches_model_and_replicas_agree(mode):
    mesh = _mesh()
    rows, hidden = 16, 1024
    x = jax.random.normal(jax.random.key(0), (R * rows, hidden), F32)
    y = allreduce_call(mesh, AllReduceSpec(R, rows, hidden, mode), interpret=INTERPRET)(
        jax.device_put(x, NamedSharding(mesh, P("d")))
    )
    y = np.asarray(y).reshape(R, rows, hidden)
    want = np.asarray(reference_allreduce(x.reshape(R, rows, hidden), mode, R)) / R
    for r in range(R):
        np.testing.assert_array_equal(y[r], want)


def _logits(seed, rows, experts):
    return jax.random.normal(jax.random.key(seed), (rows, experts), F32)


@pytest.mark.parametrize("live", [32, 7, 0])
def test_router_selection_is_the_stock_selection(live):
    cfg = MoEConfig(R=R, n_experts=128, hidden=1024, w1_chunks=2, w2_cols=512)
    logits = _logits(1, 32, 128)
    # ties across a group, decided only by the lowest-index rule
    logits = logits.at[:, 5].set(logits[:, 3]).at[:, 77].set(logits[:, 3])
    bias = jax.random.normal(jax.random.key(2), (1, 128), F32) * 0.1
    bias = bias.at[0, jnp.array([3, 5, 77])].set(5.0)
    ids, ws = rt.route(
        logits,
        bias=bias,
        live_rows=live,
        topk=6,
        n_experts=128,
        routed_scaling_factor=cfg.routed_scaling_factor,
    )
    sids, sws = ref.stock_route(logits, cfg, live, bias=bias)
    np.testing.assert_array_equal(np.concatenate(ids, 1), np.asarray(sids))
    np.testing.assert_allclose(
        np.concatenate(ws, 1), np.asarray(sws), rtol=2e-7, atol=0
    )
    if live:
        np.testing.assert_array_equal(
            np.concatenate(ids, 1)[:live, :3], [[3, 5, 77]] * live
        )


def test_hash_router_matches_stock():
    cfg = MoEConfig(R=R, n_experts=128, hidden=1024, w1_chunks=2, w2_cols=512)
    logits = _logits(3, 32, 128)
    hid = jax.random.randint(jax.random.key(4), (32, 6), 0, 128, jnp.int32)
    ids, ws = rt.route(
        logits,
        hash_ids=[hid[:, k : k + 1] for k in range(6)],
        live_rows=20,
        topk=6,
        n_experts=128,
        routed_scaling_factor=cfg.routed_scaling_factor,
    )
    sids, sws = ref.stock_route(logits, cfg, 20, hash_ids=hid)
    np.testing.assert_array_equal(np.concatenate(ids, 1), np.asarray(sids))
    np.testing.assert_allclose(
        np.concatenate(ws, 1), np.asarray(sws), rtol=2e-7, atol=0
    )


def test_slice_layout_round_trips():
    cfg = MoEConfig(R=R, hidden=1024, n_experts=128, w1_chunks=2, w2_cols=512)
    w = [a[:3] for a in ref.synthetic_weights(jax.random.key(0), cfg)]
    back = ref.logical_experts(*slice_expert_weights(*w, R=R), cfg)
    for e in range(3):
        for a, b in zip(w, back):
            np.testing.assert_array_equal(
                np.asarray(a[e].astype(F32)), np.asarray(b[e])
            )


def test_hrsag_across_groups_matches_model():
    mesh = _mesh()
    spec = AllReduceSpec(R, 16, 1024, "hrsag", group=4)
    x = jax.random.normal(jax.random.key(5), (R * 16, 1024), F32)
    y = allreduce_call(mesh, spec, interpret=INTERPRET)(
        jax.device_put(x, NamedSharding(mesh, P("d")))
    )
    want = np.asarray(reference_allreduce(x.reshape(R, 16, 1024), "hrsag", R, 4)) / R
    for r in range(R):
        np.testing.assert_array_equal(np.asarray(y).reshape(R, 16, 1024)[r], want)


def test_alltoall_transposes_blocks():
    mesh = _mesh()
    rows, width = 16, 256
    x = jax.random.normal(jax.random.key(6), (R * R * rows, width)).astype(BF16)
    y = alltoall_call(mesh, rows, width, n_iter=3, interpret=INTERPRET)(
        jax.device_put(x, NamedSharding(mesh, P("d")))
    )
    want = np.asarray(x.astype(F32)).reshape(R, R, rows, width).transpose(1, 0, 2, 3)
    np.testing.assert_array_equal(
        np.asarray(y.astype(F32)).reshape(R, R, rows, width), want
    )


def _layer(mode, T, w2_scale_swap=False, ar_mode="flat", layout="sliced"):
    cfg = MoEConfig(
        R=R,
        mode=mode,
        rows=32,
        ar_rows=16 if T <= 16 else 32,
        hidden=1024,
        inter=3072,
        n_experts=128,
        w1_chunks=2,
        w2_cols=512,
        ar_mode=ar_mode,
        layout=layout,
    )
    mesh = _mesh()
    w = ref.synthetic_weights(jax.random.key(0), cfg)
    if layout == "stock":  # the served operands (fp4 as its packed-u32 bits)
        w1, w1s, w2, w2s = w
        wp = [
            pack_fp4_rows(w1),
            pad_scale_rows(w1s),
            pack_fp4_rows(w2),
            pad_scale_rows(w2s),
        ]
    else:
        wp = list(slice_expert_weights(*w, R=R))
    if w2_scale_swap:
        wp[3] = wp[3][:, :, ::-1]
    x = (jax.random.normal(jax.random.key(1), (cfg.rows, cfg.hidden), F32) * 2).astype(
        BF16
    )
    gate_w = (
        jax.random.normal(jax.random.key(2), (cfg.hidden, cfg.n_experts), F32) / 32
    ).astype(BF16)
    bias = jax.random.normal(jax.random.key(3), (1, cfg.n_experts), F32) * 0.1
    ws = NamedSharding(mesh, weight_spec(mode, "d"))
    rep = NamedSharding(mesh, P())
    meta = jnp.stack([jnp.arange(R, dtype=jnp.int32), jnp.full((R,), T, jnp.int32)], 1)
    out, ids, wts = moe_layer_call(mesh, cfg, interpret=INTERPRET)(
        jax.device_put(meta, NamedSharding(mesh, P("d"))),
        jax.device_put(x, rep),
        jax.device_put(gate_w, rep),
        jax.device_put(bias, rep),
        *[jax.device_put(a, ws) for a in wp],
    )
    out, ids, wts = (np.asarray(a).reshape(R, cfg.rows, -1) for a in (out, ids, wts))
    sids, swts = ref.stock_route(ref.gate_logits(x, gate_w), cfg, T, bias=bias)
    return cfg, out, ids, wts, (x, sids, swts, w)


@pytest.mark.parametrize(
    "mode,T,ar_mode,layout",
    [
        ("tp", 7, "flat", "sliced"),
        ("ep", 7, "flat", "sliced"),
        ("tp", 28, "pair", "sliced"),
        ("ep", 28, "rsag", "stock"),
    ],
)
def test_moe_layer_matches_model(mode, T, ar_mode, layout):
    cfg, out, ids, wts, (x, sids, swts, w) = _layer(
        mode, T, ar_mode=ar_mode, layout=layout
    )
    for r in range(R):
        np.testing.assert_array_equal(ids[r, :, :6], np.asarray(sids))
        np.testing.assert_array_equal(out[r], out[0])
    model = ref.model_moe(x, sids, swts, *w, cfg)
    assert _rel(out[0][:T], np.asarray(model)[:T]) < MODEL_RTOL
    # and no worse than the stock path against the unquantized layer
    exact = np.asarray(ref.exact_moe(x, sids, swts, *w, cfg))[:T]
    stock = np.asarray(ref.stock_experts(x, sids, swts, *w, cfg))[:T]
    assert _rel(out[0][:T], exact) <= _rel(stock, exact)


def test_positive_control_wrong_scale_block_fails():
    cfg, out, _, _, (x, sids, swts, w) = _layer("tp", 7, w2_scale_swap=True)
    model = ref.model_moe(x, sids, swts, *w, cfg)
    assert _rel(out[0][:7], np.asarray(model)[:7]) > 100 * MODEL_RTOL


def test_ep_stock_layout_with_shared_expert():
    """L4e's shape: routed experts in the served layout, the shared expert's
    1/R slice added to the same all-reduce."""
    T = 7
    cfg = MoEConfig(
        R=R,
        mode="ep",
        layout="stock",
        rows=32,
        ar_rows=16,
        hidden=1024,
        inter=3072,
        n_experts=128,
        w1_chunks=2,
        ar_mode="rsag",
        shared=True,
    )
    mesh = _mesh()
    w1, w1s, w2, w2s = ref.synthetic_weights(jax.random.key(0), cfg)
    shared = slice_shared_weights(*ref.synthetic_shared(jax.random.key(7), cfg), R=R)
    x = (jax.random.normal(jax.random.key(1), (32, 1024), F32) * 2).astype(BF16)
    gate_w = (jax.random.normal(jax.random.key(2), (1024, 128), F32) / 32).astype(BF16)
    bias = jax.random.normal(jax.random.key(3), (1, 128), F32) * 0.1
    rep, ep = NamedSharding(mesh, P()), NamedSharding(mesh, P("d"))
    meta = jnp.stack([jnp.arange(R, dtype=jnp.int32), jnp.full((R,), T, jnp.int32)], 1)
    routed = (
        pack_fp4_rows(w1),
        pad_scale_rows(w1s),
        pack_fp4_rows(w2),
        pad_scale_rows(w2s),
    )
    out, _, _ = moe_layer_call(mesh, cfg, interpret=INTERPRET)(
        jax.device_put(meta, ep),
        jax.device_put(x, rep),
        jax.device_put(gate_w, rep),
        jax.device_put(bias, rep),
        *[jax.device_put(a, ep) for a in routed + shared],
    )
    out = np.asarray(out).reshape(R, 32, -1)
    sids, swts = ref.stock_route(ref.gate_logits(x, gate_w), cfg, T, bias=bias)
    model = ref.model_moe(x, sids, swts, w1, w1s, w2, w2s, cfg, shared=shared)
    for r in range(R):
        np.testing.assert_array_equal(out[r], out[0])
    assert _rel(out[0][:T], np.asarray(model)[:T]) < MODEL_RTOL


def _compact_layer(rows, T, compact, fault=""):
    """EP stock-layout layer at `rows` compute rows, routing skewed so one
    expert takes every live row (> 32) while others take 0 or 1."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import moe_experts

    cfg = MoEConfig(R=R, mode="ep", layout="stock", rows=rows, ar_rows=rows,
                    hidden=1024, inter=3072, n_experts=128, w1_chunks=2,
                    ar_mode="rsag", compact=compact)  # fmt: skip
    mesh = _mesh()
    w1, w1s, w2, w2s = ref.synthetic_weights(jax.random.key(0), cfg)
    wp = [
        pack_fp4_rows(w1),
        pad_scale_rows(w1s),
        pack_fp4_rows(w2),
        pad_scale_rows(w2s),
    ]
    x = (jax.random.normal(jax.random.key(1), (rows, 1024), F32) * 2).astype(BF16)
    gate_w = (jax.random.normal(jax.random.key(2), (1024, 128), F32) / 32).astype(BF16)
    bias = jax.random.normal(jax.random.key(3), (1, 128), F32) * 0.1
    bias = bias.at[0, 21].set(5.0)  # expert 21 (rank 1) on every live row
    rep, ep = NamedSharding(mesh, P()), NamedSharding(mesh, P("d"))
    meta = jnp.stack([jnp.arange(R, dtype=jnp.int32), jnp.full((R,), T, jnp.int32)], 1)
    old, moe_experts.COMPACT_FAULT = moe_experts.COMPACT_FAULT, fault
    try:
        out, ids, _ = moe_layer_call(mesh, cfg, interpret=INTERPRET)(
            jax.device_put(meta, ep), jax.device_put(x, rep),
            jax.device_put(gate_w, rep), jax.device_put(bias, rep),
            *[jax.device_put(a, ep) for a in wp],
        )  # fmt: skip
    finally:
        moe_experts.COMPACT_FAULT = old
    sids, swts = ref.stock_route(ref.gate_logits(x, gate_w), cfg, T, bias=bias)
    out = np.asarray(out).reshape(R, rows, -1)
    return (
        cfg,
        out,
        np.asarray(ids).reshape(R, rows, -1),
        (x, sids, swts, (w1, w1s, w2, w2s)),
    )


@pytest.mark.parametrize("rows,T", [(64, 61), (128, 120)])
def test_compact_rows_bit_identical(rows, T):
    """Flag on vs off: every output bit equal, with an expert over 32 rows and
    experts at 0 and 1 rows; the reference still holds; a row scattered to
    the wrong accumulator row is caught (TPU: step_device2 compact_check)."""
    _, off, ids_off, (x, sids, swts, w) = _compact_layer(rows, T, False)
    cfg, on, ids_on, _ = _compact_layer(rows, T, True)
    counts = np.bincount(np.asarray(sids)[:T].reshape(-1), minlength=129)[:128]
    assert counts.max() > 32 and (counts == 1).any() and (counts == 0).any()
    np.testing.assert_array_equal(ids_on, ids_off)
    np.testing.assert_array_equal(on, off)
    model = ref.model_moe(x, sids, swts, *w, cfg)
    assert _rel(on[0][:T], np.asarray(model)[:T]) < MODEL_RTOL
    _, bad, _, _ = _compact_layer(rows, T, True, fault="row")
    assert not np.array_equal(bad, off)
