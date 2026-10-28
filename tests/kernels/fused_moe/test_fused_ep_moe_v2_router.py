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
"""`pallas_select`'s two-table mode: select by one score table, weigh by another.

DeepSeek-V4's `noaux_tc` decides the top-k on scores that carry a per-expert
correction bias and then weighs the chosen experts by the scores WITHOUT it.
Folding the bias into the logits cannot express that, so the selector takes a
second [R, N] table and reads the selected weights from it.

The failure mode this file is built around is that the two tables come apart
silently. Selecting correctly and weighing from the wrong table is not an
error at any layer below -- every index is a real expert, every weight is a
plausible float, and the renormalization downstream rescales whatever it is
handed. What it produces is a model that routes right and weighs wrong, which
has no symptom other than quality. So the assertions here are equalities
against a numpy gather, not tolerances, and the bias in the fixtures is large
enough to REORDER the selection rather than perturb it -- a test whose bias
changed nothing would pass against a kernel that ignored the bias entirely.

`interpret=True` runs the kernel body on the host, which is what lets the
selection rule be gated without a chip, the same way `tests/kernels/
test_router_topk.py` gates the sibling selector. It does not gate Mosaic
lowering: `tests/kernels/fused_moe/test_fused_ep_moe_v2.py` is what runs this
code on a TPU, and it needs eight of them.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import router_ops  # noqa: E402

# Wide enough that a top-k spans several 128-lane blocks at the shapes the
# router actually sees, small enough to check exhaustively against numpy.
EXPERTS = 256
TOPK = 8


def _select(scores, weight_scores=None, topk=TOPK, block_rows=None):
    """The kernel on the host, as numpy."""
    rows = scores.shape[0]
    w, i = router_ops.pallas_select(
        jnp.asarray(scores),
        topk=topk,
        block_rows=rows if block_rows is None else block_rows,
        weight_scores=None if weight_scores is None else jnp.asarray(weight_scores),
        interpret=True,
    )
    return np.asarray(w), np.asarray(i)


def _reference(scores, weight_scores, topk=TOPK):
    """Top-k by `scores`, weights gathered from `weight_scores`.

    Ties go to the lowest column, which is the kernel's rule: it takes the
    row max and then the lowest column equal to it. `argsort(-x, stable)`
    is the same rule, and `np.argpartition` is NOT -- it is unordered within
    the partition, so it disagrees on exactly the rows that matter.
    """
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :topk]
    return np.take_along_axis(weight_scores, idx, axis=1), idx


def _scores(rows, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.random((rows, EXPERTS), dtype=np.float32) * scale).astype(np.float32)


@pytest.mark.parametrize(
    "rows,block_rows", [(8, 8), (256, 256), (512, 256), (1024, 256)]
)
def test_one_table_mode_is_unchanged(rows, block_rows):
    """The default path still returns the selected scores as the weights.

    The two-table mode was added by threading an optional ref through the
    same selection loop, so the one-table path is reachable code that no
    longer has a test of its own anywhere off-chip.
    """
    scores = _scores(rows, seed=rows)
    w, i = _select(scores, block_rows=block_rows)
    ref_w, ref_i = _reference(scores, scores)
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)


@pytest.mark.parametrize(
    "rows,block_rows", [(8, 8), (256, 256), (512, 256), (1024, 256)]
)
def test_weights_come_from_the_second_table(rows, block_rows):
    """Indices from `scores`, weights from `weight_scores`, exactly."""
    scores = _scores(rows, seed=rows)
    weights = _scores(rows, seed=rows + 1)
    w, i = _select(scores, weights, block_rows=block_rows)
    ref_w, ref_i = _reference(scores, weights)
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)


def test_the_bias_moves_the_selection_and_not_the_weights():
    """The `noaux_tc` call as the layer makes it, end to end.

    The bias is large against the score spread, so the biased top-k is a
    genuinely different expert set from the unbiased one -- asserted here,
    because a fixture where it is not would pass against a kernel that
    dropped the bias on the floor.
    """
    raw = _scores(64, seed=7)
    rng = np.random.default_rng(8)
    bias = (rng.random(EXPERTS, dtype=np.float32) * 4.0).astype(np.float32)
    biased = raw + bias[None, :]

    w, i = _select(biased, raw)
    ref_w, ref_i = _reference(biased, raw)
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)

    # The weights are the unbiased scores at the biased picks: neither the
    # biased scores, nor the weights the unbiased selection would have given.
    np.testing.assert_array_equal(w, np.take_along_axis(raw, i, axis=1))
    assert not np.array_equal(w, np.take_along_axis(biased, i, axis=1))
    _, unbiased_i = _reference(raw, raw)
    assert not np.array_equal(i, unbiased_i), (
        "the fixture's bias does not reorder the top-k, so this test cannot "
        "tell a kernel that applies it from one that ignores it"
    )


@pytest.mark.parametrize(
    "case", ["biased", "ties", "nonfinite", "tiny", "every_expert"]
)
@pytest.mark.cpu_test
def test_packed_top6_normalizes_unbiased_weights(case):
    """Packing must preserve expert order and normalize the unbiased table."""
    rng = np.random.default_rng(41)
    raw = rng.random((288, 384), dtype=np.float32)
    bias = rng.uniform(-3, 0, 384).astype(np.float32)
    if case == "ties":
        raw[:] = 1
        bias[:] = -100
        bias[[0, 127, 128, 255, 256, 383]] = 0
    elif case == "nonfinite":
        raw[0] = np.nan
        raw[1] = np.inf
        raw[2] = 0
        raw[3, ::3] = np.nan
        raw[4, ::3] = np.inf
        bias[::7] = np.nan
    elif case == "tiny":
        raw *= np.float32(1e-22)
    scores = raw + bias
    if case == "every_expert":
        # Exercise every expert ID, including adjacent IDs above 256.
        expected_ids = (np.arange(288)[:, None] * 6 + np.arange(6)) % 384
        scores[:] = -1
        np.put_along_axis(scores, expected_ids, np.arange(6, 0, -1)[None, :], axis=1)
    original_w, original_i = _select(scores, raw, topk=6, block_rows=32)
    with np.errstate(invalid="ignore"):
        expected = original_w / np.maximum(
            original_w.sum(axis=-1, keepdims=True), np.float32(1e-20)
        )
    weights, indices = router_ops.pallas_select(
        jnp.asarray(scores),
        topk=6,
        block_rows=288,
        weight_scores=jnp.asarray(raw),
        normalize=True,
        interpret=True,
    )
    np.testing.assert_array_equal(indices, original_i)
    np.testing.assert_allclose(weights, expected, rtol=2e-6, atol=1e-6)
    if case == "biased":
        _, unbiased_ids = _reference(raw, raw, topk=6)
        assert not np.array_equal(indices, unbiased_ids)
    elif case == "ties":
        np.testing.assert_array_equal(indices[0], [0, 127, 128, 255, 256, 383])
    elif case == "every_expert":
        np.testing.assert_array_equal(indices, expected_ids)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"rows": 256},
        {"experts": 128},
        {"topk": 3},
        {"scoring_fn": "softmax"},
        {"scoring_fn": "sigmoid"},
        {"renormalize": False},
        {"has_bias": False},
        {"has_hash": True},
        {"dtype": jnp.float16},
        {"backend": "cpu"},
        {"chip": router_ops.pltpu.ChipVersion.TPU_V6E},
    ],
)
@pytest.mark.cpu_test
def test_normalized_router_only_uses_measured_envelope(monkeypatch, change):
    """Never normalize hash/no-bias routes or widen unmeasured tiles/devices."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    config = dict(
        rows=288,
        experts=384,
        topk=6,
        scoring_fn="sqrtsoftplus",
        renormalize=True,
        has_bias=True,
        has_hash=False,
        dtype=jnp.bfloat16,
        backend="tpu",
        chip=router_ops.pltpu.ChipVersion.TPU_7X,
    )
    config.update(change)
    monkeypatch.setattr(router_ops.jax, "default_backend", lambda: config["backend"])
    monkeypatch.setattr(
        router_ops.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(chip_version=config["chip"]),
    )
    monkeypatch.setattr(router_ops, "logger", Mock())
    gate = jax.ShapeDtypeStruct((config["rows"], config["experts"]), config["dtype"])
    tile = router_ops.normalized_router_block_rows(
        gate,
        topk=config["topk"],
        scoring_fn=config["scoring_fn"],
        renormalize=config["renormalize"],
        has_bias=config["has_bias"],
        has_hash=config["has_hash"],
    )
    assert tile == (None if change else 288)


@pytest.mark.parametrize("topk,block_rows", [(3, 288), (6, 32)])
@pytest.mark.cpu_test
def test_normalized_selector_rejects_unsupported_tiles(topk, block_rows):
    scores = jnp.ones((288, 384), jnp.float32)
    with pytest.raises(ValueError, match="Normalized selection requires"):
        router_ops.pallas_select(
            scores,
            topk=topk,
            block_rows=block_rows,
            weight_scores=scores,
            normalize=True,
            interpret=True,
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("adapter_widen", [False, True])
@pytest.mark.parametrize("near_ties", [False, True])
def test_normalized_route_stage_keeps_scaling_and_invalid_sentinels(
    monkeypatch, adapter_widen, near_ties
):
    """Exercise production wiring without running experts or TPU collectives."""
    import functools
    from unittest.mock import Mock, patch

    from vllm_torchtpu.kernels.fused_moe.v2 import host, kernel, layer

    info = router_ops.pltpu.get_tpu_info_for_chip(
        router_ops.pltpu.ChipVersion.TPU_7X, 1
    )
    monkeypatch.setattr(router_ops.pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(kernel, "logger", Mock())
    monkeypatch.setattr(
        layer,
        "pallas_select",
        functools.partial(router_ops.pallas_select, interpret=True),
    )
    # Isolate the layer's function cache from other tests and from this A/B.
    monkeypatch.setattr(layer, "_LAYER_SM_CACHE", {})
    with jax.default_device(jax.devices("cpu")[0]):
        rng = np.random.default_rng(43)
        gate = jnp.asarray(rng.normal(size=(288, 384)), dtype=jnp.bfloat16)
        gate = gate.at[:3].set(
            jnp.array([jnp.nan, -jnp.inf, jnp.inf], jnp.bfloat16)[:, None]
        )
        if adapter_widen:
            # fused_moe_ep._build_op.call passes this widened form to the layer.
            gate = gate.astype(jnp.float32)
        if near_ties:
            # Adjacent logits and cancelling bias expose changed Top6 membership.
            ranks = rng.permutation(384).astype(np.float32)
            near = np.float32(0.25) + ranks * np.spacing(np.float32(0.25))
            if not adapter_widen:
                near = np.float32(0.25) + ranks * np.float32(2**-11)
            gate = jnp.broadcast_to(jnp.asarray(near, gate.dtype), (288, 384))
            gate = gate.at[:3].set(
                jnp.array([jnp.nan, -jnp.inf, jnp.inf], gate.dtype)[:, None]
            )
            old_scores = layer._apply_scoring_fn(gate[3], "sqrtsoftplus")
            bias = -old_scores + jnp.asarray((np.arange(384) % 7) * 2**-23, jnp.float32)
        else:
            bias = jnp.asarray(rng.normal(size=(384,)), dtype=jnp.float32)
        mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:1]), ("d",))
        args = (
            jnp.zeros((288, 128), jnp.bfloat16),
            jnp.zeros((384, 128, 256), jnp.bfloat16),
            jnp.zeros((384, 128, 128), jnp.bfloat16),
            None,
            None,
            gate,
        )
        selected_tiles = []

        def production_tile(gate, **kwargs):
            with patch.object(router_ops.jax, "default_backend", return_value="tpu"):
                tile = router_ops.normalized_router_block_rows(gate, **kwargs)
            selected_tiles.append(tile)
            return tile

        outputs = []
        for choose_tile in (lambda *a, **kw: None, production_tile):
            layer._LAYER_SM_CACHE.clear()
            monkeypatch.setattr(layer, "normalized_router_block_rows", choose_tile)
            outputs.append(
                tuple(
                    np.asarray(value)
                    for value in layer.fused_ep_moe_v2(
                        *args,
                        topk=6,
                        renormalize=True,
                        mesh=mesh,
                        capacity=128,
                        weight_format=host.WeightFormat.BF16,
                        scoring_fn="sqrtsoftplus",
                        score_bias=bias,
                        routed_scaling_factor=2.5,
                        stage="route",
                    )
                )
            )
        assert selected_tiles and all(tile == 288 for tile in selected_tiles)
        (old_wire, old_weights, old_payload), (wire, weights, payload) = outputs
        np.testing.assert_array_equal(wire, old_wire)
        np.testing.assert_array_equal(payload, old_payload)
        np.testing.assert_allclose(weights, old_weights, rtol=2e-6, atol=1e-6)
        np.testing.assert_array_equal(weights[:3], 0)
        np.testing.assert_array_equal(payload[0, : 3 * 6], 384)
        np.testing.assert_allclose(weights[3:].sum(axis=-1), 2.5, rtol=2e-6)


def test_packed_top6_tpu_lowering_matches_numpy():
    """Run the normalized selector through Mosaic on one supported device."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires a TPU7x device")
    if (
        router_ops.pltpu.get_tpu_info().chip_version
        != router_ops.pltpu.ChipVersion.TPU_7X
    ):
        pytest.skip("normalized selector is calibrated on TPU7x")
    rng = np.random.default_rng(42)
    raw = rng.random((288, 384), dtype=np.float32)
    biased = raw + rng.uniform(-3, 0, 384).astype(np.float32)
    biased[0] = 1
    expected_weights, expected_ids = _reference(biased, raw, topk=6)
    expected_weights /= expected_weights.sum(axis=-1, keepdims=True)
    weights, indices = jax.jit(
        lambda scores, values: router_ops.pallas_select(
            scores, topk=6, block_rows=288, weight_scores=values, normalize=True
        )
    )(jnp.asarray(biased), jnp.asarray(raw))
    np.testing.assert_array_equal(indices, expected_ids)
    np.testing.assert_allclose(weights, expected_weights, rtol=2e-6, atol=1e-6)


@pytest.mark.cpu_test
def test_fp32_sqrtsoftplus_validity_matches_generic_row_check():
    """FP32 admission must preserve invalid rows without rounding logits to BF16."""
    from vllm_torchtpu.kernels.fused_moe.v2.layer import _apply_scoring_fn, _row_routes

    rng = np.random.default_rng(47)
    bits = rng.integers(0, 2**32, (288, 384), dtype=np.uint32)
    values = bits.view(np.float32)
    extremes = np.array(
        [
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            0,
            -100,
            np.nan,
            np.inf,
            -np.inf,
            np.finfo(np.float32).tiny,
        ],
        dtype=np.float32,
    )
    values[:8] = extremes[:, None]
    values[8:16] = np.nan
    values[8:16, 383] = extremes
    with jax.default_device(jax.devices("cpu")[0]):
        gate = jnp.asarray(values)
        scores = _apply_scoring_fn(gate, "sqrtsoftplus")
        original = _row_routes(scores, gate)
        simplified = jnp.any(jnp.isfinite(gate), axis=-1, keepdims=True)
        np.testing.assert_array_equal(simplified, original)
        assert np.isfinite(np.asarray(scores)[0]).all()


@pytest.mark.cpu_test
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_specialized_softplus_preserves_finite_and_nonfinite_scores(dtype):
    """The measured formula must retain overflow safety and invalid-row semantics."""
    from vllm_torchtpu.kernels.fused_moe.v2.layer import _apply_scoring_fn

    rng = np.random.default_rng(48)
    bits = rng.integers(0, 2**32, 4096, dtype=np.uint32)
    edge = np.array(
        [
            0.0,
            -0.0,
            np.nan,
            np.inf,
            -np.inf,
            100,
            -100,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
        ],
        np.float32,
    )
    logits = jnp.asarray(np.concatenate((edge, bits.view(np.float32))), dtype)
    generic = _apply_scoring_fn(logits, "sqrtsoftplus")
    specialized = _apply_scoring_fn(logits, "sqrtsoftplus", specialize_softplus=True)
    np.testing.assert_array_equal(np.isnan(specialized), np.isnan(generic))
    np.testing.assert_array_equal(np.isposinf(specialized), np.isposinf(generic))
    np.testing.assert_allclose(specialized, generic, rtol=2e-6, atol=1e-6)
