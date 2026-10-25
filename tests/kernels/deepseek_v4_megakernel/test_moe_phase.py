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
"""CPU-interpret check of step_phases.moe_phase on 16 simulated ranks (two
8-rank groups, so the cross-group all-reduce round runs), hash and noaux_tc
layers, a non-prefix live-row pattern, per-rank shared-expert seeds."""

import contextlib

import jax
import pytest

with contextlib.suppress(RuntimeError):
    jax.config.update("jax_num_cpu_devices", 16)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import (
    moe_reference as ref,  # noqa: E402
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel.allreduce import (  # noqa: E402
    reference_allreduce,
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel.moe_experts import (  # noqa: E402
    BF16,
    F32,
    pack_fp4_rows,
    pad_scale_rows,
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel.moe_phase import (
    phase_config,  # noqa: E402
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel.moe_phase_harness import (  # noqa: E402
    phase_harness,
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel.step_phases import (  # noqa: E402
    StepGeometry,
)

R, L, T = 16, 4, 16
pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu" or len(jax.devices()) < R,
    reason="needs 16 CPU devices (JAX_PLATFORMS=cpu)",
)


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize(
    "layer,shared_ar,stacked,hidden,inter",
    [
        (1, False, False, 1024, 3072),
        (3, False, True, 1024, 3072),
        (3, True, False, 1024, 3072),
        (3, True, False, 512, 512),  # the step check's small widths (64-column slices)
    ],
)  # hash and noaux_tc layers; stacked weights; the skeleton's shared all-reduce
def test_moe_phase_matches_reference(layer, shared_ar, stacked, hidden, inter):
    geom = StepGeometry(
        R=R,
        T=T,
        q=7,
        B=1,
        layer_kinds=("hca",) * L,
        hidden=hidden,
        n_experts=128,
        moe_inter=inter,
    )
    cfg = phase_config(geom)
    mesh = jax.make_mesh((R,), ("d",))
    keys = jax.random.split(jax.random.key(0), L + 6)
    weights = [ref.synthetic_weights(keys[i], cfg) for i in range(L)]
    x = (jax.random.normal(keys[L], (T, hidden), F32) * 2).astype(BF16)
    logits = jax.random.normal(keys[L + 1], (T, 128), F32)
    bias = jax.random.normal(keys[L + 2], (L, 128), F32) * 0.1
    seeds = jax.random.normal(keys[L + 3], (R, T, hidden), F32) * 0.1
    hids = jax.random.randint(keys[L + 4], (3, T, 8), 0, 128, jnp.int32)
    valid = (np.arange(T) % 8 < 7).astype(
        np.int32
    )  # 7 live rows per 8-row request block
    valid[13] = 0

    ep, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    ops = []
    for w1, w1s, w2, w2s in weights:
        ops += [
            pack_fp4_rows(w1),
            pad_scale_rows(w1s),
            pack_fp4_rows(w2),
            pad_scale_rows(w2s),
        ]
    l0 = layer - 1 if shared_ar else layer  # two layers: the slots alternate twice
    out = phase_harness(
        mesh,
        geom,
        l0,
        layer + 1,
        interpret=pltpu.InterpretParams(),
        shared_ar=shared_ar,
        stacked=stacked,
    )(
        jax.device_put(jnp.arange(R, dtype=jnp.int32).reshape(R, 1), ep),
        jax.device_put(x, rep),
        jax.device_put(logits, rep),
        jax.device_put(seeds.reshape(R * T, hidden), ep),
        jax.device_put(hids.reshape(3, T * 8), rep),
        jax.device_put(jnp.asarray(valid), rep),
        jax.device_put(bias, rep),
        *(
            [
                jax.device_put(jnp.stack(ops[j::4]), NamedSharding(mesh, P(None, "d")))
                for j in range(4)
            ]
            if stacked
            else [jax.device_put(a, ep) for a in ops]
        ),
    )
    out = np.asarray(out).reshape(R, T, hidden)

    masked = jnp.where(jnp.asarray(valid)[:, None] > 0, logits, -jnp.inf)
    if layer < 3:
        sids, swts = ref.stock_route(masked, cfg, T, hash_ids=hids[layer, :, :6])
    else:
        sids, swts = ref.stock_route(masked, cfg, T, bias=bias[layer])
    x32 = jnp.zeros((cfg.rows, hidden), BF16).at[:T].set(x)
    sids32 = jnp.full((cfg.rows, 6), 128, jnp.int32).at[:T].set(sids)
    swts32 = jnp.zeros((cfg.rows, 6), F32).at[:T].set(swts)
    parts = jnp.stack(
        [
            ref.model_partial(x32, sids32, swts32, *weights[layer], cfg, r)[:T]
            + seeds[r]
            for r in range(R)
        ]
    )
    want = np.asarray(reference_allreduce(parts, "hrsag", R, 8).astype(BF16)).astype(
        np.float32
    )
    for r in range(R):
        np.testing.assert_array_equal(out[r], out[0])
    assert _rel(out[0], want) < 1e-3


def _random_placement(rng, L, R, E, copies):
    """Per layer: a random permutation of the experts over the ranks, and with
    `copies` one more slot per rank holding a copy of another rank's expert."""
    from vllm_torchtpu.expert_placement import parse

    el = E // R
    layers = {}
    for li in range(L):
        perm = rng.permutation(E)
        ranks = [list(map(int, perm[r * el : (r + 1) * el])) for r in range(R)]
        if copies:  # every rank one more slot: a second copy of a distinct expert
            hot, used = rng.choice(E, 2 * R, replace=False), set()
            for r in range(R):
                extra = next(int(e) for e in hot if e not in used and e not in ranks[r])
                used.add(extra)
                ranks[r].append(extra)
        layers[str(li)] = ranks
    return parse({"experts": E, "shards": R, "layers": layers})


@pytest.mark.parametrize("copies", [False, True])
def test_moe_phase_with_expert_placement(copies):
    from vllm_torchtpu.kernels.deepseek_v4_megakernel.moe_phase import placement_tables

    hidden, inter, layer = 1024, 3072, 3
    geom = StepGeometry(R=R, T=T, q=7, B=1, layer_kinds=("hca",) * L, hidden=hidden,
                        n_experts=128, moe_inter=inter)  # fmt: skip
    cfg = phase_config(geom)
    mesh = jax.make_mesh((R,), ("d",))
    rng = np.random.default_rng(5)
    place = _random_placement(rng, L, R, 128, copies)
    held, copy = placement_tables(place, L, R, 128)
    assert (copy is not None) == copies
    keys = jax.random.split(jax.random.key(1), L + 4)
    weights = [ref.synthetic_weights(keys[i], cfg) for i in range(L)]
    x = (jax.random.normal(keys[L], (T, hidden), F32) * 2).astype(BF16)
    logits = jax.random.normal(keys[L + 1], (T, 128), F32)
    bias = jax.random.normal(keys[L + 2], (L, 128), F32) * 0.1
    seeds = jax.random.normal(keys[L + 3], (R, T, hidden), F32) * 0.1
    valid = np.ones(T, np.int32)
    ep, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    ops = []
    for li, (w1, w1s, w2, w2s) in enumerate(
        weights
    ):  # each rank's slots, in slot order
        order = held[:, li].reshape(-1)
        ops += [pack_fp4_rows(w1)[order], pad_scale_rows(w1s)[order],
                pack_fp4_rows(w2)[order], pad_scale_rows(w2s)[order]]  # fmt: skip
    run = phase_harness(mesh, geom, layer, layer + 1, interpret=pltpu.InterpretParams(),
                        placed=True, copies=copies)  # fmt: skip

    def call(tables):
        out = run(
            jax.device_put(jnp.arange(R, dtype=jnp.int32).reshape(R, 1), ep),
            jax.device_put(x, rep), jax.device_put(logits, rep),
            jax.device_put(seeds.reshape(R * T, hidden), ep),
            jax.device_put(jnp.zeros((3, T * 8), jnp.int32), rep),
            jax.device_put(jnp.asarray(valid), rep), jax.device_put(bias, rep),
            *[jax.device_put(jnp.asarray(t), ep) for t in tables],
            *[jax.device_put(a, ep) for a in ops],
        )  # fmt: skip
        return np.asarray(out).reshape(R, T, hidden)

    out = call([held] + ([copy] if copies else []))
    sids, swts = ref.stock_route(logits, cfg, T, bias=bias[layer])
    x32 = jnp.zeros((cfg.rows, hidden), BF16).at[:T].set(x)
    sids32 = jnp.full((cfg.rows, 6), 128, jnp.int32).at[:T].set(sids)
    swts32 = jnp.zeros((cfg.rows, 6), F32).at[:T].set(swts)
    parts = jnp.stack([
        ref.model_partial(
            x32, sids32, swts32, *weights[layer], cfg, r, held=held[r, layer],
            copy=None if copy is None else copy[r, layer],
        )[:T] + seeds[r]
        for r in range(R)
    ])  # fmt: skip
    want = np.asarray(reference_allreduce(parts, "hrsag", R, 8).astype(BF16)).astype(
        np.float32
    )
    for r in range(R):
        np.testing.assert_array_equal(out[r], out[0])
    assert _rel(out[0], want) < 1e-3
    if not copies:  # positive control: the table must match the slot order
        wrong = call([np.roll(held, 1, axis=2)])
        assert _rel(wrong[0], want) > 100 * 1e-3


@pytest.mark.parametrize("T,layer", [(64, 1), (64, 3), (128, 1)])
def test_moe_phase_compact_bit_identical(T, layer):
    """MoEConfig.compact on vs off over skewed routing (one expert on more
    than 32 rows, experts on 0 and 1 rows), hash and noaux_tc layers: every
    output bit equal; a row scattered to the wrong accumulator row is caught."""
    import dataclasses

    from vllm_torchtpu.kernels.deepseek_v4_megakernel import moe_experts

    hidden, inter = 512, 512
    geom = StepGeometry(R=R, T=T, q=7, B=1, layer_kinds=("hca",) * L, hidden=hidden,
                        n_experts=128, moe_inter=inter)  # fmt: skip
    cfg = phase_config(geom)
    mesh = jax.make_mesh((R,), ("d",))
    keys = jax.random.split(jax.random.key(2), L + 6)
    weights = [ref.synthetic_weights(keys[i], cfg) for i in range(L)]
    x = (jax.random.normal(keys[L], (T, hidden), F32) * 2).astype(BF16)
    logits = jax.random.normal(keys[L + 1], (T, 128), F32)
    bias = (jax.random.normal(keys[L + 2], (L, 128), F32) * 0.1).at[:, 21].set(5.0)
    seeds = jax.random.normal(keys[L + 3], (R, T, hidden), F32) * 0.1
    hids = jax.random.randint(keys[L + 4], (3, T, 8), 0, 128, jnp.int32)
    hids = hids.at[:, : T // 2 + 8, 0].set(21)  # expert 21 on > 32 rows
    valid = (np.arange(T) % 8 < 7).astype(np.int32)
    ep, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    ops = []
    for w1, w1s, w2, w2s in weights:
        ops += [pack_fp4_rows(w1), pad_scale_rows(w1s), pack_fp4_rows(w2),
                pad_scale_rows(w2s)]  # fmt: skip
    args = (
        jax.device_put(jnp.arange(R, dtype=jnp.int32).reshape(R, 1), ep),
        jax.device_put(x, rep), jax.device_put(logits, rep),
        jax.device_put(seeds.reshape(R * T, hidden), ep),
        jax.device_put(hids.reshape(3, T * 8), rep),
        jax.device_put(jnp.asarray(valid), rep), jax.device_put(bias, rep),
        *[jax.device_put(a, ep) for a in ops],
    )  # fmt: skip

    def run(compact, fault=""):
        g = dataclasses.replace(geom, moe_compact=compact)
        assert phase_config(g).compact == compact
        old, moe_experts.COMPACT_FAULT = moe_experts.COMPACT_FAULT, fault
        try:
            out = phase_harness(mesh, g, layer, layer + 1,
                                interpret=pltpu.InterpretParams())(*args)  # fmt: skip
        finally:
            moe_experts.COMPACT_FAULT = old
        return np.asarray(out).reshape(R, T, hidden)

    masked = jnp.where(jnp.asarray(valid)[:, None] > 0, logits, -jnp.inf)
    if layer < 3:
        sids, swts = ref.stock_route(masked, cfg, T, hash_ids=hids[layer, :, :6])
    else:
        sids, swts = ref.stock_route(masked, cfg, T, bias=bias[layer])
    live = np.asarray(sids)[valid > 0].reshape(-1)
    counts = np.bincount(live, minlength=128)[:128]
    assert counts.max() > 32 and (counts == 1).any() and (counts == 0).any()
    off, on = run(False), run(True)
    np.testing.assert_array_equal(on, off)
    rows = cfg.rows
    x32 = jnp.zeros((rows, hidden), BF16).at[:T].set(x)
    sids32 = jnp.full((rows, 6), 128, jnp.int32).at[:T].set(sids)
    swts32 = jnp.zeros((rows, 6), F32).at[:T].set(swts)
    parts = jnp.stack([
        ref.model_partial(x32, sids32, swts32, *weights[layer], cfg, r)[:T] + seeds[r]
        for r in range(R)
    ])  # fmt: skip
    want = np.asarray(reference_allreduce(parts, "hrsag", R, 8).astype(BF16))
    assert _rel(on[0], want.astype(np.float32)) < 1e-3
    if (T, layer) == (64, 1):
        assert not np.array_equal(run(True, fault="row"), off)
