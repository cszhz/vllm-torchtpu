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
"""moe_routed_ffn must match gmm_v2_ffn over the sorted routes plus the combine."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox.gmm_v2_ffn import gmm_v2_ffn
from vllm_torchtpu.kernels.megablox.moe_routed_ffn import moe_routed_ffn

NUM_EXPERTS, EP, TOPK, SIZE_K, SIZE_I = 8, 4, 4, 512, 256


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _grouped_reference(x, ids, weights, w1, w1_scale, w2, w2_scale):
    """The grouped-GEMM path: sort routes by expert, run gmm_v2_ffn, combine."""
    t_idx, k_idx = np.nonzero(ids >= 0)
    order = np.argsort(ids[t_idx, k_idx], kind="stable")
    t_idx, k_idx = t_idx[order], k_idx[order]
    experts = ids[t_idx, k_idx]
    group_sizes = np.bincount(experts, minlength=NUM_EXPERTS).astype(np.int32)
    out = np.zeros((x.shape[0], SIZE_K), np.float32)
    if len(t_idx) == 0:
        return out
    rows = gmm_v2_ffn(
        x[t_idx],
        w1,
        w2,
        jnp.asarray(group_sizes),
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        fuse_act="silu",
        preferred_element_type=jnp.bfloat16,
    )
    rows = np.asarray(rows[: len(t_idx)].astype(jnp.float32))
    w = np.asarray(weights.astype(jnp.float32))[t_idx, k_idx]
    np.add.at(out, t_idx, rows * w[:, None])
    return out


@pytest.mark.parametrize("num_tokens", [4, 8, 16, 32, 48, 160, 512])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("rank", [0, 2])
@pytest.mark.parametrize("concentrated", [False, True])
def test_moe_routed_ffn_matches_grouped_path(num_tokens, seed, rank, concentrated):
    _require_tpu()
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((num_tokens, SIZE_K)), jnp.bfloat16)
    w1 = jnp.asarray(
        rng.standard_normal((NUM_EXPERTS, SIZE_K, 2 * SIZE_I)) * 0.5, jnp.float8_e4m3fn
    )
    w2 = jnp.asarray(
        rng.standard_normal((NUM_EXPERTS, SIZE_I, SIZE_K)) * 0.5, jnp.float8_e4m3fn
    )
    w1_scale = jnp.asarray(
        rng.uniform(0.02, 0.2, (NUM_EXPERTS, 1, 1, 2 * SIZE_I)), jnp.float32
    )
    w2_scale = jnp.asarray(
        rng.uniform(0.01, 0.1, (NUM_EXPERTS, 1, 1, SIZE_K)), jnp.float32
    )
    # Global top-k over EP * NUM_EXPERTS experts; this rank owns block `rank`.
    global_ids = np.stack(
        [rng.permutation(EP * NUM_EXPERTS)[:TOPK] for _ in range(num_tokens)]
    ).astype(np.int32)
    if concentrated:
        # Warmup's identical dummy tokens: every token on the same experts, so
        # an expert's routes span several row chunks.
        global_ids[:] = global_ids[0]
    weights = rng.uniform(0, 0.5, (num_tokens, TOPK))
    weights[num_tokens * 3 // 4 :] = 0  # padded tokens
    weights = jnp.asarray(weights, jnp.bfloat16)
    local_ids = global_ids - rank * NUM_EXPERTS
    is_local = (local_ids >= 0) & (local_ids < NUM_EXPERTS)
    ids = np.where(is_local & (np.asarray(weights) != 0), local_ids, -1)

    got = moe_routed_ffn(
        x,
        jnp.asarray(global_ids),
        weights,
        w1,
        w1_scale,
        w2,
        w2_scale,
        jnp.int32(rank * NUM_EXPERTS),
        fuse_act="silu",
        skip_zero_weights=True,
    )
    want = _grouped_reference(x, ids, weights, w1, w1_scale, w2, w2_scale)

    got = np.asarray(got.astype(jnp.float32))
    want = np.asarray(jnp.asarray(want).astype(jnp.bfloat16).astype(jnp.float32))
    # Only the f32 order of a token's expert sum differs from the grouped path.
    np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2 * np.abs(want).max())
    assert np.mean(got == want) > 0.99
    assert np.all(got[(ids < 0).all(axis=1)] == 0)


@pytest.mark.parametrize("num_tokens", [16, 128, 512])
@pytest.mark.parametrize("ranks", [1, 8])
def test_prequantized_dispatch_is_bitexact(num_tokens, ranks):
    """Rows quantized per rank and dispatched give the bf16 path's exact output."""
    _require_tpu()
    from vllm_torchtpu.kernels.megablox import moe_local_dispatch as dispatch
    from vllm_torchtpu.kernels.megablox.moe_routed_ffn import (
        _f32_scale,
        make_routed_ffn_configs,
        quantize_rows,
    )

    rng = np.random.default_rng(num_tokens)
    x = rng.standard_normal((num_tokens, SIZE_K)) * rng.choice(
        [1e-3, 1, 30], (num_tokens, 1)
    )
    x[1] = 0  # zero block scale
    x = jnp.asarray(x, jnp.bfloat16)
    w1 = jnp.asarray(
        rng.standard_normal((NUM_EXPERTS, SIZE_K, 2 * SIZE_I)) * 0.5, jnp.float8_e4m3fn
    )
    w2 = jnp.asarray(
        rng.standard_normal((NUM_EXPERTS, SIZE_I, SIZE_K)) * 0.5, jnp.float8_e4m3fn
    )
    w1_scale = jnp.asarray(
        rng.uniform(0.02, 0.2, (NUM_EXPERTS, 1, 1, 2 * SIZE_I)), jnp.float32
    )
    w2_scale = jnp.asarray(
        rng.uniform(0.01, 0.1, (NUM_EXPERTS, 1, 1, SIZE_K)), jnp.float32
    )
    ids = np.stack(
        [rng.permutation(EP * NUM_EXPERTS)[:TOPK] for _ in range(num_tokens)]
    ).astype(np.int32)
    ids = jnp.asarray(ids)
    weights = jnp.asarray(rng.uniform(0, 0.5, (num_tokens, TOPK)), jnp.bfloat16)
    start = jnp.int32(NUM_EXPERTS)
    lhs = make_routed_ffn_configs(
        x, w1, _f32_scale(w1_scale), w2, _f32_scale(w2_scale), fuse_act="silu"
    )[0].lhs_cfgs

    words = []
    for part in np.split(np.arange(num_tokens), ranks):
        x_q, scales = quantize_rows(
            x[part], q_dtype=lhs.quant_dtype, block=lhs.quant_block_size
        )
        meta = dispatch._meta(scales, ids[part], weights[part], x.dtype)
        words.append(dispatch._pack_words(x_q, meta))
    rows = dispatch._unpack_words(
        jnp.concatenate(words), SIZE_K, lhs.quant_dtype, x.dtype
    )
    _, got_ids, got_weights = dispatch.split_dispatched(rows, SIZE_K, TOPK)
    np.testing.assert_array_equal(np.asarray(got_ids), np.asarray(ids))

    want = moe_routed_ffn(
        x, ids, weights, w1, w1_scale, w2, w2_scale, start, fuse_act="silu"
    )
    got = moe_routed_ffn(
        rows,
        got_ids,
        got_weights,
        w1,
        w1_scale,
        w2,
        w2_scale,
        start,
        fuse_act="silu",
        prequantized=True,
    )
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
    )
