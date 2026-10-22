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
"""GLM5_MKD_STACK_C on TPU: stacked == per-layer, bitwise, for the DCP ops.

A small multi-layer GLM-5.2-shaped setup (page 256, interleave 4, SparseCore
MLA caches, SAL indexer cache, top-k 2048, 64 query heads) runs three steps --
a prefill, a mixed MTP-verify-sized step and a decode step -- through the
real DCP indexer step (`streamidx_topk_dcp_step`: SAL Pallas append +
`streamindex_topk_dcp`) feeding the real DCP sparse attention
(`sparse_mla_attention_dcp`: SparseCore writer + gather kernel), once on
per-layer caches and once on [n_layers, num_blocks, ...] stacks shared by every
layer (`kv_stack=(row, num_blocks)`). Every output and every cache must match
bit for bit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    KVLayout,
    get_kv_cache_shape,
)
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)

pytestmark = [pytest.mark.tpu_test, pytest.mark.multichip]

P = jax.sharding.PartitionSpec

N_LAYERS = 3
NUM_BLOCKS = 10  # per rank
PAGE = 256
INTERLEAVE = 4
PAGES_PER_SEQ = 2
LKV, ROPE, HEADS = 512, 64, 64
H_IDX, D_IDX, TOPK = 32, 128, 2048
SM_SCALE = 1.0 / 16

# (q_lens, seq_lens, request_distribution, padded token count)
STEPS = (
    ([3000, 700, 300, 0], [3000, 700, 300, 0], [0, 0, 3], 4096),  # prefill
    ([4, 64, 4, 0], [3004, 764, 304, 0], [0, 0, 3], 128),  # mixed / verify
    ([1, 1, 1, 0], [3005, 765, 305, 0], [3, 3, 3], 16),  # decode
)


def _dcp_size():
    n = len(jax.devices())
    d = 8 if n >= 8 else (4 if n >= 4 else 2)
    if n < 2:
        pytest.skip(f"needs >= 2 devices, have {n}")
    return d


def _specs():
    nope = SparseMLAKVCacheSpec.create(
        KVCacheType.NOPE, KVCacheLayout.SPARSECORE, NUM_BLOCKS, PAGE, LKV, 4
    )
    rope = SparseMLAKVCacheSpec.create(
        KVCacheType.ROPE, KVCacheLayout.SPARSECORE, NUM_BLOCKS, PAGE, ROPE, 4
    )
    return nope, rope


def _step_inputs(step, layer, q_lens, num_tokens):
    rng = np.random.default_rng(1000 * step + layer)
    f = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731
    fp8 = lambda x: jnp.asarray(x).astype(jnp.float8_e4m3fn)  # noqa: E731
    kq = jax.lax.bitcast_convert_type(fp8(f(num_tokens, D_IDX)), jnp.uint8)
    k_packed = jnp.concatenate([kq, jnp.full((num_tokens, 1), 127, jnp.uint8)], axis=-1)
    return dict(
        q_bytes=jax.lax.bitcast_convert_type(
            fp8(f(num_tokens, H_IDX, D_IDX) * 0.5), jnp.uint8
        ),
        weights=jnp.asarray(f(num_tokens, H_IDX)).astype(jnp.bfloat16),
        k_packed=k_packed,
        ql_nope=jnp.asarray(f(num_tokens, HEADS, LKV)).astype(jnp.bfloat16),
        q_pe=jnp.asarray(f(num_tokens, HEADS, ROPE)).astype(jnp.bfloat16),
        kv_c=fp8(f(num_tokens, LKV)),
        k_pe=fp8(f(num_tokens, ROPE)),
    )


def test_stacked_caches_match_per_layer_bitwise():
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        streamidx_topk_dcp_step,
    )
    from vllm_torchtpu.layers.core.attention_interface import (
        sparse_mla_attention_dcp,
    )

    dcp = _dcp_size()
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp]), (DCP_AXIS_NAME,))
    sharded = jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME))
    nope_spec, rope_spec = _specs()
    idx_shape = get_kv_cache_shape(NUM_BLOCKS, PAGE, D_IDX, KVLayout.SEQ_ALONG_LANE)

    # Virtual block table, replicated; one id past the pool (resolved by `%`).
    table = np.random.default_rng(7).permutation(NUM_BLOCKS)[: 4 * PAGES_PER_SEQ]
    table = table.astype(np.int32).reshape(4, PAGES_PER_SEQ)
    table[1, 1] += NUM_BLOCKS
    block_tables = jnp.asarray(table.reshape(-1))

    def zeros(per_layer_shape, dtype, stacked):
        lead = (dcp * N_LAYERS,) if stacked else (dcp * per_layer_shape[0],)
        rest = per_layer_shape if stacked else per_layer_shape[1:]
        return jax.device_put(jnp.zeros(lead + tuple(rest), dtype), sharded)

    caches = {}
    for stacked in (False, True):
        if stacked:
            caches[stacked] = [
                zeros(nope_spec.shape, nope_spec.jax_dtype, True),
                zeros(rope_spec.shape, rope_spec.jax_dtype, True),
                zeros(idx_shape, jnp.uint8, True),
            ]
        else:
            caches[stacked] = [
                [
                    zeros(nope_spec.shape, nope_spec.jax_dtype, False),
                    zeros(rope_spec.shape, rope_spec.jax_dtype, False),
                    zeros(idx_shape, jnp.uint8, False),
                ]
                for _ in range(N_LAYERS)
            ]

    nontrivial_topk = False
    for step, (q_lens, seq_lens, rd, num_tokens) in enumerate(STEPS):
        qsl = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
        seq_lens_a = jnp.asarray(seq_lens, jnp.int32)
        dist_attn = jnp.asarray(rd, jnp.int32)
        dist_idx = jnp.asarray([rd[0], rd[0], rd[2]], jnp.int32)
        for layer in range(N_LAYERS):
            x = _step_inputs(step, layer, q_lens, num_tokens)
            outs = {}
            for stacked in (False, True):
                if stacked:
                    nope, rope, idx = caches[True]
                    kv_stack = (layer, NUM_BLOCKS)
                else:
                    nope, rope, idx = caches[False][layer]
                    kv_stack = None
                idx, local_topk = streamidx_topk_dcp_step(
                    x["q_bytes"],
                    x["weights"],
                    idx,
                    x["k_packed"],
                    seq_lens_a,
                    block_tables,
                    qsl,
                    dist_idx,
                    mesh=mesh,
                    topk=TOPK,
                    dcp_size=dcp,
                    interleave_size=INTERLEAVE,
                    chunk_tokens=None,
                    kv_cache_mode="sal",
                    kv_stack=kv_stack,
                )
                nope, rope, out, lse = sparse_mla_attention_dcp(
                    x["ql_nope"],
                    x["q_pe"],
                    x["kv_c"],
                    x["k_pe"],
                    nope,
                    rope,
                    local_topk,
                    seq_lens_a,
                    block_tables,
                    qsl,
                    dist_attn,
                    mesh,
                    nope_spec,
                    rope_spec,
                    sm_scale=SM_SCALE,
                    k_scale=1.0,
                    dcp_size=dcp,
                    interleave_size=INTERLEAVE,
                    kv_stack=kv_stack,
                )
                if stacked:
                    caches[True] = [nope, rope, idx]
                else:
                    caches[False][layer] = [nope, rope, idx]
                # Valid token rows only: rows past the batch are padding the
                # kernel leaves unspecified.
                total = int(sum(q_lens))
                outs[stacked] = [
                    np.asarray(a).reshape(dcp, num_tokens, *a.shape[1:])[:, :total]
                    for a in (local_topk, out, lse)
                ]
            for name, a, b in zip(
                ("local_topk", "out", "lse"), outs[False], outs[True]
            ):
                np.testing.assert_array_equal(
                    b, a, err_msg=f"step {step} layer {layer}: {name}"
                )
            topk = outs[False][0]
            assert (topk >= 0).any(), f"step {step} layer {layer}: empty top-k"
            assert np.isfinite(outs[False][1].astype(np.float32)).all()
            if step == 0:
                # The 3000-token row has more context than k: selection is real.
                per_tok = (topk >= 0).sum(axis=(0, 2))
                nontrivial_topk |= bool((per_tok[: q_lens[0]] == TOPK).any())
    assert nontrivial_topk, "no row selected a strict top-k subset"

    # Final caches: the stack's row l on every rank == layer l's cache.
    for kind in range(3):
        stack = np.asarray(caches[True][kind])
        stack = stack.reshape(dcp, N_LAYERS, *stack.shape[1:])
        for layer in range(N_LAYERS):
            per = np.asarray(caches[False][layer][kind])
            per = per.reshape(dcp, *stack.shape[2:])
            assert per.any(), f"kind {kind} layer {layer}: nothing written"
            np.testing.assert_array_equal(
                stack[:, layer], per, err_msg=f"cache kind {kind} layer {layer}"
            )
