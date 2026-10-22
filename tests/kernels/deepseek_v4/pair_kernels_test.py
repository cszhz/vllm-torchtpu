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
"""The chip-pair wo_a kernel against the XLA exchange + op it replaces, bitwise."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4.o_projection import (
    fused_reverse_rope_wo_a_projection,
    fused_reverse_rope_wo_a_projection_pair_swap,
)

from .rope_test import make_cos_sin_cache

pytestmark = pytest.mark.tpu_test

# DeepSeek-V4-Pro under intra-chip TP: 128 heads of 512, 8 heads per group,
# o_lora_rank 1024; each core holds 8 of the 16 wo_a groups.
NUM_HEADS = 128
HEAD_DIM = 512
LORA_RANK = 1024
ROTARY_DIM = 64
MAX_POSITION = 4096


def _pair_mesh():
    return Mesh(np.asarray(jax.devices()[:2]), ("x",))


def _per_core(kernel, *stacked):
    """`kernel` on each core of the pair; operand i of core r is stacked[i][r]."""
    mesh = _pair_mesh()
    spec = P("x")
    args = [
        jax.device_put(jnp.concatenate(list(a)), NamedSharding(mesh, spec))
        for a in stacked
    ]
    out = jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(spec,) * len(args),
        out_specs=spec,
        check_vma=False,
    )(*args)
    return np.asarray(out).reshape(2, -1, *out.shape[1:])


def _peer_rank():
    """Per core: the peer's mesh index, and my index in the pair."""
    peer = (np.asarray([1], np.int32), np.asarray([0], np.int32))
    rank = (np.asarray([0], np.int32), np.asarray([1], np.int32))
    return peer, rank


class PairKernelsTest(parameterized.TestCase):
    @parameterized.parameters(8, 16, 32, 64, 128, 256)
    def test_wo_a_swap_matches_exchange_then_fused_op(self, num_tokens):
        rng = np.random.default_rng(num_tokens)
        half = NUM_HEADS // 2
        groups = half // 8
        o = [
            jnp.asarray(
                rng.standard_normal((num_tokens, NUM_HEADS, HEAD_DIM)), jnp.bfloat16
            )
            for _ in range(2)
        ]
        wo_a = [
            jnp.asarray(
                rng.standard_normal((8 * HEAD_DIM, groups * LORA_RANK)),
                jnp.float8_e4m3fn,
            )
            for _ in range(2)
        ]
        scale = [
            jnp.asarray(rng.uniform(0.5, 1.5, groups * LORA_RANK), jnp.float32)
            for _ in range(2)
        ]
        positions = jnp.asarray(
            rng.integers(0, MAX_POSITION, 2 * num_tokens), jnp.int32
        )
        cache = jnp.asarray(make_cos_sin_cache(MAX_POSITION, ROTARY_DIM))
        got = _per_core(
            functools.partial(
                fused_reverse_rope_wo_a_projection_pair_swap, head_dim=HEAD_DIM
            ),
            o,
            (positions, positions),
            (cache, cache),
            wo_a,
            scale,
            *_peer_rank(),
        )
        for r in range(2):
            mine = slice(r * half, (r + 1) * half)
            o_half = jnp.concatenate([o[0][:, mine], o[1][:, mine]])
            want = fused_reverse_rope_wo_a_projection(
                o_half, positions, cache, wo_a[r], scale[r], head_dim=HEAD_DIM
            )
            np.testing.assert_array_equal(got[r], np.asarray(want), err_msg=f"{r=}")


if __name__ == "__main__":
    absltest.main()
