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
"""GLM5_MKD_STACK_C: stacked [n_layers, num_blocks, ...] KV caches on the DCP path.

Pure jnp, no Pallas: these run on CPU. The DCP writers are simulated one rank
at a time (`dcp_rank` is a plain argument), the same way
`test_mla_kv_cache_utils.py` does. `test_mla_kv_stack_dcp_tpu.py` checks the
real ops (SparseCore writer, sparse attention kernel, indexer) on TPU.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    get_dcp_page_and_slot,
)
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    OOB_PAGE,
    SPARSECORE_MAX_TENSOR_BYTES,
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
    glm5_stack_group_sizes,
    glm5_stacked_kv_kinds,
    merge_kv_stack,
    split_kv_stack,
    stack_page_table_dcp,
    update_sparse_mla_kv_cache_dcp_jax,
)

LKV_DIM = 512
ROPE_DIM = 64
KV_PACKING = 4  # fp8


def _random_like(shape, dtype, seed):
    """Nonzero prior content: a write landing in the wrong row shows up."""
    rng = np.random.default_rng(seed)
    if jnp.dtype(dtype) == jnp.uint32:
        return jnp.asarray(rng.integers(0, 2**32, shape, dtype=np.uint64), jnp.uint32)
    return jnp.asarray(rng.integers(0, 256, shape, dtype=np.int64), jnp.uint8)


class MergeSplitTest(parameterized.TestCase):
    def test_merge_is_row_major_page_pool(self):
        stack = jnp.arange(3 * 5 * 4 * 8, dtype=jnp.uint32).reshape(3, 5, 4, 8)
        for row in range(3):
            merged = merge_kv_stack(stack, (row, 5))
            self.assertEqual(merged.shape, (15, 4, 8))
            for page in range(5):
                np.testing.assert_array_equal(
                    np.asarray(merged[row * 5 + page]), np.asarray(stack[row, page])
                )
            np.testing.assert_array_equal(
                np.asarray(split_kv_stack(merged, stack.shape)), np.asarray(stack)
            )

    def test_merge_rejects_a_foreign_stack(self):
        stack = jnp.zeros((3, 5, 4, 8), jnp.uint8)
        with self.assertRaisesRegex(ValueError, "kv_stack"):
            merge_kv_stack(stack, (0, 4))  # wrong num_blocks
        with self.assertRaisesRegex(ValueError, "kv_stack"):
            merge_kv_stack(stack, (3, 5))  # row out of range


class StackPageTableTest(parameterized.TestCase):
    @parameterized.parameters((1, 7), (3, 7), (21, 16), (78, 5))
    def test_resolves_to_the_per_layer_page_in_this_row(self, n_layers, num_blocks):
        """`id % (n * nb)` of the rewritten table == row * nb + `id % nb`.

        That is the resolution every DCP reader and writer applies; the table
        holds in-range ids, ids >= num_blocks and negative padding.
        """
        rng = np.random.default_rng(n_layers)
        table = np.concatenate(
            [
                rng.integers(0, num_blocks, 32),
                rng.integers(num_blocks, 10 * num_blocks, 8),
                [-1, -2, -num_blocks, 0, num_blocks - 1, num_blocks],
            ]
        ).astype(np.int32)
        for row in range(n_layers):
            stacked = np.asarray(
                stack_page_table_dcp(jnp.asarray(table), (row, num_blocks))
            )
            self.assertTrue(
                (
                    (stacked >= row * num_blocks) & (stacked < (row + 1) * num_blocks)
                ).all()
            )
            resolved = np.mod(stacked, n_layers * num_blocks)
            np.testing.assert_array_equal(
                resolved, row * num_blocks + np.mod(table, num_blocks)
            )


def _dcp_inputs(dcp_size, interleave_size, page_size, pages_per_seq, num_blocks):
    """Two sequences: a chunked-prefill continuation and a decode/verify-sized
    tail, positions covering every rank; block table with an id >= num_blocks."""
    vps = page_size * dcp_size
    cycle = dcp_size * interleave_size
    seq_lens = np.asarray([2 * vps + 11, cycle + 3, 0], np.int32)
    q_lens = np.asarray([2 * cycle + 5, 4, 0], np.int32)
    qsl = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    total = int(qsl[-1]) + 6  # batch padding rows
    table = np.zeros((3, pages_per_seq), np.int32)
    perm = np.random.default_rng(5).permutation(num_blocks)
    table[0] = perm[:pages_per_seq]
    table[1] = perm[pages_per_seq : 2 * pages_per_seq]
    table[1, 0] += num_blocks  # a virtual id past the pool: resolved by `%`
    rng = np.random.default_rng(11)
    kv_c = jnp.asarray(rng.standard_normal((total, LKV_DIM)), jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    k_pe = jnp.asarray(rng.standard_normal((total, ROPE_DIM)), jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    return dict(
        kv_c=kv_c,
        k_pe=k_pe,
        seq_lens=jnp.asarray(seq_lens),
        block_tables=jnp.asarray(table.reshape(-1)),
        query_start_loc=jnp.asarray(qsl),
    )


class StackedDcpWriterTest(parameterized.TestCase):
    """`update_sparse_mla_kv_cache_dcp` on the merged stack writes exactly what
    the per-layer writer writes, into this layer's row, and nothing else."""

    @parameterized.named_parameters(
        dict(testcase_name="d4_i4_sc", dcp_size=4, layout=KVCacheLayout.SPARSECORE),
        dict(testcase_name="d8_i4_sc", dcp_size=8, layout=KVCacheLayout.SPARSECORE),
        dict(testcase_name="d4_i4_tc", dcp_size=4, layout=KVCacheLayout.TENSORCORE),
    )
    def test_stacked_matches_per_layer(self, dcp_size, layout):
        n_layers, num_blocks, page_size, interleave = 3, 12, 32, 4
        nope_spec = SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE, layout, num_blocks, page_size, LKV_DIM, KV_PACKING
        )
        rope_spec = SparseMLAKVCacheSpec.create(
            KVCacheType.ROPE, layout, num_blocks, page_size, ROPE_DIM, KV_PACKING
        )
        merged_nope_spec = dataclasses.replace(
            nope_spec, num_pages=n_layers * num_blocks
        )
        merged_rope_spec = dataclasses.replace(
            rope_spec, num_pages=n_layers * num_blocks
        )
        inputs = _dcp_inputs(dcp_size, interleave, page_size, 4, num_blocks)
        for rank in range(dcp_size):
            nope_stack = _random_like(
                (n_layers, *nope_spec.shape), nope_spec.jax_dtype, rank
            )
            rope_stack = _random_like(
                (n_layers, *rope_spec.shape), rope_spec.jax_dtype, 100 + rank
            )
            for row in range(n_layers):
                kv_stack = (row, num_blocks)
                want_nope, want_rope = update_sparse_mla_kv_cache_dcp_jax(
                    nope_stack[row],
                    rope_stack[row],
                    inputs["kv_c"],
                    inputs["k_pe"],
                    inputs["seq_lens"],
                    inputs["block_tables"],
                    inputs["query_start_loc"],
                    jnp.int32(rank),
                    nope_spec=nope_spec,
                    rope_spec=rope_spec,
                    dcp_size=dcp_size,
                    interleave_size=interleave,
                )
                got_nope, got_rope = update_sparse_mla_kv_cache_dcp_jax(
                    merge_kv_stack(nope_stack, kv_stack),
                    merge_kv_stack(rope_stack, kv_stack),
                    inputs["kv_c"],
                    inputs["k_pe"],
                    inputs["seq_lens"],
                    stack_page_table_dcp(inputs["block_tables"], kv_stack),
                    inputs["query_start_loc"],
                    jnp.int32(rank),
                    nope_spec=merged_nope_spec,
                    rope_spec=merged_rope_spec,
                    dcp_size=dcp_size,
                    interleave_size=interleave,
                )
                got_nope = np.asarray(split_kv_stack(got_nope, nope_stack.shape))
                got_rope = np.asarray(split_kv_stack(got_rope, rope_stack.shape))
                self.assertFalse(
                    np.array_equal(np.asarray(want_nope), np.asarray(nope_stack[row])),
                    "degenerate: this rank wrote nothing",
                )
                np.testing.assert_array_equal(got_nope[row], np.asarray(want_nope))
                np.testing.assert_array_equal(got_rope[row], np.asarray(want_rope))
                for other in range(n_layers):
                    if other != row:
                        np.testing.assert_array_equal(
                            got_nope[other], np.asarray(nope_stack[other])
                        )
                        np.testing.assert_array_equal(
                            got_rope[other], np.asarray(rope_stack[other])
                        )


class StackedIndexerPageSlotTest(parameterized.TestCase):
    """The indexer writer's resolver (`get_dcp_page_and_slot`) on the merged
    stack: same slot, page offset by row * num_blocks, OOB rows stay OOB."""

    @parameterized.parameters(4, 8, 16)
    def test_offset_pages(self, dcp_size):
        n_layers, num_blocks, page_size, interleave = 21, 12, 256, 4
        inputs = _dcp_inputs(dcp_size, interleave, page_size, 4, num_blocks)
        num_tokens = inputs["kv_c"].shape[0]
        for rank in range(dcp_size):
            want_page, want_slot = get_dcp_page_and_slot(
                num_tokens,
                inputs["seq_lens"],
                inputs["block_tables"],
                inputs["query_start_loc"],
                jnp.int32(rank),
                page_size=page_size,
                num_pages=num_blocks,
                dcp_size=dcp_size,
                interleave_size=interleave,
            )
            want_page, want_slot = np.asarray(want_page), np.asarray(want_slot)
            valid = want_page != OOB_PAGE
            for row in (0, 7, n_layers - 1):
                page, slot = get_dcp_page_and_slot(
                    num_tokens,
                    inputs["seq_lens"],
                    stack_page_table_dcp(inputs["block_tables"], (row, num_blocks)),
                    inputs["query_start_loc"],
                    jnp.int32(rank),
                    page_size=page_size,
                    num_pages=n_layers * num_blocks,
                    dcp_size=dcp_size,
                    interleave_size=interleave,
                )
                page, slot = np.asarray(page), np.asarray(slot)
                np.testing.assert_array_equal(page[~valid], want_page[~valid])
                np.testing.assert_array_equal(
                    page[valid], want_page[valid] + row * num_blocks
                )
                np.testing.assert_array_equal(slot[valid], want_slot[valid])


class StackKindsEnvTest(parameterized.TestCase):
    def test_parse(self):
        self.assertEqual(glm5_stacked_kv_kinds(""), frozenset())
        self.assertEqual(glm5_stacked_kv_kinds("0"), frozenset())
        self.assertEqual(glm5_stacked_kv_kinds("1"), frozenset({"nope", "rope", "idx"}))
        self.assertEqual(glm5_stacked_kv_kinds("idx"), frozenset({"idx"}))
        self.assertEqual(
            glm5_stacked_kv_kinds("nope,rope"), frozenset({"nope", "rope"})
        )
        with self.assertRaisesRegex(ValueError, "together"):
            glm5_stacked_kv_kinds("nope")
        with self.assertRaisesRegex(ValueError, "unknown"):
            glm5_stacked_kv_kinds("kv")


class StackGroupSizesTest(parameterized.TestCase):
    def test_glm52_full_depth_nope_splits_in_two(self):
        # 78 layers x 2085 pages x 128 KiB = 20.8 GB > the 16 GiB SC limit.
        per_layer = 2085 * 256 * 128 * 4
        self.assertEqual(glm5_stack_group_sizes(78, per_layer), [39, 39])
        # rope (32 KiB pages) and the 21 indexer layers fit in one tensor.
        self.assertEqual(glm5_stack_group_sizes(78, 2085 * 64 * 128 * 4), [78])
        self.assertEqual(glm5_stack_group_sizes(21, 2085 * 33 * 4 * 256), [21])

    @parameterized.parameters((78, 7, 100), (10, 3, 9), (5, 1, 1), (21, 5, 49))
    def test_fewest_balanced_groups_under_the_limit(self, n, per, limit):
        sizes = glm5_stack_group_sizes(n, per, limit)
        self.assertEqual(sum(sizes), n)
        self.assertLessEqual(max(sizes) * per, limit)
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        if len(sizes) > 1:  # one fewer group would not fit
            self.assertGreater(-(-n // (len(sizes) - 1)) * per, limit)

    def test_default_limit_and_oversized_layer(self):
        self.assertEqual(
            glm5_stack_group_sizes(3, SPARSECORE_MAX_TENSOR_BYTES // 3), [3]
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            glm5_stack_group_sizes(2, 11, 10)
