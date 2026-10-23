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
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax import lax

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention.csa_gather import (
    csa_gather,
    csa_gather_topk,
)

pytestmark = pytest.mark.tpu_test


def gather_nope_ref_impl(cache, indices):
    return cache.reshape(-1, csa_cache_layout.ROW_WORDS)[indices]


def gather_rope_ref_impl(cache, indices, top_k=1024):
    # Bit-level throughout: random words are NaN-laden as bf16.
    words = lax.bitcast_convert_type(
        cache.reshape(-1, csa_cache_layout.ROPE_WORDS)[indices], jnp.uint32
    )
    rope = jnp.concatenate([words & 0xFFFF, words >> 16], axis=-1)  # (n, 64)
    rope = rope.reshape(-1, 2, top_k // 2, 64)
    rows = jnp.concatenate([rope[:, 0], rope[:, 1]], axis=-1).reshape(-1, 2, 128)
    # bf16 rows 2t and 2t + 1 are the low and high halves of int32 row t.
    return lax.bitcast_convert_type(rows[:, 0] | (rows[:, 1] << 16), jnp.int32)


@functools.partial(jax.jit, static_argnames=("top_k",))
def gather_ref_impl(nope_cache, rope_cache, indices, top_k=1024):
    nope_out = gather_nope_ref_impl(nope_cache, indices)
    rope_out = gather_rope_ref_impl(rope_cache, indices, top_k=top_k)
    return nope_out, rope_out


def topk_rows_ref_impl(topk, page_indices, cu_q_lens, page_size, total_pages):
    """The cache rows of `topk`'s entries, as the TensorCore index chain had them."""
    num_tokens, top_k = topk.shape
    num_seqs = cu_q_lens.shape[0] - 1
    pages_per_seq = page_indices.shape[0] // num_seqs
    seq_ids = jnp.repeat(
        jnp.arange(num_seqs),
        cu_q_lens[1:] - cu_q_lens[:-1],
        total_repeat_length=num_tokens,
    )
    windows = page_indices.reshape(num_seqs, pages_per_seq)[seq_ids]
    logical = jnp.clip(topk // page_size, 0, pages_per_seq - 1)
    page_ids = jnp.take_along_axis(windows, logical, axis=1)
    flat = jnp.arange(num_tokens * top_k, dtype=jnp.int32).reshape(num_tokens, top_k)
    page_ids = jnp.where(topk == -1, (flat * 104729) % total_pages, page_ids)
    offsets = jnp.where(topk == -1, (flat * 15485863) % page_size, topk % page_size)
    return (page_ids * page_size + offsets).reshape(-1), seq_ids


def assert_valid_prefix_equal(ref, out, topk):
    """`csa_gather_topk` outputs agree on every token's first `kv_len`
    entries, `kv_len` being its count of entries other than -1; the kernel
    may leave the rest unwritten."""
    num_tokens, top_k = topk.shape
    kv_lens = np.asarray((topk != -1).sum(axis=1))[:, None]
    entry = np.arange(top_k)[None, :]
    nope_valid = (entry < kv_lens).reshape(-1)
    np.testing.assert_array_equal(
        np.asarray(ref[0])[nope_valid], np.asarray(out[0])[nope_valid]
    )
    # int32 rope row j of a token holds entries 2j (low bf16) and 2j + 1
    # (high bf16) in lanes 0:64, and the same entries + top_k // 2 in 64:128.
    shape = (num_tokens, top_k // 4, 2, 64)
    j = np.arange(top_k // 4)[None, :, None, None]
    half = np.arange(2)[None, None, :, None]
    first = half * (top_k // 2) + 2 * j
    kv = kv_lens[:, :, None, None]
    for shift, e in ((0, first), (16, first + 1)):
        valid = np.broadcast_to(e < kv, shape)
        ref_bits = (np.asarray(ref[1]).view(np.uint32).reshape(shape) >> shift) & 0xFFFF
        out_bits = (np.asarray(out[1]).view(np.uint32).reshape(shape) >> shift) & 0xFFFF
        np.testing.assert_array_equal(ref_bits[valid], out_bits[valid])


def _random_words(key, shape):
    bits = jax.random.bits(key, shape, dtype=jnp.uint32)
    return lax.bitcast_convert_type(bits, jnp.int32)


@jax.jit
def create_nope_cache():
    key = jax.random.key(0)
    key, perm_key, cache_key = jax.random.split(key, 3)
    num_pages = 1000
    page_size = 256
    cache = _random_words(cache_key, (num_pages, page_size, 128))
    return cache, perm_key


@jax.jit
def create_rope_cache():
    key = jax.random.key(41)
    key, perm_key, cache_key = jax.random.split(key, 3)
    num_pages = 1000
    page_size = 256
    cache = _random_words(cache_key, (num_pages, page_size // 4, 128))
    return cache, perm_key


class GatherTest(parameterized.TestCase):
    @parameterized.parameters(
        (256, 256),
        (4096, 256),
        (4096, 512),
        (128 * 1024, 1024),
        (128 * 1024, 2048),
        # Not a whole number of atom batches: exercises the padding.
        (25 * 512, 512),
    )
    def test_correctness(self, n, top_k):
        nope_cache, perm_key = create_nope_cache()
        rope_cache, _ = create_rope_cache()

        max_index = nope_cache.shape[0] * nope_cache.shape[1]
        indices = jax.random.randint(perm_key, (n,), 0, max_index, dtype=jnp.int32)

        nope_ref, rope_ref = gather_ref_impl(
            nope_cache, rope_cache, indices, top_k=top_k
        )
        nope_sc, rope_sc = csa_gather(nope_cache, rope_cache, indices, top_k=top_k)

        np.testing.assert_array_equal(nope_ref, nope_sc)
        np.testing.assert_array_equal(rope_ref, rope_sc)

    @parameterized.parameters(
        (4096, 1024),
        (128 * 1024, 16 * 1024),
        (128 * 1024, 1024),
    )
    def test_num_valid_indices(self, n, num_valid):
        nope_cache, perm_key = create_nope_cache()
        rope_cache, _ = create_rope_cache()

        max_index = nope_cache.shape[0] * nope_cache.shape[1]
        indices = jax.random.randint(perm_key, (n,), 0, max_index, dtype=jnp.int32)

        nope_ref, rope_ref = gather_ref_impl(
            nope_cache, rope_cache, indices[:num_valid]
        )
        nope_sc, rope_sc = csa_gather(
            nope_cache, rope_cache, indices, num_valid_indices=num_valid
        )

        np.testing.assert_array_equal(nope_ref, nope_sc[:num_valid])
        np.testing.assert_array_equal(rope_ref, rope_sc[: num_valid // 4])

    @parameterized.parameters(
        (40, 256, 32),
        (128, 512, 32),
        (256, 512, 128),
        (24, 128, 16),
    )
    def test_topk_lookup(self, num_tokens, top_k, chunk):
        nope_cache, perm_key = create_nope_cache()
        rope_cache, _ = create_rope_cache()
        total_pages, page_size, _ = nope_cache.shape
        num_seqs, pages_per_seq = 6, 9
        rng = np.random.default_rng(0)
        cu_q_lens = np.concatenate(
            [[0], np.sort(rng.choice(num_tokens - 3, num_seqs - 1, replace=False) + 1)]
        )
        cu_q_lens = jnp.asarray(np.append(cu_q_lens, num_tokens - 3), jnp.int32)
        topk = rng.integers(0, pages_per_seq * page_size, (num_tokens, top_k))
        for t in range(num_tokens):
            topk[t, rng.integers(0, top_k + 1) :] = -1
        topk = jnp.asarray(topk, jnp.int32)
        page_indices = jax.random.randint(
            perm_key, (num_seqs * pages_per_seq,), 0, total_pages, dtype=jnp.int32
        )
        rows, seq_ids = topk_rows_ref_impl(
            topk, page_indices, cu_q_lens, page_size, total_pages
        )

        for start in range(0, num_tokens, chunk):
            n = min(chunk, num_tokens - start)
            nope_sc, rope_sc = csa_gather_topk(
                nope_cache,
                rope_cache,
                topk,
                seq_ids,
                page_indices,
                jnp.int32(n * top_k),
                token_start=start,
                num_tokens=n,
                top_k=top_k,
                pages_per_seq=pages_per_seq,
            )
            ref = gather_ref_impl(
                nope_cache, rope_cache, rows[start * top_k : (start + n) * top_k], top_k
            )
            assert_valid_prefix_equal(ref, (nope_sc, rope_sc), topk[start : start + n])


if __name__ == "__main__":
    absltest.main()
