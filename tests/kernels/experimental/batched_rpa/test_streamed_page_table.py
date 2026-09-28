"""Parity for page-table groups staged through bounded SMEM windows."""

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper


def _run_case(history, active_seqs=(0, 128)):
    num_seqs = active_seqs[-1] + 1
    num_active = len(active_seqs)
    pages_per_seq = 257
    page_size = 128
    dtype = jnp.dtype("float8_e4m3fn")
    cache_shape = wrapper.get_kv_cache_shape(
        num_active * pages_per_seq,
        page_size,
        2,
        128,
        dtype,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    page_ids = jnp.zeros((num_seqs, pages_per_seq), jnp.int32)
    q_lens = np.zeros(num_seqs, np.int32)
    kv_lens_host = np.zeros(num_seqs, np.int32)
    for i, seq in enumerate(active_seqs):
        page_ids = page_ids.at[seq].set(
            jnp.arange(i * pages_per_seq, (i + 1) * pages_per_seq, dtype=jnp.int32)
        )
        q_lens[seq] = 1
        kv_lens_host[seq] = history + 1
    page_ids = page_ids.ravel()
    cu_q_lens = jnp.asarray(np.concatenate(([0], np.cumsum(q_lens))), jnp.int32)
    kv_lens = jnp.asarray(kv_lens_host)
    page_values = ((jnp.arange(num_active * pages_per_seq) % 7) - 3).astype(dtype)
    cache = page_values[:, None, None, None, None] * jnp.ones(cache_shape, dtype)
    q = jnp.ones((num_active, 4, 128), jnp.bfloat16)
    k = jnp.ones((num_active, 2, 128), dtype)
    v = jnp.ones((num_active, 2, 128), dtype)
    distribution = jnp.array([num_seqs] * 3, jnp.int32)

    def run(stream):
        return jax.block_until_ready(
            wrapper.ragged_paged_attention(
                q,
                k,
                v,
                cache.copy(),
                kv_lens,
                page_ids,
                cu_q_lens,
                distribution,
                kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
                stream_page_indices=stream,
            )
        )

    expected_out, expected_cache = run(False)
    actual_out, actual_cache = run(True)
    np.testing.assert_array_equal(np.asarray(actual_out), np.asarray(expected_out))
    np.testing.assert_array_equal(np.asarray(actual_cache), np.asarray(expected_cache))


def test_streamed_page_table_cross_group():
    for history in (2 * 128 - 1, 257 * 128 - 1):
        _run_case(history)
    _run_case(2 * 128 - 1, (0, 128, 256))


if __name__ == "__main__":
    test_streamed_page_table_cross_group()
