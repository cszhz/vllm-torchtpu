"""Parity for page-table groups staged through bounded SMEM windows."""

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper


def _run_case(
    history, active_seqs=(0, 128), pages_per_seq=257, compact_reference=False
):
    num_seqs = active_seqs[-1] + 1
    num_active = len(active_seqs)
    page_size = 128
    dtype = jnp.dtype("float8_e4m3fn")
    used_pages = (history + page_size) // page_size
    cache_shape = wrapper.get_kv_cache_shape(
        num_active * used_pages,
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
        page_ids = page_ids.at[seq, :used_pages].set(
            jnp.arange(i * used_pages, (i + 1) * used_pages, dtype=jnp.int32)
        )
        q_lens[seq] = 1
        kv_lens_host[seq] = history + 1
    page_ids = page_ids.ravel()
    cu_q_lens = jnp.asarray(np.concatenate(([0], np.cumsum(q_lens))), jnp.int32)
    kv_lens = jnp.asarray(kv_lens_host)
    page_values = ((jnp.arange(num_active * used_pages) % 7) - 3).astype(dtype)
    cache = page_values[:, None, None, None, None] * jnp.ones(cache_shape, dtype)
    q = jnp.ones((num_active, 4, 128), jnp.bfloat16)
    k = jnp.ones((num_active, 2, 128), dtype)
    v = jnp.ones((num_active, 2, 128), dtype)
    distribution = jnp.array([num_seqs] * 3, jnp.int32)

    def run(stream):
        compact = compact_reference and not stream
        run_kv_lens = kv_lens[jnp.array(active_seqs)] if compact else kv_lens
        run_pages = (
            page_ids.reshape(num_seqs, pages_per_seq)[
                jnp.array(active_seqs), :used_pages
            ].ravel()
            if compact
            else page_ids
        )
        run_cu_q = jnp.arange(num_active + 1, dtype=jnp.int32) if compact else cu_q_lens
        run_dist = jnp.array([num_active] * 3, jnp.int32) if compact else distribution
        return jax.block_until_ready(
            wrapper.ragged_paged_attention(
                q,
                k,
                v,
                cache.copy(),
                run_kv_lens,
                run_pages,
                run_cu_q,
                run_dist,
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


def test_streamed_page_table_unaligned_1m():
    # Seven padded rows per window: exercise both boundaries and buffer reuse.
    _run_case(255, (0, 6, 7, 13, 14, 191), 8217, compact_reference=True)


def test_streamed_page_table_partial_batches():
    # Also exercise the existing unpadded path with four partial chunks.
    _run_case(255, (0, 127, 128, 255, 256, 384), 257, compact_reference=True)


def test_streamed_page_table_1m_history():
    # Full 1M attention on the last slot of the production-sized page table.
    _run_case(1048575, (191,), 8217, compact_reference=True)


if __name__ == "__main__":
    test_streamed_page_table_cross_group()
    test_streamed_page_table_unaligned_1m()
    test_streamed_page_table_partial_batches()
    test_streamed_page_table_1m_history()
