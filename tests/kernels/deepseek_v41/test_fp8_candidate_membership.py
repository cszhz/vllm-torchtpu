# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-query membership, including duplicates and invalid union slots."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PackedFP8Layout,
    read_fp8_records,
    write_fp8_records,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.candidates import (
    candidate_membership,
    canonical_candidate_blocks,
    gather_candidate_keys,
    prepare_candidates,
)


@pytest.mark.parametrize(
    "rows,count,columns,blocks",
    [
        (1, 3, 7, 17),
        (7, 129, 258, 257),
        (288, 2048, 4096, 12352),
    ],
)
def test_candidate_membership_matches_per_query_set(rows, count, columns, blocks):
    rng = np.random.default_rng(891)
    candidates = rng.integers(-3, blocks + 3, (rows, count), dtype=np.int32)
    canonical = rng.integers(-3, blocks + 3, (rows, columns), dtype=np.int32)
    candidates[0] = -1
    if rows > 1:
        candidates[1] = 0
        canonical[1, :3] = [0, -1, blocks]
    expected = np.stack(
        [
            np.isin(union, source) & (union >= 0) & (union < blocks)
            for source, union in zip(candidates, canonical, strict=True)
        ]
    )
    run = jax.jit(functools.partial(candidate_membership, blocks=blocks))
    actual = run(jnp.asarray(candidates), jnp.asarray(canonical))
    np.testing.assert_array_equal(np.asarray(actual), expected)
    # Reuse the same executable with changed data: no stale per-row histogram.
    actual = run(jnp.full_like(candidates, -1), jnp.asarray(canonical))
    assert not np.any(np.asarray(actual))


@pytest.mark.parametrize("owners", [[0, 0], [0, 0, 1, 1, 1, 2, -1]])
@pytest.mark.parametrize("page_size", [8, 128])
def test_prepared_keys_and_membership_keep_original_token_mapping(owners, page_size):
    if jax.default_backend() != "tpu":
        pytest.skip("Exercises real SparseCore metadata and DMA")
    width, block = 64, 16
    requests = max(owners) + 1
    pages_per_request = (width + page_size - 1) // page_size
    layout = PackedFP8Layout(requests * pages_per_request, page_size=page_size)
    rng = np.random.default_rng(192)
    pages = jnp.asarray(rng.permutation(layout.pages).reshape(requests, -1))
    pool = jnp.zeros((layout.pages * layout.page_rows, 128), jnp.int32)
    banks = tuple(jnp.split(pool, 2)) if page_size == 8 else (pool,)
    keys = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    banks = write_fp8_records(banks, jnp.arange(keys.shape[0]), keys, layout)
    patterns = np.array([[0, 2, 2], [1, 3, -1], [2, 0, 99], [-1, -1, -1]], np.int32)
    candidates = jnp.asarray(patterns[np.arange(len(owners)) % len(patterns)])
    prepared = jax.jit(
        functools.partial(prepare_candidates, layout=layout, block=block, width=width)
    )(banks, pages, candidates, jnp.asarray(owners))
    offsets = np.asarray(prepared.offsets)
    compact_layout = PackedFP8Layout(prepared.keys.shape[0] // 33)
    for row, owner in enumerate(owners):
        canonical = np.asarray(prepared.block_ids[row])
        expected_ids = set(np.asarray(candidates[row])) & set(range(width // block))
        if owner < 0:
            expected_ids = set()
        assert set(canonical[np.asarray(prepared.membership[row])]) == expected_ids
        group = np.searchsorted(offsets[1:], row, side="right")
        live = np.flatnonzero(canonical >= 0)
        if not len(live):
            continue
        compact = (live[:, None] * block + np.arange(block)).reshape(-1)
        group_capacity = prepared.keys.shape[0] // (len(offsets) - 1) // 33 * 128
        compact_records = group * group_capacity + compact
        logical = (canonical[live, None] * block + np.arange(block)).reshape(-1)
        records = pages[owner, logical // page_size] * page_size + logical % page_size
        np.testing.assert_array_equal(
            np.asarray(
                read_fp8_records((prepared.keys,), compact_records, compact_layout)
            ),
            np.asarray(read_fp8_records(banks, records, layout)),
        )


@pytest.mark.parametrize("page_size", [8, 128])
@pytest.mark.parametrize("bank_count", [1, 2])
def test_gather_skips_empty_tiles_without_changing_live_slots(page_size, bank_count):
    if jax.default_backend() != "tpu":
        pytest.skip("Exercises SparseCore DMA")
    layout = PackedFP8Layout(128, page_size=page_size)
    rng = np.random.default_rng(729)
    pool = jnp.zeros((layout.pages * layout.page_rows, 128), jnp.int32)
    banks = tuple(jnp.split(pool, bank_count))
    keys = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    banks = write_fp8_records(banks, jnp.arange(keys.shape[0]), keys, layout)
    physical = rng.integers(0, keys.shape[0] // 8, (3, 4096), dtype=np.int32)
    physical[0, 129:] = -1
    physical[1] = -1
    physical[2, 2049:] = -1
    offsets = jnp.array([0, 1, 1, 2], jnp.int32)
    run = jax.jit(functools.partial(gather_candidate_keys, block=8, layout=layout))
    # Changing occupancy also exercises buffer reuse and final partial tiles.
    for ids in (physical, physical[:, ::-1].copy()):
        gathered = run(banks, jnp.asarray(ids), offsets)
        slots = np.flatnonzero(ids.reshape(-1) >= 0)
        target = (slots[:, None] * 8 + np.arange(8)).reshape(-1)
        source = (ids.reshape(-1)[slots, None] * 8 + np.arange(8)).reshape(-1)
        np.testing.assert_array_equal(
            np.asarray(
                read_fp8_records(
                    (gathered,),
                    jnp.asarray(target),
                    PackedFP8Layout(gathered.shape[0] // 33),
                )
            ),
            np.asarray(read_fp8_records(banks, jnp.asarray(source), layout)),
        )
    run(banks, jnp.full_like(physical, -1), offsets).block_until_ready()


@pytest.mark.cpu_test
@pytest.mark.parametrize("page_size", [8, 128])
def test_candidate_addresses_deduplicate_and_drop_invalid_pages(monkeypatch, page_size):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with jax.default_device(jax.devices("cpu")[0]):
        candidates = jnp.array([[2, 0, 2, 3, -1, 999]], jnp.int32)
        table = [[2, 3, -1, 4, 5, 6, -1, -1]] if page_size == 8 else [[2]]
        canonical, physical = canonical_candidate_blocks(
            candidates,
            jnp.array(table, jnp.int32),
            jnp.array([0, 1], jnp.int32),
            block=16,
            width=64,
            cache_pages=7,
            page_size=page_size,
        )
        ids = [0, 2] if page_size == 8 else [0, 2, 3]
        addresses = [2, 3, 5, 6] if page_size == 8 else [16, 18, 19]
        np.testing.assert_array_equal(canonical[0], ids + [-1] * (6 - len(ids)))
        np.testing.assert_array_equal(
            physical[0], addresses + [-1] * (physical.shape[1] - len(addresses))
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("page_size", [8, 128])
def test_candidate_long_context_partial_block(page_size):
    width, block = 2**20 + 5, 16
    last = width // block
    table = np.zeros((1, (width + page_size - 1) // page_size), np.int32)
    table[0, 0] = -1
    ids = jnp.array([[last, 0, last, last + 1, -1]], jnp.int32)
    canonical, physical = canonical_candidate_blocks(
        ids,
        jnp.asarray(table),
        jnp.array([0, 1]),
        block=block,
        width=width,
        cache_pages=1,
        page_size=page_size,
    )
    np.testing.assert_array_equal(canonical, [[last, -1, -1, -1, -1]])
    expected = np.full(physical.shape, -1, np.int32)
    expected[0, 0] = 0
    np.testing.assert_array_equal(physical, expected)
