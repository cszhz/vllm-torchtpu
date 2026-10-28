# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-query membership, including duplicates and invalid union slots."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8_candidate_cache import (
    candidate_membership,
    gather_candidate_keys,
    prepare_candidates,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PackedFP8Layout,
    read_fp8_records,
    write_fp8_records,
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


@pytest.mark.parametrize("owners", [[0], [0, 0], [0, 0, 0], [1, 0, 1, -1, 0, 1, 2]])
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
    for row, original in enumerate(np.asarray(prepared.order)):
        canonical = np.asarray(prepared.block_ids[row])
        expected_ids = set(np.asarray(candidates[original])) & set(
            range(width // block)
        )
        if owners[original] < 0:
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
        records = (
            pages[owners[original], logical // page_size] * page_size
            + logical % page_size
        )
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
