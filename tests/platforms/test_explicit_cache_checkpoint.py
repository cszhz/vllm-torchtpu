# SPDX-License-Identifier: Apache-2.0
"""Explicit checkpoints preserve real hit counts and sparse Mamba reuse."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.core.explicit_cache_checkpoint import (
    patch_explicit_cache_checkpoint,
)

pytestmark = pytest.mark.cpu_test
SCHEDULER_BLOCK_SIZE = 32


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    for name in ("get_computed_blocks", "get_computed_blocks_for_connector"):
        monkeypatch.setattr(
            KVCacheManager, name, inspect.unwrap(getattr(KVCacheManager, name))
        )
    # Model writes are not run in this CPU cache-metadata contract test.
    for name in ("get_cached_block", "_insert_block_hash"):
        monkeypatch.setattr(BlockPool, name, inspect.unwrap(getattr(BlockPool, name)))
    patch_explicit_cache_checkpoint()


@pytest.fixture
def manager():
    init_none_hash(sha256)
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=8, num_kv_heads=1, head_size=4, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=16,
                    shapes=((4, 4),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
        prefix_cache_retention_interval=0,
    )
    return KVCacheManager(
        config,
        max_model_len=256,
        scheduler_block_size=SCHEDULER_BLOCK_SIZE,
        hash_block_size=8,
    )


def _request(name, ids, cache_control=None):
    return Request(
        name,
        ids,
        SamplingParams(
            max_tokens=1, extra_args={"cache_control_params": cache_control}
        ),
        None,
        block_hasher=get_request_block_hasher(8, sha256),
    )


def _seed(manager):
    seed = _request("seed", list(range(136)))
    assert manager.allocate_slots(seed, 128) is not None
    attention_blocks = manager.get_block_ids(seed.request_id)[0]
    manager.free(seed)
    manager.new_step_starts()
    return attention_blocks


@pytest.mark.parametrize(
    "cache_control, expected_boundary",
    [
        (None, 64),
        ([], 64),
        ({"token_positions": "111"}, 64),
        ({"token_positions": [True, "111", -1, 0, 144, 7]}, 64),
        ({"token_positions": [32]}, 32),
        ({"token_positions": [79, 119, 95, 119, 1000]}, 96),
    ],
)
def test_explicit_position_overrides_only_boundary(
    manager, cache_control, expected_boundary
):
    _seed(manager)
    ids = list(range(64)) + list(range(1000, 1079))
    baseline = manager.get_computed_blocks(_request("baseline", ids))
    assert baseline[1:] == (0, 64)
    request = _request("hint", ids, cache_control)

    result = manager.get_computed_blocks(request)

    assert result[:2] == baseline[:2]
    assert result[2] == expected_boundary
    assert request.sampling_params.extra_args["cache_control_params"] == cache_control


@pytest.mark.parametrize("fallback", [False, True])
def test_connector_direct_and_reconciled_results_keep_hits_and_divergence(
    manager, fallback
):
    attention_blocks = _seed(manager)
    if fallback:
        manager.block_pool.evict_blocks(set(attention_blocks[8:]))
        ids = list(range(143))
    else:
        ids = list(range(64)) + list(range(1000, 1079))
    baseline = manager.get_computed_blocks_for_connector(_request("baseline", ids))
    assert baseline[1:] == ((0, 64, False) if fallback else (64, 0, True))

    result = manager.get_computed_blocks_for_connector(
        _request("hint", ids, {"token_positions": [119]})
    )

    assert result[:2] == baseline[:2]
    assert result[2:] == (96, baseline[3])


@pytest.mark.parametrize("skip_read", [False, True])
def test_disabled_or_skipped_lookup_keeps_zero_boundary(manager, skip_read):
    request = _request("hint", list(range(143)), {"token_positions": [119]})
    manager.enable_caching = skip_read
    request.skip_reading_prefix_cache = skip_read
    assert manager.get_computed_blocks(request)[1:] == (0, 0)
    assert manager.get_computed_blocks_for_connector(request)[1:] == (0, 0, False)


def test_aligned_hint_creates_reusable_mamba_checkpoint_under_zero_retention(manager):
    request = _request("producer", list(range(143)), {"token_positions": [119]})
    blocks, hits, request.shared_prefix_boundary = manager.get_computed_blocks(request)
    assert hits == 0
    assert request.shared_prefix_boundary == 96
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=SCHEDULER_BLOCK_SIZE),
        use_eagle=False,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_partial_cache_hit=False,
        hash_block_size=8,
        max_num_scheduled_tokens=256,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
    )
    count = inspect.unwrap(Scheduler._mamba_block_aligned_split)(
        scheduler, request, 143
    )
    assert count == 96
    assert manager.allocate_slots(request, count, hits, blocks) is not None
    handoffs = manager.take_boundary_state_offloads()[request.request_id]
    assert [(group, boundary) for group, _, boundary in handoffs] == [(1, 96)]
    request.num_computed_tokens = count
    assert manager.allocate_slots(request, request.num_tokens - count) is not None
    manager.free(request)
    manager.new_step_starts()

    follower = _request("follower", list(range(96)) + list(range(1000, 1047)))
    assert manager.get_computed_blocks(follower)[1] == 96


def test_patch_is_idempotent():
    methods = (
        KVCacheManager.get_computed_blocks,
        KVCacheManager.get_computed_blocks_for_connector,
    )
    patch_explicit_cache_checkpoint()
    assert methods == (
        KVCacheManager.get_computed_blocks,
        KVCacheManager.get_computed_blocks_for_connector,
    )
