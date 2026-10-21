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
"""TPU_KV_DROP_HCA_STATE_CHECKPOINT on vLLM's KVCacheManager with DeepSeek-V4-shaped
groups (layer names as the model registers them): hits are unchanged, and the
128-ratio compressor-state group is satisfied by null blocks and caches nothing."""

import pytest
import test_fine_prefix_hits as fph_test
import torch
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_torchtpu.core import checkpoint_free_order as cfo
from vllm_torchtpu.core import fine_prefix_hits as fph
from vllm_torchtpu.core import hca_state_checkpoint as hsc

FULL = 1024
HCA_STATE = 4


def _swa(block_size, window, head_size):
    return SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
        sliding_window=window,
    )


def _manager(num_blocks):
    groups = [
        KVCacheGroupSpec(["model.layers.0.attn.swa_cache"], _swa(128, 128, 512)),
        KVCacheGroupSpec(["mtp.0.attn.swa_cache"], _swa(128, 128, 512)),
        KVCacheGroupSpec(
            ["model.layers.0.attn", "model.layers.2.attn"],
            MLAAttentionSpec(
                block_size=FULL,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.uint8,
                tokens_per_state=128,
            ),
        ),
        KVCacheGroupSpec(
            [
                "model.layers.2.attn.compressor.state_cache",
                "model.layers.2.attn.indexer.compressor.state_cache",
            ],
            _swa(16, 8, 4096),
        ),
        KVCacheGroupSpec(
            ["model.layers.0.attn.compressor.state_cache"], _swa(32, 128, 2048)
        ),
    ]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=32768,
    )
    return KVCacheManager(
        config,
        max_model_len=1 << 20,
        scheduler_block_size=FULL,
        hash_block_size=16,
        enable_caching=True,
        use_eagle=False,
    )


init_none_hash(sha256)
fph_test._HASHER = get_request_block_hasher(16, sha256)
_request, _prefill, _decode, _run = (
    fph_test._request,
    fph_test._prefill,
    fph_test._decode,
    fph_test._run,
)
TURN1, TURN2 = fph_test.TURN1, fph_test.TURN2


def test_finds_only_the_128_ratio_state_group():
    assert hsc.hca_state_group_ids(_manager(64).kv_cache_config) == {HCA_STATE}


def _scenario(mgr, held):
    """Hits of: next turn, a truncated history, and a next turn after churn."""
    hits = []
    _run(mgr, TURN1, decode_steps=300)
    hits.append(mgr.get_computed_blocks(_request(TURN2, 4))[1])
    hits.append(mgr.get_computed_blocks(_request(TURN1[:40_100] + [3] * 900, 4))[1])
    req = _request(TURN2, 1300)
    hits.append(_prefill(mgr, req))
    for step in range(300):
        _decode(mgr, req, 1)
        if step % 10 == 0:
            base = 10_000_000 + step * 100_000
            _run(mgr, range(base, base + 3000), decode_steps=5)
    mgr.free(req)
    turn3 = TURN2[:-2] + list(range(60_000_000, 60_002_000))
    hits.append(mgr.get_computed_blocks(_request(turn3, 10))[1])
    return hits


@pytest.mark.parametrize("fine", [False, True], ids=["block-hits", "fine-hits"])
def test_hits_are_unchanged_and_the_group_caches_nothing(fine, monkeypatch):
    if fine:
        fph.install(128, monkeypatch.setattr)
    fph_test._hold(monkeypatch)
    expected = _scenario(_manager(130), held=True)

    hsc.install(monkeypatch.setattr)
    mgr = _manager(130)
    assert _scenario(mgr, held=True) == expected
    assert expected[0] == (40_320 if fine else 39 * FULL)
    manager = mgr.coordinator.single_type_managers[HCA_STATE]
    assert cfo.held_checkpoint_indices(manager, len(TURN1)) == set()

    fresh = _manager(130)
    _run(fresh, TURN1)
    blocks, hit, _ = fresh.get_computed_blocks(_request(TURN2, 4))
    assert hit == expected[0]
    assert blocks.blocks[HCA_STATE] and all(b.is_null for b in blocks.blocks[HCA_STATE])
    assert _cached_blocks(fresh, HCA_STATE) == 0


def _cached_blocks(mgr, group):
    return sum(
        b.block_hash is not None and get_group_id(b.block_hash) == group
        for b in mgr.block_pool.blocks
    )


def test_caches_and_holds_nothing_for_the_group(monkeypatch):
    fph_test._hold(monkeypatch)
    base = _manager(400)
    _run(base, TURN1)
    hsc.install(monkeypatch.setattr)
    lean = _manager(400)
    _run(lean, TURN1)
    assert _cached_blocks(base, HCA_STATE) == 8
    assert _cached_blocks(lean, HCA_STATE) == 0
    for group in range(4):
        assert _cached_blocks(lean, group) == _cached_blocks(base, group)
    held = [
        len(cfo.held_checkpoint_indices(m, len(TURN1)))
        for m in lean.coordinator.single_type_managers
        if isinstance(m, SlidingWindowManager)
    ]
    assert held == [1, 1, 1, 0]
