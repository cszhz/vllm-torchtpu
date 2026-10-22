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
"""TPU_KV_DROP_HCA_STATE_CHECKPOINT on vLLM's KVCacheManager with
DeepSeek-V4-shaped groups (layer names as the model registers them) at both
served block sizes: hits are unchanged, and the 128-ratio compressor-state
group is satisfied by null blocks and caches nothing."""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.core import hca_state_checkpoint as hsc

HCA_STATE = 4
# --block-size -> (SWA block, CSA/indexer state block = hash block, HCA state
# block), as the TPU DeepSeek-V4 layers page them.
GEOMETRY = {1024: (128, 16, 32), 512: (64, 8, 16)}

init_none_hash(sha256)
_ids = iter(range(1, 1 << 30))
# 40,333-token first turn: replay boundary 39,936 at either block size.
TURN1 = list(range(1, 40_334))
# The next turn shares all but the last two prompt tokens (the generation
# prompt), as a chat template renders it.
TURN2 = TURN1[:-2] + list(range(50_000_000, 50_003_000))


@pytest.fixture(params=sorted(GEOMETRY), ids=lambda bs: f"bs{bs}")
def block_size(request):
    return request.param


def _swa(block_size, window, head_size):
    return SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
        sliding_window=window,
    )


def _manager(num_blocks, block_size):
    swa, csa_state, hca_state = GEOMETRY[block_size]
    groups = [
        KVCacheGroupSpec(["model.layers.0.attn.swa_cache"], _swa(swa, 128, 512)),
        KVCacheGroupSpec(["mtp.0.attn.swa_cache"], _swa(swa, 128, 512)),
        KVCacheGroupSpec(
            ["model.layers.0.attn", "model.layers.2.attn"],
            MLAAttentionSpec(
                block_size=block_size,
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
            _swa(csa_state, 8, 4096),
        ),
        KVCacheGroupSpec(
            ["model.layers.0.attn.compressor.state_cache"], _swa(hca_state, 128, 2048)
        ),
    ]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=32768,
    )
    mgr = KVCacheManager(
        config,
        max_model_len=1 << 20,
        scheduler_block_size=block_size,
        hash_block_size=csa_state,
        enable_caching=True,
        use_eagle=False,
    )
    mgr._test_hasher = get_request_block_hasher(csa_state, sha256)
    return mgr


def _request(mgr, tokens, max_tokens):
    return Request(
        f"r{next(_ids)}",
        list(tokens),
        SamplingParams(max_tokens=max_tokens),
        None,
        block_hasher=mgr._test_hasher,
    )


def _prefill(mgr, req, step=128):
    blocks, hit, _ = mgr.get_computed_blocks(req)
    first = True
    while req.num_computed_tokens < req.num_prompt_tokens:
        local = hit if first else 0
        n = min(step, req.num_prompt_tokens - req.num_computed_tokens - local)
        out = mgr.allocate_slots(
            req,
            n,
            num_new_computed_tokens=local,
            new_computed_blocks=blocks if first else None,
        )
        assert out is not None
        req.num_computed_tokens += n + local
        first = False
    return hit


def _decode(mgr, req, steps, accept=4):
    for _ in range(steps):
        for _ in range(accept):
            req.append_output_token_ids(7_000_000 + len(req.output_token_ids))
        assert mgr.allocate_slots(req, accept, num_lookahead_tokens=6) is not None
        req.num_computed_tokens += accept


def _run(mgr, tokens, decode_steps=50):
    req = _request(mgr, tokens, decode_steps * 4)
    hit = _prefill(mgr, req)
    _decode(mgr, req, decode_steps)
    mgr.free(req)
    return hit


def _cached_blocks(mgr, group):
    return sum(
        b.block_hash is not None and get_group_id(b.block_hash) == group
        for b in mgr.block_pool.blocks
    )


def test_finds_only_the_128_ratio_state_group(block_size):
    mgr = _manager(64, block_size)
    assert hsc.hca_state_group_ids(mgr.kv_cache_config) == {HCA_STATE}


def _scenario(mgr):
    """Hits of: next turn, a truncated history, and a next turn after churn."""
    hits = []
    _run(mgr, TURN1, decode_steps=300)
    hits.append(mgr.get_computed_blocks(_request(mgr, TURN2, 4))[1])
    truncated = TURN1[:40_100] + [3] * 900
    hits.append(mgr.get_computed_blocks(_request(mgr, truncated, 4))[1])
    req = _request(mgr, TURN2, 1300)
    hits.append(_prefill(mgr, req))
    for step in range(300):
        _decode(mgr, req, 1)
        if step % 10 == 0:
            base = 10_000_000 + step * 100_000
            _run(mgr, range(base, base + 3000), decode_steps=5)
    mgr.free(req)
    turn3 = TURN2[:-2] + list(range(60_000_000, 60_002_000))
    hits.append(mgr.get_computed_blocks(_request(mgr, turn3, 10))[1])
    return hits


def test_hits_are_unchanged_and_the_group_caches_nothing(block_size, monkeypatch):
    num_blocks = 130 * 1024 // block_size
    expected = _scenario(_manager(num_blocks, block_size))

    hsc.install(monkeypatch.setattr)
    mgr = _manager(num_blocks, block_size)
    assert _scenario(mgr) == expected
    assert expected[0] == 39_936

    fresh = _manager(num_blocks, block_size)
    _run(fresh, TURN1)
    blocks, hit, _ = fresh.get_computed_blocks(_request(fresh, TURN2, 4))
    assert hit == expected[0]
    assert blocks.blocks[HCA_STATE] and all(b.is_null for b in blocks.blocks[HCA_STATE])
    assert _cached_blocks(fresh, HCA_STATE) == 0


def test_caches_nothing_for_the_group(block_size, monkeypatch):
    num_blocks = 400 * 1024 // block_size
    base = _manager(num_blocks, block_size)
    _run(base, TURN1)
    hsc.install(monkeypatch.setattr)
    lean = _manager(num_blocks, block_size)
    _run(lean, TURN1)
    # Two checkpoints (32k and the replay boundary), each a 128-token window.
    hca_blocks_per_checkpoint = 128 // GEOMETRY[block_size][2]
    assert _cached_blocks(base, HCA_STATE) == 2 * hca_blocks_per_checkpoint
    assert _cached_blocks(lean, HCA_STATE) == 0
    for group in range(4):
        assert _cached_blocks(lean, group) == _cached_blocks(base, group)
