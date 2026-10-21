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
"""Raiden offload of DeepSeek-V4-shaped KV cache groups.

Drives vLLM's real offloading scheduler (store-job building and the hybrid
lookup) through `TPURaidenOffloadingScheduler` over a FakeKVCacheStore: one
1024-token full-attention group, two 128-token sliding-window groups (one of
them the EAGLE/MTP group), a 16-token state group with an 8-token window and
the 128-ratio compressor's 32-token state group with a 128-token window.
"""

import unittest
from types import SimpleNamespace

import torch
from test_raiden_store_offload import FakeKVCacheStore, make_manager, sub_hashes
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.kv_offload.base import OffloadingKVEventsConfig, get_offload_group_idx
from vllm.v1.request import Request

FULL = 1024
TOKENS_PER_HASH = 16
# (layer name, block size, sliding window or None, EAGLE/MTP group)
DSV4_GROUPS = [
    ("model.layers.0.attn.swa_cache", 128, 128, False),
    ("mtp.0.attn.swa_cache", 128, 128, True),
    ("model.layers.0.attn", FULL, None, False),
    ("model.layers.2.attn.compressor.state_cache", 16, 8, False),
    ("model.layers.0.attn.compressor.state_cache", 32, 128, False),
]
PROMPT = 10_000  # replay boundary 9216, 9 full chunks

init_none_hash(sha256)
_HASHER = get_request_block_hasher(TOKENS_PER_HASH, sha256)


def dsv4_kv_cache_config() -> KVCacheConfig:
    groups = []
    for name, block_size, window, eagle in DSV4_GROUPS:
        if window is None:
            spec = MLAAttentionSpec(
                block_size=block_size, num_kv_heads=1, head_size=64, dtype=torch.uint8
            )
        else:
            spec = SlidingWindowMLASpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.uint8,
                sliding_window=window,
            )
        groups.append(KVCacheGroupSpec([name], spec, is_eagle_group=eagle))
    return KVCacheConfig(
        num_blocks=1 << 16, kv_cache_tensors=[], kv_cache_groups=groups
    )


def make_scheduler(retention_interval: int | None):
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

    store = FakeKVCacheStore()
    manager = make_manager(store, device_block_size_factor=1, num_blocks=1 << 20)
    block_sizes = [g[1] for g in DSV4_GROUPS]
    spec = SimpleNamespace(
        config=SimpleNamespace(
            groups=[
                SimpleNamespace(group_id=i, tokens_per_block=bs)
                for i, bs in enumerate(block_sizes)
            ]
        ),
        blocks_per_chunk=1,
        tokens_per_block=block_sizes,
        tokens_per_hash=TOKENS_PER_HASH,
        offload_prompt_only=True,
        kv_events_config=OffloadingKVEventsConfig(
            enable_kv_cache_events=False, self_describing_kv_events=False
        ),
        get_manager=lambda: manager,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            prefix_cache_retention_interval=retention_interval,
            enable_prefix_caching=True,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(world_size=1, decode_context_parallel_size=1),
    )
    sched = TPURaidenOffloadingScheduler(spec, vllm_config, dsv4_kv_cache_config())
    return sched, store


def make_request(req_id: str, tokens: list[int]) -> Request:
    params = SamplingParams(max_tokens=1)
    params.update_from_generation_config({}, 2)
    return Request(req_id, tokens, params, None, block_hasher=_HASHER)


def prefill(sched, store, req, step_tokens=256) -> list[bytes]:
    """Chunked prefill of `req`, storing whatever the scheduler offers."""
    mgr = sched._raiden_manager
    sched.on_new_request(req)
    state = sched._req_status[req.request_id]
    next_block = iter(range(1, 1 << 30))
    stored = []
    while req.num_computed_tokens < req.num_prompt_tokens:
        n = min(step_tokens, req.num_prompt_tokens - req.num_computed_tokens)
        upto = req.num_computed_tokens + n
        for config, group_state in zip(
            sched.config.kv_group_configs, state.group_states
        ):
            while len(group_state.block_ids) < -(-upto // config.tokens_per_block):
                group_state.block_ids.append(next(next_block))
        state.update_offload_keys()
        jobs = sched._build_store_jobs(
            SimpleNamespace(
                num_scheduled_tokens={req.request_id: n}, finished_req_ids=set()
            )
        )
        for job_id, job in jobs.items():
            keys = job.dst_spec.keys
            mgr.submit_store_job(job_id, keys, job.src_spec.block_ids.tolist())
            store.complete_save([h for k in keys for h in sub_hashes(k, 1)])
            stored.extend(keys)
        mgr.poll_finished_jobs()
        req.num_computed_tokens += n
    return stored


def stored_per_group(keys) -> list[int]:
    out = [0] * len(DSV4_GROUPS)
    for key in keys:
        out[get_offload_group_idx(key)] += 1
    return out


def host_hit(sched, tokens, local=0, req_id="b") -> int | None:
    req = make_request(req_id, tokens)
    sched.on_new_request(req)
    hit, _ = sched.get_num_new_matched_tokens(req, local)
    return hit


TURN1 = list(range(1, PROMPT + 1))
TURN2 = TURN1 + list(range(10**6, 10**6 + 3000))


class TestRetentionIntervalStores(unittest.TestCase):
    """vLLM's own store policy under --prefix-cache-retention-interval keeps
    windowed chunks only at the replay boundary and interval segment tails."""

    def test_window_chunks_follow_the_retention_interval(self):
        expected = {
            32768: [1, 2, 9, 1, 4],
            4096: [3, 6, 9, 3, 12],
        }
        for interval, per_group in expected.items():
            sched, store = make_scheduler(interval)
            stored = prefill(sched, store, make_request("a", TURN1))
            with self.subTest(interval=interval):
                self.assertEqual(stored_per_group(stored), per_group)
                self.assertEqual(host_hit(sched, TURN2), 9216)


if __name__ == "__main__":
    unittest.main()
