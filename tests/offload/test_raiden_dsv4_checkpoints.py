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
lookup) through `TPURaidenOffloadingScheduler` over a FakeKVCacheStore, at both
served block sizes: one full-attention group, two sliding-window groups with a
128-token window (one of them the EAGLE/MTP group), the 4-ratio compressor
state group with an 8-token window and the 128-ratio compressor state group
with a 128-token window.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_raiden_store_offload import FakeKVCacheStore, make_manager, sub_hashes
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.kv_offload.base import OffloadingKVEventsConfig, get_offload_group_idx
from vllm.v1.request import Request

# --block-size -> (SWA block, 4-ratio state block = hash block, HCA state
# block), as the TPU DeepSeek-V4 layers page them.
GEOMETRY = {1024: (128, 16, 32), 512: (64, 8, 16)}
FULL_GROUP = 2
HCA_STATE_GROUP = 4
PROMPT = 10_000
# Replay boundary of PROMPT: 9 full chunks at 1024, 19 at 512.
REPLAY = {1024: 9216, 512: 9728}

init_none_hash(sha256)


def dsv4_groups(full: int) -> list[tuple[str, int, int | None, bool]]:
    """(layer name, block size, sliding window or None, EAGLE/MTP group)."""
    swa, csa_state, hca_state = GEOMETRY[full]
    return [
        ("model.layers.0.attn.swa_cache", swa, 128, False),
        ("mtp.0.attn.swa_cache", swa, 128, True),
        ("model.layers.0.attn", full, None, False),
        ("model.layers.2.attn.compressor.state_cache", csa_state, 8, False),
        ("model.layers.0.attn.compressor.state_cache", hca_state, 128, False),
    ]


def dsv4_kv_cache_config(full: int) -> KVCacheConfig:
    groups = []
    for name, block_size, window, eagle in dsv4_groups(full):
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


def make_scheduler(retention_interval: int | None, full: int):
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

    store = FakeKVCacheStore()
    manager = make_manager(store, device_block_size_factor=1, num_blocks=1 << 20)
    block_sizes = [g[1] for g in dsv4_groups(full)]
    spec = SimpleNamespace(
        config=SimpleNamespace(
            groups=[
                SimpleNamespace(group_id=i, tokens_per_block=bs)
                for i, bs in enumerate(block_sizes)
            ]
        ),
        blocks_per_chunk=1,
        tokens_per_block=block_sizes,
        tokens_per_hash=GEOMETRY[full][1],
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
    sched = TPURaidenOffloadingScheduler(spec, vllm_config, dsv4_kv_cache_config(full))
    return sched, store


def make_request(sched, req_id: str, tokens: list[int]) -> Request:
    params = SamplingParams(max_tokens=1)
    params.update_from_generation_config({}, 2)
    hasher = get_request_block_hasher(sched.config.tokens_per_hash, sha256)
    return Request(req_id, tokens, params, None, block_hasher=hasher)


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
    out = [0] * (HCA_STATE_GROUP + 1)
    for key in keys:
        out[get_offload_group_idx(key)] += 1
    return out


def host_hit(sched, tokens, local=0, req_id="b") -> int | None:
    req = make_request(sched, req_id, tokens)
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
            (1024, 32768): [1, 2, 9, 1, 4],
            (1024, 4096): [3, 6, 9, 3, 12],
            (512, 32768): [2, 3, 19, 1, 8],
        }
        for (full, interval), per_group in expected.items():
            sched, store = make_scheduler(interval, full)
            stored = prefill(sched, store, make_request(sched, "a", TURN1))
            with self.subTest(full=full, interval=interval):
                self.assertEqual(stored_per_group(stored), per_group)
                self.assertEqual(host_hit(sched, TURN2), REPLAY[full])


class TestDropHcaStateCheckpoint(unittest.TestCase):
    def test_hca_state_group_is_never_stored_looked_up_or_loaded(self):
        for full, per_group in ((1024, [1, 2, 9, 1, 0]), (512, [2, 3, 19, 1, 0])):
            with self.subTest(full=full):
                self._check(full, per_group)

    def _check(self, full, per_group):
        with patch.dict("os.environ", {"TPU_KV_DROP_HCA_STATE_CHECKPOINT": "1"}):
            sched, store = make_scheduler(32768, full)
        stored = prefill(sched, store, make_request(sched, "a", TURN1))
        self.assertEqual(stored_per_group(stored), per_group)

        req = make_request(sched, "b", TURN2)
        sched.on_new_request(req)
        hit, _ = sched.get_num_new_matched_tokens(req, 0)
        self.assertEqual(hit, REPLAY[full])
        # Blocks as the KV cache manager allocates them for this external
        # hit: windowed groups hold only their window.
        next_id = iter(range(1000, 2000))
        null = KVCacheBlock(block_id=0)
        null.is_null = True
        groups = []
        for _, block_size, window, eagle in dsv4_groups(full):
            n = hit // block_size
            w = n if window is None else -(-window // block_size) + int(eagle)
            groups.append(
                [null] * (n - w) + [KVCacheBlock(next(next_id)) for _ in range(w)]
            )
        sched.update_state_after_alloc(req, KVCacheBlocks(tuple(groups)), hit)
        (job,) = sched._current_batch_load_jobs.values()
        loaded = stored_per_group(job.src_spec.keys)
        self.assertEqual(loaded[HCA_STATE_GROUP], 0)
        self.assertEqual(job.dst_spec.group_sizes[HCA_STATE_GROUP], 0)
        self.assertEqual(loaded[FULL_GROUP], hit // full)


if __name__ == "__main__":
    unittest.main()
