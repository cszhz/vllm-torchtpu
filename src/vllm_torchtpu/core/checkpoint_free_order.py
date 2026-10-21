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
"""Hold a request's replay-boundary window checkpoint until the request ends.

With sparse prefix-cache retention, the next turn of a conversation hits at the
previous request's replay boundary B (its last scheduler block before the final
prompt token) only if every full-attention block before B *and* each windowed
group's few blocks just before B are cached. vLLM frees those window blocks as
soon as the window slides past them, early in decode, while the full-attention
blocks stay held until the request finishes. Through a long decode the
checkpoint ages at the head of the LRU free queue; under memory pressure it is
evicted (or reallocated before the KV-offload store of it has launched, which
cancels that store), and the next turn falls back to an older checkpoint even
though the whole prefix is still resident.

With ``TPU_KV_HOLD_REPLAY_CHECKPOINT=1`` windowed groups keep those blocks
until the request is freed, and a freed request queues them with the prefix
they serve: the tail past B is evicted first, then the checkpoint, then the
full-attention blocks before B, head last.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

logger = init_logger(__name__)


def replay_checkpoint_range(manager, num_prompt_tokens: int) -> range:
    """Block indices of a windowed group's replay-boundary checkpoint."""
    alignment = manager.cache_hit_alignment_tokens
    boundary = (num_prompt_tokens - 1) // alignment * alignment
    end = boundary // manager.block_size
    need = manager._contiguous_blocks_for_hit(
        manager.sliding_window, manager.block_size, manager.use_eagle
    )
    return range(max(0, end - need), end) if boundary > 0 else range(0)


def _remove_skipped_blocks_holding_checkpoint(
    self, request_id: str, processed_computed_tokens: int, num_prompt_tokens=None
) -> None:
    num_skipped = self.get_num_skipped_tokens(processed_computed_tokens)
    if num_skipped <= 0 or request_id not in self.req_to_blocks:
        return
    assert num_prompt_tokens is not None
    blocks = self.req_to_blocks[request_id]
    hold = replay_checkpoint_range(self, num_prompt_tokens)
    freed: list[KVCacheBlock] = []
    # Backwards to the first freed block, stepping over the held checkpoint.
    for i in range(min(num_skipped // self.block_size, len(blocks)) - 1, -1, -1):
        if i in hold:
            continue
        if blocks[i] == self._null_block:
            break
        freed.append(blocks[i])
        blocks[i] = self._null_block
    if freed:
        self.block_pool.free_blocks(freed)


def chain_ordered_blocks(kv_cache_manager, request: Request) -> list[KVCacheBlock]:
    """Pop the request's blocks in eviction order (first entry evicted first)."""
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        SlidingWindowManager,
    )

    managers = kv_cache_manager.coordinator.single_type_managers
    held = [m.pop_blocks_for_free(request.request_id) for m in managers]
    full = [i for i, m in enumerate(managers) if isinstance(m, FullAttentionManager)]
    if len(full) != 1:
        raise ValueError(
            "TPU_KV_HOLD_REPLAY_CHECKPOINT needs exactly one full-attention group"
        )
    full_mgr = managers[full[0]]
    alignment = full_mgr.cache_hit_alignment_tokens
    boundary = (request.num_prompt_tokens - 1) // alignment * alignment
    split = boundary // full_mgr.block_size

    rest: list[KVCacheBlock] = []
    checkpoint: list[KVCacheBlock] = []
    for i, (manager, blocks) in enumerate(zip(managers, held)):
        if i == full[0]:
            continue
        hold = (
            replay_checkpoint_range(manager, request.num_prompt_tokens)
            if isinstance(manager, SlidingWindowManager)
            else range(0)
        )
        for idx in range(len(blocks) - 1, -1, -1):
            (checkpoint if idx in hold else rest).append(blocks[idx])
    chain = held[full[0]]
    return rest + chain[split:][::-1] + checkpoint + chain[:split][::-1]


def _free(self, request) -> None:
    self.block_pool.free_blocks(chain_ordered_blocks(self, request))


def _pop_blocks_for_free(self, request) -> list[KVCacheBlock]:
    # Callers free the returned list reversed.
    return chain_ordered_blocks(self, request)[::-1]


def patch_hold_replay_checkpoint() -> None:
    if not envs.TPU_KV_HOLD_REPLAY_CHECKPOINT:
        return
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    if KVCacheManager.free is _free:
        return
    KVCacheManager.free = _free
    KVCacheManager.pop_blocks_for_free = _pop_blocks_for_free
    SlidingWindowManager.remove_skipped_blocks = (
        _remove_skipped_blocks_holding_checkpoint
    )
    logger.info(
        "TPU_KV_HOLD_REPLAY_CHECKPOINT: windowed groups hold each request's "
        "replay-boundary checkpoint until the request is freed"
    )
