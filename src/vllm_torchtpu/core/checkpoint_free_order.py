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

``TPU_OFFLOAD_HASH_CHECKPOINT_BLOCKS=1`` also hashes every windowed block the
host offload stores, so a freed one is not reused before its store launches.
"""

from __future__ import annotations

import dataclasses
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


def held_checkpoint_indices(manager, num_prompt_tokens: int) -> set[int]:
    """The replay checkpoint; none for a group that needs no checkpoint
    (``hca_state_checkpoint`` sets ``_tpu_no_checkpoint``)."""
    if getattr(manager, "_tpu_no_checkpoint", False):
        return set()
    return set(replay_checkpoint_range(manager, num_prompt_tokens))


def _remove_skipped_blocks_holding_checkpoint(
    self, request_id: str, processed_computed_tokens: int, num_prompt_tokens=None
) -> None:
    num_skipped = self.get_num_skipped_tokens(processed_computed_tokens)
    if num_skipped <= 0 or request_id not in self.req_to_blocks:
        return
    assert num_prompt_tokens is not None
    blocks = self.req_to_blocks[request_id]
    hold = held_checkpoint_indices(self, num_prompt_tokens)
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
            held_checkpoint_indices(manager, request.num_prompt_tokens)
            if isinstance(manager, SlidingWindowManager)
            else set()
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


def _hash_offload_checkpoint_blocks(coordinator) -> None:
    """Hash every windowed block the host offload stores.

    A freed block without a hash goes to the front of the free queue and is
    reused at once. The offload connector cancels a store job when one of its
    source blocks is reallocated before the job launches, and with it every
    full-attention chunk in the same job, which is then never offered again.
    The offload policy also stores each checkpoint's EAGLE peek block, which
    the KV cache does not hash without the EAGLE block drop. Hashed, a freed
    block waits its turn at the back of the queue.
    """
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    groups = coordinator.kv_cache_config.kv_cache_groups
    peek_managers = []
    for group, manager in zip(groups, coordinator.single_type_managers):
        if isinstance(manager, SlidingWindowManager):
            _cache_offload_checkpoint_blocks(
                manager, coordinator.scheduler_block_size, group.is_eagle_group
            )
            if group.is_eagle_group:
                peek_managers.append(manager)
    if peek_managers:
        _cache_peek_blocks(coordinator, peek_managers)


def _cache_peek_blocks(coordinator, managers) -> None:
    """Let EAGLE-annotated groups cache one block past the aligned point.

    vLLM does this only for groups that still drop the EAGLE block; with the
    drop off (``VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP``) a checkpoint's peek block
    would leave the window before it became cacheable.
    """
    original = coordinator.cache_blocks

    def cache_blocks(request, num_computed_tokens: int) -> None:
        original(request, num_computed_tokens)
        finalized = max(0, num_computed_tokens - coordinator.num_reprefillable_tokens)
        aligned = coordinator._align_cacheable(finalized)
        if aligned == 0:
            return
        boundaries = coordinator.get_replay_boundaries(request)
        for manager in managers:
            if manager.use_eagle or not manager.enable_caching:
                continue  # vLLM's own EAGLE path covers it
            manager.cache_blocks(
                request,
                min(finalized, aligned + manager.block_size),
                retention_interval=coordinator.retention_interval,
                replay_boundaries=boundaries,
            )

    coordinator.cache_blocks = cache_blocks


def _cache_offload_checkpoint_blocks(manager, alignment: int, eagle: bool) -> None:
    """Also cache the blocks the offload connector's store mask keeps.

    The connector keeps a windowed chunk where ``reachable_block_mask``
    (aligned to the full-attention chunk, EAGLE peek per the group's
    annotation) does, with the same retention and replay boundaries.
    """
    block_mask = manager.reachable_block_mask
    store_mask = type(manager).reachable_block_mask

    def reachable_block_mask(
        start_block,
        end_block,
        alignment_tokens,
        kv_cache_spec,
        use_eagle,
        retention_interval=None,
        reachable_boundaries=(),
        **kwargs,
    ):
        mask = block_mask(
            start_block=start_block,
            end_block=end_block,
            alignment_tokens=alignment_tokens,
            kv_cache_spec=kv_cache_spec,
            use_eagle=use_eagle,
            retention_interval=retention_interval,
            reachable_boundaries=reachable_boundaries,
            **kwargs,
        )
        if mask is None:
            return None
        if kv_cache_spec.extra_retained_tokens:
            # The connector stores the widened window (see its
            # _reachable_store_block_mask).
            kv_cache_spec = dataclasses.replace(
                kv_cache_spec,
                sliding_window=kv_cache_spec.sliding_window
                + kv_cache_spec.extra_retained_tokens,
                extra_retained_tokens=0,
            )
        stored = store_mask(
            start_block=start_block,
            end_block=end_block,
            alignment_tokens=alignment,
            kv_cache_spec=kv_cache_spec,
            use_eagle=eagle,
            retention_interval=retention_interval,
            reachable_boundaries=reachable_boundaries,
            dcp_world_size=kwargs.get("dcp_world_size", 1),
        )
        if stored is None:
            return None
        return [a or b for a, b in zip(mask, stored)]

    manager.reachable_block_mask = reachable_block_mask


def install_offload_checkpoint_hashing(set_attr) -> None:
    """Hash the host offload's windowed checkpoint blocks at coordinator
    construction (``setattr`` or a test's monkeypatch)."""
    import functools

    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    original_init = HybridKVCacheCoordinator.__init__
    if getattr(original_init, "_tpu_offload_checkpoint_hashing", False):
        return

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _hash_offload_checkpoint_blocks(self)

    __init__._tpu_offload_checkpoint_hashing = True  # type: ignore[attr-defined]
    set_attr(HybridKVCacheCoordinator, "__init__", __init__)


def patch_hash_offload_checkpoint_blocks() -> None:
    if not envs.TPU_OFFLOAD_HASH_CHECKPOINT_BLOCKS:
        return
    install_offload_checkpoint_hashing(setattr)
    logger.info(
        "TPU_OFFLOAD_HASH_CHECKPOINT_BLOCKS: windowed blocks of host-offload "
        "checkpoints are hashed"
    )
