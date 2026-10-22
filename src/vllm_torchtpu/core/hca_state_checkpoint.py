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
"""No prefix-cache checkpoint for DeepSeek-V4's 128-ratio compressor state
(``TPU_KV_DROP_HCA_STATE_CHECKPOINT``).

The compressor of a 128-ratio (HCA) layer pools each run of 128 tokens into
one compressed entry, written at the run's last position and read from a
128-token state window that is exactly that run (no overlap, unlike the
4-ratio compressor's 8-token window). A prefix-cache hit always lands on a
multiple of 128 tokens, i.e. at the start of a run, so nothing in that state
group before the hit point is ever read into a stored entry.

vLLM still treats the group as a sliding window of 128 tokens: every hit needs
its four 32-token blocks before the boundary, so they are cached, held with
``TPU_KV_HOLD_REPLAY_CHECKPOINT`` and offloaded as four host keys per
checkpoint. With this flag the group is satisfied by null blocks at any hit
point, caches and holds nothing, and the Raiden connector neither stores,
looks up nor loads it. Its blocks for tokens being computed are unchanged.
"""

from __future__ import annotations

import functools

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_STATE_SUFFIX = ".attn.compressor.state_cache"
# The DSv4 compressor's state window is ratio * (2 if ratio == 4 else 1), so a
# 128-token window is the non-overlapping 128-ratio compressor.
_NON_OVERLAP_WINDOW = 128


def enabled() -> bool:
    return envs.TPU_KV_DROP_HCA_STATE_CHECKPOINT


def hca_state_group_ids(kv_cache_config) -> frozenset[int]:
    """KV cache groups holding only 128-ratio main-compressor state caches."""
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    ids = set()
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        names = group.layer_names
        spec = group.kv_cache_spec
        if not names or not all(
            n.endswith(_STATE_SUFFIX) and ".indexer." not in n for n in names
        ):
            continue
        if isinstance(spec, SlidingWindowSpec) and spec.sliding_window == (
            _NON_OVERLAP_WINDOW
        ):
            ids.add(gid)
    return frozenset(ids)


def _no_checkpoint_mask(start_block, end_block, *args, **kwargs) -> list[bool]:
    return [False] * (end_block - start_block)


def _stateless_manager_cls(base):
    class StatelessResumeManager(base):
        """A windowed group whose state before a hit point is never read."""

        @classmethod
        def find_longest_cache_hit(
            cls,
            block_hashes,
            max_length,
            kv_cache_group_ids,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
            alignment_tokens,
            dcp_world_size=1,
            pcp_world_size=1,
        ):
            if alignment_tokens % kv_cache_spec.sliding_window:
                raise ValueError(
                    "TPU_KV_DROP_HCA_STATE_CHECKPOINT needs prefix hits aligned "
                    f"to the {kv_cache_spec.sliding_window}-token compressor run, "
                    f"got {alignment_tokens}"
                )
            hit = max_length // alignment_tokens * alignment_tokens
            num_blocks = hit // kv_cache_spec.block_size
            return (
                tuple([block_pool.null_block] * num_blocks for _ in kv_cache_group_ids),
                hit,
            )

    return StatelessResumeManager


def _apply(coordinator) -> None:
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    groups = coordinator.kv_cache_config.kv_cache_groups
    ids = hca_state_group_ids(coordinator.kv_cache_config)
    if not ids:
        raise ValueError(
            "TPU_KV_DROP_HCA_STATE_CHECKPOINT: no 128-ratio compressor state group "
            f"in {[g.layer_names[:2] for g in groups]}"
        )
    for gid in ids:
        manager = coordinator.single_type_managers[gid]
        if type(manager) is not SlidingWindowManager:
            raise ValueError(
                f"TPU_KV_DROP_HCA_STATE_CHECKPOINT: group {gid} is managed by "
                f"{type(manager).__name__}"
            )
        manager.reachable_block_mask = _no_checkpoint_mask
        manager._tpu_no_checkpoint = True
    attention_groups = []
    for group in coordinator.attention_groups:
        if ids.intersection(group.group_ids):
            if not ids.issuperset(group.group_ids):
                raise ValueError(
                    "TPU_KV_DROP_HCA_STATE_CHECKPOINT: groups "
                    f"{group.group_ids} share a spec with other groups"
                )
            group = group._replace(
                manager_cls=_stateless_manager_cls(group.manager_cls)
            )
        attention_groups.append(group)
    coordinator.attention_groups = attention_groups
    logger.info(
        "TPU_KV_DROP_HCA_STATE_CHECKPOINT: KV cache groups %s need no prefix-cache "
        "checkpoint",
        sorted(ids),
    )


def patch_drop_hca_state_checkpoint() -> None:
    if not enabled():
        return
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    if getattr(HybridKVCacheCoordinator, "_tpu_hca_state_patch", False):
        return
    install(setattr)
    HybridKVCacheCoordinator._tpu_hca_state_patch = True


def install(set_attr) -> None:
    """Install with ``set_attr`` (``setattr`` or a test's monkeypatch)."""
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    original_init = HybridKVCacheCoordinator.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _apply(self)

    set_attr(HybridKVCacheCoordinator, "__init__", __init__)
