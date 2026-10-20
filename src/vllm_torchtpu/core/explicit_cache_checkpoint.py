# SPDX-License-Identifier: Apache-2.0
"""Select one explicit cache checkpoint through vLLM's shared-prefix boundary."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request

_PATCH_MARKER = "_tpu_explicit_cache_checkpoint"


def _explicit_boundary(manager: KVCacheManager, request: Request) -> int | None:
    if not manager.prefix_cache_lookup_enabled(request):
        return None
    params = request.sampling_params
    if params is None or not isinstance(params.extra_args, dict):
        return None
    cache_control = params.extra_args.get("cache_control_params")
    if not isinstance(cache_control, dict):
        return None
    positions = cache_control.get("token_positions")
    if not isinstance(positions, list):
        return None
    position = max(
        (p for p in positions if type(p) is int and 0 < p <= request.num_prompt_tokens),
        default=0,
    )
    scheduler_block_size = manager.coordinator.scheduler_block_size
    boundary = position // scheduler_block_size * scheduler_block_size
    return boundary or None


def patch_explicit_cache_checkpoint() -> None:
    """Wrap both vLLM 0.29 prefix lookups without changing computed-token hits."""
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager

    original = KVCacheManager.get_computed_blocks
    if not original.__dict__.get(_PATCH_MARKER, False):

        @wraps(original)
        def get_computed_blocks(
            self: KVCacheManager, request: Request
        ) -> tuple[KVCacheBlocks, int, int]:
            result = original(self, request)
            boundary = _explicit_boundary(self, request)
            if boundary is None:
                return result
            blocks, num_computed_tokens, _ = result
            return blocks, num_computed_tokens, boundary

        get_computed_blocks.__dict__[_PATCH_MARKER] = True
        KVCacheManager.get_computed_blocks = get_computed_blocks

    original_connector = KVCacheManager.get_computed_blocks_for_connector
    if not original_connector.__dict__.get(_PATCH_MARKER, False):

        @wraps(original_connector)
        def get_computed_blocks_for_connector(
            self: KVCacheManager, request: Request
        ) -> tuple[KVCacheBlocks, int, int, bool]:
            result = original_connector(self, request)
            boundary = _explicit_boundary(self, request)
            if boundary is None:
                return result
            blocks, num_computed_tokens, _, hit_diverged = result
            return blocks, num_computed_tokens, boundary, hit_diverged

        get_computed_blocks_for_connector.__dict__[_PATCH_MARKER] = True
        KVCacheManager.get_computed_blocks_for_connector = (
            get_computed_blocks_for_connector
        )
