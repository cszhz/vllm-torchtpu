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
"""Conservative cache accounting for the opt-in remote Draft schedulers."""

from collections.abc import Iterable

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request


def peak_blocks(
    cache_manager: KVCacheManager,
    request: Request,
    lookahead: int = 2,
    recycle_align: bool = False,
) -> int:
    """Overestimate physical demand without querying mutable predictors.

    Use the validated recycling bound only for align mode; non-align modes
    retain a conservative dense positional bound.
    Prefix sharing, other cache kinds and async execution require separate
    reasoning and are unsupported by this adapter.
    """
    if cache_manager.enable_caching:
        raise NotImplementedError("remote reservation requires prefix caching off")
    cap = request.max_tokens
    if cap is None or cap < 1:
        raise ValueError("remote reservation requires a bounded generation cap")
    tokens = min(
        request.num_prompt_tokens + cap + lookahead, cache_manager.max_model_len
    )
    total = 0
    for group in cache_manager.kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            leaves = tuple(spec.kv_cache_specs.values())
            if not leaves or any(s != leaves[0] for s in leaves):
                raise NotImplementedError("nonuniform reservation group")
            # A uniform group consumes one shared page index per position,
            # irrespective of its number of backing per-layer tensors.
            spec = leaves[0]
        if type(spec) not in (FullAttentionSpec, MLAAttentionSpec, MambaSpec):
            raise NotImplementedError(f"unvalidated cache specification: {type(spec)}")
        if spec.block_size <= 0:
            raise ValueError("reservation requires positive cache block sizes")
        if isinstance(spec, MambaSpec):
            if spec.mamba_cache_mode not in ("none", "align"):
                raise NotImplementedError(
                    f"unvalidated Mamba mode: {spec.mamba_cache_mode}"
                )
            if min(spec.num_speculative_blocks, spec.num_prefill_checkpoint_blocks) < 0:
                raise ValueError("negative recurrent scratch reservation")
        if (
            isinstance(spec, MambaSpec)
            and recycle_align
            and spec.mamba_cache_mode == "align"
        ):
            # Pinned MambaSpec.max_memory_usage_bytes: old/current states,
            # speculative scratch and retained prefill checkpoint. Reserve
            # two additional scratch pages for a partial-tail allocation.
            total += (
                4 + spec.num_speculative_blocks + spec.num_prefill_checkpoint_blocks
            )
        else:
            total += (tokens + spec.block_size - 1) // spec.block_size
            if isinstance(spec, MambaSpec):
                # Pinned non-align allocation is ceil(tokens/block) + scratch.
                # Prefix caching is rejected above, so no shared-tail CoW
                # or align checkpoint pages can be allocated in this branch.
                total += spec.num_speculative_blocks
                if spec.mamba_cache_mode == "align":
                    total += 2
    return total


def remaining_reservation(
    cache_manager: KVCacheManager,
    requests: Iterable[Request],
    lookahead: int = 2,
    recycle_align: bool = False,
) -> int:
    """Count future demand for every live owner, including held P rows.

    Only credit blocks currently owned by that request. A connector's popped
    pages remain unavailable in block_pool, even when absent from req_to_blocks.
    The caller must use all admitted requests, not temporarily selected rows.
    """
    reserved = 0
    for request in requests:
        blocks = cache_manager.coordinator.get_blocks(request.request_id)
        owned = {
            block.block_id for group in blocks for block in group if not block.is_null
        }
        reserved += max(
            0,
            peak_blocks(cache_manager, request, lookahead, recycle_align) - len(owned),
        )
    return reserved


def can_admit(
    cache_manager: KVCacheManager,
    request: Request,
    admitted: Iterable[Request],
    lookahead: int = 2,
    recycle_align: bool = False,
) -> bool:
    required = peak_blocks(cache_manager, request, lookahead, recycle_align)
    required += remaining_reservation(cache_manager, admitted, lookahead, recycle_align)
    return required <= cache_manager.block_pool.get_num_free_blocks()
