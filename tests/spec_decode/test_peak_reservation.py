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

"""Exercise future reservations against the actual upstream hybrid allocator."""

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu.core.peak_reservation import peak_blocks, remaining_reservation
from vllm_torchtpu.core.reserving_manager import ReservingManager


@pytest.mark.parametrize("k,expected", [(1, 27), (5, 11)])
@pytest.mark.parametrize("prompt", [791, 1295, 1297])
def test_native_hybrid_full_growth_and_deferred_release(k, expected, prompt):
    register_all_kvcache_specs(None)
    init_none_hash(sha256)
    layer = SimpleNamespace(
        get_state_shape=lambda: ((8192,),),
        get_state_dtype=lambda: (torch.float16,),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
        is_kv_cache_tp_replicated=False,
    )
    native = MambaBase.get_kv_cache_spec(
        layer,
        SimpleNamespace(
            cache_config=CacheConfig(
                mamba_block_size=4096,
                mamba_page_size_padded=829440,
                mamba_cache_mode="none",
            ),
            num_speculative_tokens=k,
        ),
    )
    specs = [
        MLAAttentionSpec(
            block_size=1296, num_kv_heads=1, head_size=64, dtype=torch.float16
        )
    ] + [native] * 3
    cache = KVCacheConfig(
        num_blocks=221,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"l{i}"], s) for i, s in enumerate(specs)],
    )
    manager = ReservingManager(
        KVCacheManager(
            cache,
            max_model_len=4096,
            scheduler_block_size=331776,
            hash_block_size=16,
            max_in_flight_tokens=1024,
            enable_caching=False,
        ),
        lookahead=k + 1,
        recycle_align=True,
    )

    def request(i):
        return Request(
            str(i), [1] * prompt, SamplingParams(max_tokens=1025, ignore_eos=True), None
        )

    rows = []
    for i in range(32):
        row = request(i)
        if manager.allocate_slots(row, prompt, num_lookahead_tokens=k + 1) is None:
            break
        row.num_computed_tokens = prompt
        row.status = RequestStatus.RUNNING
        rows.append(row)
    # These are fixture bounds, not a promise of client concurrency.
    assert len(rows) == expected
    assert manager.blocked == 1
    assert set(manager.admitted) == {row.request_id for row in rows}
    all_remaining = remaining_reservation(manager.manager, rows, k + 1, True)
    held_remaining = remaining_reservation(manager.manager, rows[1:], k + 1, True)
    if prompt == 791:
        assert all_remaining > held_remaining
    else:
        # Near the MLA page boundary, prefill lookahead already allocates
        # the second page; all future demand is physically charged up front.
        assert all_remaining == held_remaining == 0
    bound = sum(peak_blocks(manager.manager, r, k + 1, True) for r in rows)
    for step in range(1024):
        for row in rows:
            row.append_output_token_ids([1])
            assert (
                manager.allocate_slots(row, 1, num_lookahead_tokens=k + 1) is not None
            )
            row.num_computed_tokens += 1
            # Allocate the speculative window then roll back on the next step.
            if step % 64 == 0:
                assert (
                    manager.allocate_slots(row, k + 1, num_lookahead_tokens=k + 1)
                    is not None
                )
        assert 220 - manager.block_pool.get_num_free_blocks() <= bound
    pending = manager.pop_blocks_for_free(rows[0])
    before = manager.block_pool.get_num_free_blocks()
    assert rows[0].request_id not in manager.admitted
    manager.block_pool.free_blocks(reversed(pending))
    assert manager.block_pool.get_num_free_blocks() > before
    for row in rows[1:]:
        manager.free(row)
    assert manager.block_pool.get_num_free_blocks() == 220
    next_row = request("next")
    assert manager.allocate_slots(next_row, prompt) is not None
    manager.free(next_row)
