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
"""KV cache spec for DeepSeek-V4's 128-ratio compressor state
(``TPU_KV_DROP_HCA_STATE_CHECKPOINT``).

The compressor of a 128-ratio (HCA) layer pools each run of 128 tokens into
one compressed entry, written at the run's last position from a state window
that is exactly that run (no overlap, unlike the 4-ratio compressor's 8-token
window). A prefix-cache hit lands on a scheduler-block boundary, a whole number
of runs, so no state before the hit point is read again.

vLLM would treat the group as a 128-token sliding window: every checkpoint
caches, holds and offloads the blocks of the window before it. Declared not
prefix-cacheable, the group is skipped by the hit lookup, caches nothing and
is left out of the KV offload groups; its blocks for tokens being computed are
unchanged. Kept free of model imports: the engine core unpickles it.
"""

from dataclasses import dataclass

from vllm.v1.kv_cache_interface import SlidingWindowMLASpec


@dataclass(frozen=True, kw_only=True)
class RunLocalStateSpec(SlidingWindowMLASpec):
    """Compressor state whose window is exactly one compressed run."""

    @property
    def prefix_cacheable(self) -> bool:
        return False


def state_spec_cls(
    compress_ratio: int, sliding_window: int, block_size: int, drop: bool
) -> type[SlidingWindowMLASpec]:
    """The spec class for a compressor state cache."""
    if not drop or sliding_window != compress_ratio:
        return SlidingWindowMLASpec
    if block_size % compress_ratio:
        raise ValueError(
            "TPU_KV_DROP_HCA_STATE_CHECKPOINT needs prefix hits on whole "
            f"compressor runs: block size {block_size} is not a multiple of "
            f"{compress_ratio}"
        )
    return RunLocalStateSpec
