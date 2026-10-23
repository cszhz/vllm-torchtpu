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
"""Guard: the single-step prefill patches must match the installed vLLM.

``tests/test_single_step_prefill.py`` skips itself when the MambaManager
patches do not match vLLM's checkpoint API, and ``PrefixAwareScheduler`` then
only logs a warning and runs the stock ``align`` split. On vLLM 0.31 that made
every warm hybrid turn take 2-4 prefill steps instead of one (the last full
block and the 128-token partial tail each end a chunk), +43% TP8 prefill
latency, without a single failing test. This file fails instead of skipping.
"""

import inspect

from vllm.v1.core.single_type_kv_cache_manager import MambaManager

from vllm_torchtpu.core import single_step_prefill as ssp


def test_single_step_patches_match_installed_vllm():
    params = list(inspect.signature(MambaManager._needs_internal_checkpoint).parameters)
    assert ssp.patches_match_vllm(), (
        "single-step prefill is unusable on this vLLM: "
        f"MambaManager._needs_internal_checkpoint{tuple(params)}"
    )

