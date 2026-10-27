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
"""Native FP8 attention runs on the target model only, never on a speculative draft.

``TPU_RPA_NATIVE_FP8`` (the decode kernel) and ``TPU_RPA_NATIVE_FP8_PREFILL``
(the prefill and mixed kernels) make the batched RPA kernel run its QK and/or
PV matmuls in fp8. The batched RPA wrapper reads them whenever its caller
passes ``native_fp8=None``. A draft model must run at the precision it ships
with, so its attention keeps the non-FP8 path whatever these variables say.

The runner marks each draft attention layer's impl
(``PallasAttentionBackendImpl.native_fp8_draft = True``) before it builds that
layer's kernels. The kernel builders then bind ``native_fp8_off=True`` into
the draft's batched, scheduled and DCP kernel entries, which pass
``native_fp8=""`` and ``native_fp8_prefill=""`` (off) to the wrapper. The flag
is part of every kernel registry key, so a draft layer never shares a target
layer's custom op.

With both variables unset (or "0") nothing changes: ``draft_off`` is False,
no entry binds anything new, and every key resolves to the op it did before.
"""

from vllm_torchtpu import envs

# The wrapper's "off" value for both arguments (configs.ServingConfigs).
OFF_KWARGS = {"native_fp8": "", "native_fp8_prefill": ""}


def _mode(raw: str | None) -> str:
    raw = (raw or "").strip()
    return "" if raw in ("", "0") else raw


def requested() -> tuple[str, str]:
    """(decode, prefill) native FP8 modes the environment asks for; "" = off."""
    return _mode(envs.TPU_RPA_NATIVE_FP8), _mode(envs.TPU_RPA_NATIVE_FP8_PREFILL)


def draft_off(is_draft: bool) -> bool:
    """Whether a layer's kernels must pin native FP8 off: a draft layer while
    either variable asks for native FP8."""
    return bool(is_draft) and any(requested())


def scope_line(num_target: int, num_draft: int) -> str:
    """The startup line that states the scope, for the compliance log check."""
    decode, prefill = requested()
    return (
        f"TPU_RPA_NATIVE_FP8: target={decode or 'off'} draft=off | "
        f"TPU_RPA_NATIVE_FP8_PREFILL: target={prefill or 'off'} draft=off | "
        f"attention layers: {num_target} target, {num_draft} draft"
    )
