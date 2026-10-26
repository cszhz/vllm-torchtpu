#!/bin/bash
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
# Point 6 (c48, DSpark) with the decode/prefill numerics flags of
# dev/k3-kernel-numerics: hot-expert fp8 copies (serving verify plan), fp8 wires
# for the t0/t10 gathers and fp8 cached-prefix attention. Needs an engine built
# from that branch (K3_ENGINE_REF).
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel-1016/p6-c48.sh
source "$(dirname "${BASH_SOURCE[0]}")/p6-c48.sh"
export K3_HOT_PLAN=hot_plan_p6c48_verify8.npz
export K3_FP8_WIRES=t0,t10
export K3_PREFILL_PREFIX_FP8=1
