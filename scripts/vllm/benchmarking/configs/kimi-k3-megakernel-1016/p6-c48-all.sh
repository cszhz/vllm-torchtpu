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
# p6-c48-best with the prefill EP16 expert placement (plan built from p6-c48
# serving prefill route counts) and the shortest-uncached-prompt-first prefill
# queue. Needs an engine with both (dev/k3-kernel-numerics d5a48b7b3 or later).
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel-1016/p6-c48-best.sh
source "$(dirname "${BASH_SOURCE[0]}")/p6-c48-best.sh"
export K3_PREFILL_EXPERT_PLAN=plan_prefill_serve.npz
export K3_PREFILL_SJF=1
