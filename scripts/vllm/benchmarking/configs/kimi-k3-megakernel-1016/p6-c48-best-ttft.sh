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
# p6-c48-best with a prefill scheduler for the TTFT tail: 8 prefill sequences
# per step instead of 4 and long prompts capped at 2,048 tokens per step instead
# of 4,096, so short turns are not queued behind long chunks.

# shellcheck disable=SC2034  # read by recipe-base.sh
RECIPE_PREFILL_MAX_NUM_SEQS=8
RECIPE_PREFILL_LONG_THRESHOLD=2048
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel-1016/p6-c48-best.sh
source "$(dirname "${BASH_SOURCE[0]}")/p6-c48-best.sh"
