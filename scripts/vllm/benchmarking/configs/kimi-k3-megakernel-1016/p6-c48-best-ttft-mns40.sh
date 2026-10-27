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


# shellcheck disable=SC2034  # read by recipe-base.sh
RECIPE_SET=oct16
RECIPE_CONCURRENCY=48
RECIPE_DECODE=dspark_dp2
RECIPE_DECODE_MAX_NUM_SEQS=40
RECIPE_DRAFTS=3
RECIPE_ACCEPTANCE=2.87
RECIPE_DRAFTS_PER_BATCH_SIZE='[[1,32,3],[33,40,0]]'
RECIPE_PREFILL=long
RECIPE_STORE_SERVERS=1
RECIPE_PREFILL_MAX_NUM_SEQS=8
RECIPE_PREFILL_LONG_THRESHOLD=2048
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel/recipe-base.sh
source "$(dirname "${BASH_SOURCE[0]}")/../kimi-k3-megakernel/recipe-base.sh"
export K3_HOT_PLAN=hot_plan_p6c48_verify8.npz
export K3_FP8_WIRES=t0,t10
export K3_PREFILL_PREFIX_FP8=1
