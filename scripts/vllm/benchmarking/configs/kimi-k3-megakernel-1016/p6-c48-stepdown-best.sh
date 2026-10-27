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
# Point 6 (c48) with 64 decode sequences and a draft count that steps down
# with the decode batch: three drafts up to 20 requests, one up to 64. The wider
# decode removes the seat wait at c48; one draft keeps speculation on above 32.
# p6-c48-best's numerics flags. Needs an engine with one-draft DSpark
# (tmp/k3-spec1) as K3_ENGINE_REF.
RECIPE_SET=oct16
RECIPE_CONCURRENCY=48
RECIPE_DECODE=dspark_dp2
RECIPE_DECODE_MAX_NUM_SEQS=64
RECIPE_DRAFTS=3
RECIPE_ACCEPTANCE=2.87
RECIPE_DRAFTS_PER_BATCH_SIZE='[[1,20,3],[21,64,1]]'
RECIPE_ACCEPTANCE_LENGTHS='{"3":2.87,"1":1.84}'
RECIPE_PREFILL=long
RECIPE_STORE_SERVERS=1
source "$(dirname "${BASH_SOURCE[0]}")/../kimi-k3-megakernel/recipe-base.sh"
export K3_HOT_PLAN=hot_plan_p6c48_verify8.npz
export K3_FP8_WIRES=t0,t10
export K3_PREFILL_PREFIX_FP8=1
