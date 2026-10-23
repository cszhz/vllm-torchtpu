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

# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Feasibility profile for splitting Kimi-K3 DSpark drafting from target verify.
# It intentionally runs the colocated implementation on one v7x-8: XProf
# records draft_context_update, draft_generate, and main_forward separately so
# dspark_pd_cost_model.py can estimate a two-engine pipeline before allocating
# a second slice.
MODEL="${MODEL_OVERRIDE:-mgoin/Kimi-K3-pruned75}"
# `${VAR-default}` intentionally preserves an explicit empty value. Falcon
# mounts the same GCS prefix locally and sets MODEL_OVERRIDE to that path,
# while CI uses this URI with the RunAI streamer.
MODEL_URI="${MODEL_URI-gs://tpu-commons-ci/moonshootai/kimi/k3-pruned75}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

# One exact-shape cell and one full batch wave keep the profile compact while
# exercising cross-request pipeline capacity. Profile runs use one wave even
# when NUM_PROMPTS is larger (run_benchmarks.sh's PROFILE_NUM_PROMPTS default).
ISL_OSL_CONFIGS="1024:256"
CONCURRENCY_OPTIONS="8"
NUM_PROMPTS=16
RANDOM_RANGE_RATIO="0.0"
BENCHMARK_TEMPERATURE="0"

MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=1024
MAX_NUM_SEQS=8
SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.82
KV_CACHE_DTYPE="fp8"
ASYNC_SCHEDULING=false
ATTENTION_BACKEND="CUSTOM"

# Target verify reaches 8 requests x (K + 1) rows. Draft-specific padding
# buckets are added by DFlashProposer; the list here covers target decode and
# the exact prefill chunk used by this profile.
_loader_args=""
if [ -n "$MODEL_URI" ]; then
    _loader_args='--model-loader-extra-config {"memory_limit":17179869184}'
fi
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} ${_loader_args} --block-size 256 --compilation-config {\"compile_sizes\":[1,8,16,32,64,1024]} --speculative-config {\"method\":\"dspark\",\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":7}"

EVAL_TOLERANCE="0.025"
