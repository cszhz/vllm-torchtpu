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

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-pruned75-dspark-tp8-ep.sh"

case "${K3_DSPARK_PD_ROLE:-}" in
    prefill) export VLLM_TPU_REMOTE_DSPARK_ROLE=producer ;;
    decode) export VLLM_TPU_REMOTE_DSPARK_ROLE=consumer ;;
    *)
        echo "ERROR: K3_DSPARK_PD_ROLE must be prefill or decode" >&2
        return 1
        ;;
esac

# The regular synthetic spec warmup drives one engine in isolation and cannot
# participate in the rank-paired protocol. The benchmark warmup compiles the
# live paired path before measurement instead.
export SPEC_WARMUP=0
export VLLM_TPU_REMOTE_DSPARK_PORT="${VLLM_TPU_REMOTE_DSPARK_PORT:-29800}"
export VLLM_TPU_REMOTE_DSPARK_TIMEOUT_S="${VLLM_TPU_REMOTE_DSPARK_TIMEOUT_S:-1800}"
