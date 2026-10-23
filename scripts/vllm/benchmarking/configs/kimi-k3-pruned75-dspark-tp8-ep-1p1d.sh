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
# Two-slice baseline for spec_in_prefll_to_decode. This exercises main's
# existing one-time target + draft cache handoff: P builds DSpark context and
# D owns DSpark proposals after handoff. It is deliberately not the remote
# per-round proposer design; its measurements are the comparison baseline.

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-pruned75-dspark-tp8-ep.sh"

case "${K3_DSPARK_PD_ROLE:-}" in
    prefill)
        _kv_role=kv_producer
        export TPU_RAIDEN_JOB_NAME=spec-in-prefll-to-decode-prefill
        export TPU_RAIDEN_ENGINE_ID=spec-in-prefll-to-decode-prefill-engine
        export TPU_KV_TRANSFER_NAMESPACE=spec-in-prefll-to-decode-prefill
        export TPU_RAIDEN_RESHARD_PORT_BASE=27340
        export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27440
        ;;
    decode)
        _kv_role=kv_consumer
        export TPU_RAIDEN_JOB_NAME=spec-in-prefll-to-decode-decode
        export TPU_RAIDEN_ENGINE_ID=spec-in-prefll-to-decode-decode-engine
        export TPU_KV_TRANSFER_NAMESPACE=spec-in-prefll-to-decode-decode
        export TPU_RAIDEN_RESHARD_PORT_BASE=28340
        export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28440
        ;;
    *)
        echo "ERROR: K3_DSPARK_PD_ROLE must be prefill or decode" >&2
        return 1
        ;;
esac

export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_CLIENT_IMPL=cpp
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_RAIDEN_KIMIK3_ADMISSION=1
export TPU_RAIDEN_QWEN35_ADMISSION=0
export TPU_RAIDEN_GLM_ADMISSION=0
export TPU_RAIDEN_DST_SHARDS=8
export TPU_RAIDEN_TRANSFER_PARALLELISM=8
export TPU_RAIDEN_PREFIX_AWARE_LOAD=1
export TPU_RAIDEN_TRANSFER_NUM_SLOTS=1
export TPU_PREMAPPED_BUFFER_SIZE=17179869184
export TPU_KV_SHM_POOL_GB=128
export TPU_P2P_WAIT_PULL_TIMEOUT=900
export TPU_KV_TRANSFER_PORT=29340
export TPU_SIDE_CHANNEL_PORT=29740

_kv_transfer_config="{\"kv_connector\":\"TPURaidenConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"${_kv_role}\"}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS} --kv-transfer-config ${_kv_transfer_config}"
