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

# Opt-in experiment: select explicitly with K3_DSPARK_PD_CONFIG on BOTH roles.
# Historical Falcon source pins do not include these scheduler adapters.
# Five candidates match the Block5 checkpoint; do not inherit the base K7.
export DSPARK_NUM_SPECULATIVE_TOKENS="${DSPARK_NUM_SPECULATIVE_TOKENS:-5}"
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-pruned75-dspark-tp8-ep-remote.sh"

case "$K3_DSPARK_PD_ROLE" in
    prefill) _remote_scheduler=CapacityRemoteDraftScheduler ;;
    decode) _remote_scheduler=CapacityRemoteConsumerScheduler ;;
esac

# Last complete KV config wins in upstream argparse. D must fail external-KV
# admission instead of replaying Target; keep the connector/role from the base.
# shellcheck disable=SC2154  # _kv_role is assigned by the sourced 1p1d config.
_kv_transfer_config="{\"kv_connector\":\"TPURaidenConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"${_kv_role}\",\"kv_load_failure_policy\":\"fail\"}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS} --kv-transfer-config ${_kv_transfer_config} --scheduler-cls vllm_torchtpu.core.capacity_scheduler.${_remote_scheduler} --distributed-executor-backend vllm_torchtpu.executors.tpu_multiproc_executor.TpuMultiprocExecutor"

# Deliberate per-RPC diagnostic bounds, overriding inherited historical YAML
# values. Executor permits socket failure to report before its fatal teardown.
export VLLM_TPU_REMOTE_DSPARK_TIMEOUT_S=300
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=360
