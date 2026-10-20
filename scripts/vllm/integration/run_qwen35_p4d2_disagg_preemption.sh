#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

PREEMPTION_EXPECT="${PREEMPTION_EXPECT:-abort}"
if [[ "${PREEMPTION_EXPECT}" != "recompute" && "${PREEMPTION_EXPECT}" != "abort" ]]; then
  echo "PREEMPTION_EXPECT must be recompute or abort: ${PREEMPTION_EXPECT}" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-${QWEN35_A3B_FP8_MODEL_PATH:-Qwen/Qwen3.5-35B-A3B-FP8}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_p4d2_disagg_preemption}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${PREEMPTION_EXPECT}_$(date +%Y%m%d_%H%M%S)_$$}"
P4D2_BIND_HOST="${P4D2_BIND_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"
VLLM_SRC="${VLLM_SRC:-${repo_root}}"
TORCHTPU_VLLM_SRC="${TORCHTPU_VLLM_SRC:-${repo_root}}"
USE_CURRENT_PY_ENV="${USE_CURRENT_PY_ENV:-1}"
PREFILL_TP="${PREFILL_TP:-1}"
PREFILL_PCP="${PREFILL_PCP:-4}"
PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE="${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE:-256}"
DECODE_TP="${DECODE_TP:-1}"
DECODE_DP="${DECODE_DP:-4}"
PREFILL_BLOCK_SIZE="${PREFILL_BLOCK_SIZE:-768}"
DECODE_BLOCK_SIZE="${DECODE_BLOCK_SIZE:-2304}"
# Qwen3.5 decode has three Mamba groups and one full-attention group. With
# MTP K=3, each initial request owns [4,4,4,1] blocks. 31 total blocks leave
# 30 after the null block, so two requests consume 26. The pressure prompt is
# placed near the next boundary; the metric assertion below proves whether the
# observed asynchronous schedule actually preempted a request.
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-31}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
PREFILL_COMPILE_SIZES="${PREFILL_COMPILE_SIZES:-4096}"
# Default recomputation reschedules the full decode-side prompt after a
# preemption, so the decode runner needs both its normal and prefill buckets.
DECODE_COMPILE_SIZES="${DECODE_COMPILE_SIZES:-256,4096}"
default_prefill_speculative_config='{"method":"mtp","num_speculative_tokens":1}'
default_decode_speculative_config='{"method":"mtp","num_speculative_tokens":3}'
PREFILL_SPECULATIVE_CONFIG="${PREFILL_SPECULATIVE_CONFIG-${default_prefill_speculative_config}}"
DECODE_SPECULATIVE_CONFIG="${DECODE_SPECULATIVE_CONFIG-${default_decode_speculative_config}}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
ENABLE_PREFIX_CACHING=1
TPU_RAIDEN_PREFIX_AWARE_LOAD="${TPU_RAIDEN_PREFIX_AWARE_LOAD:-1}"
RESTART_EXISTING="${RESTART_EXISTING:-0}"
owner_marker="${RUN_DIR}/.capacity_preemption_runner_owner"

if ((NUM_GPU_BLOCKS_OVERRIDE != 31)); then
  echo "NUM_GPU_BLOCKS_OVERRIDE must be 31 for the verified Qwen3.5 MTP3 capacity geometry" >&2
  exit 2
fi
if [[ "${PREFILL_SPECULATIVE_CONFIG}" != "${default_prefill_speculative_config}" \
   || "${DECODE_SPECULATIVE_CONFIG}" != "${default_decode_speculative_config}" ]]; then
  echo "This capacity geometry requires Qwen3.5 MTP K1 on prefill and K3 on decode" >&2
  exit 2
fi
if [[ "${PREFILL_TP},${PREFILL_PCP},${DECODE_TP},${DECODE_DP}" != "1,4,1,4" ]]; then
  echo "This capacity test requires the validated P4D4 topology" >&2
  exit 2
fi
if ((PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE != 256)); then
  echo "PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE must be 256 for the validated PCP4 kernel" >&2
  exit 2
fi
if ((MAX_NUM_BATCHED_TOKENS != 4096)); then
  echo "MAX_NUM_BATCHED_TOKENS must be 4096 for the fixed pressure configuration" >&2
  exit 2
fi
if [[ "${PREFILL_COMPILE_SIZES},${DECODE_COMPILE_SIZES}" != "4096,256,4096" ]]; then
  echo "This pressure configuration requires P compile 4096 and D compile 256,4096" >&2
  exit 2
fi

cleanup() {
  local pid_file pid
  if [[ ! -f "${owner_marker}" || "$(<"${owner_marker}")" != "$$" ]]; then
    return
  fi
  for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
    [[ -f "${pid_file}" ]] || continue
    pid="$(<"${pid_file}")"
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 2
  for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
    [[ -f "${pid_file}" ]] || continue
    pid="$(<"${pid_file}")"
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    if kill -0 -- "-${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
    elif kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
  rm -f "${owner_marker}"
}

mkdir -p "${RUN_ROOT}" "${RUN_DIR}"
for artifact in \
  launch_params.txt prefill.pid decode.pid proxy.pid role_wrapper.log \
  capacity_preemption_result.json logs/launch.log logs/prefill.log logs/decode.log; do
  if [[ -e "${RUN_DIR}/${artifact}" ]]; then
    echo "RUN_DIR contains an old service artifact: ${RUN_DIR}/${artifact}" >&2
    exit 2
  fi
done
if ! (set -o noclobber; printf '%s\n' "$$" >"${owner_marker}"); then
  echo "RUN_DIR is already owned by another preemption runner: ${RUN_DIR}" >&2
  exit 2
fi
trap cleanup EXIT

mkdir -p "${RUN_DIR}/logs"
EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-$(python -c 'import importlib.metadata as m; print(m.version("vllm"))')}"

export MODEL_PATH SERVED_MODEL_NAME RUN_ROOT RUN_DIR P4D2_BIND_HOST PREEMPTION_EXPECT
export PREFILL_PORT DECODE_PORT PROXY_PORT VLLM_SRC TORCHTPU_VLLM_SRC
export USE_CURRENT_PY_ENV EXPECTED_VLLM_VERSION PREFILL_TP PREFILL_PCP
export PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE
export DECODE_TP DECODE_DP PREFILL_BLOCK_SIZE DECODE_BLOCK_SIZE
export NUM_GPU_BLOCKS_OVERRIDE MAX_MODEL_LEN MAX_NUM_SEQS
export MAX_NUM_BATCHED_TOKENS
export PREFILL_COMPILE_SIZES DECODE_COMPILE_SIZES
export PREFILL_SPECULATIVE_CONFIG DECODE_SPECULATIVE_CONFIG
export ASYNC_SCHEDULING ENABLE_PREFIX_CACHING TPU_RAIDEN_PREFIX_AWARE_LOAD
export RESTART_EXISTING

export PD_CAPACITY_PREEMPTION_INNER_WRAPPER="${VLLM_DISAGG_ROLE_WRAPPER:-}"
export VLLM_DISAGG_ROLE_WRAPPER="${script_dir}/with_pd_abort_on_preemption.sh"

cd "${repo_root}"
bash examples/disagg/launch_qwen35_p4d2_v2_baseline.sh \
  2>&1 | tee "${RUN_DIR}/logs/launch.log"

grep -Fqx "NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE}" "${RUN_DIR}/launch_params.txt"
grep -Fqx "ASYNC_SCHEDULING=${ASYNC_SCHEDULING}" "${RUN_DIR}/launch_params.txt"
grep -Fqx "TPU_RAIDEN_PREFIX_AWARE_LOAD=${TPU_RAIDEN_PREFIX_AWARE_LOAD}" \
  "${RUN_DIR}/launch_params.txt"
grep -Fqx "PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE=${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE}" \
  "${RUN_DIR}/launch_params.txt"
grep -Fqx "DECODE_SPECULATIVE_CONFIG=${DECODE_SPECULATIVE_CONFIG}" "${RUN_DIR}/launch_params.txt"
expected_load_failure_policy=recompute
if [[ "${PREEMPTION_EXPECT}" == "abort" ]]; then
  expected_load_failure_policy=fail
fi
grep -Fqx "role=decode kv_load_failure_policy=${expected_load_failure_policy}" \
  "${RUN_DIR}/role_wrapper.log"
if grep -Fq -- '--scheduler-cls' "${RUN_DIR}/prefill_vllm_args.quoted" \
  "${RUN_DIR}/decode_vllm_args.quoted"; then
  echo "Capacity test must run the upstream scheduler selected by async_scheduling" >&2
  exit 1
fi
if grep -Fq 'raiden_pd_send_completion' "${RUN_DIR}/prefill_vllm_args.quoted"; then
  echo "Capacity test must leave producer send completion disabled" >&2
  exit 1
fi
python scripts/vllm/integration/with_pd_capacity_preemption.py \
  --host "${P4D2_BIND_HOST}" \
  --prefill-port "${PREFILL_PORT}" \
  --decode-port "${DECODE_PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --expect "${PREEMPTION_EXPECT}" \
  --decode-block-size "${DECODE_BLOCK_SIZE}" \
  --num-gpu-blocks "${NUM_GPU_BLOCKS_OVERRIDE}" \
  --decode-log "${RUN_DIR}/logs/decode.log" \
  --result-path "${RUN_DIR}/capacity_preemption_result.json" \
  2>&1 | tee "${RUN_DIR}/logs/capacity_preemption.log"

echo "PD_CAPACITY_PREEMPTION_E2E_OK expect=${PREEMPTION_EXPECT} async=${ASYNC_SCHEDULING} mtp=${DECODE_SPECULATIVE_CONFIG}"
