#!/usr/bin/env bash
# DeepSeek-V4.1-Flash 1P1D (vllm-torchtpu, Stage-3 Raiden) container entrypoint.
#
#   docker run ... dsv41-pd prefill          # P on :8100
#   docker run ... dsv41-pd decode           # D on :8200
#   docker run ... -e PREFILL_HOST=.. -e DECODE_HOST=.. dsv41-pd proxy    # :8000
#   docker run ... -e PROXY_URL=http://..:8000 dsv41-pd bench [extra aiperf args]
#   docker run ... dsv41-pd bash             # anything else runs as a command
#
# Defaults are the configuration validated on AgentX c512 3600s (9.02 req/s,
# ITL 10.1 ms); every one can be overridden with the same environment variable.
set -euo pipefail

export VENV=/opt/venv-epic CODE_DIR=/opt/vllm-torchtpu
# start_1p1d_epic.sh and the proxy read the tokenizer from this fixed path.
[[ -e /tmp/dsv41-tokenizer ]] || ln -s /opt/dsv41-tokenizer /tmp/dsv41-tokenizer
# Compile caches are 20-45 GB per configuration: mount a host directory at
# /cache (e.g. -v /var/tmp/dsv41-cache:/cache) so restarts reuse them.
export TMPDIR="${TMPDIR:-/cache/tmp}" VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/cache/vllm}"
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT"

role="${1:-}"
[[ $# -gt 0 ]] && shift
case "$role" in
  prefill)
    export TPU_DSPARK_SAMPLED_DRAFT="${TPU_DSPARK_SAMPLED_DRAFT:-1}"
    # Whale prefills (> 16k tokens left) share 1024 tokens/step while the rank
    # has other work; the per-step attention-work cap stays off.
    export TPU_P_WHALE_TOKENS="${TPU_P_WHALE_TOKENS:-16384}"
    export TPU_P_WHALE_STEP_TOKENS="${TPU_P_WHALE_STEP_TOKENS:-1024}"
    export TPU_DP_ATTN_STEP_CAP="${TPU_DP_ATTN_STEP_CAP:-0}"
    export EXTRA_ARGS="--scheduler-cls vllm_torchtpu.core.dp_attn_balance_scheduler.TpuDpAttnBalanceScheduler ${EXTRA_ARGS:-}"
    exec bash "$CODE_DIR/scripts/dsv41_pd/start_1p1d_epic.sh" prefill
    ;;
  decode)
    export TPU_DSPARK_SAMPLED_DRAFT="${TPU_DSPARK_SAMPLED_DRAFT:-1}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
    export DECODE_COMPILE_SIZES="${DECODE_COMPILE_SIZES:-[16,48,96,144,192,288,384,2048]}"
    # The prefill-tuned MoE input pipeline (#2972) is slower on decode shapes.
    export TPU_MOE_V2_INPUT_PIPELINE="${TPU_MOE_V2_INPUT_PIPELINE:-off}"
    export TPU_ENGRAM_GATHER_THREADS="${TPU_ENGRAM_GATHER_THREADS:-6}"
    exec bash "$CODE_DIR/scripts/dsv41_pd/start_1p1d_epic.sh" decode
    ;;
  proxy)
    : "${PREFILL_HOST:?set PREFILL_HOST}" "${DECODE_HOST:?set DECODE_HOST}"
    export VLLM_PLUGINS="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1
    export PROXY_TOKENIZER="${PROXY_TOKENIZER:-/tmp/dsv41-tokenizer}"
    export PROXY_SESSION_AFFINITY_DP="${PROXY_SESSION_AFFINITY_DP:-8}"
    export PROXY_DECODE_ROUTING="${PROXY_DECODE_ROUTING:-1}"
    exec "$VENV/bin/python" "$CODE_DIR/scripts/dsv41_pd/toy_proxy_server.py" \
      --host 0.0.0.0 --port "${PORT:-8000}" \
      --prefiller-host "$PREFILL_HOST" --prefiller-port "${PREFILL_PORT:-8100}" \
      --decoder-host "$DECODE_HOST" --decoder-port "${DECODE_PORT:-8200}" "$@"
    ;;
  bench)
    # InferenceX AgentX, same flags as the customer's command. The dataset is
    # downloaded on first use into $HF_HOME.
    : "${PROXY_URL:?set PROXY_URL, e.g. http://<proxy>:8000}"
    export HF_HOME="${HF_HOME:-/cache/hf}" AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 \
      AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
    out="${OUT_DIR:-/cache/results/agentx-c${CONC:-512}-$(date +%m%d-%H%M)}"
    mkdir -p "$out" && cd "$out"
    exec /opt/venv-aiperf/bin/aiperf profile \
      --scenario inferencex-agentx-mvp --url "$PROXY_URL" --model deepseek-v41 \
      --tokenizer /tmp/dsv41-tokenizer \
      --public-dataset semianalysis_cc_traces_weka_062126 --endpoint-type chat \
      --concurrency "${CONC:-512}" --benchmark-duration "${DURATION:-3600}" \
      --random-seed 2026110501 --use-server-token-count --streaming \
      --extra-inputs ignore_eos:true --cache-bust first_turn_prefix \
      --system-idle-gap-cap-seconds 10 --trajectory-start-min-ratio 0.0 \
      --trajectory-start-max-ratio 1.0 --ui simple --output-artifact-dir "$out/artifacts" "$@"
    ;;
  "" )
    echo "usage: prefill | decode | proxy | bench | <command>" >&2; exit 2 ;;
  *)
    exec "$role" "$@" ;;
esac
