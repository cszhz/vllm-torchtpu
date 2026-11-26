#!/usr/bin/env bash
# AgentX through the llm-d router (Envoy :8081), same aiperf flags as the dsv41-pd image's `bench` role
# (VM result from the image: 8.98 req/s at c512 3600 s). Runs inside dsv41-bench; scrapes P :8100 and D :8300.
#   P_IP=.. D_IP=.. CONC=512 DURATION=3600 bash llmd_agentx.sh
set -uo pipefail
URL=${URL:-http://llmd-epp.default.svc.cluster.local:8081}
CONC=${CONC:-512}; DURATION=${DURATION:-3600}; T=$(date +%m%d-%H%M)
OUT=${OUT:-/zzlfs/temu/results/agentx-c$CONC-gke-llmd-dp-$T}; mkdir -p "$OUT/metrics"
echo "url=$URL p=$P_IP:8100 d=$D_IP:8300 conc=$CONC duration=$DURATION" > "$OUT/config.txt"
export HF_HOME=/zzlfs/temu/hf-cache AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
( while true; do ts=$(date +%s)
    curl -s --max-time 10 "$P_IP:8100/metrics" > "$OUT/metrics/p-$ts.prom"
    curl -s --max-time 10 "$D_IP:8300/metrics" > "$OUT/metrics/d-$ts.prom"
    sleep 30; done ) & SCR=$!; trap 'kill $SCR 2>/dev/null' EXIT
echo "aiperf c$CONC ${DURATION}s -> $OUT"
cd "$OUT" && /zzlfs/temu/.venv-aiperf/bin/aiperf profile --scenario inferencex-agentx-mvp --url "$URL" --model deepseek-v41 \
  --tokenizer /zzlfs/temu/tools/dsv41-tokenizer --public-dataset semianalysis_cc_traces_weka_062126 --endpoint-type chat \
  --concurrency "$CONC" --benchmark-duration "$DURATION" --random-seed 2026110501 --use-server-token-count --streaming \
  --extra-inputs ignore_eos:true --cache-bust first_turn_prefix --system-idle-gap-cap-seconds 10 \
  --trajectory-start-min-ratio 0.0 --trajectory-start-max-ratio 1.0 --ui simple --output-artifact-dir "$OUT/artifacts" > "$OUT/aiperf.log" 2>&1
echo "aiperf rc=$? out=$OUT"
