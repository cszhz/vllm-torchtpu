#!/usr/bin/env bash
# End-to-end test of the dsv41-pd image, run on the P host after P and D are
# started from the image (README.md):
#   1. wait for P :8100 and D :8200 health
#   2. (re)start the proxy container on :8000
#   3. smoke completion
#   4. PD logprob check: P->D vs D-only prefill, 3k-300k tokens, with and
#      without a D prefix hit (first tokens must match, max|dlogprob| < ~1)
#   5. optional AgentX benchmark (BENCH=1; CONC / DURATION)
#
#   P_IP=10.128.0.62 D_IP=10.128.0.39 bash test_image.sh
#   P_IP=... D_IP=... BENCH=1 CONC=512 DURATION=3600 bash test_image.sh
set -uo pipefail
: "${P_IP:?set P_IP}" "${D_IP:?set D_IP}"
IMAGE=${IMAGE:-dsv41-pd:latest}
CACHE=${CACHE:-/var/tmp/dsv41-cache}
DOCKER=${DOCKER:-sudo docker}
# docker's default 1024 open files is too few for the proxy / aiperf at c512.
U="--ulimit nofile=1048576:1048576"
T=$(date +%m%d-%H%M)

echo "== wait for P/D health"
for i in $(seq 1 720); do
  pc=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://$P_IP:8100/health")
  dc=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://$D_IP:8200/health")
  [[ $pc == 200 && $dc == 200 ]] && { echo "ready after $((i * 10)) s"; break; }
  (( i == 720 )) && { echo "P=$pc D=$dc: not ready after 2 h"; exit 1; }
  sleep 10
done

echo "== proxy"
$DOCKER rm -f dsv41-proxy >/dev/null 2>&1
$DOCKER run --rm -d $U --net=host --name dsv41-proxy \
  -e PREFILL_HOST="$P_IP" -e DECODE_HOST="$D_IP" "$IMAGE" proxy >/dev/null
for i in $(seq 60); do curl -s -o /dev/null -m 2 "http://$P_IP:8000/" && break; sleep 2; done

echo "== smoke"
out=$(curl -s -m 300 "http://$P_IP:8000/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41","prompt":"The capital of France is","max_tokens":12,"temperature":0}')
echo "$out" | head -c 300; echo
grep -q '"text"' <<<"$out" || { echo "smoke FAILED"; exit 1; }

echo "== PD logprob check"
# pd_logprob_check.py reads its corpus from /zzlfs/temu/vllm-torchtpu/**/*.py.
mkdir -p "$CACHE/results"
$DOCKER run --rm --net=host "$IMAGE" bash -c "
  mkdir -p /zzlfs/temu && ln -s /opt/vllm-torchtpu /zzlfs/temu/vllm-torchtpu &&
  python -I /opt/vllm-torchtpu/scripts/dsv41_pd/pd_logprob_check.py \
    --proxy http://$P_IP:8000 --direct http://$D_IP:8200 \
    --tokens ${LOGPROB_TOKENS:-3000 20000 100000 300000} --offset 1000000" 2>&1 \
  | sudo tee "$CACHE/results/logprob-$T.log"
lines=$(grep -c "max|dlogprob|" "$CACHE/results/logprob-$T.log")
bad=$(awk '/max\|dlogprob\|/ {for (i = 1; i <= NF; i++) if ($i == "max|dlogprob|") v = $(i + 1);
           if (v == "nan" || v == "inf" || v + 0 > 2) n++} END {print n + 0}' "$CACHE/results/logprob-$T.log")
echo "logprob: $lines rows, $bad suspicious (nan/inf or > 2)"
[[ $lines -ge 2 && $bad == 0 ]] || { echo "logprob check FAILED"; exit 1; }

if [[ "${BENCH:-0}" == 1 ]]; then
  echo "== AgentX c${CONC:-512} ${DURATION:-3600}s"
  curl -s -X POST "http://$P_IP:8100/reset_prefix_cache"; curl -s -X POST "http://$D_IP:8200/reset_prefix_cache"; echo
  $DOCKER run --rm $U --net=host -v "$CACHE:/cache" -e PROXY_URL="http://$P_IP:8000" \
    -e CONC="${CONC:-512}" -e DURATION="${DURATION:-3600}" \
    -e OUT_DIR="/cache/results/agentx-c${CONC:-512}-$T" "$IMAGE" bench 2>&1 \
    | grep -E "Request Throughput|Output Token Throughput \(|Inter Token Latency|Time to First Token|Error Rate|Benchmark Duration|Exit Errors|Reason"
  echo "results: $CACHE/results/agentx-c${CONC:-512}-$T"
fi
echo "== PASS"
