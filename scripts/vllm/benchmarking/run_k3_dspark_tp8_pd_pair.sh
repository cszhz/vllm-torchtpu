#!/usr/bin/env bash
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

# Launch one role of the two-v7x-8 Kimi-K3 + DSpark P/D baseline. The prefill
# role drives the proxy workload and profiles both engines; the decode role
# serves until the prefill role publishes completion through the shared mount.
set -Eeuo pipefail

role="${K3_DSPARK_PD_ROLE:-}"
case "$role" in
    prefill|decode) ;;
    *) echo "K3_DSPARK_PD_ROLE must be prefill or decode" >&2; exit 2 ;;
esac
: "${K3_DSPARK_PD_COORD_DIR:?}"
: "${ARTIFACT_LOCAL_DIR:?}"

coord="$K3_DSPARK_PD_COORD_DIR"
out="$ARTIFACT_LOCAL_DIR/results"
server_out="$out/$role"
mkdir -p "$coord" "$server_out"

read_nonempty_file() {
    local path="$1"
    local value=""
    while [[ -z "$value" ]]; do
        if [[ -s "$path" ]]; then
            value="$(<"$path")" || value=""
        fi
        [[ -n "$value" ]] || sleep 2
    done
    printf '%s' "$value"
}

role_ip="$(hostname -i | awk '{print $1}')"
claim_dir="$coord/$role.claim"
if ! mkdir "$claim_dir"; then
    echo "A $role instance already claimed coordination directory $coord" >&2
    exit 1
fi
ip_tmp="$coord/$role.ip.${FALCON_EXP_ID:-$$}.tmp"
printf '%s\n' "$role_ip" >"$ip_tmp"
mv "$ip_tmp" "$coord/$role.ip"
export VLLM_HOST_IP="$role_ip"
export TPU_RAIDEN_ADVERTISE_HOST="$role_ip"
controller_port=28340
[[ "$role" == prefill ]] && controller_port=27340
export TPU_RAIDEN_CONTROLLER_ADDRESS="$role_ip:$controller_port"
export TMPDIR="${TMPDIR:-/tmp/tpu_logs/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/tpu_logs/pip-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/tpu_logs/uv-cache}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
ulimit -c 0

config="${K3_DSPARK_PD_CONFIG:-kimi-k3-pruned75-dspark-tp8-ep-1p1d}"
remote_dspark=0
if [[ "$config" == kimi-k3-pruned75-dspark-tp8-ep-remote ]]; then
    remote_dspark=1
    peer=decode
    [[ "$role" == prefill ]] || peer=prefill
    peer_ip="$(read_nonempty_file "$coord/$peer.ip")"
    export VLLM_TPU_REMOTE_DSPARK_PRODUCER_HOST
    if [[ "$role" == prefill ]]; then
        VLLM_TPU_REMOTE_DSPARK_PRODUCER_HOST="$role_ip"
    else
        VLLM_TPU_REMOTE_DSPARK_PRODUCER_HOST="$peer_ip"
    fi
fi

server_pid=""
proxy_pid=""
# shellcheck disable=SC2317  # Invoked through the EXIT trap.
cleanup() {
    status=$?
    trap - EXIT
    if [[ "$status" -ne 0 ]]; then
        echo "========== Last 200 lines of $role server.log ==========" >&2
        tail -200 "$server_out/server.log" >&2 || true
        echo "========================================================" >&2
    fi
    [[ -z "$proxy_pid" ]] || kill "$proxy_pid" 2>/dev/null || true
    [[ -z "$server_pid" ]] || kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    printf '%s\n' "$status" >"$coord/$role.exit"
    exit "$status"
}
trap cleanup EXIT

RESULTS_DIR="$server_out" \
CAPTURE_PROFILE=1 \
K3_DSPARK_PD_ROLE="$role" \
  bash scripts/vllm/benchmarking/run_benchmarks.sh \
    --config "$config" --serve-only &
server_pid=$!

deadline=$((SECONDS + 14400))
until curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; do
    kill -0 "$server_pid" 2>/dev/null || {
        echo "$role server exited before readiness" >&2
        exit 1
    }
    ((SECONDS < deadline)) || {
        echo "$role server did not become ready within four hours" >&2
        exit 1
    }
    sleep 10
done
touch "$coord/$role.ready"

peer=decode
[[ "$role" == prefill ]] || peer=prefill
until [[ -f "$coord/$peer.ready" ]]; do
    [[ ! -f "$coord/$peer.exit" ]] || {
        echo "$peer exited before readiness" >&2
        exit 1
    }
    ((SECONDS < deadline)) || {
        echo "$peer did not become ready within four hours" >&2
        exit 1
    }
    sleep 5
done

if [[ "$role" == decode ]]; then
    until [[ -f "$coord/benchmark.done" ]]; do
        [[ ! -f "$coord/prefill.exit" ]] || exit 1
        kill -0 "$server_pid" 2>/dev/null || {
            echo "decode server exited while the benchmark was running" >&2
            exit 1
        }
        sleep 5
    done
else
    decode_ip="$(read_nonempty_file "$coord/decode.ip")"
    verification_path="$out/remote-dspark-verification.jsonl"
    rm -f "$verification_path"
    concurrency="${K3_DSPARK_PD_CONCURRENCY:-8}"
    num_prompts="${K3_DSPARK_PD_NUM_PROMPTS:-8}"
    warmup_prompts="${K3_DSPARK_PD_WARMUP_PROMPTS:-$num_prompts}"
    remote_batch_size="$concurrency"
    remote_bypass_requests=0
    if ((remote_dspark && concurrency > 1)); then
        remote_batch_size=$((concurrency - 1))
        remote_bypass_requests=1
    fi
    VLLM_TPU_REMOTE_DSPARK_SHADOW="$remote_dspark" \
      VLLM_TPU_REMOTE_DSPARK_BATCH_SIZE="$remote_batch_size" \
      VLLM_TPU_REMOTE_DSPARK_BYPASS_REQUESTS="$remote_bypass_requests" \
      VLLM_TPU_REMOTE_DSPARK_VERIFICATION_PATH="$verification_path" \
      python3 examples/disagg/toy_proxy_server.py \
        --host 0.0.0.0 --port 8050 \
        --prefiller-hosts 127.0.0.1 --prefiller-ports 8000 \
        --decoder-hosts "$decode_ip" --decoder-ports 8000 \
        >"$out/proxy.log" 2>&1 &
    proxy_pid=$!
    until curl -fsS --max-time 3 http://127.0.0.1:8050/healthcheck >/dev/null 2>&1; do
        kill -0 "$proxy_pid" 2>/dev/null || {
            tail -100 "$out/proxy.log" >&2 || true
            exit 1
        }
        sleep 2
    done

    if [[ "${K3_DSPARK_PD_DIAGNOSTIC:-0}" == 1 ]]; then
        python3 - <<'PY'
import json
import os
import urllib.request

request = urllib.request.Request(
    "http://127.0.0.1:8050/v1/completions",
    data=json.dumps({
        "model": "/models/Kimi-K3-pruned75",
        "prompt": [1] * int(os.environ.get("K3_DSPARK_PD_INPUT_LEN", "32")),
        "max_tokens": int(os.environ.get("K3_DSPARK_PD_OUTPUT_LEN", "8")),
        "ignore_eos": True,
        "temperature": 0,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=600) as response:
    result = response.read()
path = os.path.join(os.environ["ARTIFACT_LOCAL_DIR"], "results", "diagnostic.json")
with open(path, "wb") as output:
    output.write(result)
PY
        touch "$coord/benchmark.done"
        exit 0
    fi

    bench=(
        vllm bench serve --backend vllm
        --model /models/Kimi-K3-pruned75
        --tokenizer /models/Kimi-K3-pruned75
        --trust-remote-code --host 127.0.0.1 --port 8050
        --dataset-name random
        --random-input-len "${K3_DSPARK_PD_INPUT_LEN:-1024}"
        --random-output-len "${K3_DSPARK_PD_OUTPUT_LEN:-256}"
        --random-range-ratio 0 --max-concurrency "$concurrency" --request-rate inf
        --percentile-metrics "ttft,tpot,itl,e2el" --ignore-eos --temperature 0
        --seed 42
    )
    if ((warmup_prompts > 0)); then
        "${bench[@]}" --num-prompts "$warmup_prompts" \
            --save-result --result-filename "$out/warmup.json"
    fi

    capture_profile="${K3_DSPARK_PD_CAPTURE_PROFILE:-1}"
    if [[ "$capture_profile" == 1 ]]; then
        curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null
        curl -fsS --max-time 3 "http://$decode_ip:8000/health" >/dev/null
        curl -fsS -X POST http://127.0.0.1:8000/start_profile >/dev/null
        curl -fsS -X POST "http://$decode_ip:8000/start_profile" >/dev/null
    fi
    "${bench[@]}" --num-prompts "$num_prompts" \
        --save-result --result-filename "$out/measurement.json"
    python3 - "$out/measurement.json" "$num_prompts" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
with open(path) as result_file:
    result = json.load(result_file)
completed = result.get("completed", result.get("successful_requests"))
if completed != expected:
    raise RuntimeError(
        f"benchmark completed {completed!r} requests; expected {expected}"
    )
PY
    if ((remote_dspark)); then
        expected_verifications=$((warmup_prompts + num_prompts))
        python3 - "$verification_path" "$expected_verifications" <<'PY'
import json
import sys
import time

path, expected = sys.argv[1], int(sys.argv[2])
deadline = time.monotonic() + 60
records = []
while time.monotonic() < deadline:
    try:
        with open(path) as verification_file:
            records = [json.loads(line) for line in verification_file if line.strip()]
    except FileNotFoundError:
        records = []
    if len(records) >= expected:
        break
    time.sleep(0.1)
if len(records) != expected:
    raise RuntimeError(
        f"remote DSpark recorded {len(records)} verifications; expected {expected}"
    )
failures = [record for record in records if record.get("status") != "complete"]
if failures:
    raise RuntimeError(f"remote DSpark output verification failed: {failures}")
PY
    fi
    if [[ "$capture_profile" == 1 ]]; then
        curl -fsS -X POST http://127.0.0.1:8000/stop_profile >/dev/null
        curl -fsS -X POST "http://$decode_ip:8000/stop_profile" >/dev/null
    fi
    curl -fsS http://127.0.0.1:8000/metrics >"$out/prefill.metrics"
    curl -fsS "http://$decode_ip:8000/metrics" >"$out/decode.metrics"
    touch "$coord/benchmark.done"
fi

if [[ "${K3_DSPARK_PD_CAPTURE_PROFILE:-1}" != 1 ]]; then
    exit 0
fi

for _ in $(seq 1 120); do
    if find "$server_out/profile" -name '*.xplane.pb' -type f -print -quit |
        grep -q .; then
        exit 0
    fi
    sleep 2
done
echo "no XProf xplane file was flushed for $role" >&2
exit 1
