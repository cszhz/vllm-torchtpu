#!/usr/bin/env bash
# Put the DP-rank-aware llm-d router in front of the running dsv41-pd P/D (../dsv41-pd-gke.yaml), next to the toy
# proxy (which keeps serving on the P pod's :8000). Re-run after P or D is re-created (Spot fallback, restart):
# the shims and the sidecar are ephemeral containers and the llm-d labels sit on the pods.
#
#   bash llmd_up.sh            # client entry afterwards: http://llmd-epp.default.svc.cluster.local:8081
#
#   P pod: dp_rank_shim.py --reuse-prompt-ids on 8110-8117 -> vLLM :8100   (one port per DP rank, rank header pinned)
#   D pod: llm-d sidecar (data-parallel mode) on 8110-8117 -> dp_rank_shim.py --decode-router on 8310-8317 -> vLLM :8300
#   router: helm release llmd (llm-d-router-standalone v1.0.0) with router-values.yaml; InferencePool ports 8110-8117
#
# Ephemeral containers cannot be removed and a container's PID 1 ignores signals from inside its namespace, so an
# older shim is killed from the node's PID namespace (node debug pod) before a new one binds the same ports.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); NS=default
IMG=${IMG:-us-central1-docker.pkg.dev/tpu-launchpad-playground/zzl-images/dsv41-pd:4b0fb058d}
SIDECAR_IMG=${SIDECAR_IMG:-ghcr.io/llm-d/llm-d-router-disagg-sidecar:v1.0.0}
CHART=${CHART:-oci://ghcr.io/llm-d/charts/llm-d-router-standalone}; CHART_VERSION=${CHART_VERSION:-v1.0.0}
GAIE_VERSION=${GAIE_VERSION:-v1.6.2}
HELM=${HELM:-$(command -v helm || echo ~/bin/helm)}

P=$(kubectl -n $NS get pods -l app=dsv41-pd,role=prefill -o json | python3 -c 'import json,sys; print(next(p["metadata"]["name"] for p in json.load(sys.stdin)["items"] if not p["metadata"].get("deletionTimestamp")))')
D=dsv41-d
P_NODE=$(kubectl -n $NS get pod $P -o jsonpath='{.spec.nodeName}'); P_IP=$(kubectl -n $NS get pod $P -o jsonpath='{.status.hostIP}')
D_NODE=$(kubectl -n $NS get pod $D -o jsonpath='{.spec.nodeName}'); D_IP=$(kubectl -n $NS get pod $D -o jsonpath='{.status.hostIP}')
echo "P=$P ($P_IP on $P_NODE)  D=$D ($D_IP on $D_NODE)"

running() { kubectl -n $NS get pod "$1" -o jsonpath='{range .status.ephemeralContainerStatuses[?(@.state.running)]}{.name}{"\n"}{end}'; }
kill_on_node() {   # node, pattern (matched against python processes only, so the debug shell itself survives)
  kubectl debug node/"$1" --image=ubuntu:24.04 --profile=sysadmin -q -- bash -c \
    "ps -eo pid=,comm=,args= | awk '\$2 ~ /^python/ && /$2/ {print \$1}' | xargs -r kill -9" >/dev/null 2>&1
  sleep 20
  kubectl -n $NS get pods --no-headers | awk "/node-debugger-$1/{print \$1}" | xargs -r kubectl -n $NS delete pod --wait=false >/dev/null
}
start_shim() {     # pod, node, base port, upstream port, extra args
  local pod=$1 node=$2 base=$3 up=$4; shift 4
  if curl -s -m 3 -o /dev/null "http://$(kubectl -n $NS get pod $pod -o jsonpath='{.status.hostIP}'):$((base + 3))/health"; then
    echo "$pod: replacing the shim on $base-$((base + 7))"; kill_on_node "$node" "--base-port $base"
  fi
  kubectl -n $NS debug "$pod" -c "dp-shim-$(date +%s)" --image="$IMG" --profile=general -q -- \
    python3 -c "$(cat "$HERE/dp_rank_shim.py")" --upstream "http://127.0.0.1:$up" --base-port "$base" --ranks 8 "$@" >/dev/null
}

echo "== 1. shims"
start_shim $P "$P_NODE" 8110 8100 --reuse-prompt-ids
start_shim $D "$D_NODE" 8310 8300 --decode-router

echo "== 2. sidecar on D (data-parallel mode)"
if ! running $D | grep -q llmd-sidecar; then
  kubectl -n $NS debug $D -c "llmd-sidecar-$(date +%s)" --image="$SIDECAR_IMG" --env=POD_IP="$D_IP" --profile=general -q --arguments-only -- \
    --port=8110 --model-server-port=8310 --data-parallel-size=8 --kv-connector=nixlv2 --secure-serving=false --zap-log-level=1 >/dev/null
fi

echo "== 3. labels"
kubectl -n $NS label pod $P llm-d.ai/guide=pd-disaggregation llm-d.ai/role=prefill --overwrite >/dev/null
kubectl -n $NS label pod $D llm-d.ai/guide=pd-disaggregation llm-d.ai/role=decode --overwrite >/dev/null

echo "== 4. router"
kubectl apply --server-side -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/$GAIE_VERSION/v1-manifests.yaml" >/dev/null
"$HELM" upgrade --install llmd "$CHART" --version "$CHART_VERSION" -n $NS -f "$HERE/router-values.yaml" --wait --timeout 5m | grep -E "STATUS|REVISION"

echo "== 5. check"
sleep 20
for ep in $P_IP:8113 $D_IP:8313 $D_IP:8113; do echo "  $ep /health -> $(curl -s -m 5 -o /dev/null -w %{http_code} http://$ep/health)"; done
E=$(kubectl -n $NS get pods -o name | grep llmd-epp | head -1)
echo "  EPP endpoints: $(kubectl -n $NS logs $E -c epp --since=60s 2>&1 | grep -oE '"endpoint":"[^"]+"' | sort -u | wc -l) (expect 16)"
echo "entry: http://llmd-epp.$NS.svc.cluster.local:8081  (pod IP $(kubectl -n $NS get pods -o wide | awk '/llmd-epp/{print $6}'):8081)"
