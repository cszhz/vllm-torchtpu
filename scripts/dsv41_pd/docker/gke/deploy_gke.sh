#!/usr/bin/env bash
# dsv41-pd 1P1D on GKE: apply, wait for P / D / proxy, smoke, PD logprob check.
#   bash deploy_gke.sh               # LOGPROB=0 skips the logprob check
# Before applying: the D node's placeholder Job must tolerate preemption (backoffLimit; each preempted pod counts
# as a failure) and the DWS node is protected from scale-down while the placeholder is Pending.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); NS=default
D_NODE=$(kubectl get nodes -l cloud.google.com/gke-nodepool=zzl-np-tpuv7-4q -o jsonpath='{.items[0].metadata.name}')
b=$(kubectl -n $NS get job zzl-tpuv7-4q-job -o jsonpath='{.spec.backoffLimit}' 2>/dev/null)
[[ -n "$b" && "$b" -lt 1000 ]] && kubectl -n $NS patch job zzl-tpuv7-4q-job --type=merge -p '{"spec":{"backoffLimit":1000}}'
kubectl annotate node "$D_NODE" cluster-autoscaler.kubernetes.io/scale-down-disabled=true --overwrite >/dev/null
kubectl apply -f "$HERE/priorityclass.yaml" -f "$HERE/dsv41-pd-gke.yaml" || exit 1

# P is a Deployment (pod name changes when it is re-created after a Spot preemption); D is a plain pod.
pod_of() { case "$1" in dsv41-p) kubectl -n $NS get pods -l app=dsv41-pd,role=prefill -o json 2>/dev/null | python3 -c 'import json,sys; [print(p["metadata"]["name"]) for p in json.load(sys.stdin)["items"] if not p["metadata"].get("deletionTimestamp")]' | head -1;; *) echo "$1";; esac; }
ready() { [[ "$(kubectl -n $NS get pod "$(pod_of $1)" -o jsonpath="{.status.containerStatuses[?(@.name==\"$2\")].ready}" 2>/dev/null)" == true ]]; }
phase() { kubectl -n $NS get pod "$(pod_of $1)" -o jsonpath='{.status.phase}' 2>/dev/null; }
echo "== waiting for P (~50 min cold) and D (~90 min cold); logs: kubectl logs -f dsv41-p -c prefill / dsv41-d"
for i in $(seq 900); do
  ready dsv41-p prefill && ready dsv41-d decode && ready dsv41-p proxy && { echo "READY $(date +%T)"; break; }
  for p in dsv41-p dsv41-d; do
    case "$(phase $p)" in Failed|Succeeded) echo "ERROR: $p exited"; kubectl -n $NS logs "$(pod_of $p)" --all-containers --tail=30 | cut -c1-250; exit 1;; esac
  done
  (( i % 30 == 1 )) && echo "$(date +%T) P=$(ready dsv41-p prefill && echo ready || phase dsv41-p) D=$(ready dsv41-d decode && echo ready || phase dsv41-d) | P: $(kubectl -n $NS logs "$(pod_of dsv41-p)" -c prefill --tail=1 2>/dev/null | cut -c1-110)"
  sleep 10
done
P_IP=$(kubectl -n $NS get pod "$(pod_of dsv41-p)" -o jsonpath='{.status.hostIP}'); D_IP=$(kubectl -n $NS get pod dsv41-d -o jsonpath='{.status.hostIP}')
echo "P=$P_IP:8100 D=$D_IP:8300 proxy=$P_IP:8000 (service dsv41-proxy:8000)"
echo "== smoke"
curl -s -m 120 "http://$P_IP:8000/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41","prompt":"The capital of France is","max_tokens":12,"temperature":0}' | head -c 400; echo
if [[ "${LOGPROB:-1}" == 1 ]]; then
  echo "== PD logprob check (3k-300k), as REPRODUCE-docker.zh.md section 5"
  kubectl -n $NS exec "$(pod_of dsv41-p)" -c proxy -- bash -c "mkdir -p /zzlfs/temu && ln -sfn /opt/vllm-torchtpu /zzlfs/temu/vllm-torchtpu &&
    python -I /opt/vllm-torchtpu/scripts/dsv41_pd/pd_logprob_check.py --proxy http://localhost:8000 --direct http://$D_IP:8300 \
      --tokens 3000 20000 100000 300000 --offset 1000000" 2>&1 | tail -10 | cut -c1-200
fi
