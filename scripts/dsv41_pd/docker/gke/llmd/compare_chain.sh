#!/usr/bin/env bash
# llm-d then toy proxy, back to back, same P/D, same aiperf flags (runs inside dsv41-bench).
set -uo pipefail
L=$(cd "$(dirname "$0")" && pwd)
drain() { for _ in $(seq 60); do n=0; for ep in $P_IP:8100 $D_IP:8300; do n=$((n + $(curl -s -m 5 http://$ep/metrics | awk '/^vllm:num_requests_(running|waiting)\{/{s+=$2} END{print int(s)}'))); done; [ "$n" = 0 ] && return; sleep 10; done; }
T=$(date +%m%d-%H%M)
OUT=/zzlfs/temu/results/agentx-c512-gke-llmd-v5-$T URL=http://llmd-epp.default.svc.cluster.local:8081 bash $L/llmd_agentx.sh
echo "router=router-values.yaml + D shim --decode-router + P shim --reuse-prompt-ids (top-level ids dropped)" >> /zzlfs/temu/results/agentx-c512-gke-llmd-v5-$T/config.txt
drain
T2=$(date +%m%d-%H%M)
OUT=/zzlfs/temu/results/agentx-c512-gke-toy-rerun-$T2 URL=http://$P_IP:8000 bash $L/llmd_agentx.sh
echo "entry=toy proxy (rerun for run-to-run variance)" >> /zzlfs/temu/results/agentx-c512-gke-toy-rerun-$T2/config.txt
echo "CHAIN DONE"
