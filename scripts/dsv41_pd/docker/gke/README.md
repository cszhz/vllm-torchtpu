# dsv41-pd 1P1D on GKE（zzl-tpu-gke），toy proxy 与 llm-d 两种入口

把 `../` 的 dsv41-pd 镜像（`4b0fb058d`）部署到 GKE 集群 `zzl-tpu-gke`（us-central1），P、D 各占一台 4 芯片 v7x。
入口有两个，指向同一套 P/D：镜像自带的 toy proxy（P pod 的 `:8000`），以及按 DP rank 路由的 llm-d（`llmd/`）。

## 文件

| 文件 | 作用 |
|---|---|
| `dsv41-pd-gke.yaml` | P（Deployment，含 proxy 容器）、D（Pod）和 `dsv41-p` / `dsv41-d` / `dsv41-proxy` 三个 Service |
| `dsv41-pd-fallback.yaml` | P 的 2x2x2 备用 Deployment（平时 0 副本）和自动切换的 watcher（`dsv41-p-failover`） |
| `priorityclass.yaml` | `zzl-tpu-workload`（1000，抢占占位 pod）、`zzl-tpu-workload-high`（1100，备用 P 抢占 1000 的负载） |
| `deploy_gke.sh` | apply → 等 P/D/proxy 就绪 → 冒烟 → PD logprob 检查（3k–300k） |
| `llmd/llmd_up.sh` | 在运行中的 P/D 上接 llm-d：转发层、sidecar、标签、router（P 或 D 重建后要重跑） |
| `llmd/dp_rank_shim.py` | 每个 DP rank 一个端口的转发层；`--reuse-prompt-ids`（P）、`--decode-router`（D） |
| `llmd/router-values.yaml` | llm-d router 的 helm values（InferencePool 8 个端口，EPP 插件链） |
| `llmd/llmd_agentx.sh` | 在 `dsv41-bench` pod 里跑 AgentX（同镜像 `bench` 角色的参数），每 30 s 抓 P/D 的 `/metrics` |
| `llmd/compare_chain.sh` | llm-d、toy proxy 背靠背各跑一轮 |

## 放置

| 角色 | 节点池 | 说明 |
|---|---|---|
| P + proxy | `v7x4`（2x2x1，**Spot**） | 被回收后由 watcher 切到 `zzl-np-tpuv7-8`（2x2x2，DWS）的一台主机，单机模式运行；不自动切回 |
| D | `zzl-np-tpuv7-4q`（2x2x1，DWS） | 用优先级抢占占位 pod `zzl-tpuv7-4q-job` |

- 镜像推在 `us-central1-docker.pkg.dev/tpu-launchpad-playground/zzl-images/dsv41-pd:4b0fb058d`。
- P 的编译缓存放 NFS（`/zzlfs/temu/gke-dsv41-cache/p`），换节点后有缓存约 20 分钟可用；D 用节点本地盘。
- 2x2x2 上的备用 P 会抢占同优先级以下的负载（例如 GLM 的两机 Job，它 `backoffLimit: 0`，会整体失败）。

```bash
kubectl apply -f priorityclass.yaml
bash deploy_gke.sh                         # P/D 冷启动约 50 / 90 分钟
kubectl apply -f dsv41-pd-fallback.yaml    # Spot 回收时自动切到 2x2x2
bash llmd/llmd_up.sh                       # 可选：llm-d 入口 http://llmd-epp.default.svc.cluster.local:8081
# 手动切回 v7x4：kubectl scale deploy/dsv41-p-fallback --replicas=0 && kubectl scale deploy/dsv41-p --replicas=1
```

## docker run → pod

| docker | pod spec |
|---|---|
| `--privileged` | `securityContext.privileged`（`/dev/vfio`） |
| `--net=host` | `hostNetwork`（启动脚本用 `hostname -I` 取 IP；Raiden 端口） |
| `--ipc=host` | `hostIPC`（`/dev/shm`：Engram 共享表约 190 GB） |
| `-v …:/cache` | P：NFS；D：hostPath `/var/tmp/dsv41-cache` |
| `--ulimit memlock/nofile` | 这些节点上 containerd 默认已是 unlimited / 1048576 |

## 踩过的坑

- **D 不能用 :8200**：GKE 节点的 `otelsvc` 在主机网络上占着 `127.0.0.1:8200-8203`，hostNetwork 的 D 报 `Address already in use`。D 用 `EXTRA_ARGS=--port 8300`（追加在 `vllm serve` 末尾，覆盖脚本里的 8200），proxy 用 `DECODE_PORT=8300`。
- **GKE Warden 每个 TPU pod 只允许一个 `gke-tpu-topology` 值**，所以 2x2x1 和 2x2x2 不能写进同一个 Deployment，备用 P 单独一个 Deployment。
- **占位 Job 的 `backoffLimit`**：被抢占的 pod 计入失败次数，`zzl-tpuv7-8-job` 原来 6 次且已失败 6 次，再抢占一次整个 Job 就 Failed、不再重建占位 pod。抢占前调到 1000（`deploy_gke.sh` 对 4q 的 Job 会做）。
- **占位 pod 重建有退避**（连续失败后最长约 6–10 分钟），期间节点是空的，给 DWS 节点加了 `cluster-autoscaler.kubernetes.io/scale-down-disabled=true`。
- **ephemeral 容器里的 PID 1 收不到容器内发的信号**（SIGTERM 被忽略、SIGKILL 不生效）：换转发层要从节点的 PID 命名空间里 `kill -9`（`llmd_up.sh` 用 node debug pod 做）。
- 删 Pod/Job 后旧 pod 还在 Terminating 时别把它当成新 pod（会误判就绪）。

## llm-d：结构

```
client → Envoy :8081 → EPP（16 个端点：P 的 rank 0-7 + D 的 rank 0-7）
  → D 节点 llm-d sidecar :8110+i（DP 模式，每个 rank 一个监听端口）
      ├─ 先 P 节点转发层 :8110+j → X-data-parallel-rank: j → P vLLM :8100
      └─ 再 D 节点转发层 :8310+i → toy 的 D 路由选 rank → D vLLM :8300
```

vLLM 的 DP8 只有一个 API 端口，靠 `X-data-parallel-rank` 指定 rank；llm-d 的 DP 支持要求每个 rank 一个端口。
转发层把每个端口固定到一个 rank，`/metrics` 只返回该 rank（`engine="i"`）的指标，于是 EPP 能按 rank 做前缀亲和。

和 toy proxy 对齐用到的改动：

| 改动 | 对应 toy proxy | 解决的问题 |
|---|---|---|
| 转发层：每个 DP rank 一个端口 | 前缀路由按 rank 记账 | 不按 rank 时 P 命中 26%，1.84 req/s |
| EPP prefill：`affinityThreshold` 0.80 → 0.05，前缀 10 : 负载 1 | 最长前缀优先 | 16% 的轮次（命中比例 < 80%）被发到最闲的 rank |
| EPP prefill：关 TTFT 负载闸门（`maxTTFTPenaltyMs: 0`） | `PROXY_PREFIX_MIGRATE_FACTOR=0` | 闸门放弃亲和占 6% |
| D 转发层 `--decode-router`（移植 `_DecodeRouter`，slack 2） | `PROXY_DECODE_ROUTING=1` | EPP 的加权打分表达不了"差超过 2 才换"：前缀权重高则 D 负载不均（16.1/6.1），低则 D 命中掉到 56% |
| P 转发层 `--reuse-prompt-ids` | `PROXY_REUSE_PROMPT_IDS=1` | D 每个请求重新套模板、分词（约 0.35 s / 13 万 token） |
| token id 以 base64 int32 穿过 sidecar | — | Go `encoding/json` 处理 10^5 个整数的数组慢，TTFT 差距随请求变大 |
| 转发层连接：上游空闲 3 s 丢弃、断连重试一次、对 sidecar 保活 120 s | — | 500 错误（vLLM 5 s keep-alive、Go sidecar 90 s 空闲连接） |

## 结果（AgentX c512 3600 s，同一套 P/D，P 在 2x2x2 主机上，2026-11-26）

| 入口 | req/s | 输出 tok/s | TTFT 平均 / P50 / P90 / P99 (s) | ITL 平均 (ms) | P / D 命中 | D rank 负载 最多/最少 |
|---|---|---|---|---|---|---|
| toy proxy | 8.98 | 9,375 | 2.06 / 1.12 / 2.09 / 17.5 | 10.4 | 94.2% / 76.4% | 13.0 / 9.7 |
| toy proxy（重跑） | 8.98 | 9,386 | 2.16 / 1.11 / 2.12 / 20.0 | 10.4 | 94.2% / 76.3% | 13.0 / 9.6 |
| llm-d，不按 rank | 1.84 | 1,867 | — / 132 / 212 / — | 6.5 | 26% / — | — |
| llm-d，按 rank，指南默认插件链 | 7.95 | 8,159 | — / 1.49 / 7.71 / — | 11.2 | 86.7% / 70.2% | 16.1 / 6.1 |
| + prefill 对齐、D 加权 1:2、复用 ids | 8.81 | 9,218 | — / 1.28 / 2.48 / — | 11.4 | 94.2% / 55.8% | 14.8 / 9.2 |
| + D 转发层用 toy 的 D 路由 | 8.98 | 9,394 | 2.11 / 1.22 / 2.33 / 17.8 | 10.4 | 94.3% / 76.1% | 12.8 / 9.4 |
| **+ ids 打包（当前）** | **8.96** | **9,376** | **2.07 / 1.18 / 2.25 / 18.5** | **10.4** | **94.2% / 76.2%** | **12.8 / 9.4** |

镜像在 VM 上（toy proxy）为 8.98 req/s，与 GKE 一致。

现状：吞吐、ITL、两侧命中率、D 负载分布、TTFT 平均 / P95 / P99 与 toy proxy 一致；TTFT P50 / P90 还高约
0.065 / 0.14 s（6–7%），随请求变大而增加。P、D 服务端看到的排队和处理时间两边相同，差在 llm-d 多出的几跳：
Envoy 要把整个请求体交给 EPP 解析后才转发（前缀亲和离不开），sidecar 用 Go 标准库解析一次、编码两次请求体，
外加两个本地转发层。

注意：吞吐压测中不要抓 xprof；aiperf 经 llm-d 抓的 `/metrics` 只是某一个 rank 的（EPP 日志里的
`parsing response: invalid character '#'` 就是它），P/D 的完整指标用 `llmd_agentx.sh` 直接抓。
