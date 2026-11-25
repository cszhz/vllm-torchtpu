# dsv41-pd 镜像（DeepSeek-V4.1-Flash 1P1D，vllm-torchtpu Stage-3 Raiden）

镜像里包含模型权重以外的全部内容：推理用的 venv（已经打好运行时补丁、换好自编的 tpu-sync `.so`）、
vllm-torchtpu 代码（`cszhz/dsv41-1p1d` 4b0fb058d）、toy proxy、aiperf 压测客户端、tokenizer。
权重在启动时从 `gs://zzl-bucket/models/DeepSeek-V4.1-Flash` 直接流式读取（runai streamer），宿主机需要能访问 GCS。

## 加载

```bash
docker load -i /zzlfs/temu/docker/dsv41-pd-4b0fb058d.tar.gz     # 得到 dsv41-pd:4b0fb058d
```

## 启动（P、D 各占一台 4 芯片 v7x，同一个镜像）

```bash
COMMON="--rm -d --privileged --net=host --ipc=host --ulimit memlock=-1 --ulimit nofile=1048576:1048576 -v /var/tmp/dsv41-cache:/cache"
# P（:8100）
docker run $COMMON --name dsv41-p dsv41-pd:4b0fb058d prefill
# D（:8200）
docker run $COMMON --name dsv41-d dsv41-pd:4b0fb058d decode
# proxy（:8000，通常和 P 在同一台）
docker run --rm -d --net=host --ulimit nofile=1048576:1048576 --name dsv41-proxy -e PREFILL_HOST=<P_IP> -e DECODE_HOST=<D_IP> dsv41-pd:4b0fb058d proxy
# AgentX 压测（客户命令同参数；结果在 /cache/results/ 下）
docker run --rm --net=host --ulimit nofile=1048576:1048576 -v /var/tmp/dsv41-cache:/cache -e PROXY_URL=http://<proxy>:8000 \
  -e CONC=512 -e DURATION=3600 dsv41-pd:4b0fb058d bench
```

- `--privileged`（访问 `/dev/vfio`）、`--net=host`（P/D 之间的 Raiden 端口、`hostname -I` 取 IP）、`--ipc=host`（`/dev/shm` 用于 Engram 共享表和 KV 传输缓冲）都需要。
- `/cache`：编译缓存（每种配置 20–45 GB，P、D 不同）、`TMPDIR`、aiperf 数据集和结果。挂宿主机本地盘，重启时复用缓存：冷启动约 90 分钟（P 约 50 分钟），有缓存时约 20 分钟。
- 就绪检查：`curl <P>:8100/health`、`curl <D>:8200/health` 返回 200。

## 默认配置（AgentX c512 3600s，docker 内：8.98 req/s，ITL 10.8 ms，TTFT P50/P90 0.99/1.93 s）

| 角色 | 默认值（都可用同名环境变量覆盖） |
|---|---|
| P | `--scheduler-cls ...TpuDpAttnBalanceScheduler`、`TPU_P_WHALE_TOKENS=16384`、`TPU_P_WHALE_STEP_TOKENS=1024`、`TPU_DP_ATTN_STEP_CAP=0`、编译桶 `[128,256,2048]`、`MAX_NUM_SEQS=16` |
| D | `MAX_NUM_SEQS=64`、`DECODE_COMPILE_SIZES=[16,48,96,144,192,288,384,2048]`、`TPU_MOE_V2_INPUT_PIPELINE=off`、`TPU_ENGRAM_GATHER_THREADS=6` |
| 两者 | `--block-size 256`（`BLOCK_SIZE`）、`TPU_DSPARK_SAMPLED_DRAFT=1`、DP8/EP8、FP8 KV/Index-K、async scheduling |
| proxy | 前缀路由 `PROXY_SESSION_AFFINITY_DP=8`、D 路由 `PROXY_DECODE_ROUTING=1` |

其他可调变量见 `scripts/dsv41_pd/start_1p1d_epic.sh`（`EXTRA_ARGS` 追加到 `vllm serve` 命令末尾）。
多 P（2P1D）时每个 P 另设 `TPU_RAIDEN_ENGINE_ID`、`TPU_RAIDEN_JOB_NAME`、`TPU_KV_TRANSFER_NAMESPACE`（参考 `gke-script/pod_start.sh`）。

## 宿主机要求

- 主机名短于 64 字符（gloo 读 `gethostname()` 的缓冲区大小限制）。
- 宿主机内存约 944 GB；`/dev/shm` 至少约 250 GB 可用。镜像里的 Python 是 Ubuntu 的非 PIE 版本，编译内存回收只能关，D 编译时内存比较紧。
- 启动前确认上一轮的 `VLLM::Worker` 进程已退出、`/dev/vfio` 已释放，并清掉 `/dev/shm` 里残留的 `psm_*`、`torch_tpu_cache`。

## 构建与测试（本目录）

| 文件 | 作用 |
|---|---|
| `Dockerfile`、`dsv41-pd-entrypoint.sh` | 镜像定义和入口脚本（角色 `prefill` / `decode` / `proxy` / `bench`） |
| `build_image.sh` | 组装构建上下文（复制宿主机 venv 并把 `/zzlfs/...` 路径改写成 `/opt/...`，浅 clone 代码，复制 tokenizer），然后 `docker build`、做不需要 TPU 的检查、导出 tar.gz 和 sha256 |
| `test_image.sh` | 在 P 机上跑端到端测试：等 P/D 就绪 → 起 proxy → 冒烟 → PD logprob 检查（3k–300k token）→ 可选 AgentX 压测（`BENCH=1`） |
| `REPRODUCE-docker.zh.md` | 给使用方的完整重现步骤（也放在 `gs://zzl-bucket/dsv41_1p1d/`） |

```bash
COMMIT=4b0fb058d bash scripts/dsv41_pd/docker/build_image.sh            # 构建 + 导出到 /zzlfs/temu/docker
P_IP=10.128.0.62 D_IP=10.128.0.39 bash scripts/dsv41_pd/docker/test_image.sh
P_IP=10.128.0.62 D_IP=10.128.0.39 BENCH=1 bash scripts/dsv41_pd/docker/test_image.sh   # 加 AgentX c512 3600s
```

基础镜像是 `ubuntu:24.04`：venv 基于它自带的 python3.12 3.12.3 构建，自编的 tpu-sync 需要 glibc 2.38。构建机需要是 Ubuntu 24.04，并且有上面这些 venv。
