# DeepSeek-V4.1-Flash 1P1D（TPU v7x）Docker 重现步骤

镜像 `dsv41-pd:4b0fb058d` 里包含模型权重以外的全部内容：

- 推理 venv：torch 2.13、torch-tpu 0.2.0、jax 0.11.2、libtpu 0.0.50、vLLM v0.31.1rc0。运行时补丁已打好，tpu-sync 扩展已换成自编版本。
- 代码：vllm-torchtpu `cszhz/dsv41-1p1d` 分支的 `4b0fb058d`，即新 KV 布局 + P 侧巨型请求调度 + D 侧优化。
- toy proxy（前缀路由）、aiperf 0.14.0 压测客户端、tokenizer。

模型权重在启动时从 `gs://zzl-bucket/models/DeepSeek-V4.1-Flash` 流式读取（runai streamer）。这个路径写死在镜像内的启动脚本里，所以运行的主机必须能读这个 bucket。

验证状态（2026-11-25）：已经在两台 v7x 上用本镜像完成端到端验证，冒烟测试和 PD logprob 检查（3k–300k token）全部通过，AgentX c512 3600s 压测结果为 8.98 req/s（见第 6 节）。冷启动约 90 分钟。

| 文件 | 位置 |
|---|---|
| 镜像 | `gs://zzl-bucket/dsv41_1p1d/dsv41-pd-4b0fb058d.tar.gz`（1.9 GB；sha256 见同名 `.sha256`） |
| 本文档 | `gs://zzl-bucket/dsv41_1p1d/REPRODUCE-docker.zh.md` |

## 0. 机器要求

- 两台 4 芯片 TPU v7x 主机：一台跑 P（prefill），一台跑 D（decode）。下文记作 `P_IP`、`D_IP`。两台之间网络互通。
- 操作系统 Ubuntu 24.04，已经装好 TPU 驱动（`/dev/vfio` 可用）。主机名不超过 63 个字符（gloo 的限制）。
- 主机内存约 944 GB，`/dev/shm` 至少约 250 GB 可用。
- 本地盘至少留 100 GB 给编译缓存。
- 主机的服务账号能读 `gs://zzl-bucket`。

## 1. 安装 docker，加载镜像（两台都做）

```bash
sudo apt-get install -y docker.io
gcloud storage cp gs://zzl-bucket/dsv41_1p1d/dsv41-pd-4b0fb058d.tar.gz* .
# 没有 GCS 权限、拿到的是公共下载链接时，改用下面两行（链接要加引号，并且存成原文件名）：
#   curl -o dsv41-pd-4b0fb058d.tar.gz "<镜像链接>"
#   curl -o dsv41-pd-4b0fb058d.tar.gz.sha256 "<sha256 链接>"
sha256sum -c dsv41-pd-4b0fb058d.tar.gz.sha256
sudo docker load -i dsv41-pd-4b0fb058d.tar.gz        # 得到 dsv41-pd:4b0fb058d
```

启动前确认本机没有残留的 vLLM 进程（`ps aux | grep VLLM`），`sudo lsof /dev/vfio/*` 为空。
如果 `/dev/shm` 里留有上一轮的 `psm_*` 或很大的 `torch_tpu_cache`，先清掉。

## 2. 启动 P 和 D

两台用同一个镜像，只是角色参数不同。

```bash
COMMON="--rm -d --privileged --net=host --ipc=host --ulimit memlock=-1 --ulimit nofile=1048576:1048576 -v /var/tmp/dsv41-cache:/cache"

# 在 P 主机上（服务端口 :8100）
sudo docker run $COMMON --name dsv41-p dsv41-pd:4b0fb058d prefill
# 在 D 主机上（服务端口 :8200）
sudo docker run $COMMON --name dsv41-d dsv41-pd:4b0fb058d decode
```

- `--privileged`：容器要访问 `/dev/vfio`。
- `--net=host`：P/D 之间的 Raiden KV 传输端口，以及用 `hostname -I` 取本机 IP。
- `--ipc=host`：`/dev/shm` 用于 Engram 共享表和 KV 传输缓冲。
- `/cache`：编译缓存、临时文件、数据集和结果都放这里，每种配置 20–45 GB。冷启动约 90 分钟（P 约 50 分钟），有缓存时重启约 20 分钟。

用下面的命令看进度，两台都返回 200 就表示就绪：

```bash
sudo docker logs -f dsv41-p           # D 主机上看 dsv41-d
curl -s -o /dev/null -w '%{http_code}\n' http://P_IP:8100/health
curl -s -o /dev/null -w '%{http_code}\n' http://D_IP:8200/health
```

## 3. 启动 proxy（通常和 P 放在同一台）

```bash
sudo docker run --rm -d --net=host --ulimit nofile=1048576:1048576 --name dsv41-proxy \
  -e PREFILL_HOST=P_IP -e DECODE_HOST=D_IP dsv41-pd:4b0fb058d proxy      # :8000
```

## 4. 冒烟测试

```bash
curl -s http://P_IP:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

## 5. PD 正确性检查（logprob 对比）

同一个 prompt 分别走两条路径：一条是 P→D（经过 proxy），另一条直接发给 D、由 D 自己做 prefill。比较两者第一个 token 的 top-5 logprob。
prompt 长度 3k–300k token，各长度都覆盖 D 端前缀命中和不命中两种情况。脚本从代码目录里取文本当语料。

```bash
sudo docker run --rm --net=host dsv41-pd:4b0fb058d bash -c '
  mkdir -p /zzlfs/temu && ln -s /opt/vllm-torchtpu /zzlfs/temu/vllm-torchtpu &&
  python -I /opt/vllm-torchtpu/scripts/dsv41_pd/pd_logprob_check.py \
    --proxy http://P_IP:8000 --direct http://D_IP:8200 \
    --tokens 3000 20000 100000 300000 --offset 1000000'
```

输出共 8 行，每个长度两行：`pd` 是全新 prompt，`pd_hit` 是 D 端前缀命中。脚本不会打印 PASS/FAIL，需要人工判断：

- **正常**：生成的文本开头基本一致（用的是 FP8 KV，偶尔第 3、4 个 token 会分叉）。`max|dlogprob|` 一般小于 1（实测 0.2–1.0），这是 bf16/FP8 数值噪声。
- **有问题**：出现 NaN、HTTP 400 或请求中断；首 token 不一致；`max|dlogprob|` 达到好几个单位。这些都说明 KV 传输有问题。

## 6. AgentX 压测

参数和 InferenceX AgentX 的标准命令相同：c512、3600 s、seed 2026110501。数据集第一次会下载到 `/cache/hf`，结果写在 `/cache/results/agentx-c512-<时间>/` 下。

```bash
sudo docker run --rm --net=host --ulimit nofile=1048576:1048576 -v /var/tmp/dsv41-cache:/cache \
  -e PROXY_URL=http://P_IP:8000 -e CONC=512 -e DURATION=3600 dsv41-pd:4b0fb058d bench
```

我们在同样的两台 v7x 上实测，代码相同（4b0fb058d）：

| 指标 | docker（本镜像） | 宿主机直接运行 |
|---|---|---|
| 请求吞吐 | 8.98 req/s | 9.02 req/s |
| 输出吞吐 | 9374 tok/s | 9420 tok/s |
| ITL（avg） | 10.8 ms | 10.1 ms |
| TTFT P50 / P90 | 0.99 s / 1.93 s | 1.02 s / 2.01 s |
| 错误率 | 0%（32596 个请求） | 0% |

两者的差别在一次压测的波动范围内，docker 本身没有带来可见的开销。

作为参照，优化前的基线是 8.16 req/s、ITL 13.5 ms、TTFT P90 11.9 s。
注意：不要在吞吐压测过程中采集 xprof，会让 P 卡住好几分钟，结果失真。

## 7. 默认配置

下表的默认值都可以用同名环境变量覆盖，例如 `docker run -e MAX_NUM_SEQS=32 ...`。

| 角色 | 默认值 |
|---|---|
| P | `--scheduler-cls vllm_torchtpu.core.dp_attn_balance_scheduler.TpuDpAttnBalanceScheduler`、`TPU_P_WHALE_TOKENS=16384`、`TPU_P_WHALE_STEP_TOKENS=1024`、`TPU_DP_ATTN_STEP_CAP=0`、`MAX_NUM_SEQS=16` |
| D | `MAX_NUM_SEQS=64`、`DECODE_COMPILE_SIZES=[16,48,96,144,192,288,384,2048]`、`TPU_MOE_V2_INPUT_PIPELINE=off`、`TPU_ENGRAM_GATHER_THREADS=6` |
| 两者 | `BLOCK_SIZE=256`（新 KV 布局要求）、`TPU_DSPARK_SAMPLED_DRAFT=1`、DP8/EP8、FP8 KV/Index-K、async scheduling、DSpark 投机解码 K=5、Engram CPU offload |
| proxy | 前缀路由 `PROXY_SESSION_AFFINITY_DP=8`、D 路由 `PROXY_DECODE_ROUTING=1` |

其他可调变量见镜像内的 `/opt/vllm-torchtpu/scripts/dsv41_pd/start_1p1d_epic.sh`。`EXTRA_ARGS` 会追加到 `vllm serve` 命令末尾。
进入容器调试用 `sudo docker run --rm -it ... dsv41-pd:4b0fb058d bash`。

## 8. 常见问题

- **压测一开始就出现 `Connection reset by peer`，aiperf 报 `warmup_failure` 后中止**：proxy 或压测容器的文件描述符上限还是 docker 默认的 1024。c512 时 proxy 要同时保持数千条连接，所以所有容器都要加 `--ulimit nofile=1048576:1048576`（上面的命令已经加了）。

- **编译阶段报 `RuntimeError: cancelled` 或 worker 被 OOM kill**：通常是 `/dev/shm/torch_tpu_cache` 或残留的 `psm_*` 占了内存。停掉容器，清理后再启动。
- **磁盘写满**：每换一种配置就会新增 20–45 GB 编译缓存。删掉 `/var/tmp/dsv41-cache/vllm` 下不用的配置目录和 `/var/tmp/dsv41-cache/tmp`。
- **停止**：`sudo docker stop dsv41-p`（D、proxy 同理）。停止后等 `/dev/vfio` 释放再重新启动。
