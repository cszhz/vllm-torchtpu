# DeepSeek-V4.1-Flash 1P1D（vllm-torchtpu，Stage-3 Raiden）复现手册

更新：2026-11-06。称呼约定：**GCP** = 我们的部署；**Temu** = 客户内部数据（对比基准，当前用 2026-11-02 版本）。

## 最新结果一览

2026-11-05/06 在新主机（g45q / pqs0）上按本手册复测，代码 `cszhz/dsv41-1p1d`。详细表格见第 5 节，旧主机结果见附录 B。

| 指标 | GCP | Temu（11-02） | 倍数（GCP ÷ Temu） |
|---|---|---|---|
| Prefill 1P（1k+1k … 32k+768k） | 0.23 – 48.4 s | 0.036 – 3.79 s | 6.3 – 15.0× |
| TTFT 1P | 0.31 – 51.0 s | 0.065 – 4.19 s | 4.7 – 12.2× |
| TTFT 4P | 0.40 – 48.1 s | 0.12 – 13.2 s | 2.2 – 4.9× |
| TTFT 8P | 0.39 – 49.0 s | — | — |
| TPOT（bs 1–256，cache 1k–768k） | 4.7 – 44.5 ms | 2.2 – 8.3 ms | 1.6 – 7.8× |
| P 的 prefill 吞吐上限 | 整机约 17.2 万 tok/s，每芯片约 4.3 万 | — | — |

---

## 1. 部署与代码

### 1.1 拓扑

| 角色 | 主机 | 地址 | 说明 |
|---|---|---|---|
| P（prefill） | tpuv7-mig-g45q | 10.128.0.145:8100 | 一台 4 芯片 v7x，DP8 / TP1 / EP8（tpu7x-16 slice 的 worker 0） |
| D（decode） | tpuv7-mig-pqs0 | 10.128.0.28:8200 | 同上（worker 1） |
| proxy | 与 P 同机 | 10.128.0.145:8000 | 客户端只打这个端口 |

- 原部署（附录 B 的结果）：P = zzl-tpu7x-slice-mig-2z6z（10.128.0.175），D = …-clsk（10.128.0.177），
  另有编译 / 单测机 …-sjd1（10.128.0.176，**不要在 P/D 上跑 pytest，会占 TPU**）。脚本里的默认地址仍是原部署，新主机用环境变量覆盖（第 3 节）。
- 每个 DP rank 是半颗 v7x（2 个 TensorCore）。EP 下 8 个 DP rank 同步（lockstep）前进，空闲 rank 每步跑 dummy forward。
- 模型：`gs://zzl-bucket/models/DeepSeek-V4.1-Flash`（runai streamer 直接从 GCS 加载）。
- 共享盘 `/zzlfs` 所有机器都挂载，下文路径都在共享盘上。

### 1.2 代码版本

- vllm-torchtpu：`https://github.com/cszhz/vllm-torchtpu`，分支 **`cszhz/dsv41-1p1d`**（本地 `/zzlfs/temu/src/vllm-torchtpu-cszhz`），
  基于上游 `vllm-project/vllm-torchtpu` 的 `epic/dsv41`（6a1ebddb7）。**所有改动只在 cszhz fork 上，不要推上游**
  （本地 clone 的 upstream push URL 已设成 `DISABLED-do-not-push-to-vllm-project`）。提交列表见附录 A。
- P、D 用同一份代码，以 editable 方式装进 venv。
- tpu-sync（KV 传输）：`GetBlockChunks` 原来每个 block 扫描整个 schedule（O(n²)），557k token 传一次要 27.9s，修复后 0.95s
  （再加上前缀命中跳过，实际只要 30–40ms）。PR **google/tpu-sync#1382**（cszhz/tpu-sync 分支 `fix/getblockchunks-entry-index-main`，基于 main）。
  **部署实际用的版本**：nightly 1012（e3f6f4fc）+ 同一修复 = cszhz/tpu-sync 分支 `fix/getblockchunks-on-nightly-1012`（3e5d3c35），
  本地 `/zzlfs/temu/src/tpu-sync-1012`。原因：main 的 Python API 和 PyPI 上的 `tpu-sync-torch==0.0.1.dev20261012084241` 对不上，而且 main 需要 libibverbs。

---

## 2. 环境搭建

### 2.1 venv（`/zzlfs/temu/.venv-epic`）

epic/dsv41 需要 jax 0.11.2（旧 venv 缺 `pltpu.ACC`），所以单独建一个新 venv。依赖清单见 `scripts/dsv41_pd/requirements-epic.txt`
（torch 2.13.0、torch-tpu 0.2.0、jax/jaxlib 0.11.2、libtpu 0.0.50、vLLM v0.31.1rc0 = e37e51dd、tpu-sync-torch dev20261012、runai-model-streamer[gcs]、xxhash）。

```bash
UV=/zzlfs/.tools/pyenv/bin/uv
$UV venv -q -p /usr/bin/python3.12 /zzlfs/temu/.venv-epic
# vLLM 用 empty 目标构建（TPU 插件由 vllm_torchtpu 提供）；torch 走 CPU wheel 源
(cd /tmp && VLLM_TARGET_DEVICE=empty CC=gcc CXX=g++ $UV pip install --python /zzlfs/temu/.venv-epic/bin/python \
    --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /zzlfs/temu/src/vllm-torchtpu-cszhz/scripts/dsv41_pd/requirements-epic.txt)
$UV pip install --python /zzlfs/temu/.venv-epic/bin/python orjson
# vllm_torchtpu 以 editable 方式指向 cszhz 这份代码
$UV pip install --python /zzlfs/temu/.venv-epic/bin/python --no-deps -e /zzlfs/temu/src/vllm-torchtpu-cszhz
```

- torch-tpu 的私有源（registry）返回 403，所以改用 PyPI 上的 torch-tpu 0.2.0 正式版（pyproject 里写的是 0.2.0rc1）。
- 这个 venv 建在非 PIE 的 `/usr/bin/python3.12` 上，所以 `TPU_RECLAIM_COMPILE_MEMORY` 只能关（见第 7 节）。
- 旧的 `/zzlfs/temu/.venv` 留给旧 PD 部署；benchmark 客户端脚本仍用 `.venv/bin/python` 跑（只需要 httpx）。

### 2.2 编译并替换 tpu-sync 的 .so（在 sjd1 上编）

```bash
# 取代码（共享盘上已有的话跳过）
git clone -b fix/getblockchunks-on-nightly-1012 https://github.com/cszhz/tpu-sync.git /zzlfs/temu/src/tpu-sync-1012
# sjd1：需要 clang-18（libibverbs-dev 已装）
cat /zzlfs/temu/src/build-tpusync-1012.sh      # 内容如下
#   cd /zzlfs/temu/src/tpu-sync-1012
#   SP=/zzlfs/temu/.venv-epic/lib/python3.12/site-packages
#   export CC=/usr/bin/clang-18 CXX=/usr/bin/clang++-18
#   export PATH=/zzlfs/temu/tools/nopip:$PATH TORCH_SOURCE=$SP TORCH_TPU_SOURCE=$SP
#   bash build.sh torch
ssh zzl-tpu7x-slice-mig-sjd1 bash /zzlfs/temu/src/build-tpusync-1012.sh

# 产物改 NEEDED 后放进 venv（原 .so 备份为 .pypi-1012）
D=/zzlfs/temu/tools/tpu_sync_epic/_tpu_raiden_torch_2_13_0.so
cp /zzlfs/temu/src/tpu-sync-1012/tpu_sync/frameworks/torch/_tpu_raiden_torch.so $D
patchelf --replace-needed libpywrap_torch_tpu_common.so libpywrap_2_13_0_common.so $D
T=/zzlfs/temu/.venv-epic/lib/python3.12/site-packages/tpu_sync/frameworks/torch
cp $T/_tpu_raiden_torch_2_13_0.so $T/_tpu_raiden_torch_2_13_0.so.pypi-1012   # 只做一次
cp $D $T/_tpu_raiden_torch_2_13_0.so
```

（`/zzlfs/temu/tools/nopip/python3` 是一个包装，用来阻止 build.sh 调用 pip 去改 venv。）

### 2.3 第三方包运行时补丁（必做，可重复执行）

```bash
/zzlfs/temu/.venv-epic/bin/python /zzlfs/temu/src/vllm-torchtpu-cszhz/scripts/dsv41_pd/apply_venv_patches.py
```

| 节 | 改动 | 原因 / 效果 |
|---|---|---|
| §3 | runai_model_streamer_gcs：`bucket()` 代替 `get_bucket()` | 不需要 `storage.buckets.get` 权限 |
| §4 | vLLM 调度器：connector 的 `skips_prefix_replay(request)` 为真时，远端加载不取整、不回放 | D 在 `--swa-bounded-replay` 下也能保留前缀缓存（Stage-3 要求完整页集，原逻辑会让 decode 引擎退出） |
| §5 | vLLM detokenizer：`DecodeStream` 只用 prompt 末尾 32 个 token 预热（如果末尾解码出 `�` 则退回整段） | 原来 557k token 时要预热 0.5s，P、D 各一次；512k+32k TTFT 4.27→3.58s |
| §6 | DP busy loop：`VLLM_DP_WAVE_START_DELAY_MS`，新 wave 开始时先等一会再收一次请求 | 让几乎同时到达、分散在不同 rank 的请求赶上同一个 lockstep 步（4P 对齐） |
| §7 | starlette `Request.json` 优先用 orjson | 10^6 个 token id 的请求体解析更快 |

原文件备份在 `/zzlfs/temu/tools/*-epic.py.orig`。

### 2.4 每台机器的准备（换主机时逐项检查）

- tokenizer：把模型目录里的 `tokenizer.json`、`tokenizer_config.json`、`config.json`、`chat_template.jinja`、`encoding/`
  拷到**每台机器**的 `/tmp/dsv41-tokenizer`（P、D、proxy 都用这个路径）。
- 主机名短于 64 字符：`sudo hostname $(hostname -s)`（GCE 主机名正好 64 字符时 gloo 会报错，重启或 guest agent 会改回去）。
- 根分区小（新主机只有 8.7 GB）时：`TMPDIR`、`VLLM_CACHE_ROOT` 放到 `/dev/shm`（第 3 节的命令已包含）。
- `/dev/shm` 里不用的缓存会直接占主机内存，启动前清掉（`stop_vllm_local.sh` 会清 `psm_*` 和超过 30G 的 `torch_tpu_cache`）。
- 各条原因见第 7 节。

---

## 3. 启动与停止

脚本都在 `scripts/dsv41_pd/`（部署时用的是 `/zzlfs/temu/scripts/` 下的同名副本，内容一致）。下面是新主机的命令；
不设 `P_HOST` / `D_HOST` / `P_ADDR` / `D_ADDR` / `PREFILL_HOST` / `DECODE_HOST` 时就是原部署的默认值。

```bash
cd /zzlfs/temu/scripts
export P_HOST=10.128.0.145 D_HOST=10.128.0.28 TMPDIR=/dev/shm/epic-tmp VLLM_CACHE_ROOT=/dev/shm/epic-vllm-cache
# P：默认配置（不开 async、compile [128,256,2048]、4 个 API server、起步等待 40ms）
ROLES=prefill bash restart_epic.sh
# D：async + MAX_NUM_SEQS=32 + compile [16,48,96,192,2048]（起步等待默认 20ms）
ROLES=decode ASYNC_SCHED=1 MAX_NUM_SEQS=32 DECODE_COMPILE_SIZES='[16,48,96,192,2048]' bash restart_epic.sh
# 等两边 /health 返回 200，并检查日志里有没有致命错误；冷启动约 30–45 分钟（有编译缓存时更短）
P_ADDR=10.128.0.145 D_ADDR=10.128.0.28 bash wait_ready.sh
# proxy（在 P 那台上）
ssh -n -f 10.128.0.145 'PREFILL_HOST=10.128.0.145 DECODE_HOST=10.128.0.28 setsid nohup bash /zzlfs/temu/scripts/start_proxy.sh \
    > /zzlfs/temu/logs/proxy-$(date +%m%d-%H%M).log 2>&1 < /dev/null &'
# 测试脚本用的地址
export PROXY_URL=http://10.128.0.145:8000 PREFILL_METRICS_URL=http://10.128.0.145:8100/metrics D_METRICS_URL=http://10.128.0.28:8200/metrics
```

- `restart_epic.sh` 通过 ssh 在对应主机上先执行 `stop_vllm_local.sh`（连残留 worker 一起停，等 `/dev/vfio` 释放；根分区超过 75% 时清缓存），
  再启动 `start_1p1d_epic.sh <role>`，并把 `TMPDIR`、`VLLM_CACHE_ROOT` 和下面的可调变量透传过去。
  日志写到 `/zzlfs/temu/logs/{prefill,decode}-epic-<MMDD-HHMM>.log`，时间戳记录在 `/tmp/epic_T`。
  P、D 分两次启动且跨了分钟时，`wait_ready.sh` 会找不到另一边的日志：用 `ln -sf` 把那边的日志链接成同名。
- `start_1p1d_epic.sh` 的关键参数：`--data-parallel-size 8 --enable-expert-parallel --kv-cache-dtype fp8 --block-size 16
  --enable-prefix-caching --swa-bounded-replay --engram-config '{"cpu_offload":true}'
  --additional-config '{"deepseek_v41":{"execution":"ced","moe":"fused_ep_w4a8"}}'
  --speculative-config dspark K=5 --max-num-batched-tokens 2048`，KV connector 为 `TPURaidenConnector`。
- 可调环境变量：`ASYNC_SCHED`、`MAX_NUM_SEQS`、`PREFILL_COMPILE_SIZES`、`DECODE_COMPILE_SIZES`、`API_SERVERS`、
  `VLLM_DP_WAVE_START_DELAY_MS`、`TP_SIZE`/`DP_SIZE`、`NO_KV_TRANSFER`、`KV_BYTES`、`EXTRA_ARGS`。
  诊断用：`VLLM_LOGGING_PREFIX='%(created).3f '`（日志带毫秒级时间戳，`pd_timeline.py` 依赖它）、`TPU_DUMMY_STEP_TIMING=1`、`TPU_STAGE3_PROFILE=1`。
- proxy 环境变量：`PROXY_TOKENIZER`（默认 `/tmp/dsv41-tokenizer`，本地分词）、`PROXY_TOKENIZER_PROCS`（默认 8）、
  `PROXY_ALIGN_MS`（默认 0=关，见 6.2）、`PROXY_RENDER_CACHE_MB`（默认 0=关，见 6.2）。

**冒烟测试**

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
/zzlfs/temu/.venv/bin/python -I /zzlfs/temu/scripts/pd_logprob_check.py --tokens 3000 20000   # PD 与直接打 D 的首 token logprob 对比
```

**停止**：`bash /zzlfs/temu/scripts/stop_vllm_local.sh`（在 P、D 各执行一次）；proxy 按 PID 停：`kill $(pgrep -f 'oy_proxy_serv[e]r')`。
调用 `stop_vllm_local.sh` 的命令行里不要出现 `VLLM::`、`.venv/bin/vllm serve` 这些字样，否则会连调用者的 shell 一起停掉。

---

## 4. 测试方法

客户端统一打 proxy（:8000），用 `/zzlfs/temu/.venv/bin/python -I` 运行。

### 4.1 Prefill / TTFT 表（cached + new）

```bash
OUT=/zzlfs/temu/results/table-$(date +%m%d-%H%M).jsonl PIN=distinct bash run_table.sh        # 11 个场景 × 1P、4P
CONCS="8" bash run_table.sh                                                                 # 8P
ONLY="1024:1024 524288:32768" bash run_table.sh                                             # 只跑部分场景
bash quick_set.sh                                                                           # 快速对比子集
```

- 每个场景先预热前缀（让 P 和 D 都缓存好 cached 部分），再发 cached+new；k = 1024 token，168k 按 172032。
- **Prefill** = P 的 vLLM 指标 `request_prefill_time`；**TTFT** = 客户端从发出请求到收到第一个 token。
- `--pin distinct`：第 i 条并发请求通过 `X-data-parallel-rank` 固定到 rank `i % 8`，4P 时 4 条请求落在 4 个不同的 rank 上。
- 请求体在计时前就编码好（大 prompt 的 JSON 编码不算进 TTFT）。短场景 5 轮、总长 ≥200k 的场景 3 轮，取中位数。

### 4.2 TPOT 表

```bash
/zzlfs/temu/.venv/bin/python -I tpot_bench.py --out /zzlfs/temu/results/tpot-$(date +%m%d-%H%M).jsonl
# 默认 cache 1k…768k × bs 1/16/64/128/256，out_len 1024；512k/768k bs256 拉满窗口太短时加 --out-len 3072
```

- 所有请求共享一个长度为 cache 的前缀，各自再接 128 个不同的 token，ignore_eos，temperature 0。语料是代码文本（`/zzlfs/temu/tools/tpot_corpus_ids.json`）。
- 只统计"batch 拼满"的窗口：从所有请求都收到第一个 token，到第一个请求收到最后一个 token。
  - **CLI** = 每条请求在窗口内的平均 token 间隔，再对请求取平均。
  - **Server** = D 的 `inter_token_latency_seconds_sum` 增量 ÷ `generation_tokens_total` 增量（投机解码一步可能出多个 token）。
- 同时记录 DSpark 的平均接受长度（约 4.3–5.8）。

### 4.3 P 的 prefill 吞吐（TPS）

**测什么**：P 满载时每秒能算多少个新 token。只测 P，不经过 proxy 和 D。

**P 怎么起**：用和线上 P 相同的配置（chunk 2048、编译桶 `[128,256,2048]`、DP8/EP8、不开 async），但不挂 KV 传输。
最好起在一台单独的机器上（原部署用 sjd1）；只有两台机器时，可以先停掉 PD 再在 P 那台上起，
也可以不重启、直接测 PD 里正在运行的 P（`--url http://<P>:8100`，不要和其他测试同时跑；5.3 节就是这样测的，没有开 async 的对比）：

```bash
ssh -n -f <host> "cd /zzlfs/temu && TMPDIR=/dev/shm/epic-tmp VLLM_CACHE_ROOT=/dev/shm/epic-vllm-cache NO_KV_TRANSFER=1 \
    setsid nohup bash /zzlfs/temu/scripts/start_1p1d_epic.sh prefill > /zzlfs/temu/logs/tps-prefill-\$(date +%m%d-%H%M).log 2>&1 < /dev/null &"
# 机器上要先有 /tmp/dsv41-tokenizer；等日志出现 "Application startup complete"。对比 async 时再加 ASYNC_SCHED=1
```

**怎么测**（`tps_bench.py`）：

```bash
cd /zzlfs/temu/scripts
OUT=/zzlfs/temu/results/tps-$(date +%m%d-%H%M).jsonl
PY=/zzlfs/temu/.venv-epic/bin/python
# 新 token 长度扫描，8 并发（每个 rank 一条）
$PY -I tps_bench.py --url http://<host>:8100 --tag scan --cached 32768 --new 32768 65536 172032 262144 --conc 8 --out $OUT
# 持续满载，每个 rank 排 2 条、3 条
$PY -I tps_bench.py --url http://<host>:8100 --tag sustained --cached 32768 --new 172032 --conc 16 24 --out $OUT
```

- 每个 rank 先预热同一个 32k 前缀；每条请求的新 token 取自然代码文本（`tools/tpot_corpus_ids.json`），各用不同的偏移，互不命中缓存；第 i 条通过 `X-data-parallel-rank` 固定到 rank `i % 8`。
- 请求直接发 token id（不含 API server 分词），`max_tokens=1`。
- **TPS = 新 token 总数 ÷ 墙钟时间**（从第一条发出到最后一条完成）。每组先跑 1 轮预热（不计入），再跑 `--rounds`（默认 2）轮正式。
- 每轮从 P 的 metrics 记录 `cache_hit_tokens`，**正式轮的命中必须正好等于 前缀长度 × 并发数**，多了就说明新 token 命中了缓存，这一轮作废。脚本每次运行用随机起点生成偏移，同一个服务上重复跑也不会撞上。

### 4.4 单请求时间线 / 耗时分解

```bash
/zzlfs/temu/.venv/bin/python -I d_breakdown.py --cached 524288 --new 32768 --rounds 3   # D 端 vLLM 指标分解
python3 pd_timeline.py <proxy.log> <prefill.log> <decode.log> [req_id 前缀]             # 毫秒级跨机时间线（需 VLLM_LOGGING_PREFIX）
python3 transfer_stats.py ...                                                           # KV 传输耗时统计
```

---

## 5. 结果（2026-11-05/06，新主机 g45q / pqs0）

配置同第 3 节，proxy 的对齐窗口和分词缓存都关；代码 54e98d673（= 1dc6984ac + 6cb686ef4）。Temu 为 2026-11-02 内部数据（`results/temu-1102.md`）。

### 5.1 Prefill / TTFT（秒，格式 Prefill / TTFT，倍数 = GCP ÷ Temu）

1P / 4P：`results/table-new-hosts-1105-1647.{jsonl,log}`；8P：`results/table-new-hosts-8p-1106-0106.{jsonl,log}`（2026-11-06，重启后同配置）。Temu 没有 8P 数据。

| 场景 | GCP 1P | Temu 1P | 1P 倍数 | GCP 4P | Temu 4P | 4P 倍数 | GCP 8P |
|---|---|---|---|---|---|---|---|
| 1k+1k | 0.227 / 0.307 | 0.036 / 0.065 | 6.3× / 4.7× | 0.192 / 0.395 | 0.051 / 0.124 | 3.8× / 3.2× | 0.180 / 0.385 |
| 8k+4k | 0.432 / 0.565 | 0.055 / 0.092 | 7.9× / 6.1× | 0.465 / 0.682 | 0.113 / 0.315 | 4.1× / 2.2× | 0.487 / 0.720 |
| 32k+16k | 1.033 / 1.223 | 0.143 / 0.197 | 7.2× / 6.2× | 0.990 / 1.267 | 0.156 / 0.511 | 6.3× / 2.5× | 0.991 / 1.300 |
| 64k+32k | 1.885 / 2.146 | 0.144 / 0.220 | 13.1× / 9.8× | 1.765 / 2.133 | 0.144 / 0.609 | 12.3× / 3.5× | 1.794 / 2.245 |
| 168k+32k | 2.043 / 2.413 | 0.148 / 0.277 | 13.8× / 8.7× | 1.944 / 2.440 | 0.149 / 0.641 | 13.0× / 3.8× | 2.043 / 2.699 |
| 32k+168k | 9.139 / 9.785 | 0.726 / 0.859 | 12.6× / 11.4× | 8.408 / 9.297 | 0.747 / 2.823 | 11.3× / 3.3× | 8.597 / 9.621 |
| 64k+64k | 3.598 / 3.937 | 0.263 / 0.358 | 13.7× / 11.0× | 3.335 / 3.797 | 0.265 / 1.073 | 12.6× / 3.5× | 3.417 / 3.970 |
| 256k+32k | 2.194 / 2.664 | 0.154 / 0.319 | 14.2× / 8.4× | 2.174 / 2.841 | 0.156 / 0.684 | 13.9× / 4.2× | 2.217 / 3.082 |
| 512k+32k | 2.634 / 3.388 | 0.176 / 0.452 | 15.0× / 7.5× | 2.711 / 3.938 | 0.176 / 0.810 | 15.4× / 4.9× | 2.707 / 4.183 |
| 768k+32k | 3.010 / 4.083 | 0.200 / 0.597 | 15.0× / 6.8× | 2.900 / 4.532 | 0.202 / 0.952 | 14.4× / 4.8× | 3.185 / 5.237 |
| 32k+768k | 48.400 / 50.985 | 3.792 / 4.186 | 12.8× / 12.2× | 45.156 / 48.122 | 3.882 / 13.198 | 11.6× / 3.6× | 45.536 / 49.021 |

- 和旧主机（附录 B.1）比：1P 的 Prefill 都在 ±3% 以内，TTFT 普遍快 2–5%（本轮代码含 a233edf04，旧表在它之前测）；
  4P 长场景 TTFT 改善更明显（512k+32k 4.259→3.938s，768k+32k 5.261→4.532s）。1k / 8k 的 4P Prefill 慢约 20%（绝对差 30–80ms，短场景单次波动大）。
- 8P 和旧主机比：短场景一致，256k 以上前缀的场景快 5–10%（512k+32k 4.650→4.183s，768k+32k 5.688→5.237s）。
- Temu 11-02 比 10 月的数据快，所以倍数比旧表大。

### 5.2 TPOT（ms，格式 CLI / Server，倍数按 CLI 算）

`results/tpot-new-hosts-1105-1748.{jsonl,log}`，默认参数 out_len 1024。

| bs | 引擎 | 1k | 16k | 32k | 64k | 128k | 256k | 512k | 768k |
|---|---|---|---|---|---|---|---|---|---|
| 1 | GCP | 4.91/5.02 | 4.68/4.61 | 4.94/4.78 | 5.40/5.35 | 5.97/5.95 | 6.30/6.35 | 7.74/7.81 | 9.16/9.22 |
| 1 | Temu | 2.22 | 2.24 | 2.24 | 2.22 | 2.25 | 2.25 | 2.24 | 2.26 |
| 16 | GCP | 5.23/5.14 | 5.15/5.07 | 5.27/5.04 | 5.42/5.26 | 5.85/5.74 | 6.56/6.39 | 7.72/7.58 | 8.66/8.64 |
| 16 | Temu | 3.10 | 3.14 | 3.15 | 3.15 | 3.15 | 3.15 | 3.19 | 3.21 |
| 64 | GCP | 7.43/7.38 | 8.28/8.08 | 9.18/8.80 | 10.99/10.65 | 8.68/8.37 | 10.27/10.21 | 14.52/13.73 | 19.07/18.31 |
| 64 | Temu | 4.33 | 4.35 | 4.35 | 4.34 | 4.35 | 4.38 | 4.49 | 4.56 |
| 128 | GCP | 9.93/9.73 | 12.47/12.25 | 15.06/14.87 | 19.18/18.74 | 14.51/14.15 | 21.03/20.55 | 31.27/30.38 | 44.47/43.89 |
| 128 | Temu | 5.21 | 5.22 | 5.18 | 5.20 | 5.25 | 5.35 | 5.51 | 5.68 |
| 256 | GCP | 14.60/14.19 | 15.04/14.62 | 15.28/14.75 | 16.86/16.33 | 19.32/18.51 | 24.97/24.88 | —‡ | OOM† |
| 256 | Temu | 7.30 | 7.38 | 7.33 | 7.41 | 7.50 | 7.64 | 7.97 | 8.27 |

- 总体是 Temu 的 1.6–7.8 倍（按 CLI）；DSpark 平均接受长度 4.36–5.09。长 cache、大 batch 差距最大（768k bs128 44.5 vs 5.7ms）。
- 和旧主机（附录 B.2）基本一致，bs1 / bs16 在 1k–512k 范围内相差 ±10% 以内。
- ‡ 512k bs256：out_len 1024 时"拼满窗口"不足 1s（-0.46s），旧表用 out_len 3072 补测，本轮没有补测。
- † 768k bs256：两次 HTTP 500 后 D 被 OOM killer 杀掉（第 7 节）。

### 5.3 P 的 prefill 吞吐上限（tok/s；`results/tps-new-hosts-1106-0224.{jsonl,log}`，方法见 4.3）

直接测 PD 里正在运行的 P（10.128.0.145:8100，挂着 KV connector；请求不带 PD 传输参数，不触发 KV 登记 / 传输），没有另起
`NO_KV_TRANSFER=1` 的 P，所以没有开 async 的对比。32k 缓存前缀，每组 1 轮预热 + 2 轮正式，所有正式轮的 `cache_hit_tokens` 都正好等于 前缀 × 并发。

| 组 | 场景 | 整机 tok/s | 每芯片 tok/s | 旧主机（sjd1，不开 async） |
|---|---|---|---|---|
| 新 token 长度扫描（8 并发） | 32k | 16.4–16.5 万 | 4.11 万 | 15.8–16.1 万 |
| | 64k | 16.4–16.7 万 | 4.11–4.17 万 | 17.7–17.8 万 |
| | **168k** | **17.1–17.2 万** | **4.27–4.30 万** | 17.3 万 |
| | 256k | 16.6–16.9 万 | 4.16–4.23 万 | 17.2 万 |
| 持续满载（168k 新 token） | 16 并发（每个 rank 2 条） | 17.2 万 | 4.29–4.30 万 | 17.0–17.4 万 |
| | 24 并发（每个 rank 3 条） | 16.6–17.1 万 | 4.15–4.27 万 | 16.8–17.0 万 |

- **新主机上 P 的 prefill 吞吐上限约为整机 17.2 万 tok/s，每芯片约 4.3 万 tok/s**，和旧主机（17.5 万 / 4.4 万）基本一致；
  旧主机上 64k 最高，这里 64k 低约 6%、168k 最高，可能和挂着 KV connector 有关，没有单独验证。
- 新 token ≥168k 不再提升；每个 rank 多排请求没有用（每个 rank 每步的预算固定是 2048 个 token）。

### 5.4 如何复现某一组数字

1. 按第 1、2 节准备代码和 venv，checkout 第 5 节注明的提交。
2. 按第 2.4 节准备机器，按第 3 节启动 P、D、proxy（`PROXY_ALIGN_MS`、`PROXY_RENDER_CACHE_MB` 都不设）。
3. 按第 4 节跑对应脚本，结果写进 `results/*.jsonl`，汇总行在同名 `.log` 里（`== ... (median): Prefill x / TTFT y`）。
4. 附录 B 的旧主机数字是在 a233edf04 之前测的（32842f3d7 + d8e22db82 的内存回收改动即可复现）；a233edf04 只降低 TTFT 中 D 这一跳的开销，不影响 Prefill 和 TPOT。

---

## 6. 改动与效果（按收益排序）

### 6.1 已生效（结果表基于这些）

1. **tpu-sync `GetBlockChunks` O(n²)→线性**：557k token 传输 27.9s→0.95s。
2. **D 开前缀缓存，只拉未命中部分**（b4d7bf2af + 补丁 §4）：可前缀缓存的 group 用 `dst_skip_bytes` 跳过命中页，SWA / 压缩器状态按 P 的长度整段拉；调度器对这类请求不取整、不回放。512k 前缀命中时实际传输只要 30–40ms。
3. **P 的编译桶加 128 / 256**（128 = sliding_window）：CED 在最后一个 chunk 上只对尾部 ≤128 行重算 decoder，原来被补齐成 2048 行，每个请求白跑约 0.37s。
   新主机上单机 DP8 cached-extend（96k+16k，C8）复测 `[128,2048]` → `[128,256,2048]`：最后一个 chunk 256.6→184.5ms，整轮 810→724ms（-10.6%），中间 chunk 不变。
4. **升级到最新 epic/dsv41**：kernel 优化让 Prefill 快 15–26%。需要配套 6b6565b38、dd89c8b2d 和新 venv。
5. **反分词器只用 prompt 末尾预热**（补丁 §5）：512k+32k TTFT 4.27→3.58s。2000 例随机测试输出逐字一致。
6. **proxy 只分词一次**：本地切块并行分词（只在"字母 空格 字母"处切，保证拼接结果和整段分词一致）+ 进程池 + orjson；P、D 不再各分词一次（557k token 每次约 2.7s）。
7. **4P 对齐**：DP wave 起步等待（P 40ms / D 20ms）、4 个 API server、orjson 解析、proxy 进程池。512k+32k 4P TTFT 7.0→约 4.0–4.5s。
8. **空闲 EP-DP rank 复用全零 dummy block table**（4250476f6）：bs1 TPOT 降低约一半。
9. **D 开 async scheduling**（P 不开）：TPOT 降低 10–46%；P 开了对 TTFT 没有帮助。

### 6.2 D 这一跳的控制路径优化（2026-10-29，已部署）

512k+32k 1P 毫秒级时间线（`pd_timeline.py`）显示，D 这一跳的 0.58s 里 KV 传输只有约 35ms，其余全是控制路径开销：

| 时刻（ms） | 事件 |
|---|---|
| 0 → 137 | proxy 分词 |
| 2950 | P 请求完成 |
| 2957 | proxy 转发给 D |
| 3161 | D 调度器第一次看到请求（D 接收请求约 200ms：JSON 解析 23ms、token 校验 8ms、前缀哈希 20–40ms、msgpack/ZMQ、起步等待 20ms…） |
| 3255 | **P worker 登记 KV block**（比请求完成晚 305ms，当时的关键路径） |
| 3290 / 3328 | D 提交传输 / 传输完成 |
| 3514 | 第一个 token（KV 到齐后约 170ms） |

- **P 端登记加速**（a233edf04）：`lower_dsv4_row_spans` 按 (每页字节数, rank, 并行度) 缓存整页 span，每次切片 + 补尾页；同时缓存 ReshardClient 的 wire 元组，`_register_request_blocks_prelowered` 直接交给 C++ client（非 C++ client 时退回公开 API）。
  效果：登记 305ms→138ms（lowering 192→72ms，RPC 72→36ms）。与原实现做了 3000 组随机等价性测试。
- **sliding-window trim 加速**（a233edf04，P、D 都部署）：cProfile（`TPU_STAGE3_PROFILE=1`）显示 P 剩下的 72ms 和 D 准备接收计划的 67ms，都花在 `_stage3_trim_window_pages` 上：
  51 个 cache group，每个约 1.7 万个已出窗口的页，逐个 `int()` 加生成器 `any(...)` 检查。改为只转换保留的页，丢弃的页用 `list.count` 检查空块（报错语义不变）。
  效果：51 个 group 74ms→3.4ms；P lowering 72→10.5ms（登记只比请求完成晚 81ms），D stage 67→5ms；**D 这一跳 581→约 500ms**。
  `test_tpu_connector` / `test_pool_byte_spans` / `test_dsv41_pool_manifest` 共 399 个测试通过。
- **proxy 对齐窗口 `PROXY_ALIGN_MS`**（原部署上已部署，当时 8ms；脚本默认 0=关，第 5 节的复测没开）：窗口内先后到达的请求归为一组，组内全部分词完成后再一起发给 P，避免分词耗时不同导致错过同一个 lockstep 步。单请求的对齐等待实测 0.01ms。4P 效果待测。
- **proxy 分词缓存 `PROXY_RENDER_CACHE_MB`**（已实现，默认关）：每个 worker 按 64KB 切块做 LRU 缓存，前缀相同的 prompt 只对尾部重新分词（约省 120ms）。随机测试与整段分词逐 token 一致。
  注意：只在文本前缀重复时有效（多轮对话 / 共享文档，即表里的 cached+new 场景）；报告结果时开、关要分开列。

### 6.3 试过但没用 / 不可行

- async scheduling 用在 P 上：TTFT 无改善。
- chunk 4096 / 8192：XLA SparseCore RET_CHECK，编译失败。
- prefix 哈希换 xxhash：82 万 token 只省 35ms。
- 8 个 API server：对 4P 没有额外帮助；P 的起步等待 60ms 也没有帮助。
- **TP8**：TP>1 时 CSA2 关闭，退回原生注意力 + RoPE 附属池，block 数和主池不同，tpu-sync KVCacheManager 拒绝；不带 KV 传输单独起 P 时输出是乱码（虽然 Prefill 约快 2 倍）。维持 DP8。

---

## 7. 已知问题与踩坑

**主机内存（新主机 944 GB）**

- **`TPU_RECLAIM_COMPILE_MEMORY` 只能关**：`host_memory.py` 用 `dladdr` 检查 malloc 是否来自 glibc。`/usr/bin/python3.12`（以及 uv 的 python-build-standalone）
  是非 PIE 可执行文件，`dlsym(RTLD_DEFAULT,"malloc")` 解析到可执行文件里的 PLT 地址而误判失败（实际 malloc 仍是 glibc）。
  用 PIE 的 Python（`--enable-shared --enable-optimizations --with-lto` 自编 CPython，与 CI 的 `python:3.12-slim` 一致）建 venv 后可以开。
- **关着回收时内存余量很小**：8 个 worker 编译完不归还堆内存，新主机上 P 启动后剩约 140 GB，D（编译桶更多）只剩约 37 GB；
  单机 DP8 的编译阶段曾因此被 OOM killer 杀（`program continuator has halted unexpectedly` / `cancelled`）。
- **D 跑不了 TPOT 768k bs256**（2026-11-05）：D 的 worker 共享内存涨到每个 150 GB 以上，先返回 HTTP 500，随后 worker 被 OOM killer 杀、整个 D 退出。
  长时间测试后 P 的可用内存也会从约 140 GB 降到约 74 GB。要跑这一格需要 PIE Python + 回收，或更大内存的机器。
- 进程被杀时 `/dev/shm/psm_*` 不会回收，`/dev/shm/torch_tpu_cache` 只增不减，都直接占主机内存；`stop_vllm_local.sh` 会清。

**磁盘与路径**

- **根分区被编译缓存写满**（原部署 TorchInductor + vLLM torch_compile_cache 一度 484G，新主机根分区只有 8.7 GB）：编译时报 ENOSPC 退出。
  `TMPDIR`、`VLLM_CACHE_ROOT` 放 `/dev/shm`；stop 脚本会在根分区超过 75% 时清缓存。
- 根分区满时 /tmp 也写不了。

**进程与端口**

- **pkill / pgrep 会匹配到自己的 shell**：停进程的匹配模式只写在 `stop_vllm_local.sh` 文件里；手动查进程用 `pgrep -f "[t]oy_proxy_server.py"` 这种写法；
  调用 `stop_vllm_local.sh` 的命令行里不要出现它匹配的字样。
- **`Failed to acquire a TPU device node '/dev/vfio/N'`**、**D 的 gRPC / Raiden 端口被占用**：上一轮的 `VLLM::Worker` 残留进程还在。一定要用 `restart_epic.sh`（会清掉残留 worker、等 `/dev/vfio` 释放）。
- **建 DP 组时 `DistBackendError: File name too long`**：GCE 主机名正好 64 字符，gloo 用 `HOST_NAME_MAX` 大小的缓冲读 `gethostname()`，放不下结尾的 `\0`。
  `GLOO_SOCKET_IFNAME` 不影响 vLLM 这条 `ProcessGroupGloo(store, rank, size)` 路径；只能改短主机名（2.4 节）。
- ssh 远程起后台进程时用 `ssh -n -f ... > /dev/null 2>&1`，否则管道会卡住。

**测试**

- `wait_ready.sh` 以 HTTP 状态码判断就绪（`curl -w ok` 会误判）。
- 首次遇到新的形状（batch / 长度）时会临时编译，那一轮的数字会偏大；要先预热，或者重跑确认。
- TPOT 512k bs256 偶发 proxy→P 的 `httpx.ReadError`（500），根因未查清，`tpot_bench.py` 会自动重试 3 次。
- **测吞吐时新 token 命中缓存会让数字虚高**（曾出现一轮 230 万 tok/s）：同一个服务上重复跑，如果请求内容和之前一样，P 的前缀缓存会直接命中。要核对每轮的 `cache_hit_tokens`（见 4.3）。
- 在 sjd1 上直接跑 JAX/Pallas 脚本时，要先设好 TPU 环境变量（`/zzlfs/temu/tools/sc_repro/env.sh`，和启动脚本里的一样），否则 JAX 初始化会卡住。

---

## 8. 下一步

512k+32k 1P 当前的关键路径（P 端登记、trim 修复后；TTFT 约 3.39–3.45s，其中 P-hop 约 2.76–2.82s）：

| 段 | 耗时 | 计划 |
|---|---|---|
| proxy 分词 | 130ms | 分词缓存（已实现，默认关） |
| D 接收请求（proxy 发出 → D 调度器） | ~195ms | 根治办法是让 proxy 在 P 计算的同时就把请求发给 D（需要提前确定 P 的 request id、uuid、DP rank，改动大） |
| D 调度器 → 传输提交完成 | ~64ms | 其中接收计划 5ms、RPC 35ms |
| 传输 | 35–66ms | — |
| KV 到齐 → 第一个 token | ~173ms | 检测→调度→前向 53ms + draft 42ms，待分析 |

其他候选：4P 对齐窗口的效果验证；动态投机 token 数；空步剩余约 20ms；共享前缀注意力 kernel；venv 换 PIE Python 以打开编译内存回收。

---

## 附录 A. 提交列表（在 epic/dsv41 6a1ebddb7 之上）

| 提交 | 内容 |
|---|---|
| 92b47e58c … 2f41443be | 旧 PD 分支的 9 个 cherry-pick（Stage-3 Raiden PD 基础） |
| b4d7bf2af | connector：D 保留前缀缓存，只拉命中之后的部分（per-tag `dst_skip_bytes`，`skips_prefix_replay`）；拉取失败按请求上报（failed_recving），不再杀引擎 |
| 0a0e11633 | producer 只构建 draft 上下文 |
| 6b6565b38 | halfplane Index-K 以 512 字节 owner 行为单位传输（pool_manifest）；去掉"禁止 kv_transfer"的检查 |
| dd89c8b2d | cache.py 补 `from vllm_torchtpu import envs` |
| f1b67377b | async scheduling 下 producer 保留 draft 提议（否则断言失败） |
| 4250476f6 | 空闲 EP-DP rank 复用全零 dummy block table（bs1 TPOT 减半）；`TPU_DUMMY_STEP_TIMING` 计时 |
| 32842f3d7 | `scripts/dsv41_pd/`：部署、venv 补丁、benchmark 脚本 |
| a233edf04 | Stage-3 DSv4：P 端 KV 登记加速（span 缓存 + 预建 wire 元组）、sliding-window trim 加速；诊断计时与 `TPU_STAGE3_PROFILE` 开关（默认关） |
| d8e22db82 | 加载权重时回收 host 内存（`_device_sync` 里 sync + `gc.collect()` + `malloc_trim`，`flush_all` 后清空 pool）；没有这项时加载阶段 host 内存可能不够 |
| 2d5d94f28 / b20ea9162 / 58dce5ae1 | 本文档、`pd_timeline.py`、proxy 对齐窗口 / 分词缓存、`restart_epic.sh` 透传诊断变量 |
| 6cb686ef4 | `tps_bench.py` 和 TPS 测试方法 / 结果 |
| 1dc6984ac | 新主机的已知问题；修正 `TPU_RECLAIM_COMPILE_MEMORY` 注释 |
| 2289534bc | 主机 / 地址可用环境变量覆盖（默认值不变）；新主机复测 1P / 4P、TPOT |
| （本次）reorganize REPRODUCE.zh.md; new-host 8P and TPS | 重新整理本文档结构；新主机 8P、TPS 结果 |

---

## 附录 B. 旧主机结果（2026-10-28/29，2z6z / clsk，Temu 为 10 月数据）

### B.1 Prefill / TTFT（秒）

1P：`results/table-epic-1028-2309`；4P/8P：`results/table-4p8p-1029-0059`；16P（前 5 个场景）：`results/table-16p-1029-0333`。

| 场景 | GCP 1P | Temu 1P | 1P 倍数 | GCP 4P | Temu 4P | 4P 倍数 | GCP 8P | GCP 16P |
|---|---|---|---|---|---|---|---|---|
| 1k+1k | 0.213 / 0.317 | 0.039 / 0.090 | 5.5× / 3.5× | 0.161 / 0.383 | 0.060 / 0.140 | 2.7× / 2.7× | 0.185 / 0.406 | 0.219 / 0.438 |
| 8k+4k | 0.419 / 0.583 | 0.171 / 0.258 | 2.5× / 2.3× | 0.383 / 0.651 | 0.173 / 0.520 | 2.2× / 1.3× | 0.469 / 0.744 | 0.500 / 0.835 |
| 32k+16k | 1.069 / 1.286 | 0.171 / 0.289 | 6.3× / 4.4× | 0.978 / 1.280 | 0.182 / 0.596 | 5.4× / 2.1× | 0.991 / 1.358 | 1.132 / 1.831 |
| 64k+32k | 1.912 / 2.240 | 0.179 / 0.341 | 10.7× / 6.6× | 1.770 / 2.182 | 0.179 / 0.759 | 9.9× / 2.9× | 1.795 / 2.253 | 2.039 / 3.299 |
| 168k+32k | 2.069 / 2.504 | 0.185 / 0.407 | 11.2× / 6.2× | 1.926 / 2.496 | 0.185 / 0.827 | 10.4× / 3.0× | 2.071 / 2.805 | 3.139 / 4.915 |
| 32k+168k | 9.172 / 9.893 | 0.892 / 1.109 | 10.3× / 8.9× | 8.372 / 9.296 | 0.915 / 3.471 | 9.1× / 2.7× | 8.650 / 9.692 | |
| 64k+64k | 3.595 / 4.032 | 0.322 / 0.505 | 11.2× / 8.0× | 3.298 / 3.801 | 0.323 / 1.334 | 10.2× / 2.8× | 3.382 / 3.975 | |
| 256k+32k | 2.189 / 2.784 | 0.189 / 0.459 | 11.6× / 6.1× | 2.069 / 3.008 | 0.189 / 0.882 | 10.9× / 3.4× | 2.338 / 3.286 | |
| 512k+32k | 2.628 / 3.508 | 0.205 / 0.609 | 12.8× / 5.8× | 2.858 / 4.259 | 0.206 / 1.055 | 13.9× / 4.0× | 2.896 / 4.650 | |
| 768k+32k | 2.997 / 4.227 | 0.225 / 0.763 | 13.3× / 5.5× | 3.401 / 5.261 | 0.225 / 1.272 | 15.1× / 4.1× | 3.472 / 5.688 | |
| 32k+768k | 48.466 / 50.967 | 4.618 / 5.151 | 10.5× / 9.9× | 45.273 / 48.958 | 4.746 / 16.275 | 9.5× / 3.0× | 45.741 / 49.432 | |

单点（2026-10-29，含反分词器修复与空闲 rank 修复）：512k+32k 1P TTFT 约 3.40–3.45s。

### B.2 TPOT（ms，CLI / Server；`results/tpot-final-1029.md`）

| bs | 引擎 | 1k | 16k | 32k | 64k | 128k | 256k | 512k | 768k |
|---|---|---|---|---|---|---|---|---|---|
| 1 | GCP | 5.24/5.29 | 4.87/4.74 | 4.54/4.35 | 5.74/5.63 | 5.27/5.22 | 6.30/6.20 | 7.96/8.06 | 7.68/7.54 |
| 1 | Temu | 2.81 | 2.77 | 2.76 | 2.76 | 2.80 | 2.76 | 2.80 | 2.81 |
| 16 | GCP | 5.24/5.17 | 5.06/4.82 | 5.33/5.10 | 5.67/5.43 | 5.80/5.69 | 6.15/5.89 | 7.62/7.51 | 8.57/8.75 |
| 16 | Temu | 3.52 | 3.51 | 3.50 | 3.56 | 3.52 | 3.56 | 3.62 | 3.72 |
| 64 | GCP | 7.21/7.05 | 8.16/8.05 | 9.49/9.21 | 10.57/10.42 | 8.36/8.13 | 10.40/10.09 | 14.10/13.92 | 19.50/19.22 |
| 64 | Temu | 5.39 | 5.41 | 5.40 | 5.41 | 5.41 | 5.39 | 5.59 | 5.81 |
| 128 | GCP | 9.66/9.44 | 12.50/12.30 | 14.70/14.29 | 18.66/18.31 | 14.43/13.99 | 21.54/21.19 | 31.32/30.11 | 45.11/44.69 |
| 128 | Temu | 6.60 | 6.64 | 6.61 | 6.65 | 6.70 | 6.82 | 7.16 | 7.53 |
| 256 | GCP | 14.51/13.97 | 12.82/12.65* | 15.68/14.78 | 16.34/15.84 | 18.31/17.51 | 23.63/22.83 | 21.72/21.35* | 27.87/27.18* |
| 256 | Temu | 9.56 | 9.60 | 9.58 | 9.61 | 9.92 | 10.05 | 10.69 | 11.28 |

\* 用 out_len 3072（out_len 1024 时拉满窗口太短）。总体是 Temu 的 1.3–6.0 倍。

### B.3 P 的 prefill 吞吐上限（tok/s；sjd1，`results/tps-sjd1-1029-1600.jsonl`）

32k 缓存前缀，每组 2 轮正式。整机 = 4 芯片 / 8 个 rank。

| 组 | 场景 | 不开 async | 开 async |
|---|---|---|---|
| 新 token 长度扫描（8 并发） | 32k | 15.8–16.1 万 | — |
| | **64k** | **17.7–17.8 万** | 16.9 万 |
| | 168k | 17.3 万 | 17.4 万 |
| | 256k | 17.2 万 | — |
| 持续满载（168k 新 token） | 16 并发（每个 rank 2 条） | 17.0–17.4 万 | 17.2–17.4 万 |
| | 24 并发（每个 rank 3 条） | 16.8–17.0 万 | — |

- 上限约为整机 17.5 万 tok/s，每芯片约 4.4 万 tok/s，条件是 8 个 rank 各有一条长请求（新 token ≥64k、缓存上下文中等长度）。
- 新 token 超过 64k 不再提升；每个 rank 多排请求没有用（每个 rank 每步的预算固定是 2048 个 token）；P 开 async 对吞吐没有帮助。
- 单请求（1P）时只有一个 rank 在算，约 1.9 万 tok/s。
- profile（32k+168k 1P，每步约 105ms 设备时间）：MoE（20 个 encoder 层）44.5ms（42%），CSA 注意力 12.8ms，indexer top-k 8.7ms，量化融合 6.4ms，all-gather 5.1ms；host 侧空隙约 4%。
