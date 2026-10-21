# GLM-5.3 TPU 优化说明

## TL;DR

本提交汇总已采用的 Indexer cache 局部更新、Native MLA 桶调参、RoPE 窄视图
和异步 logprob D2H。基线为 `ab6a92e61584c53d5046c95fac4359118977d094`。
TP8、7 层 scaled dummy、LP20 实验中，kernel 组合使长输入 e2e 延迟下降
38.58%–58.77%；异步 D2H 在组合版上进一步下降 28.07%–46.10%。
两组为不同时段的独立配对，百分比不能相加，也不代表完整模型收益。

后续研究方向见 [GLM-5.3 与 TPU v7 优化分析方案](glm53-tpu-v7-analysis.md)。
该方案区分源码事实、历史证据和待验证假设，不包含新的性能验收结论。

## 改动与边界

- **Indexer cache**：Pallas 对目标四 token 的 packed group 局部读改写，
  保持原 uint8 raw4 布局及未更新的邻居字节，避免 XLA 重排整个 cache 池。
  仅在 1024-token 桶、cache 尾部 shape 为 `(256, 4, 256)`、页数至少
  512 时启用；其他 geometry 继续原 scatter。这是实测性能策略。
  依赖既有调度契约：同一请求的新 token 连续排列，物理写入位置唯一。
- **Native MLA**：符合 BF16、topk=2048、page=1024、head_dim=576
  的 H8/H32/H64 geometry 时，16-token 桶的 attention batch 从 16 改为 1。
  H64 的 1024-token 桶另将 gather chunk 从 64 改为 128；TP8/H8 不命中
  后一项。16-token 桶也可能包含短 prefill 或 mixed batch。
- **RoPE**：旋转前对 `q_pe` 调用 `contiguous()`，让编译器抽取窄 RoPE
  切片，减少完整 Q 的布局转换；旋转算术不变。
- **异步 logprob D2H**：复用 `AsyncTPUCopyState` 和上游
  `LogprobsTensors.to_cpu_nonblocking()`；先提交完整 padded tensor 的复制，
  在覆盖全部复制的 event 完成后，才在 CPU 裁剪、拼接和转为输出列表。
  保留源 tensor 至 event 完成，维持请求行序及未请求 logprob 时的输出语义。

## 性能与验收范围

TPU7x TP8；输入长度/batch 如下，输出均为 16 token，返回 top20 logprob。
每臂每项预热 2 次、计时 10 次，profile 另采；保留全部样本。
降幅按 `(对照平均延迟 - 候选平均延迟) / 对照平均延迟` 计算。

| 输入 / batch | main → kernel 组合 | 新鲜组合版 → 异步 D2H |
| --- | --- | --- |
| 256 / 1 | 0.02% | 46.10% |
| 1024 / 1 | 38.58% | 40.25% |
| 2049 / 1 | 47.97% | 34.88% |
| 4096 / 1 | 58.77% | 28.07% |
| 1024 / 4 | 58.11% | 33.20% |

这些配对的生成 token、top20 成员和值零容差一致；单项 kernel 在 TP8
的独立贡献尚未完全拆分。后续长时间 B4/HTTP 复验仍出现数值变体，main
也有复现，根因未闭环，因此不能宣称 B4 全量稳定性验收完成。
valid-query gather、K-tail、collective offload 等未采用实验不在本提交内。

## 验证与复现

固定依赖：vLLM 0.29.0、Torch 2.13.0+cpu、
TorchTPU 0.1.1.dev20260912075415、JAX/jaxlib 0.10.2、libtpu 0.0.47、
transformers 5.10.4。性能模型为前 7 层 scaled dummy，不下载权重；
BF16/FP8、max_model_len=4112、max_num_batched_tokens=1024、
max_num_seqs=4、seed=42、异步调度、禁用 prefix cache。

在已安装上述依赖的环境，从本仓库根目录运行长期测试：

```bash
# CPU：验证复制完成顺序、源 tensor 保活、padding 裁剪和请求行序。
env -u VLLM_PLUGINS TPU_SKIP_MDS_QUERY=true JAX_PLATFORMS=cpu \
  PJRT_DEVICE=CPU PYTHONPATH=src \
  python -m pytest -q tests/runner/test_async_logprob_output.py
# TPU 空闲时：逐字节检查 packed group 邻居、跨页和全 padding。
TPU_SKIP_MDS_QUERY=true JAX_PLATFORMS=tpu PYTHONPATH=src \
  python -m pytest -q tests/kernels/test_indexer_cache_update.py
```

性能 harness 和原始证据位于配套 `tpu-misc` 工作区的
`results/glm53-kernel-opt-20260919/`，不随本提交纳入缓存或 trace。
在该归档可用、设备空闲的环境中，使用干净 shell，从本仓库根目录复跑：

```bash
GLM_RESULT=/mnt/data/tpu-misc/results/glm53-kernel-opt-20260919
GLM_HARNESS="$GLM_RESULT/environment/profile-no-regions-matrix-harness"
GLM_HARNESS="$GLM_HARNESS/benchmarks/glm53"
GLM_OUT="$GLM_RESULT/repro-$(date +%Y%m%d-%H%M%S)"
TPU_SKIP_MDS_QUERY=true VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$GLM_HARNESS:$PWD/src" \
  "$GLM_RESULT/venv/bin/python" "$GLM_HARNESS/run_profile_cleanup_e2e.py" \
  --model "$GLM_RESULT/model/dummy-7layers" --output "$GLM_OUT" \
  --tp 8 --dummy-init scaled --warmups 2 --rounds 10 --profile --llo \
  --case 256:1:16 --case 1024:1:16 --case 2049:1:16 \
  --case 4096:1:16 --case 1024:4:16
```

对照臂在相同环境串行执行，切换 `PYTHONPATH` 的产品源码目录并使用新输出目录。
用 `tpu-misc/benchmarks/glm53/compare_runs.py` 比较两臂及各臂自身，传入
`--rtol 0 --atol 0`；全部 token、top20 成员和值应一致，失败样本不得删除。
统计必须包含 avg/min/max/count/stddev/p90/p99。

归档中的 `final-tp8-main-r2` / `final-tp8-combined-r2` 对应 kernel 组合；
`async-logprob-tp8-control-r1` / `async-logprob-tp8-candidate-r1`
对应异步复制。完整统计分别见组合版 `metrics-ms.csv`、异步候选
`latency-statistics.csv`；各 case 的 `samples.jsonl` 和运行 `manifest.json`
保存原始输出、源码 hash 和配置。HTTP 限制见
`http-tp8-async-logprob-normal-r5/failure-analysis.md`。
