# PP 调度 CPU 回放：4K 与 1K bucket

这个例子用固定的 16 条请求，检查每次非空调度提交包含哪些请求、
每条请求分配多少 token，以及 checkpoint 对齐如何产生短尾部和剩余预算。
它执行保存的 vLLM 调度函数和本分支的 TPU checkpoint 包装函数，
只需要 **Python 3.12 标准库**，不需要安装 vLLM、torch 或模型依赖。

这里的“批次”指一次 scheduler 提交，不是设备 profiler 测得的 forward 次数。
测试不执行 TPU，不计算真实吞吐或 TTFT，也不包含 MTP drafter 的设备调用。

## 如何运行

```bash
git clone --single-branch --branch handsomehow/pp_schedule_replay \
  https://github.com/vllm-project/vllm-torchtpu.git
cd vllm-torchtpu

# 验证固定输入、checkpoint 边界和已记录的逐批分配结果。
python3.12 -B experiments/cpu_bucket_replay/test_replay.py

# 运行全部六组对照，打印汇总并重新生成 results/ 中的明细。
python3.12 -B experiments/cpu_bucket_replay/replay.py
```

无需创建虚拟环境、设置环境变量或安装软件包。脚本通过自身位置定位文件，
也可以从其他工作目录使用脚本的绝对路径运行。
不要使用 `python -O`，测试通过断言检查 token 守恒和预算约束。

## 比较的配置

两组共用 PP4、PCP8、`max_num_seqs=64`、关闭动态 chunk、
`long_prefill_token_threshold=0`。保留 MTP1 的 checkpoint 规则，
关闭 EAGLE block drop；物理 block 为 384，调度/hash 单元为 3,072。

| 配置 | 每个 PCP rank 的 compile sizes | 全局每轮 token 预算 |
| --- | --- | ---: |
| 4K | `[4096]` | 32,768 |
| 当前 1K | `[256, 512, 1024]` | 8,192 |

CPU 调度直接使用的是全局预算。`compile_sizes` 在这里仅记录对应的编译配置，
不会参与 `schedule()`；只改 compile sizes、不改全局预算，分配结果不变。
本例比较的是 **bucket 与对应 token 预算一起调整**的效果。

动态 chunk 关闭时使用普通 Scheduler，不启用 PP 动态成本和自适应请求限额。
当前配置启用的 lazy push 属于执行器行为，本例的 FIFO 完成模型不模拟它。

## 16 条请求

输入来自一次 dummy-load 服务测试的匿名长度和相对到达时刻，
不是用户原始文本或 token 内容。缓存命中长度固定为当次记录的值，
因此“新算”包括当时因缓存边界而需要重新计算的部分。
编号沿用采集记录，16 条请求中没有 r5。

| 请求 | 输入总长度 | 固定缓存命中 | 实际新算 |
| --- | ---: | ---: | ---: |
| r1 | 49,391 | 39,936 | 9,455 |
| r2 | 49,391 | 39,936 | 9,455 |
| r3 | 93,027 | 79,872 | 13,155 |
| r4 | 93,027 | 79,872 | 13,155 |
| r6 | 49,391 | 39,936 | 9,455 |
| r7 | 93,027 | 82,944 | 10,083 |
| r8 | 93,027 | 82,944 | 10,083 |
| r9 | 348,891 | 307,200 | 41,691 |
| r10 | 49,391 | 43,008 | 6,383 |
| r11 | 49,391 | 43,008 | 6,383 |
| r12 | 49,391 | 43,008 | 6,383 |
| r13 | 49,391 | 43,008 | 6,383 |
| r14 | 93,027 | 82,944 | 10,083 |
| r15 | 93,027 | 82,944 | 10,083 |
| r16 | 93,027 | 82,944 | 10,083 |
| r17 | 172,036 | 153,600 | 18,436 |
| 合计 | 1,517,853 | 1,327,104 | 190,749 |

平均每条新算 11,921.8 tokens。这是一组混合历史长度，
并非每条都有相同的历史长度或相同的新增长度。

## 到达与完成模型

所有场景最多保留四个在途调度批次，按 FIFO 消费模拟结果。
请求的最终 prefill 批次返回后，回放程序将请求移除。
不模拟真实输出采样、PD 传输完成、KV 释放或完整服务请求生命周期。

- `closed_loop_c4`：先放入四条请求，每完成一条就按输入顺序补入下一条。
  每批模拟服务间隔为 1，调度 tick 为 0.001；这些是任意时间单位，
  输出 JSON 中的 `synthetic_submit_ms` 不能当作真实性能数据。
- `historical_arrivals`：固定历史到达时刻，用历史结果消费时刻推导的
  FIFO 间隔，调度 tick 为 1 ms。间隔序列用尽后重复最后一个间隔。
  结果消费不是设备完成；这里也不会因策略变化重建闭环客户端的到达流。
- `all_ready_16`：16 条同时入队，使用与第一组相同的模拟完成规则，
  用于观察候选请求充足时能否合批。

所有场景均假设缓存到达即就绪，忽略真实 cache 查询及搬运等待。
同一组长度在不同到达和完成节奏下，批次组合可能不同。

## 运行结果

下表是 Python 3.12 下的 CPU 结果。六组均完成 16 条请求，
分配总量均为 190,749 tokens。

| 场景 | bucket | 调度批数 | 请求 chunk 数 | 平均有效 tokens/批 | 预算填充率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 闭环 C4 | 4K | 16 | 33 | 11,921.81 | 36.38% |
| 闭环 C4 | 1K | 32 | 51 | 5,960.91 | 72.76% |
| 历史到达 | 4K | 28 | 33 | 6,812.46 | 20.79% |
| 历史到达 | 1K | 33 | 55 | 5,780.27 | 70.56% |
| 16 条同时就绪 | 4K | 7 | 34 | 27,249.86 | 83.16% |
| 16 条同时就绪 | 1K | 31 | 51 | 6,153.19 | 75.11% |

请求 chunk 数是各请求分块数之和，一批可以容纳多个请求 chunk。
预算填充率是有效 tokens /（批数 × 全局预算），不是实测设备利用率，
也不是考虑每个 rank 的 padding 后得到的编译 bucket 利用率。

闭环 C4 的前六批如下，所有 token 数均为 PCP8 全局有效数量。

| 批次 | 4K 分配 | 合计 | 1K 分配 | 合计 |
| ---: | --- | ---: | --- | ---: |
| 1 | r1:9216, r2:9216, r3:12288 | 30,720 | r1:6144 | 6,144 |
| 2 | r1:239, r2:239, r3:867, r4:12288 | 13,633 | r1:3072, r2:3072 | 6,144 |
| 3 | r4:867 | 867 | r1:239, r2:6144 | 6,383 |
| 4 | r6:9216, r7:9216, r8:9216 | 27,648 | r2:239, r3:6144 | 6,383 |
| 5 | r6:239, r7:867, r8:867 | 1,973 | r3:6144 | 6,144 |
| 6 | r9:30720 | 30,720 | r3:867, r4:6144 | 7,011 |

完整逐批和逐请求分配：

- 闭环 C4：[4K](results/closed_loop_c4-4096.txt)、
  [1K](results/closed_loop_c4-1024.txt)。
- 历史到达：[4K](results/historical_arrivals-4096.txt)、
  [1K](results/historical_arrivals-1024.txt)。
- 16 条同时就绪：[4K](results/all_ready_16-4096.txt)、
  [1K](results/all_ready_16-1024.txt)。
- [comparison.json](results/comparison.json) 包含每批队列、分配和切分调用，
  `offered` 是进入切分函数前的额度，`allocated` 是对齐后的分配量。

## 这个例子说明什么

1. 1K 对应 8,192 的全局预算。长请求受 3,072 对齐约束，
   常只能获得 6,144 tokens；剩余 2,048 不够下一条长请求的一个对齐块。
   短尾部仍可能装入剩余预算。
2. waiting 队列遇到对齐后为零的请求，会停止本轮 admission；
   running 队列对零分配使用 `continue`。回放保留了这段源码行为。
3. 4K 能处理更大的 chunk，但仍保留尾部 checkpoint。
   闭环 C4 的 r17 在 4K 下为 `18432 + 4`，
   在 1K 下为 `3072 + 6144 + 6144 + 3072 + 4`。
   后者来自预算和对齐，不是旧的 MTP 额外回退一个 hash 单元。
4. 这组部分尾部请求在当前修复下，打开或关闭 MTP checkpoint 规则，
   闭环 C4 的分配相同。这不代表 MTP 在设备上没有额外开销。

批数减少不等于吞吐提高。本例没有测量不同 bucket 的执行耗时，
也没有模拟 lazy push，因此不能凭这些结果推断实际吞吐或 TTFT。

## 源码、替身和验证边界

- vLLM 调度源码快照提交：`2ba984a5d06db414f3b2474fe9338faf6cd80a1c`，
  路径为 `vllm/v1/core/sched/scheduler.py`，已完整附带在
  [inputs/vllm_scheduler.py.txt](inputs/vllm_scheduler.py.txt)。
  保留原许可证头，以 `.txt` 保存避免工具重格式化固定快照。
  不需要另外 checkout 该 vLLM 提交。
- vllm-torchtpu 基础提交：`194a765fb77ae162cde2289831477a7a9ee40f7e`。
  checkpoint 包装函数直接取自本 checkout 的 `src/vllm_torchtpu/`。
- 通过 AST 提取并原样执行 `schedule()`、token 记账、lookahead 和对齐函数，
  不导入完整 vLLM。构造 Scheduler 的初始化状态，使用无限容量 KV 替身
  和输出元数据替身；不执行真实混合 Mamba cache manager。
- 缓存长度固定，不模拟缓存变热、容量抢占、真实 hybrid 状态、
  connector 等待或 PD 搬运。未采集的 shared-prefix boundary 设为零。
- [inputs/provenance.json](inputs/provenance.json) 记录输入文件 SHA256；
  主脚本运行时检查固定输入，防止不知情地换了测试数据。
- 这是独立实验，不自动接入仓库的设备 CI。九项 unittest 覆盖输入来源、
  配置、checkpoint、预算与 token 守恒、MTP 规则开关，以及六组结果重现；
  ruff 检查通过。没有修改生产调度实现。

运行本目录的测试时使用 `python -S` 禁用 site-packages，也能得到相同结果。
仓库全局 pre-commit 的 pyrefly 检查在基础提交的 `src/` 中报告 14 个
已有类型错误；本次没有修改 `src/`、类型检查配置或 baseline，
不能将本目录测试通过解读为全仓库类型检查通过。
