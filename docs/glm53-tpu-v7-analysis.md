# GLM-5.3 与 TPU v7 优化分析方案

## TL;DR

截至 2026-09-22，当前 7 层实验应优先减少 Native MLA 的无效及重复 KV
搬运；完整 78 层模型还必须优先处理 MoE 权重容量、复用和并行布局。
TPU v7 的 SparseCore 适合 gather/scatter，TensorCore 负责矩阵计算，
但两者的并行收益受数据依赖、HBM 搬运和 VMEM 分配共同约束。

本文是分析与后续验证方案，基于已提交的
`a5bd5513976ef103d11c4011927f3578e390f6cf` 和既有证据。
本轮暂停测试，不新增实现或设备测量。下列排序表示研究优先级，
不是已经测得的 e2e 收益排名；不能为未验证方案赋予收益百分比。
已采用优化及历史性能见 [优化说明](glm53-optimizations.md)。

## 模型范围与硬件约束

| 项目 | 当前 cropped dummy | 完整配置 |
| --- | --- | --- |
| Transformer 层数 | 7 | 78 |
| Dense / MoE 层数 | 3 / 4 | 3 / 75 |
| 完整 Indexer 层数 | 4：0、1、2、6 | 21：0、1、2、6、10…74 |
| hidden / attention heads | 6144 / 64 | 相同 |
| MLA latent / RoPE 维度 | 512 / 64 | 相同 |
| routed experts / top-k | 256 / 8 | 相同 |
| sparse attention top-k | 2048 | 相同 |

其他 Indexer 层复用上一完整 Indexer 的 indices，各层 KV 并不共享。
7 层模型放大了 Indexer 与固定输出开销的相对占比，低估了 MoE 的占比；
其收益不能直接乘以层数外推。Dummy 权重也不能证明真实权重的模型质量。

TPU v7（Ironwood / TPU7x）每个物理芯片有两个 chiplet，每个 chiplet
包含一个 TensorCore、两个 SparseCore 和 96 GiB HBM。
本实验 TP8 对应四个物理芯片、八个逻辑设备。
芯片总 HBM 192 GiB；不能把整芯片带宽或算力当作单 rank 的可用能力。
来源：[Google TPU7x 架构](https://docs.cloud.google.com/tpu/docs/tpu7x)。

每个 TensorCore 的 VMEM 为 64 MiB；SparseCore 每核有 16 个 subcore，
适合不规则访存。SC 原生 gather 以 32-bit word 工作，低精度数据要考虑
packing 和访问粒度，不能按有效 FP8 字节数直接估计 DMA 成本。
来源：[Pallas 硬件表](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)、
[SparseCore 文档](https://docs.jax.dev/en/latest/pallas/tpu/sparsecore.html)。

VMEM 同时服务 scoped kernel 和后续权重预取，增大 tile 或缓冲深度会改变
资源竞争。现有 `65472 KiB` 配置是上限，不是实测占用。
来源：[Ironwood 性能指南](https://docs.cloud.google.com/tpu/docs/ironwood-performance)。

## 优先级与优化边界

### 1. Native MLA：跳过无效 query 与 K-tail

**已确认事实：** 当前路径按 query 展开 K=2048 的 gather；padding indices
即使分散到不同 cache 地址，也仍然产生搬运。有效 query 前缀候选已有历史实验，
尚未晋升；K-tail 只有准备与合同审查，没有设备编译或性能验收。

现有布局每个 KV occurrence 为 NOPE 512 B + ROPE carrier 128 B。
因此 K=2048 时，逻辑 gather 输出为每 query 1.25 MiB、G64 每 chunk
80 MiB、Q1024 每层每 rank 1.25 GiB。这是展开张量容量，
不是实测 HBM transactions，更不是可直接兑现的延迟降幅。

**方案：** 先按 TC 实际消费范围跳过全无效 query chunk，再按各 query 的
有效 K 跳过完整无效 owner 块。保持未启用时的原路径，并把新增合同限制在
native gather 与直接消费端之间。

**边界与缺口：** 当前 K2048/A16 的 owner 粒度为 1024，K有效长度1024
可跳过后半块，1025 则不能整块跳过；不能把尾长比例当作可跳过比例。
TC 仍可能 DMA 尾部，再在算术前 mask，因此必须审查未写区域的消费合同，
不能只凭 mask 推断设备安全。后续需覆盖零长、1023/1024/1025、整长、
partial query bucket 和旧长上下文。valid-query 的既有 i513 诊断收益
不能覆盖 B4 自身一致性失败，不能视为全矩阵通过。

### 2. Native MLA：复用 KV 与压缩物理布局

**已确认事实：** 相邻短 causal query 大量引用相同 KV，而当前按 query
展开。ROPE 有效内容64 B，carrier128 B；SC 已按每选中 token128 B读取，
不存在已证明的四倍 ROPE 读取放大。

**方案：** 研究短连续前缀跨 query 复用，减少中间 HBM 展开；另行评估
ROPE carrier 压紧。64 B padding 占当前640 B carrier的10%，只是布局
容量机会，不是10%的 e2e 承诺。只压源 cache、保留原输出展开不足以消除
整条链路的开销。

**边界与缺口：** 128 B 是当前布局策略，未证明是硬件硬约束；writer、
SC gather、TC unpack 和容量比例检查必须形成一致的新合同。保持 sparse
attention 的归约和量化语义；先前 masked-dense 数值失败的路径仍不采用。
入口涉及 `native_sc_cache_layout.py`、`native_sc_gather.py` 和
`kernel.py`，均位于 `src/vllm_torchtpu/kernels/mla/sparse/`。

### 3. 完整模型 MoE：容量、路由搬运与并行布局

**已确认事实：** 当前 TP8 每 rank 仍持有256 experts，intermediate2048
切为256；W13 为 `[256,6144,512]`，W2 为 `[256,256,6144]`。
单层 routed FP8 矩阵载荷为
`256 × 3 × 6144 × 2048 bytes = 9 GiB`，75层均匀 TP8 后为
84.375 GiB/rank，尚未计 scales、shared experts、attention 和 KV。
这是权重容量，不能据此声称每个 decode token 都读取所有 experts。

路由均匀时，N1024/top8/256 experts 的理论平均为32 tokens/expert；
真实负载分布尚未测量。现有 MoE 先构造 expert-major 输入、执行两次 GMM，
再重排和 combine。`[1024×8,6144]` BF16 张量逻辑容量为96 MiB，
是否全部物化须以 HLO/LLO 为准。

**方案：** 优先审查 GMM 加载时按 token indices 取输入、输出端局部 combine
的可能性，并参考 vLLM 已有 sequence-parallel MoE 与 TPU 的
`pipelined_fused_moe.py`。完整模型另评估较低 TP 配合 EP 的容量、算术密度
与跨 chiplet 通信取舍；不能预设 EP 必然更快。

**边界与缺口：** 维持量化粒度、route 顺序和归约舍入。已有 GMM 包含 lhs
双缓冲、rhs 三缓冲、output 双缓冲，不能把增加缓冲当作尚未实现的优化。
旧 padding 开关、tiling 扫描的失败不撤销。大 batch 提升吞吐不等于改善
B1 延迟；小 decode 与大 prefill 需要分别判断。

### 4. SC / TC 与权重预取资源平衡

**已确认事实：** 现有 i1024 第一层16个 gather update 都在首个 TC attention
之前结束，已经有跨 chunk 提前发起；不是完全串行流水。
同一 TC 时钟上 chunk1/2 的 call-done 约28.491/34.590微秒，
包括等待和簿记，不能解释为纯 SC 计算时间。

**方案：** 在有效搬运量减少后，审查在途深度、chunk 大小、TC scoped VMEM
与权重预取的平衡。代表性 Native B16 静态 scoped 空间约43.5 MiB；
其中名为 SPILL 的分配不能单独证明发生 HBM spilling。

**边界与缺口：** 历史 A32 比 A16 慢约4%–7%，B8 比 B16 慢约0.54%–1.18%；
减小 scratch 不自动产生收益。SC 实际执行时间和完整硬件计数仍缺失，
不宣称带宽饱和、perfect overlap 或全 kernel 最优。

### 5. 长上下文 Indexer 与重复激活量化

**已确认事实：** Indexer 投影使用 replicated linear，TP8 每 rank仍32 heads；
短上下文已有提前返回，长上下文才进入 score/top-k。当前最多约4K的实验
不足以确定长上下文优先级。

**方案：** 先参考已有 DCP/PCP 能力，分析分片 KV 上的 local top-k 与全局
合并；单 rank计算后广播并不自动缩短拥有者的关键路径。另外审查
`q_b_proj(q_c)` 与 `indexer.wq_b(q_c)` 的共同输入量化是否已被编译器复用。
若 HLO 已共享 absmax/scale/FP8转换，就停止该方向。

**边界与缺口：** 完整 Indexer 层才有共同输入机会；改变 dot 形状、权重布局
或量化实现仍需精确数值验证。不能将原 checkpoint 的 block128×128 配置
当成所有运行时 linear 的实际量化方式。

### 6. 通信与输出路径

**已确认事实：** 已接受版本的 i1024、7层、TP8、LP20 decode 中，扣除入口
barrier 后 forward 约2.592–2.594 ms；每步15次 all-reduce 的区间并集约
0.263–0.267 ms，LM 约0.139 ms，logprob gather 约0.108 ms。
这些不是可相加的 e2e 节省上限，也不能把 barrier 全部归因于某个慢 rank。

**方案：** 先分析 activation 分片边界，避免重复 norm/router/搬运，再考虑
collective 放置。SC collective 会与 gather 争用资源，历史 offload 阈值
实验未被采用。分布式采样需覆盖全局归一化、rank、ties 和 RNG 语义；
已有异步 D2H 优化不能代替这些合同。

## 后续验证顺序与停止条件

当前仅发布方案；以下步骤在恢复测试后执行。

1. 固定 accepted 源码、依赖、模型配置及 baseline；先修正旧 harness
   对基线 HEAD 的硬编码，不因 HEAD 改变而放宽产品源码 hash 校验。
2. 对单一候选检查入口、数据所有权、消费边界及最小反例；新增状态若泄漏到
   多个原有模块，先重新设计边界，不堆叠兜底分支。
3. 先逐字节 kernel/消费合同，再做正式 paired e2e；保留 padding、K-tail、
   长上下文、B1/B4 的失败样本。B4 既有变体尚未闭环，不豁免自身一致性 gate。
4. 每臂多次重复，报告 count/min/max/avg/stddev/p90/p99；profile 与计时
   分开，绑定实际 program、shape、源码 hash 与 final LLO。
5. 只有数值合同、重复性能和关键路径证据同时支持，才晋升候选。
   没有收益或已被编译器优化掉的方向及时停止；保留反证记录。

所有未来 TPU 启动必须设置 `TPU_SKIP_MDS_QUERY=true`，设备空闲时串行运行。
本轮没有启动 CPU/TPU 测试、benchmark 或服务。

## 证据定位与只读复核

产品源码固定在上述 commit。配套归档根目录为
`/mnt/data/tpu-misc/results/glm53-kernel-opt-20260919`，未随 Git 上传。
没有该归档的新环境只能复核公开硬件资料与产品源码，不能独立重现历史 trace
数字；需要取得原始工件和对应 manifest 后才能复核这些数字。

| 结论 | 归档内证据 |
| --- | --- |
| 完整 / cropped 配置 | `model/upstream/config.json`、`model/dummy-7layers/config.json` |
| 当前 TP8 geometry / LLO 绑定 | `environment/native-mla-h8-current-macro-contract.json` |
| 提前发起与消费等待 | `environment/native-mla-cross-chunk-audit.md` 及同名 JSON |
| decode、通信、输出开销 | `environment/async-logprob-tp8-trace-audit.md` 及同名 JSON |
| valid-query 未晋升原因 | `native-valid-gather-optimization.md` |
| K-tail 合同与待验证内容 | `environment/native-k-tail-plan-r1.md` |
| B4 历史失败 | `http-tp8-async-logprob-normal-r5/failure-analysis.md` |

以下仅读取版本、源码与既有报告，不执行模型或测试；在产品仓库根目录使用：

```bash
git show --stat a5bd5513976ef103d11c4011927f3578e390f6cf
rg -n 'num_valid_indices|sparsecore_gather|batch_end|kv_lens' \
  src/vllm_torchtpu/kernels/mla/sparse/kernel.py
rg -n '128|batches_per_query' \
  src/vllm_torchtpu/kernels/mla/sparse/native_sc_gather.py
GLM_RESULT=/mnt/data/tpu-misc/results/glm53-kernel-opt-20260919
cat "$GLM_RESULT/environment/native-mla-cross-chunk-audit.md"
cat "$GLM_RESULT/environment/async-logprob-tp8-trace-audit.md"
```

复核时区分事实、结构推断与待验证收益；动态耗时只在同一时钟域解释。
旧报告中的服务存活状态是历史快照，不代表当前服务在线。
