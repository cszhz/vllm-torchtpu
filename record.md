
配置沿用你成功的实验：v7x 单设备、48×6=288 tokens、2048 个候选块×8 records、topk=512。

| 路径 | 设备耗时 |
|---|---:|
| gather | 0.616 ms |
| logits | 0.634 ms |
| topk | 0.243 ms |
| logits + topk 流水线 | 0.752 ms |
| gather + logits + topk | 1.339 ms |
| 再包含 metadata 准备 | 1.583 ms |

设备时间取 5 次 XPlane 中位数，host 计时取预热后 100 次中位数。

已修复 queries 参数缺失及 Pallas gather/VMEM 对齐问题；19 个回归测试、benchmark 正确性、ruff、pyrefly 全部通过。

结果与复现文件 (.dsv41_bench/sparse-layer-20261028/summary.json) · 实验 exp-q5er027vy7
