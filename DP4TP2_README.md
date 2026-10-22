# DP4×TP2 decode for Qwen3.5-397B P/D (2026-10-22)

This branch is the frozen tree behind the DP4×TP2 serving points on the IX dashboards (sheet `ix-tpu-pd-dp4tp2-oct22`). It is o6q (`69c56a021`, prefill v2) plus two engine fixes and a bench commit:

| Commit | What |
|---|---|
| `cf54717bc` | Greedy top tokens under EP-DP with a TP2 head: the target keeps full logits instead of raising; the draft head keeps the vocab-parallel argmax. Tests included. |
| `2564cf137` | `tpu_communicator`: a replicated (below `min_rows`) sequence-parallel MoE step under DP>1 gathers over DP and combines as DP reduce-scatter + TP all-reduce. Before this, it mixed the DP engines' rows (wrong output). Tests included. |
| `bd10e8d5a`, `b44438a09` | Bench harness only (`p3_d_leg_bench.py`); no engine change. |

Quality on this tree: GSM8K 97.80% (j-09687ab0), LongBench v2 65.41% (j-180bd829).

## What DP4×TP2 is

Each decode host runs 4 vLLM engines (DP4). Each engine is TP2 on the 2 TensorCores of one chip, so the tensor-parallel all-reduce stays on-chip. The MoE is EP8 across all 8 cores. DCP is 1 (the model has 2 KV heads). The megakernel is off.

## Where it helps

It wins when a decode host runs about 32 or more requests at once, and loses below that.

| Requests per decode host | DP4×TP2 step vs TP8 (microbench) |
|---|---|
| 8 | +56% (slower) |
| 16 | +27% (slower) |
| 32 | −1.7% |
| 64 | −10% |
| 112 | −13% |

Serving, same-shape twins against TP8 + DCP4:

| Shape | TP8 | DP4×TP2 |
|---|---|---|
| Band A, 4P3D cap 112 c560 | 197.2 Mtok/$ @ p90 57.1 (j-2b0b7cd9) | **210.7 @ 69.8** (j-9f655413) |
| 1P1D cap 64 c152 | 200.5 @ 84.4 (j-8fcd357f) | **205.9 @ 88.5** (j-47d903e4); repeat 205.2 @ 88.4 (j-c09cf937) |
| 1P1D cap 16 c48 | 92.2 @ 230.9 (j-527d00b1) | 79.2 @ 166.3 (j-cd9a9823): loses |

## Decode configuration (as served)

Recipe `agentx-397b-pd-baseline`, starting from the TP8 o6q command, with these decode changes:

```
TENSOR_PARALLEL_SIZE_DECODE=2
DATA_PARALLEL_SIZE_DECODE=4
DECODE_CONTEXT_PARALLEL_SIZE_DECODE=1
CP_KV_CACHE_INTERLEAVE_SIZE_DECODE=1
DECODE_VLLM_TPU_QWEN35_MEGAKERNEL=0
DECODE_VLLM_TPU_QWEN35_MTP_MK=0
DECODE_TPU_MOE_SEQUENCE_PARALLEL=1
DECODE_TPU_SP_MIN_ROWS=88
DECODE_TPU_MOE_FUSED_EP_STEP_MIN_TOKENS=44
DECODE_TPU_EXPERT_PLACEMENT=/workspace/vllm-torchtpu/expert_placement/expert_placement_dp8_ep8.json
MAX_NUM_SEQS_DECODE=<cap per host / 4>          # 16 for cap 64, 28 for cap 112
DECODE_TPU_SCHED_MK_MAX_RUNNING=<same as above>
COMPILE_SIZES_DECODE=8,16,24,32,48,64,72,80,96,112,128,256,512,2048
TPU_TOKEN_BUCKET_EXTRA_DECODE=8,48,2048
KV_CACHE_MEMORY_BYTES_DECODE=39500000000
DECODE_TPU_KV_BYTES_EXPLICIT=1
```

`TPU_RAIDEN_DST_SHARDS` follows the decode TP size (2) automatically on both legs.

**Recipe passthroughs needed** (not on cloud-devkit-recipes master yet; they were in a private clone): `KV_CACHE_MEMORY_BYTES_DECODE` → `--kv-cache-memory-bytes`, and `BENCH_MEMORY_REQUEST` (a memory request on `sidecar-bench`; use 48–64Gi, otherwise the bench can be OOM-killed on hour-long replays).

The expert-placement file is tracked under `scripts/vllm/benchmark/inferencex/qwen3.5/`; the recipe copies it into `expert_placement/` when you use `--map-dir`.

## Known limits

- DP>1 with DCP>1 on a P/D KV consumer raises `NotImplementedError` (the Raiden Stage-3 admission names `dp{N}tp{T}_decode` without a DCP axis).
- Each request's KV lands on 2 cores instead of 8, so KV transfer is 1.7–2.6× slower at the median. Median TTFT is 0.1–0.3 s higher; p95 TTFT was equal or better in 4 of 5 twins.
- Real MTP acceptance under greedy decoding is about 2% lower than the stock path (the serving numbers above use synthetic acceptance 3.39, like every point on the dashboard).
