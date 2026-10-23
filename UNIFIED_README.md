# ix-unified-oct23: one tree for TP8+DCP4 and DP4×TP2 decode

This branch serves both decode layouts of the Qwen3.5-397B P/D curve from one tree. The layout is chosen only by runtime flags and environment variables. Both layouts run the same code.

- **TP8 + DCP4** (bands C to H with the megakernel; bands A and B with the E6 or o6 configs): this is `shriharim/ix-curve-oct21-merged` @ `52925e2eb`. See `CURVE_README.md` in this tree for the per-band trees, environment and full commands.
- **DP4 × TP2** (bands A and B): this is `shriharim/ix-dp4tp2-oct22` @ `7bad8dfde` plus the env-gated DP4×TP2 candidate commits.

## What the branch is

The branch is `52925e2eb` plus these cherry-picks, in order. Every cherry-pick applied without a textual conflict. Only `tpu_runner.py` needed a 3-way auto-merge, for `cf54717bc` on top of R5.

| Commit here | Source | Change | Default |
|---|---|---|---|
| `824352014` | `cf54717bc` | Greedy top tokens under EP-DP with a TP head. The target keeps full logits, and the draft head keeps the vocab-parallel argmax. | Acts only when DP>1 with EP and TP>1. |
| `3843e3c7f` | `2564cf137` | `tpu_communicator`: a replicated (below `TPU_SP_MIN_ROWS`) sequence-parallel MoE step under DP>1 gathers over DP, and combines as a DP reduce-scatter followed by a TP all-reduce. | Acts only when the DP group size is above 1. |
| `42b037cbb` | `4afe2bb67` (= `a82efefdd`, same patch-id) | `TPU_RPA_DECODE_LANES`: sets the decode lanes of the batched RPA decode kernel. | 0 = off |
| `e42a24bc1` | `54a54831f` (= `58f792c52`, same patch-id) | `TPU_DP_PACKED_ROUTER_GATHER`: one packed DP all-gather of the hidden rows and the router logits, instead of two. Lossless. | off; acts only under DP>1 |
| `fa40006aa` | `933c26ed8` | `TPU_MOE_FUSED_EP_V3_CONFIG` (JSON overrides of the fused EP v3 `Config`) and `TPU_MOE_FUSED_EP_V3_TILE_ROWS`. | empty and 128, which are the old values |
| `12c582ff9` | `815539b67` | Fused EP v3 `Config.combine_real_tokens`. | off |
| `6dbc33888` | `a994fc7a3` | Fused EP v3 `Config.sort_index_compare`. | off |
| `fd68f98a2`, `910e1a6de` | `bd10e8d5a`, `b44438a09` | Bench harness only (`p3_d_leg_bench.py`, `smallb_job.sh`, `build_prompts.py`). No engine change. | not applicable |

`TPU_GDN_FUSED_1DOT` is an env-only switch that was already in the merged tree.

The merged branch already contained all of o6q's lineage (`69c56a021`, prefill v2). `git cherry` matched these by patch-id: `d5c40130f`≡`c297c20e0`, `71d6e2c75`≡`4d61bcd7d`, `46e5e0103`≡`f596f8587`, `c784ad231`≡`e1c7027a7`, `09079d1d5`≡`f3c9b3a34` and `69c56a021`≡`ed366ff3a`.

## Why TP8 behaviour is unchanged

**Tree identity (measured with git):**

- The unified tree minus the DP4×TP2 candidate tree (`a994fc7a3`) is exactly the merged branch's own delta over o6q (`69c56a021..52925e2eb`). It covers the same 11 files, and the patch is identical once hunk offsets are ignored.
- The unified tree minus the merged branch (`52925e2eb`) is 17 files. Six are tests, three are bench scripts, and eight are source files.

**Code reading of the 8 source files, with every new variable unset and DP=1:**

- `tpu_communicator.py`: `_replicated_dp_group` returns `None` when the DP group has one rank. Every changed branch then returns the old expression. The packed gather is reached only through that DP path.
- `greedy_top_tokens.py`, `tpu_runner.py`, `eagle3.py`: `_dp_lockstep_enabled()` is `dp_size > 1 and enable_expert_parallel`, so it is False at DP=1. The target head then takes the old "fast" call. The draft owner is resolved with `False`, which is what it was passed before.
- `batched_rpa_decode.py`: lanes 0 returns the same kwargs as before.
- `fused_moe_ep_v3.py`: an empty override string returns the config unchanged. `tile_rows()` returns 128, which equals `TILE_ROWS`.
- `kernels/fused_moe/v3/layer.py`, `combine.py`, `config.py`: with `combine_real_tokens` and `sort_index_compare` off, `combine_tokens` equals `message_tokens` and the pair index is the same slice. `math.gcd(combine_unroll, combine_tokens_per_tile)` equals `combine_unroll`, because `Config` already rejects an unroll that does not divide the tile. `in_place=False` keeps the old combine path.

**Unit tests** ran in the server image `vllm-torchtpu-dev@sha256:ba3be4a9…` on CPU under docker, with the tree mounted read-only as `/workspace/vllm-torchtpu`:

| Suite | Unified | Merged base `52925e2eb` |
|---|---|---|
| CURVE_README list (16 files) | 242 passed, 3 failed | 242 passed, 3 failed (the same 3) |
| Cherry-picked commits' tests (`test_tpu_communicator`, `test_batched_rpa_decode`, `test_fused_moe_ep_v3_config`, `test_greedy_top_tokens`) | 114 passed | 91 passed (the new tests are absent) |
| MoE adapters and fused MoE kernels (`test_fused_moe_ep`, `test_moe_runner`, `test_moe_routing`, `tests/kernels/fused_moe`) | 298 passed, 92 skipped, 2 failed | 298 passed, 92 skipped, 2 failed (the same 2) |

All five failures are pre-existing and fail identically on the base:

- The 3 megakernel failures: `test_qwen35_megakernel_buckets` `fake_step`, which does not accept `out_rows=`.
- `test_moe_routing.py::test_routing_path_compiles_fullgraph[sort|rowmax]`.

**Mutation checks.** Each check broke one guard in a scratch worktree, ran that commit's tests, and restored the tree. The no-edit control passed 114/114. All 12 mutations were caught:

- `2564cf137`: DP gather removed (6 failed); DP=1 guard removed (4 failed).
- `54a54831f`: packed gather never used (2 failed); packed gather on by default (3 failed).
- `cf54717bc`: target head always fast (1 failed); target head stock at TP1 (1 failed); draft lockstep passed through (1 failed).
- `4afe2bb67`: lanes env ignored (5 failed); lanes applied at 0 (5 failed).
- `933c26ed8`: overrides ignored (4 failed); unknown-field check off (1 failed); tile-rows env ignored (1 failed).

`combine_real_tokens` and `sort_index_compare` have no CPU unit test, because their kernels need a TPU. Their evidence comes from TPU runs:

- Kernel-level bit-exactness in `j-d52da5cb`.
- Engine token identity of 176/176 in `j-394f71be`.
- The engine microbench `j-50ecf56a`.

**Hardware: this tree has not run on TPU yet. Status at push (2026-10-23): the two GSM8K gates below are pending; use the source branches for gated numbers until they pass.** The planned gates are:

- GSM8K, TP8+DCP4 1P1D, against `j-e0de41ff` (97.57%).
- GSM8K, DP4×TP2 candidate, against `j-489694e3` (97.80%).

## Flags per layout (decode leg; the prefill leg is identical across all three)

Cap = requests per decode host. Set `MAX_NUM_SEQS_DECODE`, `DECODE_TPU_SCHED_MK_MAX_RUNNING` and `PROXY_DECODE_MK_CEILING` together, as the table shows.

| Variable | TP8+DCP4, megakernel (bands C–H) | TP8+DCP4, band A E6 / 1P1D gate | DP4×TP2 (bands A–B) |
|---|---|---|---|
| `TENSOR_PARALLEL_SIZE_DECODE` | 8 | 8 | **2** |
| `DATA_PARALLEL_SIZE_DECODE` | 1 | 1 | **4** |
| `DECODE_CONTEXT_PARALLEL_SIZE_DECODE` | 4 | 4 | **1** |
| `CP_KV_CACHE_INTERLEAVE_SIZE_DECODE` | 128 | 128 | **1** |
| `DECODE_VLLM_TPU_QWEN35_MEGAKERNEL` | 1 | 1 | **0** |
| `DECODE_VLLM_TPU_QWEN35_MTP_MK` | 1 | 1 | **0** |
| `DECODE_VLLM_TPU_QWEN35_MK_MODE` | `tp` for caps 1–4, `ep` for caps ≥ 8 | `ep` | not read (MK off) |
| `DECODE_TPU_SP_MIN_ROWS` | 512 | 352 (E6) / 512 (gate) | **88** |
| `DECODE_TPU_MOE_FUSED_EP_STEP_MIN_TOKENS` | unset (64) | 44 (E6) / unset (gate) | 44 |
| `DECODE_TPU_EXPERT_PLACEMENT` (`/workspace/vllm-torchtpu/expert_placement/…`) | `…_tp8_ep8_agentx_corr_c64_mk2p.json` (C), `…_c32.json` (D–H) | `expert_placement_dp8_ep8.json` (E6) / `…_c32.json` (gate) | `expert_placement_dp8_ep8.json` |
| `MAX_NUM_SEQS_DECODE` = `DECODE_TPU_SCHED_MK_MAX_RUNNING` | cap | cap | **cap ÷ 4** (16 for cap 64, 28 for cap 112) |
| `PROXY_DECODE_MK_CEILING` | cap | cap | cap (the host total) |
| `COMPILE_SIZES_DECODE` | `4,…,384,448,512,2048` | without 448 (E6) / with 448 (gate) | `8,16,24,32,48,64,72,80,96,112,128,256,512,2048` |
| `TPU_TOKEN_BUCKET_EXTRA_DECODE` | unset | unset | `8,48,2048` |
| `KV_CACHE_MEMORY_BYTES_DECODE` | unset | unset | `39500000000` (passthrough, below) |
| `DECODE_TPU_KV_BYTES_EXPLICIT` | unset | unset | `1` (passthrough) |
| `PROXY_*_ROUTING` | `llmd` (D–H), `prefix_hash` (C) | `prefix_hash` (A) / `llmd` (gate) | `prefix_hash` |
| `DECODE_TPU_RPA_DECODE_V2` | 1 | 1 | 1 (a no-op at DCP 1) |
| **DP4×TP2 candidate:** `DECODE_TPU_RPA_DECODE_LANES` | unset | unset | `4` |
| `DECODE_TPU_DP_PACKED_ROUTER_GATHER` | unset (a no-op at DP 1) | unset | `1` |
| `DECODE_TPU_GDN_FUSED_1DOT` | unset | unset | `1` |
| `DECODE_TPU_MOE_FUSED_EP_V3_CONFIG` | unset | unset | `{"combine_real_tokens":true,"sort_index_compare":true}` |

The candidate variables have been measured only on DP4×TP2. On TP8, lanes, GDN 1-dot and the fused-EP overrides would change real kernels, and they have not been gated there. Leave them unset for TP8.

**Never set `scatter_chunk_pairs=128` with `scatter_ring=8`** in `TPU_MOE_FUSED_EP_V3_CONFIG`. That pair hangs the fused EP kernel at 8192 global rows (1024 per shard, the 2048-row SP bucket that engine warmup compiles). The hang was found in `j-d52da5cb` and hung the engines of `j-1c54d05a` and `j-87e1aff5`.

Everything else in the decode environment is common to all three layouts. It is listed in `CURVE_README.md` ("Env common to every point" and the upstream knob set).

## Recipe passthroughs

`agentx-397b-pd-baseline` on cloud-devkit-recipes master does not pass the variables below. They live in the private clone `~/claude/cloud-devkit-dp4tp2` (`4b1623bd`, `4b3470f9`, `ca4a447e`). Each is opt-in, and an unset variable renders unchanged.

| Variable | Recipe effect | Needed by |
|---|---|---|
| `KV_CACHE_MEMORY_BYTES_DECODE` | `--kv-cache-memory-bytes` on decode | DP4×TP2 (the TP2 decode KV pin) |
| `TPU_KV_BYTES_EXPLICIT` (as `DECODE_…`) | env passthrough | DP4×TP2 |
| `BENCH_MEMORY_REQUEST` | memory request on `sidecar-bench`; use 48–64Gi on hour-long replays to avoid an OOM-kill | DP4×TP2 serving and LongBench |
| `TPU_RPA_DECODE_LANES`, `TPU_DP_PACKED_ROUTER_GATHER`, `TPU_GDN_FUSED_1DOT`, `TPU_MOE_FUSED_EP_V3_CONFIG` (as `DECODE_…`) | env passthroughs | DP4×TP2 candidate |

`CURVE_README.md` lists the TP8-side passthroughs that master drops: `PREFILL_TPU_RPA_PREFILL_BLOCKS`, `PREFILL_VLLM_TPU_LIGHTWEIGHT_STEP_STATS` and `LIBTPU_EXTRA_PREFILL`.

## Launch lines

Set `TREE` to a clone of this branch.

- **TP8+DCP4.** Use the commands in `CURVE_README.md` with `--map-dir "$TREE":*:/workspace/vllm-torchtpu`. The 1P1D GSM8K gate is `j-e0de41ff`'s command with only `--map-dir` and `BENCH_NAME` changed.
- **DP4×TP2.** Start from the TP8 command of the same shape. Replace the decode variables with the DP4×TP2 column above, add the four candidate variables, and render through a recipe that has the passthroughs. The gated 1P1D form is `j-489694e3`'s command (`cdk job desc j-489694e3-fa94-4d6a-97f6`) with `--map-dir "$TREE":*:/workspace/vllm-torchtpu` and `DECODE_TPU_MOE_FUSED_EP_V3_CONFIG={"combine_real_tokens":true,"sort_index_compare":true}` added.

## Known limits

- DP>1 with DCP>1 on a P/D KV consumer raises `NotImplementedError`. The Raiden Stage-3 admission names the layout `dp{N}tp{T}_decode`, with no DCP axis.
- DP4×TP2 loses to TP8 as served (megakernel up to 28 per host, XLA from 32) below about 31 requests per decode host. Same-harness step times, DP4×TP2 vs TP8 as served (2026-10-23, real weights, ctx 98,560): +58.5% at 8, +26.8% at 16, +3.6% at 28, −1.0% at 32, −9.3% at 64, −13.8% at 112; the cross is where TP8 leaves the megakernel (128 rows > its 112-row limit).
- Under DP4×TP2, each request's KV lands on 2 cores instead of 8. The median KV transfer is therefore 1.7–2.6× slower, and median TTFT is 0.1–0.3 s higher.
- R4 (HR + hx collectives) and the MK 112-row buckets are on by default. They live inside the megakernel, which DP4×TP2 turns off.
