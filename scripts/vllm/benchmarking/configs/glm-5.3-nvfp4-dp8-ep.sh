#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Perf benchmark for GLM-5.3-NVFP4 on a single-host TPU v7x-8 (4 chips,
# 8 TensorCores): attention-DP layout TP=1 x DP=8 with EP, the same layout
# as the GLM-5.2 v7x-16 nightly (glm-5.2-fp8-dp16-ep.sh) and the GLM-5.2
# NVFP4 DP8 recipe (cloud-devkit-recipes glm52-nvfp4-disagg-raiden-dp8).
#
# GLM-5.3 shares GLM-5.2's architecture (GlmMoeDsaForCausalLM: 78 layers,
# 64-head MLA + DSA indexer top-k 2048, 256 routed experts top-8), so the
# calibrated GLM-5.2 kernel profiles apply. The FP8 checkpoint (~756 GB)
# does not fit 8 x 94.7 GiB, hence the NVFP4 checkpoint (~465 GB: NVFP4
# routed experts, FP8 dense/attention/shared experts, FP8 KV).
#
# Sized for decode batch 512 (64 / device) at ISL 8192 / OSL 1024.
# Throughput/device = throughput / 8 (TPxDP recorded in config.json).
#
# The geometry knobs below take a GLM53_* environment override so a
# hill-climb campaign can sweep them without editing the config; the nightly
# uses the defaults. The prefix is deliberate: run_benchmarks.sh presets
# MODEL, ISL_OSL_CONFIGS, CONCURRENCY_OPTIONS and NUM_PROMPTS before sourcing
# this file, so the plain names cannot be overridden from the environment.
# GLM53_MODEL_PATH points the server at a local copy of the checkpoint.
MODEL="${GLM53_MODEL_PATH:-RadixArk/GLM-5.3-NVFP4}"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=8
ENABLE_EP=true
# No --quantization: hf_quant_config.json self-describes as NVFP4 and vLLM
# resolves it to modelopt_fp4.
QUANTIZATION=""
ISL_OSL_CONFIGS="${GLM53_ISL_OSL_CONFIGS:-8192:1024}"
CONCURRENCY_OPTIONS="${GLM53_CONCURRENCY_OPTIONS:-512}"
NUM_PROMPTS="${GLM53_NUM_PROMPTS:-1024}"

# Disable random length jitter: benchmark requests use exact nominal 8192:1024 lengths.
RANDOM_RANGE_RATIO="0.0"

# Block-512 requantization of the checkpoint's block-16 NVFP4 MoE weights
# (nvfp4.py): ~32x less scale data and no in-kernel dequant. Same setting as
# the GLM-5.2 NVFP4 recipe.
export MOE_REQUANTIZE_BLOCK_SIZE="${MOE_REQUANTIZE_BLOCK_SIZE:-512}"

# Server configuration:
# - max-model-len=9216: exact nominal length for 8192 input + 1024 output.
MAX_MODEL_LEN="${GLM53_MAX_MODEL_LEN:-9216}"
# Per DP engine. 1024 keeps the MoE dispatch dimension
# topk(8) x DP(8) x batched-tokens at 65536, the bound the GLM-5.2 DP16
# config documents for the SparseCore path.
MAX_NUM_BATCHED_TOKENS="${GLM53_MAX_NUM_BATCHED_TOKENS:-1024}"
# max-num-seqs is PER DP ENGINE: 64 x DP8 = 512 concurrent sequences,
# matching the benchmark concurrency.
MAX_NUM_SEQS="${GLM53_MAX_NUM_SEQS:-64}"

SERVER_READY_WAIT_MIN="${GLM53_SERVER_READY_WAIT_MIN:-180}"
GPU_MEMORY_UTILIZATION="${GLM53_GPU_MEMORY_UTILIZATION:-0.90}"
KV_CACHE_DTYPE="fp8"

# GLM-5.3 is an MLA model: tpu_platform.get_attn_backend_cls forces
# FLASH_ATTN_MLA whenever use_mla is set.

# --enable-ep-weight-filter: each rank reads only its own experts plus dense
# weights (model_loader_patches.py). --safetensors-load-strategy=prefetch
# streams the 47 shards from local disk.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--enable-ep-weight-filter --safetensors-load-strategy=prefetch --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45"

# Greedy decoding for bench requests.
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
LM_EVAL_TASKS="mmlu_pro"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
EXTRA_LM_EVAL_MODEL_ARGS="num_concurrent=32,timeout=7200"
LM_EVAL_GEN_KWARGS="until=['<|im_end|>']"
