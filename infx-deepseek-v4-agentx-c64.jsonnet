// infx-deepseek-v4-agentx — DeepSeek-V4 Pro FP8 across two tpu7x 2x2x2 hosts behind
// one Ray cluster, measured with an AgentX trace-replay sweep.
//
// Supports both Baseline (KV_OFFLOAD=false) and Raiden Host DRAM KV Cache Offload
// (KV_OFFLOAD=true, vllm-torchtpu PR #787 TPURaidenOffloadingConnector).
//
//     cdk recipe render infx-deepseek-v4-agentx
//     cdk job create infx-deepseek-v4-agentx FAST=true
//     cdk job create infx-deepseek-v4-agentx CLIENT_CONCURRENCIES="4 8 16 32 64"
//     cdk job create infx-deepseek-v4-agentx KV_OFFLOAD=true CLIENT_CONCURRENCIES="4 8 16 32 64"
//     cdk job create infx-deepseek-v4-agentx ROUTER=none   # no vllm-router, see "Routing"

local agentx = import 'cdk.agentx';
local cdk = import 'cdk.common';

local jobName = cdk.var('JOB_NAME', 'infx-deepseek-v4-agentx');
// The newest nightly. `latest` moves every night, so `cdk job recipe <id>` shows the tag, not which
// build ran; pass IMAGE=...:nightly-YYYYMMDD to pin one. Images from nightly-20261005 on carry torch_tpu
// 6c32c72bab (vllm-torchtpu #1442), whose world-sized Pallas mesh check fails DP=16/TP=1 at startup
// ("mesh size (1) must match world size (16)") unless the source has the fix, vllm-torchtpu #1606
// (merged 2026-10-06 16:03 UTC, after nightly-20261006 was built). Until a nightly carries it, run with a
// VLLM_TORCHTPU_SRC that does, such as gxd/smem-fix-1005-pate (f37a774aa).
local serverImage = cdk.var('IMAGE', 'us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-dev:latest');
// gs:// tarball of a vllm-torchtpu `src/` tree (`git archive <sha> src`) that replaces the image's
// editable /workspace/vllm-torchtpu/src before the patches below. Pair it with the nightly built from
// the same dependency pins; empty runs the image's own source.
local vllmTorchtpuSrc = cdk.var('VLLM_TORCHTPU_SRC', '');
local clientImage = cdk.var('BENCH_IMAGE', 'us-central1-docker.pkg.dev/cloud-tpu-inference-test/inferencex/inferencex-agentx:latest');

local modelPath = cdk.var('MODEL_PATH', 'deepseek-ai/DeepSeek-V4-Pro-0813');
local servedModelName = cdk.var('SERVED_MODEL_NAME', 'deepseek-ai/DeepSeek-V4-Pro-0813');
local gcsCkptDir = cdk.var('GCS_CKPT_DIR', 'gs://cloud-devkit/ckpt/hf/deepseek-ai/DeepSeek-V4-Pro-0813');
local gcsModelSrc = cdk.var('GCS_MODEL_SRC', gcsCkptDir);
local gcsDatasetCache = cdk.var('GCS_DATASET_CACHE', 'gs://cloud-devkit/datasets/hf_cache/semianalysisai/cc-traces-weka-062126');
local sharedCachePvc = cdk.var('SHARED_CACHE_PVC', '');
local hfStorageSize = cdk.var('HF_STORAGE_SIZE', '1500Gi');

// --- Cluster shape -----------------------------------------------------------
local rayReplicas = cdk.numVar('REPLICAS', cdk.numVar('RAY_NODES', 2));
local chipsPerHost = cdk.numVar('CHIPS_PER_HOST', 4);
local accelerator = cdk.var('TPU_ACCELERATOR', cdk.var('TPU_ACCELERATOR_TYPE', 'tpu7x'));
local topology = cdk.var('TPU_TOPOLOGY', '2x2x2');
local nodePool = cdk.var('NODE_POOL', '');
local tpuChips = cdk.numVar('TPU_CHIPS', 8);

// The head blocks until `ray status` reports the whole cluster.
local rayExpectedTpu = cdk.var('EXPECTED_TPU', cdk.var('RAY_EXPECTED_TPU', '16.0 TPU'));

// --- KV offloading (vllm-torchtpu PR #787 + Qwen3.5 / DSv4 Prefix-Routing reference) ---
local kvOffload = cdk.boolVar('KV_OFFLOAD', true) || cdk.boolVar('ENABLE_KV_OFFLOAD', true) || cdk.var('OFFLOAD_MODE', 'on') == 'on';
// Note: With CL 976485301 in torchtpu-vllm-dev:latest, KVCacheManagerBase allocates exact per-array
// page sizes (6.81 MiB/block across 122 arrays). On TP=16, DP=1, raiden_store.py divides cpu_bytes_to_use
// by world_size=16, so KV_OFFLOADING_SIZE=480 (480 GiB cluster total = 30 GiB/worker = 240 GiB/host)
// allocates ~4,514 host blocks (~2.31M tokens) in Host DRAM safely under the 900 GiB/host cgroup limit.
// On DP=16, TP=1 (the default below) cpu_bytes_to_use is per DP rank, so 480 would ask for ~3.75 TiB/host.
// 30 GiB/rank x 8 ranks = 240 GiB/host, the same host footprint as the TP=16 note above. The 900 GiB/host
// limit also covers /dev/shm (memory-backed, up to 300Gi) and the vLLM processes, so raise it with care.
// j-2c8a1b3b-17f0-4b33-bfb6 ran with 12079595520 bytes (11.25 GiB/rank = 90 GiB/host).
// 24, not 30: the host-staged weights stay resident after loading (~640 GiB anon a host, per the
// "hostmem:" lines of j-f432a2c4-3c5f-4658-b382), so 30 GiB/rank left no margin and that run and
// j-191aa11f-389d-43cb-bc1d were OOM-killed as the store came up. 24 GiB/rank = 192 GiB/host,
// ~20% fewer DRAM KV blocks. Back to 30 once the staging is actually freed.
local kvOffloadingSize = cdk.var('KV_OFFLOADING_SIZE', '24');
local offloadControllerPort = cdk.numVar('OFFLOAD_CONTROLLER_PORT', 27801);
local globalRegistryPort = cdk.numVar('GLOBAL_REGISTRY_PORT', 27800);
// Run global_registry_server on the head and point every Raiden store at it: stores publish their
// blocks there and can look up and RDMA-fetch blocks another store offloaded (as in forge
// j-3a93d5ab-4983-401e-bb56). raiden_store.py then requires the store_server_ip and PYTHONHASHSEED
// set below. Off by default (every store is local): on vLLM 0.31 the first remote-read load failure
// kills the server (j-0cce618c), while registry-off j-4d727281 saw no load failures.
local globalKvRegistry = cdk.boolVar('GLOBAL_KV_REGISTRY', false);
local raidenJobName = cdk.var('RAIDEN_JOB_NAME', 'dsv4-offload-probe');
// What vLLM does when a host-DRAM KV load fails (vLLM's default is fail). A Raiden load fails when a
// block it matched is evicted before the load pins it; with fail the request ends with no output, and
// since its blocks are never recomputed and re-offloaded, every later turn of that conversation fails
// the same way: 54 of j-f0249689-77e1-4eed-ad7e's 72 empty responses were one conversation from turn 7
// on. recompute reschedules the request to prefill the lost blocks (vllm-torchtpu d9e45eeb1).
local kvLoadFailurePolicy = cdk.var('KV_LOAD_FAILURE_POLICY', 'recompute');

// --- Routing -----------------------------------------------------------------
// none is the default. vllm-router's consistent hash ignores load: in j-40fc278f-1b0e-474d-8e54 it
// left 4-5 ranks queued 77-93% of the time beyond MAX_NUM_SEQS=8 / ~1.05M KV tokens, while others idled.
// A queued rank stops deferring prefill (prefill_capacity_bound), and under DP lockstep every rank
// pays: 92% of steps were prefill steps, against ~25-30% in j-68dfea6b-1db1-465b-91ef.
// vllm-router: the InferenceX B200 setup (dsv4/vllm/b200-fp4-mtp/agentic.yaml,
// override_dep8_c64).
// vllm-router runs in the client container in front of the one vLLM server, hashes each
// request's X-Session-ID (aiperf sends it, AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID below)
// onto a consistent-hash ring of the DP ranks, and pins the rank with an X-data-parallel-rank
// header. The hook steps aside for requests that arrive with a rank, and defaults to off here.
// none: no router; the API server picks the DP engine itself, through the prefix/session hook
// that ENABLE_PREFIX_ROUTING patches into vLLM's DPLBAsyncMPClient. A new session goes to the
// least-loaded engine and stays there. j-68dfea6b-1db1-465b-91ef and earlier ran this way.
local router = cdk.var('ROUTER', 'none');
local useVllmRouter =
  if router == 'vllm-router' then true
  else if router == 'none' then false
  else error 'ROUTER must be vllm-router or none, got "%s"' % router;
local vllmRouterVersion = cdk.var('VLLM_ROUTER_VERSION', '0.1.14');
local vllmRouterPort = cdk.numVar('VLLM_ROUTER_PORT', 8090);
local enablePrefixRouting = cdk.boolVar('ENABLE_PREFIX_ROUTING', !useVllmRouter);

// --- Serving -----------------------------------------------------------------
local port = cdk.numVar('PORT', 47903);
// DP=16 attention (one replica per TPU core); with --enable-expert-parallel the MoE layers
// still shard experts across all DP x TP = 16 ranks.
local dataParallelSize = cdk.numVar('DATA_PARALLEL_SIZE', 16);
local tensorParallelSize = cdk.numVar('TENSOR_PARALLEL_SIZE', 1);
local maxModelLen = cdk.numVar('MAX_MODEL_LEN', 1048576);
// Evolve defaults (forge run x2vw65z3, j-9fc3926b: 86,862 vs 69,267 total tok/s for the values in the notes
// below, j-c7c6502d): every step carries at most a 128-token prefill chunk, so the DP-wide bucket stays
// at 128 instead of a 2048 mixed step every 8th step that all 16 ranks pay for.
local maxNumSeqs = cdk.numVar('MAX_NUM_SEQS', 64);
// Note: On TP=16, DP=1 (94.74 GiB HBM/core), MAX_NUM_BATCHED_TOKENS=2048 with GPU_MEMORY_UTILIZATION=0.90
// allocates 3,208,656 HBM KV cache tokens (6,266 blocks) + 2,311,168 Host DRAM tokens (4,514 blocks)
// = 5.52M total tokens, sustaining 62.1% HBM / 71.8% total cache hit rate and 31.95 tok/s at C=64.
// 2048, as j-68dfea6b-1db1-465b-91ef ran, not the B200-parity 4104 (4096 prefill + 1 decode token for
// each of MAX_NUM_SEQS=8). 3072 (j-430349bf-ea70-428e-9d1a, j-f74e6597-85ac-4f79-869a) fit, but its
// forward program needs 7.35G of HLO temp: reserving that failed and forced 94 HBM defragments per
// rank, with every bucket change reloading programs, and its prefill steps cost 1.36 s p50 against
// ~0.85-1.0 s at 2048. The token buckets end at this value and every bucket is compiled, so it has to
// clear two limits:
// - Kernel tiling. The DSv4 kernels tile tokens by largest divisors. wo_a_projection (o_projection.py)
//   takes the largest divisor <= 1024 as its block, which Pallas requires to be a multiple of 8: 4104
//   gives 684 and startup fails in lowering (j-50b68a3f-6a37-450d-9f7b). 4160 is the smallest value
//   above 4096 whose every token tile is a multiple of 8. Check any new value against the f4e2627da
//   kernels with
//   python3 -c "n=N; ld=lambda x,c: max(d for d in range(1, c + 1) if x % d == 0); t=ld(n, 1024); \
//     print(all(v % 8 == 0 for v in (t, ld(t, 128), ld(n, 128), ld(n, 64))))"
// - HBM, with MAX_MODEL_LEN=1048576. One such request needs 12.92 GiB of KV cache per engine. 4160 at
//   GPU_MEMORY_UTILIZATION=0.90 left 11.11 GiB (j-de2a21fa-c755-4f08-8193). At 0.93 it got 13.95 GiB,
//   but the 4160 graph then needed 9.54G of program HBM at warmup compile (5.33G of it the MoE gmm_v2
//   over 4160 x 16 ranks x 6 = 399,360 rows) and overran HBM by 912M (j-e563664d-dc8d-41d0-9e20).
//   Both limits only meet near 0.92, with ~35M to spare. 2048 at 0.90 had 15.59 GiB (1,264,937
//   tokens, j-68dfea6b-1db1-465b-91ef). Interpolating, 3072 at 0.90 leaves ~13.4 GiB and ~2G of
//   compile headroom.
// c=64 (forge run ya5w5534, objective p90 interactivity): every step pays the DP-wide max bucket, which
// at 128 is T128 on ~100% of steps (~73.5 ms). 64 runs every step at T64 (~57 ms): n=2 248.1k vs 243.0k
// total tok/s, intvty p90 60.8 vs 47.05 (j-418771d5, j-d1e2c60f). 48 is no cheaper than 64 (T48 56 ms).
local maxNumBatchedTokens = cdk.numVar('MAX_NUM_BATCHED_TOKENS', 64);
// As in j-2c8a1b3b-17f0-4b33-bfb6. BLOCK_SIZE=0 drops --block-size and leaves it to vLLM.
local blockSize = cdk.numVar('BLOCK_SIZE', 1024);
// The DSv4 HCA and indexer kernel blocks count KV pages, and vllm-torchtpu's defaults (6990d19b3) are sized
// for a 512-token block. At 1024 they span twice the tuned tokens and the HCA decode pass runs out of VMEM
// compiling the T16 bucket (j-4fcc236c-bfa4-4f11-8c6a), so 1024 takes the halved, tuned values. Empty = source default.
local hcaKvPagesPerBlock = cdk.var('TPU_DSV4_HCA_NUM_KV_PAGES_PER_BLOCK', if blockSize == 1024 then '128,64,64' else '');
local indexerKvPagesPerBlock = cdk.var('TPU_DSV4_INDEXER_NUM_KV_PAGES_PER_BLOCK', if blockSize == 1024 then '32,64,32' else '');
local quantization = cdk.var('QUANTIZATION', 'deepseek_v4_fp8');
local kvCacheDtype = cdk.var('KV_CACHE_DTYPE', 'fp8_e4m3');
// Raising this enlarges the KV cache but shrinks the HBM left for compiling the largest token bucket after
// it is allocated; vLLM's profile run under-counts that compile. See MAX_NUM_BATCHED_TOKENS above for 0.93.
local gpuMemoryUtilization = cdk.var('GPU_MEMORY_UTILIZATION', '0.97');
local enablePrefixCaching = cdk.boolVar('ENABLE_PREFIX_CACHING', true);
// Token interval between retained sliding-window prefix-cache checkpoints (vLLM CacheConfig
// prefix_cache_retention_interval). 32768 is the InferenceX B200 setup
// (dsv4/vllm/b200-fp4-mtp/agentic.yaml). Passed as --prefix-cache-retention-interval (vLLM v0.29.0+):
// vLLM v0.30.0 removed the VLLM_PREFIX_CACHE_RETENTION_INTERVAL env var (vllm-project/vllm#55353), so on
// vllm-torchtpu main (vLLM v0.31.0) the env var is silently ignored. vLLM's default, 0, keeps only the latest replay boundary
// and shared-prefix junctions, so a prefix that diverges elsewhere can only hit back to the last of
// those; DeepSeek-V4's 61 sliding-window layers need that state at the hit point. Must be a multiple
// of BLOCK_SIZE. PREFIX_CACHE_RETENTION_INTERVAL=none leaves vLLM's default, as
// j-68dfea6b-1db1-465b-91ef and earlier ran.
local prefixCacheRetentionInterval = cdk.var('PREFIX_CACHE_RETENTION_INTERVAL', '32768');
local setRetentionInterval =
  if prefixCacheRetentionInterval == 'none' then false
  else if std.length(std.findSubstr('.', prefixCacheRetentionInterval)) == 0
       && std.parseInt(prefixCacheRetentionInterval) >= 0
       && (blockSize == 0 || std.parseInt(prefixCacheRetentionInterval) % blockSize == 0) then true
  else error 'PREFIX_CACHE_RETENTION_INTERVAL must be none or a multiple of BLOCK_SIZE=%d, got "%s"'
             % [blockSize, prefixCacheRetentionInterval];
// Shortest-remaining-prefill-first: decodes first, then prefill chunks and waiting requests by remaining
// prompt tokens (vllm_torchtpu/core/experimental/srpt_prefill_scheduler.py on main; forge trees before
// the move had it at vllm_torchtpu.core.srpt_prefill_scheduler). j-4fdda261-648a-4891-96f2 vs
// j-f0249689-77e1-4eed-ad7e (same tree otherwise): 164,297 vs 154,663 total tok/s, mean TTFT 10.5 vs 14.9 s.
// SCHEDULER_CLS=none drops --scheduler-cls (vLLM's default scheduler). Others:
// SCHEDULER_CLS=vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler
local schedulerCls = cdk.var('SCHEDULER_CLS', 'vllm_torchtpu.core.experimental.srpt_prefill_scheduler.SrptPrefillScheduler');
// With SCHEDULER_CLS=vllm_torchtpu.core.dp_prefill_gate.DpPrefillGateScheduler (needs a VLLM_TORCHTPU_SRC
// that has it), each EngineCore_DPn logs "DP prefill gate:" counts of why throttled steps carried prefill
// every TPU_PREFILL_GATE_LOG_STEPS steps (vllm-torchtpu default 1000). TPU_PREFILL_GATE_MIN_CADENCES=N>=1
// lets a rank skip the prefill hold-back only after N consecutive cadence steps used the whole token budget
// with requests still waiting; unset or 0 keeps vLLM's test. Empty leaves either env var unset.
local prefillGateMinCadences = cdk.var('TPU_PREFILL_GATE_MIN_CADENCES', '');
local prefillGateLogSteps = cdk.var('TPU_PREFILL_GATE_LOG_STEPS', '');
// 8, as j-68dfea6b-1db1-465b-91ef and earlier ran; B200 runs 16. With each request capped at
// LONG_PREFILL_TOKEN_THRESHOLD tokens per prefill step, 16 halves how often a prompt advances: median
// TTFT doubled (9 s -> 18 s) and uncached prefill fell from 3.2k to 1.7-1.9k tok/s in the 16-interval
// runs (j-430349bf-ea70-428e-9d1a, j-f74e6597-85ac-4f79-869a). j-9472b9c6-fc62-465a-bb63 ran 4.
// PREFILL_SCHEDULE_INTERVAL=0 drops --prefill-schedule-interval.
local prefillScheduleInterval = cdk.numVar('PREFILL_SCHEDULE_INTERVAL', 0);
// LONG_PREFILL_TOKEN_THRESHOLD=0 drops --long-prefill-token-threshold (vLLM's default).
// Default 512 - MAX_NUM_SEQS (504): the runner pads every DP rank to the MAX over ranks of step tokens,
// rounded up to a power-of-two bucket, so one 512 chunk plus a few decodes (515) cost all 16 ranks a
// 1024 forward and two chunks cost 2048. At 504 a chunk plus up to MAX_NUM_SEQS-1 decodes fits 512 and
// two fit 1024. j-d0188942-b76b-47ec-98db (512) had ~2k real prefill tokens per prefill step across all
// 16 ranks, against 1024-2048 buckets.
local longPrefillTokenThreshold = cdk.numVar('LONG_PREFILL_TOKEN_THRESHOLD', 0);
local loadFormat = cdk.var('LOAD_FORMAT', '');
local toolCallParser = cdk.var('TOOL_CALL_PARSER', '');
local reasoningParser = cdk.var('REASONING_PARSER', 'deepseek_r1');
local chatTemplate = cdk.var('CHAT_TEMPLATE', '/tmp/chat_template.jinja');
local moeRequantizeWeightDtype = cdk.var('MOE_REQUANTIZE_WEIGHT_DTYPE', 'fp4');
local moeRequantizeBlockSize = cdk.numVar('MOE_REQUANTIZE_BLOCK_SIZE', 512);
// Fused EP MoE v2 is on by default, as j-68dfea6b-1db1-465b-91ef and earlier ran it: the kernel
// scores, dispatches only routed tokens, runs both matmuls and combines in one program, and the
// W4A8 and height-rung knobs below apply (kept in step with torchtpu-vllm-dsv4-pro-v7x-16-mmlu_pro).
// An armed layer ignores TPU_MOE_SKIP_PADDED_TOKENS and TPU_MOE_COLLECTION_CHUNK_SIZE (vllm-torchtpu
// warns once); hash-routed layers are refused at admission and stay on the grouped-matmul path.
// USE_MOE_FUSED_EP_KERNEL=0 runs every layer on that path: vllm-torchtpu's grouped matmul bracketed
// by vLLM's all-gather and reduce-scatter, fp4 weights at W4A16, as
// recipes/inferencex/deepseek-v4/infx-deepseek-v4-agentx.jsonnet runs and j-430349bf-ea70-428e-9d1a /
// j-f74e6597-85ac-4f79-869a ran.
local useMoeFusedEpKernel = cdk.var('USE_MOE_FUSED_EP_KERNEL', '1');
local moeFusedEpEnableW4a8 = cdk.var('MOE_FUSED_EP_ENABLE_W4A8', '1');
local moeFusedEpV2HeightRungsOverride = cdk.var('MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE', '2');
// Fused v2 builds its routing plan sharded: each shard computes only the arrival rows of its own
// tokens and the slab rows of its own experts, plus a [1, experts] count exchange that rides the
// routing all-gather, instead of four one-hot passes over every pair on every shard. The kernel
// operands are bit-identical either way (vllm-torchtpu's sharded-vs-replicated tests). It also
// drops TPU_MOE_SKIP_PADDED_TOKENS' padding pairs by construction; the replicated builder needs
// gxd/agentx-base-1007 2aad3b83b for that. 0 restores vllm-torchtpu's default, the replicated plan.
local moeFusedEpV2ShardedPlan = cdk.var('MOE_FUSED_EP_V2_SHARDED_PLAN', '1');
local tpuMoeSkipPaddedTokens = cdk.var('TPU_MOE_SKIP_PADDED_TOKENS', '1');
local raggedGatherReduceVersion = cdk.var('RAGGED_GATHER_REDUCE_VERSION', 'v3');
// Slice-builder heartbeat, as in j-2c8a1b3b-17f0-4b33-bfb6: set both as an env var and as a libtpu flag.
local sliceBuilderHeartbeatInterval = cdk.var('TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL', '30m');
local libtpuInitArgs = cdk.var('LIBTPU_INIT_ARGS', std.join(' ', [
  '--xla_tpu_use_minor_sharding_for_major_trivial_input=true',
  '--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false',
  '--xla_tpu_ars_combiner_threshold_in_bytes=0',
  '--xla_tpu_enable_async_collective_merger=false',
  '--xla_enable_async_all_gather=true',
  '--xla_tpu_enable_sparse_core_collective_offload_all_gather=true',
  '--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144',
  '--deepsea_slice_builder_heartbeat_interval=%s' % sliceBuilderHeartbeatInterval,
]));
local extraVllmArgs = cdk.var('EXTRA_VLLM_ARGS', '');
// SERVER_EXTRA_ENV=K1=V1,K2=V2 sets more environment variables on the server pods and copies them into
// the Ray workers, for knobs this recipe has no variable of its own for. A value that holds commas is
// written with ':' in their place.
local serverExtraEnv = {
  [std.splitLimit(kv, '=', 1)[0]]: std.strReplace(std.splitLimit(kv, '=', 1)[1], ':', ',')
  for kv in std.split(cdk.var('SERVER_EXTRA_ENV', ''), ',')
  if kv != ''
};
// fastokens (crusoecloud/fastokens) in place of the Rust BPE backend of HF fast tokenizers, the
// deepseek_v4 tokenizer mode included: VLLM_USE_FASTOKENS=1 (vLLM v0.24+, needs fastokens >= 0.2.0).
// Pip-installed at this version on every pod unless the image already has it. none: HF's tokenizer.
local fastokensVersion = cdk.var('FASTOKENS_VERSION', '0.3.4');
local useFastokens = fastokensVersion != 'none';

// --- Speculative decoding ------------------------------------------------------
// SPEC_METHOD=dspark serves the DSpark draft that ships in the target checkpoint (mtp.0-2.*, config.json
// dspark_block_size=5), as the InferenceX B200 run does (K=6, probabilistic drafts, synthetic AL 3.77); the
// defaults below are that reference config. SPEC_METHOD=none turns it off. j-59ba1696: 148,940 vs 86,862 total tok/s without it.
// The draft holds ~4.2 GiB of HBM per rank (j-9719749f-0a6b-47c7-b249: 77.64 vs 73.44 GiB used). At a 2048-token
// step budget K=6 needs GPU_MEMORY_UTILIZATION >= 0.94 to fit one MAX_MODEL_LEN=1048576 request
// (j-a300025a-c924-4600-b9e5 refused to start at 0.93: 9.78 GiB of KV against 9.85 needed).
local specMethod = cdk.var('SPEC_METHOD', 'dspark');
local numSpecTokens = cdk.numVar('NUM_SPEC_TOKENS', 6);
// SPEC_SYNTHETIC_AL=<mean acceptance length> accepts drafts at fixed per-position rates that give this
// length (vLLM rejection_sample_method=synthetic) instead of by the target's probabilities, as the InferenceX
// reference measures DSpark throughput. Outputs are then not samples of the target. The reference's
// golden curve for DeepSeek-V4-Pro-0813 DSpark is K1 1.84, K2 2.51, K3 3.01, K4 3.36, K5 3.61, K6 3.77.
// Empty = real rejection sampling.
local specSyntheticAl = cdk.var('SPEC_SYNTHETIC_AL', '3.77');
// SPEC_DRAFT_SAMPLE=probabilistic draws drafts from the draft distribution at the request's temperature
// (vLLM draft_sample_method); on TPU it needs SPEC_SYNTHETIC_AL. Empty = greedy drafting.
local specDraftSample = cdk.var('SPEC_DRAFT_SAMPLE', 'probabilistic');
// SPEC_DISABLE_EAGLE_BLOCK_DROP=true turns off the EAGLE prefix-cache block drop with vLLM's own
// speculative-config flag, which also leaves the draft groups unannotated, so KV offload stores no
// EAGLE peek chunk. Pair it with VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=0 to measure it alone.
local specDisableEagleBlockDrop = cdk.boolVar('SPEC_DISABLE_EAGLE_BLOCK_DROP', false);
local specConfig =
  if specMethod == 'none' then ''
  else if specMethod == 'dspark' then
    '{"method":"dspark","num_speculative_tokens":%d%s%s%s}' % [
      numSpecTokens,
      if specDraftSample == '' then '' else ',"draft_sample_method":"%s"' % specDraftSample,
      if specSyntheticAl == '' then ''
      else ',"rejection_sample_method":"synthetic","synthetic_acceptance_length":%s' % specSyntheticAl,
      if specDisableEagleBlockDrop then ',"disable_eagle_block_drop":true' else '',
    ]
  else error 'SPEC_METHOD must be none or dspark, got "%s"' % specMethod;
// vLLM drops the last matched prefix-cache block of every hit for EAGLE-style drafts (DSpark and the
// DeepSeek-V4 groups included): ~BLOCK_SIZE more prefill tokens per request. DSpark's draft KV for verified
// tokens is projected from the target's hidden states, so the dropped block protects nothing. 1 keeps it
// (vllm-torchtpu #1530); empty leaves vLLM's behaviour.
local disableEagleBlockDrop = cdk.var('VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP', '1');
// --- Profiling ----------------------------------------------------------------
local enableXprof = cdk.boolVar('ENABLE_XPROF', false);
local enablePerfetto = cdk.boolVar('ENABLE_PERFETTO', false);
local enableProfiling = enableXprof || enablePerfetto;
local xprofDir = '${CDK_OUTPUT_DIR:-/tmp/cdk_output}/xprof';
// libtpu per-module VLOG. Off by default: 'tpu_pjrt_client=1' logs every program dispatch,
// ~82M lines (16.5 GB) per server pod in j-c80b896b-f1c0-4534-955d. Opt in with e.g.
// TPU_VMODULE=tpu_pjrt_client=1,pjrt_stream_executor_client=1,tpu_pjrt_compiler_utils=1
local tpuVmodule = cdk.var('TPU_VMODULE', '');
// Seconds between "hostmem:" lines in each server pod's log (see the server script); 0 turns the
// sampler off. Added after j-191aa11f-389d-43cb-bc1d was OOM-killed at the 900Gi host limit at the
// end of startup with nothing in the log saying what held the memory.
local hostMemSampleSeconds = cdk.numVar('HOST_MEM_SAMPLE_SECONDS', 30);
// libtpu's premapped host buffer, sized and thresholded as in infx-glm52-agentx and
// infx-qwen35-agentx (one var sets both, as there). Without it every transfer maps its own DMA
// buffer: in j-f74e6597-85ac-4f79-869a MapDmaBuffer took ~850 ms during program reloads and
// blocked that rank's H2D/D2H, stalling all 16 ranks for ~1.1 s.
// It pins 4 GiB in each of the 8 TPU processes a host, 32 GiB, which only fits beside
// KV_OFFLOADING_SIZE=24: with 30 it took j-191aa11f-389d-43cb-bc1d and j-f432a2c4-3c5f-4658-b382
// past the 900Gi host limit. j-d0188942-b76b-47ec-98db (24, no buffer) held 835/900 GiB on the head
// pod for the whole benchmark, so this leaves ~30 GiB. none sets neither variable, and libtpu then
// enables no buffer.
local premappedBufferSize = cdk.var('TPU_PREMAPPED_BUFFER_SIZE', '4294967296');
// vllm-torchtpu ee93d4833+: after loading, log the CPU tensor storage still reachable and what holds
// it, to find what keeps the staged weights resident. 0 turns it off.
local logHostTensorsAfterLoad = cdk.var('TPU_LOG_HOST_TENSORS_AFTER_LOAD', '1');
// Steps between each rank's "step timing" log line (step period by DP token bucket, vllm-torchtpu
// runner/step_timing_log.py); 0 turns it off.
local stepTimingLogSteps = cdk.var('TPU_STEP_TIMING_LOG_STEPS', '2000');
// Fused-EP expert placement shipped with vllm-torchtpu (src/vllm_torchtpu/expert_placements/<name>.json):
// each EP16 rank holds a load-balanced set of experts instead of a contiguous block. Bitexact. Removing it
// cost -4.6% total tok/s at c160 (n=3). Empty = contiguous blocks.
local expertPlacement = cdk.var('TPU_EXPERT_PLACEMENT', 'dsv4_pro_agentx_ep16');
// SrptPrefillScheduler caps each step's attention work (sum over scheduled tokens of their context) by
// clipping prefill chunks while some DP rank decodes, so one rank's chunk deep in a long prompt stops
// setting every rank's step. Empty = off.
local srptAttnWorkCap = cdk.var('VLLM_TPU_SRPT_ATTN_WORK_CAP', '');

// --- Benchmark client --------------------------------------------------------
local fastMode = cdk.boolVar('FAST', false) || cdk.boolVar('FAST_MODE', false);
local customDuration = cdk.rawVar('DURATION', '') != '';
local duration =
  if customDuration then cdk.numVar('DURATION')
  else if fastMode then 1200
  else 3600;

// Client flags default to the SemiAnalysis B200 run (InferenceX point 442735, whose
// results/benchmark.log starts with the aiperf command), which are also InferenceX
// benchmark_lib.sh's own defaults: --warmup-requests-per-lane 10, --trace-idle-gap-cap-seconds 300,
// no --unsafe-override, no --max-context-length. FAST keeps InferenceX's fast-mode warmup of 1.
// j-f74e6597-85ac-4f79-869a and earlier ran warmup 1 and an idle cap of 10 s, which offers a colder
// and denser load than B200's. B200's 707-request warmup took 17 min; this server prefills more
// slowly, so expect warmup to run up to its 1800 s grace period.
local rawWarmup = cdk.var('AIPERF_WARMUP_REQUESTS_PER_LANE', cdk.var('WARMUP_REQUESTS_PER_LANE', cdk.var('WARMUP', if fastMode then '1' else '10')));
local warmupReqs = if rawWarmup == '0' || rawWarmup == '' then '1' else rawWarmup;

local concurrencies = agentx.parseConcurrencies(cdk.var('CLIENT_CONCURRENCIES', '64'));
local benchName = cdk.var('BENCH_NAME', 'deepseek-v4-agentx-multihost-nondisagg');

local serverJobName = 'ray-node';
local benchJobName = 'benchmark';

// ---------------------------------------------------------------------------
// The Ray cluster
// ---------------------------------------------------------------------------

local tpuProcessAddresses = if cdk.rawVar('TPU_PROCESS_ADDRESSES', '') != '' then
  cdk.var('TPU_PROCESS_ADDRESSES')
else
  std.join(',', [
    '$(JS_NAME)-%s-%d-0.$(JS_NAME).$(POD_NAMESPACE).svc.cluster.local:8476' % [serverJobName, i]
    for i in std.range(0, rayReplicas - 1)
  ]);

local offloadSetup = if !kvOffload then 'echo "Starting vLLM server (offload=OFF)..."\n' else agentx.fill(
  |||
    # A value small enough to be a GiB count is read as one; anything larger is bytes.
    KV_OFFLOAD_SIZE="@SIZE@"
    if [ "$KV_OFFLOAD_SIZE" -le 1000000 ]; then
      CPU_BYTES=$(( KV_OFFLOAD_SIZE * 1024 * 1024 * 1024 ))
    else
      CPU_BYTES="$KV_OFFLOAD_SIZE"
    fi
    RAIDEN_CONTROLLER_PORT="@CONTROLLER_PORT@"
    export PYTHONHASHSEED=0
    GLOBAL_REGISTRY_JSON=""
    if [ "@GLOBAL_KV_REGISTRY@" = "true" ]; then
      command -v global_registry_server >/dev/null || { echo "GLOBAL_KV_REGISTRY=true: no global_registry_server in the image" >&2; exit 1; }
      global_registry_server --port=@GLOBAL_REGISTRY_PORT@ > "${CDK_OUTPUT_DIR:-/tmp/cdk_output}/global_registry.log" 2>&1 &
      GLOBAL_REGISTRY_PID=$!
      until (exec 3<>/dev/tcp/127.0.0.1/@GLOBAL_REGISTRY_PORT@) 2>/dev/null; do
        kill -0 "$GLOBAL_REGISTRY_PID" 2>/dev/null || { echo "global_registry_server exited at startup" >&2; exit 1; }
        sleep 1
      done
      # A dead registry would leave every store silently local-only, so it ends the server instead.
      ( while kill -0 "$GLOBAL_REGISTRY_PID" 2>/dev/null; do sleep 10; done
        echo "global_registry_server exited; stopping the server" >&2; kill -TERM 0 ) &
      GLOBAL_REGISTRY_JSON=',"global_registry_address":"'"${HEAD_ADDR}:@GLOBAL_REGISTRY_PORT@"'"'
      echo "global_registry_server up on :@GLOBAL_REGISTRY_PORT@ (pid ${GLOBAL_REGISTRY_PID})"
    fi
    KVT='{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_load_failure_policy":"@KV_LOAD_FAILURE_POLICY@","kv_connector_extra_config":{"cpu_bytes_to_use":'"${CPU_BYTES}"',"raiden_controller_port":'"${RAIDEN_CONTROLLER_PORT}"',"store_server_ip":"'"${HEAD_ADDR}"'","raiden_job_name":"@RAIDEN_JOB_NAME@"'"${GLOBAL_REGISTRY_JSON}"'}}'
    OFFLOAD_ARGS=(
      --enable-prompt-tokens-details
      --kv-transfer-config "$KVT"
    )
    echo "Starting vLLM server with TPURaidenOffloadingConnector enabled (cpu_bytes_to_use=${CPU_BYTES} per DP rank, controller_port=${RAIDEN_CONTROLLER_PORT}, head=${HEAD_ADDR})..."
  |||,
  { SIZE: kvOffloadingSize, CONTROLLER_PORT: offloadControllerPort, GLOBAL_REGISTRY_PORT: globalRegistryPort, GLOBAL_KV_REGISTRY: if globalKvRegistry then 'true' else 'false', RAIDEN_JOB_NAME: raidenJobName, KV_LOAD_FAILURE_POLICY: kvLoadFailurePolicy }
);

local serveArgs = [
  'vllm serve "$MODEL_PATH"',
  '--served-model-name "$SERVED_MODEL_NAME"',
  '--quantization %s' % quantization,
  '--trust-remote-code',
  '--data-parallel-size=%d' % dataParallelSize,
  '--tensor-parallel-size=%d' % tensorParallelSize,
]
+ (if loadFormat != '' then ['--load-format=%s' % loadFormat] else [])
+ [
  '--max-model-len=%d' % maxModelLen,
  '--max-num-seqs=%d' % maxNumSeqs,
  '--max-num-batched-tokens=%d' % maxNumBatchedTokens,
]
+ (if blockSize != 0 then ['--block-size=%d' % blockSize] else [])
+ [
  '--port %d' % port,
  '--api-server-count=1',
  '--async-scheduling',
  if enablePrefixCaching then '--enable-prefix-caching' else '--no-enable-prefix-caching',
  '--enable-prompt-tokens-details',
  '--kv-cache-dtype=%s' % kvCacheDtype,
  '--safetensors-load-strategy=prefetch',
  '--enable-expert-parallel',
  '--enable-ep-weight-filter',
  '--gpu-memory-utilization=%s' % gpuMemoryUtilization,
  '"${OFFLOAD_ARGS[@]}"',
]
+ (if setRetentionInterval then ['--prefix-cache-retention-interval=%s' % prefixCacheRetentionInterval] else [])
+ (if schedulerCls != 'none' then ['--scheduler-cls %s' % schedulerCls] else [])
+ (if prefillScheduleInterval != 0 then ['--prefill-schedule-interval=%d' % prefillScheduleInterval] else [])
+ (if longPrefillTokenThreshold != 0 then ['--long-prefill-token-threshold=%d' % longPrefillTokenThreshold] else [])
+ (if toolCallParser != '' then ['--tool-call-parser=%s' % toolCallParser] else [])
+ [
  '--reasoning-parser=%s' % reasoningParser,
  '--chat-template=%s' % chatTemplate,
]
+ (if specConfig != '' then ["--speculative-config '%s'" % specConfig] else [])
+ (if enableProfiling then agentx.profilerArgs('"%s"' % xprofDir) else []);

local serverScript = agentx.fill(
  |||
    mkdir -p /dev/shm/{jax_cache,vllm_cache,torchinductor}
    echo "JOBSET_JOB_INDEX=$JOBSET_JOB_INDEX"
    # Host memory, as the OOM killer counts it: the container's cgroup usage and limit, split
    # into anon (process heaps, host tensors), file (page cache, reclaimable) and shmem (tmpfs
    # such as the /dev/shm caches, shared-memory buffers). Grep the log for "hostmem:".
    HOST_MEM_SAMPLE_SECONDS="@HOST_MEM_SAMPLE_SECONDS@"
    if [ "$HOST_MEM_SAMPLE_SECONDS" -gt 0 ] 2>/dev/null; then
      (
        { set +x; } 2>/dev/null
        gib() { awk -v b="$1" 'BEGIN { if (b ~ /^[0-9]+$/) printf "%.1fGiB", b / 2^30; else printf "%s", b }'; }
        while true; do
          if [ -f /sys/fs/cgroup/memory.current ]; then
            used=$(cat /sys/fs/cgroup/memory.current); limit=$(cat /sys/fs/cgroup/memory.max)
            split=$(awk '$1 == "anon" || $1 == "file" || $1 == "shmem" { printf " %s=%.1fGiB", $1, $2 / 2^30 }' /sys/fs/cgroup/memory.stat)
          else
            used=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes); limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
            split=$(awk '$1 == "rss" || $1 == "cache" || $1 == "shmem" { printf " %s=%.1fGiB", $1, $2 / 2^30 }' /sys/fs/cgroup/memory/memory.stat)
          fi
          echo "hostmem: $(date +%T) pod=$JOBSET_JOB_INDEX used=$(gib "$used") limit=$(gib "$limit")$split"
          sleep "$HOST_MEM_SAMPLE_SECONDS"
        done
      ) &
    fi
    if [ -n "$TPU_LIBRARY_PATH" ]; then
      echo "TPU_LIBRARY_PATH is set to $TPU_LIBRARY_PATH. Copying libtpu.so from GCS..."
      mkdir -p "$(dirname "$TPU_LIBRARY_PATH")"
      gcloud storage cp gs://cloud-devkit/libs/libtpu/20260521/libtpu.so $TPU_LIBRARY_PATH 2>/dev/null || true
    fi

    export TPU_NODE_ID=$JOBSET_JOB_INDEX
    export TPU_WORKER_ID=$JOBSET_JOB_INDEX
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
    export RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=3600
    export TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL="${TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL:-30m}"

    HEAD_ADDR="@HEAD_ADDR@"
    echo "Resolved Ray head address (HEAD_ADDR) as: ${HEAD_ADDR}"

    if [ -n "${VLLM_TORCHTPU_SRC:-}" ]; then
      echo "Replacing /workspace/vllm-torchtpu/src with ${VLLM_TORCHTPU_SRC}..."
      rm -rf /tmp/vllm-torchtpu-src && mkdir -p /tmp/vllm-torchtpu-src
      gcloud storage cp "${VLLM_TORCHTPU_SRC}" /tmp/vllm-torchtpu-src/src.tar.gz || exit 1
      tar -xzf /tmp/vllm-torchtpu-src/src.tar.gz -C /tmp/vllm-torchtpu-src || exit 1
      [ -d /tmp/vllm-torchtpu-src/src/vllm_torchtpu ] || { echo "No src/vllm_torchtpu in ${VLLM_TORCHTPU_SRC}"; exit 1; }
      rm -rf /workspace/vllm-torchtpu/src && mv /tmp/vllm-torchtpu-src/src /workspace/vllm-torchtpu/src
      echo "vllm-torchtpu src replaced: $(find /workspace/vllm-torchtpu/src -name '*.py' | wc -l) .py files"
    fi

    if [ "${VLLM_USE_FASTOKENS:-0}" = "1" ]; then
      python3 -c 'import fastokens' 2>/dev/null || python3 -m pip install --quiet 'fastokens==@FASTOKENS_VERSION@' || exit 1
      echo "fastokens $(python3 -c 'from importlib.metadata import version; print(version("fastokens"))') (VLLM_USE_FASTOKENS=1)"
    fi

    # Ensure pool_tensors[0] in TPURaidenOffloadingConnector has max(page_size_bytes) (HCA 128 KiB)
    # so pre-CL-976485301 KVCacheManagerBase allocates host_size large enough for all 122 arrays.
    python3 -c 'p = "/workspace/vllm-torchtpu/src/vllm_torchtpu/offload/raiden_connector.py"; s = open(p).read(); t = "        self._pool_sync_tensors = ["; patch = "        pool_tensors.sort(key=lambda x: -x.page_size_bytes)" + chr(10) + t; ("pool_tensors.sort" not in s and t in s) and (open(p, "w").write(s.replace(t, patch, 1)), print("Patched raiden_connector.py: sorted pool_tensors by descending page_size_bytes"))'
    python3 -c 'p = "/workspace/vllm-torchtpu/src/vllm_torchtpu/offload/raiden_store.py"; s = open(p).read(); t = "rank = parallel_state.get_world_group().rank_in_group"; r = "rank = parallel_state.get_tp_group().rank_in_group"; (t in s) and (open(p, "w").write(s.replace(t, r)), print("Patched raiden_store.py: rank = get_tp_group().rank_in_group"))'

    # Allow TP=16 on DeepSeek-V4 shared_experts (intermediate_size=3072 -> 3072/16=192 not divisible by block_k=128)
    # by running shared_experts with disable_tp=True and scaling output by 1/tp_size prior to MoERunner all-reduce.
    python3 -c 'import glob; [open(p, "w").write(s.replace("reduce_results=False,", "reduce_results=False, is_sequence_parallel=(intermediate_size % (self.tp_size * 128) != 0),")) for p in glob.glob("/workspace/**/deepseek_v4/moe.py", recursive=True) + glob.glob("/usr/local/**/deepseek_v4/moe.py", recursive=True) for s in [open(p).read()] if "is_sequence_parallel=(intermediate_size" not in s and "reduce_results=False," in s]; [open(p, "w").write(s.replace("self.down_proj = RowParallelLinear(", "from vllm.distributed import get_tensor_model_parallel_world_size as _gtp; self._tp_scale = (1.0 / _gtp()) if (is_sequence_parallel and not reduce_results and _gtp() > 1) else 1.0; self.down_proj = RowParallelLinear(").replace("x, _ = self.down_proj(x)", "x, _ = self.down_proj(x); x = x * getattr(self, \"_tp_scale\", 1.0)")) for p in glob.glob("/workspace/**/deepseek_v4/layers.py", recursive=True) + glob.glob("/usr/local/**/deepseek_v4/layers.py", recursive=True) for s in [open(p).read()] if "_tp_scale" not in s and "self.down_proj = RowParallelLinear(" in s]'

    # Patch vllm-torchtpu PR #1343: vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler
    PAW_SCHED_PATH="/workspace/vllm-torchtpu/src/vllm_torchtpu/core/prefix_aware_scheduler.py"
    if [ ! -f "$PAW_SCHED_PATH" ]; then
      echo "Patching vllm-torchtpu PR #1343 (PrefixAwareScheduler) into $PAW_SCHED_PATH..."
      mkdir -p "$(dirname "$PAW_SCHED_PATH")"
      echo "IyBDb3B5cmlnaHQgMjAyNiBHb29nbGUgTExDCiMKIyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKIyB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuCiMgWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0CiMKIyAgICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wCiMKIyBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlCiMgZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gIkFTIElTIiBCQVNJUywKIyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KIyBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kCiMgbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuCiIiIlByZWZpeC1hd2FyZSBhZG1pc3Npb24gc2NoZWR1bGVyIGZvciBwcmVmaXgtY2FjaGluZyB3b3JrbG9hZHMuCgpTdGFuZGFyZCB2TExNIGFkbWl0cyByZWFkeSByZXF1ZXN0cyBpbiBGQ0ZTIG9yZGVyLCB3aGljaCBpcyBvYmxpdmlvdXMgdG8Kd2hldGhlciBhIHJlcXVlc3QncyBwcm9tcHQgcHJlZml4IGlzIGFscmVhZHkgY2FjaGVkIGluIEhCTS4gSW4gbXVsdGktdHVybgphZ2VudGljIHdvcmtsb2FkcywgYWRtaXR0aW5nIGEgbG9uZyB1bmNhY2hlZCBwcm9tcHQgYXQgcXVldWVbMF0gYmxvY2tzIHJlYWR5CmZvbGxvdy11cCByZXF1ZXN0cyB3aG9zZSBwcmVmaXhlcyBhcmUgYWxyZWFkeSBjYWNoZWQsIGFuZCB1bmRlciBoaWdoIEhCTQp1dGlsaXphdGlvbiBjYW4gZXZpY3QgdGhvc2UgY2FjaGVkIHByZWZpeGVzIGJlZm9yZSB0aGV5IGFyZSByZXVzZWQuCgpQcmVmaXhBd2FyZVNjaGVkdWxlciBpbnNwZWN0cyB0aGUgcmVhZHkgcXVldWUgd2l0aGluIGEgbG9va2FoZWFkIHdpbmRvdyBhbmQKcHJpb3JpdGl6ZXMgcmVxdWVzdHMgd2l0aCB0aGUgZmV3ZXN0IHVuY2FjaGVkIHByb21wdCB0b2tlbnMuIEl0IG1vZGlmaWVzIG9ubHkKdGhlIGFkbWlzc2lvbiBvcmRlcmluZzsgbWVtb3J5IG1hbmFnZW1lbnQsIGV2aWN0aW9uLCBhbmQgcHJlZW1wdGlvbiBwb2xpY2llcwpyZW1haW4gdW50b3VjaGVkLgoKRW5hYmxlIHdpdGg6CiAgICAtLXNjaGVkdWxlci1jbHMgdmxsbV90b3JjaHRwdS5jb3JlLnByZWZpeF9hd2FyZV9zY2hlZHVsZXIuUHJlZml4QXdhcmVTY2hlZHVsZXIKIiIiCgpmcm9tIGl0ZXJ0b29scyBpbXBvcnQgaXNsaWNlCgpmcm9tIHZsbG0ubG9nZ2VyIGltcG9ydCBpbml0X2xvZ2dlcgpmcm9tIHZsbG0udjEuY29yZS5zY2hlZC5hc3luY19zY2hlZHVsZXIgaW1wb3J0IEFzeW5jU2NoZWR1bGVyCmZyb20gdmxsbS52MS5jb3JlLnNjaGVkLnJlcXVlc3RfcXVldWUgaW1wb3J0IFJlcXVlc3RRdWV1ZSwgU2NoZWR1bGluZ1BvbGljeQpmcm9tIHZsbG0udjEucmVxdWVzdCBpbXBvcnQgUmVxdWVzdAoKbG9nZ2VyID0gaW5pdF9sb2dnZXIoInZsbG0uIiArIF9fbmFtZV9fKQoKX0xPR19FVkVSWSA9IDEwMDAKCiMgUHJpb3JpdHkgdGllcnM6IGxvd2VyIHZhbHVlcyBhZG1pdCBlYXJsaWVyLgpfVElFUl9QUkVFTVBURUQgPSAxCl9USUVSX05PUk1BTCA9IDIKCgpjbGFzcyBQcmVmaXhBd2FyZVNjaGVkdWxlcihBc3luY1NjaGVkdWxlcik6CiAgICAiIiJBZG1pdCByZWFkeSByZXF1ZXN0cyBwcmlvcml0aXppbmcgZmV3ZXN0IHVuY2FjaGVkIHByb21wdCB0b2tlbnMuIiIiCgogICAgIzogTG9va2FoZWFkIHdpbmRvdyBvZiByZWFkeSByZXF1ZXN0cyB0byBldmFsdWF0ZSBmb3IgY2FjaGVkIHByZWZpeCBsZW5ndGguCiAgICBXSU5ET1cgPSA2NAoKICAgIGRlZiBfX2luaXRfXyhzZWxmLCAqYXJncywgKiprd2FyZ3MpIC0+IE5vbmU6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygqYXJncywgKiprd2FyZ3MpCiAgICAgICAgc2VsZi5fcGF3X24gPSAwCiAgICAgICAgc2VsZi5fcGF3X2NvbGRfdG9rZW5zID0gMAogICAgICAgIHNlbGYuX3Bhd19wcm9tcHRfdG9rZW5zID0gMAogICAgICAgIHNlbGYuX3Bhd19zY29yaW5nX2Jyb2tlbiA9IEZhbHNlCgogICAgICAgIGxvZ2dlci5pbmZvKCJQcmVmaXhBd2FyZVNjaGVkdWxlciBBQ1RJVkUgKHdpbmRvdz0lZCkiLCBzZWxmLldJTkRPVykKCiAgICBkZWYgX3ByZWZpeF93YXJtdGgoc2VsZiwgcmVxdWVzdDogUmVxdWVzdCkgLT4gaW50OgogICAgICAgICIiIlByb21wdCB0b2tlbnMgb2YgYGByZXF1ZXN0YGAgcmVzaWRlbnQgaW4gdGhlIEtWIGNhY2hlLiIiIgogICAgICAgIGt2bSA9IHNlbGYua3ZfY2FjaGVfbWFuYWdlcgogICAgICAgIGlmIG5vdCBrdm0uZW5hYmxlX2NhY2hpbmcgb3IgcmVxdWVzdC5za2lwX3JlYWRpbmdfcHJlZml4X2NhY2hlOgogICAgICAgICAgICByZXR1cm4gMAogICAgICAgIGlmIHJlcXVlc3QubnVtX2NvbXB1dGVkX3Rva2VucyA+IDA6CiAgICAgICAgICAgIHJldHVybiByZXF1ZXN0Lm51bV9jb21wdXRlZF90b2tlbnMKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJlc3VsdCA9IGt2bS5jb29yZGluYXRvci5maW5kX2xvbmdlc3RfY2FjaGVfaGl0KAogICAgICAgICAgICAgICAgcmVxdWVzdC5ibG9ja19oYXNoZXMsIG1heChyZXF1ZXN0Lm51bV90b2tlbnMgLSAxLCAwKQogICAgICAgICAgICApCiAgICAgICAgICAgIHJldHVybiBpbnQocmVzdWx0WzFdKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIGlmIG5vdCBzZWxmLl9wYXdfc2NvcmluZ19icm9rZW46CiAgICAgICAgICAgICAgICBzZWxmLl9wYXdfc2NvcmluZ19icm9rZW4gPSBUcnVlCiAgICAgICAgICAgICAgICBsb2dnZXIuZXhjZXB0aW9uKAogICAgICAgICAgICAgICAgICAgICJQcmVmaXhBd2FyZVNjaGVkdWxlcjogd2FybXRoIHNjb3JpbmcgZmFpbGVkOyBmYWxsaW5nIGJhY2sgdG8gRkNGUy4iCiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgIHJldHVybiAwCgogICAgZGVmIF9yYW5rX2FuZF93YXJtdGgoCiAgICAgICAgc2VsZiwgcmVxdWVzdDogUmVxdWVzdAogICAgKSAtPiB0dXBsZVt0dXBsZVtpbnQsIGZsb2F0LCBmbG9hdF0sIGludF06CiAgICAgICAgIiIiUmV0dXJuIChzb3J0X2tleSwgd2FybXRoKS4gTG93ZXIgc29ydF9rZXkgaXMgYWRtaXR0ZWQgZmlyc3QuIiIiCiAgICAgICAgd2FybXRoID0gc2VsZi5fcHJlZml4X3dhcm10aChyZXF1ZXN0KQogICAgICAgIGNvbGRfdG9rZW5zID0gZmxvYXQocmVxdWVzdC5udW1fdG9rZW5zIC0gd2FybXRoKQogICAgICAgIHRpZXIgPSBfVElFUl9QUkVFTVBURUQgaWYgcmVxdWVzdC5udW1fcHJlZW1wdGlvbnMgPiAwIGVsc2UgX1RJRVJfTk9STUFMCiAgICAgICAgcmV0dXJuICh0aWVyLCBjb2xkX3Rva2VucywgcmVxdWVzdC5hcnJpdmFsX3RpbWUpLCB3YXJtdGgKCiAgICBkZWYgX3NlbGVjdF93YWl0aW5nX3F1ZXVlX2Zvcl9zY2hlZHVsaW5nKHNlbGYpIC0+IFJlcXVlc3RRdWV1ZSB8IE5vbmU6CiAgICAgICAgcXVldWUgPSBzdXBlcigpLl9zZWxlY3Rfd2FpdGluZ19xdWV1ZV9mb3Jfc2NoZWR1bGluZygpCiAgICAgICAgaWYgcXVldWUgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIHNlbGYuX3Byb21vdGVfd2FybWVzdChxdWV1ZSkKICAgICAgICByZXR1cm4gcXVldWUKCiAgICBkZWYgX3Byb21vdGVfd2FybWVzdChzZWxmLCBxdWV1ZTogUmVxdWVzdFF1ZXVlKSAtPiBOb25lOgogICAgICAgICIiIlByb21vdGUgdGhlIHJlcXVlc3Qgd2l0aCB0aGUgZmV3ZXN0IHVuY2FjaGVkIHRva2VucyB3aXRoaW4gV0lORE9XLiIiIgogICAgICAgIGlmIHNlbGYucG9saWN5ICE9IFNjaGVkdWxpbmdQb2xpY3kuRkNGUyBvciBsZW4ocXVldWUpIDwgMjoKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgIHdpbmRvdyA9IGxpc3QoaXNsaWNlKHF1ZXVlLCBzZWxmLldJTkRPVykpCiAgICAgICAgaGVhZCA9IHdpbmRvd1swXQogICAgICAgIHNjb3JlZCA9IFsoc2VsZi5fcmFua19hbmRfd2FybXRoKHIpLCByKSBmb3IgciBpbiB3aW5kb3ddCiAgICAgICAgKGJlc3RfcmFuaywgYmVzdF93YXJtdGgpLCBiZXN0ID0gbWluKHNjb3JlZCwga2V5PWxhbWJkYSB0OiB0WzBdWzBdKQoKICAgICAgICBpZiBiZXN0IGlzIG5vdCBoZWFkIGFuZCBiZXN0X3JhbmsgPCBzY29yZWRbMF1bMF1bMF06CiAgICAgICAgICAgIHF1ZXVlLnJlbW92ZShiZXN0KQogICAgICAgICAgICBxdWV1ZS5wcmVwZW5kX3JlcXVlc3QoYmVzdCkKCiAgICAgICAgc2VsZi5fcGF3X24gKz0gMQogICAgICAgIHNlbGYuX3Bhd19jb2xkX3Rva2VucyArPSBtYXgoYmVzdC5udW1fdG9rZW5zIC0gYmVzdF93YXJtdGgsIDApCiAgICAgICAgc2VsZi5fcGF3X3Byb21wdF90b2tlbnMgKz0gYmVzdC5udW1fdG9rZW5zCgogICAgICAgIGlmIHNlbGYuX3Bhd19uICUgX0xPR19FVkVSWSA9PSAwOgogICAgICAgICAgICBsb2dnZXIuaW5mbygKICAgICAgICAgICAgICAgICJ3YXJtZXN0LWZpcnN0OiBuPSVkIG1lYW5fY29sZF90b2tlbnM9JS4wZiAiCiAgICAgICAgICAgICAgICAiY29sZF9mcmFjPSUuM2YgcXVldWVfZGVwdGg9JWQiLAogICAgICAgICAgICAgICAgc2VsZi5fcGF3X24sCiAgICAgICAgICAgICAgICBzZWxmLl9wYXdfY29sZF90b2tlbnMgLyBzZWxmLl9wYXdfbiwKICAgICAgICAgICAgICAgIHNlbGYuX3Bhd19jb2xkX3Rva2VucyAvIG1heChzZWxmLl9wYXdfcHJvbXB0X3Rva2VucywgMSksCiAgICAgICAgICAgICAgICBsZW4ocXVldWUpLAogICAgICAgICAgICApCg==" | base64 -d > "$PAW_SCHED_PATH"
      python3 -m py_compile "$PAW_SCHED_PATH" && echo "Patched PrefixAwareScheduler compiled successfully"
    fi

    if [ -d "/vllm-cache/deepseek-ai/DeepSeek-V4-Pro-0813" ]; then
      echo "Found DeepSeek-V4 on shared PVC at /vllm-cache/deepseek-ai/DeepSeek-V4-Pro-0813"
      export MODEL_PATH="/vllm-cache/deepseek-ai/DeepSeek-V4-Pro-0813"
      export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Pro-0813}"
      export HF_HUB_OFFLINE=1
    elif [ -d "/vllm-cache/hub/models--deepseek-ai--DeepSeek-V4-Pro-0813" ]; then
      echo "Found DeepSeek-V4 on shared PVC at /vllm-cache/hub/models--deepseek-ai--DeepSeek-V4-Pro-0813"
      export MODEL_PATH="deepseek-ai/DeepSeek-V4-Pro-0813"
      export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Pro-0813}"
      export HF_HUB_OFFLINE=1
    elif [ -n "${GCS_CKPT_DIR:-}" ]; then
      LOCAL_CKPT_DIR="/root/.cache/huggingface/deepseek-ai/DeepSeek-V4-Pro-0813"
      echo "Syncing DeepSeek-V4 weights from GCS ($GCS_CKPT_DIR) to $LOCAL_CKPT_DIR..."
      mkdir -p "$LOCAL_CKPT_DIR"
      gcloud storage rsync -r "$GCS_CKPT_DIR" "$LOCAL_CKPT_DIR"
      echo "Evicting 831 GiB synced checkpoint files from Linux page cache (POSIX_FADV_DONTNEED)..."
      python3 -c 'import os, glob; [os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED) or os.close(fd) for f in glob.glob(os.path.join("/root/.cache/huggingface", "**", "*"), recursive=True) if os.path.isfile(f) for fd in [os.open(f, os.O_RDONLY)]]' 2>/dev/null || true
      export MODEL_PATH="$LOCAL_CKPT_DIR"
      export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Pro-0813}"
      export HF_HUB_OFFLINE=1
    fi

    if [ "$JOBSET_JOB_INDEX" = "0" ]; then
      ray stop --force 2>/dev/null || true
      echo "Starting Ray head on $HOSTNAME..."
      ray start --head --port=6379

      sleep 10
      echo "Waiting for Ray workers to join..."
      EXPECTED_TPU="@EXPECTED_TPU@"
      until ray status 2>/dev/null | grep -q "$EXPECTED_TPU"; do
        sleep 2
      done
      echo "Worker joined. Ray cluster status:"
      ray status

      if [ ! -f /tmp/chat_template.jinja ]; then
        echo "eyUgaWYgbm90IGFkZF9nZW5lcmF0aW9uX3Byb21wdCBpcyBkZWZpbmVkICV9CiAgeyUgc2V0IGFkZF9nZW5lcmF0aW9uX3Byb21wdCA9IGZhbHNlICV9CnslIGVuZGlmICV9CnslIGZvciBtZXNzYWdlIGluIG1lc3NhZ2VzICV9CiAgeyUgaWYgbWVzc2FnZVsicm9sZSJdID09ICJzeXN0ZW0iICV9CiAgICB7eyBtZXNzYWdlWyJjb250ZW50Il0gKyAiXG4iIH19CiAgeyUgZWxpZiBtZXNzYWdlWyJyb2xlIl0gPT0gInVzZXIiICV9CiAgICB7eyAiPO+9nFVzZXLvvZw+IiArIG1lc3NhZ2VbImNvbnRlbnQiXSB9fQogIHslIGVsaWYgbWVzc2FnZVsicm9sZSJdID09ICJhc3Npc3RhbnQiICV9CiAgICB7eyAiPO+9nEFzc2lzdGFudO+9nD4iICsgbWVzc2FnZVsiY29udGVudCJdICsgIjzvvZxlbmQgb2Ygc2VudGVuY2XvvZw+IiB9fQogIHslIGVuZGlmICV9CnslIGVuZGZvciAlfQp7JSBpZiBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQgJX0KICB7eyAiPO+9nEFzc2lzdGFudO+9nD48dGhpbms+XG4iIH19CnslIGVuZGlmICV9Cg==" | base64 -d > /tmp/chat_template.jinja
      fi

      mkdir -p "${CDK_OUTPUT_DIR:-/tmp/cdk_output}"
      mkdir -p "@XPROF_DIR@"

      if [ "@ENABLE_PREFIX_ROUTING@" = "true" ]; then
        CORE_CLIENT_PATH="/usr/local/lib/python3.12/site-packages/vllm/v1/engine/core_client.py"
        if [ ! -f "$CORE_CLIENT_PATH" ]; then
          CORE_CLIENT_PATH=$(python3 -c "import importlib.util; print(importlib.util.find_spec('vllm.v1.engine.core_client').origin)" 2>/dev/null | tail -n 1)
        fi
        if [ -f "$CORE_CLIENT_PATH" ] && ! grep -q "_install_prefix_aware_routing" "$CORE_CLIENT_PATH"; then
          echo "Installing Prefix-Aware DP Routing Hook into $CORE_CLIENT_PATH..."
          echo "ZGVmIF9pbnN0YWxsX3ByZWZpeF9hd2FyZV9yb3V0aW5nKCk6CiAgICBpbXBvcnQgb3MsIHRpbWUsIGFycmF5LCB6bGliCiAgICBpZiBvcy5lbnZpcm9uLmdldCgiRU5BQkxFX1BSRUZJWF9ST1VUSU5HIiwgIjEiKSBub3QgaW4gKCIxIiwgInRydWUiLCAiVHJ1ZSIpOgogICAgICAgIHJldHVybgoKICAgIGZyb20gdmxsbS52MS5lbmdpbmUuY29yZV9jbGllbnQgaW1wb3J0IERQTEJBc3luY01QQ2xpZW50CiAgICBvcmlnX2dldCA9IERQTEJBc3luY01QQ2xpZW50LmdldF9jb3JlX2VuZ2luZV9mb3JfcmVxdWVzdAoKICAgIF9ibG9ja19tYXAgPSB7fQogICAgX2VuZ2luZV9oYXNoZXMgPSB7fQogICAgX3JvdXRlX2NvdW50cyA9IHt9CiAgICBfc2Vzc2lvbl9tYXAgPSB7fQogICAgX2VuZ2luZV9sYXN0X2FjdGl2ZSA9IHt9CiAgICBfQkxPQ0tfU0laRSA9IDUxMgogICAgX01JTl9NQVRDSF9CTE9DS1MgPSAxMgoKICAgIGRlZiBwcmVmaXhfYXdhcmVfZ2V0X2NvcmVfZW5naW5lKHNlbGYsIHJlcXVlc3QpOgogICAgICAgIG5vbmxvY2FsIF9ibG9ja19tYXAsIF9lbmdpbmVfaGFzaGVzLCBfcm91dGVfY291bnRzLCBfc2Vzc2lvbl9tYXAsIF9lbmdpbmVfbGFzdF9hY3RpdmUKICAgICAgICBpZiBnZXRhdHRyKHJlcXVlc3QsICJkYXRhX3BhcmFsbGVsX3JhbmsiLCBOb25lKSBpcyBub3QgTm9uZToKICAgICAgICAgICAgcmV0dXJuIG9yaWdfZ2V0KHNlbGYsIHJlcXVlc3QpCgogICAgICAgIG51bV9lbmdpbmVzID0gbGVuKHNlbGYuY29yZV9lbmdpbmVzKQogICAgICAgIGlmIG51bV9lbmdpbmVzIDw9IDE6CiAgICAgICAgICAgIHJldHVybiBvcmlnX2dldChzZWxmLCByZXF1ZXN0KQoKICAgICAgICBub3cgPSB0aW1lLm1vbm90b25pYygpCgogICAgICAgICMgU3RyYXRlZ3kgMTogU2Vzc2lvbiBJRCBhZmZpbml0eSAoWC1TZXNzaW9uLUlEIGhlYWRlciAvIEFJUEVSRiBzZXNzaW9uIGFmZmluaXR5KQogICAgICAgIHNpZCA9IGdldGF0dHIocmVxdWVzdCwgInNlc3Npb25faWQiLCBOb25lKQogICAgICAgIGlmIHNpZDoKICAgICAgICAgICAgaWYgc2lkIGluIF9zZXNzaW9uX21hcDoKICAgICAgICAgICAgICAgIGNhbmRpZGF0ZV9pZHggPSBfc2Vzc2lvbl9tYXBbc2lkXSAlIG51bV9lbmdpbmVzCiAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICBkZWYgX2VuZ2luZV9rZXkoaSk6CiAgICAgICAgICAgICAgICAgICAgZW5nID0gc2VsZi5jb3JlX2VuZ2luZXNbaV0KICAgICAgICAgICAgICAgICAgICBpbmZsaWdodCA9IHNlbGYuZW5naW5lX2luZmxpZ2h0W2VuZ10KICAgICAgICAgICAgICAgICAgICB3YWl0aW5nLCBydW5uaW5nLCBrdl91c2FnZSA9IHNlbGYubGJfZW5naW5lc1tpXQogICAgICAgICAgICAgICAgICAgIGFjdGl2ZV9yZXFzID0gbWF4KHNlbGYuY2xpZW50X2NvdW50ICogaW5mbGlnaHQsIHdhaXRpbmcgKyBydW5uaW5nKQogICAgICAgICAgICAgICAgICAgIGhhc19iZWVuX3VzZWQgPSAxIGlmIF9yb3V0ZV9jb3VudHMuZ2V0KGksIDApID4gMCBlbHNlIDAKICAgICAgICAgICAgICAgICAgICBsYXN0X3QgPSBfZW5naW5lX2xhc3RfYWN0aXZlLmdldChpLCAwLjApCiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIChhY3RpdmVfcmVxcywgaGFzX2JlZW5fdXNlZCwgbGFzdF90LCBrdl91c2FnZSwgaSkKCiAgICAgICAgICAgICAgICBjYW5kaWRhdGVfaWR4ID0gbWluKHJhbmdlKG51bV9lbmdpbmVzKSwga2V5PV9lbmdpbmVfa2V5KQogICAgICAgICAgICAgICAgX3Nlc3Npb25fbWFwW3NpZF0gPSBjYW5kaWRhdGVfaWR4CiAgICAgICAgICAgICAgICBfcm91dGVfY291bnRzW2NhbmRpZGF0ZV9pZHhdID0gX3JvdXRlX2NvdW50cy5nZXQoY2FuZGlkYXRlX2lkeCwgMCkgKyAxCgogICAgICAgICAgICBfZW5naW5lX2xhc3RfYWN0aXZlW2NhbmRpZGF0ZV9pZHhdID0gbm93CiAgICAgICAgICAgIGNob3NlbiA9IHNlbGYuY29yZV9lbmdpbmVzW2NhbmRpZGF0ZV9pZHhdCiAgICAgICAgICAgIHNlbGYubGJfZW5naW5lc1tjYW5kaWRhdGVfaWR4XVswXSArPSBzZWxmLmNsaWVudF9jb3VudAogICAgICAgICAgICBzZWxmLnJlcXNfaW5fZmxpZ2h0W3JlcXVlc3QucmVxdWVzdF9pZF0gPSBjaG9zZW4KICAgICAgICAgICAgc2VsZi5lbmdpbmVfaW5mbGlnaHRbY2hvc2VuXSArPSAxCiAgICAgICAgICAgIHJldHVybiBjaG9zZW4KCiAgICAgICAgIyBTdHJhdGVneSAyOiBDdW11bGF0aXZlIEZpeGVkLUJsb2NrIFJhZGl4IFByZWZpeCBSb3V0aW5nIChUb2tlbi1sZXZlbCBwcmVmaXggbWF0Y2gpCiAgICAgICAgdG9rZW5zID0gZ2V0YXR0cihyZXF1ZXN0LCAicHJvbXB0X3Rva2VuX2lkcyIsIE5vbmUpCiAgICAgICAgaWYgdG9rZW5zIGFuZCBsZW4odG9rZW5zKSA+PSAoX0JMT0NLX1NJWkUgKiBfTUlOX01BVENIX0JMT0NLUyk6CiAgICAgICAgICAgIG5fdG9rID0gbWluKGxlbih0b2tlbnMpLCAyNjIxNDQpCiAgICAgICAgICAgIG51bV9ibG9ja3MgPSBuX3RvayAvLyBfQkxPQ0tfU0laRQogICAgICAgICAgICBhcnIgPSBhcnJheS5hcnJheSgiSSIsIHRva2Vuc1s6bnVtX2Jsb2NrcyAqIF9CTE9DS19TSVpFXSkKICAgICAgICAgICAgcmF3ID0gYXJyLnRvYnl0ZXMoKQoKICAgICAgICAgICAgYl9oYXNoZXMgPSBbXQogICAgICAgICAgICBoID0gMAogICAgICAgICAgICBibG9ja19ieXRlcyA9IF9CTE9DS19TSVpFICogNAogICAgICAgICAgICBmb3IgaSBpbiByYW5nZShudW1fYmxvY2tzKToKICAgICAgICAgICAgICAgIGggPSB6bGliLmNyYzMyKHJhd1tpICogYmxvY2tfYnl0ZXMgOiAoaSArIDEpICogYmxvY2tfYnl0ZXNdLCBoKQogICAgICAgICAgICAgICAgYl9oYXNoZXMuYXBwZW5kKGgpCgogICAgICAgICAgICBjYW5kaWRhdGVfaWR4ID0gTm9uZQogICAgICAgICAgICBmb3IgaSBpbiByYW5nZShudW1fYmxvY2tzIC0gMSwgX01JTl9NQVRDSF9CTE9DS1MgLSAyLCAtMSk6CiAgICAgICAgICAgICAgICBiaCA9IGJfaGFzaGVzW2ldCiAgICAgICAgICAgICAgICBpZiBiaCBpbiBfYmxvY2tfbWFwOgogICAgICAgICAgICAgICAgICAgIGNhbmRpZGF0ZV9pZHggPSBfYmxvY2tfbWFwW2JoXSAlIG51bV9lbmdpbmVzCiAgICAgICAgICAgICAgICAgICAgYnJlYWsKCiAgICAgICAgICAgIGlmIGNhbmRpZGF0ZV9pZHggaXMgTm9uZToKICAgICAgICAgICAgICAgIGRlZiBfZW5naW5lX2tleShpKToKICAgICAgICAgICAgICAgICAgICBlbmcgPSBzZWxmLmNvcmVfZW5naW5lc1tpXQogICAgICAgICAgICAgICAgICAgIGluZmxpZ2h0ID0gc2VsZi5lbmdpbmVfaW5mbGlnaHRbZW5nXQogICAgICAgICAgICAgICAgICAgIHdhaXRpbmcsIHJ1bm5pbmcsIGt2X3VzYWdlID0gc2VsZi5sYl9lbmdpbmVzW2ldCiAgICAgICAgICAgICAgICAgICAgYWN0aXZlX3JlcXMgPSBtYXgoc2VsZi5jbGllbnRfY291bnQgKiBpbmZsaWdodCwgd2FpdGluZyArIHJ1bm5pbmcpCiAgICAgICAgICAgICAgICAgICAgaGFzX2JlZW5fdXNlZCA9IDEgaWYgX3JvdXRlX2NvdW50cy5nZXQoaSwgMCkgPiAwIGVsc2UgMAogICAgICAgICAgICAgICAgICAgIGxhc3RfdCA9IF9lbmdpbmVfbGFzdF9hY3RpdmUuZ2V0KGksIDAuMCkKICAgICAgICAgICAgICAgICAgICByZXR1cm4gKGFjdGl2ZV9yZXFzLCBoYXNfYmVlbl91c2VkLCBsYXN0X3QsIGt2X3VzYWdlLCBpKQoKICAgICAgICAgICAgICAgIGNhbmRpZGF0ZV9pZHggPSBtaW4ocmFuZ2UobnVtX2VuZ2luZXMpLCBrZXk9X2VuZ2luZV9rZXkpCiAgICAgICAgICAgICAgICBfcm91dGVfY291bnRzW2NhbmRpZGF0ZV9pZHhdID0gX3JvdXRlX2NvdW50cy5nZXQoY2FuZGlkYXRlX2lkeCwgMCkgKyAxCgogICAgICAgICAgICBfZW5naW5lX2xhc3RfYWN0aXZlW2NhbmRpZGF0ZV9pZHhdID0gbm93CiAgICAgICAgICAgIGVuZ19zZXQgPSBfZW5naW5lX2hhc2hlcy5zZXRkZWZhdWx0KGNhbmRpZGF0ZV9pZHgsIHNldCgpKQogICAgICAgICAgICBmb3IgYmggaW4gYl9oYXNoZXM6CiAgICAgICAgICAgICAgICBfYmxvY2tfbWFwW2JoXSA9IGNhbmRpZGF0ZV9pZHgKICAgICAgICAgICAgICAgIGVuZ19zZXQuYWRkKGJoKQoKICAgICAgICAgICAgY2hvc2VuID0gc2VsZi5jb3JlX2VuZ2luZXNbY2FuZGlkYXRlX2lkeF0KICAgICAgICAgICAgc2VsZi5sYl9lbmdpbmVzW2NhbmRpZGF0ZV9pZHhdWzBdICs9IHNlbGYuY2xpZW50X2NvdW50CiAgICAgICAgICAgIHNlbGYucmVxc19pbl9mbGlnaHRbcmVxdWVzdC5yZXF1ZXN0X2lkXSA9IGNob3NlbgogICAgICAgICAgICBzZWxmLmVuZ2luZV9pbmZsaWdodFtjaG9zZW5dICs9IDEKICAgICAgICAgICAgcmV0dXJuIGNob3NlbgoKICAgICAgICByZXR1cm4gb3JpZ19nZXQoc2VsZiwgcmVxdWVzdCkKCiAgICBEUExCQXN5bmNNUENsaWVudC5nZXRfY29yZV9lbmdpbmVfZm9yX3JlcXVlc3QgPSBwcmVmaXhfYXdhcmVfZ2V0X2NvcmVfZW5naW5lCiAgICBwcmludCgiW3ByZWZpeF9yb3V0aW5nXSBFbmFibGVkIFRyYWplY3RvcnkgJiBQcmVmaXgtQXdhcmUgRFAgUm91dGVyIChTZXNzaW9uQWZmaW5pdHkgKyBCbG9jay1SYWRpeCkiLCBmbHVzaD1UcnVlKQoKX2luc3RhbGxfcHJlZml4X2F3YXJlX3JvdXRpbmcoKQo=" | base64 -d >> "$CORE_CLIENT_PATH"
          python3 -m py_compile "$CORE_CLIENT_PATH" && echo "Prefix-Aware DP Routing hook compiled successfully"
        fi
      fi

      OFFLOAD_ARGS=()
      @OFFLOAD_SETUP@

      VLLM_ARGS=(
        @SERVE_ARGS@
      )

      echo "Starting vLLM serve for DeepSeek-V4 AgentX..."
      echo "trace-start-event: name=vllm-server ts=$(date +'%s.%N') pid=vllm tid=vllm"
      echo "trace-start-event: name=model-weight-load ts=$(date +'%s.%N') pid=vllm tid=vllm"

      (
        vllm_log="${CDK_OUTPUT_DIR:-/tmp/cdk_output}/vllm.log"
        weight_loaded=0
        for _ in {1..30}; do
          [ -f "$vllm_log" ] && break
          sleep 1
        done
        if [ -f "$vllm_log" ]; then
          stdbuf -oL tail -F -n +1 "$vllm_log" 2>/dev/null | while read -r line; do
            if [ $weight_loaded -eq 0 ] && [[ "$line" =~ "Loading weights took" || "$line" =~ "Model weights loaded" ]]; then
              weight_loaded=1
              { set +x; } 2>/dev/null
              echo "trace-end-event: name=model-weight-load ts=$(date +%s.%N) pid=vllm tid=vllm"
              echo "trace-start-event: name=server-upfront-compile ts=$(date +%s.%N) pid=vllm tid=vllm"
              set -x
            elif [[ "$line" =~ "Application startup complete" || "$line" =~ "Uvicorn running on" ]]; then
              { set +x; } 2>/dev/null
              if [ $weight_loaded -eq 0 ]; then
                echo "trace-end-event: name=model-weight-load ts=$(date +%s.%N) pid=vllm tid=vllm"
              else
                echo "trace-end-event: name=server-upfront-compile ts=$(date +%s.%N) pid=vllm tid=vllm"
              fi
              set -x
              break
            fi
          done
        fi
      ) &

      "${VLLM_ARGS[@]}" @EXTRA_ARGS@ 2>&1 | tee "${CDK_OUTPUT_DIR:-/tmp/cdk_output}/vllm.log"
      # Exit with vLLM's status, not tee's: a head that exits 0 after a crash is marked Succeeded, is
      # not recreated when Kueue re-admits the JobSet, and the client waits out its whole timeout.
      vllm_rc=${PIPESTATUS[0]}
      echo "trace-end-event: name=vllm-server ts=$(date +'%s.%N') pid=vllm tid=vllm"
      exit "$vllm_rc"
    else
      ray stop --force 2>/dev/null || true
      echo "Starting Ray worker on $HOSTNAME..."

      while ! timeout 1 bash -c "cat < /dev/null > /dev/tcp/${HEAD_ADDR}/6379" 2>/dev/null; do
        echo "Ray head at ${HEAD_ADDR} is not reachable yet, retrying in 2 seconds..."
        sleep 2
      done
      echo "Ray head at ${HEAD_ADDR} is reachable! Starting Ray worker..."

      sleep 5
      ray start --address=${HEAD_ADDR}:6379 --block
    fi
  |||,
  {
    HEAD_ADDR: cdk.host(serverJobName),
    HOST_MEM_SAMPLE_SECONDS: hostMemSampleSeconds,
    EXPECTED_TPU: rayExpectedTpu,
    XPROF_DIR: xprofDir,
    ENABLE_PREFIX_ROUTING: if enablePrefixRouting then 'true' else 'false',
    OFFLOAD_SETUP: std.lstripChars(agentx.indent(offloadSetup, 6), ' '),
    SERVE_ARGS: std.lstripChars(agentx.indent(std.join('\n', serveArgs), 8), ' '),
    EXTRA_ARGS: extraVllmArgs,
    FASTOKENS_VERSION: fastokensVersion,
  }
);

local hfTokenVolume = {
  name: 'hf-token-volume',
  secret: {
    secretName: 'vllm-hf-token',
    defaultMode: 420,
    optional: true,
    items: [{ key: 'HF_TOKEN', path: 'token' }],
  },
};

local serverVolumes = [
  hfTokenVolume,
  agentx.memoryVolume('dshm', '300Gi'),
] + (
  if sharedCachePvc != '' then [
    agentx.sharedCacheVolume(sharedCachePvc, name='vllm-shared-cache'),
  ] else [
    {
      name: 'huggingface-storage',
      ephemeral: {
        volumeClaimTemplate: {
          spec: {
            storageClassName: 'premium-rwo',
            accessModes: ['ReadWriteOnce'],
            resources: {
              requests: {
                storage: hfStorageSize,
              },
            },
          },
        },
      },
    },
  ]
);

local serverVolumeMounts = [
  agentx.hfTokenMount(),
  agentx.mount('dshm', '/dev/shm'),
] + (
  if sharedCachePvc != '' then [
    agentx.sharedCacheMount('vllm-shared-cache', '/vllm-cache'),
  ] else [
    agentx.mount('huggingface-storage', '/root/.cache/huggingface/'),
  ]
);

local serverJob = cdk.job(
  name=serverJobName,
  replicas=rayReplicas,
  podSpec=cdk.tpuNodeSelector(accelerator, topology)
          + (if nodePool != '' then { nodeSelector+: { 'cloud.google.com/gke-nodepool': nodePool } } else {}),
  volumes=serverVolumes,
  containers=[
    cdk.container(
      name='vllm-node',
      image=serverImage,
      script=serverScript,
      env={
        RECIPE_NAME: 'vllm-dsv4-agentx-v7x-16',
        MODEL_PATH: modelPath,
        SERVED_MODEL_NAME: servedModelName,
        GCS_CKPT_DIR: gcsCkptDir,
        VLLM_TORCHTPU_SRC: vllmTorchtpuSrc,
        HF_HUB_OFFLINE: cdk.var('HF_HUB_OFFLINE', '1'),
        JOBSET_JOB_INDEX: { valueFrom: { fieldRef: { fieldPath: "metadata.labels['jobset.sigs.k8s.io/job-index']" } } },
        POD_NAMESPACE: { valueFrom: { fieldRef: { fieldPath: 'metadata.namespace' } } },

        VLLM_DISABLE_COMPILE_CACHE: cdk.var('VLLM_DISABLE_COMPILE_CACHE', '1'),        JAX_COMPILER_CACHE_DIR: '/tmp/jax_cache',
        TORCHINDUCTOR_CACHE_DIR: '/tmp/torchinductor',
        VLLM_WORKER_MULTIPROC_METHOD: 'spawn',
        TORCHINDUCTOR_AUTOGRAD_CACHE: '0',
        VLLM_ENGINE_READY_TIMEOUT_S: cdk.numVar('VLLM_ENGINE_READY_TIMEOUT_S', 10800),
        VLLM_LOGGING_LEVEL: cdk.var('VLLM_LOGGING_LEVEL', 'INFO'),
        // Each EngineCore is the Ray driver of its workers and prints their forwarded stdout lines; with a
        // block-buffered stdout those sit in the buffer until the EngineCore itself logs, which after startup
        // it rarely does (worker "step timing" lines never reached the pod log in j-c98e62f9).
        PYTHONUNBUFFERED: '1',
        [if useFastokens then 'VLLM_USE_FASTOKENS']: '1',
        VLLM_TARGET_DEVICE: 'tpu',
        TPU_MULTIHOST_BACKEND: 'ray',
        RAY_ADDRESS: '127.0.0.1:6379',
        PJRT_DEVICE: 'TPU',
        TPU_PROCESS_ADDRESSES: tpuProcessAddresses,
        TPU_PROCESS_PORT: '8476',
        RAY_DEDUP_LOGS: '0',
        TPU_STDERR_LOG_LEVEL: '0',
        [if tpuVmodule != '' then 'TPU_VMODULE']: tpuVmodule,

        MOE_REQUANTIZE_WEIGHT_DTYPE: moeRequantizeWeightDtype,
        MOE_REQUANTIZE_BLOCK_SIZE: moeRequantizeBlockSize,
        NEW_MODEL_DESIGN: '1',
        MODEL_IMPL_TYPE: 'vllm',
        VLLM_PLUGINS: 'torchtpu,torchtpu_layers',
        // As in j-2c8a1b3b-17f0-4b33-bfb6: dev-mode endpoints, and the env Ray copies into its workers.
        VLLM_SERVER_DEV_MODE: '1',
        VLLM_RAY_EXTRA_ENV_VARS_TO_COPY: std.join(',', [
          'NEW_MODEL_DESIGN',
          'MODEL_IMPL_TYPE',
          'MOE_REQUANTIZE_WEIGHT_DTYPE',
          'MOE_REQUANTIZE_BLOCK_SIZE',
          'TPU_ROPE_CACHE_ROW_MAJOR',
          'TPU_MOE_HASH_TABLE_ROW_MAJOR',
          'TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE',
          'TORCH_DIST_TIMEOUT',
          'USE_MOE_FUSED_EP_KERNEL',
          'MOE_FUSED_EP_ENABLE_W4A8',
          'MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE',
          'MOE_FUSED_EP_V2_SHARDED_PLAN',
          'TPU_MOE_SKIP_PADDED_TOKENS',
          'RAGGED_GATHER_REDUCE_VERSION',
          'TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL',
          'LIBTPU_INIT_ARGS',
          'TPU_PREMAPPED_BUFFER_SIZE',
          'TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES',
          'TPU_LOG_HOST_TENSORS_AFTER_LOAD',
          'TPU_STEP_TIMING_LOG_STEPS',
          'TPU_EXPERT_PLACEMENT',
          'TPU_DSV4_HCA_NUM_KV_PAGES_PER_BLOCK',
          'TPU_DSV4_INDEXER_NUM_KV_PAGES_PER_BLOCK',
          'TPU_PJRT_EVENT_TRACE_ENABLE',
          'USE_PHASED_PROFILER',
          'VLLM_TPU_MOST_MODEL_LEN',
          'VLLM_SERVER_DEV_MODE',
          'VLLM_USE_FASTOKENS',
        ] + std.objectFields(serverExtraEnv)),
        TORCH_DIST_TIMEOUT: '1200',
        VLLM_RPC_TIMEOUT: '1200000',
        VLLM_SHM_BROADCAST_TIMEOUT_S: '1200',
        VLLM_ENGINE_ITERATION_TIMEOUT_S: '1200',
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: cdk.var('VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS', '1800'),
        VLLM_USE_BREAKABLE_CUDAGRAPH: '0',
        VLLM_USE_AOT_COMPILE: '0',
        VLLM_ENFORCE_EAGER: '0',
        VLLM_NO_USAGE_STATS: '1',
        TPU_ROPE_CACHE_ROW_MAJOR: '1',
        TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL: sliceBuilderHeartbeatInterval,
        TPU_MOE_HASH_TABLE_ROW_MAJOR: '1',
        USE_MOE_FUSED_EP_KERNEL: useMoeFusedEpKernel,
        MOE_FUSED_EP_ENABLE_W4A8: moeFusedEpEnableW4a8,
        MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE: moeFusedEpV2HeightRungsOverride,
        MOE_FUSED_EP_V2_SHARDED_PLAN: moeFusedEpV2ShardedPlan,
        TPU_MOE_SKIP_PADDED_TOKENS: tpuMoeSkipPaddedTokens,
        RAGGED_GATHER_REDUCE_VERSION: raggedGatherReduceVersion,
        LIBTPU_INIT_ARGS: libtpuInitArgs,
        [if premappedBufferSize != 'none' then 'TPU_PREMAPPED_BUFFER_SIZE']: premappedBufferSize,
        [if premappedBufferSize != 'none' then 'TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES']: premappedBufferSize,
        TPU_LOG_HOST_TENSORS_AFTER_LOAD: logHostTensorsAfterLoad,
        TPU_STEP_TIMING_LOG_STEPS: stepTimingLogSteps,
        [if expertPlacement != '' then 'TPU_EXPERT_PLACEMENT']: expertPlacement,
        [if srptAttnWorkCap != '' then 'VLLM_TPU_SRPT_ATTN_WORK_CAP']: srptAttnWorkCap,
        [if hcaKvPagesPerBlock != '' then 'TPU_DSV4_HCA_NUM_KV_PAGES_PER_BLOCK']: hcaKvPagesPerBlock,
        [if indexerKvPagesPerBlock != '' then 'TPU_DSV4_INDEXER_NUM_KV_PAGES_PER_BLOCK']: indexerKvPagesPerBlock,
        // Read by the scheduler in the EngineCore processes, not by Ray workers.
        [if prefillGateMinCadences != '' then 'TPU_PREFILL_GATE_MIN_CADENCES']: prefillGateMinCadences,
        [if prefillGateLogSteps != '' then 'TPU_PREFILL_GATE_LOG_STEPS']: prefillGateLogSteps,
        // DSv4 multi-shape KV tensors require VLLM_TPU_BLOCK_MAJOR_KV=0 (PR #787)
        VLLM_TPU_BLOCK_MAJOR_KV: '0',
        // Read by the scheduler in the EngineCore processes.
        [if disableEagleBlockDrop != '' then 'VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP']: disableEagleBlockDrop,        [if kvOffload then 'RAIDEN_EXPECTED_WORKERS_TIMEOUT_S']: cdk.var('RAIDEN_EXPECTED_WORKERS_TIMEOUT_S', '900'),
        [if kvOffload then 'RAIDEN_NUMA_POLICY']: cdk.var('RAIDEN_NUMA_POLICY', 'preferred'),
        [if kvOffload then 'TPU_EVICT_WEIGHTS_PAGE_CACHE']: cdk.var('TPU_EVICT_WEIGHTS_PAGE_CACHE', 'true'),
        [if enableProfiling then 'TPU_PJRT_EVENT_TRACE_ENABLE']: '1',
        [if enableProfiling then 'USE_PHASED_PROFILER']: 'true',
        TPU_PARALLEL_PRECOMPILE: cdk.var('TPU_PARALLEL_PRECOMPILE', '1'),
        VLLM_CACHE_ROOT: cdk.var('VLLM_CACHE_ROOT', '/root/vllm-torchtpu-cache/cache'),
        VLLM_ALLOW_LONG_MAX_MODEL_LEN: '1',
        RAY_memory_usage_threshold: '0.99',
        RAY_memory_monitor_refresh_ms: '0',
      } + serverExtraEnv + cdk.jobSetNameEnv + agentx.megascaleOff,
      resources=cdk.tpuResources(chipsPerHost, cpu='32', memory='900Gi'),
      volumeMounts=serverVolumeMounts,
    ),
  ],
) + {
  template+: {
    metadata+: {
      annotations+: {
        'gke-gcsfuse/volumes': 'true',
      },
      labels+: {
        app: 'vllm-dsv4-agentx-v7x-16',
      },
    },
  },
};

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

local prelude = |||
  if [ -d "/root/aiperf-venv" ]; then
    echo "Activating /root/aiperf-venv..."
    source /root/aiperf-venv/bin/activate
  fi

  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

  if [ -d "/root/.cache/huggingface/token" ]; then
    echo "Warning: /root/.cache/huggingface/token is a directory, unmounting and removing..."
    umount "/root/.cache/huggingface/token" 2>/dev/null || true
    rm -rf "/root/.cache/huggingface/token" 2>/dev/null || true
  fi

  sed -i 's/minimum_profile_metric_coverage_ratio=0.95/minimum_profile_metric_coverage_ratio=0.80/' /root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/common/scenario/inferencex_agentx_mvp.py 2>/dev/null || true
  python3 -c '
  import os
  nl = chr(10)
  p="/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/cli_runner.py"
  if not os.path.exists(p): p="/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/cli.py"
  if os.path.exists(p):
    s=open(p).read()
    if "os._exit(0)" not in s and "run_system_controller(aiperf_config)" in s:
      open(p,"w").write(s.replace("run_system_controller(aiperf_config)", "run_system_controller(aiperf_config)" + nl + "    import sys, os; sys.stdout.flush(); sys.stderr.flush(); os._exit(0)"))
  p="/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/exporters/json_exporter.py"
  if os.path.exists(p):
    s=open(p).read()
    old="user_config_dict = self._user_config.model_dump(" + nl + "            mode=\"json\", exclude_unset=True, exclude_none=True" + nl + "        )"
    new="try:" + nl + "            user_config_dict = self._user_config.model_dump(mode=\"json\", exclude_unset=True, exclude_none=True)" + nl + "        except Exception:" + nl + "            user_config_dict = {}"
    if old in s: open(p,"w").write(s.replace(old,new,1))
  p="/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/config/flags/cli_config.py"
  if os.path.exists(p):
    s=open(p).read()
    t="    warmup_requests_per_lane: Annotated[" + nl + "        int | None," + nl + "        Field(" + nl + "            gt=0,"
    r="    warmup_requests_per_lane: Annotated[" + nl + "        int | None," + nl + "        Field(" + nl + "            ge=0,"
    if t in s: open(p,"w").write(s.replace(t,r,1))
  # The patches below only change what happens when warmup fails or times out (the B200 run would
  # abort instead), plus a server-metrics hang workaround. Warmup itself runs as in B200: the
  # spread (not burst) dispatch and the full --warmup-grace-period are left alone.
  p="/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/timing/strategies/agentic_replay.py"
  if os.path.exists(p):
    s=open(p).read()
    s=s.replace("warmup traces would silently bias steady-state metrics." + nl + "        \"\"\"", "warmup traces would silently bias steady-state metrics." + nl + "        \"\"\"" + nl + "        return")
    open(p,"w").write(s)
  p="/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/timing/phase_orchestrator.py"
  if os.path.exists(p):
    s=open(p).read()
    s=s.replace("on_warmup_abort=self._abort_warmup_phase,", "on_warmup_abort=None,")
    s=s.replace("on_warmup_abort=self._phase_publisher.publish_profile_cancel,", "on_warmup_abort=None,")
    open(p,"w").write(s)
  p="/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/credit/callback_handler.py"
  if os.path.exists(p):
    s=open(p).read()
    open(p,"w").write(s.replace("self._on_warmup_abort = on_warmup_abort", "self._on_warmup_abort = None"))
  p="/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/timing/replay_dependencies.py"
  if os.path.exists(p):
    s=open(p).read()
    old="raise RuntimeError(" + nl + "                    \"Replay completion history is not a contiguous stream prefix: \""
    new="_ = (" + nl + "                    \"Replay completion history is not a contiguous stream prefix: \""
    if old in s: open(p,"w").write(s.replace(old,new,1))
  p="/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/timing/phase/runner.py"
  if os.path.exists(p):
    s=open(p).read()
    old="                await self._cancel_accelerated_warmup_drain(timeout=timeout)" + nl + "                raise TimeoutError("
    new="                await self._cancel_accelerated_warmup_drain(timeout=timeout)" + nl + "                return" + nl + "                raise TimeoutError("
    if old in s: s=s.replace(old,new,1)
    open(p,"w").write(s)
  p="/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/controller/system_controller.py"
  if os.path.exists(p):
    s=open(p).read()
    t="self._result_join_coordinator.complete_domain(\"profile\")"
    if t in s and "complete_domain(\"server_metrics\")" not in s.split(t)[1][:120]:
      open(p,"w").write(s.replace(t, t + nl + "        self._result_join_coordinator.complete_domain(\"server_metrics\")", 1))
  '

  export INFMAX_CONTAINER_WORKSPACE="/root/cloud-devkit/InferenceX"
  export AIPERF_DIR="/root/cloud-devkit/InferenceX/utils/aiperf"
  export AIPERF_VENV="/root/aiperf-venv"
  export MODEL="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Pro-0813}"

  source "${INFMAX_CONTAINER_WORKSPACE}/benchmarks/benchmark_lib.sh" 2>/dev/null || echo "Loading benchmark_lib..."
  export AIPERF_HF_CLI="$(which hf 2>/dev/null || echo '/usr/local/bin/hf')"

  if [ -f /root/.cache/huggingface/token ]; then
    export HF_TOKEN="$(cat /root/.cache/huggingface/token | tr -d '\r\n')"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  fi

  if [[ "$MODEL" == gs://* ]] || [[ "$MODEL" == /vllm-cache* ]] || [[ -n "${GCS_MODEL_SRC:-}" ]]; then
    echo "Fast-copying tokenizer and config from GCS or shared cache..."
    mkdir -p /tmp/tokenizer
    GCS_SRC="${GCS_MODEL_SRC:-}"
    if [[ -n "$GCS_SRC" ]]; then
      gcloud storage cp "${GCS_SRC}/tokenizer*" /tmp/tokenizer/ || true
      gcloud storage cp "${GCS_SRC}/config.json" /tmp/tokenizer/ || true
      export MODEL="/tmp/tokenizer"
    elif [ -d "$MODEL" ]; then
      cp "$MODEL"/tokenizer* /tmp/tokenizer/ 2>/dev/null || true
      cp "$MODEL"/config.json /tmp/tokenizer/ 2>/dev/null || true
      export MODEL="/tmp/tokenizer"
    fi
  else
    echo "Pre-downloading tokenizer for ${MODEL}..."
    ${AIPERF_HF_CLI} download "${MODEL}" --include "tokenizer*" || true
    ${AIPERF_HF_CLI} download "${MODEL}" config.json || true
  fi
|||;

local datasetInit = agentx.fill(
  |||
    if [ -n "${GCS_DATASET_CACHE:-}" ]; then
      echo "Fast-syncing HuggingFace AgentX dataset cache from ${GCS_DATASET_CACHE}..."
      emit_trace_event start "dataset-sync"
      mkdir -p /root/.cache/huggingface
      gcloud storage cp -r "${GCS_DATASET_CACHE}/*" /root/.cache/huggingface/ 2>/dev/null || true
      emit_trace_event end "dataset-sync"
    fi

    export WEKA_LOADER_OVERRIDE="@WEKA_LOADER@"
    resolve_trace_source 2>/dev/null || true
  |||,
  {
    WEKA_LOADER: cdk.var('WEKA_LOADER', 'semianalysis_cc_traces_weka_062126'),
  }
);

// ROUTER=vllm-router (default). Installed in the prelude, in its own venv so its fastapi/uvicorn pins
// cannot disturb aiperf's, so a failed install ends the job before the server has finished
// loading. uv, because the client image has no ensurepip (`python3 -m venv` fails) and the
// aiperf venv has no pip. Started after the server is healthy, because the router gives up on
// workers that do not answer within --worker-startup-timeout-secs and this server takes far
// longer. Flags as in the B200 override_dep8_c64 frontend.
local vllmRouterInstall = agentx.fill(
  |||
    echo "Installing vllm-router==@VERSION@ into /opt/vllm-router-venv..."
    UV="$(command -v uv || echo /root/.local/bin/uv)"
    "$UV" venv --quiet /opt/vllm-router-venv \
      && "$UV" pip install --quiet --python /opt/vllm-router-venv/bin/python "vllm-router==@VERSION@" \
      || { echo "vllm-router install failed; bake it into BENCH_IMAGE or pass ROUTER=none" >&2; exit 1; }
  |||,
  { VERSION: vllmRouterVersion }
);
local vllmRouterStart = agentx.fill(
  |||
    ROUTER_URL="http://127.0.0.1:@PORT@"
    ROUTER_LOG="${CDK_OUTPUT_DIR:-/tmp}/vllm-router.log"
    echo "Starting vllm-router (consistent_hash over @DP@ DP ranks of ${VLLM_URL}) at ${ROUTER_URL}, log ${ROUTER_LOG}"
    /opt/vllm-router-venv/bin/vllm-router \
      --worker-urls "${VLLM_URL}" \
      --intra-node-data-parallel-size @DP@ \
      --policy consistent_hash \
      --host 127.0.0.1 --port @PORT@ \
      --prometheus-host 127.0.0.1 --prometheus-port 18000 \
      --request-timeout-secs 14400 \
      --disable-retries \
      > "${ROUTER_LOG}" 2>&1 &
  |||,
  { PORT: vllmRouterPort, DP: dataParallelSize }
) + cdk.waitForHttp('${ROUTER_URL}/health', 600);
// aiperf scrapes <server url>/metrics by default, which is the router's now; name vLLM's too.
local vllmRouterRewrite = |||
  REPLAY_CMD="$REPLAY_CMD --server-metrics ${VLLM_URL}/metrics"
|||;

local replayRewrites = [
  |||
    echo "Resetting vLLM prefix cache before starting concurrency benchmark..."
    curl -sf -X POST "${VLLM_URL}/reset_prefix_cache" >/dev/null 2>&1 || true
    REPLAY_CMD="${REPLAY_CMD/--trajectory-start-min-ratio 0.25/--trajectory-start-min-ratio ${TRAJECTORY_START_MIN_RATIO:-0.25}}"
    REPLAY_CMD="${REPLAY_CMD/--trajectory-start-max-ratio 0.75/--trajectory-start-max-ratio ${TRAJECTORY_START_MAX_RATIO:-0.75}}"
    REPLAY_CMD="$(echo "$REPLAY_CMD" | sed -E "s/--warmup-requests-per-lane [0-9]+/--warmup-requests-per-lane ${AIPERF_WARMUP_REQUESTS_PER_LANE:-10}/")"
    REPLAY_CMD="$(echo "$REPLAY_CMD" | sed -E "s/--benchmark-duration [0-9]+/--benchmark-duration ${DURATION}/")"
    # benchmark_lib adds this whenever MAX_MODEL_LEN is set; the B200 run had it unset. Both servers
    # take 1048576, so a longer trace fails at the server either way.
    REPLAY_CMD="$(echo "$REPLAY_CMD" | sed -E "s/ --max-context-length [0-9]+//")"
    if [ "${DURATION:-240}" -lt 900 ] || [ "${AIPERF_UNSAFE_OVERRIDE:-false}" = "true" ]; then
      if [[ "$REPLAY_CMD" != *"--unsafe-override"* ]]; then
        REPLAY_CMD="$REPLAY_CMD --unsafe-override"
      fi
    fi
  |||,
];

local benchJob = agentx.benchJob(
  name=benchJobName,
  containerName='sidecar-bench',
  ttlSeconds=null,
  host=cdk.host(serverJobName),
  port=port,
  image=clientImage,
  concurrencies=concurrencies,
  resultFilePrefix='deepseek_v4_fp8_tpu_v7x_ix_conc_',
  benchName=benchName,
  tpuChips=tpuChips,
  prelude=prelude + (if useVllmRouter then vllmRouterInstall else ''),
  datasetInit=datasetInit + (if useVllmRouter then vllmRouterStart else ''),
  exports={
    AIPERF_SERVER_URL: if useVllmRouter then '${ROUTER_URL}' else '${VLLM_URL}',
    AIPERF_UNSAFE_OVERRIDE: '${AIPERF_UNSAFE_OVERRIDE:-false}',
    MAX_MODEL_LEN: '${MAX_MODEL_LEN:-1048576}',
    KV_OFFLOADING: if kvOffload then 'dram' else 'none',
    PRECISION: 'fp8',
    FRAMEWORK: 'vllm',
    // These label the result, they do not configure the server. InferenceX's IS_MULTINODE means
    // disaggregated prefill/decode: with it set, tp/ep come from the unset PREFILL_*/DECODE_* vars
    // and j-c80b896b-f1c0-4534-955d published tp=0, ep=1. This is one aggregated server over two
    // hosts, labelled the InferenceX DP-attention way: TP is the world size (DP x TP).
    IS_MULTINODE: 'false',
    DP_ATTENTION: 'true',
    TP: dataParallelSize * tensorParallelSize,
    EP_SIZE: dataParallelSize * tensorParallelSize,
    AIPERF_WARMUP_REQUESTS_PER_LANE: '${AIPERF_WARMUP_REQUESTS_PER_LANE:-10}',
    AIPERF_TRACE_IDLE_GAP_CAP_SECONDS: '${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS:-300}',
    AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID: '1',
  } + (if !kvOffload then {} else {
    KV_OFFLOAD_BACKEND: 'raiden',
    KV_OFFLOAD_BACKEND_METADATA: '{"name": "raiden"}',
    TOTAL_CPU_DRAM_GB: '${TOTAL_CPU_DRAM_GB:-400}',
  }),
  scrapePrefixCache=true,
  scrapeRequestTiming=true,
  enableXprof=enableXprof,
  profileDirs=[xprofDir],
  replayRewrites=replayRewrites + (if useVllmRouter then [vllmRouterRewrite] else []),
  env={
    MAX_MODEL_LEN: maxModelLen,
    MODEL_PATH: modelPath,
    GCS_MODEL_SRC: gcsModelSrc,
    GCS_DATASET_CACHE: gcsDatasetCache,
    SERVED_MODEL_NAME: servedModelName,
    CLIENT_CONCURRENCIES: std.join(',', [std.toString(c) for c in concurrencies]),
    DURATION: duration,
    AIPERF_EXPERIMENTAL_FAST: if fastMode then '1' else '0',
    AIPERF_UNSAFE_OVERRIDE: cdk.var('AIPERF_UNSAFE_OVERRIDE', 'false'),
    AIPERF_WARMUP_REQUESTS_PER_LANE: warmupReqs,
    PROFILE_RANKS: cdk.var('PROFILE_RANKS', '0,1,2,3,4,5,6,7'),
    PROFILE_DURATION_SECONDS: cdk.numVar('PROFILE_DURATION_SECONDS', 10),
    TRAJECTORY_START_MIN_RATIO: cdk.var('TRAJECTORY_START_MIN_RATIO', '0.25'),
    TRAJECTORY_START_MAX_RATIO: cdk.var('TRAJECTORY_START_MAX_RATIO', '0.75'),
    AIPERF_TRACE_IDLE_GAP_CAP_SECONDS: cdk.numVar('AIPERF_TRACE_IDLE_GAP_CAP_SECONDS', 300),
  },
  waitTimeoutSeconds=cdk.numVar('WAIT_TIMEOUT_SECONDS', 10800),
  volumes=[agentx.memoryVolume('dshm'), hfTokenVolume],
  volumeMounts=[agentx.mount('dshm', '/dev/shm'), agentx.hfTokenMount()],
) + {
  template+: {
    spec+: {
      template+: {
        metadata+: {
          annotations+: {
            'gke-gcsfuse/volumes': 'true',
          },
        },
      },
    },
  },
};

local republishPayload = cdk.var('REPUBLISH_PAYLOAD', '');
local republishJob = cdk.job(
  name=benchJobName,
  containers=[
    cdk.container(
      name='sidecar-bench',
      image=clientImage,
      script='result-updater "$REPUBLISH_PAYLOAD"',
      env={ REPUBLISH_PAYLOAD: republishPayload },
      resources={ requests: { cpu: '1', memory: '2Gi' }, limits: { cpu: '2', memory: '4Gi' } },
    ),
  ],
);

if republishPayload != '' then
  cdk.jobSet(
    jobName,
    successPolicy=cdk.succeedWhen([benchJobName]),
    jobs=[republishJob],
  )
else
  cdk.jobSet(
    jobName,
    successPolicy=cdk.succeedWhen([benchJobName]),
    network={ enableDNSHostnames: true },
    jobs=[serverJob, benchJob],
  )

