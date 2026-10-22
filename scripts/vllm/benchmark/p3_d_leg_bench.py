#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode leg step time of Qwen3.5-397B under a chosen layout (phase 3 harness).

Port of agentx-p2-e1 scripts/vllm/benchmark/d_leg_bench.py onto the vLLM 0.31
/ torch_tpu 0.2.0rc1 tree, with the harness defects of that version fixed and
the phase 4 axes exposed:

  * running count: every request gets max_tokens sized for the whole window at
    K+1 tokens per step (the old 1.5 x acceptance budget let requests finish
    early under the real sampler) with ignore_eos, max_num_seqs and the
    megakernel running cap are raised to the case's REQS, and the scheduler
    hook records the running count of every step; the summary reports whether
    r was held (held_running) and the number of steps that were not at r,
    which are excluded from the step time;
  * K = 0 to 4 (--mtp), synthetic acceptance (--acceptance synthetic
    --synthetic-acceptance 3.39) or the real rejection sampler
    (--acceptance real, acceptance measured from sampled tokens per step);
  * request sampling: greedy by default, or the served requests'
    --temperature / --top-k / --top-p;
  * batch definition: --batch-mode replica (REQS per engine replica) or
    per_chip (REQS per chip, multiplied by the replica's chips); under
    --data-parallel-size N every one of the N engines holds REQS requests
    (request ids carry the engine index, the client routes by it), step
    records are grouped by engine process, and the summary reports the
    per-replica step and the per-chip token rate over all replicas;
  * layout: --tensor-parallel-size, --decode-context-parallel-size,
    --pipeline-parallel-size, --data-parallel-size, --expert-parallel,
    --megakernel and --mtp-mk switches, --num-hosts;
  * timing: queue, init, load, compile, warmup and timed windows separately,
    per-step intervals (steps.jsonl), one xprof window per case with --profile.

If the MTP megakernel warmup hangs on torch_tpu 0.2.0rc1 (the other session's
_warmup_spec_decode investigation), run with --mtp-mk 0 and say so in the
record; that is a configuration note, never a performance result.

    python scripts/vllm/benchmark/p3_d_leg_bench.py --case 92160:8 --case 92160:28 \
        --mtp 3 --acceptance synthetic --synthetic-acceptance 3.39

Results: <out_dir>/result.json, steps.jsonl, env.json and a ``P3_D_LEG_RESULT
{...}`` line on stdout.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3_bench_common as common  # noqa: E402

REPO_ROOT = common.REPO_ROOT

# Decode container env of the agentx-397b-pd-baseline cell on this tree
# (scripts/agentx-cell.sh, DECODE_* prefix stripped) layered over the
# j-776e8d47 container env captured by agentx-p2-e1, minus the KV transfer,
# compile-cache path and frontend knobs, which do not reach the forward step.
D_LEG_ENV = {
    "ATTN_BUCKETIZED_NUM_REQS": "true",
    "ATTN_CUSTOM_NUM_REQS_BUCKETS": "8,16,32,64",
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_OFFLINE": "1",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PLATFORMS": "tpu,cpu",
    "MODEL_IMPL_TYPE": "vllm",
    "MOE_FUSED_EP_KERNEL_IMPL": "v3",
    "MOE_FUSED_EP_KERNEL_MIN_TOKENS": "1024",
    "MOE_LOCAL_EXPERT_AFFINITY_EPSILON": "0.01",
    "NEW_MODEL_DESIGN": "0",
    "OMP_NUM_THREADS": "8",
    "ONEHOT_MOE_PERMUTE_THRESHOLD": "2048",
    "PJRT_DEVICE": "TPU",
    "RAGGED_GATED_DELTA_RULE_IMPL": "chunked_kernel_v3_pd",
    "RAGGED_GATHER_REDUCE_VERSION": "v3",
    "SKIP_JAX_PRECOMPILE": "1",
    "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
    "TORCHINDUCTOR_MAX_WORKERS": "4",
    "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK": "0",
    "TPUMISC_HMA_CHECKSUM_TRACE": "0",
    "TPU_BACKEND_TYPE": "jax",
    "TPU_ENABLE_GDN_DYNAMIC_TILING": "0",
    "TPU_EXPERT_PLACEMENT": str(
        common.PLACEMENT_DIR / "expert_placement_tp8_ep8_agentx_corr_c32.json"
    ),
    "TPU_FUSE_POST_ATTN_NORM": "1",
    "TPU_FUSE_PRE_ATTN_NORM": "1",
    "TPU_GDN_CONV_QK_PAIR_LAYOUT": "1",
    "TPU_GDN_CONV_STATE_TILE_PAD": "1",
    "TPU_GDN_FUSED_NORM_QUANT": "1",
    "TPU_KERNEL_PAGE_TOKENS": "128",
    "TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER": "1",
    "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS": "33",
    "TPU_MOE_LOWTOK": "0",
    "TPU_MOE_LOWTOK_MAX_TOKENS": "256",
    "TPU_MOE_SEQUENCE_PARALLEL": "1",
    "TPU_MOE_SKIP_PADDED_TOKENS": "1",
    "TPU_PJRT_EVENT_TRACE_ENABLE": "1",
    "TPU_PREFILL_SKIP_VERIFY_WARMUP": "1",
    "TPU_QWEN35_AUTO_DEFAULTS": "1",
    "TPU_ROPE_CACHE_TRUNCATE": "1",
    "TPU_RPA_FOLD_KV_HEAD_DIM": "0",
    "TPU_SCHED_FINISH_DECODE_STEPS": "0",
    "TPU_SCHED_MK_MAX_RUNNING": "24",
    "TPU_SCHED_WARM_STREAK": "0",
    "TPU_SMALL_ALL_REDUCE": "1",
    "TPU_SP_FP8_GATHER": "1",
    "TPU_SP_MIN_ROWS": "512",
    "TPU_TOKEN_BUCKET_EXTRA": "4,8,48,2048",
    "TPU_TOKEN_BUCKET_LINEAR_INTERVAL": "16",
    "TPU_TOKEN_BUCKET_LINEAR_UNTIL": "64",
    "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1",
    "USE_BATCHED_RPA_KERNEL": "1",
    "USE_BATCHED_RPA_LONGCTX": "1",
    "USE_FUSED_MOE_GMM": "1",
    "USE_MOE_COUNTING_SORT": "1",
    "USE_MOE_FUSED_EP_KERNEL": "1",
    "USE_MOE_FUSED_TP_AR": "0",
    "USE_MOE_SPARSE_CORE": "1",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_ENGINE_READY_TIMEOUT_S": "2700",
    "VLLM_KV_CACHE_LAYOUT": "HND",
    "VLLM_LOGGING_LEVEL": "INFO",
    "VLLM_MOE_CHUNK_SIZE": "256",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_TARGET_DEVICE": "tpu",
    "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP": "1",
    "VLLM_TPU_QWEN35_MEGAKERNEL": "1",
    "VLLM_TPU_QWEN35_MK_BUCKETS": "0",
    "VLLM_TPU_QWEN35_MK_FUSED_POOL_IO_MAX_B": "128",
    "VLLM_TPU_QWEN35_MK_MODE": "ep",
    "VLLM_TPU_QWEN35_MK_SPLIT_WIDE": "1",
    "VLLM_TPU_QWEN35_MTP_MK": "1",
    "VLLM_TPU_QWEN35_MTP_MK_VERIFY_BUCKETS": "4,8,12,16,20,24,32,40,48,56,64",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_XLA_CHECK_RECOMPILATION": "0",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    # The recipe's string; vllm_torchtpu.env_override rewrites it the same way
    # it does in the server.
    "LIBTPU_INIT_ARGS": (
        " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
        " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
        " --xla_tpu_ars_combiner_threshold_in_bytes=0"
        " --xla_tpu_enable_async_collective_merger=false"
        " --xla_enable_async_all_gather=true"
        " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
        " --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144"
    ),
}


@dataclasses.dataclass
class Case:
    ctx: int
    reqs: int
    real: bool = False  # with --decode-fill: prefill this case for real (A/B against the fill)

    @classmethod
    def parse(cls, text: str) -> Case:
        parts = text.split(":")
        if len(parts) == 3 and parts[2] == "real":
            return cls(int(parts[0]), int(parts[1]), True)
        ctx, reqs = (int(x) for x in parts)
        return cls(ctx, reqs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--case", action="append", type=Case.parse,
        help="CTX:REQS, repeatable (default 92160:8 and 92160:24); REQS per replica, "
        "or per chip with --batch-mode per_chip",
    )
    p.add_argument(
        "--prompt-file", default="",
        help="JSONL of {\"prefix\": [ids]} (frozen input contexts); replaces the random "
        "CTX prompts, records cycled over REQS",
    )
    p.add_argument("--batch-mode", choices=("replica", "per_chip"), default="replica")
    p.add_argument(
        "--shapes-from", default="",
        help="per_request.csv(.gz) of the frozen input: contexts (and with --hold-running "
        "replenish the output lengths) of the requests come from real rows instead of CTX",
    )
    p.add_argument("--shapes-select", choices=("sample", "quantiles"), default="sample")
    p.add_argument("--shapes-subset", choices=("all", "first_turn", "later_turn", "cold"), default="all")
    p.add_argument("--shapes-seed", type=int, default=0)
    p.add_argument(
        "--hold-running", choices=("window", "replenish"), default="window",
        help="window: every request's max_tokens covers the whole measured window so none "
        "finishes (steady r by construction); replenish: max_tokens is the real output length "
        "(--shapes-from, ignore_eos) and a finished request is replaced at once by a new one, "
        "so r is held while outputs follow the input distribution; steps that carry the "
        "replacement's prefill are counted and left out of the pure decode step time",
    )
    p.add_argument("--cores-per-chip", type=int, default=2, help="tpu7x: 2 TensorCores per chip")
    p.add_argument("--warm-steps", type=int, default=20)
    p.add_argument("--prof-steps", type=int, default=30)
    p.add_argument("--meas-steps", type=int, default=100)
    p.add_argument("--windows", type=int, default=1,
                   help="measured windows per case on the resident batch (one prefill); each window "
                   "is reported as its own case record (a repeat)")
    p.add_argument("--profile", action="store_true", help="profile PROF steps per case")
    p.add_argument("--dump-tokens", type=int, default=0,
                   help="write the first N generated token ids of every request to <out>/tokens.jsonl "
                   "(numerics check: compare two layouts' greedy outputs on the same prompts)")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=0, help="0: the largest case's REQS")
    # The serving D leg's sizes (4 .. 256) plus the prefill chunks up to
    # --max-num-batched-tokens: this engine prefills its prompts itself, and the
    # runner sizes its per-token buffers from the largest compile size.
    p.add_argument("--compile-sizes", default="4,8,16,24,32,48,64,96,128,256,512,1024,2048")
    p.add_argument("--block-size", type=int, default=640)
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--kv-cache-memory-bytes", type=int, default=0,
                   help="KV pool bytes per rank; 0 leaves the platform's auto sizing, which assumes "
                   "TP8 dense weights per rank")
    p.add_argument("--mtp", type=int, default=3, help="MTP draft tokens K, 0 to 4 (0: off)")
    p.add_argument("--acceptance", choices=("synthetic", "real"), default="synthetic")
    p.add_argument("--synthetic-acceptance", type=float, default=3.39,
                   help="synthetic acceptance length (used with --acceptance synthetic)")
    # A served request without sampling fields gets the checkpoint's generation_config
    # (Qwen3.5-397B: --temperature 0.6 --top-k 20 --top-p 0.95), which verifies drafts on the
    # sampling path instead of the greedy one.
    p.add_argument("--temperature", type=float, default=0.0, help="request temperature (0: greedy)")
    p.add_argument("--top-k", type=int, default=0, help="request top_k (0: off)")
    p.add_argument("--top-p", type=float, default=1.0, help="request top_p")
    p.add_argument(
        "--scheduler-cls",
        default="vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler",
        help="empty: vLLM's scheduler",
    )
    p.add_argument("--no-async-scheduling", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--decode-context-parallel-size", type=int, default=1)
    p.add_argument("--pipeline-parallel-size", type=int, default=1)
    p.add_argument("--data-parallel-size", type=int, default=1)
    p.add_argument("--cp-kv-cache-interleave-size", type=int, default=1)
    p.add_argument("--expert-parallel", type=int, choices=(0, 1), default=1)
    p.add_argument("--megakernel", type=int, choices=(0, 1), default=1,
                   help="VLLM_TPU_QWEN35_MEGAKERNEL")
    p.add_argument("--mtp-mk", type=int, choices=(0, 1), default=1,
                   help="VLLM_TPU_QWEN35_MTP_MK (0 works around the 0.2.0rc1 warmup hang)")
    p.add_argument("--num-hosts", type=int, default=1)
    p.add_argument("--distributed-executor-backend", default="")
    p.add_argument("--routing", choices=("hash", "model", "trace", "idx"), default="model",
                   help="fused EP routing override for the prefill of the prompts and any "
                   "non-megakernel decode path (the megakernel routes on its own)")
    p.add_argument("--routing-file", default="")
    p.add_argument("--weights", choices=("synthetic", "real"), default="synthetic")
    p.add_argument("--load-format", default="")
    p.add_argument("--kv-cache-dtype", default="fp8")
    p.add_argument("--quantization", default="fp8")
    p.add_argument("--env", action="append", default=[],
                   help="KEY=VALUE set after D_LEG_ENV; KEY= with no value unsets KEY")
    p.add_argument("--engine-arg", action="append", default=[],
                   help="KEY=VALUE set on EngineArgs after the CLI parse, before the engine is "
                   "built; VALUE is a Python literal (True, 3, {...}) or else a string")
    p.add_argument("--model-dir", default=common.MODEL_CONFIG_DIR)
    p.add_argument("--out-root", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--cache-root", default="/workspace/tmp/p3-harness/compile-cache/d-leg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--decode-fill", action="store_true",
        help="skip the real prefill: a DecodeBenchConnector (p3_decode_fill.py) reports the prompt "
        "as loaded and fills its KV blocks and GDN state with a constant; cases written "
        "CTX:REQS:real still prefill for real (in-job A/B)",
    )

    a = p.parse_args(argv)
    if not a.case:
        a.case = [Case(92160, 8), Case(92160, 24)]
    if not 0 <= a.mtp <= 4:
        p.error("--mtp must be 0 to 4")
    if a.routing == "trace" and not a.routing_file:
        p.error("--routing trace needs --routing-file")
    if a.hold_running == "replenish" and not a.shapes_from:
        p.error("--hold-running replenish needs --shapes-from (the output length distribution)")
    for item in a.engine_arg:
        if "=" not in item or not item.split("=", 1)[0]:
            p.error(f"--engine-arg {item!r}: expected KEY=VALUE")
    return a


def engine_arg_overrides(items: list[str]) -> dict:
    """Parses --engine-arg KEY=VALUE items into EngineArgs field values."""
    out = {}
    for item in items:
        key, raw = item.split("=", 1)
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def ranks_per_replica(a: argparse.Namespace) -> int:
    # DCP ranks are a subset of the TP ranks in vLLM, so they do not add ranks.
    return a.tensor_parallel_size * a.pipeline_parallel_size


def case_reqs(a: argparse.Namespace, case: Case) -> int:
    if a.batch_mode == "per_chip":
        chips = max(1, ranks_per_replica(a) // a.cores_per_chip)
        return case.reqs * chips
    return case.reqs


def layout_tag(a: argparse.Namespace) -> str:
    parts = [f"tp{a.tensor_parallel_size}"]
    if a.decode_context_parallel_size > 1:
        parts.append(f"dcp{a.decode_context_parallel_size}")
    if a.pipeline_parallel_size > 1:
        parts.append(f"pp{a.pipeline_parallel_size}")
    if a.data_parallel_size > 1:
        parts.append(f"dp{a.data_parallel_size}")
    parts.append("ep" if a.expert_parallel else "noep")
    parts.append(f"k{a.mtp}")
    parts.append("mk" if a.megakernel else "nomk")
    if a.megakernel and not a.mtp_mk:
        parts.append("mtpmk0")
    parts.append(a.acceptance)
    return "-".join(parts)


def build_env(a: argparse.Namespace, out_dir: str) -> tuple[dict[str, str], list[str]]:
    env = dict(D_LEG_ENV)
    env.update(common.topology_env(a.num_hosts))
    unset: list[str] = []
    max_reqs = max(case_reqs(a, c) for c in a.case)
    env["TPU_SCHED_MK_MAX_RUNNING"] = str(max(max_reqs, int(env["TPU_SCHED_MK_MAX_RUNNING"])))
    env["VLLM_TPU_QWEN35_MEGAKERNEL"] = str(a.megakernel)
    env["VLLM_TPU_QWEN35_MTP_MK"] = str(a.mtp_mk if (a.megakernel and a.mtp > 0) else 0)
    if a.tensor_parallel_size != 8 or not a.expert_parallel or a.pipeline_parallel_size > 1:
        env.pop("TPU_EXPERT_PLACEMENT", None)
        unset.append("TPU_EXPERT_PLACEMENT")
    if a.pipeline_parallel_size > 1:
        env.update(TPU_MOE_SEQUENCE_PARALLEL="0", TPU_SP_FP8_GATHER="0", USE_MOE_FUSED_EP_KERNEL="0")
    if a.kv_cache_memory_bytes > 0:
        # The platform replaces a requested pool size with its TP8-calibrated
        # auto value unless told the request is deliberate.
        env["TPU_KV_BYTES_EXPLICIT"] = "1"
    if a.distributed_executor_backend == "ray":
        env["TPU_MULTIHOST_BACKEND"] = "ray"
    env, more = common.parse_env_overrides(a.env, env)
    if a.decode_fill:
        # The fill connector runs as kv_both: kv_producer would clamp MTP to K=1
        # (tpu_platform.py prefill clamp). PrefixAwareScheduler treats any connector
        # that is not a producer as a KV consumer; pin the settings a connector-less
        # engine uses so the decode step matches the control harness: hybrid-state
        # retention on (also sets retention interval 0 / match unit 128 in the
        # platform), no consumer KV prefetch reserve, SRPF on, the non-disagg step
        # and cold-chunk caps.
        total_cap = max(1024, a.max_num_batched_tokens // 2)
        pinned = {
            "TPU_HYBRID_STATE_RETENTION": "1",
            "TPU_SCHED_KV_PREFETCH_WAITING": "0",
            "TPU_SCHED_SRPF_RUNNING": "1",
            "TPU_SCHED_STEP_TOTAL_CAP": str(total_cap),
            "TPU_SCHED_COLD_CHUNK_CAP": str(max(512, total_cap // 2)),
        }
        env.update(pinned)
        env["P3_DECODE_FILL_PINNED"] = ",".join(f"{k}={v}" for k, v in pinned.items())
    # an --env KEY=VALUE wins over the layout's unset of KEY (e.g. a TP16 placement table)
    unset = [k for k in unset if k not in env] + more
    src = common.source_identity()
    key = common.cache_key([
        *sorted(a.env),
        *("engine_arg=" + x for x in sorted(a.engine_arg)),
        "routing=" + a.routing,
        "routing_file=" + (os.path.basename(a.routing_file) if a.routing_file else ""),
        "source=" + src["source_sha"] + "+" + src["source_dirty_sha"],
        "layout=" + layout_tag(a),
        f"quant={a.quantization}/{a.kv_cache_dtype}",
        f"weights={a.weights}",
        f"max_running={env['TPU_SCHED_MK_MAX_RUNNING']}",
        f"kv_bytes={a.kv_cache_memory_bytes}",
        f"block={a.block_size}",
        f"compile={a.compile_sizes}",
        f"chunk={a.max_num_batched_tokens}",
        common.SITECUSTOMIZE,
    ])
    for k, v in common.cache_dirs(a.cache_root, key).items():
        env.setdefault(k, v)
    env["P3_STEP_LOG"] = os.path.join(out_dir, "steps.jsonl")
    env["P3_LOG_PREFIX"] = "p3_d_leg"
    env["P3_ROUTING"] = a.routing
    if a.routing_file:
        env["P3_ROUTING_FILE"] = os.path.abspath(a.routing_file)
    env["P3_WEIGHTS"] = a.weights
    env["P3_COMPILE_SIZES"] = a.compile_sizes
    site_dir = common.install_sitecustomize(out_dir)
    inherited = os.environ.get("PYTHONPATH", "")
    if a.distributed_executor_backend == "ray":
        # Ray workers start from the raylet's environment, so the pod puts a copy of
        # SITECUSTOMIZE on PYTHONPATH before `ray start` (P3_RAY_SITE); prepending site_dir as
        # well would load it twice in the engine core and record every step twice.
        ray_site = os.path.join(os.environ.get("P3_RAY_SITE", ""), "sitecustomize.py")
        if not (
            os.path.isfile(ray_site)
            and open(ray_site).read() == common.SITECUSTOMIZE
            and os.path.dirname(ray_site) in inherited.split(os.pathsep)
        ):
            raise SystemExit(
                "--distributed-executor-backend ray needs P3_RAY_SITE=<dir holding "
                "common.SITECUSTOMIZE as sitecustomize.py> on PYTHONPATH before ray start "
                "(EXTRA_ENV)"
            )
        env["PYTHONPATH"] = inherited
        # Ray copies only VLLM_* and the platform's listed variables to its workers.
        env["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(
            sorted(k for k in env if not k.startswith("VLLM_"))
        )
    else:
        env["PYTHONPATH"] = os.pathsep.join(x for x in (site_dir, inherited) if x)
    if a.decode_fill:
        # p3_decode_fill must import in the engine core and every worker.
        env["PYTHONPATH"] = os.pathsep.join(
            x for x in (env["PYTHONPATH"], os.path.dirname(os.path.abspath(__file__))) if x
        )
    env["P3_CACHE_KEY"] = key
    return env, unset


def engine_argv(a: argparse.Namespace, profile_dir: str) -> list[str]:
    compile_sizes = [int(x) for x in a.compile_sizes.split(",") if x]
    load_format = a.load_format or ("dummy" if a.weights == "synthetic" else "auto")
    max_num_seqs = a.max_num_seqs or max(case_reqs(a, c) for c in a.case)
    argv = [
        "--model", a.model_dir,
        "--load-format", load_format,
        "--trust-remote-code",
        "--seed", "42",
        "--max-model-len", str(a.max_model_len),
        "--disable-custom-all-reduce",
        "--gpu-memory-utilization", str(a.gpu_memory_utilization),
        *(["--kv-cache-memory-bytes", str(a.kv_cache_memory_bytes)] if a.kv_cache_memory_bytes > 0 else []),
        "--kv-cache-dtype", a.kv_cache_dtype,
        "--quantization", a.quantization,
        "--language-model-only",
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
        "--max-num-batched-tokens", str(a.max_num_batched_tokens),
        "--max-num-seqs", str(max_num_seqs),
        "--no-disable-hybrid-kv-cache-manager",
        "--attention-backend", "CUSTOM",
        "--mamba-cache-mode", "align",
        "--compilation-config", json.dumps({
            "backend": "vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor",
            "compile_sizes": compile_sizes,
            "inductor_compile_config": {
                "enable_auto_functionalized_v2": False,
                "size_asserts": False,
                "alignment_asserts": False,
                "scalar_asserts": False,
            },
        }),
        "--enable-prefix-caching",
        "--block-size", str(a.block_size),
        "--disable-sliding-window",
        "--tensor-parallel-size", str(a.tensor_parallel_size),
        "--decode-context-parallel-size", str(a.decode_context_parallel_size),
        "--pipeline-parallel-size", str(a.pipeline_parallel_size),
        "--prefill-context-parallel-size", "1",
        "--cp-kv-cache-interleave-size", str(a.cp_kv_cache_interleave_size),
        "--data-parallel-size", str(a.data_parallel_size),
        "--profiler-config.profiler", "torch",
        "--profiler-config.torch_profiler_dir", profile_dir,
        "--profiler-config.ignore_frontend", "true",
    ]  # fmt: skip
    if a.expert_parallel:
        argv.append("--enable-expert-parallel")
    if a.no_async_scheduling or a.pipeline_parallel_size > 1:
        argv.append("--no-async-scheduling")
    else:
        argv.append("--async-scheduling")
    if a.scheduler_cls:
        argv += ["--scheduler-cls", a.scheduler_cls]
    if a.mtp > 0:
        spec = {"method": "mtp", "num_speculative_tokens": a.mtp}
        if a.acceptance == "synthetic":
            spec.update(rejection_sample_method="synthetic",
                        synthetic_acceptance_length=a.synthetic_acceptance)
        argv += ["--speculative-config", json.dumps(spec)]
    if a.distributed_executor_backend:
        argv += ["--distributed-executor-backend", a.distributed_executor_backend]
    if a.decode_fill:
        argv += ["--kv-transfer-config", json.dumps({
            "kv_connector": "DecodeBenchConnector",
            "kv_connector_module_path": "p3_decode_fill",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"fill_mean": 0.015, "fill_std": 0.0},
        })]
    return argv


def _window_stats(steps: list[dict], t0: float, t1: float, reqs: int, k1: int) -> dict:
    """Step records inside [t0, t1]: the steady decode steps are those that
    scheduled exactly reqs requests and at most reqs x (K+1) tokens; the step
    time is the median interval between consecutive ones. Steps at another
    running count, and steps that also carried a prefill chunk (replenishment),
    are counted and left out. Records are grouped by the engine process that
    wrote them (one per data-parallel replica), so an interval never spans two
    engines; reqs is the per-replica count."""
    win = [s for s in steps if t0 <= s["t"] <= t1]
    at_r = [s for s in win if s["reqs"] == reqs and 0 < s["tok"] <= reqs * k1]
    mixed = [s for s in win if s["reqs"] == reqs and s["tok"] > reqs * k1]
    hist: dict[int, int] = {}
    for s in win:
        hist[s["tok"]] = hist.get(s["tok"], 0) + 1
    by_engine: dict = {}
    for s in win:
        by_engine.setdefault(s.get("pid"), []).append(s)
    out = {
        "steps": len(win),
        "engines": len(by_engine),
        "steps_at_r": len(at_r),
        "steps_mixed_prefill": len(mixed),
        "steps_not_at_r": len(win) - len(at_r) - len(mixed),
        "held_running": bool(win) and all(s["reqs"] == reqs and s.get("running_after", reqs) >= reqs
                                           for s in win if s["tok"] > 0),
        "min_running_after": min((s.get("running_after", reqs) for s in win), default=None),
        "step_tokens_hist": dict(sorted(hist.items())),
        "full_tok": reqs * k1,
    }
    def pure(s):
        return s["reqs"] == reqs and 0 < s["tok"] <= reqs * k1

    # Intervals between log-adjacent steady steps of one engine only, so a gap
    # that spans a mixed or off-r step never enters the step time.
    iv: list[float] = []
    per_engine: dict[str, float] = {}
    for pid, recs in by_engine.items():
        e_iv = [(b["t"] - a["t"]) * 1e3 for a, b in zip(recs, recs[1:]) if pure(a) and pure(b)]
        if e_iv:
            per_engine[str(pid)] = round(statistics.median(e_iv), 3)
        iv += e_iv
    if len(by_engine) > 1:
        out["step_ms_median_per_engine"] = per_engine
    if iv:
        q = statistics.quantiles(iv, n=10) if len(iv) >= 2 else [iv[0]] * 9
        med = statistics.median(iv)
        out.update(
            step_ms_median=round(med, 3),
            step_ms_mean=round(statistics.mean(iv), 3),
            step_ms_p10=round(q[0], 3),
            step_ms_p90=round(q[-1], 3),
            step_intervals=len(iv),
            scheduled_tok_per_step=round(statistics.mean(s["tok"] for s in at_r), 2),
        )
        sampled = [s["sampled"] for s in at_r if "sampled" in s]
        if sampled:
            # Tokens emitted per request per step: 1 plus accepted drafts.
            acc = statistics.mean(sampled) / reqs
            out["accepted_len_mean"] = round(acc, 3)
            # Per replica; every replica steps in lockstep, so all replicas
            # emit engines x this.
            out["tokens_per_s"] = round(acc * reqs / (med / 1e3))
            out["tokens_per_s_all_replicas"] = round(out["tokens_per_s"] * max(1, len(by_engine)))
        else:
            out["tokens_per_s_scheduled"] = round(out["scheduled_tok_per_step"] / (med / 1e3))
    return out


def _install_dp_request_routing(dp: int) -> None:
    """Routes each request to the engine named in its id under data parallelism.

    The offline client sends every request to its first engine core, so a
    data-parallel engine would hold every request on one replica while the
    others run dummy batches. Request ids carry `-e<idx>-`; adds and aborts go
    to that engine. Call before the engine is built.
    """
    if dp <= 1:
        return
    import re

    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core_client import SyncMPClient

    pat = re.compile(r"-e(\d+)-")

    def engine_of(self, request_id: str):
        m = pat.search(request_id)
        idx = int(m.group(1)) if m else 0
        return self.core_engines[idx % len(self.core_engines)]

    def send(self, engine, request_type, request) -> None:
        self.ensure_alive()
        msg = (engine, request_type.value, *self.encoder.encode(request))
        self.input_socket.send_multipart(msg, copy=False)

    def add_request(self, request) -> None:
        if self.is_dp:
            self.engines_running = True
        send(self, engine_of(self, request.request_id), EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: list[str]) -> None:
        if not request_ids or self.resources.engine_dead:
            return
        by_engine: dict = {}
        for rid in request_ids:
            by_engine.setdefault(engine_of(self, rid), []).append(rid)
        for engine, ids in by_engine.items():
            send(self, engine, EngineCoreRequestType.ABORT, ids)

    SyncMPClient.add_request = add_request
    SyncMPClient.abort_requests = abort_requests


def _kv_capacity(vllm_config) -> dict:
    """KV pool the engine core sized: num_gpu_blocks (the core reports it back to the
    frontend's cache_config at startup), the block size and the decode context parallel width.
    vLLM's attention block manager takes block_size x dcp tokens per block under DCP
    (single_type_kv_cache_manager), so token_capacity is the attention view of the pool; the
    GatedDeltaNet state blocks of the same pool keep dcp 1."""
    cc, pc = vllm_config.cache_config, vllm_config.parallel_config
    dcp = int(pc.decode_context_parallel_size)
    blocks = cc.num_gpu_blocks
    return {
        "num_gpu_blocks": blocks,
        "block_size": cc.block_size,
        "dcp": dcp,
        "tokens_per_block_per_seq": cc.block_size * dcp,
        "token_capacity": blocks * cc.block_size * dcp if blocks else None,
    }


def run(a: argparse.Namespace, out_dir: str, timeline: common.Timeline) -> dict:
    import numpy as np
    from vllm import SamplingParams
    from vllm.config import FaultToleranceConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.engine.llm_engine import LLMEngine

    profile_dir = os.path.join(out_dir, "xprof")
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    ns = parser.parse_args(engine_argv(a, profile_dir))
    if isinstance(ns.fault_tolerance_config, dict):
        ns.fault_tolerance_config = FaultToleranceConfig()
    engine_args = EngineArgs.from_cli_args(ns)
    fields = {f.name for f in dataclasses.fields(engine_args)}
    for key, value in engine_arg_overrides(a.engine_arg).items():
        if key not in fields:
            raise SystemExit(f"--engine-arg {key}: EngineArgs has no field {key!r}")
        setattr(engine_args, key, value)
        print(f"[p3_d_leg] engine arg {key}={value!r}", flush=True)
    step_log = os.environ["P3_STEP_LOG"]
    open(step_log, "w").close()
    dp = a.data_parallel_size
    _install_dp_request_routing(dp)
    chips = max(1, ranks_per_replica(a) * dp // a.cores_per_chip)
    timeline.mark("engine_init_start")
    # The engine directly, not through `LLM`: `LLM.__init__` refuses
    # data_parallel_size > 1 in one process, while this is the process layout
    # `vllm serve --data-parallel-size` uses (one parent spawns every engine
    # core and the coordinator), which the TPU platform's single-host DP
    # bootstrap is written for. `_install_dp_request_routing` gives every
    # engine its own requests, so no engine idles.
    eng = LLMEngine.from_engine_args(engine_args, usage_context=UsageContext.LLM_CLASS)
    timeline.mark("engine_ready")
    startup_s = timeline.marks["engine_ready"] - timeline.marks["engine_init_start"]
    print(f"[p3_d_leg] engine ready in {startup_s:.0f} s", flush=True)
    kv_cache = _kv_capacity(eng.vllm_config)
    print(f"[p3_d_leg] kv_cache {json.dumps(kv_cache)}", flush=True)
    rng = np.random.default_rng(a.seed)
    k1 = a.mtp + 1
    n_steps = a.warm_steps + (a.prof_steps if a.profile else 0) + a.meas_steps * max(1, a.windows)
    records = None
    if a.prompt_file:
        records = []
        with open(a.prompt_file) as f:
            for line in f:
                if line.strip():
                    records.append(list(json.loads(line)["prefix"]))
    shapes_pool: list[dict] = []
    if a.shapes_from:
        # Enough rows for the initial batch of every case plus replacements.
        n_shapes = 8 * max(case_reqs(a, c) for c in a.case) + 64
        shapes_pool = common.load_shapes(a.shapes_from, n_shapes, a.shapes_seed, a.shapes_select,
                                         a.shapes_subset)
    shape_iter = iter(shapes_pool)
    counters = {"added": 0, "finished": 0, "client_steps": 0, "last_output_t": time.time()}
    live_ids: set[str] = set()
    dumped: dict[str, list[int]] = {}

    def unfinished() -> bool:
        # Under DP the sync client's engines_running flag never clears: the engines
        # report wave_complete to the DP coordinator, not to this client, so
        # eng.has_unfinished_requests() stays True once idle and eng.step() would
        # block in get_output() forever. The output processor's own view is exact.
        if dp > 1:
            return eng.output_processor.has_unfinished_requests()
        return eng.has_unfinished_requests()

    import threading

    def watchdog() -> None:
        while not watchdog_stop.wait(60.0):
            stall = time.time() - counters["last_output_t"]
            if stall >= 120.0:
                print(f"[p3_d_leg] watchdog: no engine output for {stall:.0f} s after "
                      f"{counters['client_steps']} client steps, {len(live_ids)} live requests",
                      flush=True)

    watchdog_stop = threading.Event()
    threading.Thread(target=watchdog, name="p3-watchdog", daemon=True).start()

    def add_one(rid: str) -> int:
        """Adds one request; returns its prompt length."""
        shape = next(shape_iter, None) if shapes_pool else None
        if records:
            prompt = records[counters["added"] % len(records)]
        elif shape is not None:
            prompt = rng.integers(1000, 200000, shape["ctx"]).tolist()
        else:
            prompt = rng.integers(1000, 200000, current_case.ctx).tolist()
        if a.hold_running == "replenish" and shape is not None:
            max_tokens = max(1, min(a.max_model_len - len(prompt) - 16, shape["out"]))
        else:
            # Budget: early-prefilled requests decode while later requests finish
            # chunked prefill, so cover both prefill steps and the timed window.
            prefill_steps = (case_reqs(a, current_case) * len(prompt)) // max(1, a.max_num_batched_tokens)
            max_tokens = min(
                a.max_model_len - len(prompt) - 16,
                (prefill_steps + n_steps + 64) * k1 * 2 + 256,
            )
        sp = SamplingParams(max_tokens=max_tokens, temperature=a.temperature, top_k=a.top_k,
                            top_p=a.top_p, ignore_eos=True, detokenize=False)
        eng.add_request(rid, {"prompt_token_ids": prompt}, sp)
        counters["added"] += 1
        live_ids.add(rid)
        return len(prompt)

    def step_once():
        if not unfinished():
            return []
        outs = eng.step()
        counters["client_steps"] += 1
        counters["last_output_t"] = time.time()
        for o in outs:
            if a.dump_tokens and o.outputs and len(dumped.get(o.request_id, ())) < a.dump_tokens:
                # Outputs are cumulative (the default RequestOutputKind): keep the first N.
                import itertools
                dumped[o.request_id] = list(itertools.islice(o.outputs[0].token_ids, a.dump_tokens))
            if o.finished:
                live_ids.discard(o.request_id)
                if a.hold_running == "replenish":
                    counters["finished"] += 1
                    add_one(f"{o.request_id}-repl{counters['finished']}")
        return outs

    def steps(n: int) -> None:
        # One eng.step() consumes one engine's outputs; dp engines step in
        # lockstep, so n device steps are n x dp client steps.
        for _ in range(n * dp):
            step_once()

    cases = []
    timeline.mark("timed_start")
    for ci, case in enumerate(a.case):
        current_case = case
        reqs = case_reqs(a, case)
        # REQS requests on every data-parallel engine; the id names the engine.
        tag = "-real" if (case.real and a.decode_fill) else ""
        ids = [f"case{ci}{tag}-e{e}-req{i}" for e in range(dp) for i in range(reqs)]
        ctx_lens = [add_one(rid) for rid in ids]
        finished_before = counters["finished"]
        t0 = time.time()
        started: set[str] = set()
        n_prefill_steps = 0
        while len(started) < reqs * dp and unfinished():
            for o in step_once():
                if o.outputs and len(o.outputs[0].token_ids) > 0:
                    started.add(o.request_id.split("-repl")[0])
            n_prefill_steps += 1
        prefill_s = time.time() - t0
        tw0 = time.time()
        steps(a.warm_steps)
        tw1 = time.time()
        prof_window = None
        if a.profile:
            eng.start_profile(profile_prefix=f"case{ci}")
            tp0 = time.time()
            steps(a.prof_steps)
            tp1 = time.time()
            eng.stop_profile()
            prof_window = (tp0, tp1)
        meas = []
        for _w in range(max(1, a.windows)):
            tm0 = time.time()
            steps(a.meas_steps)
            tm1 = time.time()
            meas.append((tm0, tm1))
        replaced = counters["finished"] - finished_before
        # Abort everything still running (including replacements) before the next case.
        eng.abort_request(sorted(live_ids))
        live_ids.clear()
        while unfinished():
            eng.step()
        log = common.step_records(common.read_jsonl(step_log))
        for wi, (tm0, tm1) in enumerate(meas):
            case_out = {
                "ctx": case.ctx if not (records or shapes_pool) else round(statistics.mean(ctx_lens)),
                "ctx_min": min(ctx_lens), "ctx_max": max(ctx_lens),
                "decode_fill": bool(a.decode_fill and not case.real),
                "reqs": reqs,
                "reqs_arg": case.reqs,
                "replicas": dp,
                "reqs_all_replicas": reqs * dp,
                "chips": chips,
                "batch_mode": a.batch_mode,
                "hold_running": a.hold_running,
                "replaced_requests": replaced,
                "k": a.mtp,
                "acceptance": a.acceptance,
                "synthetic_acceptance": a.synthetic_acceptance if a.acceptance == "synthetic" else None,
                "sampling": {"temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p},
                "prefill_s": round(prefill_s, 1),
                "prefill_steps": n_prefill_steps,
                "all_started": len(started) == reqs * dp,
                "warm_window_s": round(tw1 - tw0, 2),
                "warm": _window_stats(log, tw0, tw1, reqs, k1),
                "measured": _window_stats(log, tm0, tm1, reqs, k1),
                "measured_wall_s": round(tm1 - tm0, 3),
                "window": wi,
            }
            if prof_window:
                case_out["profiled"] = _window_stats(log, *prof_window, reqs, k1)
            for key in ("warm", "measured", "profiled"):
                w = case_out.get(key)
                if w and w.get("tokens_per_s_all_replicas"):
                    w["tokens_per_s_per_chip"] = round(w["tokens_per_s_all_replicas"] / chips)
            print(f"[p3_d_leg] case {ci}: {json.dumps(case_out)}", flush=True)
            cases.append(case_out)
    timeline.mark("timed_end")
    watchdog_stop.set()
    if a.dump_tokens:
        with open(os.path.join(out_dir, "tokens.jsonl"), "w") as f:
            for rid in sorted(dumped):
                f.write(json.dumps({"id": rid, "tokens": dumped[rid]}) + "\n")
        print(f"[p3_d_leg] dumped tokens of {len(dumped)} requests", flush=True)
    return {"startup_s": round(startup_s, 1), "kv_cache": kv_cache, "cases": cases}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    a = parse_args(argv)
    timeline = common.Timeline()
    out_dir = a.out_dir or common.run_dir("d_leg", "-".join(x for x in (a.tag, layout_tag(a)) if x),
                                          a.out_root or None)
    os.makedirs(out_dir, exist_ok=True)
    env, unset = build_env(a, out_dir)
    argv_engine = engine_argv(a, os.path.join(out_dir, "xprof"))
    if a.dry_run:
        print(json.dumps({"env": env, "unset": unset, "engine_argv": argv_engine,
                          "out_dir": out_dir}, indent=1))
        return 0
    os.environ.update(env)
    for key in unset:
        os.environ.pop(key, None)
    env_id = common.write_env(out_dir, {"bench": "d_leg", "cache_key": env["P3_CACHE_KEY"]})
    config = {
        "argv": argv,
        "cases": [dataclasses.asdict(c) for c in a.case],
        "layout": {
            "tp": a.tensor_parallel_size, "dcp": a.decode_context_parallel_size,
            "pp": a.pipeline_parallel_size, "dp": a.data_parallel_size,
            "ep": bool(a.expert_parallel), "num_hosts": a.num_hosts,
            "ranks": ranks_per_replica(a) * a.data_parallel_size,
            "megakernel": a.megakernel, "mtp_mk": int(env["VLLM_TPU_QWEN35_MTP_MK"]),
        },
        "k": a.mtp,
        "acceptance": a.acceptance,
        "synthetic_acceptance": a.synthetic_acceptance,
        "sampling": {"temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p},
        "batch_mode": a.batch_mode,
        "routing": a.routing,
        "weights": a.weights,
        "env_overrides": a.env,
        "engine_argv": argv_engine,
        "engine_arg_overrides": a.engine_arg,
        "stack": common.stack_tag(),
        "cache_key": env["P3_CACHE_KEY"],
        "out_dir": out_dir,
    }
    result = {"bench": "d_leg", "environment_id": env_id["id"], "config": config}
    try:
        result.update(run(a, out_dir, timeline))
        # In replenish mode r dips by one between a finish and its replacement
        # inside the same step, so the hold check is the share of steady steps.
        result["ok"] = all(
            c["all_started"] and c["measured"].get("step_intervals", 0) > 0
            and (c["measured"]["held_running"] if a.hold_running == "window"
                 else c["measured"]["steps_at_r"] >= 0.5 * c["measured"]["steps"])
            for c in result["cases"]
        )
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
        result["ok"] = False
        raise
    finally:
        records = common.read_jsonl(os.environ["P3_STEP_LOG"])
        timeline.absorb_phases(records)
        result["windows"] = timeline.windows()
        result["phases"] = common.summarize_phases(records)
        common.write_json(os.path.join(out_dir, "result.json"), result)
        print("P3_D_LEG_RESULT " + json.dumps(result, default=str), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
