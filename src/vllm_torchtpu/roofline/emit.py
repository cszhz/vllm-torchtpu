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
"""Stage-3 server flag emitter (vLLM & SGLang) and CPU-only CDK ceiling generator."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from .buckets import BucketRegimePlan
from .modules import Dtypes
from .registry import KNOBS, default_envs_path, env_keys
from .solver import PoolMeshPlan, ServingPlan


def _pool_cli_args(
    pool: PoolMeshPlan,
    engine: Literal["vllm", "sglang"],
    kv_dtype_str: str,
) -> list[str]:
    if engine == "vllm":
        args = [
            "--tensor-parallel-size",
            str(pool.tp),
            "--data-parallel-size",
            str(pool.dp),
            "--kv-cache-dtype",
            kv_dtype_str,
        ]
        if pool.pp > 1:
            args.extend(["--pipeline-parallel-size", str(pool.pp)])
        if pool.pcp > 1:
            args.extend(["--prefill-context-parallel-size", str(pool.pcp)])
        if pool.kv_split > 1:
            args.extend(["--decode-context-parallel-size", str(pool.kv_split)])
        if pool.ep > 1:
            args.append("--enable-expert-parallel")
        return args

    sgl_kv = "fp8_e4m3" if kv_dtype_str == "fp8" else kv_dtype_str
    dp_attn = pool.dp > 1 and pool.ep > 1
    args = [
        "--tp-size",
        str(pool.tp * pool.dp if dp_attn else pool.tp),
        "--dp-size",
        str(pool.dp),
        "--kv-cache-dtype",
        sgl_kv,
    ]
    if dp_attn:
        args.append("--enable-dp-attention")
    if pool.pp > 1:
        args.extend(["--pp-size", str(pool.pp)])
    if pool.ep > 1:
        args.extend(["--ep-size", str(pool.ep)])
    return args


def emit_engine_args(
    plan: ServingPlan,
    bucket_plan: BucketRegimePlan,
    engine: Literal["vllm", "sglang"] = "vllm",
    *,
    dtypes: Dtypes | None = None,
    target_envs: str | Path | None = None,
) -> dict[str, Any]:
    """Emit engine CLI flags, per-band threshold env vars, and target-env refusals."""
    kv_str = "fp8" if (dtypes is not None and dtypes.kv <= 1.0) else "auto"
    rec = plan.recommended
    if rec is not None:
        cli_args = _pool_cli_args(rec.decode_pool, engine, kv_str)
        prefill_cli_args = (
            _pool_cli_args(rec.prefill_pool, engine, kv_str)
            if rec.prefill_pool is not None
            else []
        )
    else:
        cli_args = (
            ["--tensor-parallel-size", "8"] if engine == "vllm" else ["--tp-size", "8"]
        )
        prefill_cli_args = []

    first_tok = bucket_plan.rows[0].bucket_tokens if bucket_plan.rows else None
    last_tok = bucket_plan.rows[-1].bucket_tokens if bucket_plan.rows else None
    sp_global = first_tok is not None and bucket_plan.sp_min_tokens == first_tok

    deferred: dict[str, Any] = {}
    refused_flags: dict[str, str] = {}

    if bucket_plan.sp_min_tokens is not None and not sp_global:
        deferred["TPU_MOE_SEQUENCE_PARALLEL"] = {
            "scope": "startup",
            "min_tokens": bucket_plan.sp_min_tokens,
            "bands": [asdict(b) for b in bucket_plan.bands if b.residual == "sp"],
        }
    if (
        bucket_plan.megakernel_max_tokens > 0
        and last_tok is not None
        and bucket_plan.megakernel_max_tokens < last_tok
    ):
        deferred["megakernel"] = {
            "scope": "startup",
            "max_tokens": bucket_plan.megakernel_max_tokens,
            "bands": [asdict(b) for b in bucket_plan.bands if b.megakernel],
        }

    if engine == "vllm":
        raw_env: dict[str, str] = {
            "TPU_MOE_SEQUENCE_PARALLEL": "1" if sp_global else "0",
        }
        if bucket_plan.fused_ep_min_tokens is not None:
            edge = str(bucket_plan.fused_ep_min_tokens)
            raw_env["USE_MOE_FUSED_EP_KERNEL"] = "1"
            raw_env["MOE_FUSED_EP_KERNEL_MIN_TOKENS"] = edge
            raw_env["TPU_MOE_FUSED_EP_STEP_MIN_TOKENS"] = edge
        else:
            raw_env["USE_MOE_FUSED_EP_KERNEL"] = "0"

        envs_path = (
            Path(target_envs) if target_envs is not None else default_envs_path()
        )
        allowed = env_keys(envs_path)
        env: dict[str, str] = {}
        for flag, val in raw_env.items():
            if flag in allowed:
                env[flag] = val
            else:
                refused_flags[flag] = f"absent from {envs_path}"
        return {
            "engine": engine,
            "cli_args": cli_args,
            "prefill_cli_args": prefill_cli_args,
            "env": env,
            "refused_flags": refused_flags,
            "deferred": deferred,
            "bucket_bands": [asdict(b) for b in bucket_plan.bands],
            "serving_plan": asdict(plan),
        }

    env_sgl: dict[str, str] = {}
    sglang_status: dict[str, str] = {}
    for knob in KNOBS:
        if knob.sglang is not None:
            sglang_status[knob.key] = knob.sglang
            if (
                knob.key == "USE_MOE_FUSED_EP_KERNEL"
                and bucket_plan.fused_ep_min_tokens is not None
            ):
                env_sgl[knob.sglang] = "1"
        else:
            sglang_status[knob.key] = "no SGLang equivalent"

    return {
        "engine": engine,
        "cli_args": cli_args,
        "prefill_cli_args": prefill_cli_args,
        "env": env_sgl,
        "refused_flags": refused_flags,
        "deferred": deferred,
        "sglang": sglang_status,
        "bucket_bands": [asdict(b) for b in bucket_plan.bands],
        "serving_plan": asdict(plan),
    }


def emit_cdk_ceiling_rows(plan: ServingPlan) -> list[dict[str, Any]]:
    """Emit CPU-only CDK ceiling rows leading with p90 interactivity and tput."""
    rows: list[dict[str, Any]] = []
    for pt in plan.frontier:
        rows.append(
            {
                "p90_interactivity_tok_s": pt.p90_interactivity_tok_s,
                "token_throughput_per_chip": pt.token_throughput_per_chip,
                "output_throughput_per_chip": pt.output_throughput_per_chip,
                "p90_ttft_ms": pt.p90_ttft_ms,
                "p90_tpot_ms": pt.p90_tpot_ms,
                "concurrency": pt.concurrency,
                "total_chips": pt.total_chips,
                "mode": pt.mode,
                "binding_unit": pt.binding,
            }
        )
    return rows
