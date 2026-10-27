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
"""Declarative registry of engine threshold and route knobs for roofline scoring."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .geometry import ModelGeometry
from .modules import Dtypes, Parallel, StepShape

KnobScope = Literal["startup", "per-bucket", "per-step"]

THRESHOLD_KEY_RE = (
    r"(_(MIN|MAX)_(TOKENS|ROWS|BYTES)$|_THRESHOLD$|_SMALL_BUCKET_|_TOKEN_BUCKET_)"
)

PENDING_ENV_KEYS: dict[str, str] = {
    "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS": (
        "per-step MoE path knob, lands with the fused-EP fallback PR"
    ),
}


@dataclass(frozen=True)
class Knob:
    """Engine runtime or compile-bucket knob and its roofline cost hook."""

    key: str
    scope: KnobScope
    mechanism: str
    supports: Callable[..., bool]
    cost: Callable[..., StepShape | Parallel] | None
    sglang: str | None = None
    env_var: bool = True


def default_envs_path() -> Path:
    """Return the in-tree ``vllm_torchtpu/envs.py`` path."""
    return Path(__file__).resolve().parents[1] / "envs.py"


def env_keys(path: str | Path | None = None) -> set[str]:
    """Parse ``envs.py`` with ``ast`` so roofline imports neither JAX nor torch."""
    target = Path(path) if path is not None else default_envs_path()
    tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    found: set[str] = set()
    for node in tree.body:
        val_node: ast.AST | None = None
        if type(node) is ast.Assign:
            for tgt in node.targets:
                if type(tgt) is ast.Name and tgt.id == "environment_variables":
                    val_node = node.value
        elif (
            type(node) is ast.AnnAssign
            and type(node.target) is ast.Name
            and node.target.id == "environment_variables"
        ):
            val_node = node.value
        if type(val_node) is ast.Dict:
            for k in val_node.keys:
                if type(k) is ast.Constant and type(k.value) is str:
                    found.add(k.value)
    if found:
        return found
    # Support minimal unit-test fixture files with top-level assignments.
    for node in tree.body:
        if type(node) is ast.Assign:
            for tgt in node.targets:
                if type(tgt) is ast.Name and tgt.id.isupper():
                    found.add(tgt.id)
        elif (
            type(node) is ast.AnnAssign
            and type(node.target) is ast.Name
            and node.target.id.isupper()
        ):
            found.add(node.target.id)
    return found


def threshold_env_keys(path: str | Path | None = None) -> set[str]:
    """Return keys in ``envs.py`` matching ``THRESHOLD_KEY_RE``."""
    pat = re.compile(THRESHOLD_KEY_RE)
    return {k for k in env_keys(path) if pat.search(k)}


def _supports_threshold(
    geom: ModelGeometry,
    parallel: Parallel,
    bucket_tokens: int,
    dtypes: Dtypes | None = None,
) -> bool:
    return (
        geom.num_layers > 0
        and parallel.tp >= 1
        and bucket_tokens >= 1
        and (dtypes is None or dtypes.w_dense > 0.0)
    )


def _supports_sp(
    geom: ModelGeometry,
    parallel: Parallel,
    bucket_tokens: int,
    dtypes: Dtypes | None = None,
) -> bool:
    return (
        bucket_tokens >= 1
        and parallel.tp > 1
        and parallel.dp <= 1
        and parallel.pp <= 1
        and (geom.num_experts == 0 or parallel.moe == "ep")
        and (dtypes is None or dtypes.w_dense > 0.0)
    )


def _supports_fused_ep_load(
    geom: ModelGeometry,
    parallel: Parallel,
    bucket_tokens: int,
    dtypes: Dtypes | None = None,
) -> bool:
    ep_ways = parallel.ep if parallel.ep > 1 else parallel.tp
    return (
        bucket_tokens >= 1
        and geom.num_experts > 0
        and parallel.moe == "ep"
        and ep_ways > 1
        and (dtypes is None or dtypes.w_expert > 0.0)
    )


def _supports_fused_ep_step(
    geom: ModelGeometry,
    parallel: Parallel,
    bucket_tokens: int,
    dtypes: Dtypes | None = None,
    *,
    sorted_gather: bool = False,
    has_hash_ids: bool = False,
) -> bool:
    ep_ways = parallel.ep if parallel.ep > 1 else parallel.tp
    w_expert = 1.0 if dtypes is None else dtypes.w_expert
    return (
        bucket_tokens >= 1
        and geom.num_experts > 0
        and parallel.moe == "ep"
        and ep_ways > 1
        and w_expert == 1.0
        and not sorted_gather
        and not has_hash_ids
    )


def _supports_megakernel(
    geom: ModelGeometry,
    parallel: Parallel,
    bucket_tokens: int,
    dtypes: Dtypes | None = None,
) -> bool:
    ep_ways = parallel.ep if parallel.ep > 1 else parallel.tp
    return (
        bucket_tokens >= 1
        and (dtypes is None or dtypes.w_dense > 0.0)
        and not (
            geom.num_experts > 0 and parallel.moe == "ep" and ep_ways > parallel.tp
        )
    )


def _apply_sp(parallel: Parallel) -> Parallel:
    return replace(parallel, residual="sp")


def _apply_fused_ep(parallel: Parallel) -> Parallel:
    return replace(parallel, moe_dispatch="a2a_push")


KNOBS: tuple[Knob, ...] = (
    # 13 threshold knobs from envs.py matched by THRESHOLD_KEY_RE.
    Knob(
        key="TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS",
        scope="per-bucket",
        mechanism="measure-only",
        supports=_supports_threshold,
        cost=None,
    ),
    Knob(
        key="TPU_FP8_LINEAR_W8A16_MAX_ROWS",
        scope="per-bucket",
        mechanism="measure-only",
        supports=_supports_threshold,
        cost=None,
    ),
    Knob(
        key="ONEHOT_MOE_PERMUTE_THRESHOLD",
        scope="per-bucket",
        mechanism="measure-only",
        supports=_supports_threshold,
        cost=None,
    ),
    Knob(
        key="TPU_MOE_DENSE_EXPERT_THRESHOLD",
        scope="startup",
        mechanism="measure-only",
        supports=_supports_threshold,
        cost=None,
    ),
    Knob(
        key="FUSED_MOE_DECODE_KERNEL_MAX_TOKENS",
        scope="per-bucket",
        mechanism="measure-only",
        supports=_supports_threshold,
        cost=None,
    ),
    Knob(
        key="MOE_FUSED_EP_KERNEL_MIN_TOKENS",
        scope="startup",
        mechanism="moe_dispatch='a2a_push' (load-time fused EP arming threshold)",
        supports=_supports_fused_ep_load,
        cost=_apply_fused_ep,
    ),
    Knob(
        key="TPU_TOKEN_BUCKET_EXTRA",
        scope="startup",
        mechanism="measure-only",
        supports=_supports_threshold,
        cost=None,
    ),
    Knob(
        key="TPU_TOKEN_BUCKET_LINEAR",
        scope="startup",
        mechanism="measure-only",
        supports=_supports_threshold,
        cost=None,
    ),
    Knob(
        key="TPU_DCP_INDEXER_A2A_MIN_TOKENS",
        scope="per-bucket",
        mechanism="measure-only",
        supports=_supports_threshold,
        cost=None,
    ),
    Knob(
        key="TPU_STREAMIDX_SMALL_BUCKET_BQ",
        scope="per-bucket",
        mechanism="measure-only",
        supports=_supports_threshold,
        cost=None,
    ),
    Knob(
        key="TPU_STREAMIDX_SMALL_BUCKET_BKVP",
        scope="per-bucket",
        mechanism="measure-only",
        supports=_supports_threshold,
        cost=None,
    ),
    Knob(
        key="TPU_STREAMIDX_SMALL_BUCKET_TOKENS",
        scope="per-bucket",
        mechanism="measure-only",
        supports=_supports_threshold,
        cost=None,
    ),
    Knob(
        key="SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES",
        scope="startup",
        mechanism="measure-only",
        supports=_supports_threshold,
        cost=None,
    ),
    # Route entries.
    Knob(
        key="USE_MOE_FUSED_EP_KERNEL",
        scope="startup",
        mechanism="moe_dispatch='a2a_push' (enable fused EP kernel)",
        supports=_supports_fused_ep_load,
        cost=_apply_fused_ep,
    ),
    Knob(
        key="TPU_MOE_FUSED_EP_STEP_MIN_TOKENS",
        scope="per-step",
        mechanism="moe_dispatch='a2a_push' at or above step threshold; gmm below",
        supports=_supports_fused_ep_step,
        cost=_apply_fused_ep,
    ),
    Knob(
        key="TPU_MOE_SEQUENCE_PARALLEL",
        scope="startup",
        mechanism="residual='sp' (sequence-parallel residual stream)",
        supports=_supports_sp,
        cost=_apply_sp,
    ),
    Knob(
        key="megakernel",
        scope="startup",
        mechanism="measure-only",
        supports=_supports_megakernel,
        cost=None,
        env_var=False,
    ),
)
