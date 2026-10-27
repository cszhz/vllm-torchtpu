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
"""End-to-end model verification (`verify_model`) and `auto_model` entry points."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax

from . import pool, vmem
from . import weights as w_mod
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .geometry import ModelGeometry
from .model_def import ModelDef, get_model
from .profile import Profile, list_profiles
from .runtime.policy import BucketPolicy
from .spec import ModelSpec
from .verify_checks import (
    VerifyCtx,
    check_derived_artifacts,
    check_loader_parity,
    prof_exec,
    run_kernel_vs_ref,
)


@dataclass(frozen=True)
class ToleranceSpec:
    """Numerical tolerance specification for `verify_model`.

    Per-field comparison vs `tests/megakernel/test_model_def.py:979-998` and
    `tests/megakernel/test_m3_cache_eplb.py`:
    - `hidden`: `atol=4e-2, rtol=2e-2, min_cos=0.9999` (`atol=4e-2` tightens
      `test_model_def.py`'s `5e-2`; `rtol=2e-2` is looser than `1e-2` for
      near-zero elements; seed-0 measured `max_abs`: primary hybrid model
      `0.039` at `T=4` / `0.019` at `T=1`, `SynthHybrid` `0.0156`).
    - `kv`: `atol=3.5e-2, rtol=1.5e-2` (looser than `test_model_def.py`'s
      `2e-2/1e-2` at `T=1` because `DirectHBM` at `T=4` has `kc` `max_abs =
      0.03125 = 1` bf16 ULP).
    - `paged_kv`: `atol=2.5e-1, rtol=1.5e-1` (matches `test_m3_cache_eplb.py`
      for FP8 `PallasWriter`, 1 FP8 ULP).
    - `conv`: `atol=1e-2, rtol=1e-2` (matches `test_model_def.py`).
    - `rec`: `atol=5e-2, rtol=2e-2` (matches `test_model_def.py`).
    - `approx_*`: governs `approx_profiles`; `hf_hidden_*`: `atol=4e-2,
      rtol=2e-2, min_cos=0.9999` (measured `max_abs ~0.0156`, `min_cos
      ~0.999998`); `max_ulp_diff` is in relative bf16 steps
      (`ldexp(1.0, floor(log2(max(|got|, |ref|, 1.0))) - 7)`).
    """

    hidden_atol: float = 4e-2
    hidden_rtol: float = 2e-2
    hidden_min_cos: float = 0.9999
    kv_atol: float = 3.5e-2
    kv_rtol: float = 1.5e-2
    paged_kv_atol: float = 2.5e-1
    paged_kv_rtol: float = 1.5e-1
    conv_atol: float = 1e-2
    conv_rtol: float = 1e-2
    rec_atol: float = 5e-2
    rec_rtol: float = 2e-2
    approx_hidden_atol: float = 4e-2
    approx_hidden_rtol: float = 2e-2
    approx_hidden_min_cos: float = 0.9999
    approx_kv_atol: float = 3.5e-2
    approx_kv_rtol: float = 1.5e-2
    approx_paged_kv_atol: float = 2.5e-1
    approx_paged_kv_rtol: float = 1.5e-1
    approx_conv_atol: float = 1e-2
    approx_conv_rtol: float = 1e-2
    approx_rec_atol: float = 5e-2
    approx_rec_rtol: float = 2e-2
    hf_hidden_atol: float = 4e-2
    hf_hidden_rtol: float = 2e-2
    hf_hidden_min_cos: float = 0.9999


DEFAULT_TOLERANCES = ToleranceSpec()


@dataclass(frozen=True)
class CheckResult:
    """Outcome and measured error metrics for a single verification check."""

    name: str
    status: Literal["passed", "failed", "skipped"]
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    max_ulp_diff: int = 0
    min_cos: float = 1.0
    tolerance: Mapping[str, float | int] = dataclasses.field(default_factory=dict)
    reason: str = ""
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class VerificationReport:
    """Structured report returned by `verify_model`."""

    model_name: str
    geometry: ModelGeometry
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(c.status in ("passed", "skipped") for c in self.checks)

    def assert_passed(self, *, allow_skipped: bool = True) -> None:
        ok_st = ("passed", "skipped") if allow_skipped else ("passed",)
        bad = [c for c in self.checks if c.status not in ok_st]
        if bad:
            pts = [f"{c.name}({c.status}: {c.reason!r})" for c in bad]
            raise AssertionError(f"verify_model({self.model_name!r}): {pts}")


@dataclass(frozen=True)
class ResolvedModel:
    """Resolved model definition, geometry, execution spec, plan, and pool backend."""

    model_def: type[ModelDef]
    profile: Profile
    geometry: ModelGeometry
    spec: ModelSpec
    exec_spec: ExecSpec
    plan: Any
    pool_layout: pool.PoolLayout
    pool_backend: pool.PoolBackend


def _safe_run(name: str, fn: Callable[[], CheckResult]) -> CheckResult:
    try:
        return fn()
    except Exception as exc:
        return CheckResult(name, "failed", reason=f"{type(exc).__name__}: {exc}")


def auto_model(
    target: Any,
    *,
    profile: str | Profile | None = None,
    R: int | None = None,
    B: int | None = None,
    T: int = 1,
    ctx: int = 2048,
    moe_tp: bool | None = None,
    e_slots: int | None = None,
) -> ResolvedModel:
    """Resolve `target` into a `ResolvedModel` without invoking `kernel.build()`."""
    try:
        m_def = get_model(target)
    except KeyError as exc:
        raise MegakernelConfigError(str(exc)) from exc
    is_cls = isinstance(target, type) and issubclass(target, ModelDef)
    hf_cfg = None if (isinstance(target, str) or is_cls) else target
    if not isinstance(hf_cfg, ModelGeometry):
        elig = m_def.validate_hf_config(hf_cfg)
        if not elig.eligible:
            raise MegakernelConfigError(f"{elig.reason}: {elig.message}")
    is_str = isinstance(profile, str) or profile is None
    prof = m_def.get_profile(profile) if is_str else profile
    if prof.model and prof.model != m_def.name:
        raise MegakernelConfigError(
            f"Profile {prof.name!r} targets model {prof.model!r}, not {m_def.name!r}"
        )
    raw_tp = prof.supported_tp_sizes or m_def.supported_tp_sizes or ()
    sup_tp = tuple(int(x) for x in raw_tp)
    if R is None and not sup_tp:
        raise MegakernelConfigError(f"Model {m_def.name!r} has no supported_tp_sizes")
    eff_R = int(R) if R is not None else sup_tp[0]
    if R is not None and sup_tp and eff_R not in sup_tp:
        raise MegakernelConfigError(
            f"Model {m_def.name!r} supports TP sizes {sup_tp}, got R={eff_R}"
        )
    eff_B = eff_R * int(T) if B is None else int(B)
    if eff_B < eff_R * int(T):
        raise MegakernelConfigError(
            f"B={eff_B} must be >= R * T ({eff_R * int(T)}) for R={eff_R}, T={T}"
        )
    g = m_def.derive_geometry(hf_cfg, moe_tp=moe_tp, e_slots=e_slots)
    spec = m_def.to_model_spec(g, moe_tp=moe_tp, e_slots=e_slots, R=eff_R)
    ex = prof_exec(prof, g, R=eff_R, B=eff_B, T=T, ctx=ctx, interp=False)
    g.check(ex)
    cp, _ = vmem.build_scratch(spec, ex)
    play, pbe = m_def.pool_layout(g, R=eff_R), m_def.pool_backend(g, R=eff_R)
    return ResolvedModel(m_def, prof, g, spec, ex, cp.plan, play, pbe)


def verify_model(
    target: Any,
    *,
    hf: bool = False,
    oracle: Callable[..., tuple[Any, Any]] | None = None,
    approx_profiles: bool = True,
    periods: int = 1,
    R: int = 2,
    B_per_T: int = 2,
    T_values: Sequence[int] | None = None,
    ctx: int = 256,
    seed: int = 0,
    moe_tp: bool | None = None,
    profile: str | Profile | None = None,
    tolerances: ToleranceSpec = DEFAULT_TOLERANCES,
) -> VerificationReport:
    """Run the end-to-end generic verification suite for a registered `ModelDef`.

    Executes on `ModelDef.shrink(periods=periods, R=R)` across `R` JAX devices:
    `derived_artifacts` (VMEM/SMEM chip fit + pool layout), `loader_parity` (exact
    0-ULP parity across `load_hf`, `MappingWeightSource`, `SyntheticVLLMWeightSource`
    for `moe_tp=True` and `False`), `kernel_vs_reference` (`DirectHBM` and
    `PallasWriter` vs `reference_step` across `T_values`, checking hidden + all
    reference/pool states), `approx_profiles` (`numerics="approx"` profiles), and
    `reference_vs_hf` (when `hf=True` or `oracle is not None`).
    """
    m_name = str(getattr(target, "name", target))
    m_def: Any = None
    g: Any = None
    prof: Any = None
    vc: Any = None
    ex_t1: Any = None
    w_hf_q: Any = None
    w_log: Any = None

    def _run_setup() -> CheckResult:
        nonlocal m_name, m_def, g, prof, vc, ex_t1, w_hf_q, w_log
        m_def = get_model(target)
        m_name = m_def.name
        eff_mtp = True if moe_tp is None else bool(moe_tp)
        g = dataclasses.replace(m_def.shrink(periods=periods, R=R), moe_tp=eff_mtp)
        is_str = isinstance(profile, str) or profile is None
        prof = m_def.get_profile(profile) if is_str else profile
        pol = BucketPolicy.resolve(prof)
        v_b = tuple(sorted({1, *(int(t) for t, _ in pol.iter_buckets())}))
        t_v = (v_b if len(v_b) > 1 else (1, 2)) if T_values is None else tuple(T_values)
        in_by_t: dict[int, tuple[Any, Any, Any, Any]] = {}
        for i, T in enumerate(t_v):
            ex_i = prof_exec(
                prof, g, R=R, B=max(1, B_per_T * int(T)), T=int(T), ctx=ctx, interp=True
            )
            in_by_t[int(T)] = w_mod.random_logical_inputs(
                jax.random.key(seed + 100 + i), g, ex_i, model_def=m_def
            )
        vc = VerifyCtx(m_def, g, R, B_per_T, t_v, ctx, seed, tolerances, in_by_t)
        ex_t1 = prof_exec(prof, g, R=R, B=max(1, B_per_T), T=1, ctx=ctx, interp=True)
        w_hf = w_mod.random_hf_weights(jax.random.key(seed), g, model_def=m_def)
        w_hf_q = w_mod.normalize_hf_for_vllm(m_def, ex_t1, w_hf, g=g)
        w_log = w_mod.load_logical_weights(m_def, ex_t1, w_hf_q, g=g)
        return CheckResult("setup", "passed", details={"T_values": t_v, "R": R})

    setup_res = _safe_run("setup", _run_setup)
    if setup_res.status != "passed":
        return VerificationReport(model_name=m_name, geometry=g, checks=(setup_res,))
    checks = [
        setup_res,
        _safe_run(
            "derived_artifacts", lambda: check_derived_artifacts(vc, prof, CheckResult)
        ),
        _safe_run(
            "loader_parity",
            lambda: check_loader_parity(vc, ex_t1, w_hf_q, w_log, CheckResult),
        ),
        _safe_run(
            "kernel_vs_reference",
            lambda: run_kernel_vs_ref(
                vc, prof, "kernel_vs_reference", CheckResult, include_direct=True
            ),
        ),
    ]
    if approx_profiles:
        ap_all = [m_def.get_profile(nm) for nm in list_profiles()]
        ap_l = [p for p in ap_all if p.model == m_def.name and p.numerics == "approx"]
        if ap_l:

            def _run_approx() -> CheckResult:
                sub = [
                    run_kernel_vs_ref(vc, p, p.name, CheckResult, include_direct=False)
                    for p in ap_l
                ]
                bad = [r for r in sub if r.status != "passed"]
                am = max(r.max_abs_err for r in sub)
                rm = max(r.max_rel_err for r in sub)
                um, cm = max(r.max_ulp_diff for r in sub), min(r.min_cos for r in sub)
                st_s: Literal["passed", "failed"] = "passed" if not bad else "failed"
                rsn = "; ".join(r.reason for r in bad if r.reason)
                tols = {"approx_hidden_atol": tolerances.approx_hidden_atol}
                return CheckResult(
                    "approx_profiles",
                    st_s,
                    am,
                    rm,
                    um,
                    cm,
                    tols,
                    rsn,
                    {r.name: r.details for r in sub},
                )

            checks.append(_safe_run("approx_profiles", _run_approx))
        else:
            checks.append(CheckResult("approx_profiles", "skipped", reason="none"))
    if hf or oracle is not None:
        from .verify_hf import verify_reference_vs_hf

        checks.append(
            _safe_run(
                "reference_vs_hf",
                lambda: verify_reference_vs_hf(
                    m_def,
                    g,
                    ex_t1,
                    w_hf_q,
                    w_log,
                    oracle=oracle,
                    tolerances=tolerances,
                    seed=seed,
                ),
            )
        )
    return VerificationReport(model_name=m_def.name, geometry=g, checks=tuple(checks))


__all__ = [
    "CheckResult",
    "DEFAULT_TOLERANCES",
    "ResolvedModel",
    "ToleranceSpec",
    "VerificationReport",
    "auto_model",
    "verify_model",
]
