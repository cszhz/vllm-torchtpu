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
"""Small-batch decode host-overhead trims (IX track 3). Every switch is
default-off and read from ``os.environ`` (not ``envs.py``, so the
compile-cache key is unchanged). None changes what the device computes.
``VLLM_TPU_T3_HOST=1`` turns every trim on; an explicitly set per-trim
variable (``=0`` or ``=1``) wins over it.

* ``VLLM_TPU_T3_AOT_SKIP_GUARDS=1``: once a vLLM ``@support_torch_compile``
  module (here the MTP draft step-0 and fused-loop programs) has run its
  freshly AOT-compiled function, skip that function's per-call Dynamo guard
  evaluation, exactly as vLLM already does for an AOT function it *loads*
  from cache when ``dynamic_shapes_config.evaluate_guards`` is False
  (``vllm/compilation/decorators.py:_try_load_aot_compiled_fn``). The
  compiled callable that runs is the same object either way.
* ``VLLM_TPU_T3_GUARD_MEMO``: see ``vllm_torchtpu._qx_fastcall``.
"""

from __future__ import annotations

import os
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _flag(name: str) -> bool:
    """``name`` if set, else the master switch ``VLLM_TPU_T3_HOST``."""
    raw = os.environ.get(name, os.environ.get("VLLM_TPU_T3_HOST", "0"))
    return raw.strip().lower() in ("1", "true", "on", "yes")


AOT_SKIP_GUARDS = _flag("VLLM_TPU_T3_AOT_SKIP_GUARDS")

# VLLM_TPU_T3_CONN_STATS_EVERY=N: poll the KV connector's stats every N decode
# steps instead of every step (default 1 = every step; 16 under
# VLLM_TPU_T3_HOST=1 unless set explicitly).
CONN_STATS_EVERY = int(
    os.environ.get(
        "VLLM_TPU_T3_CONN_STATS_EVERY",
        "16" if _flag("VLLM_TPU_T3_CONN_STATS") else "1",
    )
    or 1
)

# VLLM_TPU_T3_AB_PERIOD_MS=N (default 0 = off): in-process A/B. The trims that
# consult ab_on() are active only in even wall-clock windows of N ms and off
# in odd ones; every process on a host sees the same phase. Timers record
# each key twice, under ".on" / ".off", so one run compares both arms under
# identical load and content.
AB_PERIOD_MS = float(os.environ.get("VLLM_TPU_T3_AB_PERIOD_MS", "0") or 0)


def ab_on() -> bool:
    """True when the A/B toggle is off (trims follow their flags) or in an
    even window."""
    if AB_PERIOD_MS <= 0:
        return True
    import time

    return int(time.time() * 1000.0 / AB_PERIOD_MS) % 2 == 0

_done: set[int] = set()
_ab_modules: dict[int, Any] = {}


def ab_toggle_aot_guards(module: Any) -> None:
    """A/B mode only: set the module's guard check for the current window."""
    fn = _ab_modules.get(id(module))
    if fn is not None:
        fn._guard_check_enabled = not ab_on()


def _evaluate_guards(module: Any) -> bool:
    """vLLM's evaluate_guards for this module (True when unknown: keep guards)."""
    cfg = getattr(module, "compilation_config", None)
    if cfg is None:
        vcfg = getattr(module, "vllm_config", None)
        cfg = getattr(vcfg, "compilation_config", None)
    ds = getattr(cfg, "dynamic_shapes_config", None)
    if ds is None:
        return True
    return bool(getattr(ds, "evaluate_guards", True))


def maybe_skip_aot_guards(module: Any) -> bool:
    """After a successful call of ``module``: disable its AOT function's guard
    check when vLLM would have done so on a cache load. Returns True when it
    was disabled by this call. Cheap after the first call per module."""
    if not AOT_SKIP_GUARDS:
        return False
    key = id(module)
    if key in _done:
        return False
    fn = getattr(module, "aot_compiled_fn", None)
    if fn is None:
        return False
    _done.add(key)
    if not getattr(fn, "_guard_check_enabled", False):
        return False
    if _evaluate_guards(module):
        logger.info(
            "VLLM_TPU_T3_AOT_SKIP_GUARDS: %s keeps guards (evaluate_guards=True)",
            type(module).__name__,
        )
        return False
    disable = getattr(fn, "disable_guard_check", None)
    if disable is None:
        return False
    if AB_PERIOD_MS > 0:
        # A/B mode: re-decide on every call (see ab_toggle_aot_guards).
        _ab_modules[key] = fn
        fn._guard_check_enabled = not ab_on()
        return True
    disable()
    logger.info(
        "VLLM_TPU_T3_AOT_SKIP_GUARDS: per-call guard check off for %s "
        "(as vLLM's AOT cache-load path does with evaluate_guards=False)",
        type(module).__name__,
    )
    return True


def _reset_for_tests() -> None:
    _done.clear()
    _ab_modules.clear()


# ---------------------------------------------------------------------------
# VLLM_TPU_T3_TIMERS=1 (default off): light host timers for serving.
# Wraps a few methods with perf_counter pairs, aggregates in memory and logs
# one summary line per key every VLLM_TPU_T3_TIMERS_PERIOD_S (default 30 s)
# per process. No per-step logging. Measurement only.
# ---------------------------------------------------------------------------

import functools as _functools
import time as _time

TIMERS = os.environ.get("VLLM_TPU_T3_TIMERS", "0").strip() == "1"
_PERIOD_S = float(os.environ.get("VLLM_TPU_T3_TIMERS_PERIOD_S", "30"))
_MAX_SAMPLES = 20000


class _Agg:
    __slots__ = ("vals",)

    def __init__(self) -> None:
        self.vals: list[float] = []

    def add(self, ms: float) -> None:
        if len(self.vals) < _MAX_SAMPLES:
            self.vals.append(ms)


_aggs: dict[str, _Agg] = {}
_last_flush = [_time.monotonic()]
_tag = [""]


def _q(v: list[float], p: float) -> float:
    return v[min(len(v) - 1, int(p * len(v)))]


def flush(force: bool = False) -> None:
    now = _time.monotonic()
    if not force and now - _last_flush[0] < _PERIOD_S:
        return
    window = now - _last_flush[0]
    _last_flush[0] = now
    items = [(k, sorted(a.vals)) for k, a in _aggs.items() if a.vals]
    for a in _aggs.values():
        a.vals = []
    for k, v in sorted(items):
        logger.info(
            "[t3timers]%s %s n=%d window_s=%.1f mean=%.4f p10=%.4f p50=%.4f "
            "p90=%.4f p99=%.4f ms",
            _tag[0], k, len(v), window, sum(v) / len(v), _q(v, 0.1), _q(v, 0.5),
            _q(v, 0.9), _q(v, 0.99),
        )


def record(key: str, ms: float) -> None:
    if AB_PERIOD_MS > 0:
        key = key + (".on" if ab_on() else ".off")
    a = _aggs.get(key)
    if a is None:
        a = _aggs[key] = _Agg()
    a.add(ms)


def _wrap_method(owner: Any, name: str, key: str, after: Any = None) -> bool:
    fn = getattr(owner, name, None)
    if fn is None or getattr(fn, "_t3_timed", False):
        return False
    pc = _time.perf_counter

    @_functools.wraps(fn)
    def timed(*a: Any, **k: Any) -> Any:
        t0 = pc()
        try:
            return fn(*a, **k)
        finally:
            t1 = pc()
            record(key, (t1 - t0) * 1e3)
            if after is not None:
                after(t0, t1)

    timed._t3_timed = True  # type: ignore[attr-defined]
    setattr(owner, name, timed)
    return True


def install_runner_timers(runner_cls: Any) -> None:
    """Worker-side timers on the TPU model runner (and, when importable, the
    drafter and KV connector worker)."""
    if not TIMERS:
        return
    _tag[0] = f"[worker pid={os.getpid()}]"
    state = {"exec_t0": None, "samp_t1": None, "fwd_t0": None, "exec_entry": None}

    def after_exec(t0: float, t1: float) -> None:
        if state["samp_t1"] is not None:
            record("w.gap_sample_end_to_exec_start", (t0 - state["samp_t1"]) * 1e3)
        if state["exec_t0"] is not None:
            record("w.exec_start_period", (t0 - state["exec_t0"]) * 1e3)
        state["exec_t0"] = t0

    def after_sample(t0: float, t1: float) -> None:
        state["samp_t1"] = t1
        flush()

    def after_fwd(t0: float, t1: float) -> None:
        if state["exec_t0"] is not None:
            record("w.exec_start_to_forward_start", (t0 - state["exec_t0"]) * 1e3)
        if state["fwd_t0"] is not None:
            record("w.forward_start_period", (t0 - state["fwd_t0"]) * 1e3)
        state["fwd_t0"] = t0

    _wrap_method(runner_cls, "execute_model", "w.execute_model", after_exec)
    _wrap_method(runner_cls, "sample_tokens", "w.sample_tokens", after_sample)
    _wrap_method(runner_cls, "forward_model", "w.forward_model", after_fwd)
    for name in (
        "_prepare_inputs",
        "_update_states",
        "get_finished_kv_transfers",
        "maybe_setup_kv_connector",
        "_sample_spec_verify_chunk",
        "_modify_prev_results",
        "_assemble_async_spec_substitution",
        "_record_input_staging_fence",
        "_wait_input_staging_fence",
    ):
        _wrap_method(runner_cls, name, f"w.{name}")
    try:
        from vllm_torchtpu.runner.speculative_decoding_manager import (
            SpeculativeDecodingManager,
        )

        _wrap_method(
            SpeculativeDecodingManager, "propose_draft_token_ids", "w.propose"
        )
    except Exception:  # noqa: BLE001
        pass
    logger.info("VLLM_TPU_T3_TIMERS: worker timers installed (period %.0f s)", _PERIOD_S)


def install_engine_core_timers() -> None:
    """Engine-core timers: the batch-queue step, its schedule / update phases,
    executor submits, and the idle gap between steps."""
    if not TIMERS:
        return
    try:
        from vllm.v1.engine import core as _core
    except Exception:  # noqa: BLE001
        return
    ec = getattr(_core, "EngineCore", None)
    fn = getattr(ec, "step_with_batch_queue", None)
    if fn is None or getattr(fn, "_t3_timed", False):
        return
    pc = _time.perf_counter
    st = {"t1": None, "inst": set()}

    def _instrument(self: Any) -> None:
        if id(self) in st["inst"]:
            return
        st["inst"].add(id(self))
        _tag[0] = f"[engine_core pid={os.getpid()}]"
        sch = getattr(self, "scheduler", None)
        if sch is not None:
            _wrap_method(sch, "schedule", "ec.schedule")
            _wrap_method(sch, "update_from_output", "ec.update_from_output")
        ex = getattr(self, "model_executor", None)
        if ex is not None:
            _wrap_method(ex, "execute_model", "ec.submit_execute_model")
            _wrap_method(ex, "sample_tokens", "ec.submit_sample_tokens")
        logger.info("VLLM_TPU_T3_TIMERS: engine-core timers installed (period %.0f s)", _PERIOD_S)

    @_functools.wraps(fn)
    def step_with_batch_queue(self: Any, *a: Any, **k: Any) -> Any:
        _instrument(self)
        t0 = pc()
        if st["t1"] is not None:
            record("ec.gap_between_steps", (t0 - st["t1"]) * 1e3)
        try:
            out = fn(self, *a, **k)
        finally:
            t1 = pc()
            st["t1"] = t1
            record("ec.step_with_batch_queue", (t1 - t0) * 1e3)
        try:
            if out is not None and out[0] is not None:
                record("ec.step_with_output", (t1 - t0) * 1e3)
        except Exception:  # noqa: BLE001
            pass
        flush()
        return out

    step_with_batch_queue._t3_timed = True  # type: ignore[attr-defined]
    ec.step_with_batch_queue = step_with_batch_queue
