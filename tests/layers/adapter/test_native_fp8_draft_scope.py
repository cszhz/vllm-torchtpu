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
"""Native FP8 attention (TPU_RPA_NATIVE_FP8 / TPU_RPA_NATIVE_FP8_PREFILL) is target-only.

Under either variable, a target attention layer's batched RPA call leaves
`native_fp8` / `native_fp8_prefill` unset (the wrapper reads the variables:
the FP8 kernel path), while a speculative draft's layer passes "" for both
(the non-FP8 path the draft ships with). With both variables unset nothing
changes: the draft binds nothing new and shares the target's op as before.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

import vllm_torchtpu.layers.adapter.attention as attention
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as rpa_wrapper
from vllm_torchtpu.layers.adapter import cp_attention, native_fp8_scope
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackendImpl,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

FP8_VARS = ("TPU_RPA_NATIVE_FP8", "TPU_RPA_NATIVE_FP8_PREFILL")


@pytest.fixture
def pvs(monkeypatch):
    for v in FP8_VARS:
        monkeypatch.setenv(v, "pvs")


@pytest.fixture
def fp8_unset(monkeypatch):
    for v in FP8_VARS:
        monkeypatch.delenv(v, raising=False)


@pytest.fixture
def cpu_vllm_config_context():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _batched_impl():
    return PallasBatchedRPAAttentionBackendImpl(
        num_heads=2,
        head_size=128,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bfloat16",
    )


def _mesh():
    return SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1})


def _vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
    )


class _FakeOp:
    def register_fake(self, _fake_impl):
        pass


@pytest.fixture
def registered(monkeypatch):
    """Fresh kernel registry; records every (op_name, wrapped_fn) registered."""
    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})
    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(lambda _mesh, _pcp, _m=object(): (_m, _m, None)),
    )
    out = []

    def fake_jax_op(name, wrapped_fn, **_kwargs):
        out.append((name, wrapped_fn))
        return _FakeOp()

    monkeypatch.setattr(attention.pallas, "jax_op", fake_jax_op)
    return out


def _rpa_call_kwargs(monkeypatch, entry_fn, n_args):
    """Run a kernel entry with `_pallas_rpa_kernel_impl` stubbed; return the
    keyword arguments its `rpa_func` would pass to the batched RPA wrapper."""
    seen = {}

    def fake_impl(*_args, rpa_func, **_kwargs):
        seen["rpa_func"] = rpa_func
        return None, None

    monkeypatch.setattr(attention, "_pallas_rpa_kernel_impl", fake_impl)
    query = jnp.zeros((4, 2, 128))
    args = [None] * n_args
    args[1] = query
    entry_fn(*args)
    fn = seen["rpa_func"]
    if fn is attention.ragged_paged_attention_batched:
        return {}
    assert fn.func is attention.ragged_paged_attention_batched
    return dict(fn.keywords)


def _resolved_serving_config(monkeypatch, **kwargs):
    """The (native_fp8, native_fp8_prefill) the batched RPA wrapper resolves
    into its ServingConfigs for one call."""

    class _Stop(Exception):
        pass

    seen = {}

    def fake_serving_configs(**kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(rpa_wrapper.configs, "ServingConfigs", fake_serving_configs)
    q = jnp.zeros((8, 2, 128), jnp.bfloat16)
    kv = jnp.zeros((4, 16, 2, 2, 128), jnp.bfloat16)
    with pytest.raises(_Stop):
        rpa_wrapper.ragged_paged_attention(
            q,
            jnp.zeros((8, 1, 128), jnp.bfloat16),
            jnp.zeros((8, 1, 128), jnp.bfloat16),
            kv,
            jnp.zeros((2,), jnp.int32),
            jnp.zeros((8,), jnp.int32),
            jnp.zeros((3,), jnp.int32),
            jnp.zeros((3,), jnp.int32),
            vmem_limit_bytes=1 << 26,
            **kwargs,
        )
    return seen["native_fp8"], seen["native_fp8_prefill"]


# --- the wrapper: what each call resolves to --------------------------------


def test_wrapper_reads_the_variables_when_unset_and_honours_explicit_off(
    monkeypatch, pvs
):
    # Target layers pass nothing: the FP8 kernel path.
    assert _resolved_serving_config(monkeypatch) == ("pvs", "pvs")
    # Draft layers pass OFF_KWARGS: the non-FP8 path, whatever the env says.
    assert _resolved_serving_config(monkeypatch, **native_fp8_scope.OFF_KWARGS) == (
        "",
        "",
    )


# --- the kernel entries ------------------------------------------------------


@pytest.mark.parametrize(
    "entry,n_args",
    [
        (attention._pallas_rpa_kernel_batched, 12),
        (attention._pallas_rpa_kernel_batched_scheduled, 37),
    ],
)
def test_entries_pin_fp8_off_only_for_a_draft(monkeypatch, pvs, entry, n_args):
    import functools

    common = dict(
        mesh=None,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=attention.batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
    )
    target = _rpa_call_kwargs(
        monkeypatch, functools.partial(entry, **common), n_args
    )
    assert "native_fp8" not in target and "native_fp8_prefill" not in target
    draft = _rpa_call_kwargs(
        monkeypatch, functools.partial(entry, native_fp8_off=True, **common), n_args
    )
    assert draft["native_fp8"] == "" and draft["native_fp8_prefill"] == ""


# --- the kernel builders -----------------------------------------------------


def test_builder_gives_a_draft_its_own_non_fp8_op_under_pvs(
    monkeypatch, pvs, registered, cpu_vllm_config_context
):
    target, draft = _batched_impl(), _batched_impl()
    draft.native_fp8_draft = True
    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        t_op = target._build_rpa_kernel(None, None, None)
        d_op = draft._build_rpa_kernel(None, None, None)
        t_sched = target._build_rpa_scheduled_kernel(None, None, None)
        d_sched = draft._build_rpa_scheduled_kernel(None, None, None)
    assert t_op is not d_op and t_sched is not d_sched
    rpa = [(n, fn) for n, fn in registered if n.startswith("pallas::rpa_kernel")]
    assert len(rpa) == 4, [n for n, _ in rpa]
    (_, t_fn), (_, d_fn), (_, ts_fn), (_, ds_fn) = rpa
    assert "native_fp8_off" not in t_fn.keywords
    assert "native_fp8_off" not in ts_fn.keywords
    assert d_fn.keywords["native_fp8_off"] is True
    assert ds_fn.keywords["native_fp8_off"] is True
    # End to end through the bound entries: the target leaves native_fp8 to
    # the env (pvs), the draft pins it off.
    # The builder binds q/k/v scales as keywords: 9 positional arguments.
    t_kw = _rpa_call_kwargs(monkeypatch, t_fn, 9)
    d_kw = _rpa_call_kwargs(monkeypatch, d_fn, 9)
    assert "native_fp8" not in t_kw
    assert d_kw["native_fp8"] == "" and d_kw["native_fp8_prefill"] == ""


def test_builder_unchanged_with_the_variables_unset(
    monkeypatch, fp8_unset, registered, cpu_vllm_config_context
):
    target, draft = _batched_impl(), _batched_impl()
    draft.native_fp8_draft = True
    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        t_op = target._build_rpa_kernel(None, None, None)
        d_op = draft._build_rpa_kernel(None, None, None)
    # As before the scoping: one shared op, nothing new bound.
    assert t_op is d_op
    rpa = [(n, fn) for n, fn in registered if n.startswith("pallas::rpa_kernel")]
    assert len(rpa) == 1
    assert "native_fp8_off" not in rpa[0][1].keywords


def test_dcp_kernels_pin_fp8_off_for_a_draft(monkeypatch, pvs):
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", lambda name, fn, **k: fn)
    kw = dict(
        sliding_window=None,
        sm_scale=1.0,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=4,
        cp_rank=0,
        decode_query_size=4,
    )
    t_cache, t_new = cp_attention.build_dcp_kernels(**kw)
    d_cache, d_new = cp_attention.build_dcp_kernels(native_fp8_off=True, **kw)
    assert t_cache is not d_cache and t_new is not d_new
    assert d_cache.keywords["native_fp8_off"] is True
    assert d_new.keywords["native_fp8_off"] is True
    assert t_cache.keywords["native_fp8_off"] is False

    calls = []

    def fake_rpa(**k):
        calls.append(k)
        return None, None, None

    monkeypatch.setattr(cp_attention._batched_rpa_wrapper, "ragged_paged_attention", fake_rpa)
    args = [jnp.zeros((4, 2, 128))] * 11
    t_new(*args)
    d_new(*args)
    assert "native_fp8" not in calls[0]
    assert calls[1]["native_fp8"] == "" and calls[1]["native_fp8_prefill"] == ""


# --- the runner marks the draft before building its kernels ------------------


def _runner(layers, draft_names):
    runner = MagicMock(spec=TPUModelRunner)
    runner._attention_kernels_initialized = False
    runner.model = None
    runner.mesh = MagicMock()
    runner.vllm_config = MagicMock()
    runner.parallel_config = SimpleNamespace(tensor_parallel_size=2)
    runner.reorder_batch_threshold = 1
    runner.drafter = SimpleNamespace(_draft_attn_layer_names=set(draft_names))
    runner.speculative_config = SimpleNamespace(draft_tensor_parallel_size=2)
    runner._initialize_attention_kernels = (
        TPUModelRunner._initialize_attention_kernels.__get__(runner)
    )
    return runner


def test_runner_marks_draft_layers_before_initialize_kernel(pvs):
    seen = {}
    layers = {}
    for name in ("model.layers.3.self_attn.attn", "mtp.layers.0.self_attn.attn"):
        impl = MagicMock(spec=PallasAttentionBackendImpl)
        impl.decode_query_size = 4
        impl.native_fp8_draft = False

        def record(_layer, impl=impl, name=name):
            seen[name] = impl.native_fp8_draft

        impl.initialize_kernel.side_effect = record
        layers[name] = SimpleNamespace(impl=impl)
    runner = _runner(layers, ["mtp.layers.0.self_attn.attn"])
    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config",
            return_value=layers,
        ),
        patch("vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context"),
        patch("vllm_torchtpu.runner.tpu_runner.logger") as log,
    ):
        runner._initialize_attention_kernels()
    assert seen == {
        "model.layers.3.self_attn.attn": False,
        "mtp.layers.0.self_attn.attn": True,
    }
    lines = [c.args[0] for c in log.info.call_args_list if c.args]
    assert (
        "TPU_RPA_NATIVE_FP8: target=pvs draft=off | "
        "TPU_RPA_NATIVE_FP8_PREFILL: target=pvs draft=off | "
        "attention layers: 1 target, 1 draft"
    ) in lines


def test_scope_helpers(monkeypatch, fp8_unset):
    assert native_fp8_scope.requested() == ("", "")
    assert native_fp8_scope.draft_off(True) is False
    monkeypatch.setenv("TPU_RPA_NATIVE_FP8_PREFILL", "0")
    assert native_fp8_scope.draft_off(True) is False
    monkeypatch.setenv("TPU_RPA_NATIVE_FP8", "pvs")
    assert native_fp8_scope.draft_off(True) is True
    assert native_fp8_scope.draft_off(False) is False
    assert "target=pvs draft=off" in native_fp8_scope.scope_line(60, 1)
    assert "TPU_RPA_NATIVE_FP8_PREFILL: target=off draft=off" in (
        native_fp8_scope.scope_line(60, 1)
    )
