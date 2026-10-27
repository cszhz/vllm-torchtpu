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
"""TPU_RPA_DECODE_BATCHED: the decode-program settings and how the batched
RPA entry hands them to the kernel."""

import functools
from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter import attention, batched_rpa_decode

pytestmark = pytest.mark.cpu_test

MIB = 1 << 20


def _vllm_config(max_num_seqs, num_speculative_tokens=0, max_model_len=9216):
    spec = (
        SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
        if num_speculative_tokens
        else None
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=spec,
    )


@pytest.fixture
def v7x(monkeypatch):
    monkeypatch.setattr(
        batched_rpa_decode.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(vmem_capacity_bytes=64 * MIB),
    )


def test_off_by_default(monkeypatch, v7x):
    monkeypatch.delenv("TPU_RPA_DECODE_BATCHED", raising=False)
    assert batched_rpa_decode.decode_tuning(_vllm_config(4)) is None
    assert batched_rpa_decode.kernel_kwargs(None, 8) == {}


# The decode batch follows the engine's request slots, capped at the
# wrapper's default of 8: 4 at 4 users, 8 from 8 users up.
@pytest.mark.parametrize("max_num_seqs, batch", [(1, 1), (4, 4), (8, 8), (32, 8)])
def test_tuning_halves_vmem_and_caps_the_batch(monkeypatch, v7x, max_num_seqs, batch):
    monkeypatch.setenv("TPU_RPA_DECODE_BATCHED", "1")
    tuning = batched_rpa_decode.decode_tuning(_vllm_config(max_num_seqs))
    assert tuning == batched_rpa_decode.DecodeTuning(
        vmem_limit_bytes=32 * MIB, decode_batch_size=batch
    )


# An 8k/1k deployment takes the decode settings; a long-context one (AgentX
# serves max_model_len 262144) keeps the default, larger KV blocks.
@pytest.mark.parametrize(
    "max_model_len, takes",
    [(9216, True), (16384, True), (16385, False), (262144, False)],
)
def test_tuning_refused_past_the_model_length_bound(
    monkeypatch, v7x, max_model_len, takes
):
    monkeypatch.setenv("TPU_RPA_DECODE_BATCHED", "1")
    tuning = batched_rpa_decode.decode_tuning(
        _vllm_config(8, 3, max_model_len=max_model_len)
    )
    assert (tuning is not None) == takes
    kwargs = batched_rpa_decode.kernel_kwargs(tuning, 32)
    assert ("vmem_limit_bytes" in kwargs) == takes


@pytest.mark.parametrize(
    "max_num_seqs, k, armed", [(16, 3, "decode"), (32, 3, "default"), (64, 0, "decode")]
)
def test_tuning_logs_which_settings_decode_takes(
    monkeypatch, v7x, max_num_seqs, k, armed
):
    monkeypatch.setenv("TPU_RPA_DECODE_BATCHED", "1")
    messages = []
    monkeypatch.setattr(
        batched_rpa_decode.logger,
        "info_once",
        lambda msg, *args: messages.append(msg % args),
    )
    batched_rpa_decode.decode_tuning(_vllm_config(max_num_seqs, k))
    (message,) = messages
    assert f"runs the {armed} settings" in message
    assert f"<= {64 // (1 + k)} requests" in message


TUNING = batched_rpa_decode.DecodeTuning(vmem_limit_bytes=32 * MIB, decode_batch_size=4)


@pytest.mark.parametrize(
    "rows, decode_query_size, tuning, expected",
    [
        (64, 1, TUNING, {"vmem_limit_bytes": 32 * MIB, "decode_batch_size": 4}),
        (128, 1, TUNING, {}),
        (
            64,
            4,
            TUNING,
            {
                "vmem_limit_bytes": 32 * MIB,
                "decode_batch_size": 4,
                "decode_query_size": 4,
            },
        ),
        (128, 4, TUNING, {"decode_query_size": 4}),
        (8, 1, None, {}),
    ],
)
def test_batched_entry_passes_the_settings_to_the_kernel(
    monkeypatch, rows, decode_query_size, tuning, expected
):
    seen = {}

    def impl(*args, rpa_func, **kwargs):
        seen["rpa_func"] = rpa_func

    monkeypatch.setattr(attention, "_pallas_rpa_kernel_impl", impl)
    query = SimpleNamespace(shape=(rows, 4, 128))
    attention._pallas_rpa_kernel_batched(
        None,
        query,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        mesh=None,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=None,
        decode_query_size=decode_query_size,
        decode_tuning=tuning,
    )
    rpa_func = seen["rpa_func"]
    if not expected:
        assert rpa_func is attention.ragged_paged_attention_batched
    else:
        assert isinstance(rpa_func, functools.partial)
        assert rpa_func.func is attention.ragged_paged_attention_batched
        assert rpa_func.keywords == expected


def test_q_tail_rungs_and_toggle(monkeypatch):
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, kernel

    monkeypatch.delenv("TPU_RPA_Q_TAIL_TRIM", raising=False)
    assert kernel.q_tail_trim_enabled()
    cfgs_1024 = SimpleNamespace(
        mode=configs.RpaCase.MIXED, bq_sz=1024, bq_c_sz=512
    )
    assert kernel._q_tail_rungs(cfgs_1024) == (128, 256, 512, 1024)
    cfgs_decode = SimpleNamespace(
        mode=configs.RpaCase.DECODE, bq_sz=4, bq_c_sz=4
    )
    assert kernel._q_tail_rungs(cfgs_decode) == (4,)

    monkeypatch.setenv("TPU_RPA_Q_TAIL_TRIM", "0")
    assert not kernel.q_tail_trim_enabled()
    assert kernel._q_tail_rungs(cfgs_1024) == (1024,)


def test_batched_rpa_schedule_scope_hoists_across_layers(monkeypatch):
    import jax.numpy as jnp
    import torch
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    monkeypatch.delenv("TPU_RPA_HOIST_SCHEDULE", raising=False)
    assert wrapper.hoist_schedule_enabled()
    d_shapes, m_shapes = wrapper.get_rpa_schedule_shapes(
        total_q_tokens=2048,
        num_q_heads=4,
        num_kv_heads=1,
        head_dim=256,
        page_size=128,
        max_num_seqs=16,
        num_page_indices=16 * 72,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        decode_query_size=4,
    )
    assert len(d_shapes) == 14
    assert len(m_shapes) == 14

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
    )
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])

    schedule_calls = []
    kernel_calls = []

    def fake_schedule_op(*args):
        schedule_calls.append(len(args))
        return tuple(torch.zeros((1,), dtype=torch.int32) for _ in range(28))

    def make_fake_scheduled_kernel(k_scale, v_scale):
        def _kernel(kv_cache, query, *args):
            kernel_calls.append((k_scale, v_scale, len(args)))
            return query

        return _kernel

    with set_current_vllm_config(config):
        layers = [
            (
                attention.PallasBatchedRPAAttentionBackendImpl(
                    num_heads=4,
                    head_size=256,
                    scale=256**-0.5,
                    num_kv_heads=1,
                    alibi_slopes=None,
                    sliding_window=None,
                    kv_cache_dtype="bfloat16",
                ),
                SimpleNamespace(
                    _k_scale_float=1.0 + 0.1 * i,
                    _v_scale_float=2.0 + 0.1 * i,
                ),
            )
            for i in range(3)
        ]

    for impl, layer_obj in layers:
        assert not impl.runs_batched_rpa_schedule()
        assert impl.supports_hoisted_rpa_schedule()
        monkeypatch.setattr(
            impl,
            "_build_rpa_schedule_op",
            lambda block_major=False: fake_schedule_op,
        )
        monkeypatch.setattr(
            impl,
            "_build_rpa_scheduled_kernel",
            lambda q_s, k_s, v_s, skip_kv_update=False, block_major=False, _l=layer_obj: (  # noqa: E501
                make_fake_scheduled_kernel(_l._k_scale_float, _l._v_scale_float)
            ),
        )
        monkeypatch.setattr(
            impl,
            "_build_rpa_kernel",
            lambda *a, **kw: (lambda kv_cache, query, *rest: query),
        )

    q = torch.zeros((32, 4, 256), dtype=torch.bfloat16)
    k = torch.zeros((32, 1, 256), dtype=torch.bfloat16)
    v = torch.zeros((32, 1, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((8, 128, 1, 2, 256), dtype=torch.bfloat16)
    md = AttentionMetadata(
        input_positions=torch.zeros((32,), dtype=torch.int32),
        block_tables=torch.zeros((64,), dtype=torch.int32),
        seq_lens=torch.ones((8,), dtype=torch.int32),
        query_start_loc=torch.arange(9, dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    mesh = SimpleNamespace(shape={"model": 1})
    with (
        set_vllm_model_wrapper_context(mesh=mesh, vllm_config=None),
        attention.batched_rpa_schedule_scope(),
    ):
        for impl, layer_obj in layers:
            impl.forward(layer_obj, q, k, v, kv_cache, md)

    # Schedule op runs once on the first layer; all 3 layers receive the 28
    # schedule tensors (7 metadata/kv/sinks + 28 schedule = 35 args after kv, q).
    assert schedule_calls == [7]
    assert len(kernel_calls) == 3
    assert all(arg_count == 35 for _, _, arg_count in kernel_calls)


def test_batched_rpa_schedule_scope_torch_compile_mark_dynamic(monkeypatch):
    import torch
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    monkeypatch.delenv("TPU_RPA_HOIST_SCHEDULE", raising=False)
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
    )
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    mesh = SimpleNamespace(shape={"model": 1})
    with (
        set_current_vllm_config(config),
        set_vllm_model_wrapper_context(mesh=mesh, vllm_config=config),
    ):
        layers = [
            (
                attention.PallasBatchedRPAAttentionBackendImpl(
                    num_heads=4,
                    head_size=256,
                    scale=256**-0.5,
                    num_kv_heads=1,
                    alibi_slopes=None,
                    sliding_window=None,
                    kv_cache_dtype="bfloat16",
                ),
                SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0),
            )
            for _ in range(3)
        ]
        for impl, layer_obj in layers:
            impl.initialize_kernel(layer_obj)

    captured_graphs = []

    def capture_backend(gm, example_inputs):
        captured_graphs.append(gm)
        return lambda *args: (torch.zeros(4, 4, 256, dtype=torch.bfloat16),)

    def step(
        q,
        k,
        v,
        kv_cache,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
    ):
        md = AttentionMetadata(
            input_positions=torch.zeros((q.shape[0],), dtype=torch.int32),
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )
        with attention.batched_rpa_schedule_scope():
            for impl, layer_obj in layers:
                q = impl.forward(layer_obj, q, k, v, kv_cache, md)
        return q

    q = torch.zeros((4, 4, 256), dtype=torch.bfloat16)
    k = torch.zeros((4, 1, 256), dtype=torch.bfloat16)
    v = torch.zeros((4, 1, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((8, 128, 1, 2, 256), dtype=torch.bfloat16)
    seq_lens = torch.ones((8,), dtype=torch.int32)
    block_tables = torch.zeros((64,), dtype=torch.int32)
    query_start_loc = torch.arange(9, dtype=torch.int32)
    request_distribution = torch.tensor([0, 0, 1], dtype=torch.int32)

    torch._dynamo.mark_dynamic(q, 0)
    torch._dynamo.mark_dynamic(k, 0)
    torch._dynamo.mark_dynamic(v, 0)

    compiled = torch.compile(step, backend=capture_backend, fullgraph=True)
    with set_vllm_model_wrapper_context(mesh=mesh, vllm_config=config):
        compiled(
            q,
            k,
            v,
            kv_cache,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
        )
    assert len(captured_graphs) == 1
    gm = captured_graphs[0]
    sched_nodes = [n.name for n in gm.graph.nodes if "rpa_schedule" in str(n.target)]
    kern_nodes = [
        n.name
        for n in gm.graph.nodes
        if "rpa_kernel_batched_scheduled" in str(n.target)
    ]
    assert len(sched_nodes) == 1
    assert len(kern_nodes) == 3




# TPU_RPA_DECODE_LANES: the decode batch (lanes) of every batched RPA program.
@pytest.mark.parametrize(
    "rows, tuning, expected",
    [
        (112, None, {"decode_batch_size": 4}),
        (64, None, {"decode_batch_size": 4}),
        (64, TUNING, {"vmem_limit_bytes": 32 * MIB, "decode_batch_size": 4}),
        (2048, TUNING, {"decode_batch_size": 4}),
    ],
)
def test_decode_lanes_env_sets_the_decode_batch(monkeypatch, rows, tuning, expected):
    monkeypatch.setenv("TPU_RPA_DECODE_LANES", "4")
    assert batched_rpa_decode.kernel_kwargs(tuning, rows) == expected


def test_decode_lanes_off_by_default(monkeypatch):
    monkeypatch.delenv("TPU_RPA_DECODE_LANES", raising=False)
    assert batched_rpa_decode.kernel_kwargs(None, 112) == {}
    monkeypatch.setenv("TPU_RPA_DECODE_LANES", "0")
    assert batched_rpa_decode.kernel_kwargs(None, 112) == {}


@pytest.mark.parametrize("rows", [64, 112])
def test_decode_lanes_reach_the_kernel_and_the_schedule(monkeypatch, rows):
    # The served DP4xTP2 path (no decode tuning, 4 query rows per request):
    # the kernel entry and the hoisted schedule must get the same lanes.
    monkeypatch.setenv("TPU_RPA_DECODE_LANES", "4")
    seen = {}

    def impl(*args, rpa_func, **kwargs):
        seen["rpa_func"] = rpa_func

    monkeypatch.setattr(attention, "_pallas_rpa_kernel_impl", impl)
    attention._pallas_rpa_kernel_batched(
        None, SimpleNamespace(shape=(rows, 16, 256)), *([None] * 10),
        mesh=None, sliding_window=None, skip_kv_update=False, kv_layout=None,
        decode_query_size=4, decode_tuning=None,
    )
    rpa_func = seen["rpa_func"]
    assert isinstance(rpa_func, functools.partial)
    assert rpa_func.keywords == {"decode_batch_size": 4, "decode_query_size": 4}
    # The schedule builder reads the same helper with the program's rows.
    assert batched_rpa_decode.kernel_kwargs(None, rows) == {"decode_batch_size": 4}
