# SPDX-License-Identifier: Apache-2.0
"""Request ownership, physical state bytes and bounded diagnostic artifacts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (
    PoolEntry,
    RegionSpec,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    SequenceLayoutPlan,
)

pytestmark = pytest.mark.cpu_test


def test_flag_defaults_off_propagates_and_does_not_change_compile_key(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    flag = "TPU_HIDDEN_NAN_KV_DUMP"
    monkeypatch.delenv(flag, raising=False)
    assert not envs.environment_variables[flag]()
    monkeypatch.setenv(flag, "1")
    assert envs.environment_variables[flag]()
    assert flag in TpuPlatform.additional_env_vars
    assert flag in _TPU_COMPILE_ENV_IGNORED


@pytest.mark.parametrize("lanes", [128, 256])
def test_fp8_carrier_preserves_bf16_bytes_without_false_nan(lanes):
    from vllm_torchtpu.runner.hidden_nan_kv_dump import pool_physical_bytes

    physical = torch.tensor(
        [0x3FFF, 0x3F80, 0x3F7F, 0x3F80] * lanes, dtype=torch.int16
    ).view(torch.uint8)
    carrier = (
        physical.reshape(1, 1, 2, lanes // 128, 128, 4)
        .permute(0, 1, 2, 5, 3, 4)
        .contiguous()
        .reshape(1, 1, 2, 4, lanes)
        .view(torch.float8_e4m3fn)
    )
    restored = pool_physical_bytes(carrier)
    assert torch.equal(restored, physical)
    assert not torch.isnan(restored.view(torch.bfloat16)).any()
    if lanes == 128:
        assert torch.isnan(
            carrier.view(torch.uint8).reshape(-1).view(torch.bfloat16)
        ).any(), "The fixture must expose the old carrier-layout false positive"


def test_decode_gdn_regions_preserves_dtype_and_discards_padding():
    from vllm_torchtpu.runner.hidden_nan_kv_dump import decode_pool_regions

    raw = torch.tensor([1.0, float("nan"), 3.0, 99.0], dtype=torch.bfloat16).view(
        torch.uint8
    )
    pool = PoolEntry(
        "gdn_ssm",
        "model.layers.0.linear_attn",
        0,
        0,
        8,
        1,
        (RegionSpec("gdn_ssm", 0, 4, 2, 2),),
        "bfloat16",
    )
    result = decode_pool_regions(raw, pool, 0, torch.bfloat16)
    torch.testing.assert_close(result, torch.tensor([1.0, 3.0], dtype=torch.bfloat16))


def test_hidden_rows_map_to_current_requests_and_ignore_pcp_padding():
    from vllm_torchtpu.runner.hidden_nan_kv_dump import bad_request_rows

    plan = SequenceLayoutPlan(
        DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
        slice(1, 5),
        5,
        8,
        3,
        4,
        _packed_to_request_major_token_indices=np.array([0, 2, -1, 1, 3, 4, -1, -1]),
    )
    flags = torch.tensor([True, True, False, False])
    assert bad_request_rows(flags, ("a", "b"), np.array([0, 2, 5]), plan) == {"b": [0]}


def test_synchronized_request_snapshot_is_once_and_bounded(
    tmp_path, monkeypatch, caplog
):
    from vllm_torchtpu.runner import hidden_nan_kv_dump as dump

    calls = []
    cache = torch.tensor([[[1.0]], [[float("nan")]], [[7.0]]])
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=0, data_parallel_index=0),
        requests={
            "bad": SimpleNamespace(block_ids=([1],)),
            "good": SimpleNamespace(block_ids=([2],)),
        },
    )
    monkeypatch.setattr(dump.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        dump,
        "synchronize_tensors",
        lambda tensors, wait=True: calls.append((tensors, wait)),
    )

    def entries(runner, req_id, metadata):
        assert req_id == "bad"
        dump.synchronize_tensors([cache], wait=True)
        yield {
            "kind": "fa",
            "layer": "model.layers.1.self_attn",
            "block_ids": [1],
            "tensor": cache[1].clone(),
        }

    monkeypatch.setattr(dump, "request_cache_entries", entries)
    plan = SequenceLayoutPlan(
        DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR, slice(0, 3), 2, 3, 2, 3
    )
    hidden = torch.tensor(
        [[float("nan"), 2.0], [1.0, 2.0], [float("nan"), float("nan")]]
    )
    dumper = dump.HiddenNanKVDumper()
    for _ in range(3):
        dumper.capture(runner, hidden, ("bad", "good"), np.array([0, 1, 2]), plan, {})
    paths = list(tmp_path.rglob("request.pt"))
    assert len(paths) == 1 and paths[0].stat().st_size <= dump.MAX_DUMP_BYTES
    result = torch.load(paths[0], weights_only=True)
    assert result["metadata"]["request_id"] == "bad"
    assert result["metadata"]["hidden_local_rows"] == [0]
    assert result["caches"][0]["kind"] == "fa"
    assert result["caches"][0]["nan_count"] == 1
    assert all(wait for _, wait in calls)
    assert len(calls) == 3, "Later steps must not scan or synchronize again"


def test_disk_error_does_not_change_hidden_or_retry_forever(tmp_path, monkeypatch):
    from vllm_torchtpu.runner import hidden_nan_kv_dump as dump

    monkeypatch.setattr(dump.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(dump, "synchronize_tensors", lambda *args, **kw: None)
    monkeypatch.setattr(dump, "request_cache_entries", lambda *args: iter(()))
    monkeypatch.setattr(
        dump.Path, "mkdir", lambda *args, **kw: (_ for _ in ()).throw(OSError("ENOSPC"))
    )
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=0, data_parallel_index=0)
    )
    hidden = torch.tensor([[float("nan")]])
    plan = SequenceLayoutPlan(
        DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR, slice(0, 1), 1, 1, 1, 1
    )
    dumper = dump.HiddenNanKVDumper()
    dumper.capture(runner, hidden, ("bad",), np.array([0, 1]), plan, {})
    dumper.capture(runner, hidden, ("bad",), np.array([0, 1]), plan, {})
    assert torch.isnan(hidden).all() and dumper.attempted


def test_request_fa_pages_are_synchronized_and_do_not_retain_other_storage(
    monkeypatch,
):
    from vllm_torchtpu.runner import hidden_nan_kv_dump as dump

    cache = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
    md = SimpleNamespace(
        **{
            name: None
            for name in (
                "mamba_state_indices",
                "mamba_slot_read_offsets",
                "mamba_ckpt_indices",
                "seq_lens",
                "block_tables",
                "mamba_request_distribution",
                "request_distribution",
            )
        }
    )
    calls = []
    monkeypatch.setattr(
        dump, "synchronize_tensors", lambda tensors, wait: calls.append((tensors, wait))
    )
    name = "model.layers.3.self_attn"
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[name],
                    kv_cache_spec=SimpleNamespace(page_size_bytes=32),
                )
            ]
        ),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={name: SimpleNamespace(kv_cache=cache)}
            )
        ),
        requests={"bad": SimpleNamespace(block_ids=([0, 1],))},
    )
    entries = list(
        dump.request_cache_entries(
            runner,
            "bad",
            {
                "attn_metadata": {name: md},
                "req_ids": ("good", "bad"),
                "scheduled": [1, 1],
            },
        )
    )
    assert len(calls) == 2 and all(wait for _, wait in calls)
    assert calls[0][0][0] is cache
    assert torch.equal(calls[1][0][0], cache[2:4])
    assert len(entries) == 1 and entries[0]["block_ids"] == [1]
    tensor = entries[0]["tensor"]
    assert torch.equal(tensor, cache[2:4])
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def test_cap_is_explicit_and_stops_reading_remaining_cache(tmp_path, monkeypatch):
    from vllm_torchtpu.runner import hidden_nan_kv_dump as dump

    monkeypatch.setattr(dump.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(dump, "MAX_TENSOR_BYTES", 128)
    monkeypatch.setattr(dump, "synchronize_tensors", lambda *args, **kw: None)

    def entries(*args):
        for block in (1, 2):
            yield {
                "kind": "fa",
                "layer": "model.layers.3.self_attn",
                "block_ids": [block],
                "tensor": torch.ones(16),
            }
        pytest.fail("The dumper must not copy more pages after exhausting its cap")

    monkeypatch.setattr(dump, "request_cache_entries", entries)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=0, data_parallel_index=0)
    )
    plan = SequenceLayoutPlan(
        DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR, slice(0, 1), 1, 1, 1, 1
    )
    dump.HiddenNanKVDumper().capture(
        runner, torch.tensor([[float("nan")]]), ("bad",), np.array([0, 1]), plan, {}
    )
    result = torch.load(next(tmp_path.rglob("request.pt")), weights_only=True)
    assert result["metadata"]["truncated"]
    assert result["metadata"]["first_omitted"]["block_ids"] == [2]
    assert len(result["caches"]) == 1


@pytest.mark.parametrize("container", [list, tuple])
def test_gdn_snapshot_selects_current_requests_read_and_written_slots(
    monkeypatch, container
):
    from vllm_torchtpu.runner import hidden_nan_kv_dump as dump

    class StateSpec:
        pass

    monkeypatch.setattr(dump, "MambaSpec", StateSpec)
    calls = []
    monkeypatch.setattr(
        dump, "synchronize_tensors", lambda tensors, wait: calls.append((tensors, wait))
    )
    conv = torch.arange(8, dtype=torch.bfloat16).reshape(8, 1)
    ssm = conv + 10
    name = "model.layers.0.linear_attn"
    md = SimpleNamespace(
        mamba_state_indices=torch.tensor([1, 4]),
        mamba_slot_read_offsets=torch.tensor([0, 0, 0, 0, 1, 0, 0, 0]),
        mamba_ckpt_indices=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        seq_lens=torch.tensor([4, 5]),
        block_tables=None,
        mamba_request_distribution=torch.tensor([0, 2, 2]),
        request_distribution=None,
    )
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=[name], kv_cache_spec=StateSpec())
            ]
        ),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    name: SimpleNamespace(kv_cache=container((conv, ssm)))
                }
            )
        ),
        _unified_kv_layout=False,
        _mamba_slot_stride=3,
    )
    entries = list(
        dump.request_cache_entries(
            runner,
            "bad",
            {
                "attn_metadata": {name: md},
                "req_ids": ("good", "bad"),
                "scheduled": [1, 2],
            },
        )
    )
    assert len(calls) == 5 and all(wait for _, wait in calls)
    assert all(isinstance(tensor, torch.Tensor) for tensor in calls[0][0])
    assert len(entries) == 4
    assert {e["block_ids"][0] for e in entries} == {4, 5}
    assert {e["kind"] for e in entries} == {"gdn_conv", "gdn_ssm"}
    assert all(e["checkpoint"]["read_slot"] == 5 for e in entries)
    assert all(e["tensor"].untyped_storage().nbytes() == 2 for e in entries)


@pytest.mark.parametrize(
    "tp_kq,tp_v,pcp,local_kq,local_v,page_bytes",
    [
        (1, 1, 1, 1, 1, 1024),
        (8, 8, 8, 1, 1, 1024),
        (1, 8, 8, 1, 1, 1024),
        (16, 64, 8, 2, 8, 3072),
    ],
)
def test_pooled_gdn_manifest_offsets_are_relative_to_raw_not_storage(
    monkeypatch, tp_kq, tp_v, pcp, local_kq, local_v, page_bytes
):
    from vllm_torchtpu.runner import hidden_nan_kv_dump as dump

    class StateSpec:
        page_size_bytes = page_bytes
        shapes = ((3, (2 * local_kq + local_v) * 8), (local_v, 8, 8))
        dtypes = (torch.bfloat16, torch.bfloat16)

    monkeypatch.setattr(dump, "MambaSpec", StateSpec)
    monkeypatch.setattr(dump, "synchronize_tensors", lambda *args, **kw: None)
    backing = torch.zeros(512 + 8 * page_bytes, dtype=torch.uint8)
    cache = (
        backing[512:].reshape(8, 1, page_bytes // 512, 4, 128).view(torch.float8_e4m3fn)
    )
    assert cache.storage_offset() == 512
    name = "model.layers.0.linear_attn"
    module = SimpleNamespace(
        kv_cache=[cache],
        head_k_dim=8,
        head_v_dim=8,
        conv_kernel_size=4,
        tp_head_geometry=SimpleNamespace(
            local_num_kq_heads=tp_kq, local_num_v_heads=tp_v
        ),
    )
    md = SimpleNamespace(
        mamba_state_indices=torch.tensor([1]),
        mamba_slot_read_offsets=None,
        mamba_ckpt_indices=None,
        seq_lens=torch.tensor([1]),
        block_tables=None,
        mamba_request_distribution=torch.tensor([0, 0, 1]),
        request_distribution=None,
    )
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=[name], kv_cache_spec=StateSpec())
            ]
        ),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={name: module})
        ),
        kv_cache_raw_tensors=[cache],
        _unified_kv_layout=True,
        _mamba_ckpt_window=4,
    )
    entries = list(
        dump.request_cache_entries(
            runner,
            "bad",
            {
                "attn_metadata": {name: md},
                "req_ids": ("bad",),
                "scheduled": [1],
            },
        )
    )
    assert {e["kind"] for e in entries} == {"gdn.conv", "gdn.ssm"}
    expected_sizes = [local_v * 64, 3 * (2 * local_kq + local_v) * 8]
    assert sorted(e["tensor"].numel() for e in entries) == sorted(expected_sizes)
    assert all(not e["tensor"].any() for e in entries)


def test_startup_pause_preserves_capture_and_restores_after_error(
    tmp_path, monkeypatch
):
    from vllm_torchtpu.runner import hidden_nan_kv_dump as dump

    calls = []
    monkeypatch.setattr(dump.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        dump, "synchronize_tensors", lambda *args, **kwargs: calls.append(args)
    )
    monkeypatch.setattr(dump, "request_cache_entries", lambda *args: iter(()))
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=0, data_parallel_index=0)
    )
    hidden = torch.tensor([[float("nan")]])
    plan = SequenceLayoutPlan(
        DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR, slice(0, 1), 1, 1, 1, 1
    )
    dumper = dump.HiddenNanKVDumper()
    # Request IDs are caller-owned; startup identity is a lifecycle, not a prefix.
    req_ids = ("__spec_warmup_0__",)
    with pytest.raises(RuntimeError, match="startup failed"), dumper.pause():
        with dumper.pause():
            dumper.capture(runner, hidden, req_ids, np.array([0, 1]), plan, {})
        dumper.capture(runner, hidden, req_ids, np.array([0, 1]), plan, {})
        assert not calls and not dumper.attempted
        raise RuntimeError("startup failed")
    dumper.capture(runner, hidden, req_ids, np.array([0, 1]), plan, {})
    assert calls and dumper.attempted
    saved = torch.load(next(tmp_path.rglob("request.pt")), weights_only=True)
    assert saved["metadata"]["request_id"] == req_ids[0]
