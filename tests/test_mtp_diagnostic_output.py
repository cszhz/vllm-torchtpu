# SPDX-License-Identifier: Apache-2.0
"""Verified draft snapshots must stay aligned across async batch changes."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.outputs import ModelRunnerOutput

from vllm_torchtpu.runner.mtp_diagnostic_output import maybe_wrap_mtp_output
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
)

pytestmark = pytest.mark.cpu_test


class CopyEvent:
    def __init__(self):
        self.recorded = False
        self.waited = False

    def record(self):
        self.recorded = True

    def synchronize(self):
        assert self.recorded
        self.waited = True


@pytest.fixture
def output(monkeypatch):
    monkeypatch.setenv("TPU_MTP_ZERO_ACCEPTANCE_LOG", "1")
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=CopyEvent), raising=False)
    base = ModelRunnerOutput(
        req_ids=["a", "b", "c", "d", "e"],
        req_id_to_index={rid: i for i, rid in enumerate("abcde")},
        prompt_logprobs_dict={"a": None},
    )
    copies = AsyncTPUCopyState.from_device_chunks(
        [torch.tensor([[99, -1], [98, -1], [97, -1], [96, -1], [95, -1], [0, 0]])],
        [5],
    )
    return AsyncTPUModelRunnerOutput(base, copies, [2]), base


def test_verified_drafts_survive_padding_reordering_and_transport(output):
    original, base = output
    config = SimpleNamespace(
        speculative_config=object(),
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    first = SimpleNamespace(
        draft_lengths_cpu=np.array([2, 1, 0]),
        draft_token_ids=torch.tensor([101, 102, 201, 999, 999, 999]),
    )
    last = SimpleNamespace(
        draft_lengths_cpu=np.array([1]), draft_token_ids=torch.tensor([501, 999])
    )
    req_ids = base.req_ids.copy()
    wrapped = maybe_wrap_mtp_output(
        original, config, req_ids, [first, None, last], [3, 1, 1]
    )
    # The next execute_model may reorder requests and replace cached metadata.
    req_ids.reverse()
    first.draft_lengths_cpu[:] = 0
    first.draft_token_ids = torch.tensor([901, 902, 903])
    result = wrapped.get_output()
    assert not wrapped.drafts
    assert result.verified_draft_token_ids == {
        "a": [101, 102],
        "b": [201],
        "c": [],
        "d": [],
        "e": [501],
    }
    assert result.sampled_token_ids == [[99], [98], [], [96], [95]]
    assert result.req_id_to_index is base.req_id_to_index
    assert result.prompt_logprobs_dict is base.prompt_logprobs_dict
    assert (
        pickle.loads(pickle.dumps(result)).verified_draft_token_ids
        == result.verified_draft_token_ids
    )


@pytest.mark.parametrize(
    "enabled,speculative,consumer",
    [
        (False, True, None),
        (False, False, None),
        (True, False, None),
        (True, True, False),
    ],
)
def test_inactive_output_preserves_identity(
    output, monkeypatch, enabled, speculative, consumer
):
    original, base = output
    monkeypatch.setenv("TPU_MTP_ZERO_ACCEPTANCE_LOG", str(int(enabled)))
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=object() if speculative else None,
        kv_transfer_config=None
        if consumer is None
        else SimpleNamespace(is_kv_consumer=consumer),
    )
    assert maybe_wrap_mtp_output(original, config, base.req_ids, [], []) is original


def test_bootstrap_without_drafts_preserves_identity(output):
    original, base = output
    config = SimpleNamespace(
        speculative_config=object(),
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    assert (
        maybe_wrap_mtp_output(original, config, base.req_ids, [None], [5]) is original
    )


@pytest.mark.parametrize("env_value", [None, "1"])
def test_low_acceptance_drafts_enabled_by_default_or_explicitly(
    output, monkeypatch, env_value
):
    original, base = output
    if env_value is None:
        monkeypatch.delenv("TPU_MTP_ZERO_ACCEPTANCE_LOG", raising=False)
    else:
        monkeypatch.setenv("TPU_MTP_ZERO_ACCEPTANCE_LOG", env_value)
    config = SimpleNamespace(
        speculative_config=object(),
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    md = SimpleNamespace(
        draft_lengths_cpu=np.array([3, 3, 0, 0, 0]),
        draft_token_ids=torch.arange(6),
    )
    wrapped = maybe_wrap_mtp_output(original, config, base.req_ids, [md], [5])
    result = pickle.loads(pickle.dumps(wrapped.get_output()))
    assert result.verified_draft_token_ids["a"] == [0, 1, 2]
    assert result.verified_draft_token_ids["b"] == [3, 4, 5]


@pytest.mark.parametrize("nan", [False, True])
def test_tensor_nan_report_saves_full_values_and_explicit_clean_status(tmp_path, nan):
    from vllm_torchtpu.runner.mtp_tensor_diagnostics import inspect_tensor

    tensor = torch.tensor([[1.0, float("nan") if nan else 2.0], [3.0, 4.0]])
    report = inspect_tensor(tensor, tmp_path / "hidden.pt", {"request_ids": ["a"]})
    assert report["has_nan"] is nan
    assert report["nan_count"] == int(nan)
    if nan:
        saved = torch.load(report["dump_path"], weights_only=True)
        torch.testing.assert_close(saved["tensor"], tensor, equal_nan=True)
        assert saved["metadata"]["request_ids"] == ["a"]
    else:
        assert report["dump_path"] is None
        assert not list(tmp_path.iterdir())


def test_pooled_nan_check_uses_typed_live_regions_not_raw_bf16_bits(tmp_path):
    from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (
        PoolEntry,
        RegionSpec,
    )
    from vllm_torchtpu.runner.mtp_tensor_diagnostics import inspect_pool

    # Finite fp32 values may contain a bf16 half-word with a NaN bit pattern.
    state = torch.tensor([0x3F807FC1, 0x40000000], dtype=torch.int32).view(
        torch.float32
    )
    raw = torch.zeros(4, 8, dtype=torch.bfloat16)
    raw[2, :4] = state.view(torch.bfloat16)
    raw[2, 4:] = float("nan")  # unused page padding must not be checked
    pool = PoolEntry(
        "gdn.ssm",
        "model.layers.0",
        0,
        0,
        16,
        4,
        (RegionSpec("ssm", 0, 8, 8, 1),),
        "float32",
    )
    clean = inspect_pool(raw, pool, [2], tmp_path, {"phase": "before_forward"})
    assert clean["has_nan"] is False
    raw[2, :4] = torch.tensor([float("nan"), 2.0]).view(torch.bfloat16)
    bad = inspect_pool(raw, pool, [2], tmp_path, {"phase": "before_forward"})
    assert bad["has_nan"] is True
    assert bad["nan_count"] == 1
    assert bad["nan_blocks"] == [2]
    saved = torch.load(bad["dump_paths"][0], weights_only=True)
    assert saved["tensor"].dtype == torch.float32
    assert saved["metadata"]["block_id"] == 2


def test_checkpoint_snapshot_uses_noncontiguous_checkpoint_table_and_copies_offsets():
    from vllm_torchtpu.runner.mtp_tensor_diagnostics import checkpoint_snapshot

    offsets = torch.zeros(24, dtype=torch.int32)
    offsets[5] = 2
    metadata = SimpleNamespace(
        mamba_state_indices=torch.tensor([5, 0]),
        mamba_slot_read_offsets=offsets,
        mamba_ckpt_indices=torch.tensor([[5, 9, 17, 20], [0, 0, 0, 0]]),
        seq_lens=torch.tensor([387, 0]),
    )
    result = checkpoint_snapshot(metadata, ["a"], [4], {"a": 1}, {"a": 0})
    offsets[5] = 0
    assert result == [
        {
            "request_id": "a",
            "slot": 5,
            "read_offset": 2,
            "read_slot": 17,
            "checkpoint_slots": [5, 9, 17, 20],
            "state_column": 1,
            "previous_state_column": 0,
            "num_computed_tokens": 383,
            "num_scheduled_tokens": 4,
        }
    ]


@pytest.mark.parametrize(
    "bad_source,bad_hidden,bad_attention,bad_written",
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, True, False),
        (False, False, False, True),
    ],
)
def test_chunk_checks_preserve_read_state_and_ignore_hidden_padding(
    tmp_path, monkeypatch, caplog, bad_source, bad_hidden, bad_attention, bad_written
):
    import logging

    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_torchtpu.runner import mtp_tensor_diagnostics as diagnostic

    monkeypatch.setattr(diagnostic.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(diagnostic, "logger", logging.getLogger("mtp_tensor_test"))
    caplog.set_level(logging.INFO, logger="mtp_tensor_test")
    name = "model.layers.0.linear_attn"
    attn_name = "model.layers.1.self_attn.attn"
    conv = torch.zeros(8, 3, 1, 6, dtype=torch.bfloat16)
    ssm = torch.zeros(8, 1, 2, 2)
    if bad_source:
        ssm[3, 0, 0, 0] = float("nan")
    attention = torch.zeros(8, 4, 1, 2, 2, dtype=torch.bfloat16)
    if bad_attention:
        attention[2, 0, 0, 0, 0] = float("nan")
    attention[7] = float("nan")  # belongs to another request
    offsets = torch.zeros(8, dtype=torch.int32)
    offsets[1] = 2
    md = SimpleNamespace(
        mamba_state_indices=torch.tensor([1, 0]),
        mamba_slot_read_offsets=offsets,
        mamba_ckpt_indices=None,
        seq_lens=torch.tensor([20, 0]),
        mamba_request_distribution=torch.tensor([1, 1, 1]),
    )
    spec = MambaSpec(
        block_size=16,
        shapes=((3, 1, 6), (1, 2, 2)),
        dtypes=(torch.bfloat16, torch.float32),
    )
    module = SimpleNamespace(
        kv_cache=(conv, ssm),
        tp_head_geometry=SimpleNamespace(local_num_kq_heads=1, local_num_v_heads=1),
        head_k_dim=2,
        head_v_dim=2,
        conv_kernel_size=4,
    )
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=0),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=spec, layer_names=[name]),
                SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(
                        block_size=4, num_kv_heads=1, head_size=2
                    ),
                    layer_names=[attn_name],
                ),
            ]
        ),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: "LBNHC"),
            compilation_config=SimpleNamespace(
                static_forward_context={
                    name: module,
                    attn_name: SimpleNamespace(kv_cache=attention),
                }
            ),
        ),
        kv_cache_raw_tensors=[],
        bm_pool_layout=None,
        _mamba_state_pos={"a": 1},
        _mamba_ckpt_window=1,
        _mamba_slot_stride=4,
        _unified_kv_layout=False,
        requests={"a": SimpleNamespace(block_ids=([1], [2]))},
    )
    capture = diagnostic.MtpTensorDiagnostics(
        runner, {name: md, attn_name: object()}, ["a"], [4], {"a": 0}
    )
    ssm.zero_()  # Forward overwrites the bad input state.
    if bad_written:
        ssm[4, 0, 0, 0] = float("nan")  # Last compact draft checkpoint.
    offsets.zero_()  # A later sampling update must not rewrite this snapshot.
    hidden = torch.tensor(
        [[float("nan") if bad_hidden else 1.0, 2.0], [float("nan"), float("nan")]]
    )
    result = capture.before_logits(hidden, 1)
    assert result["hidden_states"]["has_nan"] is bad_hidden
    assert result["kv_cache"]["has_nan"] is (bad_source or bad_attention or bad_written)
    assert result["kv_cache"]["status"] == "checked"
    assert result["gdn_checkpoints"][0]["requests"][0]["read_offset"] == 2
    assert result["gdn_checkpoints"][0]["requests"][0]["read_slot"] == 3
    assert all(7 not in c["block_ids"] for c in result["kv_cache"]["checks"])
    if bad_source:
        before = next(
            c
            for c in result["kv_cache"]["checks"]
            if c["phase"] == "before_forward" and c["has_nan"]
        )
        saved = torch.load(before["dump_paths"][0], weights_only=True)
        assert torch.isnan(saved["tensor"]).any()
    if not (bad_source or bad_hidden or bad_attention or bad_written):
        assert not list(tmp_path.iterdir())
    pp_result = capture.after_forward()
    assert pp_result["hidden_states"] == {"has_nan": None, "status": "not_logits_stage"}
    assert pp_result["coverage"] == "rank_local_layers"

    ticket = diagnostic.MtpDiagnosticStep()
    deferred = diagnostic.DeferredMtpChunk(
        ticket,
        runner,
        {name: md, attn_name: object()},
        ["a"],
        [4],
        {"a": 0},
        {"a": 1},
        hidden=hidden,
        real_rows=1,
    )
    late = deferred.collect()
    assert late["cache_observation"] == "after_sampling"
    assert late["input_state_snapshot_available"] is False
    assert late["gdn_checkpoints"][0]["requests"][0]["read_offset"] == 0
    assert (
        late["gdn_checkpoints"][0]["requests"][0]["read_offset_phase"]
        == "after_sampling"
    )
    assert late["kv_cache"]["has_nan"] is (bad_attention or bad_written)
    ticket.invalidate()
    ssm.fill_(float("nan"))
    stale = deferred.collect()
    assert stale["kv_cache"]["has_nan"] is None
    assert stale["kv_cache"]["status"] == "unavailable"
    assert stale["hidden_states"]["has_nan"] is bad_hidden


def test_invalid_checkpoint_offset_is_reported_instead_of_indexing_another_slot():
    from vllm_torchtpu.runner.mtp_tensor_diagnostics import checkpoint_snapshot

    md = SimpleNamespace(
        mamba_state_indices=torch.tensor([2]),
        mamba_slot_read_offsets=torch.tensor([0, 0, 8]),
        mamba_ckpt_indices=torch.tensor([[2, 4, 7, 9]]),
        seq_lens=torch.tensor([16]),
    )
    record = checkpoint_snapshot(md, ["a"], [4], {"a": 1}, {"a": 0})[0]
    assert record["read_offset"] == 8
    assert record["read_slot"] is None
    assert "outside" in record["address_error"]


def test_dump_io_error_does_not_hide_nan_or_abort_sampling(tmp_path, monkeypatch):
    from vllm_torchtpu.runner.mtp_tensor_diagnostics import inspect_tensor

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    report = inspect_tensor(torch.tensor([float("nan")]), tmp_path / "bad.pt", {})
    assert report["has_nan"] is True
    assert report["dump_path"] is None
    assert report["dump_error"] == "disk full"


@pytest.mark.parametrize("bad", [False, True])
@pytest.mark.parametrize("legacy_env", [None, "HND", "NHD"])
def test_hnd_attention_pages_use_seq_on_lane_manifest(
    tmp_path, monkeypatch, bad, legacy_env
):
    from vllm.v1.kv_cache_layout import KVCacheLayout

    from vllm_torchtpu.runner import mtp_tensor_diagnostics as diagnostic

    if legacy_env is None:
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", legacy_env)
    monkeypatch.setattr(diagnostic.tempfile, "gettempdir", lambda: str(tmp_path))
    name = "model.layers.0.self_attn.attn"
    cache = torch.zeros((8, 2, 8, 4, 128), dtype=torch.float8_e4m3fn)
    # Manager block spans two kernel pages; verify the second page too.
    if bad:
        cache[3, 0, 0, 0, 0] = float("nan")
    module = SimpleNamespace(kv_cache=cache)
    spec = SimpleNamespace(block_size=256, num_kv_heads=1, head_size=32)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=0),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=[name])]
        ),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC
            ),
            compilation_config=SimpleNamespace(static_forward_context={name: module}),
        ),
        kv_cache_raw_tensors=[],
        bm_pool_layout=None,
        requests={"a": SimpleNamespace(block_ids=([1],))},
    )
    capture = diagnostic.MtpTensorDiagnostics(runner, {name: object()}, ["a"], [1], {})
    result = capture.before_logits(torch.ones(1, 2), 1)
    assert result["kv_cache"]["status"] == "checked"
    assert result["kv_cache"]["has_nan"] is bad
