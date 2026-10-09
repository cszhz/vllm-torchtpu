# SPDX-License-Identifier: Apache-2.0
"""Normal MTP batches must never issue diagnostic tensor work."""

import logging
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


@pytest.fixture(autouse=True)
def capture_diagnostic_logger(monkeypatch):
    from vllm_torchtpu.runner import mtp_diagnostic_output

    monkeypatch.setattr(mtp_diagnostic_output, "logger", logging.getLogger("mtp_gate"))


class ReadyEvent:
    created = 0

    def __init__(self):
        type(self).created += 1

    def record(self):
        pass

    def synchronize(self):
        pass


@pytest.mark.parametrize("accepted", [1, 4, 10])
def test_normal_batch_does_not_copy_drafts_or_log(monkeypatch, caplog, accepted):
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=ReadyEvent), raising=False)
    original = AsyncTPUModelRunnerOutput(
        ModelRunnerOutput(req_ids=["a"], req_id_to_index={"a": 0}),
        AsyncTPUCopyState.from_device_chunks(
            [torch.tensor([list(range(accepted + 1)) + [-1] * (10 - accepted)])],
            [1],
        ),
        [],
    )
    md = SimpleNamespace(
        draft_lengths_cpu=np.array([10]), draft_token_ids=torch.arange(10)
    )
    before = ReadyEvent.created
    output = maybe_wrap_mtp_output(
        original,
        SimpleNamespace(
            speculative_config=object(),
            kv_transfer_config=None,
            scheduler_config=SimpleNamespace(async_scheduling=False),
        ),
        ["a"],
        [md],
        [1],
    ).get_output()
    assert output.sampled_token_ids == [list(range(accepted + 1))]
    assert ReadyEvent.created == before, "normal batches must not issue draft D2H"
    assert not caplog.records


@pytest.mark.parametrize(
    "accepted,invalid,low",
    [(0, 0, True), (1, 0, True), (1, 2, False), (2, 0, False), (0, 12, False)],
)
@pytest.mark.parametrize("reused", [False, True])
def test_only_low_effective_batch_checks_nan_and_preserves_hidden(
    tmp_path, monkeypatch, caplog, accepted, invalid, low, reused
):
    from vllm_torchtpu.runner import mtp_tensor_diagnostics as diag

    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=ReadyEvent), raising=False)
    monkeypatch.setattr(diag.tempfile, "gettempdir", lambda: str(tmp_path))
    req_ids = ["a", "b", "c", "d"]
    tokens = torch.full((4, 4), -1, dtype=torch.int32)
    tokens[:, 0] = 99
    tokens[0, 1 : accepted + 1] = 100
    original = AsyncTPUModelRunnerOutput(
        ModelRunnerOutput(
            req_ids=req_ids, req_id_to_index=dict(zip(req_ids, range(4)))
        ),
        AsyncTPUCopyState.from_device_chunks([tokens], [4]),
        [],
    )
    step = diag.MtpDiagnosticStep()
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=0),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        _mamba_state_pos={},
    )
    hidden = torch.tensor([[float("nan"), 2.0], [float("nan"), 0.0]])
    capture = diag.DeferredMtpChunk(
        step, runner, {}, req_ids, [4] * 4, {}, {}, hidden=hidden, real_rows=1
    )
    if reused:
        step.invalidate()
    md = SimpleNamespace(
        draft_lengths_cpu=np.array([3] * 4), draft_token_ids=torch.arange(12)
    )
    wrapped = maybe_wrap_mtp_output(
        original,
        SimpleNamespace(
            speculative_config=object(),
            kv_transfer_config=None,
            scheduler_config=SimpleNamespace(async_scheduling=False),
        ),
        req_ids,
        [md],
        [4],
        [capture],
        {"d": invalid} if invalid < 12 else dict.fromkeys(req_ids, 3),
    )
    assert not list(tmp_path.iterdir())
    result = wrapped.get_output()
    assert bool(list(tmp_path.rglob("*.pt"))) is low
    assert bool(caplog.records) is low
    if low:
        report = result.tensor_diagnostics[0]
        assert report["hidden_states"]["has_nan"] is True
        saved = torch.load(report["hidden_states"]["dump_path"], weights_only=True)
        torch.testing.assert_close(saved["tensor"], hidden[:1], equal_nan=True)
        assert report["kv_cache"]["has_nan"] is (None if reused else False)
        assert report["kv_cache"]["status"] == (
            "unavailable" if reused else "no_related_cache"
        )
    assert not wrapped.captures
    assert not wrapped.drafts
    count = len(caplog.records)
    assert wrapped.get_output() is result
    assert len(caplog.records) == count


def test_discarded_rows_do_not_trigger_a_low_batch(monkeypatch, caplog):
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=ReadyEvent), raising=False)
    original = AsyncTPUModelRunnerOutput(
        ModelRunnerOutput(
            req_ids=["discard", "keep"], req_id_to_index={"discard": 0, "keep": 1}
        ),
        AsyncTPUCopyState.from_device_chunks([torch.tensor([[9, -1], [9, 10]])], [2]),
        [0],
    )
    md = SimpleNamespace(
        draft_lengths_cpu=np.array([100, 1]), draft_token_ids=torch.arange(101)
    )
    before = ReadyEvent.created
    result = maybe_wrap_mtp_output(
        original,
        SimpleNamespace(
            speculative_config=object(),
            kv_transfer_config=None,
            scheduler_config=SimpleNamespace(async_scheduling=False),
        ),
        ["discard", "keep"],
        [md],
        [2],
    ).get_output()
    assert result.sampled_token_ids == [[], [9, 10]]
    assert ReadyEvent.created == before
    assert not caplog.records


def test_cache_reuse_waits_until_triggered_capture_finishes(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from vllm_torchtpu.runner import mtp_tensor_diagnostics as diag

    entered, release, invalidating, invalidated = (Event() for _ in range(4))
    inspect = diag.inspect_tensor

    def slow_copy(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return inspect(*args, **kwargs)

    monkeypatch.setattr(diag, "inspect_tensor", slow_copy)
    step = diag.MtpDiagnosticStep()
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=0),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        _mamba_state_pos={},
    )
    capture = diag.DeferredMtpChunk(
        step, runner, {}, ["a"], [4], {}, {}, hidden=torch.ones(1, 2), real_rows=1
    )

    def reuse():
        invalidating.set()
        step.invalidate()
        invalidated.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(capture.collect)
        assert entered.wait(timeout=5)
        next_step = executor.submit(reuse)
        assert invalidating.wait(timeout=5)
        try:
            assert not invalidated.wait(timeout=0.05)
        finally:
            release.set()
        assert pending.result(timeout=5)["cache_observation"] == "after_sampling"
        next_step.result(timeout=5)
    assert invalidated.is_set()
