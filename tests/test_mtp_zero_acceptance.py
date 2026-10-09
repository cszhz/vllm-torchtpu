# SPDX-License-Identifier: Apache-2.0
"""Batch low-acceptance logging contracts; no model or TPU execution."""

import json
import logging
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core import mtp_zero_acceptance as hook
from vllm_torchtpu.runner.mtp_diagnostic_output import MtpDiagnosticModelRunnerOutput

pytestmark = pytest.mark.cpu_test


def diagnostic_output(*request_ids):
    return MtpDiagnosticModelRunnerOutput(
        req_ids=list(request_ids),
        req_id_to_index={rid: i for i, rid in enumerate(request_ids)},
        verified_draft_token_ids={rid: [101, 102, 103] for rid in request_ids},
    )


@pytest.fixture
def scheduler(monkeypatch, caplog):
    monkeypatch.setattr(hook, "logger", logging.getLogger("mtp_hook_test"))
    caplog.set_level(logging.WARNING, logger="mtp_hook_test")
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.log_stats = True
    scheduler.num_spec_tokens = 3
    scheduler.parallel_config = SimpleNamespace(data_parallel_index=2)
    scheduler.result = {}

    def consume(scheduler_output, model_runner_output):
        stats = None
        # The scheduler decides which live outputs contribute to statistics.
        # Here the consumer supplies already-resolved IDs, not runner row order.
        for req_id, drafted, accepted, invalid in scheduler_output:
            if not drafted:
                continue
            stats = scheduler.make_spec_decoding_stats(
                stats, drafted, accepted, invalid, req_id
            )
        scheduler.stats = stats
        return scheduler.result

    scheduler.update_from_output = consume
    hook._BatchLowAcceptance(scheduler)
    return scheduler


@pytest.mark.parametrize("log_stats", [True, False])
def test_zero_batch_logs_all_effective_request_ids(scheduler, caplog, log_stats):
    scheduler.log_stats = log_stats
    rows = [
        ("req-b", 3, 0, None),
        ("req-a", 2, 0, {"req-a": 1}),
        ("grammar-only", 3, 0, {"grammar-only": 3}),
    ]
    output = diagnostic_output("req-b", "req-a", "grammar-only", "prefill")
    output.verified_draft_token_ids["prefill"] = []
    output.tensor_diagnostics = [
        {
            "capture_id": "batch-a",
            "hidden_states": {"has_nan": False},
            "kv_cache": {"has_nan": False},
        }
    ]
    result = scheduler.update_from_output(rows, output)

    assert result is scheduler.result
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "MTP_LOW_ACCEPTANCE" in message
    assert "scope=batch" in message
    assert "dp_index=2" in message
    assert "num_draft_tokens=4" in message
    assert "num_accepted_tokens=0" in message
    diagnostics = message.split("tensor_diagnostics=", 1)[1].split(
        " verified_draft_token_ids=", 1
    )[0]
    assert json.loads(diagnostics) == output.tensor_diagnostics
    assert json.loads(message.split("request_ids=", 1)[1]) == ["req-b", "req-a"]
    drafts = message.split("verified_draft_token_ids=", 1)[1].split(" request_ids=", 1)[
        0
    ]
    assert json.loads(drafts) == output.verified_draft_token_ids
    if log_stats:
        assert scheduler.stats.num_draft_tokens == 4
        assert scheduler.stats.num_accepted_tokens == 0
    else:
        assert scheduler.stats is None


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [("prefill", 0, 0, None)],
        [("invalid", 3, 0, {"invalid": 3})],
        [("zero", 3, 0, None), ("accepted", 3, 1, None)],
    ],
)
def test_healthy_or_empty_batch_does_not_log(scheduler, caplog, rows):
    scheduler.update_from_output(rows, None)
    assert not caplog.records


def test_batches_do_not_retain_previous_acceptance_or_request_ids(scheduler, caplog):
    scheduler.update_from_output([("old", 3, 1, None)], None)
    scheduler.update_from_output([("zero-a", 3, 0, None)], diagnostic_output("zero-a"))
    scheduler.update_from_output([("zero-b", 1, 0, None)], diagnostic_output("zero-b"))
    scheduler.update_from_output([], None)
    assert len(caplog.records) == 2
    assert json.loads(caplog.messages[0].split("request_ids=", 1)[1]) == ["zero-a"]
    assert json.loads(caplog.messages[1].split("request_ids=", 1)[1]) == ["zero-b"]


def test_scheduler_subset_can_be_low_without_a_worker_snapshot(scheduler, caplog):
    from vllm.v1.outputs import ModelRunnerOutput

    output = ModelRunnerOutput(
        req_ids=["aborted-accepted", "remaining"],
        req_id_to_index={"aborted-accepted": 0, "remaining": 1},
        sampled_token_ids=[[1, 2, 3], [4]],
    )
    scheduler.update_from_output([("remaining", 3, 0, None)], output)
    assert len(caplog.records) == 1
    assert "worker_batch_not_below_threshold_or_unavailable" in caplog.text


def test_failed_batch_does_not_contaminate_the_next_batch(scheduler, caplog):
    state = scheduler.update_from_output.__self__
    original_update = state.original_update

    def fail_after_observation(scheduler_output, model_runner_output):
        scheduler.make_spec_decoding_stats(None, 3, 1, None, "failed")
        raise RuntimeError("output consumer failed")

    state.original_update = fail_after_observation
    with pytest.raises(RuntimeError, match="output consumer failed"):
        scheduler.update_from_output([], None)
    state.original_update = original_update
    scheduler.update_from_output([("next", 3, 0, None)], diagnostic_output("next"))
    assert len(caplog.records) == 1
    assert json.loads(caplog.messages[0].split("request_ids=", 1)[1]) == ["next"]


@pytest.mark.parametrize(
    "enabled,speculative,kv_consumer",
    [
        ("0", True, None),
        ("1", True, None),
        (None, True, None),
        ("1", False, None),
        ("1", True, False),
        ("1", True, True),
    ],
)
def test_hook_only_installs_when_enabled(
    monkeypatch, caplog, enabled, speculative, kv_consumer
):
    if enabled is None:
        monkeypatch.delenv("TPU_MTP_ZERO_ACCEPTANCE_LOG", raising=False)
    else:
        monkeypatch.setenv("TPU_MTP_ZERO_ACCEPTANCE_LOG", enabled)
    monkeypatch.setattr(hook, "logger", logging.getLogger("mtp_hook_test"))
    caplog.set_level(logging.WARNING, logger="mtp_hook_test")

    def init(self, config):
        self.log_stats = True
        self.num_spec_tokens = 3
        self.parallel_config = SimpleNamespace(data_parallel_index=0)

    def consume(self, scheduler_output, model_runner_output):
        return self.make_spec_decoding_stats(None, 3, 0, None, "enabled-request")

    monkeypatch.setattr(Scheduler, "__init__", init)
    monkeypatch.setattr(Scheduler, "update_from_output", consume)
    hook.patch_scheduler_init()
    config = SimpleNamespace(
        speculative_config=object() if speculative else None,
        kv_transfer_config=(
            None if kv_consumer is None else SimpleNamespace(is_kv_consumer=kv_consumer)
        ),
    )
    scheduler = Scheduler(config)
    stats = scheduler.update_from_output(None, diagnostic_output("enabled-request"))
    assert stats.num_draft_tokens == 3
    assert stats.num_accepted_tokens == 0
    assert len(caplog.records) == int(
        enabled != "0" and speculative and kv_consumer is not False
    )
    assert (Scheduler.__init__ is not init) == (enabled != "0")


@pytest.mark.parametrize("log_stats", [True, False])
@pytest.mark.parametrize(
    "accepted,invalid_last,should_log",
    [(0, 0, True), (1, 0, True), (1, 1, True), (1, 2, False), (1, 3, False)],
)
def test_batch_dump_threshold_uses_effective_drafts(
    scheduler, caplog, log_stats, accepted, invalid_last, should_log
):
    scheduler.log_stats = log_stats
    rows = [
        ("a", 3, accepted, None),
        ("b", 3, 0, None),
        ("c", 3, 0, None),
        ("d", 3, 0, {"d": invalid_last}),
    ]
    output = diagnostic_output("a", "b", "c", "d")
    result = scheduler.update_from_output(rows, output)
    assert result is scheduler.result
    assert len(caplog.records) == int(should_log)
    if should_log:
        message = caplog.messages[0]
        assert f"num_draft_tokens={12 - invalid_last} " in message
        assert f"num_accepted_tokens={accepted} " in message
        rate = float(message.split("acceptance_rate=", 1)[1].split()[0])
        assert rate == pytest.approx(accepted / (12 - invalid_last))
        drafts = message.split("verified_draft_token_ids=", 1)[1].split(
            " request_ids=", 1
        )[0]
        assert json.loads(drafts) == output.verified_draft_token_ids
        assert json.loads(message.split("request_ids=", 1)[1]) == ["a", "b", "c", "d"]
    if log_stats:
        assert scheduler.stats.num_draft_tokens == 12 - invalid_last
        assert scheduler.stats.num_accepted_tokens == accepted
    else:
        assert scheduler.stats is None
