# SPDX-License-Identifier: Apache-2.0
"""Batch zero-acceptance logging contracts; no model or TPU execution."""

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
    hook._BatchZeroAcceptance(scheduler)
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
    result = scheduler.update_from_output(rows, output)

    assert result is scheduler.result
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "MTP_ZERO_ACCEPTANCE" in message
    assert "scope=batch" in message
    assert "dp_index=2" in message
    assert "num_draft_tokens=4" in message
    assert "num_accepted_tokens=0" in message
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
def test_nonzero_or_empty_batch_does_not_log(scheduler, caplog, rows):
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
        (False, True, None),
        (True, True, None),
        (True, False, None),
        (True, True, False),
        (True, True, True),
    ],
)
def test_hook_only_installs_when_enabled(
    monkeypatch, caplog, enabled, speculative, kv_consumer
):
    monkeypatch.setenv("TPU_MTP_ZERO_ACCEPTANCE_LOG", str(int(enabled)))
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
        enabled and speculative and kv_consumer is not False
    )
    if not enabled:
        assert Scheduler.__init__ is init
