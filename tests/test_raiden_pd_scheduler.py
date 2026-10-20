# SPDX-License-Identifier: Apache-2.0
"""P response lifetime and send-result contracts; no TPU execution required."""

from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorWorkerMetadata,
)
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from vllm_torchtpu.core.raiden_pd import _ProducerCompletion
from vllm_torchtpu.distributed.kv_transfer.raiden_pd_completion import (
    RaidenSendOutcome,
    RaidenSendResults,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPURaidenConnector,
    TPURaidenConnectorScheduler,
)


@pytest.fixture(params=[Scheduler, AsyncScheduler])
def scheduler_cls(request):
    return request.param


def _scheduler(scheduler_cls, *, multi_connector=False):
    scheduler = scheduler_cls.__new__(scheduler_cls)
    connector = TPURaidenConnector.__new__(TPURaidenConnector)
    connector.connector_scheduler = TPURaidenConnectorScheduler.__new__(
        TPURaidenConnectorScheduler
    )
    connector.connector_scheduler._stage3_finished_sends = {}
    connector.connector_scheduler._stage3_sends_to_expire = {}
    connector.connector_scheduler.reqs_to_release = {}
    scheduler.connector = connector
    if multi_connector:
        scheduler.connector = MultiConnector.__new__(MultiConnector)
        scheduler.connector._connectors = [object(), connector]
    scheduler._next_outputs = {}
    scheduler.update_from_output = lambda *_args: scheduler._next_outputs
    scheduler.finish_requests = lambda *_args: []
    return scheduler, _ProducerCompletion(scheduler)


def _step(scheduler, outputs=None, results=None, finished=()):
    kv_output = KVConnectorOutput(
        finished_sending=set(finished),
        kv_connector_worker_meta=MultiKVConnectorWorkerMetadata(
            metadata=(results, None)
        ),
    )
    runner_output = SimpleNamespace(kv_connector_output=kv_output)
    scheduler._next_outputs = outputs if outputs is not None else {}
    return scheduler.update_from_output(None, runner_output)


def _prefill(scheduler):
    stats = PrefillStats(num_cached_tokens=16384)
    output = EngineCoreOutput(
        request_id="p-request",
        new_token_ids=[42],
        finish_reason=FinishReason.LENGTH,
        prefill_stats=stats,
        kv_transfer_params={
            "req_id": "p-request",
            "uuid": 7,
            "kv_transfer_pending": True,
        },
    )
    outputs = _step(scheduler, {3: EngineCoreOutputs(outputs=[output])})
    return outputs[3].outputs[0]


def test_prefill_remains_open_until_success_without_replaying_tokens(scheduler_cls):
    scheduler, state = _scheduler(scheduler_cls)
    first = _prefill(scheduler)
    assert first.finish_reason is None
    assert first.new_token_ids == [42]
    assert first.prefill_stats.num_cached_tokens == 16384
    assert first.kv_transfer_params["uuid"] == 7

    # A subset of ranks finishes first. The existing aggregator, not this
    # scheduler, decides when the whole request reaches finished_sending.
    assert (
        _step(
            scheduler,
            results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
        )
        == {}
    )
    final = _step(
        scheduler,
        results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
        finished=["p-request"],
    )[3].outputs
    assert len(final) == 1
    assert final[0].new_token_ids == []
    assert final[0].finish_reason == FinishReason.STOP
    assert final[0].kv_transfer_params == {"kv_transfer_done": True}
    assert state.pending_responses == {}
    # A repeated/late terminal must not finish the response twice.
    assert _step(scheduler, finished=["p-request"]) == {}


@pytest.mark.parametrize(
    ("outcome", "status_name", "status_message"),
    [
        (
            RaidenSendOutcome.SEND_FAILURE,
            "model.KVTPrefillWriteFailure",
            "P-side failed to write KV cache data. Category: internal.",
        ),
        (
            RaidenSendOutcome.UNCLAIMED_TIMEOUT,
            "model.KVTPrefillTransferTimeout",
            "P-side timed out waiting for KV cache transfer completion.",
        ),
        (
            None,
            "model.KVTPrefillWriteFailure",
            "P-side KV cache transfer finished without a sender result. "
            "Category: internal.",
        ),
    ],
)
def test_failed_or_missing_result_has_structured_error(
    outcome, status_name, status_message, scheduler_cls
):
    scheduler, _ = _scheduler(scheduler_cls)
    _prefill(scheduler)
    if outcome is not None:
        # Failure must survive successful ranks arriving in later steps.
        _step(scheduler, results=RaidenSendResults({"p-request": outcome}))
        _step(
            scheduler,
            results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
        )
    final = _step(scheduler, finished=["p-request"])[3].outputs[0]
    assert final.finish_reason == FinishReason.ERROR
    assert final.kv_transfer_params == {
        "kv_transfer_error": {
            "status_code": 500,
            "status_name": status_name,
            "status_message": status_message,
        }
    }


def test_rank_result_aggregation_preserves_failure_in_both_orders():
    for values in [
        (RaidenSendOutcome.SUCCESS, RaidenSendOutcome.SEND_FAILURE),
        (RaidenSendOutcome.SEND_FAILURE, RaidenSendOutcome.SUCCESS),
    ]:
        result = MultiKVConnectorWorkerMetadata(
            metadata=(RaidenSendResults({"p-request": values[0]}), None)
        )
        result = result.aggregate(
            MultiKVConnectorWorkerMetadata(
                metadata=(RaidenSendResults({"p-request": values[1]}), None)
            )
        )
        assert result.metadata[0].outcomes == {
            "p-request": RaidenSendOutcome.SEND_FAILURE
        }


def test_rank_result_aggregation_prefers_failure_over_timeout():
    result = RaidenSendResults({"p-request": RaidenSendOutcome.UNCLAIMED_TIMEOUT})
    result.aggregate(RaidenSendResults({"p-request": RaidenSendOutcome.SEND_FAILURE}))
    assert result.outcomes == {"p-request": RaidenSendOutcome.SEND_FAILURE}


def test_cancelled_response_does_not_emit_late_pkg2(scheduler_cls):
    scheduler, state = _scheduler(scheduler_cls)
    _prefill(scheduler)
    scheduler.finish_requests("p-request", RequestStatus.FINISHED_ABORTED)
    assert (
        _step(
            scheduler,
            results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
            finished=["p-request"],
        )
        == {}
    )
    assert state.pending_responses == {}


@pytest.mark.parametrize("request_ids", [None, "p-request", ["p-request", "missing"]])
@pytest.mark.parametrize("multi_connector", [False, True])
def test_abort_reaches_model_finished_send(scheduler_cls, request_ids, multi_connector):
    scheduler, state = _scheduler(scheduler_cls, multi_connector=multi_connector)
    _prefill(scheduler)
    scheduler.requests = {"p-request": SimpleNamespace(is_finished=lambda: True)}
    state.original_finish = Scheduler.finish_requests.__get__(scheduler)
    connector = (
        scheduler.connector._connectors[-1] if multi_connector else scheduler.connector
    )
    connector_scheduler = connector.connector_scheduler
    connector_scheduler._stage3_finished_sends["p-request"] = (7, (), 1, {})
    connector_scheduler.is_producer = True
    connector_scheduler.reqs_to_send = {}

    assert scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED) == []
    metadata = connector.build_connector_meta(None)
    assert metadata.reqs_to_expire == {"p-request": 7}
    assert connector.build_connector_meta(None).reqs_to_expire == {}
    assert state.pending_responses == {}
    assert "p-request" in scheduler.requests


def test_abort_keeps_model_running_registration_on_upstream_path(scheduler_cls):
    scheduler, state = _scheduler(scheduler_cls)
    connector_scheduler = scheduler.connector.connector_scheduler
    connector_scheduler._stage3_finished_sends["running"] = (8, (), 1, {})
    original_calls = []
    state.original_finish = (
        lambda ids, status: original_calls.append((set(ids), status)) or []
    )

    scheduler.finish_requests(["running"], RequestStatus.FINISHED_ABORTED)

    assert connector_scheduler._stage3_sends_to_expire == {}
    assert original_calls == [({"running"}, RequestStatus.FINISHED_ABORTED)]


def test_prefill_and_send_completion_in_same_step(scheduler_cls):
    scheduler, state = _scheduler(scheduler_cls)
    first = EngineCoreOutput(
        request_id="p-request",
        new_token_ids=[42],
        finish_reason=FinishReason.LENGTH,
        kv_transfer_params={"kv_transfer_pending": True},
    )
    outputs = _step(
        scheduler,
        {3: EngineCoreOutputs(outputs=[first])},
        results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
        finished=["p-request"],
    )[3].outputs
    assert len(outputs) == 2
    assert outputs[0].new_token_ids == [42]
    assert outputs[0].finish_reason is None
    assert outputs[1].new_token_ids == []
    assert outputs[1].finish_reason == FinishReason.STOP
    assert outputs[1].kv_transfer_params == {"kv_transfer_done": True}
    assert state.pending_responses == {}


def test_output_processor_keeps_cached_stats_until_pkg2(scheduler_cls):
    # Regression: finishing pkg1 deletes OutputProcessor's state, so a later
    # kv_transfer_done is silently dropped. Exercise the real processor.
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    processor.add_request(
        EngineCoreRequest(
            request_id="p-request",
            external_req_id="client-request",
            prompt_token_ids=[1] * 20000,
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=1, detokenize=False, output_kind=RequestOutputKind.DELTA
            ),
            pooling_params=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
        prompt=None,
    )
    scheduler, _ = _scheduler(scheduler_cls)
    first = processor.process_outputs(
        [_prefill(scheduler)], engine_core_timestamp=0
    ).request_outputs[0]
    assert not first.finished
    assert first.outputs[0].token_ids == [42]
    assert first.num_cached_tokens == 16384
    assert processor.has_unfinished_requests()
    final_outputs = _step(
        scheduler,
        results=RaidenSendResults({"p-request": RaidenSendOutcome.SUCCESS}),
        finished=["p-request"],
    )[3].outputs
    final = processor.process_outputs(
        final_outputs, engine_core_timestamp=0
    ).request_outputs[0]
    assert final.request_id == "client-request"
    assert final.finished
    assert final.outputs[0].token_ids == []
    assert final.num_cached_tokens == 16384
    assert final.kv_transfer_params == {"kv_transfer_done": True}
    assert not processor.has_unfinished_requests()


def test_output_processor_keeps_structured_error_after_prefill_output(scheduler_cls):
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    processor.add_request(
        EngineCoreRequest(
            request_id="p-request",
            external_req_id="client-request",
            prompt_token_ids=[1] * 20000,
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=1, detokenize=False, output_kind=RequestOutputKind.DELTA
            ),
            pooling_params=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
        prompt=None,
    )
    scheduler, _ = _scheduler(scheduler_cls)
    first = processor.process_outputs(
        [_prefill(scheduler)], engine_core_timestamp=0
    ).request_outputs[0]
    assert not first.finished

    terminal_outputs = _step(
        scheduler,
        results=RaidenSendResults({"p-request": RaidenSendOutcome.SEND_FAILURE}),
        finished=["p-request"],
    )[3].outputs
    terminal = processor.process_outputs(
        terminal_outputs, engine_core_timestamp=0
    ).request_outputs[0]

    assert terminal.finished
    assert terminal.outputs[0].token_ids == []
    assert terminal.kv_transfer_params == {
        "kv_transfer_error": {
            "status_code": 500,
            "status_name": "model.KVTPrefillWriteFailure",
            "status_message": (
                "P-side failed to write KV cache data. Category: internal."
            ),
        }
    }
    assert not processor.has_unfinished_requests()
