# SPDX-License-Identifier: Apache-2.0
"""Decode-side Raiden KV load error output contracts."""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason

from vllm_torchtpu.core.raiden_pd import _DecodeLoadFailures


@pytest.fixture(params=[Scheduler, AsyncScheduler])
def scheduler_cls(request):
    return request.param


def test_only_failed_kv_load_request_gets_structured_error(scheduler_cls):
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.recompute_kv_load_failures = False
    outputs = {
        0: EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id="kv-load-failed",
                    new_token_ids=[],
                    finish_reason=FinishReason.ERROR,
                ),
                EngineCoreOutput(
                    request_id="unrelated-error",
                    new_token_ids=[],
                    finish_reason=FinishReason.ERROR,
                ),
            ]
        )
    }

    def base_update(*_args, **_kwargs):
        scheduler._handle_invalid_blocks({11}, {})
        return outputs

    scheduler._handle_invalid_blocks = lambda *_args: {"kv-load-failed"}
    scheduler.update_from_output = base_update
    state = _DecodeLoadFailures(scheduler)
    actual = scheduler.update_from_output(
        None, SimpleNamespace(kv_connector_output=None)
    )

    kv_error, unrelated = actual[0].outputs
    assert kv_error.kv_transfer_params == {
        "kv_transfer_error": {
            "status_code": 500,
            "status_name": "model.KVTDecodeLoadFailure",
            "status_message": (
                "D-side failed to load KV cache data. Category: internal."
            ),
        }
    }
    assert unrelated.kv_transfer_params is None
    assert state.failed_req_ids == set()


def _lifecycle_scheduler(scheduler_cls):
    """Keep scheduling/finish/free code real; replace only cache/device services."""
    from unittest.mock import MagicMock

    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    from vllm.v1.request import Request, RequestStatus

    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.recompute_kv_load_failures = False
    scheduler.requests = {}
    for req_id in ("bad", "good"):
        req = Request(req_id, [1, 2, 3], SamplingParams(max_tokens=8), None)
        req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        scheduler.requests[req_id] = req
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue()
    scheduler.skipped_waiting = FCFSRequestQueue(scheduler.requests.values())
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._inflight_prefills = set()
    scheduler.grammar_compile_error_reqs = []
    scheduler.defer_block_free = False
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.encoder_cache_manager = MagicMock()
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.perf_metrics = None
    scheduler.make_stats = lambda *args: None
    _DecodeLoadFailures(scheduler)
    return scheduler


def _lifecycle_step(
    scheduler, failures=(), finished=(), *, wrapped=False, scheduled=None
):
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorWorkerMetadata,
    )
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

    from vllm_torchtpu.distributed.kv_transfer.raiden_pd_completion import (
        RaidenLoadFailures,
    )

    metadata = RaidenLoadFailures(set(failures)) if failures else None
    if wrapped:
        metadata = MultiKVConnectorWorkerMetadata([None, metadata])
    output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=KVConnectorOutput(
            finished_recving=set(finished), kv_connector_worker_meta=metadata
        ),
    )
    return scheduler.update_from_output(
        SimpleNamespace(
            num_scheduled_tokens=scheduled or {},
            total_num_scheduled_tokens=sum((scheduled or {}).values()),
            scheduled_spec_decode_tokens={},
        ),
        output,
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_load_error_finishes_request_but_waits_for_receive_barrier(
    scheduler_cls, wrapped
):
    from vllm.v1.request import RequestStatus

    scheduler = _lifecycle_scheduler(scheduler_cls)
    bad = scheduler.requests["bad"]
    result = _lifecycle_step(scheduler, ["bad"], wrapped=wrapped)
    (terminal,) = result[bad.client_index].outputs
    assert terminal.new_token_ids == []
    assert terminal.finish_reason is FinishReason.ERROR
    assert (
        terminal.kv_transfer_params["kv_transfer_error"]["status_name"]
        == "model.KVTDecodeLoadFailure"
    )
    assert bad.status == RequestStatus.FINISHED_ERROR
    assert set(scheduler.requests) == {"bad", "good"}
    scheduler.kv_cache_manager.free.assert_not_called()
    assert [r.request_id for r in scheduler.skipped_waiting] == ["good"]

    # Another rank may repeat a failure before all receive completions arrive.
    assert _lifecycle_step(scheduler, ["bad"]) == {}
    assert _lifecycle_step(scheduler, finished=["bad", "good"]) == {}
    scheduler.kv_cache_manager.free.assert_called_once_with(bad)
    assert set(scheduler.requests) == {"good"}
    assert scheduler.finished_recving_kv_req_ids == {"good"}
    assert scheduler.requests["good"].status == RequestStatus.WAITING_FOR_REMOTE_KVS
    # Later metadata cannot resurrect a completed request or emit another error.
    assert _lifecycle_step(scheduler, ["bad"]) == {}


def test_cancelled_load_failure_only_releases_buffers(scheduler_cls):
    from vllm.v1.request import RequestStatus

    scheduler = _lifecycle_scheduler(scheduler_cls)
    bad = scheduler.requests["bad"]
    scheduler.finish_requests("bad", RequestStatus.FINISHED_ABORTED)
    scheduler.kv_cache_manager.free.assert_not_called()
    assert _lifecycle_step(scheduler, ["bad"], ["bad"]) == {}
    scheduler.kv_cache_manager.free.assert_called_once_with(bad)
    assert bad.status == RequestStatus.FINISHED_ABORTED


def test_failed_load_discards_inflight_tokens(scheduler_cls):
    scheduler = _lifecycle_scheduler(scheduler_cls)
    bad = scheduler.requests["bad"]
    bad.num_in_flight_tokens = 1
    result = _lifecycle_step(scheduler, ["bad"], ["bad"], scheduled={"bad": 1})
    (terminal,) = result[bad.client_index].outputs
    assert terminal.new_token_ids == []
    assert list(bad.output_token_ids) == []
    assert bad.num_in_flight_tokens == 0
    scheduler.kv_cache_manager.cache_blocks.assert_not_called()
