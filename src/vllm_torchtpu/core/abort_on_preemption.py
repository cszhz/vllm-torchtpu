# SPDX-License-Identifier: Apache-2.0
"""Capacity preemption errors for vLLM 0.29 KV consumers using fail policy.

Only enabled scheduler instances use the adapted allocation loop. The init
patch leaves the upstream scheduler class, configuration dataclasses and
administrative ``_preempt_request`` path intact.
"""

import inspect
import linecache
import textwrap
from functools import lru_cache, wraps
from types import MethodType

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import (
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
)
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def _capacity_schedule(original):
    """Adapt only the capacity branch, retaining upstream scheduling accounts.

    vLLM has no hook before removing a preemption victim. Apply these exact
    source hunks to the pinned upstream method instead of maintaining a copy
    of its entire scheduling algorithm. A changed upstream context is an
    unsupported version, not a reason to silently lose the in-flight fence.
    Compilation is lazy: disabled instances never need this adaptation.
    """
    source = textwrap.dedent(inspect.getsource(original))
    replacements = (
        (
            '        with record_function_or_nullcontext("schedule: allocate_slots"):',
            "        preemption_deferred = False\n"
            '        with record_function_or_nullcontext("schedule: allocate_slots"):',
        ),
        (
            "                    victim_index = self.running.index(preempted_req)",
            "                    if self._tpu_capacity_preemption.in_flight("
            "preempted_req):\n"
            "                        preemption_deferred = True\n"
            "                        break\n"
            "                    victim_index = self.running.index(preempted_req)",
        ),
        (
            "                    preempted_req = self.running.pop()",
            "                    preempted_req = self.running[-1]\n"
            "                    if self._tpu_capacity_preemption.in_flight("
            "preempted_req):\n"
            "                        preemption_deferred = True\n"
            "                        break\n"
            "                    preempted_req = self.running.pop()",
        ),
        (
            "                self._preempt_request(\n"
            "                    preempted_req,\n"
            "                    scheduled_timestamp,\n"
            "                    drop_stale_output=self.requires_kv_delivery,\n"
            "                )",
            "                self._tpu_capacity_preemption.abort(\n"
            "                    preempted_req, scheduled_timestamp\n"
            "                )",
        ),
        (
            "        if new_blocks is None:\n"
            "            # Cannot schedule this request.\n"
            "            break",
            "        if new_blocks is None:\n"
            "            # Try other requests while the victim is in flight.\n"
            "            if preemption_deferred:\n"
            "                req_index += 1\n"
            "                continue\n"
            "            break",
        ),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError(
                "KV consumer fail policy requires the vLLM 0.29 capacity loop; "
                f"expected one patch context: {before!r}"
            )
        source = source.replace(before, after, 1)
    source = "from __future__ import annotations\n" + source
    filename = f"<{__name__}.schedule>"
    namespace = original.__globals__.copy()
    exec(compile(source, filename, "exec"), namespace)
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    return namespace[original.__name__]


class _CapacityPreemption:
    """Own the terminal snapshots and preserve the scheduler's output chain."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.original_update = scheduler.update_from_output
        self.pending: dict[str, tuple[int, EngineCoreOutput]] = {}

    def in_flight(self, request: Request) -> bool:
        scheduler = self.scheduler
        return (
            scheduler.defer_block_free
            and request.last_sched_seq > scheduler.processed_step_seq
        )

    def abort(self, request: Request, timestamp: float) -> None:
        scheduler = self.scheduler
        assert request.status == RequestStatus.RUNNING, (
            "Capacity preemption requires a RUNNING victim: "
            f"request_id={request.request_id}, status={request.status}"
        )
        request.num_preemptions += 1
        if scheduler.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
        request.stop_reason = "kv_cache_preempted"
        self.pending[request.request_id] = (
            request.client_index,
            EngineCoreOutput(
                request_id=request.request_id,
                new_token_ids=[],
                finish_reason=FinishReason.ERROR,
                stop_reason=request.stop_reason,
                events=request.take_events(),
                trace_headers=request.trace_headers,
                prefill_stats=request.take_prefill_stats(),
                kv_transfer_params={
                    "kv_transfer_error": {
                        "status_code": 500,
                        "status_name": "model.KVCachePreempt",
                        "status_message": (
                            "D-side request was preempted due to KV cache capacity."
                        ),
                    }
                },
            ),
        )
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ERROR)
        scheduler.reset_preempted_req_ids.add(request.request_id)
        logger.warning(
            "Aborted capacity-preempted request_id=%s error=model.KVCachePreempt",
            request.request_id,
        )

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = self.original_update(scheduler_output, model_runner_output)
        # An older async output must not consume a later step's terminal.
        # finished_req_ids is also delivered for a zero-token cleanup step.
        for req_id in scheduler_output.finished_req_ids:
            pending = self.pending.pop(req_id, None)
            if pending is not None:
                client_index, terminal = pending
                batch = outputs.setdefault(client_index, EngineCoreOutputs())
                batch.outputs.append(terminal)
        return outputs


def patch_scheduler_init() -> None:
    if Scheduler.__dict__.get("_tpu_abort_on_preemption_patch", False):
        return
    original_init = Scheduler.__init__
    original_schedule = inspect.unwrap(Scheduler.schedule)

    @wraps(original_init)
    def init(self, vllm_config, *args, **kwargs):
        kv = vllm_config.kv_transfer_config
        enabled = (
            kv is not None
            and kv.kv_role == "kv_consumer"
            and kv.kv_load_failure_policy == "fail"
        )
        if enabled:
            class_schedule = type(self).schedule
            if class_schedule is original_schedule:
                schedule = _capacity_schedule(original_schedule)
            elif class_schedule.__dict__.get("_tpu_same_step_prefix_hit_patch", False):
                schedule = class_schedule._tpu_wrap_same_step_prefix_hits(
                    _capacity_schedule(original_schedule)
                )
            else:
                raise ValueError(
                    "KV consumer fail policy requires the upstream Scheduler.schedule; "
                    f"{type(self).__name__} overrides it"
                )
        original_init(self, vllm_config, *args, **kwargs)
        if enabled:
            state = _CapacityPreemption(self)
            self._tpu_capacity_preemption = state
            self.schedule = MethodType(schedule, self)
            self.update_from_output = state.update_from_output
            logger.info(
                "Enabled capacity preemption abort for KV consumer "
                "with kv_load_failure_policy=fail"
            )

    Scheduler.__init__ = init
    Scheduler._tpu_abort_on_preemption_patch = True
