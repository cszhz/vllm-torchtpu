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
"""Opt-in pre-accounting selection for D-planned remote Draft execution."""

from collections.abc import Iterable
from dataclasses import fields
from typing import Any

from vllm.config import SchedulerConfig
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.spec_decode.remote_draft_control import (
    GENERATION_KEY,
    RemoteDraftConsumerOutput,
    RemoteDraftPlan,
    RemoteDraftPlanReceiver,
    RemoteDraftSchedulerOutput,
    RemoteDraftTerminal,
)
from vllm_torchtpu.spec_decode.remote_dspark import (
    RemoteDraftProtocolError,
    is_remote_dspark_request,
)


def remote_draft_scheduler_role(config: SchedulerConfig) -> str | None:
    scheduler_cls = config.get_scheduler_cls()
    if issubclass(scheduler_cls, RemoteDraftConsumerScheduler):
        return "consumer"
    if issubclass(scheduler_cls, RemoteDraftScheduler):
        return "producer"
    return None


class _RemoteDraftSchedulerBase(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        parallel = self.vllm_config.parallel_config
        if (
            self.scheduler_config.async_scheduling
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.nnodes != 1
        ):
            raise NotImplementedError("planned remote Draft requires sync TP-only 1P1D")

    def set_pause_state(self, pause_state: PauseState) -> None:
        if pause_state != PauseState.UNPAUSED:
            raise NotImplementedError(
                "planned remote Draft pause requires coordinated P/D draining"
            )
        super().set_pause_state(pause_state)

    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False
    ) -> None:
        if is_remote_dspark_request(request.kv_transfer_params):
            # TODO: coordinated P/D reclamation must precede remote preemption.
            raise NotImplementedError(
                "remote Draft preemption must not replay Target prefill"
            )
        super()._preempt_request(request, timestamp, drop_stale_output)


class RemoteDraftConsumerScheduler(_RemoteDraftSchedulerBase):
    """Explicit consumer opt-in paired with RemoteDraftScheduler on P."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (
            self.vllm_config.kv_transfer_config is not None
            and self.recompute_kv_load_failures
        ):
            raise NotImplementedError(
                "planned remote Draft consumer requires "
                "kv_load_failure_policy='fail' to prevent Target prefill replay"
            )
        self._remote_terminals: list[RemoteDraftTerminal] = []
        self._mamba_alignment = self.need_mamba_block_aligned_split
        self.need_mamba_block_aligned_split = True

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        if is_remote_dspark_request(request.kv_transfer_params) and (
            request.num_output_tokens == 0
        ):
            position = (
                request.num_computed_tokens
                + num_new_local_computed_tokens
                + num_external_computed_tokens
            )
            if position != request.num_prompt_tokens - 1:
                raise RemoteDraftProtocolError(
                    "D bootstrap requires transferred KV through the penultimate "
                    "prompt token; refusing Target prefill replay"
                )
            # Pinned upstream drops initial speculative padding when this hook
            # clips its span, before allocating slots or charging counters.
            return 1
        if self._mamba_alignment:
            return super()._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        return num_new_tokens

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        params = request.kv_transfer_params or {}
        if is_remote_dspark_request(params):
            self._remote_terminals.append(
                RemoteDraftTerminal(
                    params["req_id"], params["uuid"], int(request.status)
                )
            )
        return super()._free_request(request, delay_free_blocks)

    def schedule(self, throttle_prefills: bool = False) -> RemoteDraftConsumerOutput:
        if self.scheduler_config.async_scheduling:
            raise NotImplementedError("remote Draft scheduling requires sync execution")
        output = super().schedule(throttle_prefills)
        terminals, self._remote_terminals = tuple(self._remote_terminals), []
        return RemoteDraftConsumerOutput(
            **{field.name: getattr(output, field.name) for field in fields(output)},
            remote_draft_terminals=terminals,
        )


class RemoteDraftScheduler(_RemoteDraftSchedulerBase):
    def __init__(
        self, *args, plan_receiver: RemoteDraftPlanReceiver | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        world_size = self.vllm_config.parallel_config.world_size
        self._receiver = plan_receiver or RemoteDraftPlanReceiver(
            f"tcp://*:{envs.VLLM_TPU_REMOTE_DSPARK_PORT + world_size}", world_size
        )
        self._plan: RemoteDraftPlan | None = None
        self._terminal_outputs: list[tuple[int, EngineCoreOutput]] = []
        self._mamba_alignment = self.need_mamba_block_aligned_split
        self.need_mamba_block_aligned_split = True

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        if request_ids is None:
            ids = tuple(self.requests)
        elif isinstance(request_ids, str):
            ids = (request_ids,)
        else:
            ids = tuple(request_ids)
        if any(
            (request := self.requests.get(request_id)) is not None
            and not request.is_finished()
            and is_remote_dspark_request(request.kv_transfer_params)
            for request_id in ids
        ):
            # TODO: route P cancellation through D before releasing remote KV.
            raise NotImplementedError(
                "cancel remote requests through D; direct P termination requires "
                "coordinated P/D draining"
            )
        return super().finish_requests(ids, finished_status)

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        if not is_remote_dspark_request(request.kv_transfer_params):
            return super()._update_request_with_output(request, new_token_ids, is_stale)
        if is_stale or request.num_tokens + len(new_token_ids) > self.max_model_len:
            raise RemoteDraftProtocolError(
                "P remote token accounting is stale or exceeds model context capacity"
            )
        # P's prefix-boundary sample is private. D owns EOS/length/stop and
        # sends an authoritative terminal after receiving the final proposal.
        request.append_output_token_ids(new_token_ids)
        return new_token_ids, False

    def schedule(self, throttle_prefills: bool = False) -> RemoteDraftSchedulerOutput:
        ready = self._receiver.poll()
        if ready is not None and ready.terminals:
            for terminal in ready.terminals:
                request = self.requests.get(terminal.request_id)
                if (
                    request is None
                    or (request.kv_transfer_params or {}).get(GENERATION_KEY)
                    != terminal.generation
                ):
                    raise RemoteDraftProtocolError(
                        "terminal has a stale request generation"
                    )
                if request.num_in_flight_tokens:
                    raise RemoteDraftProtocolError(
                        "terminal arrived before P execution completed"
                    )
            for terminal in ready.terminals:
                finished = super().finish_requests(
                    [terminal.request_id], RequestStatus(terminal.status)
                )
                for request in finished:
                    # The external-finish hook frees KV but emits no client
                    # completion: unlike a client abort, D's stop must close P's
                    # still-live HTTP request through the normal output channel.
                    self._terminal_outputs.append(
                        (
                            request.client_index,
                            EngineCoreOutput(
                                request_id=request.request_id,
                                new_token_ids=[],
                                finish_reason=request.get_finished_reason(),
                                stop_reason=request.stop_reason,
                                events=request.take_events(),
                                trace_headers=request.trace_headers,
                            ),
                        )
                    )
            if not ready.queries:
                self._receiver.consume(ready)
                ready = None
        has_prefill = bool(self.waiting) or any(
            request.num_computed_tokens < request.num_prompt_tokens
            for request in self.running
        )
        self._plan = ready if not (has_prefill and self.current_step % 2 == 1) else None
        if self._plan is not None:
            if sum(query.num_tokens for query in self._plan.queries) > (
                self.max_num_scheduled_tokens
            ):
                raise RemoteDraftProtocolError(
                    "D execution plan exceeds P token budget"
                )
            by_id = {request.request_id: request for request in self.running}
            selected = []
            for query in self._plan.queries:
                request = by_id.get(query.request_id)
                if request is None or not is_remote_dspark_request(
                    request.kv_transfer_params
                ):
                    raise RemoteDraftProtocolError(
                        "D plan names a non-live remote request"
                    )
                assert request.kv_transfer_params is not None
                if request.kv_transfer_params.get(GENERATION_KEY) != query.generation:
                    raise RemoteDraftProtocolError(
                        "D plan has a stale request generation"
                    )
                if (
                    request.num_in_flight_tokens
                    or request.num_computed_tokens != query.position
                    or request.num_computed_tokens < request.num_prompt_tokens
                ):
                    raise RemoteDraftProtocolError(
                        "D query span differs from P bookkeeping"
                    )
                selected.append(request)
        else:
            selected = [
                request
                for request in self.running
                if not (
                    is_remote_dspark_request(request.kv_transfer_params)
                    and request.num_computed_tokens >= request.num_prompt_tokens
                )
            ]
        selected_ids = {request.request_id for request in selected}
        held = [
            request
            for request in self.running
            if request.request_id not in selected_ids
        ]
        previous_pause = self._pause_state
        self.running[:] = selected
        # Like SrptPrefillScheduler, hold rows without releasing their KV or
        # admitting replacements beyond either upstream capacity limit.
        self.max_num_running_reqs -= len(held)
        self.max_num_active_reqs -= len(held)
        if self._plan is not None and previous_pause == PauseState.UNPAUSED:
            self._pause_state = PauseState.PAUSED_NEW
        try:
            output = super().schedule(throttle_prefills)
            plan = self._plan
            if plan is not None:
                self._receiver.consume(plan)
            return RemoteDraftSchedulerOutput(
                **{field.name: getattr(output, field.name) for field in fields(output)},
                remote_draft_plan=plan,
            )
        finally:
            self._pause_state = previous_pause
            self.max_num_running_reqs += len(held)
            self.max_num_active_reqs += len(held)
            self.running.extend(held)
            self._plan = None

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        for client_index, terminal in self._terminal_outputs:
            if client_index not in outputs:
                outputs[client_index] = EngineCoreOutputs()
            outputs[client_index].outputs.append(terminal)
        self._terminal_outputs.clear()
        return outputs

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        if self._plan is not None:
            query = next(
                query
                for query in self._plan.queries
                if query.request_id == request.request_id
            )
            if num_new_tokens < query.num_tokens:
                raise RemoteDraftProtocolError(
                    "P cannot schedule D's complete query span"
                )
            return query.num_tokens
        if self._mamba_alignment:
            return super()._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        return num_new_tokens

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        if self._plan is not None:
            expected = {
                query.request_id: query.num_tokens for query in self._plan.queries
            }
            if list(scheduler_output.num_scheduled_tokens.items()) != list(
                expected.items()
            ):
                raise RemoteDraftProtocolError(
                    "P scheduled a partial or reordered D plan"
                )
        super()._update_after_schedule(scheduler_output)

    def shutdown(self) -> None:
        self._receiver.close()
        super().shutdown()
