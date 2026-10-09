# SPDX-License-Identifier: Apache-2.0
"""Temporary attribution for batches with draft acceptance below 10%."""

from __future__ import annotations

import json
from functools import wraps
from typing import TYPE_CHECKING

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.spec_decode.metrics import SpecDecodingStats

logger = init_logger(__name__)


def is_low_mtp_acceptance(drafted: int, accepted: int) -> bool:
    """Use effective draft counts; exactly 10% is outside the trigger."""
    return drafted > 0 and accepted * 10 < drafted


class _BatchLowAcceptance:
    def __init__(self, scheduler: Scheduler) -> None:
        self.original_update = scheduler.update_from_output
        self.original_stats = scheduler.make_spec_decoding_stats
        self.dp_index = scheduler.parallel_config.data_parallel_index
        self.batch_index = 0
        self.request_ids: list[str] = []
        self.num_draft_tokens = 0
        self.num_accepted_tokens = 0
        scheduler.update_from_output = self.update_from_output
        scheduler.make_spec_decoding_stats = self.make_spec_decoding_stats

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        stats = self.original_stats(
            spec_decoding_stats,
            num_draft_tokens,
            num_accepted_tokens,
            num_invalid_spec_tokens,
            request_id,
        )
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        if num_draft_tokens > 0:
            self.request_ids.append(request_id)
            self.num_draft_tokens += num_draft_tokens
            self.num_accepted_tokens += num_accepted_tokens
        return stats

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        self.batch_index += 1
        try:
            outputs = self.original_update(scheduler_output, model_runner_output)
            # Integer comparison keeps exactly 10% outside the trigger.
            if is_low_mtp_acceptance(self.num_draft_tokens, self.num_accepted_tokens):
                from vllm_torchtpu.runner.mtp_diagnostic_output import (
                    MtpDiagnosticModelRunnerOutput,
                )

                # Aborts/load failures can remove rows after the worker's
                # whole-batch gate. Never force a normal worker batch to copy
                # tensors just because the scheduler later keeps a subset.
                diagnostics = []
                drafts = {}
                diagnostic_status = "worker_batch_not_below_threshold_or_unavailable"
                if isinstance(model_runner_output, MtpDiagnosticModelRunnerOutput):
                    diagnostics = model_runner_output.tensor_diagnostics
                    drafts = model_runner_output.verified_draft_token_ids
                    diagnostic_status = "worker_low_acceptance_capture"
                logger.warning(
                    "[MTP_LOW_ACCEPTANCE] scope=batch "
                    "reason=acceptance_rate_below_threshold "
                    "dp_index=%s batch_index=%d num_draft_tokens=%d "
                    "num_accepted_tokens=%d acceptance_rate=%s "
                    "acceptance_threshold=0.1 "
                    "diagnostic_status=%s "
                    "tensor_diagnostics=%s "
                    "verified_draft_token_ids=%s request_ids=%s",
                    self.dp_index,
                    self.batch_index,
                    self.num_draft_tokens,
                    self.num_accepted_tokens,
                    self.num_accepted_tokens / self.num_draft_tokens,
                    diagnostic_status,
                    json.dumps(diagnostics),
                    json.dumps(drafts),
                    json.dumps(self.request_ids),
                )
            return outputs
        finally:
            self.request_ids.clear()
            self.num_draft_tokens = 0
            self.num_accepted_tokens = 0


def is_mtp_diagnostic_enabled(config: VllmConfig) -> bool:
    """Independent scheduler/worker entries share the same PD role policy."""
    kv_config = config.kv_transfer_config
    return (
        envs.TPU_MTP_ZERO_ACCEPTANCE_LOG
        and config.speculative_config is not None
        and (kv_config is None or kv_config.is_kv_consumer)
    )


def is_mtp_tensor_diagnostic_enabled(config: VllmConfig) -> bool:
    """Keep TPU diagnostic work on the model execution thread.

    Diagnostic D2H on async output threads can fail when TorchTPU creates
    chained tasks from retained deferred tensors. Keep device work out of
    that thread; scheduler attribution remains enabled.
    """
    return is_mtp_diagnostic_enabled(config) and not (
        config.scheduler_config.async_scheduling
    )


def patch_scheduler_init() -> None:
    """Install at startup; disabled runs keep their original scheduler methods."""
    if not envs.TPU_MTP_ZERO_ACCEPTANCE_LOG:
        return

    from vllm.v1.core.sched.scheduler import Scheduler

    original_init = Scheduler.__init__

    @wraps(original_init)
    def init(self, vllm_config, *args, **kwargs):
        original_init(self, vllm_config, *args, **kwargs)
        if is_mtp_diagnostic_enabled(vllm_config):
            _BatchLowAcceptance(self)

    Scheduler.__init__ = init
