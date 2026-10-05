# SPDX-License-Identifier: Apache-2.0
"""Opt-in request attribution for batches with draft acceptance below 10%."""

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
            if (
                self.num_draft_tokens > 0
                and self.num_accepted_tokens * 10 < self.num_draft_tokens
            ):
                from vllm_torchtpu.runner.mtp_diagnostic_output import (
                    MtpDiagnosticModelRunnerOutput,
                )

                assert isinstance(
                    model_runner_output, MtpDiagnosticModelRunnerOutput
                ), "MTP diagnostic runner output must carry the verified draft snapshot"
                logger.warning(
                    "[MTP_LOW_ACCEPTANCE] scope=batch "
                    "reason=acceptance_rate_below_threshold "
                    "dp_index=%s batch_index=%d num_draft_tokens=%d "
                    "num_accepted_tokens=%d acceptance_rate=%s "
                    "acceptance_threshold=0.1 "
                    "verified_draft_token_ids=%s request_ids=%s",
                    self.dp_index,
                    self.batch_index,
                    self.num_draft_tokens,
                    self.num_accepted_tokens,
                    self.num_accepted_tokens / self.num_draft_tokens,
                    json.dumps(model_runner_output.verified_draft_token_ids),
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


def patch_scheduler_init() -> None:
    """Install only at startup; disabled runs keep their original methods."""
    if not envs.TPU_MTP_ZERO_ACCEPTANCE_LOG:
        return

    from vllm.v1.core.sched.scheduler import Scheduler

    original_init = Scheduler.__init__

    @wraps(original_init)
    def init(self, vllm_config, *args, **kwargs):
        original_init(self, vllm_config, *args, **kwargs)
        if is_mtp_diagnostic_enabled(vllm_config):
            _BatchLowAcceptance(self)
            logger.info("Enabled batch MTP request logging for acceptance below 10%")

    Scheduler.__init__ = init
