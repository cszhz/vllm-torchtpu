# SPDX-License-Identifier: Apache-2.0
"""Temporary MTP diagnostics, materialized only below 10% batch acceptance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.v1.outputs import ModelRunnerOutput

from vllm_torchtpu.core.mtp_zero_acceptance import (
    is_low_mtp_acceptance,
    is_mtp_tensor_diagnostic_enabled,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig

    from vllm_torchtpu.runner.mtp_tensor_diagnostics import DeferredMtpChunk
    from vllm_torchtpu.runner.speculative_decoding_manager import SpecDecodeMetadata

logger = init_logger(__name__)


@dataclass
class MtpDiagnosticModelRunnerOutput(ModelRunnerOutput):
    verified_draft_token_ids: dict[str, list[int]] = field(default_factory=dict)
    tensor_diagnostics: list[dict[str, Any]] = field(default_factory=list)


class _MtpDiagnosticOutput(AsyncTPUModelRunnerOutput):
    def __init__(
        self,
        original: AsyncTPUModelRunnerOutput,
        req_ids: list[str],
        draft_lengths: list[int],
        drafts: list[tuple[torch.Tensor, int]],
        captures: list[DeferredMtpChunk],
        invalid_drafts: dict[str, int],
    ) -> None:
        super().__init__(
            original._model_runner_output,
            original.copy_state,
            original.discard_sampled_tokens_req_indices,
            kv_ready_event=original.kv_ready_event,
        )
        self.req_ids = req_ids.copy()
        self.draft_lengths = draft_lengths
        self.drafts = drafts
        self.captures = captures.copy()
        self.invalid_drafts = invalid_drafts.copy()
        self._resolved_output: ModelRunnerOutput | None = None

    def get_output(self) -> ModelRunnerOutput:
        if self._resolved_output is not None:
            return self._resolved_output
        # This is the existing sampling D2H wait. The gate adds no device
        # operations, event, copy or synchronization on a normal batch.
        output = super().get_output()
        drafted = accepted = 0
        for req_id, length, tokens in zip(
            self.req_ids, self.draft_lengths, output.sampled_token_ids
        ):
            effective = length - self.invalid_drafts.get(req_id, 0)
            if tokens and effective > 0:
                drafted += effective
                accepted += max(len(tokens) - 1, 0)
        try:
            if not is_low_mtp_acceptance(drafted, accepted):
                self._resolved_output = output
                return output

            # Draft/hidden tensors are immutable results retained by reference,
            # never copied eagerly. Mutable KV is guarded by a step ticket.
            flat_ids = []
            for tensor, count in self.drafts:
                flat_ids.extend(tensor.detach().to("cpu").tolist()[:count])
            verified = {}
            offset = 0
            for req_id, length in zip(self.req_ids, self.draft_lengths):
                verified[req_id] = flat_ids[offset : offset + length]
                offset += length
            assert offset == len(flat_ids)
            reports = [capture.collect() for capture in self.captures]
            logger.warning(
                "[MTP_TENSOR_DIAGNOSTIC] scope=worker_output_batch "
                "num_draft_tokens=%d num_accepted_tokens=%d acceptance_rate=%s "
                "acceptance_threshold=0.1 diagnostics=%s",
                drafted,
                accepted,
                accepted / drafted,
                json.dumps(reports),
            )
            self._resolved_output = MtpDiagnosticModelRunnerOutput(
                **vars(output),
                verified_draft_token_ids=verified,
                tensor_diagnostics=reports,
            )
            return self._resolved_output
        finally:
            # Do not pin old activations beyond output consumption.
            self.drafts.clear()
            self.captures.clear()


def maybe_wrap_mtp_output(
    output: AsyncTPUModelRunnerOutput,
    config: VllmConfig,
    req_ids: list[str],
    metadata: list[SpecDecodeMetadata | None],
    num_reqs: list[int],
    captures: list[DeferredMtpChunk] | None = None,
    invalid_drafts: dict[str, int] | None = None,
) -> AsyncTPUModelRunnerOutput:
    """Retain references only; defer all tensor diagnostics until acceptance."""
    if not is_mtp_tensor_diagnostic_enabled(config):
        return output
    assert len(metadata) == len(num_reqs) and sum(num_reqs) == len(req_ids)
    lengths: list[int] = []
    drafts: list[tuple[torch.Tensor, int]] = []
    for md, count in zip(metadata, num_reqs):
        if md is None:
            lengths.extend([0] * count)
        else:
            chunk_lengths = md.draft_lengths_cpu.tolist()
            assert len(chunk_lengths) == count
            lengths.extend(chunk_lengths)
            real_len = sum(chunk_lengths)
            if real_len:
                drafts.append((md.draft_token_ids, real_len))
    if not drafts:
        return output
    return _MtpDiagnosticOutput(
        output, req_ids, lengths, drafts, captures or [], invalid_drafts or {}
    )
