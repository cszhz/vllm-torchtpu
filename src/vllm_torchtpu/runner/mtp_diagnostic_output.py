# SPDX-License-Identifier: Apache-2.0
"""Feature-owned transport of the drafts verified by one runner output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.v1.outputs import ModelRunnerOutput

from vllm_torchtpu.core.mtp_zero_acceptance import is_mtp_diagnostic_enabled
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
)

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig

    from vllm_torchtpu.runner.speculative_decoding_manager import SpecDecodeMetadata


@dataclass
class MtpDiagnosticModelRunnerOutput(ModelRunnerOutput):
    verified_draft_token_ids: dict[str, list[int]] = field(default_factory=dict)


class _MtpDiagnosticOutput(AsyncTPUModelRunnerOutput):
    def __init__(
        self,
        original: AsyncTPUModelRunnerOutput,
        req_ids: list[str],
        draft_lengths: list[int],
        draft_copy: AsyncTPUCopyState,
    ) -> None:
        super().__init__(
            original._model_runner_output,
            original.copy_state,
            original.discard_sampled_tokens_req_indices,
        )
        self.req_ids = req_ids.copy()
        self.draft_lengths = draft_lengths
        self.draft_copy = draft_copy

    def get_output(self) -> MtpDiagnosticModelRunnerOutput:
        output = super().get_output()
        self.draft_copy.wait()
        flat_ids = self.draft_copy.sampled_token_ids_cpu.tolist()
        drafts = {}
        offset = 0
        for req_id, length in zip(self.req_ids, self.draft_lengths):
            drafts[req_id] = flat_ids[offset : offset + length]
            offset += length
        assert offset == len(flat_ids), (
            f"MTP draft snapshot has {len(flat_ids)} tokens but lengths sum to {offset}"
        )
        return MtpDiagnosticModelRunnerOutput(
            **vars(output), verified_draft_token_ids=drafts
        )


def maybe_wrap_mtp_output(
    output: AsyncTPUModelRunnerOutput,
    config: VllmConfig,
    req_ids: list[str],
    metadata: list[SpecDecodeMetadata | None],
    num_reqs: list[int],
) -> AsyncTPUModelRunnerOutput:
    """Snapshot actual verification IDs before the next execute_model mutates state."""
    if not is_mtp_diagnostic_enabled(config):
        return output
    assert len(metadata) == len(num_reqs) and sum(num_reqs) == len(req_ids), (
        f"MTP chunk/request alignment: chunks={len(metadata)}, counts={num_reqs}, "
        f"requests={len(req_ids)}"
    )
    lengths: list[int] = []
    tensors: list[torch.Tensor] = []
    real_lens: list[int] = []
    for md, count in zip(metadata, num_reqs):
        if md is None:
            lengths.extend([0] * count)
        else:
            chunk_lengths = md.draft_lengths_cpu.tolist()
            assert len(chunk_lengths) == count, (
                f"MTP draft lengths have {len(chunk_lengths)} rows for {count} requests"
            )
            lengths.extend(chunk_lengths)
            tensors.append(md.draft_token_ids)
            real_lens.append(sum(chunk_lengths))
    if not any(real_lens):
        return output
    # Keep bucket shapes on device; trim padding and split variable lengths on CPU.
    draft_copy = AsyncTPUCopyState.from_device_chunks(tensors, real_lens)
    return _MtpDiagnosticOutput(output, req_ids, lengths, draft_copy)
