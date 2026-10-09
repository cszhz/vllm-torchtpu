# SPDX-License-Identifier: Apache-2.0
"""Admission for the experimental MTP prefill producer."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def validate_prefill_only_config(config: "VllmConfig") -> None:
    """Reject modes whose MTP cache boundary this diagnostic cannot validate."""
    pc = config.parallel_config
    spec = config.speculative_config
    if spec is None or spec.method != "mtp" or spec.num_speculative_tokens != 1:
        raise ValueError("TPU_MTP_PREFILL_ONLY requires method=mtp and K=1.")
    if (
        pc.tensor_parallel_size != 1
        or pc.decode_context_parallel_size != 1
        or pc.data_parallel_size != 1
    ):
        raise NotImplementedError("MTP prefill diagnostic requires TP=DCP=DP=1.")
    if config.kv_transfer_config is not None:
        raise NotImplementedError("MTP prefill diagnostic does not transfer KV.")
    if config.scheduler_config.async_scheduling:
        raise NotImplementedError("MTP prefill diagnostic requires sync scheduling.")


def validate_prefill_span(computed: int, scheduled: int, prompt: int) -> None:
    """Allow fresh, chunked and cache-hit prefill, but never decode."""
    if computed < 0 or scheduled <= 0 or computed + scheduled > prompt:
        raise NotImplementedError(
            "MTP prefill diagnostic requires a span within the prompt: "
            f"computed={computed}, scheduled={scheduled}, prompt={prompt}."
        )
