# SPDX-License-Identifier: Apache-2.0
"""Admission for the experimental, unchunked MTP prefill diagnostic."""

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
    if config.scheduler_config.enable_chunked_prefill:
        raise NotImplementedError("MTP prefill diagnostic requires unchunked prefill.")
    if config.cache_config.enable_prefix_caching:
        raise NotImplementedError("MTP prefill diagnostic requires prefix caching off.")


def validate_prefill_span(computed: int, scheduled: int, prompt: int) -> None:
    """Require a complete fresh prompt; no cache hits, chunks or decode."""
    if computed != 0 or scheduled != prompt:
        raise NotImplementedError(
            "MTP prefill diagnostic requires a complete fresh prompt: "
            f"computed={computed}, scheduled={scheduled}, prompt={prompt}."
        )
