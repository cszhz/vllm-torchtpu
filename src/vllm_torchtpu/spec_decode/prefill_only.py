# SPDX-License-Identifier: Apache-2.0
"""Admission for the experimental MTP prefill producer."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


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
    transfer = config.kv_transfer_config
    if transfer is not None:
        from vllm_torchtpu import envs

        if (
            transfer.kv_connector != "TPURaidenConnector"
            or transfer.kv_role != "kv_producer"
            or envs.TPU_KV_RESHARD_TRANSPORT != "raiden"
            or not envs.TPU_RAIDEN_QWEN35_ADMISSION
            or not envs.TPU_RAIDEN_POOL_TAGS_PER_LAYER
            or pc.pipeline_parallel_size <= 1
        ):
            raise NotImplementedError(
                "MTP prefill KV transfer requires a pipeline Raiden Stage-3 "
                "producer with Qwen hybrid admission and per-layer pool tags."
            )
    if config.scheduler_config.async_scheduling:
        raise NotImplementedError("MTP prefill diagnostic requires sync scheduling.")


def validate_prefill_span(computed: int, scheduled: int, prompt: int) -> None:
    """Allow fresh, chunked and cache-hit prefill, but never decode."""
    if computed < 0 or scheduled <= 0 or computed + scheduled > prompt:
        raise NotImplementedError(
            "MTP prefill diagnostic requires a span within the prompt: "
            f"computed={computed}, scheduled={scheduled}, prompt={prompt}."
        )


def handoff_seed_tokens(
    device_seed: "torch.Tensor",
    req_ids: list[str],
    metadata: "KVConnectorMetadata | None",
) -> "torch.Tensor":
    """Use the known continuation, rather than P's sampled replacement."""
    import torch

    from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
        TPUConnectorMetadata,
    )

    if metadata is None:
        return device_seed
    assert isinstance(metadata, TPUConnectorMetadata)
    indices, tokens = [], []
    for index, req_id in enumerate(req_ids):
        send = metadata.reqs_to_send.get(req_id)
        if send is not None and send.next_prompt_token_id is not None:
            indices.append(index)
            tokens.append(send.next_prompt_token_id)
    if not indices:
        return device_seed
    result = device_seed.clone()
    result[torch.tensor(indices, device=result.device)] = torch.tensor(
        tokens, dtype=result.dtype, device=result.device
    )
    return result
