# SPDX-License-Identifier: Apache-2.0
"""Static checks for pipeline parallelism on TPU."""

from typing import Any


def _legacy_ray_executor() -> bool:
    import vllm.envs as vllm_envs

    from vllm_torchtpu import envs

    return (
        envs.TPU_MULTIHOST_BACKEND == "ray"
        and not vllm_envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND
    )


_PP_KV_CONNECTORS = {"TPURaidenConnector", "TPUV6dObjectConnector"}

_FULL_ATTENTION_LAYER_TYPES = ["full_attention", "deepseek_sparse_attention"]


def is_supported_pp_transfer(transfer: Any) -> bool:
    """Admit stage-aware Raiden alone or its explicit V6D composition."""
    if transfer.kv_connector in _PP_KV_CONNECTORS:
        return True
    if transfer.kv_connector != "TPUMultiConnector":
        return False
    children = getattr(transfer, "kv_connector_extra_config", {}).get("connectors", [])
    if len(children) != 2:
        return False
    by_name = {child.get("kv_connector"): child for child in children}
    if set(by_name) != {"TPURaidenConnector", "TPUV6dObjectConnector"}:
        return False
    return (
        by_name["TPURaidenConnector"].get("kv_role", transfer.kv_role)
        == transfer.kv_role
        and by_name["TPUV6dObjectConnector"].get("kv_role") == "kv_both"
        and by_name["TPUV6dObjectConnector"].get("kv_connector_module_path")
        == "kvs_connector.v6d_object_connector"
    )


def prefill_only_mtp(vllm_config: Any) -> bool:
    """Whether the speculative config is the prefill-only MTP a pipeline
    admits: one MTP token on a KV producer, which drafts to fill the draft
    layer's KV for a decode peer and never verifies."""
    spec = vllm_config.speculative_config
    transfer = vllm_config.kv_transfer_config
    return (
        spec is not None
        and spec.method == "mtp"
        and spec.num_speculative_tokens == 1
        and transfer is not None
        and bool(transfer.is_kv_producer)
    )


def _stage_without_full_attention(vllm_config: Any) -> str | None:
    """Describe the first stage holding no full-attention layer, or None.

    Only hybrid models declare ``layer_types``; every layer of the others
    has a KV cache.
    """
    from vllm.distributed.utils import get_pp_indices

    hf_config = vllm_config.model_config.hf_text_config
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is None:
        return None
    stages = vllm_config.parallel_config.pipeline_parallel_size
    for stage in range(stages):
        start, end = get_pp_indices(len(layer_types), stage, stages)
        kinds = layer_types[start:end]
        if not any(kind in _FULL_ATTENTION_LAYER_TYPES for kind in kinds):
            return (
                f"stage {stage} holds layers {start}-{end - 1}, "
                f"{'/'.join(sorted(set(kinds)))} only"
            )
    return None


def validate_pipeline_parallel_config(vllm_config: Any) -> None:
    """Reject configurations the TPU pipeline-parallel path does not serve.

    Raises NotImplementedError naming the unsupported feature.
    """
    from vllm_torchtpu import envs
    from vllm_torchtpu.spec_decode.prefill_only import validate_prefill_only_config

    if envs.TPU_MTP_PREFILL_ONLY:
        validate_prefill_only_config(vllm_config)
    parallel_config = vllm_config.parallel_config
    if parallel_config.pipeline_parallel_size <= 1:
        return

    unsupported = None
    remedy = ""
    if parallel_config.tensor_parallel_size > 1:
        # On torch_tpu (dev20260813) a stage's TP collectives stall on the
        # first serving step, when the other stages are not launching the
        # same collectives; warmup, where every stage runs them, passes.
        unsupported = "tensor parallelism inside a stage"
        remedy = " Use --tensor-parallel-size 1."
    elif (
        parallel_config.prefill_context_parallel_size > 1
        and vllm_config.kv_transfer_config is not None
        and not is_supported_pp_transfer(vllm_config.kv_transfer_config)
    ):
        # These connectors pair a stage with its peer through one pool per
        # device; a context-parallel stage spreads a request over several.
        # The raiden connector lowers each PCP worker's chunks of the
        # stage's layers.
        unsupported = "KV transfer connectors with context-parallel stages"
    elif vllm_config.scheduler_config.async_scheduling:
        # Stages before the last never see the sampled tokens the async
        # placeholder substitution reads from the device.
        unsupported = "async scheduling"
        remedy = " Pass --no-async-scheduling."
    elif vllm_config.speculative_config is not None and not envs.TPU_MTP_PREFILL_ONLY:
        unsupported = "speculative decoding"
    elif vllm_config.kv_transfer_config is not None and not is_supported_pp_transfer(
        vllm_config.kv_transfer_config
    ):
        # The raiden connector transfers each stage's own layers; the
        # other connectors assume every worker holds every layer.
        unsupported = "KV transfer connectors other than TPURaidenConnector"
    elif (
        vllm_config.kv_transfer_config is not None
        and (vllm_config.kv_transfer_config.kv_connector != "TPUV6dObjectConnector")
        and (stage := _stage_without_full_attention(vllm_config))
    ):
        # The raiden connector pairs a stage with its peer through the
        # measured layout of a full-attention KV pool, so a stage made of
        # linear-attention layers alone has nothing to register.
        unsupported = (
            f"TPURaidenConnector with a stage holding no full-attention layer ({stage})"
        )
        remedy = (
            " Choose a pipeline size or VLLM_PP_LAYER_PARTITION that "
            "gives every stage a full-attention layer."
        )
    elif _legacy_ray_executor():
        # The pipeline's hand-off pushes go through the executor hook that
        # the multiprocess executor and the Ray V2 executor share; the
        # legacy Ray executor has no such hook.
        unsupported = "the legacy Ray executor"
        remedy = " Set VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1."
    if unsupported is not None:
        raise NotImplementedError(
            f"Pipeline parallelism on TPU does not support {unsupported} yet.{remedy}"
        )
