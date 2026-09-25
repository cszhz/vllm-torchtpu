# SPDX-License-Identifier: Apache-2.0
"""Opt-in Raiden send results carried by vLLM's worker metadata channel."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata

from vllm_torchtpu import envs
from vllm_torchtpu.distributed import utils as dist_utils

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def raiden_pd_completion_modes(vllm_config: "VllmConfig") -> tuple[bool, bool]:
    """Select response contracts shared by scheduler and worker entry points."""
    additional = vllm_config.additional_config
    send = (
        additional.get("raiden_pd_send_completion", False)
        if isinstance(additional, dict)
        else False
    )
    if not isinstance(send, bool):
        raise ValueError("additional_config.raiden_pd_send_completion must be a bool")

    kv = vllm_config.kv_transfer_config
    stage3 = str(envs.TPU_KV_RESHARD_TRANSPORT).strip().lower() == "raiden"
    decode = (
        stage3
        and kv is not None
        and kv.kv_role == "kv_consumer"
        and kv.kv_load_failure_policy == "fail"
    )
    if not send and not decode:
        return False, False

    if send:
        parallel = vllm_config.parallel_config
        if (
            not stage3
            or kv is None
            or kv.kv_role != "kv_producer"
            or parallel.tensor_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
        ):
            raise ValueError(
                "raiden_pd_send_completion requires a Raiden Stage-3 "
                "kv_producer with TP1/DP1/PP1"
            )

    connectors = (
        kv.kv_connector_extra_config["connectors"]
        if kv.kv_connector == "TPUMultiConnector"
        else [{"kv_connector": kv.kv_connector}]
    )
    raiden_count = sum(c["kv_connector"] == "TPURaidenConnector" for c in connectors)
    if (send and raiden_count != 1) or (decode and raiden_count > 1):
        mode = "raiden_pd_send_completion" if send else "Raiden PD decode fail policy"
        raise ValueError(f"{mode} requires exactly one TPURaidenConnector")
    decode = decode and raiden_count == 1
    if decode and dist_utils.get_raiden_inline_load():
        raise ValueError(
            "Raiden PD decode fail policy requires TPU_RAIDEN_INLINE_LOAD=0"
        )
    return send, decode


@dataclass
class RaidenLoadFailures(KVConnectorWorkerMetadata):
    # Request identity is independent of hybrid cache-group block numbering.
    # finished_recving remains the separate barrier for releasing buffers.
    failed_req_ids: set[str] = field(default_factory=set)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "RaidenLoadFailures":
        assert isinstance(other, RaidenLoadFailures), (
            "Raiden load failures must aggregate with the same connector type"
        )
        self.failed_req_ids.update(other.failed_req_ids)
        return self


class RaidenSendOutcome(StrEnum):
    SUCCESS = "success"
    SEND_FAILURE = "send_failure"
    UNCLAIMED_TIMEOUT = "unclaimed_timeout"


_OUTCOME_PRIORITY = {
    RaidenSendOutcome.SUCCESS: 0,
    RaidenSendOutcome.UNCLAIMED_TIMEOUT: 1,
    RaidenSendOutcome.SEND_FAILURE: 2,
}


def merge_raiden_send_outcomes(
    current: RaidenSendOutcome | None,
    candidate: RaidenSendOutcome,
) -> RaidenSendOutcome:
    if current is None or _OUTCOME_PRIORITY[candidate] > _OUTCOME_PRIORITY[current]:
        return candidate
    return current


@dataclass
class RaidenSendResults(KVConnectorWorkerMetadata):
    # Preserve the terminal reason across worker/rank aggregation.
    # finished_sending remains the independent cleanup barrier.
    outcomes: dict[str, RaidenSendOutcome] = field(default_factory=dict)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "RaidenSendResults":
        assert isinstance(other, RaidenSendResults), (
            "Raiden send results must aggregate with the same connector type"
        )
        for req_id, outcome in other.outcomes.items():
            self.outcomes[req_id] = merge_raiden_send_outcomes(
                self.outcomes.get(req_id), outcome
            )
        return self
