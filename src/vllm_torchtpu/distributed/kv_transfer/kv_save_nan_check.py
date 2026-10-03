# SPDX-License-Identifier: Apache-2.0
"""Step-owned NaN checks consumed by asynchronous KV stores."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch_tpu._internal.stream.streams import TpuEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutPlan


@dataclass
class _Chunk:
    req_ids: tuple[str, ...]
    request_indices: np.ndarray
    flags_cpu: torch.Tensor
    flags_device: torch.Tensor
    ready_event: TpuEvent


class KVSaveNanCheck:
    """Snapshot all real forward chunks without synchronizing the issuer."""

    def __init__(self) -> None:
        self._chunks: list[_Chunk] = []
        self._bad_req_ids: set[str] | None = None

    def record(
        self,
        hidden_states: torch.Tensor,
        req_ids: tuple[str, ...],
        query_start_loc: np.ndarray,
        plan: SequenceLayoutPlan,
    ) -> None:
        flags = torch.isnan(hidden_states).any(dim=-1)
        flags_cpu = flags.to("cpu", non_blocking=True)
        ready_event = TpuEvent()
        ready_event.record()

        token_indices = plan._packed_to_request_major_token_indices
        if token_indices is None:
            token_indices = np.arange(plan.global_padded_num_tokens)
        token_indices = token_indices[plan.token_slice]
        assert len(token_indices) == hidden_states.shape[0], (
            "NaN check token mapping must cover the rank-local hidden states: "
            f"mapping={len(token_indices)}, hidden_states={hidden_states.shape}"
        )
        request_indices = np.searchsorted(
            query_start_loc[1:], token_indices, side="right"
        )
        request_indices[token_indices < 0] = len(req_ids)
        self._chunks.append(
            _Chunk(req_ids, request_indices, flags_cpu, flags, ready_event)
        )

    def is_ready(self) -> bool:
        return all(chunk.ready_event.query() for chunk in self._chunks)

    def bad_request_ids(self) -> set[str]:
        """Read the snapshot on the store thread, preserving event errors."""
        if self._bad_req_ids is None:
            bad_req_ids: set[str] = set()
            for chunk in self._chunks:
                chunk.ready_event.synchronize()
                indices = chunk.request_indices[chunk.flags_cpu.numpy()]
                bad_req_ids.update(
                    chunk.req_ids[i]
                    for i in np.unique(indices)
                    if i < len(chunk.req_ids)
                )
            self._bad_req_ids = bad_req_ids
        return self._bad_req_ids


class SupportsKVSaveNanCheck(ABC):
    """Opt-in store interface; ordinary KV connectors keep their original API."""

    @property
    @abstractmethod
    def needs_kv_save_nan_check(self) -> bool: ...

    @abstractmethod
    def wait_for_save(self, nan_check: KVSaveNanCheck | None = None) -> None: ...


def _leaf_connectors(connector: KVConnectorBase_V1) -> Iterator[KVConnectorBase_V1]:
    if isinstance(connector, MultiConnector):
        for child in connector._connectors:
            yield from _leaf_connectors(child)
    else:
        yield connector


def begin_kv_save_nan_check(connector: KVConnectorBase_V1) -> KVSaveNanCheck | None:
    if any(
        isinstance(child, SupportsKVSaveNanCheck) and child.needs_kv_save_nan_check
        for child in _leaf_connectors(connector)
    ):
        return KVSaveNanCheck()
    return None


def wait_for_save_with_nan_check(
    connector: KVConnectorBase_V1, nan_check: KVSaveNanCheck
) -> None:
    for child in _leaf_connectors(connector):
        if isinstance(child, SupportsKVSaveNanCheck) and child.needs_kv_save_nan_check:
            child.wait_for_save(nan_check=nan_check)
        else:
            child.wait_for_save()
