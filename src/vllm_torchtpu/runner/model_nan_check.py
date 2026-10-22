# SPDX-License-Identifier: Apache-2.0
"""Opt-in final hidden/logits NaN/Inf detection without changing model outputs."""

from __future__ import annotations

import functools
import socket
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal import pallas
from torch_tpu._internal.stream.streams import TpuEvent
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorStats,
)

from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorStats,
)
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutPlan

logger = init_logger(__name__)


def _has_nan_jax(tensor: jax.Array, valid_rows: jax.Array) -> jax.Array:
    return (~jnp.isfinite(tensor)).any(where=valid_rows[:, None])


@functools.cache
def _get_has_nan_op() -> torch.library.CustomOpDef:
    return pallas.jax_op("pallas::model_has_nan", _has_nan_jax)


def _stats_paths(
    connector_cls: type[KVConnectorBase_V1], vllm_config: VllmConfig
) -> list[tuple[str, ...]]:
    if issubclass(connector_cls, MultiConnector):
        children = connector_cls._get_connector_classes_and_configs(vllm_config)
        return [
            (child_cls.__name__, *path)
            for child_cls, child_config in children
            for path in _stats_paths(child_cls, child_config)
        ]
    return (
        [()]
        if isinstance(connector_cls.build_kv_connector_stats(), TpuKVConnectorStats)
        else []
    )


def _merge_stats(
    stats: KVConnectorStats | None,
    path: tuple[str, ...],
    rounds: list[tuple[int, int, int]],
) -> KVConnectorStats:
    leaf = TpuKVConnectorStats()
    for round_id, rank, has_nan in rounds:
        leaf.record_model_nan(round_id, rank, bool(has_nan))
    update: KVConnectorStats = leaf
    for key in reversed(path):
        update = MultiKVConnectorStats(data={key: update})
    return update if stats is None else stats.aggregate(update)


@dataclass
class _Check:
    name: str
    shape: tuple[int, ...]
    result: torch.Tensor


@dataclass
class _PendingRound:
    round_id: int
    checks: tuple[_Check, ...]
    results_cpu: torch.Tensor
    # Keep the source alive until its asynchronous D2H completes.
    results_tpu: torch.Tensor
    ready_event: TpuEvent


class ModelNaNChecker:
    """Queue device flags and poll completed rounds on the runner thread."""

    def __init__(
        self,
        rank: int,
        stats_path: tuple[str, ...] | None = None,
        *,
        engine_idx: int = 0,
    ) -> None:
        self.rank = rank
        self.engine_idx = engine_idx
        self._hostname = socket.gethostname()
        self._stats_path = stats_path
        self._checks: list[_Check] = []
        self._pending: deque[_PendingRound] = deque()
        self._next_round_id = 0
        self._completed: list[tuple[int, int, int]] = []

    @classmethod
    def from_vllm_config(cls, rank: int, vllm_config: VllmConfig) -> ModelNaNChecker:
        """Resolve the telemetry carrier once using registered connector APIs."""
        engine_idx = vllm_config.parallel_config.data_parallel_rank
        transfer_config = vllm_config.kv_transfer_config
        if transfer_config is None:
            return cls(rank, engine_idx=engine_idx)
        connector_cls = KVConnectorFactory.get_connector_class(transfer_config)
        paths = _stats_paths(connector_cls, vllm_config)
        if len(paths) > 1:
            raise ValueError(
                "NaN/Inf summary reporting requires at most one TPU stats leaf"
            )
        return cls(rank, paths[0] if paths else None, engine_idx=engine_idx)

    def prepare_stats(self, stats: KVConnectorStats | None) -> KVConnectorStats | None:
        """Attach completed rounds once, preserving the connector's own samples."""
        if not self._completed:
            return stats
        assert self._stats_path is not None, (
            "Completed NaN/Inf metrics require a TPU sink"
        )
        rounds, self._completed = self._completed, []
        return _merge_stats(stats, self._stats_path, rounds)

    def enqueue(self) -> None:
        """Submit one round's small D2H copy without waiting for its result."""
        if not self._checks:
            return
        checks = tuple(self._checks)
        self._checks = []
        results = torch.stack([check.result for check in checks])
        results_cpu = results.to("cpu", non_blocking=True)
        event = TpuEvent()
        event.record()
        self._pending.append(
            _PendingRound(self._next_round_id, checks, results_cpu, results, event)
        )
        self._next_round_id += 1

    def poll(self) -> None:
        """Drain the ready FIFO prefix; leave an unfinished head for the next call."""
        while self._pending:
            if not self._pending[0].ready_event.query():
                break
            pending = self._pending.popleft()
            flags = pending.results_cpu.tolist()
            for check, has_nan in zip(pending.checks, flags):
                if has_nan:
                    logger.error(
                        "NaN/Inf detected in %s: host=%s, engine=%d, rank=%d, shape=%s",
                        check.name,
                        self._hostname,
                        self.engine_idx,
                        self.rank,
                        check.shape,
                    )
            if self._stats_path is not None:
                self._completed.append((pending.round_id, self.rank, int(any(flags))))

    def check_hidden(self, hidden: torch.Tensor, plan: SequenceLayoutPlan) -> None:
        """Check all real rank-local tokens before hidden-state gathering."""
        token_indices = plan._packed_to_request_major_token_indices
        if token_indices is None:
            token_indices = np.arange(plan.global_padded_num_tokens)
        token_indices = token_indices[plan.token_slice]
        valid_rows = torch.from_numpy(
            (token_indices >= 0) & (token_indices < plan.global_num_tokens)
        )
        self._check(hidden, valid_rows, "last_hidden")

    def check_logits(self, logits: torch.Tensor, num_valid_rows: int) -> None:
        """Check real sampling/verification rows before logits processing."""
        assert 0 <= num_valid_rows <= logits.shape[0], (
            "Real sampling/verification rows must fit the logits buffer: "
            f"rank={self.rank}, real={num_valid_rows}, shape={logits.shape}"
        )
        valid_rows = torch.arange(logits.shape[0], device="cpu") < num_valid_rows
        self._check(logits, valid_rows, "logits")

    def _check(self, tensor: torch.Tensor, valid_rows: torch.Tensor, name: str) -> None:
        assert valid_rows.shape == (tensor.shape[0],), (
            "NaN/Inf check row mapping must cover the rank-local output: "
            f"rank={self.rank}, tensor={name}, shape={tensor.shape}, "
            f"mapping={valid_rows.shape}"
        )
        result = _get_has_nan_op()(
            tensor, valid_rows.to(tensor.device, non_blocking=True)
        )
        self._checks.append(_Check(name, tuple(tensor.shape), result))
