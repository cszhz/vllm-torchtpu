# SPDX-License-Identifier: Apache-2.0
"""NaN round results survive transfer idleness and reach injected metrics."""

import logging
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import msgspec
import pytest
import torch
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.parser import text_string_to_metric_families
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorStats,
)

from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics,
    TpuKVConnectorStats,
)
from vllm_torchtpu.runner import model_nan_check, tpu_runner
from vllm_torchtpu.runner.model_nan_check import ModelNaNChecker

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def cpu_logger(monkeypatch, model_nan_cpu):
    monkeypatch.setattr(model_nan_check, "logger", logging.getLogger(__name__))


def consume(check, stats=None):
    check.enqueue()
    check.poll()
    return check.prepare_stats(stats)


def test_zero_round_is_nonempty_and_survives_serialization_and_reset():
    stats = TpuKVConnectorStats()
    stats.record_model_nan(0, 1, False)
    snapshot = stats.clone_and_reset()
    restored = TpuKVConnectorStats(
        data=msgspec.msgpack.decode(msgspec.msgpack.encode(snapshot.data))
    )
    assert not restored.is_empty()
    assert restored.data["model_nan_detected"] == [[0, 1, 0]]
    stats.record_model_nan(1, 1, True)
    assert snapshot.data["model_nan_detected"] == [(0, 1, 0)]


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_result_preserves_existing_samples_and_creates_missing_stats(
    value, multi, existing
):
    path = ("TPURaidenConnector",) if multi else ()
    check = ModelNaNChecker(rank=1, stats_path=path)
    other = KVConnectorStats(data={"other": [7]})
    leaf = TpuKVConnectorStats() if existing else None
    if leaf is not None:
        leaf.record_kv_pull(2.5)
    original: KVConnectorStats | None = leaf
    if multi:
        original = MultiKVConnectorStats(data={"OtherConnector": other})
        if leaf is not None:
            original.data["TPURaidenConnector"] = leaf
    check.check_logits(torch.tensor([[value]]), 1)
    result = consume(check, original)
    if multi:
        assert result is original
        assert result.data["OtherConnector"] is other
        result = result.data["TPURaidenConnector"]
    assert isinstance(result, TpuKVConnectorStats)
    assert result.data["model_nan_detected"] == [(0, 1, int(value != 0.0))]
    if leaf is not None:
        assert result is leaf
        assert result.data["kv_pull_time"] == [2.5]
    assert check.prepare_stats(original) is original


def test_runner_collects_results_even_without_transfer_samples(monkeypatch):
    check = ModelNaNChecker(rank=1, stats_path=())
    check.check_logits(torch.tensor([[float("nan")]]), 1)
    connector = SimpleNamespace(
        get_finished=lambda _: (None, None),
        get_block_ids_with_load_errors=lambda: set(),
        build_connector_worker_meta=lambda: None,
        get_kv_connector_stats=lambda: None,
        clear_connector_metadata=lambda: None,
    )
    runner = SimpleNamespace(
        _nan_checker=check,
        _stage_timing=SimpleNamespace(take_samples=lambda: ([], [])),
    )
    monkeypatch.setattr(tpu_runner, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(tpu_runner, "get_kv_transfer_group", lambda: connector)
    check.enqueue()
    output = tpu_runner.TPUModelRunner.get_finished_kv_transfers(
        runner, SimpleNamespace(finished_req_ids=set())
    )
    assert output[-1].data["model_nan_detected"] == [(0, 1, 1)]
    assert check.prepare_stats(None) is None


def test_runner_polls_without_a_kv_connector(monkeypatch, caplog):
    check = ModelNaNChecker(rank=1)
    check.check_logits(torch.tensor([[float("nan")]]), 1)
    check.enqueue()
    runner = SimpleNamespace(
        _nan_checker=check,
        _stage_timing=SimpleNamespace(take_samples=lambda: ([], [])),
    )
    monkeypatch.setattr(tpu_runner, "has_kv_transfer_group", lambda: False)
    with caplog.at_level(logging.ERROR):
        output = tpu_runner.TPUModelRunner.get_finished_kv_transfers(runner, None)
    assert output[-1] is None
    assert len(caplog.records) == 1


def test_zero_token_call_without_connector_polls_the_idle_tail(monkeypatch, caplog):
    check = ModelNaNChecker(rank=1)
    check.check_logits(torch.tensor([[float("nan")]]), 1)
    check.enqueue()
    runner = SimpleNamespace(
        _nan_checker=check,
        execute_model_state=None,
        _pp_wave=None,
        _update_states=lambda _: None,
        vllm_config=None,
        maybe_setup_kv_connector=lambda *_, **__: None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    monkeypatch.setattr(tpu_runner, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(tpu_runner, "set_forward_context", lambda *_: nullcontext())
    monkeypatch.setattr(tpu_runner.dist_utils, "get_raiden_inline_load", lambda: False)
    with caplog.at_level(logging.ERROR):
        result = tpu_runner.TPUModelRunner.execute_model(
            runner, SimpleNamespace(total_num_scheduled_tokens=0)
        )
    assert result is tpu_runner.EMPTY_MODEL_RUNNER_OUTPUT
    assert len(caplog.records) == 1


def test_deferred_rounds_keep_distinct_metrics_and_are_reported_once(caplog):
    check = ModelNaNChecker(rank=1, stats_path=())
    for has_nan in (True, False, True):
        check.check_logits(torch.tensor([[float("nan") if has_nan else 1.0]]), 1)
        check.check_logits(torch.ones(2, 3), 2)
        check.enqueue()
    assert check.prepare_stats(None) is None
    with caplog.at_level(logging.ERROR):
        check.poll()
    assert len(caplog.records) == 2
    assert check.prepare_stats(None).data["model_nan_detected"] == [
        (0, 1, 1),
        (1, 1, 0),
        (2, 1, 1),
    ]
    check.poll()
    assert check.prepare_stats(None) is None


@pytest.mark.parametrize("tp,pcp", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_single_counter_counts_delayed_rounds_once_per_engine(monkeypatch, tp, pcp):
    registry = CollectorRegistry()
    monkeypatch.setattr(
        TpuKVConnectorPromMetrics, "_get_raiden_metric_definitions", lambda *_: {}
    )
    metrics = TpuKVConnectorPromMetrics(
        SimpleNamespace(
            kv_transfer_config=None,
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tp, prefill_context_parallel_size=pcp
            ),
        ),
        {
            Gauge: partial(Gauge, registry=registry),
            Counter: partial(Counter, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        },
        ["model_name", "engine"],
        {0: ["model", "0"], 1: ["model", "1"]},
    )
    # One rank can deliver several rounds before its peers catch up. Round 0
    # is clean; anomalies on one rank or all ranks each count just once.
    for rank in range(tp * pcp):
        records = [
            (0, rank, 0),
            (1, rank, int(rank == 0)),
            (2, rank, 1),
            (3, rank, int(rank == tp * pcp - 1)),
        ]
        metrics.observe({"model_nan_detected": records}, engine_idx=0)
        metrics.observe({"model_nan_detected": [(0, rank, 0)]}, engine_idx=1)
    metrics.observe({}, engine_idx=0)
    values = {
        (sample.name, sample.labels["engine"]): sample.value
        for family in text_string_to_metric_families(generate_latest(registry).decode())
        for sample in family.samples
        if sample.name
        in {
            "vllm:tpu_model_nan_rounds_total",
            "vllm:tpu_model_checked_rounds_total",
        }
    }
    assert values == {
        ("vllm:tpu_model_nan_rounds_total", "0"): 3.0,
        ("vllm:tpu_model_nan_rounds_total", "1"): 0.0,
    }


@pytest.mark.parametrize("tp,pcp", [(2, 1), (1, 2)])
def test_configuration_uses_multi_connector_class_keys_and_rejects_multiple_sinks(
    monkeypatch, tp, pcp
):
    class TpuLeaf:
        @classmethod
        def build_kv_connector_stats(cls):
            return TpuKVConnectorStats()

    class OtherLeaf:
        @classmethod
        def build_kv_connector_stats(cls):
            return None

    class Multi(MultiConnector):
        children = [TpuLeaf, OtherLeaf]

        @classmethod
        def _get_connector_classes_and_configs(cls, config):
            return [(child, config) for child in cls.children]

    monkeypatch.setattr(KVConnectorFactory, "get_connector_class", lambda _: Multi)
    config = SimpleNamespace(
        kv_transfer_config=object(),
        parallel_config=SimpleNamespace(
            data_parallel_rank=2,
            tensor_parallel_size=tp,
            prefill_context_parallel_size=pcp,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )
    check = ModelNaNChecker.from_vllm_config(rank=1, vllm_config=config)
    assert check.engine_idx == 2
    check.check_logits(torch.ones(1, 2), 1)
    result = consume(check)
    assert result.data["TpuLeaf"].data["model_nan_detected"] == [(0, 1, 0)]

    class SecondTpuLeaf(TpuLeaf):
        pass

    Multi.children = [TpuLeaf, SecondTpuLeaf]
    with pytest.raises(ValueError, match="one TPU stats"):
        ModelNaNChecker.from_vllm_config(rank=1, vllm_config=config)


@pytest.mark.parametrize("tp,pcp", [(1, 1), (2, 1), (1, 2)])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_multirank_without_connector_is_independent_of_output_consumption(
    tp, pcp, async_scheduling
):
    config = SimpleNamespace(
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            tensor_parallel_size=tp,
            prefill_context_parallel_size=pcp,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )
    ModelNaNChecker.from_vllm_config(0, config)
