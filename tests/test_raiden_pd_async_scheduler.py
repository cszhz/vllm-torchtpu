# SPDX-License-Identifier: Apache-2.0
"""Raiden completion installs on vLLM's official scheduler entrypoints."""

from types import SimpleNamespace

import pytest
from vllm.config import SchedulerConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.core.raiden_pd import (
    _DecodeLoadFailures,
    _ProducerCompletion,
    patch_scheduler_init,
)
from vllm_torchtpu.distributed import utils as dist_utils
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnector


@pytest.mark.parametrize(
    ("async_scheduling", "expected"),
    [
        (False, Scheduler),
        (True, AsyncScheduler),
    ],
)
def test_default_entrypoint_selects_the_official_scheduler(async_scheduling, expected):
    config = SchedulerConfig(
        async_scheduling=async_scheduling, max_model_len=32, is_encoder_decoder=False
    )
    assert config.scheduler_cls is None
    assert config.get_scheduler_cls() is expected


def _config(*, async_scheduling, producer):
    return SimpleNamespace(
        additional_config={
            "raiden_pd_send_completion": producer,
        },
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1
        ),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_producer" if producer else "kv_consumer",
            kv_load_failure_policy="recompute" if producer else "fail",
            kv_connector="TPURaidenConnector",
            kv_connector_extra_config={},
        ),
    )


@pytest.mark.parametrize(
    ("scheduler_cls", "producer", "expected_state"),
    [
        (Scheduler, True, _ProducerCompletion),
        (AsyncScheduler, False, _DecodeLoadFailures),
    ],
)
def test_init_patch_installs_completion_on_official_schedulers(
    scheduler_cls, producer, expected_state, monkeypatch
):
    async_scheduling = scheduler_cls is AsyncScheduler
    config = _config(async_scheduling=async_scheduling, producer=producer)
    monkeypatch.setattr(tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(dist_utils, "get_raiden_inline_load", lambda: False)

    def base_init(self, *_args, **_kwargs):
        self.connector = TPURaidenConnector.__new__(TPURaidenConnector)
        self.finished_req_ids_dict = None
        self.num_spec_tokens = 0
        self.parallel_config = config.parallel_config
        self.update_from_output = lambda *_args: {}
        self.finish_requests = lambda *_args: []
        self._handle_invalid_blocks = lambda *_args: set()

    monkeypatch.setattr(Scheduler, "__init__", base_init)
    monkeypatch.delattr(Scheduler, "_tpu_raiden_pd_completion_patch", raising=False)
    patch_scheduler_init()

    scheduler = scheduler_cls(config, None)
    state = scheduler.update_from_output.__self__
    assert isinstance(state, expected_state)
    assert type(scheduler) is scheduler_cls
    if scheduler_cls is AsyncScheduler:
        assert (
            scheduler._update_after_schedule.__func__
            is AsyncScheduler._update_after_schedule
        )
        assert (
            scheduler._update_request_with_output.__func__
            is AsyncScheduler._update_request_with_output
        )
