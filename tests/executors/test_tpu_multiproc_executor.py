# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import signal
from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.config import CacheConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import core as engine_core_module
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.outputs import DraftTokenIds

from vllm_torchtpu.core.remote_draft_scheduler import (
    RemoteDraftConsumerScheduler,
    RemoteDraftScheduler,
)
from vllm_torchtpu.executors.tpu_multiproc_executor import TpuMultiprocExecutor

# Sentinel returned by the parent get_kv_cache_specs; the override plumbing must
# pass this through unchanged.
_SPECS = [{"layer": MagicMock()}]


def _blocked_cpu_worker(ready, ignore_sigterm):
    if ignore_sigterm:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    signal.pause()


@pytest.mark.parametrize("ignore_sigterm", [False, True])
def test_engine_fatal_rpc_error_tears_down_blocked_child(monkeypatch, ignore_sigterm):
    """Actual fatal-loop envelope and executor teardown with a real CPU child.

    Model construction, engine I/O and RPC failure are explicit stand-ins.
    Forked child only signals readiness and blocks in the OS; no torch/device work.
    """
    from vllm import envs

    context = multiprocessing.get_context("fork")
    ready = context.Event()
    child = context.Process(target=_blocked_cpu_worker, args=(ready, ignore_sigterm))
    child.start()
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    response_queue, broadcast_queue = MagicMock(), MagicMock()
    executor.workers = [
        SimpleNamespace(
            proc=child, death_writer=None, worker_response_mq=response_queue
        )
    ]
    executor.rpc_broadcast_mq = broadcast_queue
    executor.response_mqs = []
    executor.shutting_down = False
    core = SimpleNamespace(
        run_busy_loop=MagicMock(side_effect=RuntimeError("injected fatal RPC")),
        _send_engine_dead=MagicMock(),
        shutdown=executor.shutdown,
    )
    config = SimpleNamespace(
        logging_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            numa_bind=False,
            reconfigure_for_independent_dp_rank=MagicMock(),
        ),
    )
    entrypoint = engine_core_module.EngineCoreProc.run_engine_core
    monkeypatch.setattr(
        engine_core_module, "EngineCoreProc", MagicMock(return_value=core)
    )
    for name in (
        "maybe_register_config_serialize_by_value",
        "set_process_title",
        "maybe_init_worker_tracer",
        "decorate_logs",
        "SignalCallback",
    ):
        monkeypatch.setattr(engine_core_module, name, MagicMock())
    monkeypatch.setattr(envs, "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", 0)
    # Preserve pytest's own signal handlers while exercising the fatal envelope.
    monkeypatch.setattr(engine_core_module.signal, "signal", MagicMock())
    try:
        assert ready.wait(10), "CPU child failed to start"
        with pytest.raises(RuntimeError, match="injected fatal RPC"):
            entrypoint(vllm_config=config)
        child.join(timeout=5)
        assert not child.is_alive()
        assert child.exitcode == (
            -signal.SIGKILL if ignore_sigterm else -signal.SIGTERM
        )
        core._send_engine_dead.assert_called_once_with()
        assert executor.shutting_down
        assert executor.rpc_broadcast_mq is None
        response_queue.shutdown.assert_called_once_with()
        broadcast_queue.shutdown.assert_called_once_with()
        executor.shutdown()
        response_queue.shutdown.assert_called_once_with()
        broadcast_queue.shutdown.assert_called_once_with()
    finally:
        if child.is_alive():
            child.kill()
        child.join(timeout=5)
        child.close()


@pytest.mark.parametrize(
    "scheduler_cls", [RemoteDraftScheduler, RemoteDraftConsumerScheduler]
)
@pytest.mark.parametrize("failing_rank", [None, 0, 1])
def test_remote_draft_rpc_reports_every_rank(scheduler_cls, failing_rank):
    """Real WorkerProc dispatch/response handling; CPU queues replace IPC only."""
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(get_scheduler_cls=lambda: scheduler_cls)
    )
    executor.output_rank = 0
    executor.is_failed = False
    executor.futures_queue = deque()
    result = DraftTokenIds(["A"], [[1, 2]])

    class ResponseQueue:
        def __init__(self):
            self.queue = Queue()

        def enqueue(self, value):
            self.queue.put(value)

        def dequeue(self, timeout):
            return self.queue.get(timeout=timeout)

    workers = []
    executor.response_mqs = []
    for rank in range(2):

        def take(rank=rank):
            if rank == failing_rank:
                raise RuntimeError(f"injected rank {rank} proposal failure")
            return result if rank == executor.output_rank else None

        worker = WorkerProc.__new__(WorkerProc)
        worker.rank = rank
        worker.worker = SimpleNamespace(take_draft_token_ids=take)
        worker.use_async_scheduling = False
        worker.worker_response_mq = ResponseQueue()
        workers.append(worker)
        executor.response_mqs.append(worker.worker_response_mq)

    class BroadcastQueue:
        def enqueue(self, request):
            for worker in workers:
                worker._execute_worker_rpc(request)

    executor.rpc_broadcast_mq = BroadcastQueue()
    if failing_rank is None:
        assert executor.take_draft_token_ids() is result
        assert all(mq.queue.empty() for mq in executor.response_mqs)
    else:
        with pytest.raises(RuntimeError, match=f"injected rank {failing_rank}"):
            executor.take_draft_token_ids()


def test_unplanned_draft_rpc_keeps_upstream_reply_contract():
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(get_scheduler_cls=lambda: Scheduler)
    )
    executor.output_rank = 1
    executor.collective_rpc = MagicMock(return_value=None)
    assert executor.take_draft_token_ids() is None
    assert executor.collective_rpc.call_args.kwargs["unique_reply_rank"] == 1


def _make_executor(engine_override, worker_overrides, pipeline_parallel_size=1):
    """Build an executor without running __init__, with a mocked engine
    cache_config and a collective_rpc that returns the per-worker overrides."""
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor.vllm_config = MagicMock()
    executor.vllm_config.cache_config = MagicMock(spec=CacheConfig)
    executor.vllm_config.cache_config.num_gpu_blocks_override = engine_override
    executor.vllm_config.parallel_config.world_size = len(worker_overrides)
    executor.vllm_config.parallel_config.pipeline_parallel_size = pipeline_parallel_size
    executor.collective_rpc = MagicMock(return_value=worker_overrides)
    return executor


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_copies_override_from_workers_when_engine_unset(mock_super):
    """Engine override is None -> copy the agreed worker value to the engine."""
    executor = _make_executor(engine_override=None, worker_overrides=[5428, 5428, 5428])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    executor.collective_rpc.assert_called_once_with("get_num_gpu_blocks_override")
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 5428


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_does_not_overwrite_existing_engine_override(mock_super):
    """A user-supplied engine override (e.g. --num-gpu-blocks-override) wins;
    the worker value is never queried."""
    executor = _make_executor(engine_override=1234, worker_overrides=[5428, 5428, 5428])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    executor.collective_rpc.assert_not_called()
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 1234


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_raises_when_workers_disagree(mock_super):
    """Workers must agree on the override; a mismatch is a sizing bug, not
    something to silently pick one of. 5428 -> 6840 is 26%, far outside the
    relative tolerance."""
    executor = _make_executor(engine_override=None, worker_overrides=[5428, 6840, 5428])

    with pytest.raises(ValueError, match="workers disagree"):
        executor.get_kv_cache_specs()


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_small_blocks_tolerate_measurement_jitter(mock_super):
    """The tolerance must scale with the block count.

    Workers size the pool from their own measured free HBM, so the counts
    never match exactly. With a small per-block size the same ~0.02 GiB of
    allocator jitter is hundreds of blocks: these are the eight values
    observed serving Kimi-Linear-48B at TP=8 (~140 KiB/block), a spread of
    158. Against an absolute tolerance of 4 this fails 100% of the time, so
    the model cannot be served at all, while Kimi-K3 (~14 MiB/block) passes
    on the same code because the identical jitter is only a few blocks."""
    observed = [551506, 551425, 551415, 551403, 551426, 551358, 551348, 551356]
    executor = _make_executor(engine_override=None, worker_overrides=observed)

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    # Smallest budget wins, so no worker is asked for more than it measured.
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 551348


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_absolute_floor_applies_to_tiny_block_counts(mock_super):
    """The relative tolerance must not shrink below the absolute floor: at
    100 blocks 0.5% rounds to 0, which would demand exact agreement."""
    executor = _make_executor(engine_override=None, worker_overrides=[100, 102, 101])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 100


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_no_override_when_workers_return_none(mock_super):
    """If no worker computed an override (uniform sizing skipped), the engine
    stays None and vLLM falls back to its own num_blocks computation."""
    executor = _make_executor(engine_override=None, worker_overrides=[None, None, None])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    assert executor.vllm_config.cache_config.num_gpu_blocks_override is None


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_pipeline_stages_may_disagree_and_the_smallest_stage_wins(mock_super):
    """Each pipeline stage holds different layers, so its workers measure a
    different count; vLLM sizes every stage's pool to the minimum."""
    executor = _make_executor(
        engine_override=None,
        worker_overrides=[100, 100, 100, 100] + [80, 80, 80, 80],
        pipeline_parallel_size=2,
    )

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 80


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_workers_within_a_pipeline_stage_must_still_agree(mock_super):
    executor = _make_executor(
        engine_override=None,
        worker_overrides=[100, 100, 100, 150] + [80, 80, 80, 80],
        pipeline_parallel_size=2,
    )

    with pytest.raises(ValueError, match="workers disagree"):
        executor.get_kv_cache_specs()


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_pipeline_stage_without_an_override_is_rejected(mock_super):
    """vLLM applies the override to every stage, so a stage that set none
    (no attention or no mamba layers in it) has no measured capacity for
    the value the others computed."""
    executor = _make_executor(
        engine_override=None,
        worker_overrides=[None, None] + [80, 81],
        pipeline_parallel_size=2,
    )

    with pytest.raises(ValueError, match=r"stage\(s\) \[0\] set no"):
        executor.get_kv_cache_specs()


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_pipeline_stages_without_any_override_leave_it_unset(mock_super):
    executor = _make_executor(
        engine_override=None,
        worker_overrides=[None, None, None, None],
        pipeline_parallel_size=2,
    )

    executor.get_kv_cache_specs()

    assert executor.vllm_config.cache_config.num_gpu_blocks_override is None


def _make_ray_executor(engine_override, worker_overrides):
    """Same shape as _make_executor but for the Ray multi-host executor."""
    from vllm_torchtpu.executors.ray_distributed_executor_v2 import (
        RayDistributedExecutorV2,
    )

    executor = RayDistributedExecutorV2.__new__(RayDistributedExecutorV2)
    executor.vllm_config = MagicMock()
    executor.vllm_config.cache_config = MagicMock(spec=CacheConfig)
    executor.vllm_config.cache_config.num_gpu_blocks_override = engine_override
    executor.vllm_config.parallel_config.world_size = len(worker_overrides)
    executor.vllm_config.parallel_config.pipeline_parallel_size = 1
    executor.collective_rpc = MagicMock(return_value=worker_overrides)
    return executor


def test_ray_executor_tolerates_measurement_jitter():
    """Regression for dev build 155: the Ray multi-host executor demanded
    EXACT agreement (assert len(set(overrides)) == 1) while the multiproc
    executor already tolerated jitter. Kimi-K3 at TP=32 landed 6456..6459
    (~14 MiB blocks, a 3-block spread from per-worker HBM measurement) and
    the whole engine start died on the bare assert."""
    from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

    observed = [6459, 6456] * 16
    executor = _make_ray_executor(engine_override=None, worker_overrides=observed)
    with patch.object(RayExecutorV2, "get_kv_cache_specs", return_value=_SPECS):
        specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    # Smallest budget wins, so no worker is asked for more than it measured.
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 6456


def test_ray_executor_raises_on_real_disagreement():
    from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

    executor = _make_ray_executor(
        engine_override=None, worker_overrides=[6456, 8000] * 16
    )
    with (
        patch.object(RayExecutorV2, "get_kv_cache_specs", return_value=_SPECS),
        pytest.raises(ValueError, match="workers disagree"),
    ):
        executor.get_kv_cache_specs()
