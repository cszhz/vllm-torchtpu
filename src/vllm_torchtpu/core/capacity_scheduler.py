# Copyright 2026 Google LLC
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
"""Explicit experiment scheduler classes, never selected by default."""

from typing import Any, cast

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core.remote_draft_scheduler import (
    RemoteDraftConsumerScheduler,
    RemoteDraftScheduler,
)
from vllm_torchtpu.core.reserving_manager import ReservingManager


def install_reservation(scheduler: Scheduler) -> None:
    if scheduler.scheduler_config.async_scheduling:
        raise NotImplementedError("capacity reservation requires sync scheduling")
    if scheduler.kv_cache_manager.enable_caching:
        raise NotImplementedError("capacity reservation requires prefix caching off")
    scheduler.kv_cache_manager = cast(
        KVCacheManager,
        ReservingManager(
            scheduler.kv_cache_manager,
            lookahead=1 + scheduler.num_spec_tokens + scheduler.num_lookahead_tokens,
            recycle_align=True,
        ),
    )
    if scheduler.connector is not None:
        scheduler.connector.bind_kv_cache_manager(scheduler.kv_cache_manager)


class CapacityRemoteDraftScheduler(RemoteDraftScheduler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        install_reservation(self)


class CapacityRemoteConsumerScheduler(RemoteDraftConsumerScheduler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        install_reservation(self)
