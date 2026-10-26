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
"""Reserve future cache growth for explicitly selected remote schedulers."""

from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request

from vllm_torchtpu.core.peak_reservation import can_admit, peak_blocks

logger = init_logger(__name__)


class ReservingManager:
    def __init__(
        self, manager: KVCacheManager, lookahead: int = 2, recycle_align: bool = False
    ) -> None:
        self.manager = manager
        self.lookahead = lookahead
        self.recycle_align = recycle_align
        self.admitted: dict[str, Request] = {}
        self.blocked = 0
        self.high_watermark = 0
        # Cache specifications are validated before the first allocation.
        logger.info(
            "REMOTE_CAPACITY_CONFIG pool_blocks=%d recycle_align=%s "
            "lookahead=%d specs=%s",
            manager.block_pool.num_gpu_blocks,
            recycle_align,
            lookahead,
            manager.kv_cache_config.kv_cache_groups,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.manager, name)

    def allocate_slots(self, request: Request, *args: Any, **kwargs: Any) -> Any:
        new = request.request_id not in self.admitted
        required = 0
        if new:
            required = peak_blocks(
                self.manager, request, self.lookahead, self.recycle_align
            )
            if required > self.manager.block_pool.num_gpu_blocks - 1:
                raise ValueError("request peak reservation exceeds physical pool")
            if not can_admit(
                self.manager,
                request,
                self.admitted.values(),
                self.lookahead,
                self.recycle_align,
            ):
                self.blocked += 1
                if self.blocked & (self.blocked - 1) == 0:
                    logger.info(
                        "REMOTE_CAPACITY_BLOCKED attempts=%d active=%d "
                        "new_peak=%d free=%d pool=%d specs=%s",
                        self.blocked,
                        len(self.admitted),
                        required,
                        self.manager.block_pool.get_num_free_blocks(),
                        self.manager.block_pool.num_gpu_blocks,
                        self.manager.kv_cache_config.kv_cache_groups,
                    )
                return None
        result = self.manager.allocate_slots(request, *args, **kwargs)
        if result is not None and new:
            self.admitted[request.request_id] = request
            if len(self.admitted) > self.high_watermark:
                self.high_watermark = len(self.admitted)
                logger.info(
                    "REMOTE_CAPACITY_ADMISSION active=%d peak_blocks=%d "
                    "free_blocks=%d blocked=%d",
                    self.high_watermark,
                    required,
                    self.manager.block_pool.get_num_free_blocks(),
                    self.blocked,
                )
        return result

    def free(self, request: Request) -> None:
        self.manager.free(request)
        self.admitted.pop(request.request_id, None)
        logger.info(
            "REMOTE_CAPACITY_RELEASE request=%s active=%d free=%d",
            request.request_id,
            len(self.admitted),
            self.manager.block_pool.get_num_free_blocks(),
        )

    def pop_blocks_for_free(self, request: Request) -> Any:
        result = self.manager.pop_blocks_for_free(request)
        self.admitted.pop(request.request_id, None)
        logger.info(
            "REMOTE_CAPACITY_RELEASE request=%s active=%d free=%d",
            request.request_id,
            len(self.admitted),
            self.manager.block_pool.get_num_free_blocks(),
        )
        return result
