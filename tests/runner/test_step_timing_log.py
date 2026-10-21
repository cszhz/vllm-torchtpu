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

from unittest.mock import patch

from vllm_torchtpu.runner import step_timing_log
from vllm_torchtpu.runner.step_timing_log import StepTimingLog


def test_periods_grouped_by_the_opening_steps_bucket():
    log = StepTimingLog(rank=3, log_every_steps=3)
    starts = iter([0.0, 0.05, 0.12])
    lines = []
    with (
        patch.object(step_timing_log.time, "perf_counter", lambda: next(starts)),
        patch.object(
            step_timing_log.logger, "info", lambda fmt, *a: lines.append(fmt % a)
        ),
    ):
        for bucket in (64, 128, 64):
            log.step_start()
            log.result_wait(0.001)
            log.dp_sync(bucket, 0.002)
    # The 3rd step's period is still open when the 3rd step flushes.
    (line,) = lines
    assert line == (
        "step timing rank=3 steps=2 period_ms mean=60.0 p50=60.0 | "
        "dp_sync_ms mean=2.00 p90=2.00 | result_wait_ms mean=1.00 p10=1.00 | "
        "T64: n=1 (50%) mean=50.0 p50=50.0 p90=50.0 | "
        "T128: n=1 (50%) mean=70.0 p50=70.0 p90=70.0"
    )


def test_idle_gaps_are_not_steps():
    log = StepTimingLog(rank=0, log_every_steps=100)
    starts = iter([0.0, 10.0])
    with patch.object(step_timing_log.time, "perf_counter", lambda: next(starts)):
        log.step_start()
        log.dp_sync(64, 0.0)
        log.step_start()
    assert not log._periods
