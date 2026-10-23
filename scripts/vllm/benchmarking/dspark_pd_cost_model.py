#!/usr/bin/env python3
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
"""Estimate the balance of remote DSpark drafting in a P/D pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import json
from typing import Any


@dataclasses.dataclass(frozen=True)
class Inputs:
    context_update_ms: float
    draft_generate_ms: float
    verify_ms: float
    batch_size: int
    proposal_tokens_per_request: int
    feedback_tokens_per_request: int
    feedback_layer_count: int
    hidden_size: int
    feedback_dtype_bytes: int
    token_id_bytes: int
    metadata_bytes_per_request: int
    p2d_latency_us: float
    p2d_bandwidth_gbps: float
    d2p_latency_us: float
    d2p_bandwidth_gbps: float
    mean_accepted_tokens: float


@dataclasses.dataclass(frozen=True)
class StageInterval:
    stage: str
    request_index: int
    round_id: int
    start_ms: float
    end_ms: float


def simulate_pipeline(
    *,
    p_stage_ms: float,
    d_stage_ms: float,
    num_requests: int,
    rounds_per_request: int,
) -> dict[str, Any]:
    """Schedule dependent P/D rounds on two independent workers.

    Requests are FIFO within each stage. A request's D round starts after its
    P round; its next P round starts after that D feedback. Independent
    requests may occupy the other stage during either wait.
    """
    positive = {
        "p_stage_ms": p_stage_ms,
        "d_stage_ms": d_stage_ms,
        "num_requests": num_requests,
        "rounds_per_request": rounds_per_request,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    sequence = 0
    ready: list[tuple[float, int, str, int, int]] = []
    for request_index in range(num_requests):
        heapq.heappush(ready, (0.0, sequence, "p", request_index, 0))
        sequence += 1
    available_at = {"p": 0.0, "d": 0.0}
    intervals: list[StageInterval] = []

    while ready:
        dependency_ready, _, stage, request_index, round_id = heapq.heappop(ready)
        start_ms = max(dependency_ready, available_at[stage])
        duration_ms = p_stage_ms if stage == "p" else d_stage_ms
        end_ms = start_ms + duration_ms
        available_at[stage] = end_ms
        intervals.append(
            StageInterval(stage, request_index, round_id, start_ms, end_ms)
        )
        if stage == "p":
            next_stage = "d"
            next_round = round_id
        elif round_id + 1 < rounds_per_request:
            next_stage = "p"
            next_round = round_id + 1
        else:
            continue
        heapq.heappush(
            ready,
            (end_ms, sequence, next_stage, request_index, next_round),
        )
        sequence += 1

    p_intervals = [interval for interval in intervals if interval.stage == "p"]
    d_intervals = [interval for interval in intervals if interval.stage == "d"]
    overlap_ms = _overlap_ms(p_intervals, d_intervals)
    makespan_ms = max(interval.end_ms for interval in intervals)
    colocated_ms = num_requests * rounds_per_request * (p_stage_ms + d_stage_ms)
    return {
        "makespan_ms": makespan_ms,
        "colocated_ms": colocated_ms,
        "overlap_ms": overlap_ms,
        "speedup_vs_colocated": colocated_ms / makespan_ms,
        "p_utilization": sum(
            interval.end_ms - interval.start_ms for interval in p_intervals
        )
        / makespan_ms,
        "d_utilization": sum(
            interval.end_ms - interval.start_ms for interval in d_intervals
        )
        / makespan_ms,
        "timeline": [dataclasses.asdict(interval) for interval in intervals],
    }


def _overlap_ms(
    p_intervals: list[StageInterval], d_intervals: list[StageInterval]
) -> float:
    p_intervals.sort(key=lambda interval: interval.start_ms)
    d_intervals.sort(key=lambda interval: interval.start_ms)
    p_index = 0
    d_index = 0
    overlap = 0.0
    while p_index < len(p_intervals) and d_index < len(d_intervals):
        p_interval = p_intervals[p_index]
        d_interval = d_intervals[d_index]
        overlap += max(
            0.0,
            min(p_interval.end_ms, d_interval.end_ms)
            - max(p_interval.start_ms, d_interval.start_ms),
        )
        if p_interval.end_ms <= d_interval.end_ms:
            p_index += 1
        else:
            d_index += 1
    return overlap


def _transfer_ms(payload_bytes: int, latency_us: float, bandwidth_gbps: float) -> float:
    return latency_us / 1_000.0 + payload_bytes * 8.0 / (bandwidth_gbps * 1e6)


def evaluate(inputs: Inputs) -> dict[str, Any]:
    """Return payload, timing, balance, and throughput estimates.

    The pipeline model assumes enough independent request microbatches to keep
    P and D busy. It does not assume that consecutive rounds of one request can
    overlap: round n+1 still depends on round n's target hidden states.
    """
    _validate(inputs)

    request_metadata_bytes = inputs.batch_size * inputs.metadata_bytes_per_request
    p2d_payload_bytes = (
        inputs.batch_size * inputs.proposal_tokens_per_request * inputs.token_id_bytes
        + request_metadata_bytes
    )
    d2p_payload_bytes = (
        inputs.batch_size
        * inputs.feedback_tokens_per_request
        * inputs.feedback_layer_count
        * inputs.hidden_size
        * inputs.feedback_dtype_bytes
        + request_metadata_bytes
    )
    p2d_transfer_ms = _transfer_ms(
        p2d_payload_bytes, inputs.p2d_latency_us, inputs.p2d_bandwidth_gbps
    )
    d2p_transfer_ms = _transfer_ms(
        d2p_payload_bytes, inputs.d2p_latency_us, inputs.d2p_bandwidth_gbps
    )

    p_stage_ms = inputs.context_update_ms + inputs.draft_generate_ms + p2d_transfer_ms
    d_stage_ms = inputs.verify_ms + d2p_transfer_ms
    pipeline_cycle_ms = max(p_stage_ms, d_stage_ms)
    split_request_round_ms = p_stage_ms + d_stage_ms
    colocated_round_ms = (
        inputs.context_update_ms + inputs.draft_generate_ms + inputs.verify_ms
    )
    accepted_tokens_per_microbatch = inputs.batch_size * inputs.mean_accepted_tokens
    p_utilization = p_stage_ms / pipeline_cycle_ms
    d_utilization = d_stage_ms / pipeline_cycle_ms
    throughput_speedup = colocated_round_ms / pipeline_cycle_ms

    return {
        "assumptions": {
            "independent_microbatches_available": True,
            "same_request_rounds_overlap": False,
            "bandwidth_unit": "decimal_gigabits_per_second",
        },
        "payload_bytes": {
            "p_to_d": p2d_payload_bytes,
            "d_to_p": d2p_payload_bytes,
        },
        "transfer_ms": {
            "p_to_d": p2d_transfer_ms,
            "d_to_p": d2p_transfer_ms,
        },
        "stage_ms": {
            "p": p_stage_ms,
            "d": d_stage_ms,
            "pipeline_cycle": pipeline_cycle_ms,
            "split_single_request_round": split_request_round_ms,
            "colocated_round": colocated_round_ms,
        },
        "balance": {
            "p_to_d_stage_ratio": p_stage_ms / d_stage_ms,
            "bottleneck": "p" if p_stage_ms >= d_stage_ms else "d",
            "p_utilization": p_utilization,
            "d_utilization": d_utilization,
        },
        "performance": {
            "accepted_tokens_per_microbatch": accepted_tokens_per_microbatch,
            "pipeline_accepted_tokens_per_second": (
                accepted_tokens_per_microbatch * 1_000.0 / pipeline_cycle_ms
            ),
            "pipeline_tpot_ms_per_request": (
                pipeline_cycle_ms / inputs.mean_accepted_tokens
            ),
            "split_single_request_tpot_ms": (
                split_request_round_ms / inputs.mean_accepted_tokens
            ),
            "colocated_tpot_ms_per_request": (
                colocated_round_ms / inputs.mean_accepted_tokens
            ),
            "pipeline_throughput_speedup_vs_colocated": throughput_speedup,
            "two_slice_efficiency_vs_one_colocated_slice": throughput_speedup / 2.0,
            "overlap_speedup_vs_unpipelined_split": (
                split_request_round_ms / pipeline_cycle_ms
            ),
        },
    }


def _validate(inputs: Inputs) -> None:
    positive = {
        "batch_size": inputs.batch_size,
        "proposal_tokens_per_request": inputs.proposal_tokens_per_request,
        "feedback_tokens_per_request": inputs.feedback_tokens_per_request,
        "feedback_layer_count": inputs.feedback_layer_count,
        "hidden_size": inputs.hidden_size,
        "feedback_dtype_bytes": inputs.feedback_dtype_bytes,
        "token_id_bytes": inputs.token_id_bytes,
        "p2d_bandwidth_gbps": inputs.p2d_bandwidth_gbps,
        "d2p_bandwidth_gbps": inputs.d2p_bandwidth_gbps,
        "mean_accepted_tokens": inputs.mean_accepted_tokens,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    nonnegative = {
        "context_update_ms": inputs.context_update_ms,
        "draft_generate_ms": inputs.draft_generate_ms,
        "verify_ms": inputs.verify_ms,
        "metadata_bytes_per_request": inputs.metadata_bytes_per_request,
        "p2d_latency_us": inputs.p2d_latency_us,
        "d2p_latency_us": inputs.d2p_latency_us,
    }
    for name, value in nonnegative.items():
        if value < 0:
            raise ValueError(f"{name} must be nonnegative, got {value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-update-ms", type=float, required=True)
    parser.add_argument("--draft-generate-ms", type=float, required=True)
    parser.add_argument("--verify-ms", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--proposal-tokens-per-request", type=int, required=True)
    parser.add_argument("--feedback-tokens-per-request", type=int, required=True)
    parser.add_argument("--feedback-layer-count", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--feedback-dtype-bytes", type=int, required=True)
    parser.add_argument("--token-id-bytes", type=int, default=4)
    parser.add_argument("--metadata-bytes-per-request", type=int, default=16)
    parser.add_argument("--p2d-latency-us", type=float, required=True)
    parser.add_argument("--p2d-bandwidth-gbps", type=float, required=True)
    parser.add_argument("--d2p-latency-us", type=float, required=True)
    parser.add_argument("--d2p-bandwidth-gbps", type=float, required=True)
    parser.add_argument("--mean-accepted-tokens", type=float, required=True)
    parser.add_argument("--simulate-requests", type=int, default=0)
    parser.add_argument("--simulate-rounds", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inputs = Inputs(
        **{
            field.name: getattr(args, field.name)
            for field in dataclasses.fields(Inputs)
        }
    )
    result = evaluate(inputs)
    if args.simulate_requests:
        result["simulation"] = simulate_pipeline(
            p_stage_ms=result["stage_ms"]["p"],
            d_stage_ms=result["stage_ms"]["d"],
            num_requests=args.simulate_requests,
            rounds_per_request=args.simulate_rounds,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
