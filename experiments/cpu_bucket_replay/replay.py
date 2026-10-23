# SPDX-License-Identifier: Apache-2.0
"""CPU source-function replay; no imports of vLLM, torch, or device runtime.

The unmodified upstream schedule() and token accounting run with capacity-
unlimited KV and output-metadata doubles. Completion is a synthetic FIFO;
the script measures allocations, not device forward counts or performance.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
import time
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
PLUGIN = ROOT.parents[1] / "src/vllm_torchtpu"
INPUTS = ROOT / "inputs"


def source_function(path: Path, name: str, **namespace: Any) -> Callable[..., Any]:
    tree = ast.parse(path.read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class RequestQueue(deque):
    add_request = deque.append
    pop_request = deque.popleft
    prepend_request = deque.appendleft
    prepend_requests = deque.extendleft

    def peek_request(self) -> Request:
        return self[0]


STATUS = SimpleNamespace(
    **{
        name: name
        for name in (
            "WAITING",
            "RUNNING",
            "PREEMPTED",
            "WAITING_FOR_REMOTE_KVS",
            "WAITING_FOR_STREAMING_REQ",
            "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR",
        )
    }
)
POLICY = SimpleNamespace(FCFS="fcfs", PRIORITY="priority")
PAUSE = SimpleNamespace(UNPAUSED=0, PAUSED_ALL=1)


class Request(SimpleNamespace):
    __hash__ = object.__hash__
    __eq__ = object.__eq__


class Blocks:
    def get_block_ids(self) -> tuple[list[int]]:
        return ([],)


class Cache:
    """Inject fixed observed hits; assume unlimited memory and no eviction."""

    def __init__(self, specs: dict[str, dict[str, Any]]) -> None:
        self.specs = specs
        self.empty_kv_cache_blocks = Blocks()

    def new_step_starts(self) -> None:
        pass

    def get_computed_blocks(self, request: Request) -> tuple[Blocks, int, int]:
        return Blocks(), self.specs[request.request_id]["cached_tokens"], 0

    def allocate_slots(self, *args: Any, **kwargs: Any) -> Blocks:
        return Blocks()

    def get_blocks(self, request_id: str) -> Blocks:
        return Blocks()

    def record_prefix_cache_stats(self, *args: Any) -> None:
        pass

    def get_num_common_prefix_blocks(self, request_id: str) -> list[int]:
        return [0]

    def take_boundary_state_offloads(self) -> dict[str, Any]:
        return {}

    def take_kv_cache_block_copies(self) -> tuple[list[Any], list[Any]]:
        return [], []

    def take_new_block_ids(self) -> list[int]:
        return []


class NewRequestData:
    @staticmethod
    def from_request(request: Request, *args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(req_id=request.request_id)


class SchedulerOutput(SimpleNamespace):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(has_structured_output_requests=False, **kwargs)


def scheduler_type(mtp: bool = True) -> type:
    namespace = {
        "time": time,
        "RequestStatus": STATUS,
        "SchedulingPolicy": POLICY,
        "PauseState": PAUSE,
        "create_request_queue": lambda policy: RequestQueue(),
        "record_function_or_nullcontext": lambda *args: nullcontext(),
        "NewRequestData": NewRequestData,
        "SchedulerOutput": SchedulerOutput,
    }
    names = [
        "schedule",
        "_update_after_schedule",
        "_mamba_block_aligned_split",
        "_reserve_prefill_lookahead",
        "_get_local_prefix_cache_hit",
        "_select_waiting_queue_for_scheduling",
        "_is_blocked_waiting_status",
        "_get_new_block_ids_to_zero",
    ]
    methods = {
        name: source_function(INPUTS / "vllm_scheduler.py.txt", name, **namespace)
        for name in names
    }
    methods["_is_blocked_waiting_status"] = staticmethod(
        methods["_is_blocked_waiting_status"]
    )
    physical = source_function(
        PLUGIN / "__init__.py",
        "split_with_scheduler_block_size",
        copy=copy,
        original_split=methods["_mamba_block_aligned_split"],
    )
    checkpoint = source_function(
        PLUGIN / "__init__.py",
        "split_at_mtp_prefix_boundary",
        envs=SimpleNamespace(
            TPU_MTP_PREFILL_ONLY=mtp,
            VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=True,
        ),
        original_split=physical,
    )
    methods["_mamba_block_aligned_split"] = source_function(
        PLUGIN / "core/eagle_block_drop.py",
        "split",
        original_split=checkpoint,
    )
    return type("SourceScheduler", (), methods)


def make_scheduler(
    specs: dict[str, dict[str, Any]], budget: int, mtp: bool = True
) -> Any:
    scheduler = scheduler_type(mtp)()
    scheduler.__dict__.update(
        current_step=0,
        max_num_scheduled_tokens=budget,
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(max_num_new_slots_for_drafting=0)
            if mtp
            else None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=budget,
            long_prefill_token_threshold=0,
            enable_chunked_prefill=True,
        ),
        cache_config=SimpleNamespace(block_size=384, enable_prefix_caching=True),
        block_size=3072,
        hash_block_size=3072,
        use_eagle=mtp,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_partial_cache_hit=False,
        _pause_state=PAUSE.UNPAUSED,
        max_num_encoder_input_tokens=0,
        kv_cache_manager=Cache(specs),
        prefill_capacity_bound=False,
        running=[],
        waiting=RequestQueue(),
        skipped_waiting=RequestQueue(),
        requests={},
        max_model_len=1048576,
        num_sampled_tokens_per_step=1,
        need_mamba_block_aligned_split=True,
        num_prefill_lookahead=int(mtp),
        num_lookahead_tokens=int(mtp),
        num_spec_tokens=int(mtp),
        dynamic_sd_lookup=None,
        lora_config=None,
        policy=POLICY.FCFS,
        num_waiting_for_streaming_input=0,
        max_num_running_reqs=64,
        connector=None,
        ec_connector=None,
        is_encoder_decoder=False,
        scheduler_reserve_full_isl=False,
        log_stats=False,
        _inflight_prefills=set(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[None]),
        use_v2_model_runner=False,
        model_uses_mrope=False,
        model_uses_xdrope=False,
        prev_step_scheduled_req_ids=set(),
        reset_preempted_req_ids=set(),
        finished_req_ids=set(),
        needs_kv_cache_zeroing=False,
        defer_block_free=False,
        enable_return_routed_experts=False,
        encoder_cache_manager=SimpleNamespace(
            get_freed_mm_hashes=lambda: [],
            get_manager_metadata=lambda: None,
        ),
    )
    scheduler._make_cached_request_data = lambda *args: SimpleNamespace(
        req_ids=[],
        new_block_ids=[],
    )
    return scheduler


def new_request(spec: dict[str, Any]) -> Request:
    return Request(
        request_id=spec["id"],
        num_prompt_tokens=spec["prompt_tokens"],
        num_tokens=spec["prompt_tokens"],
        num_tokens_with_spec=spec["prompt_tokens"],
        num_computed_tokens=0,
        num_output_placeholders=0,
        max_tokens=1,
        next_decode_eligible_step=0,
        is_prefill_chunk=True,
        has_encoder_inputs=False,
        spec_token_ids=[],
        status=STATUS.WAITING,
        num_stale_output_tokens=0,
        drop_stale_output=False,
        num_preemptions=0,
        prefill_stats=None,
        shared_prefix_boundary=0,
        num_in_flight_tokens=0,
        use_structured_output=False,
    )


def observed_intervals() -> list[float]:
    rows = json.loads((INPUTS / "observed_batches.json").read_text())
    intervals = []
    last = 0
    for row in rows:
        intervals.append(max(0.001, row["consumed_ms"] - max(last, row["submit_ms"])))
        last = row["consumed_ms"]
    return intervals


def run(
    mode: str, budget: int, local_buckets: list[int], mtp: bool = True
) -> dict[str, Any]:
    if mode not in {"closed_loop_c4", "historical_arrivals", "all_ready_16"}:
        raise ValueError(f"Unknown arrival mode: {mode}")
    workload = json.loads((INPUTS / "workload.json").read_text())["requests"]
    specs = {row["id"]: row for row in workload}
    scheduler = make_scheduler(specs, budget, mtp)
    batches, pending, completed = [], deque(), set()
    next_request, clock = 0, 0.0
    intervals = observed_intervals() if mode == "historical_arrivals" else [1.0]
    chunks = {row["id"]: [] for row in workload}
    active = set()
    # Synthetic 4-slot PP FIFO, not the device-side lazy-push state machine.
    for _ in range(100000):
        submitted = False
        while pending and pending[0]["due"] <= clock:
            done = pending.popleft()
            for rid, count in done["assigned"].items():
                scheduler.requests[rid].num_in_flight_tokens -= count
            for rid in done["final"]:
                request = scheduler.requests[rid]
                assert request.num_in_flight_tokens == 0
                scheduler.running.remove(request)
                completed.add(rid)
                active.remove(rid)
        if len(completed) == len(workload):
            break
        while next_request < len(workload):
            row = workload[next_request]
            if mode == "historical_arrivals" and row["arrival_ms"] > clock:
                break
            if mode == "closed_loop_c4" and len(active) >= 4:
                break
            request = new_request(row)
            scheduler.requests[row["id"]] = request
            scheduler.waiting.add_request(request)
            active.add(row["id"])
            next_request += 1
        if len(pending) < 4 and active:
            before = {
                "running": [r.request_id for r in scheduler.running],
                "waiting": [r.request_id for r in scheduler.waiting],
            }
            split_calls = []
            real_split = scheduler._mamba_block_aligned_split

            def traced_split(
                request,
                tokens,
                local=0,
                external=0,
                original=real_split,
                calls=split_calls,
            ):
                result = original(request, tokens, local, external)
                calls.append(
                    {
                        "request": request.request_id,
                        "start": request.num_computed_tokens + local + external,
                        "offered": tokens,
                        "allocated": result,
                    }
                )
                return result

            scheduler._mamba_block_aligned_split = traced_split
            output = scheduler.schedule()
            scheduler._mamba_block_aligned_split = real_split
            assigned = dict(output.num_scheduled_tokens)
            if assigned:
                submitted = True
                total = sum(assigned.values())
                assert total == output.total_num_scheduled_tokens <= budget
                for rid, count in assigned.items():
                    assert count > 0
                    chunks[rid].append(count)
                final = {
                    rid
                    for rid in assigned
                    if scheduler.requests[rid].num_computed_tokens
                    == specs[rid]["prompt_tokens"]
                }
                duration = intervals[min(len(batches), len(intervals) - 1)]
                due = max(clock, pending[-1]["due"] if pending else clock) + duration
                batches.append(
                    {
                        "batch": len(batches) + 1,
                        "synthetic_submit_ms": clock,
                        "assigned": assigned,
                        "tokens": total,
                        "unused_budget": budget - total,
                        "queues_before": before,
                        "split_calls": split_calls,
                    }
                )
                pending.append({"due": due, "assigned": assigned, "final": final})
        future = [row["due"] for row in pending if row["due"] > clock]
        if mode == "historical_arrivals" and next_request < len(workload):
            future.append(workload[next_request]["arrival_ms"])
        if len(pending) < 4 and active and submitted:
            future.append(clock + (0.001 if mode != "historical_arrivals" else 1.0))
        if not future:
            raise AssertionError("No progress with unfinished requests")
        clock = min(future)
    else:
        raise AssertionError("Replay did not finish")
    for row in workload:
        assert sum(chunks[row["id"]]) == row["prompt_tokens"] - row["cached_tokens"]
    total = sum(b["tokens"] for b in batches)
    assert total == 190749
    assert len(completed) == 16
    assert not any(
        n.split(".")[0] in {"torch", "torch_tpu", "libtpu"} for n in sys.modules
    )
    return {
        "mode": mode,
        "compile_sizes_per_pcp_rank": local_buckets,
        "global_token_budget": budget,
        "mtp_checkpoint": mtp,
        "batch_count": len(batches),
        "request_chunk_count": sum(map(len, chunks.values())),
        "computed_tokens": total,
        "mean_tokens_per_batch": total / len(batches),
        "budget_utilization": total / (len(batches) * budget),
        "small_batches_le_1024": sum(b["tokens"] <= 1024 for b in batches),
        "per_request": chunks,
        "batches": batches,
    }


def verify_inputs() -> None:
    provenance = json.loads((INPUTS / "provenance.json").read_text())
    for filename, expected in provenance["sha256"].items():
        actual = hashlib.sha256((INPUTS / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Fixture SHA256 mismatch: {filename}")


def main() -> None:
    verify_inputs()
    results = []
    for mode in ["closed_loop_c4", "historical_arrivals", "all_ready_16"]:
        for budget, buckets in [
            (32768, [4096]),
            (8192, [256, 512, 1024]),
        ]:
            results.append(run(mode, budget, buckets))
    metadata = json.loads((INPUTS / "provenance.json").read_text())
    metadata["plugin_sources"] = {
        str(path.relative_to(PLUGIN)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [PLUGIN / "__init__.py", PLUGIN / "core/eagle_block_drop.py"]
    }
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(
            {
                "provenance": metadata,
                "results": results,
                "limitations": [
                    "Exact schedule/split/accounting functions; "
                    "initialization and KV/output metadata are doubles",
                    "Fixed observed cache hits; no KV capacity pressure, "
                    "shared-prefix boundary or cache warming",
                    "Four-slot synthetic FIFO; lazy push, TPU/MTP execution "
                    "and real KV transfer are not modeled",
                    "Historical intervals are a sensitivity assumption, "
                    "not a performance prediction",
                    "Compile sizes do not enter schedule(); "
                    "global token budget controls chunk capacity",
                    "Batch counts are scheduler submissions, "
                    "not measured TPU forward or MTP draft counts",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    for result in results:
        name = f"{result['mode']}-{result['global_token_budget'] // 8}"
        lines = [name, "batch | total valid tokens | per-request allocation"]
        for batch in result["batches"]:
            allocation = ", ".join(f"{r}:{n}" for r, n in batch["assigned"].items())
            lines.append(f"{batch['batch']:2} | {batch['tokens']:5} | {allocation}")
        lines.extend(["", "request | chunk tokens"])
        lines.extend(
            f"{rid} | {' + '.join(map(str, values))}"
            for rid, values in result["per_request"].items()
        )
        (output / f"{name}.txt").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            [
                {k: v for k, v in row.items() if k not in {"batches", "per_request"}}
                for row in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
