#!/usr/bin/env python3
"""Exercise real decode-side KV capacity preemption through the OpenAI API."""

import argparse
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any, float]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=request_headers,
        method="POST",
    )
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        try:
            parsed: Any = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw_body": body}
        return error.code, parsed, time.monotonic() - started
    with response:
        body = response.read().decode()
        return response.status, json.loads(body), time.monotonic() - started


def get_text(url: str, timeout: int) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def metric_sum(text: str, name: str, required_label: str | None = None) -> float:
    total = 0.0
    found = False
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if not (line.startswith(f"{name}{{") or line.startswith(f"{name} ")):
            continue
        if required_label is not None and required_label not in line:
            continue
        try:
            total += float(line.rsplit(maxsplit=1)[1])
        except (IndexError, ValueError) as error:
            raise AssertionError(f"Malformed metric line: {line}") from error
        found = True
    if not found:
        label = f" with {required_label}" if required_label else ""
        raise AssertionError(f"Metric {name}{label} was not exported")
    return total


def write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))


def validate_decode_geometry(path: Path, num_blocks: int, block_size: int) -> None:
    text = path.read_text(errors="replace")
    layouts = re.findall(r"num_blocks=(\d+) num_groups=(\d+) num_raw_tensors=\d+", text)
    expected_layout = (str(num_blocks), "4")
    if not layouts or any(layout != expected_layout for layout in layouts):
        raise AssertionError(
            "Unexpected decode KV layout: "
            f"expected={expected_layout}, observed={layouts}, log={path}"
        )
    mamba_groups = re.findall(
        rf"group\[\d+\]: spec=MambaSpec .* block_size={block_size}(?: |$)",
        text,
    )
    attention_groups = re.findall(
        rf"group\[\d+\]: spec=FullAttentionSpec .* block_size={block_size}(?: |$)",
        text,
    )
    if len(mamba_groups) != 3 * len(layouts) or len(attention_groups) != len(layouts):
        raise AssertionError(
            "Capacity geometry requires exactly three Mamba groups and one "
            "full-attention group per decode worker: "
            f"layouts={len(layouts)}, mamba={len(mamba_groups)}, "
            f"full_attention={len(attention_groups)}, log={path}"
        )


def render_prompt_ids(
    base_url: str, model: str, target: int, timeout: int
) -> list[int]:
    # Render only a short seed: the render endpoint also enforces max_model_len.
    # Expand token IDs locally to obtain the exact capacity-boundary length.
    fixture = (
        "Continue the token pattern: alpha beta gamma delta epsilon zeta eta theta. "
    )
    status, body, _ = post_json(
        f"{base_url}/v1/completions/render",
        {"model": model, "prompt": fixture},
        timeout,
    )
    if status != 200 or not isinstance(body, list) or not body:
        raise AssertionError(f"Prompt render failed: status={status}, body={body}")
    token_ids = body[0].get("token_ids") if isinstance(body[0], dict) else None
    if not isinstance(token_ids, list) or not token_ids:
        raise AssertionError(f"Prompt render returned no token IDs: {body}")
    return (token_ids * ((target + len(token_ids) - 1) // len(token_ids)))[:target]


def request_payload(
    model: str, prompt_ids: list[int], max_tokens: int, request_id: str
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": False,
        "request_id": request_id,
        # Prefix caching is required by the hybrid Mamba layout. A per-request
        # salt keeps the two identical prompts from sharing full-attention
        # blocks, while using the same salt on P and D preserves the transfer.
        "cache_salt": request_id,
    }


def prefill(
    base_url: str,
    model: str,
    prompt_ids: list[int],
    request_id: str,
    timeout: int,
) -> dict[str, Any]:
    payload = request_payload(model, prompt_ids, 1, request_id)
    status, body, elapsed = post_json(
        f"{base_url}/v1/completions",
        payload,
        timeout,
        {"X-Request-Id": request_id},
    )
    if status != 200 or not isinstance(body, dict):
        raise AssertionError(
            f"Prefill failed for {request_id}: status={status}, body={body}"
        )
    kv_transfer_params = body.get("kv_transfer_params")
    if not isinstance(kv_transfer_params, dict) or not kv_transfer_params:
        raise AssertionError(
            f"Prefill returned no KV transfer handle for {request_id}: {body}"
        )
    expected_transfer_tokens = len(prompt_ids) - 1
    if kv_transfer_params.get("num_tokens") != expected_transfer_tokens:
        raise AssertionError(
            "Prefill handle did not publish the expected remote prefix: "
            f"request_id={request_id}, expected={expected_transfer_tokens}, "
            f"handle={kv_transfer_params}"
        )
    print(f"PREFILL_OK request_id={request_id} elapsed_s={elapsed:.3f}", flush=True)
    return kv_transfer_params


def decode(
    base_url: str,
    model: str,
    prompt_ids: list[int],
    kv_transfer_params: dict[str, Any],
    request_id: str,
    max_tokens: int,
    timeout: int,
    barrier: threading.Barrier | None = None,
) -> dict[str, Any]:
    payload = request_payload(model, prompt_ids, max_tokens, request_id)
    payload["kv_transfer_params"] = kv_transfer_params
    if barrier is not None:
        barrier.wait(timeout=timeout)
    status, body, elapsed = post_json(
        f"{base_url}/v1/completions",
        payload,
        timeout,
        {
            "X-Request-Id": request_id,
            "X-Data-Parallel-Rank": "0",
        },
    )
    return {
        "request_id": request_id,
        "status": status,
        "body": body,
        "elapsed_s": round(elapsed, 3),
    }


def success_token_ids(
    result: dict[str, Any], max_tokens: int, expected_cached_tokens: int
) -> list[int]:
    if result["status"] != 200:
        raise AssertionError(f"Expected success: {result}")
    body = result["body"]
    try:
        choice = body["choices"][0]
        token_ids = choice["token_ids"]
        completion_tokens = body["usage"]["completion_tokens"]
        cached_tokens = body["usage"]["prompt_tokens_details"]["cached_tokens"]
    except (KeyError, IndexError, TypeError) as error:
        raise AssertionError(f"Malformed success response: {result}") from error
    if choice.get("finish_reason") != "length":
        raise AssertionError(f"Expected finish_reason=length: {result}")
    if completion_tokens != max_tokens or len(token_ids) != max_tokens:
        raise AssertionError(f"Expected exactly {max_tokens} output tokens: {result}")
    if cached_tokens != expected_cached_tokens:
        raise AssertionError(
            "Decode did not reuse the producer's remote prefix: "
            f"expected_cached_tokens={expected_cached_tokens}, result={result}"
        )
    return token_ids


def abort_log_lines(path: Path, request_id: str) -> list[str]:
    text = path.read_text(errors="replace") if path.exists() else ""
    return [
        line
        for line in text.splitlines()
        if request_id in line and "error=model.KVCachePreempt" in line
    ]


def wait_for_abort_log(path: Path, request_id: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while True:
        matches = abort_log_lines(path, request_id)
        if len(matches) == 1:
            return
        if len(matches) > 1:
            raise AssertionError(
                f"Capacity victim logged duplicate abort terminals: {matches}"
            )
        if time.monotonic() >= deadline:
            raise AssertionError(
                "Decode log did not provide the capacity-abort evidence chain: "
                f"request_id={request_id}, error=model.KVCachePreempt, log={path}"
            )
        time.sleep(0.2)


def receiver_complete_records(path: Path) -> list[dict[str, Any]]:
    records = []
    text = path.read_text(errors="replace") if path.exists() else ""
    for line in text.splitlines():
        if '"event": "raiden_stage3_receiver_complete"' not in line:
            continue
        try:
            record = json.loads(line[line.index("{") :])
        except (ValueError, json.JSONDecodeError):
            continue
        records.append(record)
    return records


def wait_for_pd_load_logs(
    path: Path, handles: list[dict[str, Any]], expected_tokens: int, timeout: int
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while True:
        records = receiver_complete_records(path)
        matches = []
        missing = []
        for handle in handles:
            match = next(
                (
                    record
                    for record in records
                    if record.get("req_id") == handle["req_id"]
                    and record.get("uuid") == handle["uuid"]
                    and record.get("num_tokens") == expected_tokens
                ),
                None,
            )
            if match is None:
                missing.append(
                    {
                        "req_id": handle["req_id"],
                        "uuid": handle["uuid"],
                    }
                )
            else:
                matches.append(match)
        if not missing:
            return matches
        if time.monotonic() >= deadline:
            raise AssertionError(
                "Decode log did not prove completed remote Stage-3 loads: "
                f"missing={missing}, expected_tokens={expected_tokens}, log={path}"
            )
        time.sleep(0.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("P4D2_BIND_HOST", "127.0.0.1"))
    parser.add_argument(
        "--prefill-port", type=int, default=int(os.environ.get("PREFILL_PORT", "8400"))
    )
    parser.add_argument(
        "--decode-port", type=int, default=int(os.environ.get("DECODE_PORT", "9400"))
    )
    parser.add_argument(
        "--model", default=os.environ.get("SERVED_MODEL_NAME", "Qwen3.5-35B-A3B-FP8")
    )
    parser.add_argument("--expect", choices=("recompute", "abort"), required=True)
    parser.add_argument("--decode-block-size", type=int, required=True)
    parser.add_argument("--num-gpu-blocks", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-margin", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--decode-log", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_prompt_tokens = args.decode_block_size - args.prompt_margin
    if target_prompt_tokens <= 0:
        raise SystemExit("prompt-margin must be smaller than decode-block-size")
    if target_prompt_tokens + args.max_tokens <= args.decode_block_size:
        raise SystemExit("prompt + output must cross a decode KV block boundary")

    prefill_url = f"http://{args.host}:{args.prefill_port}"
    decode_url = f"http://{args.host}:{args.decode_port}"
    validate_decode_geometry(
        args.decode_log, args.num_gpu_blocks, args.decode_block_size
    )
    prompt_ids = render_prompt_ids(
        prefill_url, args.model, target_prompt_tokens, args.timeout
    )
    metrics_before = get_text(f"{decode_url}/metrics", args.timeout)
    preemptions_before = metric_sum(metrics_before, "vllm:num_preemptions_total")
    draft_tokens_before = metric_sum(
        metrics_before, "vllm:spec_decode_num_draft_tokens_total"
    )
    evidence: dict[str, Any] = {
        "expect": args.expect,
        "prompt_tokens": target_prompt_tokens,
        "max_tokens": args.max_tokens,
        "num_gpu_blocks": args.num_gpu_blocks,
        "metrics_before": metrics_before,
    }
    write_evidence(args.result_path, evidence)

    run_id = f"capacity-{args.expect}-{os.getpid()}-{time.time_ns()}"
    pressure_ids = [f"{run_id}-a", f"{run_id}-b"]
    handles = [
        prefill(prefill_url, args.model, prompt_ids, request_id, args.timeout)
        for request_id in pressure_ids
    ]
    evidence["prefill_handles"] = handles
    write_evidence(args.result_path, evidence)
    barrier = threading.Barrier(len(pressure_ids))
    with ThreadPoolExecutor(max_workers=len(pressure_ids)) as pool:
        futures = [
            pool.submit(
                decode,
                decode_url,
                args.model,
                prompt_ids,
                handle,
                request_id,
                args.max_tokens,
                args.timeout,
                barrier,
            )
            for request_id, handle in zip(pressure_ids, handles, strict=True)
        ]
        pressure = [future.result() for future in futures]
    evidence["pressure"] = pressure
    write_evidence(args.result_path, evidence)
    metrics_after_pressure = get_text(f"{decode_url}/metrics", args.timeout)
    evidence["metrics_after_pressure"] = metrics_after_pressure
    write_evidence(args.result_path, evidence)
    evidence["pressure_receiver_complete"] = wait_for_pd_load_logs(
        args.decode_log, handles, target_prompt_tokens - 1, 10
    )
    write_evidence(args.result_path, evidence)

    successes = [result for result in pressure if result["status"] == 200]
    failures = [result for result in pressure if result["status"] != 200]
    if args.expect == "recompute":
        if len(successes) != 2 or failures:
            raise AssertionError(
                f"Default policy must recompute both requests: {pressure}"
            )
    else:
        if len(successes) != 1 or len(failures) != 1:
            raise AssertionError(
                "Abort policy must fail one victim and preserve one survivor: "
                f"{pressure}"
            )
        if failures[0]["status"] != 500:
            raise AssertionError(
                f"Capacity victim must end with HTTP 500: {failures[0]}"
            )
        wait_for_abort_log(args.decode_log, failures[0]["request_id"], 10)

    running_after_pressure = metric_sum(
        metrics_after_pressure, "vllm:num_requests_running"
    )
    waiting_after_pressure = metric_sum(
        metrics_after_pressure, "vllm:num_requests_waiting"
    )
    if running_after_pressure != 0 or waiting_after_pressure != 0:
        raise AssertionError(
            "Pressure requests left scheduler state behind: "
            f"running={running_after_pressure}, waiting={waiting_after_pressure}"
        )

    success_tokens = [
        success_token_ids(result, args.max_tokens, target_prompt_tokens - 1)
        for result in successes
    ]
    if len(success_tokens) == 2 and success_tokens[0] != success_tokens[1]:
        raise AssertionError("The two recomputed baseline responses diverged")

    recovery_id = f"{run_id}-recovery"
    recovery_handle = prefill(
        prefill_url, args.model, prompt_ids, recovery_id, args.timeout
    )
    recovery = decode(
        decode_url,
        args.model,
        prompt_ids,
        recovery_handle,
        recovery_id,
        args.max_tokens,
        args.timeout,
    )
    evidence["recovery"] = recovery
    write_evidence(args.result_path, evidence)
    evidence["recovery_receiver_complete"] = wait_for_pd_load_logs(
        args.decode_log, [recovery_handle], target_prompt_tokens - 1, 10
    )
    write_evidence(args.result_path, evidence)
    recovery_tokens = success_token_ids(
        recovery, args.max_tokens, target_prompt_tokens - 1
    )
    if success_tokens[0] != recovery_tokens:
        raise AssertionError(
            "Survivor output differs from the post-pressure recovery request"
        )

    metrics_after = get_text(f"{decode_url}/metrics", args.timeout)
    preemptions_after = metric_sum(metrics_after, "vllm:num_preemptions_total")
    draft_tokens_after = metric_sum(
        metrics_after, "vllm:spec_decode_num_draft_tokens_total"
    )
    preemption_delta = preemptions_after - preemptions_before
    draft_tokens_delta = draft_tokens_after - draft_tokens_before
    if preemption_delta < 1:
        raise AssertionError(
            "The pressure pair did not trigger real scheduler preemption: "
            f"delta={preemption_delta}"
        )
    if draft_tokens_delta < 1:
        raise AssertionError(
            "MTP produced no draft tokens during the capacity scenario: "
            f"delta={draft_tokens_delta}"
        )
    if args.expect == "abort":
        matches = abort_log_lines(args.decode_log, failures[0]["request_id"])
        if len(matches) != 1:
            raise AssertionError(
                f"Capacity victim must have exactly one abort terminal: {matches}"
            )
    else:
        unexpected_abort_lines = [
            line
            for request_id in (*pressure_ids, recovery_id)
            for line in abort_log_lines(args.decode_log, request_id)
        ]
        if unexpected_abort_lines:
            raise AssertionError(
                "Default policy emitted capacity-abort terminals: "
                f"{unexpected_abort_lines}"
            )

    evidence.update(
        {
            "metrics_after": metrics_after,
            "preemption_delta": preemption_delta,
            "draft_tokens_delta": draft_tokens_delta,
            "running_after_pressure": running_after_pressure,
            "waiting_after_pressure": waiting_after_pressure,
        }
    )
    write_evidence(args.result_path, evidence)
    print(f"CAPACITY_PREEMPTION_TRIGGERED delta={preemption_delta:g}")
    print(f"CAPACITY_PREEMPTION_MTP_OK draft_tokens_delta={draft_tokens_delta:g}")
    print("CAPACITY_PREEMPTION_DRAINED running=0 waiting=0")
    print(f"CAPACITY_PREEMPTION_SURVIVOR_OK request_id={successes[0]['request_id']}")
    print(f"CAPACITY_PREEMPTION_RECOVERY_OK request_id={recovery_id}")
    if args.expect == "abort":
        print(f"CAPACITY_PREEMPTION_ABORT_OK request_id={failures[0]['request_id']}")
    else:
        print("CAPACITY_PREEMPTION_RECOMPUTE_BASELINE_OK requests=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
