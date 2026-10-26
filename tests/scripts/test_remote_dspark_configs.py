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
"""Validate opt-in launch arguments without loading model weights or a device."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.mark.parametrize("status,expected", [("complete", 0), ("error", 1)])
def test_pair_lifecycle_gate_rejects_proxy_finalization_error(
    tmp_path, status, expected
):
    root = Path(__file__).resolve().parents[2]
    script = (
        root / "scripts/vllm/benchmarking/run_k3_dspark_tp8_pd_pair.sh"
    ).read_text()
    function = script.split("    wait_remote_verifications() {", 1)[1].split(
        "\n    }", 1
    )[0]
    records = tmp_path / "verification.jsonl"
    records.write_text(json.dumps({"request_id": "a", "status": status}) + "\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            "wait_remote_verifications() {"
            + function
            + '\n}\nverification_path="$1"\nwait_remote_verifications 1',
            "gate",
            str(records),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == expected, result.stderr
    diagnostic = script.split('if [[ "${K3_DSPARK_PD_DIAGNOSTIC:-0}" == 1 ]]', 1)[1]
    assert diagnostic.index("wait_remote_verifications 1") < diagnostic.index(
        'touch "$coord/benchmark.done"'
    )


@pytest.mark.parametrize(
    "role,scheduler,kv_role",
    [
        ("prefill", "CapacityRemoteDraftScheduler", "kv_producer"),
        ("decode", "CapacityRemoteConsumerScheduler", "kv_consumer"),
    ],
)
def test_planned_profile_resolves_scheduler_and_fail_policy(role, scheduler, kv_role):
    """Use upstream parsing, including last-wins replacement of base KV config."""
    root = Path(__file__).resolve().parents[2]
    config = root / (
        "scripts/vllm/benchmarking/configs/"
        "kimi-k3-pruned75-dspark-tp8-ep-remote-planned.sh"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" || exit; printf "%s\\n%s\\n%s\\n%s" '
            '"$EXTRA_SERVE_ARGS" "$VLLM_TPU_REMOTE_DSPARK_ROLE" '
            '"$VLLM_TPU_REMOTE_DSPARK_TIMEOUT_S" '
            '"$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"',
            "bash",
            str(config),
        ],
        env={
            **os.environ,
            "K3_DSPARK_PD_ROLE": role,
            "EXTRA_SERVE_ARGS": "",
            "VLLM_TPU_REMOTE_DSPARK_TIMEOUT_S": "1800",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "10800",
            "DSPARK_NUM_SPECULATIVE_TOKENS": "",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    extra, wire_role, socket_timeout, rpc_timeout = result.stdout.splitlines()
    # Match run_benchmarks.sh's word splitting, not shell quote interpretation.
    words = extra.split()
    flags = {
        "--kv-transfer-config",
        "--scheduler-cls",
        "--distributed-executor-backend",
        "--speculative-config",
    }
    selected = []
    for index, word in enumerate(words):
        if word in flags:
            selected.extend([word, words[index + 1]])
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(selected)
    assert args.scheduler_cls == f"vllm_torchtpu.core.capacity_scheduler.{scheduler}"
    assert args.kv_transfer_config.kv_role == kv_role
    assert args.speculative_config["num_speculative_tokens"] == 5
    assert args.kv_transfer_config.kv_load_failure_policy == "fail"
    assert args.distributed_executor_backend == (
        "vllm_torchtpu.executors.tpu_multiproc_executor.TpuMultiprocExecutor"
    )
    assert wire_role == {"prefill": "producer", "decode": "consumer"}[role]
    assert (socket_timeout, rpc_timeout) == ("300", "360")
