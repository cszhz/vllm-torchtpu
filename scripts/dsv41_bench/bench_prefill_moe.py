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
"""Fused MoE v2: TP1/DP8/EP8, P7, 16384 tokens, XProf device timing.

Each DP rank owns 2048 token rows and 48 experts. All eight DP ranks form
one EP group; token inputs are distinct, with no TP replicas.

On an eight-device TPU host (with the repo and src on PYTHONPATH):
  python -m scripts.dsv41_bench.bench_prefill_moe \\
    --result-dir /tmp/moe/results --profile /tmp/moe/xprof \\
    --dump-llo /tmp/moe/compiler --golden-dir /tmp/moe/golden --update-golden

Use fresh result/profile directories and --verify with the same golden directory
for subsequent runs. Golden checks measure repeatability against the baseline,
not agreement with an independent reference implementation.

run_prefill_moe.sh publishes one baseline per Falcon experiment at
ARTIFACT_LOCAL_DIR/{manifest.json,profiling/,rank-<n>/}. Each rank contains
benchmark/metrics.jsonl, compiler/llo/, and profiling/xprof/. Smoke traces stay
local; failed runs are copied to failed/<run-name>/ for diagnosis.
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import os
import pathlib
import re
import shlex
import shutil
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import unquote, urlparse

from scripts.dsv41_bench import common

GLOBAL_TOKENS = 16384
EP_SIZE = 8
VLLM_COMMIT = "e37e51dd246cf421c554e7d6f53e178e3fd29085"
RUNTIME_PINS = {
    "torch": "2.13.0+cpu",
    "torch-tpu": "0.2.0rc1",
    "jax": "0.11.2",
    "jaxlib": "0.11.2",
    "libtpu": "0.0.50",
    "tpu-sync-torch": "0.0.1.dev20261012084241",
    "numpy": "2.3.5",
    "transformers": "5.18.0",
}
EVENT_PREFIXES = {
    "pallas_main": "fused_ep_moe_v2_fp4w",
    "combine": "moe_v2_combine",
}


def check_runtime() -> dict[str, str]:
    """Verify the installed packages and editable vLLM source before TPU imports."""
    versions = {
        name: md.version(name) for name in (*RUNTIME_PINS, "vllm", "vllm_torchtpu")
    }
    for name, expected in RUNTIME_PINS.items():
        if versions[name] != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {versions[name]}")
    direct_url = md.distribution("vllm").read_text("direct_url.json")
    if direct_url is None:
        raise RuntimeError("Install vLLM from the pinned Git source in editable mode")
    source_info = json.loads(direct_url)
    source = urlparse(source_info["url"])
    if source.scheme != "file" or not source_info.get("dir_info", {}).get("editable"):
        raise RuntimeError(
            "Expected editable local vLLM source for commit verification"
        )
    path = unquote(source.path)
    commit = subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != VLLM_COMMIT:
        raise RuntimeError(f"vLLM source: expected {VLLM_COMMIT}, found {commit}")
    subprocess.run(["git", "-C", path, "diff", "--exit-code", "HEAD"], check=True)
    plugins = md.distribution("vllm_torchtpu").entry_points
    if not any(
        entry.group == "vllm.platform_plugins" and entry.name == "torchtpu"
        for entry in plugins
    ):
        raise RuntimeError("The TorchTPU platform entry point is not installed")
    return {**versions, "vllm_git_sha": commit}


def configure_runtime_env() -> None:
    """Force P7 and profiling metadata flags before TPU runtime initialization."""
    required = {
        "--xla_tpu_dvfs_p_state": "7",
        "--xla_xprof_enable_custom_call_tracing": "true",
        "--xla_xprof_register_llo_debug_info": "true",
    }
    flags = shlex.split(os.environ.get("LIBTPU_INIT_ARGS", ""))
    flags = [flag for flag in flags if flag.split("=", 1)[0] not in required]
    flags.extend(f"{flag}={value}" for flag, value in required.items())
    os.environ["LIBTPU_INIT_ARGS"] = shlex.join(flags)


def summarize_device_events(
    planes: list[dict[str, Any]], profile_steps: int
) -> dict[str, Any]:
    """Reject incomplete traces, then take median of each step's slowest core."""
    if profile_steps < 1:
        raise ValueError("profile_steps must be positive")
    cores = [
        plane
        for plane in planes
        if plane["plane"].startswith("/device:TPU:")
        and "SparseCore" not in plane["plane"]
    ]
    identities = {(plane["file"], plane["plane"]) for plane in cores}
    if len(cores) != EP_SIZE or len(identities) != EP_SIZE:
        raise ValueError(
            f"Expected {EP_SIZE} unique TensorCore planes, got {len(cores)}"
        )
    per_core = []
    for plane in cores:
        scopes: dict[str, list[dict[str, Any]]] = {
            "module": [],
            "pallas_main": [],
            "combine": [],
        }
        for line in plane["lines"]:
            for event in line["events"]:
                scope = None
                if line["line"] == "XLA Modules":
                    scope = "module"
                elif line["line"] == "XLA Ops":
                    op_name = event["name"].split(" = ", 1)[0].lstrip("%")
                    scope = next(
                        (
                            name
                            for name, prefix in EVENT_PREFIXES.items()
                            if op_name.startswith(prefix)
                        ),
                        None,
                    )
                if scope is not None:
                    duration = float(event["duration_ns"]) / 1e6
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError(f"Invalid {scope} duration: {duration}")
                    scopes[scope].append(event)
        durations = {}
        for scope, events in scopes.items():
            if len(events) != profile_steps:
                raise ValueError(
                    f"{plane['file']}:{plane['plane']} {scope}: "
                    f"expected {profile_steps} events, got {len(events)}"
                )
            timestamped = ["start_ns" in event for event in events]
            if any(timestamped):
                if not all(timestamped):
                    raise ValueError(f"{scope}: mixed missing start_ns timestamps")
                if any(not math.isfinite(event["start_ns"]) for event in events):
                    raise ValueError(f"{scope}: invalid start_ns timestamp")
                events.sort(key=lambda event: event["start_ns"])
            # Older saved captures contain durations only, already in step order.
            durations[scope] = [float(event["duration_ns"]) / 1e6 for event in events]
        per_core.append(durations)
    metrics = {}
    for scope in ("module", "pallas_main", "combine"):
        samples = [
            max(core[scope][step] for core in per_core) for step in range(profile_steps)
        ]
        metrics[scope] = {
            "device_median_ms": statistics.median(samples),
            "slowest_core_samples_ms": samples,
        }
    return {
        "global_tokens": GLOBAL_TOKENS,
        "local_tokens": GLOBAL_TOKENS // EP_SIZE,
        "ep_size": EP_SIZE,
        "profile_steps": profile_steps,
        "timing_source": "XProf device events only",
        "aggregation": "per-iteration max over 8 TensorCores, then median",
        "primary_metric": "module",
        "metric_scopes": {
            "module": "full fused_ep_moe_v2 invocation, including routing and combine",
            "pallas_main": "fused_ep_moe_v2 main Pallas custom call",
            "combine": "MoE output combine custom call",
        },
        "metrics": metrics,
        "per_core": [
            {
                "file": plane["file"],
                "plane": plane["plane"],
                "samples_ms": samples,
                **({"rank": plane["rank"]} if "rank" in plane else {}),
            }
            for plane, samples in zip(cores, per_core, strict=True)
        ],
    }


def assign_trace_ranks(planes: list[dict[str, Any]], results: pathlib.Path) -> None:
    """Map chip-scoped XPlanes to workers using their recorded TPU coordinates."""
    by_file: dict[str, list[dict[str, Any]]] = {}
    for plane in planes:
        by_file.setdefault(plane["file"], []).append(plane)
    devices = None
    for filename, cores in by_file.items():
        match = re.match(r"rank(\d+)[_.]", filename)
        if match is None:
            raise ValueError(f"Trace has no worker rank prefix: {filename}")
        owner = int(match[1])
        if len(cores) == 1:
            cores[0]["rank"] = owner
            continue
        if devices is None:
            devices = {
                rank: json.loads((results / f"rank_{rank}.json").read_text())["device"]
                for rank in range(EP_SIZE)
            }
        chip = devices[owner]["coords"]
        chip_ranks = {
            device["core_on_chip"]: rank
            for rank, device in devices.items()
            if device["coords"] == chip
        }
        if len(chip_ranks) != len(cores):
            raise ValueError(f"Trace core count does not match worker chip: {filename}")
        for core in cores:
            core_id = int(core["plane"].removeprefix("/device:TPU:"))
            if core_id not in chip_ranks:
                raise ValueError(f"Unknown core {core_id} on captured chip: {filename}")
            core["rank"] = chip_ranks[core_id]
    ranks = [plane["rank"] for plane in planes]
    if sorted(ranks) != list(range(EP_SIZE)):
        raise ValueError(f"Trace must map each of eight worker ranks once: {ranks}")


def export_falcon_artifact(run_dir: pathlib.Path, output: pathlib.Path) -> None:
    """Stage a validated, rank-scoped operator artifact for the shell publisher."""
    baseline = run_dir / "baseline"
    results = baseline / "results"
    metrics = json.loads((results / "device-metrics.json").read_text())
    if metrics["replay_only"] or (
        metrics["global_tokens"],
        metrics["tp_size"],
        metrics["dp_size"],
        metrics["ep_size"],
    ) != (GLOBAL_TOKENS, 1, EP_SIZE, EP_SIZE):
        raise ValueError("Artifact requires a captured TP1/DP8/EP8 baseline")
    events = json.loads((results / "device-events.json").read_text())
    assign_trace_ranks(events, results)
    measured = summarize_device_events(events, metrics["profile_steps"])
    if measured["metrics"] != metrics["metrics"]:
        raise ValueError("Device metrics do not match the captured events")
    by_rank = {}
    for core in measured["per_core"]:
        rank = core["rank"]
        if rank in by_rank:
            raise ValueError(f"Multiple TensorCore planes for rank {rank}")
        by_rank[rank] = core
    if set(by_rank) != set(range(EP_SIZE)):
        raise ValueError("Artifact requires traces from all eight ranks")

    output.mkdir(parents=True, exist_ok=False)
    profiling = output / "profiling"
    profiling.mkdir()
    shutil.copyfile(run_dir / "python-packages.txt", profiling / "python-packages.txt")
    shutil.copytree(run_dir / "setup", profiling / "setup")
    shutil.copyfile(run_dir / "launcher.log", profiling / "launcher.log")
    shutil.copyfile(__file__, profiling / "exporter-source.py")
    env_lines = []
    for rank, core in sorted(by_rank.items()):
        worker = json.loads((results / f"rank_{rank}.json").read_text())
        if (
            "error" in worker
            or worker["rank"] != rank
            or (worker["tp_size"], worker["dp_size"], worker["ep_size"])
            != (1, EP_SIZE, EP_SIZE)
            or worker["output_finite"] is not True
            or worker["verification"]["golden"] != "verified"
            or worker["verification"]["report"]["output"]["ok"] is not True
        ):
            raise ValueError(f"Invalid worker result for rank {rank}")
        compiler = baseline / "compiler" / f"rank{rank}"
        llos = [
            path
            for path in compiler.rglob("*")
            if path.is_file()
            and re.search(
                r"(^|[._/-])llo([._/-]|$)", path.relative_to(compiler).as_posix()
            )
            and path.stat().st_size > 0
        ]
        if not llos:
            raise ValueError(f"Missing nonempty LLO dump for rank {rank}: {compiler}")
        trace_root = baseline / "xprof"
        match = re.match(r"rank(\d+)_", core["file"])
        if match is None:
            raise ValueError(f"Trace has no capture owner: {core['file']}")
        owner = int(match[1])
        traces = sorted(trace_root.glob(f"plugins/profile/*/rank{owner}_*"))
        xplanes = [path for path in traces if path.name.endswith(".xplane.pb")]
        if (
            len(xplanes) != 1
            or xplanes[0].name != core["file"]
            or not (xplanes[0].stat().st_size)
        ):
            raise ValueError(f"Missing or ambiguous XPlane capture for rank {rank}")
        rank_dir = output / f"rank-{rank}"
        for dump in llos:
            destination = rank_dir / "compiler" / "llo" / dump.relative_to(compiler)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dump, destination)
        for trace in sorted(trace_root.glob(f"plugins/profile/*/rank{rank}_*")):
            destination = (
                rank_dir / "profiling" / "xprof" / trace.relative_to(trace_root)
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(trace, destination)
        capture = rank_dir / "profiling" / "capture.json"
        capture.parent.mkdir(parents=True, exist_ok=True)
        capture.write_text(
            json.dumps(
                {
                    "rank": rank,
                    "plane": core["plane"],
                    "trace": (
                        f"rank-{owner}/profiling/xprof/"
                        f"{xplanes[0].relative_to(trace_root).as_posix()}"
                    ),
                },
                indent=2,
            )
        )
        benchmark = rank_dir / "benchmark"
        benchmark.mkdir()
        shutil.copyfile(results / f"rank_{rank}.json", benchmark / "worker.json")
        rows = [
            {
                "variant": "baseline",
                "rank": rank,
                "step": step,
                "tokens": metrics["local_tokens"],
                "global_tokens": metrics["global_tokens"],
                "tp_size": worker["tp_size"],
                "dp_size": worker["dp_size"],
                "ep_size": worker["ep_size"],
                "latency_ms": latency,
                "pallas_main_ms": core["samples_ms"]["pallas_main"][step],
                "combine_ms": core["samples_ms"]["combine"][step],
                "timing_source": metrics["timing_source"],
                "scope": "module",
            }
            for step, latency in enumerate(core["samples_ms"]["module"])
        ]
        (benchmark / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        env_lines.append(f"rank-{rank} LIBTPU_INIT_ARGS={worker['libtpu_init_args']}\n")
    (profiling / "env.txt").write_text("".join(env_lines))
    diagnostics = output / "rank-0" / "benchmark" / "diagnostics"
    diagnostics.mkdir()
    # Exclude smoke traces so recursive profile discovery sees only baseline.
    for phase in ("smoke", "baseline"):
        shutil.copytree(run_dir / phase / "results", diagnostics / phase / "results")
        shutil.copyfile(
            run_dir / phase / "bench.log", diagnostics / phase / "bench.log"
        )
    shutil.copytree(run_dir / "golden", diagnostics / "golden")
    for name in ("runtime-versions.json", "source-files.json", "exit-status.txt"):
        shutil.copyfile(run_dir / name, profiling / name)
    manifest = {
        "schema_version": 1,
        "workflow": "operator-optimization",
        "operator_family": "fused_moe",
        "operator_name": "fused_ep_moe_v2",
        "run_name": run_dir.name,
        "dimensions": {
            key: metrics[key]
            for key in (
                "global_tokens",
                "local_tokens",
                "tp_size",
                "dp_size",
                "ep_size",
            )
        },
        "profile_steps": metrics["profile_steps"],
        "primary_metric": "latency_ms",
        "metric_scope": metrics["metric_scopes"]["module"],
        "distributed_summary": (
            "rank-0/benchmark/diagnostics/baseline/results/device-metrics.json"
        ),
        "distributed_aggregation": metrics["aggregation"],
        "rank_coverage": sorted(by_rank),
        "exporter_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def publish_falcon_artifact(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Copy independent files concurrently and publish the root manifest last."""
    manifest = source / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Missing validated artifact manifest: {manifest}")
    files = [path for path in source.rglob("*") if path.is_file() and path != manifest]

    def copy(path: pathlib.Path) -> None:
        output = destination / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)

    with ThreadPoolExecutor() as executor:
        list(executor.map(copy, files))
    shutil.copyfile(manifest, destination / "manifest.json")


def collect_device_events(profile_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Read captured events without initializing a TPU execution client."""
    from jax.profiler import ProfileData

    planes = []
    for path in sorted(profile_dir.rglob("*.xplane.pb")):
        for plane in ProfileData.from_file(str(path)).planes:
            if not plane.name.startswith("/device:TPU:") or "SparseCore" in plane.name:
                continue
            lines = []
            for line in plane.lines:
                if line.name in ("XLA Modules", "XLA Ops"):
                    lines.append(
                        {
                            "line": line.name,
                            "events": [
                                {
                                    "name": event.name,
                                    "start_ns": event.start_ns,
                                    "duration_ns": event.duration_ns,
                                }
                                for event in line.events
                            ],
                        }
                    )
            planes.append(
                {
                    "file": path.name,
                    "plane": plane.name,
                    "lines": lines,
                    "track_event_counts": {
                        line.name: len(line.events) for line in plane.lines
                    },
                }
            )
    return planes


def worker(ctx: common.WorkerContext, args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas
    from vllm.distributed import get_dp_group, get_ep_group, get_tp_group

    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME, build_ep_mesh
    from vllm_torchtpu.distributed.mesh_utils import _get_tpu_global_device_id
    from vllm_torchtpu.kernels.fused_moe.v2 import WeightFormat, fused_ep_moe_v2

    ep_size = get_ep_group().world_size
    tp_size = get_tp_group().world_size
    dp_size = get_dp_group().world_size
    if (
        ctx.world_size != EP_SIZE
        or ctx.tp_size != 1
        or ctx.dp_size != EP_SIZE
        or tp_size != 1
        or dp_size != EP_SIZE
        or ep_size != EP_SIZE
    ):
        raise ValueError("This fixed workload requires TP=1, DP=8 and EP=8")
    mesh = build_ep_mesh()
    if mesh is None or mesh.shape[EP_AXIS_NAME] != ep_size:
        raise ValueError("Expected an initialized 8-device EP mesh")
    device_id = _get_tpu_global_device_id()
    device = next(device for device in jax.devices() if device.id == device_id)
    tokens_local = GLOBAL_TOKENS // ep_size
    token_groups = tuple((rank,) for rank in range(ep_size))
    hidden, inter, experts_global, topk, qb = 5120, 2560, 384, 6, 512
    local_experts = experts_global // ep_size
    with common.model_wrapper_context():
        generator = torch.Generator(device="cpu").manual_seed(args.seed + ctx.rank)
        w1_bytes = torch.randint(
            0,
            256,
            (local_experts, hidden, inter),
            generator=generator,
            dtype=torch.uint8,
        )
        w2_bytes = torch.randint(
            0,
            256,
            (local_experts, inter, hidden // 2),
            generator=generator,
            dtype=torch.uint8,
        )
        # Preserve the model's 2304 intermediate channels inside the 2560 padding.
        w1_bytes[:, :, 2304 // 2 : inter // 2] = 0
        w1_bytes[:, :, (inter + 2304) // 2 :] = 0
        w2_bytes[:, 2304:, :] = 0
        w1 = w1_bytes.to("tpu").view(torch.float4_e2m1fn_x2)
        w2 = w2_bytes.to("tpu").view(torch.float4_e2m1fn_x2)
        w1s = torch.full(
            (local_experts, hidden // qb, 2 * inter),
            0.01,
            dtype=torch.float32,
            device="tpu",
        )
        w2s = torch.full(
            (local_experts, inter // qb, hidden),
            0.01,
            dtype=torch.float32,
            device="tpu",
        )
        rank_buf = torch.tensor([[ctx.rank]], dtype=torch.int32, device="tpu")
        token_generator = torch.Generator(device="cpu").manual_seed(
            args.seed + 1000 + ctx.rank
        )
        x = (
            torch.randn(
                (tokens_local, hidden), generator=token_generator, dtype=torch.bfloat16
            )
            * 0.1
        ).to("tpu")
        gating = (
            torch.randn(
                (tokens_local, experts_global),
                generator=token_generator,
                dtype=torch.float32,
            )
            * 0.1
        ).to("tpu")
        common.synchronize((x, w1, w2, w1s, w2s, gating, rank_buf))

        def invoke(
            x: jax.Array,
            a1: jax.Array,
            a2: jax.Array,
            s1: jax.Array,
            s2: jax.Array,
            scores: jax.Array,
            rank: jax.Array,
        ) -> jax.Array:
            return fused_ep_moe_v2(
                x,
                a1,
                a2,
                s1,
                s2,
                scores.astype(jnp.float32),
                rank=rank,
                topk=topk,
                renormalize=True,
                mesh=mesh,
                capacity=128,
                block=None,
                ragged_stride=None,
                weight_format=WeightFormat.FP4,
                rhs_qb=qb,
                act_fn="silu_and_mul_with_clamp",
                mesh_ep_ranks=None,
                sharded_plan=True,
                token_replica_groups=token_groups,
                scoring_fn="sigmoid",
                routed_scaling_factor=2.5,
            )

        spec = P(EP_AXIS_NAME)
        op = pallas.jax_op(
            "pallas::bench_fused_moe_v2_prefill_16384",
            invoke,
            mesh=mesh,
            input_partition_specs=(spec,) * 7,
            output_partition_specs=spec,
        )
        op.register_fake(lambda x, *_: torch.empty_like(x))

        def step():
            return op(x, w1, w2, w1s, w2s, gating, rank_buf)

        for _ in range(args.warmup):
            out = step()
            common.synchronize(out)
        if tuple(out.shape) != (tokens_local, hidden):
            raise ValueError(f"Unexpected output shape: {tuple(out.shape)}")
        if not torch.isfinite(out.cpu()).all().item():
            raise ValueError("MoE output contains non-finite values")
        verification = common.check_against_golden(
            ctx,
            args,
            "fused_moe_v2_prefill_global16384",
            {"output": out},
            common.LOSSLESS,
        )
        profile_dir = common.profile_steps(
            ctx, args, step, label="fused_moe_v2_prefill_global16384"
        )
    return {
        "global_tokens": GLOBAL_TOKENS,
        "local_tokens": tokens_local,
        "tp_size": tp_size,
        "dp_size": dp_size,
        "ep_size": ep_size,
        "token_layout": "sharded_no_replicas",
        "device": {
            "id": device_id,
            "coords": list(device.coords),
            "core_on_chip": device.core_on_chip,
        },
        "hidden": hidden,
        "intermediate_original": 2304,
        "intermediate_padded": inter,
        "experts_global": experts_global,
        "experts_local": local_experts,
        "topk": topk,
        "weight_format": "fp4",
        "rhs_qb": qb,
        "capacity": 128,
        "seed": args.seed,
        "sharded_plan": True,
        "token_replica_groups": token_groups,
        "output_finite": True,
        "verification": verification,
        "profile_dir": profile_dir,
        "profile_steps": args.profile_steps,
        "trace_mode": args.trace_mode,
        "libtpu_init_args": os.environ["LIBTPU_INIT_ARGS"],
        "dump_llo": str(args.dump_llo.resolve()) if args.dump_llo else None,
        "hbm": common.hbm_stats(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser, ("moe_only",))
    parser.add_argument("--global-tokens", type=int, default=GLOBAL_TOKENS)
    parser.add_argument("--replay-device-events", type=pathlib.Path)
    parser.set_defaults(
        tp_size=1,
        dp_size=EP_SIZE,
        warmup=5,
        profile_steps=5,
        trace_mode="TRACE_ONLY_XLA",
    )
    args = parser.parse_args()
    if (
        args.global_tokens != GLOBAL_TOKENS
        or args.tp_size != 1
        or args.dp_size != EP_SIZE
    ):
        parser.error(
            "This benchmark requires --global-tokens 16384 --tp-size 1 --dp-size 8"
        )
    if args.warmup < 1 or args.profile_steps < 1:
        parser.error("--warmup and --profile-steps must be positive")
    if args.result_dir is None:
        parser.error("--result-dir is required")
    if not args.worker and (args.result_dir / "device-metrics.json").exists():
        parser.error("--result-dir contains device metrics; use a fresh directory")
    if args.replay_device_events:
        planes = json.loads(args.replay_device_events.read_text())
    else:
        if args.profile is None:
            parser.error("--profile is required; timing comes only from XProf")
        if not args.worker:
            if args.profile.exists() and (
                not args.profile.is_dir() or any(args.profile.iterdir())
            ):
                parser.error("--profile must name a fresh, empty directory")
            golden_paths = [
                common.golden_path(
                    args.golden_dir, "fused_moe_v2_prefill_global16384", rank
                )
                for rank in range(EP_SIZE)
            ]
            if args.update_golden and any(path.exists() for path in golden_paths):
                parser.error("Reference outputs exist; use a fresh --golden-dir")
            if args.verify and not args.update_golden:
                missing = [str(path) for path in golden_paths if not path.is_file()]
                if missing:
                    parser.error(
                        "Reference outputs missing; record the unchanged baseline "
                        "with --update-golden first"
                    )
        configure_runtime_env()
        if not args.worker:
            check_runtime()
        common.run_main("fused_moe_v2_prefill_16384", parser, worker)
        if args.worker:
            return
        planes = collect_device_events(args.profile)
        assign_trace_ranks(planes, args.result_dir)
    metrics = summarize_device_events(planes, args.profile_steps)
    metrics["replay_only"] = args.replay_device_events is not None
    metrics["benchmark_sha256"] = hashlib.sha256(
        pathlib.Path(__file__).read_bytes()
    ).hexdigest()
    if args.replay_device_events is None:
        metrics["tp_size"] = args.tp_size
        metrics["dp_size"] = args.dp_size
        metrics["token_layout"] = "sharded_no_replicas"
        metrics["kernel_git_sha"] = common.git_sha()
        metrics["profile_dir"] = str(args.profile.resolve())
        metrics["trace_mode"] = args.trace_mode
        metrics["execution"] = "TorchTPU pallas.jax_op"
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "device-events.json").write_text(json.dumps(planes, indent=2))
    (args.result_dir / "device-metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("REPLAY_OK" if args.replay_device_events else "BENCHMARK_OK")


if __name__ == "__main__":
    main()
