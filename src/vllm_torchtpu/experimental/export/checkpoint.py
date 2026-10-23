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
"""Writes the real weights of an exported artifact as an Orbax checkpoint.

`python -m vllm_torchtpu.experimental.export.checkpoint --artifact-dir DIR ...`
takes the same model arguments as the export that produced `DIR`. It loads the
checkpoint through vLLM on CPU, one process per tensor-parallel rank, so every
weight goes through vLLM's processing (fused projections, FP8 requantization,
stacked experts) exactly as when serving. It then saves each `abstract_params`
input `i` as array `array_{i}` of an Orbax checkpoint (Zarr v2, no OCDBT) in
`DIR/serving_checkpoint/data`, each rank writing its own shards, and removes
`DIR/serving_checkpoint/weight.json`, whose sources cannot express that
processing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import portpicker

_CPU_ENV = {"VLLM_TARGET_DEVICE": "cpu", "JAX_PLATFORMS": "cpu", "PJRT_DEVICE": "CPU"}
_JAX_PORT_ENV = "EXPORT_CHECKPOINT_JAX_PORT"
_AXIS_NAMES = ("data", "model")
# Checkpoint arrays `array_{i}` live here, relative to the artifact directory.
CHECKPOINT_DATA_DIR = os.path.join("serving_checkpoint", "data")


def build_arg_parser() -> argparse.ArgumentParser:
    """`EngineArgs` plus `--artifact-dir`."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        prog="python -m vllm_torchtpu.experimental.export.checkpoint",
        description=(
            "Save the real weights of an exported artifact as an Orbax checkpoint."
        ),
    )
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--artifact-dir",
        type=str,
        required=True,
        help="Exported artifact whose abstract_params the checkpoint fills.",
    )
    parser.add_argument(
        "--checkpoint-worker", action="store_true", help=argparse.SUPPRESS
    )
    return parser


def abstract_params_inputs(artifact_dir: str) -> list[dict[str, Any]]:
    """The `abstract_params` inputs of the artifact, in argument order."""
    with open(os.path.join(artifact_dir, "metadata.json")) as f:
        meta = json.load(f)
    inputs = meta["functions"]["abstract_params"]["default"]["inputs"]
    return sorted(inputs, key=lambda t: t["arg_index"])


def _axis_sizes(artifact_dir: str) -> dict[str, int]:
    with open(os.path.join(artifact_dir, "metadata.json")) as f:
        mesh = json.load(f)["mesh"]
    return dict(zip(mesh["axis_names"], mesh["mesh_shape"], strict=True))


def shard_shape(spec: Mapping[str, Any], axis_sizes: Mapping[str, int]) -> list[int]:
    """Shape of one device's shard of the global tensor `spec`."""
    shape = []
    for dim, axis in zip(spec["shape"], spec["sharding"], strict=True):
        axes = [] if axis is None else [axis] if isinstance(axis, str) else list(axis)
        div = 1
        for a in axes:
            div *= axis_sizes[a]
        shape.append(int(dim) // div)
    return shape


def local_tensors(model: Any) -> dict[str, Any]:
    """The model's parameters and buffers by name.

    A tied `lm_head.weight` is the embedding tensor.
    """
    tensors = dict(model.named_buffers())
    tensors.update(model.named_parameters())
    if "lm_head.weight" not in tensors and "model.embed_tokens.weight" in tensors:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    return tensors


def to_numpy(tensor: Any) -> Any:
    """A CPU numpy view of `tensor`, with `ml_dtypes` types for bf16 and FP8."""
    import ml_dtypes
    import torch

    views = {
        torch.bfloat16: (torch.int16, ml_dtypes.bfloat16),
        torch.float8_e4m3fn: (torch.uint8, ml_dtypes.float8_e4m3fn),
        torch.float8_e5m2: (torch.uint8, ml_dtypes.float8_e5m2),
    }
    t = tensor.detach().cpu().contiguous()
    if t.dtype in views:
        int_dtype, np_dtype = views[t.dtype]
        return t.view(int_dtype).numpy().view(np_dtype)
    return t.numpy()


def checkpoint_shards(
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Any],
    axis_sizes: Mapping[str, int],
) -> list[Any]:
    """This rank's shard of every `abstract_params` input, as numpy arrays.

    A tensor's shard is the rank's own tensor. A per-device constant, declared as
    a global `[tp, ...]` array, has a `[1, ...]` shard holding the rank's value.
    """
    from vllm_torchtpu.experimental.export import tpu_lowering

    shards = []
    for spec in inputs:
        name = spec["name"]
        if name not in tensors:
            raise KeyError(f"abstract_params input {name} is not in the loaded model")
        tensor = tensors[name]
        dtype = tpu_lowering.dtype_name(tensor.dtype)
        if dtype != spec["dtype"]:
            raise ValueError(
                f"{name}: loaded dtype {dtype}, artifact declares {spec['dtype']}"
            )
        expected = shard_shape(spec, axis_sizes)
        array = to_numpy(tensor)
        if list(array.shape) != expected:
            if [1, *array.shape] != expected:
                raise ValueError(
                    f"{name}: loaded shape {list(array.shape)}, "
                    f"artifact shard shape {expected}"
                )
            array = array.reshape(expected)
        shards.append(array)
    return shards


def save_checkpoint(
    inputs: Sequence[Mapping[str, Any]],
    shards: Sequence[Any],
    axis_sizes: Mapping[str, int],
    data_dir: str,
) -> None:
    """Saves the global arrays assembled from every process's `shards` with Orbax.

    Each process holds one CPU device, the mesh position of its rank; Orbax writes
    each array's chunks from the processes that hold them.
    """
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
    mesh = Mesh(
        np.asarray(devices).reshape([axis_sizes[a] for a in _AXIS_NAMES]), _AXIS_NAMES
    )
    (local,) = jax.local_devices()
    tree = {}
    for i, (spec, shard) in enumerate(zip(inputs, shards, strict=True)):
        sharding = NamedSharding(mesh, PartitionSpec(*spec["sharding"]))
        tree[f"array_{i}"] = jax.make_array_from_single_device_arrays(
            tuple(spec["shape"]), sharding, [jax.device_put(shard, local)]
        )
    checkpointer = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler(use_ocdbt=False, use_zarr3=False)
    )
    checkpointer.save(
        os.path.abspath(data_dir), args=ocp.args.PyTreeSave(tree), force=True
    )


def run_worker(args: argparse.Namespace) -> None:
    """Loads this rank's weights on CPU and saves its shards."""
    import jax
    import torch.distributed as dist
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment
    from vllm.engine.arg_utils import EngineArgs

    from vllm_torchtpu.experimental.export import universal_adapter

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    jax.config.update("jax_enable_x64", True)  # Keep i64 constants 64-bit.
    if world_size > 1:
        jax.distributed.initialize(
            f"{os.environ['MASTER_ADDR']}:{os.environ[_JAX_PORT_ENV]}",
            num_processes=world_size,
            process_id=rank,
        )

    from vllm_torchtpu import patch_registry

    vllm_config = EngineArgs.from_cli_args(args).create_engine_config()
    # vLLM runs on its CPU platform here, which skips the TPU platform's patches;
    # apply them so the model has the parameters and buffers it has on TPU.
    patch_registry.apply("platform_activation", model_config=vllm_config.model_config)
    universal_adapter.check_supported_parallelism(vllm_config.parallel_config)
    load_format = "auto" if args.load_format == "dummy" else args.load_format
    with set_current_vllm_config(vllm_config):
        if not dist.is_initialized():
            # TPU collectives are unavailable on CPU; loading only needs gloo.
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                local_rank=rank,
                distributed_init_method="env://",
                backend="gloo",
            )
        model = universal_adapter.load_vllm_model(vllm_config, load_format=load_format)

    inputs = abstract_params_inputs(args.artifact_dir)
    axis_sizes = _axis_sizes(args.artifact_dir)
    shards = checkpoint_shards(inputs, local_tensors(model), axis_sizes)
    save_checkpoint(
        inputs, shards, axis_sizes, os.path.join(args.artifact_dir, CHECKPOINT_DATA_DIR)
    )
    if rank == 0:
        weight_json = os.path.join(
            args.artifact_dir, "serving_checkpoint", "weight.json"
        )
        if os.path.exists(weight_json):
            os.remove(weight_json)


def _launch_workers(argv: Sequence[str], tp_size: int) -> None:
    """Runs one CPU worker per tensor-parallel rank."""
    env = dict(
        os.environ,
        **_CPU_ENV,
        WORLD_SIZE=str(tp_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(portpicker.pick_unused_port()),
        **{_JAX_PORT_ENV: str(portpicker.pick_unused_port())},
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "vllm_torchtpu.experimental.export.checkpoint"]
            + [*argv, "--checkpoint-worker"],
            env=dict(env, RANK=str(rank), LOCAL_RANK=str(rank)),
        )
        for rank in range(tp_size)
    ]
    try:
        failed = [r for r, proc in enumerate(procs) if proc.wait() != 0]
        if failed:
            raise RuntimeError(f"checkpoint workers for ranks {failed} failed")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m vllm_torchtpu.experimental.export.checkpoint`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--checkpoint-worker" in argv:
        run_worker(build_arg_parser().parse_args(argv))
        return 0
    # vLLM picks its device when imported, so only the workers, started with
    # `_CPU_ENV`, parse the full engine arguments.
    launcher = argparse.ArgumentParser(add_help=False)
    launcher.add_argument("--artifact-dir", required=True)
    launcher.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    args, _ = launcher.parse_known_args(argv)
    if not os.path.isfile(os.path.join(args.artifact_dir, "metadata.json")):
        raise FileNotFoundError(f"{args.artifact_dir} has no metadata.json")
    data_dir = os.path.join(args.artifact_dir, CHECKPOINT_DATA_DIR)
    shutil.rmtree(data_dir, ignore_errors=True)
    _launch_workers(argv, int(args.tensor_parallel_size or 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
