# SPDX-License-Identifier: Apache-2.0
"""First-hit snapshots with bounded asynchronous export and a round cooldown."""

import os
import pickle
import socket
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import torch
from torch_tpu._internal.stream.streams import TpuEvent
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import supports_multimodal

from vllm_torchtpu.layers.adapter.custom_ops.hidden_nan_flags_op import record_layer_nan
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MAX_DUMP_ROUNDS = 10

if TYPE_CHECKING:
    from vllm.config import ModelConfig


@dataclass
class ModelHiddenNaNSnapshot:
    role: str
    layer_ids: tuple[int, ...]
    state: torch.Tensor
    hidden: torch.Tensor
    input_ids: torch.Tensor
    current_input_ids: torch.Tensor
    current_input_rows: torch.Tensor
    hidden_width: int

    @property
    def hidden_shape(self) -> tuple[int, int]:
        return self.hidden.numel() // self.hidden_width, self.hidden_width


@dataclass
class HiddenNaNReadback:
    round_id: int
    req_ids: tuple[str, ...]
    model_diagnostics: tuple[ModelHiddenNaNSnapshot, ...]
    states_cpu: tuple[torch.Tensor, ...]
    ready_event: TpuEvent
    detector: "HiddenNaNDetector"

    def consume(self) -> None:
        self.ready_event.synchronize()
        self.detector._submit_dump(self)


@dataclass
class HiddenNaNDump:
    role: str
    layer_id: int
    hidden_rows: int
    token_rows: int
    hidden_cpu: torch.Tensor
    tokens_cpu: torch.Tensor
    hidden_width: int


def _write_dump(
    snapshots: tuple[HiddenNaNDump, ...],
    ready_event: TpuEvent,
    dump_dir: Path,
    round_id: int,
    req_ids: tuple[str, ...],
) -> None:
    ready_event.synchronize()
    dump_dir.mkdir(parents=True, exist_ok=True)
    for snapshot in snapshots:
        path = dump_dir / (
            f"hidden-nan-{socket.gethostname()}-{os.getpid()}-"
            f"{snapshot.role}-round{round_id}-layer{snapshot.layer_id}.pkl"
        )
        temporary = path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as stream:
            pickle.dump(
                {
                    "role": snapshot.role,
                    "round": round_id,
                    "layer_id": snapshot.layer_id,
                    "req_ids": list(req_ids),
                    "hidden_states": snapshot.hidden_cpu.reshape(
                        -1, snapshot.hidden_width
                    )[: snapshot.hidden_rows].clone(),
                    "input_ids": snapshot.tokens_cpu[: snapshot.token_rows].clone(),
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary.replace(path)
        logger.error(
            "Hidden NaN/Inf snapshot saved: role=%s round=%d layer=%d "
            "hidden_shape=%s input_tokens=%d path=%s",
            snapshot.role,
            round_id,
            snapshot.layer_id,
            (snapshot.hidden_rows, snapshot.hidden_width),
            snapshot.token_rows,
            path,
        )


class HiddenNaNDetector:
    """Feature-owned buffers capture the first bad layer across a round's forwards."""

    def __init__(
        self, model_diagnostics: tuple[ModelHiddenNaNSnapshot, ...], dump_dir: Path
    ) -> None:
        self.model_diagnostics = model_diagnostics
        self.dump_dir = dump_dir
        self.armed_tpu = torch.zeros(
            1, dtype=torch.bool, device=model_diagnostics[0].state.device
        )
        self.round_state = torch.zeros(
            2, dtype=torch.int32, device=model_diagnostics[0].state.device
        )
        self._armed = False
        self._round_id = 0
        self._req_ids: tuple[str, ...] = ()
        self._dump_lock = Lock()
        self._dump_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hidden-nan"
        )
        self._dump_future: Future[None] | None = None
        self._last_dump_round = 0
        self._resume_round = 0
        self._dump_round_count = 0

    def record_input_ids(self, input_ids: torch.Tensor) -> None:
        # The target may receive only inputs_embeds. Preserve its actual device
        # tokens after async placeholder substitution, before that conversion.
        self._bind_input_ids(0, input_ids)

    def _bind_input_ids(self, model_index: int, input_ids: torch.Tensor) -> None:
        diagnostic = self.model_diagnostics[model_index]
        diagnostic.current_input_ids[: input_ids.shape[0]].copy_(input_ids)
        diagnostic.current_input_rows.fill_(input_ids.shape[0])

    def before_model(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        model_index: int,
    ) -> None:
        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else inputs[0]
        if input_ids is not None:
            self._bind_input_ids(model_index, input_ids)

    def before_embedding(
        self, module: torch.nn.Module, inputs: tuple[Any, ...], *, model_index: int
    ) -> None:
        # MTP loop programs call the model's forward unbound, bypassing its
        # root prehook. The token embedding still receives each step's IDs.
        self._bind_input_ids(model_index, inputs[0])

    def after_layer(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: torch.Tensor | tuple[Any, ...],
        *,
        model_index: int,
        layer_slot: int,
    ) -> torch.Tensor | tuple[Any, ...]:
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        diagnostic = self.model_diagnostics[model_index]
        assert hidden.ndim == 2 and hidden.shape[1] == diagnostic.hidden_width, (
            f"Hidden width mismatch: role={diagnostic.role} shape={hidden.shape}, "
            f"expected width={diagnostic.hidden_width}"
        )
        hidden = record_layer_nan(
            diagnostic.state,
            diagnostic.hidden,
            diagnostic.input_ids,
            self.armed_tpu,
            hidden,
            diagnostic.current_input_ids,
            diagnostic.current_input_rows,
            layer_slot,
            self.round_state,
        )
        return hidden if isinstance(output, torch.Tensor) else (hidden, *output[1:])

    def arm(self) -> None:
        self.armed_tpu.fill_(False)
        self._armed = True
        self._round_id = 0
        self.round_state.zero_()
        for diagnostic in self.model_diagnostics:
            diagnostic.state.copy_(
                torch.tensor(
                    [-1, 0, 0, 0], dtype=torch.int32, device=diagnostic.state.device
                )
            )

    def begin_round(self, req_ids: list[str]) -> None:
        self._round_id += 1
        self._req_ids = tuple(req_ids)
        with self._dump_lock:
            if (
                self._dump_future is not None
                and self._dump_future.done()
                and self._round_id >= self._resume_round
            ):
                self._dump_future.result()
                self._dump_future = None
                for diagnostic in self.model_diagnostics:
                    diagnostic.state[0].fill_(-1)
                self.round_state[1].zero_()
            should_check = self._armed and self._dump_round_count < _MAX_DUMP_ROUNDS
        # Reuse a device increment: distinct Python fill values are cached as
        # separate TPU scalar buffers, otherwise adding HBM on every round.
        self.round_state[0].add_(1)
        self.armed_tpu.fill_(should_check)

    def readback(self) -> HiddenNaNReadback | None:
        # Warmup outputs may be discarded; never enqueue an unconsumed D2H then.
        if not self._armed:
            return None
        self.armed_tpu.fill_(False)
        with self._dump_lock:
            if (
                self._dump_future is not None
                or self._dump_round_count >= _MAX_DUMP_ROUNDS
            ):
                return None
        states_cpu = tuple(
            diagnostic.state.to("cpu", non_blocking=True)
            for diagnostic in self.model_diagnostics
        )
        event = TpuEvent()
        event.record()
        return HiddenNaNReadback(
            self._round_id,
            self._req_ids,
            self.model_diagnostics,
            states_cpu,
            event,
            self,
        )

    def _submit_dump(self, readback: HiddenNaNReadback) -> None:
        hits = [
            (diagnostic, state.tolist())
            for diagnostic, state in zip(
                readback.model_diagnostics, readback.states_cpu
            )
            if state[0].item() >= 0 and state[1].item() == readback.round_id
        ]
        if not hits:
            return
        with self._dump_lock:
            if (
                self._dump_future is not None
                or readback.round_id <= self._last_dump_round
                or self._dump_round_count >= _MAX_DUMP_ROUNDS
            ):
                return
            # Device first-hit guards freeze both snapshots until this job is
            # complete. No GPU clones or queued snapshot generations are needed.
            snapshots = tuple(
                HiddenNaNDump(
                    diagnostic.role,
                    diagnostic.layer_ids[slot],
                    hidden_rows,
                    token_rows,
                    diagnostic.hidden.to("cpu", non_blocking=True),
                    diagnostic.input_ids.to("cpu", non_blocking=True),
                    diagnostic.hidden_width,
                )
                for diagnostic, (slot, _, hidden_rows, token_rows) in hits
            )
            event = TpuEvent()
            event.record()
            self._last_dump_round = readback.round_id
            self._resume_round = readback.round_id + 21
            self._dump_future = self._dump_executor.submit(
                _write_dump,
                snapshots,
                event,
                self.dump_dir,
                readback.round_id,
                readback.req_ids,
            )
            self._dump_round_count += 1


def install_hidden_nan_detector(
    models: list[tuple[str, torch.nn.Module, "ModelConfig", int]],
    device: torch.device,
    *,
    dump_dir: Path | None = None,
) -> HiddenNaNDetector:
    """Validate the model boundary once and attach standard decoder-layer hooks."""
    model_layers: list[tuple[str, list[tuple[int, torch.nn.Module]]]] = []
    model_embeddings: list[list[torch.nn.Module]] = []
    for role, model, _, _ in models:
        if supports_multimodal(model):
            model = cast(torch.nn.Module, model.get_language_model())
        model_embeddings.append(
            [
                module
                for name, module in model.named_modules()
                if (
                    isinstance(module, VocabParallelEmbedding)
                    and not isinstance(module, ParallelLMHead)
                )
                # The draft loader replaces embed_tokens with nn.Embedding
                # when its TP differs. Position embeddings are not token inputs.
                or (
                    name.rsplit(".", 1)[-1] == "embed_tokens"
                    and isinstance(module, torch.nn.Embedding)
                )
            ]
        )
        layers = []
        containers = []
        for name, module in model.named_modules():
            if not isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                continue
            # A block's nested containers are implementation details, not layers.
            if any(
                not parent or name.startswith(parent + ".") for parent in containers
            ):
                continue
            containers.append(name)
            for index, layer in module.named_children():
                if not index.isdecimal():
                    raise ValueError(
                        "Hidden NaN/Inf detection requires integer-indexed layer "
                        f"containers: role={role} module={name}.{index}"
                    )
                layers.append((int(index), layer))
        if not layers:
            raise ValueError(
                "Hidden NaN/Inf detection requires layers registered in a ModuleList "
                f"or an integer-indexed ModuleDict: role={role}"
            )
        model_layers.append((role, sorted(layers, key=lambda item: item[0])))
    # Target and Draft may share a token embedding. Both hooks receive its
    # inputs, including draft query shapes larger than the target's buckets.
    token_capacity = max(capacity for _, _, _, capacity in models)
    detector = HiddenNaNDetector(
        tuple(
            ModelHiddenNaNSnapshot(
                role,
                tuple(index for index, _ in layers),
                torch.tensor([-1, 0, 0, 0], dtype=torch.int32, device=device),
                torch.zeros(
                    capacity * config.get_hidden_size(),
                    dtype=cast(torch.dtype, config.dtype),
                    device=device,
                ),
                torch.zeros(token_capacity, dtype=torch.int32, device=device),
                torch.zeros(token_capacity, dtype=torch.int32, device=device),
                torch.zeros(1, dtype=torch.int32, device=device),
                config.get_hidden_size(),
            )
            for (role, layers), (_, _, config, capacity) in zip(model_layers, models)
        ),
        dump_dir
        if dump_dir is not None
        else Path(tempfile.gettempdir()) / "vllm-hidden-nan",
    )
    for model_index, ((_, model, _, _), embeddings) in enumerate(
        zip(models, model_embeddings)
    ):
        if embeddings:
            for embedding in embeddings:
                embedding.register_forward_pre_hook(
                    partial(detector.before_embedding, model_index=model_index)
                )
        else:
            model.register_forward_pre_hook(
                partial(detector.before_model, model_index=model_index),
                with_kwargs=True,
            )
    for model_index, ((role, layers), diagnostic) in enumerate(
        zip(model_layers, detector.model_diagnostics)
    ):
        for slot, (_, layer) in enumerate(layers):
            layer.register_forward_hook(
                partial(detector.after_layer, model_index=model_index, layer_slot=slot)
            )
        logger.info(
            "Hidden NaN/Inf detection installed: role=%s layers=%d snapshot_shape=%s "
            "cooldown=20_rounds max_pending_dumps=1 max_dump_rounds=%d dump_dir=%s",
            role,
            len(layers),
            diagnostic.hidden_shape,
            _MAX_DUMP_ROUNDS,
            detector.dump_dir,
        )
    return detector
