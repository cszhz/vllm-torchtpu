# SPDX-License-Identifier: Apache-2.0
"""Decode-only, model-owned NaN captures on the execution thread."""

from __future__ import annotations

import contextlib
import math
import os
import pickle
import socket
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


def _nan_flags(hidden: torch.Tensor, indices: torch.Tensor | None) -> torch.Tensor:
    if indices is not None:
        hidden = torch.index_select(hidden, 0, indices)
    return hidden.isnan().any(dim=-1)


_compiled_nan_flags = torch.compile(
    _nan_flags, backend="tpu", fullgraph=True, dynamic=False
)


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == "tpu":
        synchronize_tensors([tensor], wait=True)
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def _snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _cpu(value)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    return value


def _save(path: Path, value: Any) -> None:
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


class DecodeNaNDumper:
    """Own references to one forward; copy cache only at anomalies 0, 20, 40..."""

    def __init__(self, role: str, layer_names: set[str], directory: Path) -> None:
        self.role = role
        self.layer_names = layer_names
        self.directory = directory
        self.anomaly_count = 0
        self._paused = False
        self._forward: dict[str, Any] | None = None

    @contextlib.contextmanager
    def pause(self) -> Iterator[None]:
        previous = self._paused
        self._paused = True
        try:
            yield
        finally:
            self._paused = previous

    def record_forward(
        self,
        runner: TPUModelRunner,
        start: int,
        count: int,
        inputs: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        if self._paused:
            return
        batch = runner.input_batch
        self._forward = {
            "request_ids": tuple(batch.req_ids[start : start + count]),
            "decode_requests": (
                batch.num_computed_tokens_cpu[start : start + count]
                >= batch.num_prompt_tokens[start : start + count]
            ).copy(),
            "inputs": inputs,
            "metadata": {
                name: md for name, md in metadata.items() if name in self.layer_names
            },
        }

    def check(
        self,
        runner: TPUModelRunner,
        hidden: torch.Tensor,
        row_request_indices: np.ndarray,
        logits_indices: torch.Tensor | None = None,
    ) -> None:
        if self._paused:
            return
        forward = self._forward
        self._forward = None
        assert forward is not None, "NaN check requires its model's recorded forward"
        decode_requests = forward["decode_requests"]
        if not decode_requests.any():
            return
        check = _compiled_nan_flags if hidden.device.type == "tpu" else _nan_flags
        flags = _cpu(check(hidden, logits_indices)).numpy()
        assert len(row_request_indices) <= len(flags), (
            "Real pre-logits rows must fit the hidden buffer: "
            f"model={self.role}, real={len(row_request_indices)}, rows={len(flags)}"
        )
        bad_rows = (
            flags[: len(row_request_indices)] & decode_requests[row_request_indices]
        )
        if not bad_rows.any():
            return
        index = self.anomaly_count
        self.anomaly_count += 1
        if index % 20:
            return
        req_ids = forward["request_ids"]
        bad_req_ids = [req_ids[i] for i in np.unique(row_request_indices[bad_rows])]
        report = {
            "schema_version": 1,
            "model_role": self.role,
            "anomaly_index": index,
            "rank": runner.parallel_config.rank,
            "data_parallel_index": runner.parallel_config.data_parallel_index,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "coverage": "rank_local_model_layers",
            "torch_version": torch.__version__,
            "request_ids": list(req_ids),
            "bad_request_ids": bad_req_ids,
            "cache_phase": "after_forward_before_logits",
            "gdn_input_state_available": False,
            "layers": [],
        }
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            folder = Path(
                tempfile.mkdtemp(
                    prefix=f"vllm-decode-nan-{self.role}-{os.getpid()}-{index}-",
                    dir=self.directory,
                )
            )
            _save(
                folder / "inputs.pkl",
                {
                    "forward": _snapshot(forward["inputs"]),
                    "attention_metadata": {
                        name: _snapshot(vars(md))
                        for name, md in forward["metadata"].items()
                    },
                    "requests": {
                        req_id: _snapshot(
                            {
                                "prompt_token_ids": runner.requests[
                                    req_id
                                ].prompt_token_ids,
                                "output_token_ids": runner.requests[
                                    req_id
                                ].output_token_ids,
                                "block_ids": runner.requests[req_id].block_ids,
                            }
                        )
                        for req_id in req_ids
                    },
                    "hidden_states": _cpu(hidden),
                    "logits_indices": _snapshot(logits_indices),
                    "row_request_indices": row_request_indices.copy(),
                    "decode_requests": decode_requests,
                },
            )
            self._save_caches(runner, folder, report, forward)
            # Only a final manifest marks the capture complete; layers stream
            # page by page so large contexts need no batch-sized host buffer.
            _save(folder / "manifest.pkl", report)
        except OSError:
            logger.exception(
                "Decode NaN dump incomplete: model=%s directory=%s",
                self.role,
                self.directory,
            )
            return
        logger.error(
            "Decode NaN dump: model=%s anomaly_index=%d rank=%d "
            "request_ids=%s bad_request_ids=%s path=%s",
            self.role,
            index,
            runner.parallel_config.rank,
            req_ids,
            bad_req_ids,
            folder,
        )

    def _save_caches(
        self,
        runner: TPUModelRunner,
        folder: Path,
        report: dict[str, Any],
        forward: dict[str, Any],
    ) -> None:
        config = runner.kv_cache_config
        assert config is not None, "Decode forward requires allocated KV cache groups"
        context = runner.vllm_config.compilation_config.static_forward_context
        for gid, group in enumerate(config.kv_cache_groups):
            spec = group.kv_cache_spec
            for name in sorted(set(group.layer_names) & self.layer_names):
                cache = context[name].kv_cache
                tensors = list(cache) if isinstance(cache, (list, tuple)) else [cache]
                blocks = sorted(
                    {
                        block
                        for req_id in forward["request_ids"]
                        for block in runner.requests[req_id].block_ids[gid]
                    }
                )
                page_bytes = math.prod(tensors[0].shape[1:]) * tensors[0].element_size()
                assert spec.page_size_bytes % page_bytes == 0, (
                    f"Manager page must contain complete kernel pages: {name}"
                )
                pages = spec.page_size_bytes // page_bytes
                assert pages > 0, f"KV page mapping must be nonempty: {name}"
                entry = {
                    "layer_name": name,
                    "group_id": gid,
                    "path": f"layer-{len(report['layers']):04d}.pkl",
                    "num_blocks": len(blocks),
                    "kernel_pages_per_block": pages,
                    "layout": "allocated_manager_pages_including_padding",
                    "tensor_shapes": [tuple(t.shape) for t in tensors],
                    "tensor_dtypes": [t.dtype for t in tensors],
                    "cache_spec": _snapshot(vars(spec)),
                }
                with (folder / entry["path"]).open("wb") as stream:
                    pickle.dump(entry, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    for block in blocks:
                        lo, hi = block * pages, (block + 1) * pages
                        assert all(0 <= lo < hi <= t.shape[0] for t in tensors), (
                            f"Request KV block outside model cache: {name}, "
                            f"block={block}"
                        )
                        pickle.dump(
                            {
                                "block_id": block,
                                # Preserve carrier dtype/bits; do not reinterpret
                                # a mixed BF16/FP32 GDN pool as numeric FP8 data.
                                "tensor_bytes": [
                                    _cpu(t[lo:hi]).view(torch.uint8) for t in tensors
                                ],
                            },
                            stream,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                report["layers"].append(entry)
