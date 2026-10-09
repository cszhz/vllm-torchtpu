# SPDX-License-Identifier: Apache-2.0
"""Temporary lazy MTP diagnostics, collected only after low acceptance."""

from __future__ import annotations

import os
import socket
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

import torch
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class MtpDiagnosticStep:
    """Serialize a rare KV read with the next step's first cache mutation."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.valid = True

    def invalidate(self) -> None:
        with self.lock:
            self.valid = False


@dataclass
class DeferredMtpChunk:
    """References/host metadata only: construction must not touch tensor data."""

    step: MtpDiagnosticStep
    runner: TPUModelRunner
    metadata: dict[str, Any]
    req_ids: list[str]
    scheduled: list[int]
    previous_positions: dict[str, int]
    positions: dict[str, int]
    hidden: torch.Tensor | None = None
    logits_indices: torch.Tensor | None = None
    real_rows: int = 0

    def collect(self) -> dict[str, Any]:
        assert self.hidden is not None
        hidden = self.hidden
        if self.logits_indices is not None:
            hidden = torch.index_select(hidden, 0, self.logits_indices)
        with self.step.lock:
            if self.step.valid:
                capture = MtpTensorDiagnostics(
                    self.runner,
                    self.metadata,
                    self.req_ids,
                    self.scheduled,
                    self.previous_positions,
                    cache_observation="after_sampling",
                )
                return capture.before_logits(hidden, self.real_rows)

        # Never inspect old metadata/cache tensors after the next execute_model
        # can have reset offsets, copied pages or written another forward.
        capture_id = uuid.uuid4().hex
        report: dict[str, Any] = {
            "capture_id": capture_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "rank": self.runner.parallel_config.rank,
            "coverage": "rank_local_layers",
            "request_ids": self.req_ids,
            "cache_observation": "unavailable",
            "gdn_checkpoints": [
                {
                    "request_id": req_id,
                    "previous_state_column": self.previous_positions.get(req_id),
                    "state_column": self.positions.get(req_id),
                    "slot": None,
                    "read_offset": None,
                    "read_slot": None,
                    "status": "cache_reused_before_acceptance_was_known",
                }
                for req_id in self.req_ids
            ],
            "kv_cache": {
                "has_nan": None,
                "status": "unavailable",
                "reason": "cache_reused_before_acceptance_was_known",
            },
        }
        directory = Path(tempfile.gettempdir()) / f"vllm-mtp-{os.getpid()}-{capture_id}"
        report["hidden_states"] = inspect_tensor(
            hidden[: self.real_rows],
            directory / "pre-logits.pt",
            {**report, "stage": "pre_logits_tensor_retained_until_sampling"},
        )
        return report


def inspect_tensor(
    tensor: torch.Tensor, path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Report both outcomes, saving the complete CPU tensor only on NaN."""
    cpu = tensor.detach().to("cpu").contiguous()
    mask = torch.isnan(
        cpu.float() if cpu.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else cpu
    )
    count = int(mask.sum())
    report: dict[str, Any] = {
        "has_nan": count > 0,
        "nan_count": count,
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "dump_path": None,
    }
    if count:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"tensor": cpu, "metadata": metadata}, path)
            report["dump_path"] = str(path)
        except (OSError, RuntimeError) as exc:
            report["dump_error"] = str(exc)
            logger.exception("MTP NaN tensor dump failed: %s", path)
    return report


def inspect_pool(
    storage: torch.Tensor,
    pool: rpm.PoolEntry,
    block_ids: list[int],
    directory: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Read only the manifest's live regions, preserving mixed-dtype pool bits."""
    if not storage.is_contiguous():
        raise NotImplementedError("noncontiguous backing storage has no byte view")
    dtype = getattr(torch, pool.dtype_tag)
    if not dtype.is_floating_point:
        raise NotImplementedError(f"encoded cache dtype {pool.dtype_tag}")
    flat = storage.view(-1)
    itemsize = storage.element_size()
    begin = min(r.offset_bytes for r in pool.regions)
    end = max(r.extent_end_bytes for r in pool.regions)
    result: dict[str, Any] = {
        "layer": pool.layer_name,
        "pool": pool.tag,
        "block_ids": block_ids,
        "dtype": pool.dtype_tag,
        "has_nan": False,
        "nan_count": 0,
        "nan_blocks": [],
        "dump_paths": [],
        "status": "checked",
    }
    for block in block_ids:
        if not 0 <= block < pool.num_blocks:
            raise ValueError(f"cache block {block} outside [0, {pool.num_blocks})")
        base = pool.base_offset_bytes + block * pool.block_stride_bytes
        lo = (base + begin) // itemsize
        hi = (base + end + itemsize - 1) // itemsize
        # Reinterpret only after D2H: a finite fp32 SSM word can contain a
        # half-word that looks like bf16 NaN. Never isnan() the raw BF16 pool.
        raw = flat[lo:hi].detach().to("cpu").contiguous().view(torch.uint8)
        pieces = [
            torch.as_strided(
                raw,
                (region.num_units, region.unit_bytes * region.units_per_stride),
                (region.stride_bytes, 1),
                storage_offset=raw.storage_offset()
                + base
                + region.offset_bytes
                - lo * itemsize,
            ).reshape(-1)
            for region in pool.regions
        ]
        values = torch.cat(pieces).view(dtype)
        info = {**metadata, "pool": asdict(pool), "block_id": block}
        check = inspect_tensor(values, directory / f"{uuid.uuid4().hex}.pt", info)
        result["nan_count"] += check["nan_count"]
        if check["has_nan"]:
            result["nan_blocks"].append(block)
            if check["dump_path"] is not None:
                result["dump_paths"].append(check["dump_path"])
            if "dump_error" in check:
                result.setdefault("dump_errors", []).append(check["dump_error"])
    result["has_nan"] = result["nan_count"] > 0
    return result


def checkpoint_snapshot(
    metadata: Any,
    req_ids: list[str],
    scheduled: list[int],
    positions: dict[str, int],
    previous_positions: dict[str, int],
) -> list[dict[str, Any]]:
    n = len(req_ids)
    indices = metadata.mamba_state_indices
    if indices is None:
        indices = metadata.block_tables.reshape(metadata.seq_lens.shape[0], -1)[:, 0]
    slots = indices[:n].cpu().tolist()
    offsets = metadata.mamba_slot_read_offsets
    offsets_cpu = offsets.cpu() if offsets is not None else None
    checkpoints = metadata.mamba_ckpt_indices
    checkpoints_cpu = (
        checkpoints[:n].cpu().tolist() if checkpoints is not None else None
    )
    seq_lens = metadata.seq_lens[:n].cpu().tolist()
    result = []
    for i, (req_id, slot) in enumerate(zip(req_ids, slots)):
        invalid = None
        if offsets_cpu is not None and not 0 <= slot < offsets_cpu.numel():
            offset = None
            invalid = "base slot is outside the read-offset buffer"
        else:
            offset = int(offsets_cpu[slot]) if offsets_cpu is not None else 0
        ckpts = checkpoints_cpu[i] if checkpoints_cpu is not None else []
        if ckpts and (offset is None or not 0 <= offset < len(ckpts)):
            invalid = "read offset is outside the checkpoint window"
        read_slot = None
        if invalid is None:
            assert offset is not None
            read_slot = ckpts[offset] if ckpts else slot + offset
        result.append(
            {
                "request_id": req_id,
                "slot": slot,
                "read_offset": offset,
                "read_slot": read_slot,
                "checkpoint_slots": ckpts,
                "state_column": positions.get(req_id),
                "previous_state_column": previous_positions.get(req_id),
                "num_computed_tokens": seq_lens[i] - scheduled[i],
                "num_scheduled_tokens": scheduled[i],
            }
        )
        if invalid:
            result[-1]["address_error"] = invalid
    return result


class MtpTensorDiagnostics:
    """Own one real forward chunk's metadata and before/after cache checks."""

    def __init__(
        self,
        runner: TPUModelRunner,
        attn_metadata: dict[str, Any],
        req_ids: list[str],
        scheduled: list[int],
        previous_positions: dict[str, int],
        cache_observation: Literal[
            "before_forward", "after_sampling"
        ] = "before_forward",
    ) -> None:
        self.runner = runner
        self.metadata = attn_metadata
        self.req_ids = req_ids.copy()
        self.scheduled = scheduled.copy()
        self.checks: list[dict[str, Any]] = []
        self.checkpoints: dict[int, list[dict[str, Any]]] = {}
        self.written_counts: dict[int, list[int]] = {}
        capture_id = uuid.uuid4().hex
        self.directory = (
            Path(tempfile.gettempdir()) / f"vllm-mtp-{os.getpid()}-{capture_id}"
        )
        self.report: dict[str, Any] = {
            "capture_id": capture_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "rank": runner.parallel_config.rank,
            "coverage": "rank_local_layers",
            "layer_names": sorted(attn_metadata),
            "request_ids": self.req_ids,
            "gdn_checkpoints": [],
            "cache_observation": cache_observation,
            "input_state_snapshot_available": cache_observation == "before_forward",
        }
        assert runner.kv_cache_config is not None
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            if not isinstance(group.kv_cache_spec, MambaSpec):
                continue
            name = next(
                (name for name in group.layer_names if name in attn_metadata), None
            )
            if name is None:
                continue
            entries = checkpoint_snapshot(
                attn_metadata[name],
                req_ids,
                scheduled,
                runner._mamba_state_pos,
                previous_positions,
            )
            self.checkpoints[gid] = entries
            for entry in entries:
                entry["read_offset_phase"] = cache_observation
                if "address_error" in entry and cache_observation == "after_sampling":
                    self.checks.append(
                        {
                            "phase": cache_observation,
                            "request_id": entry["request_id"],
                            "has_nan": None,
                            "status": "invalid_checkpoint_address",
                            "error": entry["address_error"],
                        }
                    )
            md = attn_metadata[name]
            distribution = md.mamba_request_distribution
            if distribution is None:
                distribution = md.request_distribution
            windowed_end = int(distribution[1].cpu())
            width = (
                runner._mamba_ckpt_window
                if runner._unified_kv_layout
                else runner._mamba_slot_stride
            )
            self.written_counts[gid] = [
                min(count, width) if row < windowed_end else 1
                for row, count in enumerate(scheduled)
            ]
            self.report["gdn_checkpoints"].append(
                {"group_id": gid, "requests": entries}
            )
        if cache_observation == "before_forward":
            self._check_caches("before_forward")

    def _layer_manifest(self, name: str, is_mamba: bool) -> rpm.PoolManifest:
        runner = self.runner
        module = runner.vllm_config.compilation_config.static_forward_context[name]
        assert runner.kv_cache_config is not None
        kwargs = dict(
            named_kv_caches={name: module.kv_cache},
            kv_cache_groups=runner.kv_cache_config.kv_cache_groups,
            raw_tensors=runner.kv_cache_raw_tensors,
        )
        geometry = None
        if is_mamba:
            if not hasattr(module, "tp_head_geometry"):
                # KDA uses the other existing manifest's padded state ABI.
                return rpm.build_kimi_k3_pool_manifest(**kwargs)
            heads = module.tp_head_geometry
            geometry = rpm.GdnHeadGeometry(
                heads.local_num_kq_heads,
                heads.local_num_v_heads,
                module.head_k_dim,
                module.head_v_dim,
                module.conv_kernel_size,
            )
        fa_layout = None
        if not is_mamba and (
            runner.vllm_config.cache_config.get_resolved_kv_cache_layout()
            in (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)
        ):
            from vllm_torchtpu.distributed.kv_transfer.raiden.seq_on_lane import (
                SolFaLayout,
            )

            fa_layout = SolFaLayout()
        return rpm.build_qwen35_pool_manifest(
            **kwargs,
            gdn_geometry=geometry,
            block_major=runner.bm_pool_layout,
            fa_layout=fa_layout,
        )

    def _check_caches(self, phase: str) -> None:
        runner = self.runner
        assert runner.kv_cache_config is not None
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            is_mamba = isinstance(group.kv_cache_spec, MambaSpec)
            if phase == "before_forward" and not is_mamba:
                continue
            blocks: set[int] = set()
            if is_mamba:
                for row, entry in enumerate(self.checkpoints.get(gid, [])):
                    if phase == "before_forward":
                        if "address_error" in entry:
                            self.checks.append(
                                {
                                    "phase": phase,
                                    "group_id": gid,
                                    "request_id": entry["request_id"],
                                    "has_nan": None,
                                    "status": "invalid_checkpoint_address",
                                    "error": entry["address_error"],
                                }
                            )
                            continue
                        if entry["num_computed_tokens"] > 0:
                            blocks.add(entry["read_slot"])
                    else:
                        count = self.written_counts[gid][row]
                        # Prefill writes the base only; verify writes one
                        # checkpoint per actual input, never unused lookahead.
                        slots = entry["checkpoint_slots"] or list(
                            range(entry["slot"], entry["slot"] + count)
                        )
                        blocks.update(slots[:count])
            else:
                for req_id in self.req_ids:
                    blocks.update(runner.requests[req_id].block_ids[gid])
            if not is_mamba:
                blocks.discard(0)
            if not blocks:
                continue
            for name in group.layer_names:
                if name not in self.metadata:
                    continue  # Another pipeline stage owns this layer.
                info = {
                    **self.report,
                    "phase": phase,
                    "group_id": gid,
                    "scope": "gdn_read_state"
                    if phase == "before_forward"
                    else (
                        "gdn_written_checkpoints"
                        if is_mamba
                        else "allocated_attention_pages"
                    ),
                }
                try:
                    manifest = self._layer_manifest(name, is_mamba)
                    for pool in manifest.pools:
                        check = inspect_pool(
                            manifest.storages[pool.storage_index],
                            pool,
                            sorted(blocks),
                            self.directory,
                            info,
                        )
                        self.checks.append({"phase": phase, "group_id": gid, **check})
                except (rpm.ManifestError, NotImplementedError, ValueError) as exc:
                    # A skipped/unsupported layout must never be reported clean.
                    self.checks.append(
                        {
                            "phase": phase,
                            "layer": name,
                            "has_nan": None,
                            "status": "unavailable",
                            "error": str(exc),
                        }
                    )

    def before_logits(
        self, selected_hidden_states: torch.Tensor, real_rows: int
    ) -> dict[str, Any]:
        self.report["hidden_states"] = inspect_tensor(
            selected_hidden_states[:real_rows],
            self.directory / "pre-logits.pt",
            {**self.report, "tensor_origin": "pre_logits"},
        )
        phase = (
            "after_sampling"
            if self.report["cache_observation"] == "after_sampling"
            else "before_logits"
        )
        return self._finish(phase)

    def after_forward(self) -> dict[str, Any]:
        """PP stages without the logits head still check their local cache."""
        self.report["hidden_states"] = {"has_nan": None, "status": "not_logits_stage"}
        return self._finish("after_forward")

    def _finish(self, phase: str) -> dict[str, Any]:
        self._check_caches(phase)
        any_nan = any(c["has_nan"] is True for c in self.checks)
        unavailable = any(c["has_nan"] is None for c in self.checks)
        self.report["kv_cache"] = {
            "has_nan": True if any_nan else (None if unavailable else False),
            "status": "incomplete"
            if unavailable
            else ("checked" if self.checks else "no_related_cache"),
            "checks": self.checks,
            "attention_scope": "allocated_pages_including_unwritten_tail_and_lookahead",
            "attention_active_tokens_has_nan": None,
        }
        return self.report
