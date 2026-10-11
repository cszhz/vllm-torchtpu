# SPDX-License-Identifier: Apache-2.0
"""One bounded, synchronized request snapshot after a real hidden NaN.

Runs on the model execution thread before sampling or another cache mutation.
The disabled path owns no state; the enabled path stops scanning after its first
snapshot. GDN pool carrier bytes must be restored before interpreting state
dtypes: a contiguous CPU FP8 view is not the TPU's packed word order.
"""

from __future__ import annotations

import io
import math
import os
import socket
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.distributed.kv_transfer.raiden.tags import class_tag
from vllm_torchtpu.kernels.gdn.head_geometry import derive_gdn_head_geometry
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.mtp_tensor_diagnostics import checkpoint_snapshot
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutPlan
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)
MAX_DUMP_BYTES = 64 * 1024**2
# Leave room for serialization and metadata; the serialized size is also checked.
MAX_TENSOR_BYTES = MAX_DUMP_BYTES - 1024**2
STATE_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def _cpu_snapshot(tensor: torch.Tensor) -> torch.Tensor:
    synchronize_tensors([tensor], wait=True)
    return tensor.to(device="cpu", copy=True)


def pool_physical_bytes(cpu: torch.Tensor) -> torch.Tensor:
    """Restore native [..., packing, lanes] words, including multi-tile lanes.

    Native pool packing is 32 / element_bits; each TPU lane tile has 128 words.
    This is an ABI conversion, not a numerical cast of the FP8 carrier.
    """
    packing = 4 // cpu.element_size()
    assert cpu.shape[-2] == packing and cpu.shape[-1] % 128 == 0, (
        f"Native pool packing invariant violated: {cpu.shape}, {cpu.dtype}"
    )
    raw = cpu.contiguous().view(torch.uint8)
    # Include the bytes of each element in the word, after its packing index.
    tiles = raw.reshape(-1, packing, cpu.shape[-1] // 128, 128, cpu.element_size())
    return tiles.permute(0, 2, 3, 1, 4).contiguous().reshape(-1)


def decode_pool_regions(
    raw: torch.Tensor, pool: rpm.PoolEntry, base_offset: int, dtype: torch.dtype
) -> torch.Tensor:
    """Read live region bytes from one selected manager page, excluding padding."""
    pieces = []
    for region in pool.regions:
        start = base_offset + region.offset_bytes
        size = region.unit_bytes * region.units_per_stride
        end = base_offset + region.extent_end_bytes
        assert 0 <= start < end <= raw.numel(), (
            f"GDN region outside selected page: {pool.layer_name}, "
            f"{region.name}, [{start}, {end}), bytes={raw.numel()}"
        )
        pieces.extend(
            raw[
                start + i * region.stride_bytes : start + i * region.stride_bytes + size
            ]
            for i in range(region.num_units)
        )
    return torch.cat(pieces).view(dtype)


def bad_request_rows(
    flags: torch.Tensor,
    req_ids: tuple[str, ...],
    query_start_loc: np.ndarray,
    plan: SequenceLayoutPlan,
) -> dict[str, list[int]]:
    token_indices = plan._packed_to_request_major_token_indices
    if token_indices is None:
        token_indices = np.arange(plan.global_padded_num_tokens)
    token_indices = token_indices[plan.token_slice]
    assert len(token_indices) == flags.numel(), "Hidden rows must match chunk layout"
    owners = np.searchsorted(query_start_loc[1:], token_indices, side="right")
    result: dict[str, list[int]] = {}
    for row in flags.nonzero().flatten().tolist():
        owner = int(owners[row])
        if token_indices[row] >= 0 and owner < len(req_ids):
            result.setdefault(req_ids[owner], []).append(row)
    return result


def request_cache_entries(
    runner: TPUModelRunner,
    req_id: str,
    metadata: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Copy only the abnormal request's rank-local FA pages and GDN slots."""
    assert runner.kv_cache_config is not None, "Real forward requires cache config"
    context = runner.vllm_config.compilation_config.static_forward_context
    attn = metadata["attn_metadata"]
    live = [
        tensor
        for name in attn
        for tensor in (
            context[name].kv_cache
            if isinstance(context[name].kv_cache, (list, tuple))
            else [context[name].kv_cache]
        )
    ]
    # Metadata contains lazy device tensors too; sync them before slot CPU reads.
    for md in attn.values():
        live.extend(
            t
            for t in (
                md.mamba_state_indices,
                md.mamba_slot_read_offsets,
                md.mamba_ckpt_indices,
                md.seq_lens,
                md.block_tables,
                md.mamba_request_distribution,
                md.request_distribution,
            )
            if isinstance(t, torch.Tensor)
        )
    synchronize_tensors(live, wait=True)
    req_ids = metadata["req_ids"]
    row = req_ids.index(req_id)
    scheduled = metadata["scheduled"]
    layers = []
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        layers.extend(
            (name, gid, group.kv_cache_spec)
            for name in group.layer_names
            if name in attn
        )
    for name, gid, spec in sorted(layers):
        module = context[name]
        cache = module.kv_cache
        info: dict[str, Any] = {"layer": name, "group_id": gid}
        if not isinstance(spec, MambaSpec):
            tensor = cache
            # Scheduler blocks contain consecutive kernel pages on both layouts.
            page_bytes = math.prod(tensor.shape[1:]) * tensor.element_size()
            pages = spec.page_size_bytes // page_bytes
            assert pages > 0 and spec.page_size_bytes % page_bytes == 0, (
                f"FA manager page must contain whole kernel pages: {name}"
            )
            for block in runner.requests[req_id].block_ids[gid]:
                if block != 0:  # vLLM's shared null block is not request history.
                    yield {
                        **info,
                        "kind": "fa",
                        "block_ids": [block],
                        "scope": "allocated_pages_including_tail_padding",
                        "tensor": _cpu_snapshot(
                            tensor[block * pages : (block + 1) * pages]
                        ),
                    }
            continue
        md = attn[name]
        snapshot = checkpoint_snapshot(md, list(req_ids), scheduled, {}, {})[row]
        distribution = md.mamba_request_distribution
        if distribution is None:
            distribution = md.request_distribution
        windowed_end = int(distribution[1].cpu())
        width = (
            runner._mamba_ckpt_window
            if runner._unified_kv_layout
            else runner._mamba_slot_stride
        )
        count = min(scheduled[row], width) if row < windowed_end else 1
        written = snapshot["checkpoint_slots"] or list(
            range(snapshot["slot"], snapshot["slot"] + count)
        )
        blocks = set(written[:count])
        if snapshot["num_computed_tokens"] > 0 and snapshot["read_slot"] is not None:
            blocks.add(snapshot["read_slot"])
        info.update(checkpoint=snapshot, scope="gdn_read_and_written_slots")
        if len(cache) == 2:
            for kind, tensor in zip(("gdn_conv", "gdn_ssm"), cache):
                for block in sorted(blocks):
                    yield {
                        **info,
                        "kind": kind,
                        "block_ids": [block],
                        "tensor": _cpu_snapshot(tensor[block]),
                    }
            continue
        tp_heads = module.tp_head_geometry
        # The model localizes TP-owned state heads again across PCP ranks.
        heads = derive_gdn_head_geometry(
            tp_heads.local_num_kq_heads,
            tp_heads.local_num_v_heads,
            runner.parallel_config.prefill_context_parallel_size,
        )
        manifest = rpm.build_qwen35_pool_manifest(
            named_kv_caches={name: cache},
            kv_cache_groups=runner.kv_cache_config.kv_cache_groups,
            raw_tensors=runner.kv_cache_raw_tensors,
            gdn_geometry=rpm.GdnHeadGeometry(
                heads.local_num_kq_heads,
                heads.local_num_v_heads,
                module.head_k_dim,
                module.head_v_dim,
                module.conv_kernel_size,
            ),
        )
        tensor = cache[0]
        page_bytes = math.prod(tensor.shape[1:]) * tensor.element_size()
        pages = spec.page_size_bytes // page_bytes
        for block in sorted(blocks):
            raw = pool_physical_bytes(
                _cpu_snapshot(tensor[block * pages : (block + 1) * pages])
            )
            for pool in manifest.pools:
                yield {
                    **info,
                    "kind": class_tag(pool.tag),
                    "block_ids": [block],
                    "regions": [region.to_dict() for region in pool.regions],
                    "tensor": decode_pool_regions(
                        raw,
                        pool,
                        pool.base_offset_bytes,
                        STATE_DTYPES[pool.dtype_tag],
                    ),
                }


class HiddenNanKVDumper:
    def __init__(self) -> None:
        self.attempted = False
        self._paused = False

    @contextmanager
    def pause(self) -> Iterator[None]:
        """Exclude startup/profile execution without consuming the live quota."""
        previous = self._paused
        self._paused = True
        try:
            yield
        finally:
            self._paused = previous

    def capture(
        self,
        runner: TPUModelRunner,
        hidden: torch.Tensor,
        req_ids: tuple[str, ...],
        query_start_loc: np.ndarray,
        plan: SequenceLayoutPlan,
        attn_metadata: dict[str, Any],
    ) -> None:
        if self.attempted or self._paused:
            return
        flags = hidden.isnan().any(dim=-1)
        synchronize_tensors([hidden, flags], wait=True)
        bad = bad_request_rows(flags.cpu(), req_ids, query_start_loc, plan)
        if not bad:
            return
        self.attempted = True
        req_id, rows = next(iter(bad.items()))
        capture_id = uuid.uuid4().hex
        directory = (
            Path(tempfile.gettempdir()) / f"vllm-hidden-nan-{os.getpid()}-{capture_id}"
        )
        selected = _cpu_snapshot(hidden[rows[:8]])
        report: dict[str, Any] = {
            "metadata": {
                "request_id": req_id,
                "other_bad_request_ids": list(bad)[1:],
                "hidden_local_rows": rows,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "rank": runner.parallel_config.rank,
                "data_parallel_index": runner.parallel_config.data_parallel_index,
                "capture_phase": "after_main_forward_before_sampling",
                "coverage": "rank_local_layers",
                "max_bytes": MAX_DUMP_BYTES,
                "truncated": False,
            },
            "hidden": selected,
            "caches": [],
        }
        budget = MAX_TENSOR_BYTES - selected.numel() * selected.element_size()
        cache_metadata = {
            "attn_metadata": attn_metadata,
            "req_ids": req_ids,
            "scheduled": np.diff(query_start_loc).tolist(),
        }
        for entry in request_cache_entries(runner, req_id, cache_metadata):
            tensor = entry["tensor"]
            size = tensor.numel() * tensor.element_size()
            if size > budget:
                report["metadata"]["truncated"] = True
                report["metadata"]["first_omitted"] = {
                    key: entry[key] for key in ("kind", "layer", "block_ids")
                }
                break
            values = tensor.float()
            entry.update(
                dtype=str(tensor.dtype),
                shape=list(tensor.shape),
                nan_count=int(values.isnan().sum()),
                inf_count=int(values.isinf().sum()),
            )
            report["caches"].append(entry)
            budget -= size
        buffer = io.BytesIO()
        torch.save(report, buffer)
        if buffer.tell() > MAX_DUMP_BYTES:
            logger.error(
                "Hidden NaN KV dump exceeded limit: request_id=%s bytes=%d",
                req_id,
                buffer.tell(),
            )
            return
        try:
            directory.mkdir(mode=0o700)
            path = directory / "request.pt"
            path.write_bytes(buffer.getbuffer())
        except OSError:
            logger.exception("Hidden NaN KV dump write failed: request_id=%s", req_id)
            return
        logger.error(
            "Hidden NaN KV dump: request_id=%s rank=%d path=%s bytes=%d "
            "entries=%d truncated=%s",
            req_id,
            runner.parallel_config.rank,
            path,
            buffer.tell(),
            len(report["caches"]),
            report["metadata"]["truncated"],
        )
