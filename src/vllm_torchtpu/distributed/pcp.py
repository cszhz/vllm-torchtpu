# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu.distributed.mesh_utils import (
    CpGroupLayout,
    get_cp_group_layout,
    get_or_create_cp_mesh,
)

PcpGroupLayout = CpGroupLayout


def get_pcp_group() -> Any | None:
    """Return vLLM's native PCP group, or None before distributed init."""
    try:
        from vllm.distributed.parallel_state import get_pcp_group as _get_group

        return _get_group()
    except (AssertionError, ImportError):
        return None


def get_pcp_rank() -> int:
    group = get_pcp_group()
    return 0 if group is None else int(group.rank_in_group)


def get_pcp_world_size() -> int:
    group = get_pcp_group()
    return 1 if group is None else int(group.world_size)


def all_gather_equal_tokens(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Gather equal-shaped tensors across the native PCP group."""
    group = get_pcp_group()
    if group is None or int(group.world_size) == 1:
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    gathered = group.all_gather(tensor, dim=dim)
    if isinstance(gathered, torch.Tensor):
        return gathered
    return torch.cat(list(gathered), dim=dim)


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across the native PCP group."""
    group = get_pcp_group()
    if group is None or int(group.world_size) == 1:
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return group.all_reduce(tensor)


def all_to_all_single(tensor: torch.Tensor) -> torch.Tensor:
    """All-to-all exchange along the leading dimension of ``tensor`` across the PCP group.

    Block ``d`` of ``tensor``'s leading axis goes to rank ``d``; block ``r`` of
    the result came from rank ``r``.
    """
    group = get_pcp_group()
    if group is None or int(group.world_size) == 1:
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if hasattr(group, "all_to_all_single"):
        return group.all_to_all_single(tensor)
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(tensor, None, None, group.device_group)
    return funcol.wait_tensor(out)


def all_to_all_head_to_token(tensor: torch.Tensor) -> torch.Tensor:
    """Exchange ``tensor`` from head-replicated / token-sharded ``[L, C, ...]``
    to token-all / head-sharded ``[P * L, C // P, ...]`` across the PCP group.
    """
    group = get_pcp_group()
    if group is None or int(group.world_size) == 1:
        return tensor
    world_size = int(group.world_size)
    L = tensor.shape[0]
    if tensor.shape[1] % world_size != 0:
        raise ValueError(
            f"Head/channel dimension {tensor.shape[1]} must be divisible by "
            f"PCP world_size {world_size}."
        )
    c_per_rank = tensor.shape[1] // world_size
    rest = tensor.shape[2:]
    send_buf = (
        tensor.reshape(L, world_size, c_per_rank, *rest)
        .transpose(0, 1)
        .contiguous()
    )
    recv_buf = all_to_all_single(send_buf)
    return recv_buf.reshape(world_size * L, c_per_rank, *rest)


def all_to_all_token_to_head(tensor: torch.Tensor) -> torch.Tensor:
    """Exchange ``tensor`` from token-all / head-sharded ``[P * L, C_local, ...]``
    back to token-sharded / head-replicated ``[L, P * C_local, ...]`` across the PCP group.
    """
    group = get_pcp_group()
    if group is None or int(group.world_size) == 1:
        return tensor
    world_size = int(group.world_size)
    if tensor.shape[0] % world_size != 0:
        raise ValueError(
            f"Leading token dimension {tensor.shape[0]} must be divisible by "
            f"PCP world_size {world_size}."
        )
    L = tensor.shape[0] // world_size
    c_per_rank = tensor.shape[1]
    rest = tensor.shape[2:]
    send_buf = tensor.reshape(world_size, L, c_per_rank, *rest).contiguous()
    recv_buf = all_to_all_single(send_buf)
    return (
        recv_buf.transpose(0, 1)
        .contiguous()
        .reshape(L, world_size * c_per_rank, *rest)
    )


def get_pcp_cache_rank() -> int:
    """
    Index of the PCP chunk whose KV this worker's device physically holds.
    """
    from torch_tpu._internal.distributed import tpu_distributed

    group = get_pcp_group()
    if group is None or int(group.world_size) == 1:
        return 0
    world_size = int(group.world_size)

    device_id = int(tpu_distributed.global_device_id())
    device_ids = [int(value) for value in tpu_distributed.all_global_device_ids()]
    # A slice wider than the PCP world means the kernels' partition space is
    # not the PCP group, so no index into it would mean what callers expect.
    if len(device_ids) != world_size:
        raise RuntimeError(
            f"torch_tpu exposes {len(device_ids)} devices {device_ids} but the "
            f"PCP group has world_size={world_size}; the kernels' partition "
            f"space does not match the PCP group, so the chunk this worker "
            f"holds is undefined."
        )
    if device_id not in device_ids:
        raise RuntimeError(
            f"This worker's TPU device id {device_id} is not in torch_tpu's "
            f"device list {device_ids}."
        )
    return device_ids.index(device_id)


def get_pcp_group_layout() -> PcpGroupLayout:
    """Return the current PCP group's rank order and TPU device-id order.

    The order of ``device_ids`` mirrors ``get_pcp_group().ranks``. This is the
    order a PCP Pallas mesh should use so TP+PCP layouts form one PCP ring per
    TP lane.
    """
    return get_cp_group_layout(get_pcp_group())


def get_or_create_pcp_mesh(axis_name: str = "pcp") -> Any:
    """Build a JAX mesh ordered by the current vLLM PCP group.

    This helper is intentionally op-local: it does not replace the runner's
    normal single-device model mesh.
    """
    return get_or_create_cp_mesh(axis_name=axis_name, group=get_pcp_group())
