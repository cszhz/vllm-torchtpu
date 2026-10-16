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


def all_to_all_qkvba_head_to_token(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    rm_to_packed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused head-to-token all-to-all (and optional token permutation) for GDN [mixed_qkv, b, a].

    Packs rank-major ``mixed_qkv`` ``[L, P * qkv_per_rank]``, ``b`` ``[L, P * v_per_rank]``,
    and ``a`` ``[L, P * v_per_rank]`` into a single ``[P, L, qkv_per_rank + 2 * v_per_rank]``
    send buffer, executes a single ``all_to_all_single``, optionally permutes tokens
    via ``torch.index_select(0, rm_to_packed)``, and splits back into
    ``(mixed_qkv_gdn, b_gdn, a_gdn)`` of shapes ``[P * L, qkv_per_rank]``,
    ``[P * L, v_per_rank]``, and ``[P * L, v_per_rank]``.
    """
    group = get_pcp_group()
    if group is None or int(group.world_size) == 1:
        if rm_to_packed is not None:
            idx = rm_to_packed[: mixed_qkv.size(0)]
            return (
                torch.index_select(mixed_qkv, 0, idx),
                torch.index_select(b, 0, idx),
                torch.index_select(a, 0, idx),
            )
        return mixed_qkv, b, a
    world_size = int(group.world_size)
    L = mixed_qkv.shape[0]
    if mixed_qkv.shape[1] % world_size != 0 or b.shape[1] % world_size != 0:
        raise ValueError(
            f"QKV ({mixed_qkv.shape[1]}) and B/A ({b.shape[1]}) channel dimensions "
            f"must be divisible by PCP world_size {world_size}."
        )
    qkv_per_rank = mixed_qkv.shape[1] // world_size
    v_per_rank = b.shape[1] // world_size
    total_per_rank = qkv_per_rank + 2 * v_per_rank

    qkv_3d = mixed_qkv.reshape(L, world_size, qkv_per_rank)
    b_3d = b.reshape(L, world_size, v_per_rank)
    a_3d = a.reshape(L, world_size, v_per_rank)
    send_buf = (
        torch.cat((qkv_3d, b_3d, a_3d), dim=-1)
        .transpose(0, 1)
        .contiguous()
    )
    recv_buf = all_to_all_single(send_buf).reshape(world_size * L, total_per_rank)
    if rm_to_packed is not None:
        recv_buf = torch.index_select(
            recv_buf, 0, rm_to_packed[: recv_buf.size(0)]
        )
    mixed_qkv_gdn, b_gdn, a_gdn = recv_buf.split(
        [qkv_per_rank, v_per_rank, v_per_rank], dim=-1
    )
    return mixed_qkv_gdn.contiguous(), b_gdn.contiguous(), a_gdn.contiguous()


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
