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
"""Small TP all-reduces through the Pallas two-shot kernel.

With TPU_SMALL_ALL_REDUCE, ``prebuild`` builds one op over the expert-parallel
mesh at model load; ``TpuDeviceCommunicator.all_reduce`` hands it every 2-D
bf16 TP all-reduce whose width splits over the TP cores. The op sums over this
rank's TP group only, so it serves TP groups smaller than the EP mesh (TP8 x
DP2 with EP16): the kernel addresses the group's cores through a per-core peer
table, and torch_tpu wants every Pallas kernel of a module on the one EP mesh.

The kernel/XLA choice is taken inside JAX, where each token bucket is exported
at its concrete shape: up to ``max_rows()`` rows the kernel, above it XLA's
all-reduce over the same groups. No branch on the token count reaches the
torch trace, so the backbone and the drafter take the same op.
"""

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_token_replica_groups,
)
from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Largest block, in rows, the kernel takes. On tpu7x 2x2x2 (two TP8 groups in
# a 16-core program), bf16 [rows, 7168]: kernel vs XLA 9.9/22.2 us at 8 rows,
# 11.8/23.7 at 16, 16.2/25.1 at 32, 25.3/29.4 at 64.
MAX_ROWS = 64

_OP = None
# MOE_FUSED_EP_V2_FOLD_RESTORE's ops (see ``fold``), keyed by routed scale
# ("add") or None ("pair"); the scales are requested at model construction.
_FOLD_OPS: dict = {}
_FOLD_SCALES: set[float] = set()
# This core's position in its TP group (int32 [1]) and the mesh index of each
# member (int32 [cores]), as device tensors: the kernel takes both as data.
_RANK = None
_PEERS = None
_CORES = 0
_TP_RANKS: tuple[int, ...] = ()


def max_rows() -> int:
    """The kernel's row cap: TPU_SMALL_ALL_REDUCE_MAX_ROWS, else MAX_ROWS."""
    return envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS or MAX_ROWS


def group_all_reduce(groups: tuple[tuple[int, ...], ...], cap: int, axis: str):
    """The shard_map body: body(x [rows, hidden], rank int32 [1], peers int32
    [cores]) -> the sum of x over the core's group, by the kernel up to ``cap``
    rows and by XLA above."""
    cores = len(groups[0])
    index_groups = [list(g) for g in groups]

    def body(block, rank, peers):
        if block.shape[0] > cap:
            return lax.psum(block, axis, axis_index_groups=index_groups)
        return all_reduce.small_all_reduce(
            block, num_cores=cores, axis=axis, rank=rank, peers=peers
        )

    return body


def bf16_scale(scale: float) -> float:
    """The scale as a bf16 multiply of a bf16 tensor by a Python float applies
    it: torch_tpu rounds the scalar to bf16 first (measured on device)."""
    return float(torch.tensor(scale, dtype=torch.bfloat16))


def group_fold(
    groups: tuple[tuple[int, ...], ...], cap: int, axis: str, scale: float | None
):
    """The shard_map body: body(x, deferred, rank, peers) -> the sum of x over
    the core's group and the group's deferred MoE rows gathered back: as the
    pair (scale None), or as ``sum + rows * scale`` with the roundings of the
    runner's separate bf16 multiply and add.

    ``deferred`` is a fused EP MoE block under defer_restore: this core's
    slice of the rows at its top, sized as TokenReplicaLayout sizes it for
    these groups. Up to ``cap`` rows one kernel does it all; above it XLA's
    all-reduce and all-gather, as the separate ops would.
    """
    from vllm_torchtpu.kernels.experimental.small_collectives import (
        all_reduce_gather,
    )
    from vllm_torchtpu.kernels.fused_moe.v2.host import U32_SUBLANE_TILE
    from vllm_torchtpu.kernels.fused_moe.v2.token_parallel import TokenReplicaLayout

    cores = len(groups[0])
    index_groups = [list(g) for g in groups]
    add = scale is not None

    def body(block, deferred, rank, peers):
        rows = block.shape[0]
        layout = TokenReplicaLayout.create(
            ep=cores * len(groups),
            input_rows=rows,
            groups=groups,
            row_alignment=U32_SUBLANE_TILE,
        )
        mine = deferred[: layout.local_rows]
        if rows <= cap:
            out = all_reduce_gather.small_all_reduce_gather(
                block,
                mine,
                num_cores=cores,
                axis=axis,
                rank=rank,
                peers=peers,
                add=add,
                scale=scale if add else 1.0,
                round_product=True,
            )
            return out if add else (out[0], out[1][:rows])
        total = lax.psum(block, axis, axis_index_groups=index_groups)
        gathered = lax.all_gather(
            mine, axis, axis=0, tiled=True, axis_index_groups=index_groups
        )[:rows]
        if not add:
            return total, gathered
        return total + gathered * jnp.asarray(scale, gathered.dtype)

    return body


def _build_fold_op(mesh, groups, cap, scale: float | None):
    spec = PartitionSpec(EP_AXIS_NAME)
    body = group_fold(groups, cap, EP_AXIS_NAME, scale)
    out_specs = spec if scale is not None else (spec, spec)
    sm = jax.shard_map(
        body, mesh=mesh, in_specs=(spec,) * 4, out_specs=out_specs, check_vma=False
    )

    def pair(
        x: jax.Array, deferred: jax.Array, rank: jax.Array, peers: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return sm(x, deferred, rank, peers)

    def add(
        x: jax.Array, deferred: jax.Array, rank: jax.Array, peers: jax.Array
    ) -> jax.Array:
        return sm(x, deferred, rank, peers)

    op = jax_op(
        "pallas::small_tp_all_reduce_gather"
        + ("" if scale is None else f"_add_{len(_FOLD_OPS)}"),
        pair if scale is None else add,
        mesh=mesh,
        input_partition_specs=(spec,) * 4,
        output_partition_specs=out_specs,
    )
    if scale is None:
        op.register_fake(lambda x, d, r, p: (torch.empty_like(x), torch.empty_like(x)))
    else:
        op.register_fake(lambda x, d, r, p: torch.empty_like(x))
    return op


def _build_op(mesh, groups: tuple[tuple[int, ...], ...], cap: int):
    """The torch op over ``mesh``: fn(x, rank, peers) per core, see
    ``group_all_reduce``."""
    spec = PartitionSpec(EP_AXIS_NAME)
    body = group_all_reduce(groups, cap, EP_AXIS_NAME)

    # Every operand annotated jax.Array: torch_tpu infers static argnums from
    # the signature and refuses an unannotated one.
    def fn(x: jax.Array, rank: jax.Array, peers: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False
        )(x, rank, peers)

    op = jax_op(
        "pallas::small_tp_all_reduce",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 3,
        output_partition_specs=spec,
    )
    # The default fake exports fn at the operands' shapes, which would pin the
    # caller's dynamic token dimension; the output is the input's shape.
    op.register_fake(lambda x, rank, peers: torch.empty_like(x))
    return op


def prebuild() -> None:
    """Build the op and this core's tables, outside any traced region (the
    groups come from a collective). A no-op unless TPU_SMALL_ALL_REDUCE."""
    global _OP, _RANK, _PEERS, _CORES, _TP_RANKS
    if envs.MOE_FUSED_EP_V2_FOLD_RESTORE and not envs.TPU_SMALL_ALL_REDUCE:
        raise ValueError(
            "MOE_FUSED_EP_V2_FOLD_RESTORE gathers the MoE rows inside the small "
            "TP all-reduce; set TPU_SMALL_ALL_REDUCE=1 as well."
        )
    if not envs.TPU_SMALL_ALL_REDUCE or _OP is not None:
        return
    mesh = build_ep_mesh()
    if mesh is None:
        raise NotImplementedError(
            "TPU_SMALL_ALL_REDUCE=1 runs its kernel over the expert-parallel "
            "mesh, and this deployment has no expert-parallel group. Unset "
            "TPU_SMALL_ALL_REDUCE."
        )
    from vllm.distributed.parallel_state import get_tp_group

    # The TP groups in mesh coordinates, the same tuple on every rank.
    groups = ep_token_replica_groups()
    me = ep_mesh_index()
    (group,) = [g for g in groups if me in g]
    _CORES = len(group)
    _TP_RANKS = tuple(int(r) for r in get_tp_group().ranks)
    _RANK = torch.tensor([group.index(me)], dtype=torch.int32, device="tpu")
    _PEERS = torch.tensor(group, dtype=torch.int32, device="tpu")
    _OP = _build_op(mesh, groups, max_rows())
    mode = envs.MOE_FUSED_EP_V2_FOLD_RESTORE
    for scale in sorted(_FOLD_SCALES) if mode == "add" else [None] * bool(mode):
        _FOLD_OPS[scale] = _build_fold_op(
            mesh, groups, max_rows(), None if scale is None else bf16_scale(scale)
        )
    logger.info(
        "Small TP all-reduce armed | cores=%d | groups=%s | max_rows=%d | "
        "one_shot_max_bytes=%d | fold_restore=%s",
        _CORES,
        groups,
        max_rows(),
        all_reduce.ONE_SHOT_MAX_BYTES,
        f"{mode or 'off'} {sorted(_FOLD_OPS, key=str)}",
    )


def all_reduce_or(x: torch.Tensor, ranks, fallback) -> torch.Tensor:
    """The sum of ``x`` over the group ``ranks``: the op for a 2-D bf16 block
    of the TP group in a compiled trace, ``fallback(x)`` otherwise."""
    if _OP is None or tuple(ranks) != _TP_RANKS:
        return fallback(x)
    if (
        not torch.compiler.is_compiling()
        or x.dim() != 2
        or x.dtype != torch.bfloat16
        or not all_reduce.supports(1, x.shape[1], _CORES)
    ):
        return fallback(x)
    return _OP(x, _RANK, _PEERS)


def request_fold(scale: float) -> None:
    """Called at MoE layer construction, before ``prebuild``: the routed
    scale the "add" form must build an op for."""
    _FOLD_SCALES.add(float(scale))


def fold(
    x: torch.Tensor, deferred: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """The TP-group sum of ``x`` with the group's deferred fused EP MoE rows
    gathered back in token order: (sum, rows) under the "pair" form, else
    ``sum + rows * scale`` rounded as the runner's bf16 ops round it. See
    ``group_fold``. Only with MOE_FUSED_EP_V2_FOLD_RESTORE armed: the rows
    have nowhere else to be gathered, so an unarmed call is an error."""
    mode = envs.MOE_FUSED_EP_V2_FOLD_RESTORE
    key = float(scale) if mode == "add" else None
    if key not in _FOLD_OPS:
        raise RuntimeError(
            f"deferred MoE rows reached the TP reduction with no "
            f"MOE_FUSED_EP_V2_FOLD_RESTORE={mode!r} op built for {key}"
        )
    if (
        x.dim() != 2
        or deferred.dim() != 2
        or x.shape[1] != deferred.shape[1]
        or x.dtype != torch.bfloat16
        or deferred.dtype != torch.bfloat16
        or not all_reduce.supports(1, x.shape[1], _CORES)
    ):
        raise ValueError(
            f"fold restore takes two bf16 [rows, hidden] blocks with hidden a "
            f"multiple of {_CORES} x 128; got {tuple(x.shape)} {x.dtype} and "
            f"{tuple(deferred.shape)} {deferred.dtype}"
        )
    if torch.compiler.is_compiling():
        return _FOLD_OPS[key](x, deferred, _RANK, _PEERS)
    # Eager, like all_reduce_or: the group's own collectives, same values.
    from vllm.distributed.parallel_state import get_tp_group

    from vllm_torchtpu.kernels.fused_moe.v2.host import U32_SUBLANE_TILE
    from vllm_torchtpu.kernels.fused_moe.v2.token_parallel import TokenReplicaLayout

    rows = x.shape[0]
    local = TokenReplicaLayout.create(
        ep=_CORES,
        input_rows=rows,
        groups=(tuple(range(_CORES)),),
        row_alignment=U32_SUBLANE_TILE,
    ).local_rows
    tp = get_tp_group()
    total = tp.all_reduce(x)
    gathered = tp.all_gather(deferred[:local].contiguous(), dim=0)[:rows]
    if key is None:
        return total, gathered
    return total + gathered * scale
