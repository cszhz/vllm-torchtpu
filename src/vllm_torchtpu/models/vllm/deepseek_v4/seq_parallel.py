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
"""Token-sharded (sequence-parallel) collectives over the TP group.

With TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS, a forward of at least that many tokens
keeps the mHC residual streams split by token over the TP group: each core
holds a contiguous 1/TP of the rows, in TP-rank order -- the same slot order
the fused EP MoE uses to split replicated rows (TokenReplicaLayout). Every
core otherwise repeats the seams' per-token work on all rows.

The ops run over the expert-parallel mesh with the TP groups as
``axis_index_groups``, as ``small_all_reduce_op`` does: torch_tpu wants every
JAX op of a module on the one mesh, and a TP group is smaller than it under
DP x TP. ``take_local`` reads this core's slot from the small TP all-reduce's
rank tensor rather than a tensor of its own: every rank then traces the same
graph, and the token-sharded traces take the same graph inputs, in the same
order, as the other buckets' traces -- the embedding's all-reduce lifts that
tensor first in both -- which ``compilation.shape_variants`` requires to hand
buckets between traces.
"""

import jax
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
from vllm_torchtpu.layers.adapter import fused_moe_ep, small_all_reduce_op
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_AG = None
_RS = None
_TAKE = None
_CORES = 0


def _build_ops(mesh, groups: tuple[tuple[int, ...], ...]):
    spec = PartitionSpec(EP_AXIS_NAME)
    index_groups = [list(g) for g in groups]
    cores = len(groups[0])

    def smap(body, n_in):
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * n_in, out_specs=spec, check_vma=False
        )

    # Every operand annotated jax.Array: torch_tpu infers static argnums from
    # the signature and refuses an unannotated one.
    def all_gather(x: jax.Array, like: jax.Array) -> jax.Array:
        return smap(
            lambda b, _: lax.all_gather(
                b, EP_AXIS_NAME, axis=0, tiled=True, axis_index_groups=index_groups
            ),
            2,
        )(x, like)

    def reduce_scatter(x: jax.Array) -> jax.Array:
        return smap(
            lambda b: lax.psum_scatter(
                b,
                EP_AXIS_NAME,
                scatter_dimension=0,
                tiled=True,
                axis_index_groups=index_groups,
            ),
            1,
        )(x)

    def take(x: jax.Array, slot: jax.Array) -> jax.Array:
        def body(b, s):
            rows = b.shape[0] // cores
            return lax.dynamic_slice_in_dim(b, s[0] * rows, rows, axis=0)

        return smap(body, 2)(x, slot)

    def op(name, fn, n_in, rows):
        o = jax_op(
            f"pallas::dsv4_sp_{name}",
            fn,
            mesh=mesh,
            input_partition_specs=(spec,) * n_in,
            output_partition_specs=spec,
        )
        # The default fake exports fn at the operands' shapes; the row count
        # is known without that.
        o.register_fake(lambda x, *a: x.new_empty((rows(x, *a), *x.shape[1:])))
        return o

    return (
        # Sized off `like` so the gathered rows carry the forward's own token
        # symbol: torch_tpu refuses allocations sized by a derived expression
        # such as cores * (tokens // cores).
        op("all_gather", all_gather, 2, lambda x, like: like.shape[0]),
        op("reduce_scatter", reduce_scatter, 1, lambda x: x.shape[0] // cores),
        op("take", take, 2, lambda x, slot: x.shape[0] // cores),
    )


def prebuild(max_num_tokens: int) -> None:
    """Build the ops outside any traced region (the groups come from a
    collective). A no-op unless the flag is set."""
    global _AG, _RS, _TAKE, _CORES
    min_tokens = envs.TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS
    if min_tokens == 0 or _AG is not None:
        return
    if min_tokens >= max_num_tokens:
        # The token-sharded trace would then hold only the largest bucket:
        # its guards pin the dynamic token dim to one value, which Dynamo
        # refuses for a dim vLLM marked dynamic.
        raise ValueError(
            f"TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS={min_tokens} must be below the "
            f"largest token bucket ({max_num_tokens})"
        )
    mesh = build_ep_mesh()
    if mesh is None:
        raise NotImplementedError(
            "TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS runs its collectives over the "
            "expert-parallel mesh, and this deployment has none. Unset it."
        )
    groups = ep_token_replica_groups()
    me = ep_mesh_index()
    (group,) = [g for g in groups if me in g]
    if fused_moe_ep._RANK_BUFFER is None:
        raise NotImplementedError(
            "TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS needs the fused EP MoE armed: it "
            "returns the MoE's rows token-sharded"
        )
    if small_all_reduce_op._RANK is None or len(group) != small_all_reduce_op._CORES:
        raise NotImplementedError(
            "TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS takes this core's TP slot from "
            "TPU_SMALL_ALL_REDUCE's rank tensor; set TPU_SMALL_ALL_REDUCE=1"
        )
    _CORES = len(group)
    _AG, _RS, _TAKE = _build_ops(mesh, groups)
    logger.info(
        "DSv4 sequence parallelism armed | min_tokens=%d | cores=%d | groups=%s",
        envs.TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS,
        _CORES,
        groups,
    )


def active(num_tokens: int) -> bool:
    """Whether a forward of ``num_tokens`` rows runs token-sharded. Graphs are
    traced per token bucket, so this is a trace-time constant."""
    min_tokens = envs.TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS
    if min_tokens == 0 or num_tokens < min_tokens:
        return False
    if _AG is None:
        raise RuntimeError(
            "TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS is set but seq_parallel.prebuild "
            "did not run before the forward"
        )
    return _CORES > 1 and num_tokens % _CORES == 0


def cores() -> int:
    return _CORES


def take_local(x: torch.Tensor) -> torch.Tensor:
    """This core's rows of a TP-replicated ``x``."""
    assert _TAKE is not None
    return _TAKE(x, small_all_reduce_op._RANK)


def all_gather(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Every core's rows of the token-sharded ``x``, in slot order. ``like``
    is any tensor whose leading dim is the forward's token count (positions):
    the result takes that dim from it."""
    assert _AG is not None
    return _AG(x, like)


def reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """This core's rows of the sum of the TP-partial ``x``."""
    assert _RS is not None
    return _RS(x)
