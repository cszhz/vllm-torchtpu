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
"""A small all-reduce and a small all-gather over one group, as one kernel.

The MoE layer of a TP group ends in two collectives over the same cores: the
shared expert's TP-partial output is summed, and the routed rows each core
computed for its slice of the tokens are gathered back. Issued separately each
pays its own launch, barrier and round-trip latency, which at decode sizes is
most of its cost. Here one barrier serves both, the gather's pushes go out in
the same round as the sum's first round, and the results are bit-identical to
``all_reduce.small_all_reduce`` and ``all_gather.small_all_gather``.

With ``add`` the gathered rows, scaled, are added to the sum on the owning core
before the sum's second round instead of being returned, so one tensor comes
back, rounded as a separate bf16 multiply and add would round it. The gather
then sends each owner only its columns of the rows.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.all_reduce import (
    COLLECTIVE_ID,
    supports,
)
from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    barrier,
    push,
    scalar_operands,
    split_refs,
    sum_slots,
    wait_bytes_of,
)


def _scaled(rows_f32, scale, round_product, dtype):
    if scale == 1.0:
        return rows_f32
    rows_f32 = rows_f32 * jnp.float32(scale)
    return rows_f32.astype(dtype).astype(jnp.float32) if round_product else rows_f32


def _kernel_twoshot(
    n, rows, yrows, own, axis, rank_op, peers_op, add, scale, round_product
):
    def kernel(*refs):
        me, group, refs = split_refs(refs, n, axis, rank_op, peers_op)
        if add:
            (x_ref, y_ref, o_ref), g_ref, rest = refs[:3], None, refs[3:]
        else:
            (x_ref, y_ref, o_ref, g_ref), rest = refs[:4], refs[4:]
        mine, slices, gathered, yslots, ymine, acc, *sems = rest
        send1, recv1, send2, recv2, send3, recv3 = sems
        barrier(group)
        if add:  # owner j only needs its columns of everyone's rows
            for j in range(n):
                ymine[j] = y_ref[:, pl.ds(j * own, own)]
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                src = ymine.at[j] if add else y_ref
                push(src, yslots.at[me], send3, recv3, j, group)

        for j in range(n):
            mine[j] = x_ref[:, pl.ds(j * own, own)]
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(mine.at[j], slices.at[me], send1, recv1, j, group)

        slices[pl.ds(me, 1)] = mine[pl.ds(me, 1)]
        if add:
            yslots[pl.ds(me, 1)] = ymine[pl.ds(me, 1)]
        else:
            yslots[pl.ds(me, 1)] = y_ref[...][None]
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv1)
        total = sum_slots(slices, n).astype(gathered.dtype)
        wait_bytes_of(yslots.at[pl.ds(0, n - 1)], recv3)
        if add:
            # The sum is rounded before the add, as a separate all-reduce's is.
            acc[...] = total.astype(jnp.float32)
            for i in range(n):
                k = min(yrows, rows - i * yrows)
                if k > 0:
                    acc[pl.ds(i * yrows, k)] += _scaled(
                        yslots[i][:k].astype(jnp.float32),
                        scale,
                        round_product,
                        gathered.dtype,
                    )
            total = acc[...].astype(gathered.dtype)
        gathered[pl.ds(me, 1)] = total[None]
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(gathered.at[me], gathered.at[me], send2, recv2, j, group)

        if not add:
            for i in range(n):
                g_ref[pl.ds(i * yrows, yrows)] = yslots[i]
        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], recv2)
        for j in range(n):
            o_ref[:, pl.ds(j * own, own)] = gathered[j]
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], send1)
        wait_bytes_of(yslots.at[pl.ds(0, n - 1)], send3)
        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], send2)

    return kernel


def small_all_reduce_gather(
    x,
    y,
    *,
    num_cores,
    axis,
    rank,
    peers,
    add=False,
    scale=1.0,
    round_product=True,
    collective_id=COLLECTIVE_ID,
):
    """The group sum of ``x`` and the group all-gather of ``y``, inside a
    shard_map over mesh axis ``axis``.

    Args:
      x: [rows, hidden], this core's partial; hidden a multiple of
        num_cores * LANES.
      y: [yrows, hidden], this core's rows, with num_cores * yrows >= rows.
      rank, peers: this core's group position (int32 [1]) and the group's mesh
        indices (int32 [num_cores]) as data, see ``common``.
      add: return ``sum + scale * gathered[:rows]`` instead of the pair, the
        sum rounded to the input dtype first as a separate all-reduce's is,
        the product rounded too if ``round_product``, and the add rounded.

    Returns:
      (sum [rows, hidden], gathered [num_cores * yrows, hidden]), or the one
      tensor with ``add``.
    """
    rows, hidden = x.shape
    yrows = y.shape[0]
    n = num_cores
    if not supports(rows, hidden, n) or y.shape[1] != hidden or y.dtype != x.dtype:
        raise ValueError(
            f"x {x.shape} {x.dtype} / y {y.shape} {y.dtype} over {n} cores"
        )
    if n * yrows < rows:
        raise ValueError(f"{n} x {yrows} gathered rows do not cover {rows}")
    own = hidden // n
    kernel = _kernel_twoshot(
        n,
        rows,
        yrows,
        own,
        axis,
        rank is not None,
        peers is not None,
        add,
        float(scale),
        bool(round_product),
    )
    out = jax.ShapeDtypeStruct((rows, hidden), x.dtype)
    gathered = jax.ShapeDtypeStruct((n * yrows, hidden), y.dtype)
    out_shape = out if add else (out, gathered)
    yslot_shape = (n, yrows, own) if add else (n, yrows, hidden)
    scratch = [
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM(yslot_shape, y.dtype),
        pltpu.VMEM((n, yrows, own) if add else (1, 8, 128), y.dtype),
        pltpu.VMEM((rows, own) if add else (8, 128), jnp.float32),
        *[pltpu.SemaphoreType.DMA] * 6,
    ]
    args, specs = scalar_operands(rank, peers)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=[*specs, pl.BlockSpec(), pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"small_all_reduce_gather{'_add' if add else ''}_n{n}_r{rows}_y{yrows}",
    )(*args, x, y)
