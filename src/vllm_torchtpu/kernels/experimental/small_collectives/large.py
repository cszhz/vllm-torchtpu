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
"""Reduce-scatter, all-gather and all-reduce of a large [rows, hidden] block
over a group of cores, as Pallas kernels that move HBM to remote HBM directly.

    reduce_scatter  Every core pushes member j's row slice to member j, which
                    sums the arrivals in float32 in group order and rounds
                    once -- the small all-reduce's order, so every member of
                    an all-reduce built from it gets the same bits.
    all_gather      Every core pushes its slice into its slot of every member.

All pushes of a phase are in flight at once; the fabric routes them. The
group is a peer table as data, as in ``common``.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    barrier,
    push,
    scalar_operands,
    split_refs,
    wait_bytes_of,
)

RS_COLLECTIVE_ID = 8
AG_COLLECTIVE_ID = 9
# Rows of each member's slice summed per pipeline step.
REDUCE_TILE_ROWS = 32


def _hbm():
    return pl.BlockSpec(memory_space=pl.ANY)


def _pinned_hbm():
    # Buffers a peer writes into, or a DMA still reads, after the kernel that
    # started it returned: XLA must not place them in VMEM, which the kernels
    # in between reuse.
    return pl.BlockSpec(memory_space=pltpu.HBM)


def _pieces(rows, chunks):
    """Row ranges splitting ``rows`` into ``chunks`` DMAs of 16-row multiples."""
    step = max(16, -(-rows // chunks) // 16 * 16)
    return [(a, min(step, rows - a)) for a in range(0, rows, step)]


def _push_chunked(src, dst, send, recv, j, group, rows, chunks):
    for a, k in _pieces(rows, chunks):
        push(src.at[pl.ds(a, k)], dst.at[pl.ds(a, k)], send, recv, j, group)


def reduce_scatter(
    x, *, num_cores, axis, rank, peers, tile_rows=REDUCE_TILE_ROWS, chunks=1
):
    """[rows / num_cores, hidden]: this member's slice of the group sum of x."""
    rows, hidden = x.shape
    n = num_cores
    if rows % n:
        raise ValueError(f"{rows} rows do not split over {n} cores")
    r = rows // n
    tr = min(tile_rows, r)
    while r % tr:
        tr //= 2
    x3 = x.reshape(n, r, hidden)

    def kernel(*refs):
        me, group, (x_ref, o_ref, land, send, recv, local) = split_refs(
            refs, n, axis, True, True
        )
        barrier(group)
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                _push_chunked(x_ref.at[j], land.at[me], send, recv, j, group, r, chunks)

        mine = pltpu.make_async_copy(x_ref.at[me], land.at[me], local)
        mine.start()
        mine.wait()
        wait_bytes_of(land.at[pl.ds(0, n - 1)], recv)

        def tile(in_ref, out_ref):
            acc = in_ref[0].astype(jnp.float32)
            for i in range(1, n):
                acc = acc + in_ref[i].astype(jnp.float32)
            out_ref[...] = acc.astype(out_ref.dtype)

        pltpu.emit_pipeline(
            tile,
            grid=(r // tr,),
            in_specs=[pl.BlockSpec((n, tr, hidden), lambda t: (0, t, 0))],
            out_specs=[pl.BlockSpec((tr, hidden), lambda t: (t, 0))],
        )(land, o_ref)
        wait_bytes_of(land.at[pl.ds(0, n - 1)], send)

    args, specs = scalar_operands(rank, peers)
    out, _ = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((r, hidden), x.dtype),
            jax.ShapeDtypeStruct((n, r, hidden), x.dtype),
        ),
        in_specs=[*specs, _hbm()],
        out_specs=(_hbm(), _hbm()),
        scratch_shapes=[pltpu.SemaphoreType.DMA] * 3,
        compiler_params=pltpu.CompilerParams(
            collective_id=RS_COLLECTIVE_ID, has_side_effects=True
        ),
        name=f"large_reduce_scatter_n{n}_r{rows}",
    )(*args, x3)
    return out


def all_gather(x, *, num_cores, axis, rank, peers, chunks=1):
    """[num_cores * rows, hidden]: every member's x in group order."""
    r, hidden = x.shape
    n = num_cores

    def kernel(*refs):
        me, group, (x_ref, o_ref, send, recv, local) = split_refs(
            refs, n, axis, True, True
        )
        barrier(group)
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                _push_chunked(x_ref, o_ref.at[me], send, recv, j, group, r, chunks)

        mine = pltpu.make_async_copy(x_ref, o_ref.at[me], local)
        mine.start()
        mine.wait()
        wait_bytes_of(o_ref.at[pl.ds(0, n - 1)], recv)
        wait_bytes_of(o_ref.at[pl.ds(0, n - 1)], send)

    args, specs = scalar_operands(rank, peers)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n, r, hidden), x.dtype),
        in_specs=[*specs, _hbm()],
        out_specs=_hbm(),
        scratch_shapes=[pltpu.SemaphoreType.DMA] * 3,
        compiler_params=pltpu.CompilerParams(
            collective_id=AG_COLLECTIVE_ID, has_side_effects=True
        ),
        name=f"large_all_gather_n{n}_r{r}",
    )(*args, x)
    return out.reshape(n * r, hidden)


def all_reduce(x, *, num_cores, axis, rank, peers, chunks=1):
    """The group sum of x, the same bits on every member."""
    kw = dict(num_cores=num_cores, axis=axis, rank=rank, peers=peers, chunks=chunks)
    return all_gather(reduce_scatter(x, **kw), **kw)


ASYNC_RS_COLLECTIVE_ID = 10


def reduce_scatter_start(x, *, num_cores, axis, rank, peers, chunks=1):
    """Issue the pushes of ``reduce_scatter`` and return without waiting.

    Returns a handle for ``reduce_scatter_done``. Between the two, the core is
    free for other kernels while the slices travel; ``x`` must not be written
    until done (the handle keeps it alive).
    """
    rows, hidden = x.shape
    n = num_cores
    if rows % n:
        raise ValueError(f"{rows} rows do not split over {n} cores")
    r = rows // n
    x3 = x.reshape(n, r, hidden)

    def kernel(*refs):
        me, group, (x_ref, send, recv, land) = split_refs(refs, n, axis, True, True)
        barrier(group)
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                _push_chunked(x_ref.at[j], land.at[me], send, recv, j, group, r, chunks)

        # The own slice lands through the same semaphore as the peers'.
        pltpu.make_async_copy(x_ref.at[me], land.at[me], recv).start()

    args, specs = scalar_operands(rank, peers)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    send, recv, land = pl.pallas_call(
        kernel,
        out_shape=(
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA(()),
            jax.ShapeDtypeStruct((n, r, hidden), x.dtype),
        ),
        in_specs=[*specs, _pinned_hbm()],
        out_specs=(sem, sem, _pinned_hbm()),
        compiler_params=pltpu.CompilerParams(
            collective_id=ASYNC_RS_COLLECTIVE_ID, has_side_effects=True
        ),
        name=f"large_reduce_scatter_start_n{n}_r{rows}",
    )(*args, x3)
    return (send, recv, land, x3)


def reduce_scatter_done(handle, *, after=None, tile_rows=REDUCE_TILE_ROWS):
    """Wait for ``reduce_scatter_start``'s slices and sum them: the same bits
    as ``reduce_scatter``. ``after``, any array, orders this after its
    producer (e.g. the kernel the transfer overlaps)."""
    send, recv, land, x3 = handle
    n, r, hidden = land.shape
    tr = min(tile_rows, r)
    while r % tr:
        tr //= 2
    extra = () if after is None else (after,)

    def kernel(send_ref, recv_ref, land_ref, x_ref, *rest):
        o_ref, land_out = rest[-2:]
        del land_out  # aliases land_ref
        wait_bytes_of(land_ref, recv_ref)
        wait_bytes_of(land_ref.at[pl.ds(0, n - 1)], send_ref)

        def tile(in_ref, out_ref):
            acc = in_ref[0].astype(jnp.float32)
            for i in range(1, n):
                acc = acc + in_ref[i].astype(jnp.float32)
            out_ref[...] = acc.astype(out_ref.dtype)

        pltpu.emit_pipeline(
            tile,
            grid=(r // tr,),
            in_specs=[pl.BlockSpec((n, tr, hidden), lambda t: (0, t, 0))],
            out_specs=[pl.BlockSpec((tr, hidden), lambda t: (t, 0))],
        )(land_ref, o_ref)

    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    out, _ = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((r, hidden), land.dtype),
            jax.ShapeDtypeStruct(land.shape, land.dtype),
        ),
        in_specs=[sem, sem, _pinned_hbm(), _pinned_hbm(), *[_hbm()] * len(extra)],
        out_specs=(_hbm(), _pinned_hbm()),
        input_output_aliases={2: 1},
        compiler_params=pltpu.CompilerParams(has_side_effects=True),
        name=f"large_reduce_scatter_done_n{n}_r{n * r}",
    )(send, recv, land, x3, *extra)
    return out
