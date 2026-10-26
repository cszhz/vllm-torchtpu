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
"""The fused EP layer's ragged all-to-all dispatch, split into an issue and
a wait.

`exchange_start` sends every destination its run of the destination-major
send buffer with one DMA each -- remote for the others, local for this
shard's own run -- into the destination's receive buffer at this shard's
fixed block (shard r's rows at r * t_local, the layout XLA's ragged
all-to-all fills), and returns at once with the receive buffer and the two
DMA semaphores. `exchange_wait` takes them back, blocks until this shard's
own sends have left its buffer and every row due here has landed, and
returns the buffer.

Neither kernel is marked side-effecting: torch_tpu's bridge recompiles the
exported module through XLA's SPMD partitioner, which refuses a side-effecting
custom call without a sharding of its own (spmd_partitioner.cc "Side-effect HLO
must have sharding"), and nothing here needs the mark -- every output is
consumed, so neither call can be dropped, and the order is the data's.

What runs between the two is the point. XLA's `ragged_all_to_all` is one
synchronous TensorCore op -- 94 us a layer at DP32 C768, at the fabric's own
rate, which a Mosaic exchange measures the same -- and the routing plan the
fused kernel needs is built from the gathered payload alone. Issued first and
waited last, the exchange runs under the plan instead of in front of it. The
layer pins that order with optimization barriers; these kernels only carry
the semaphores from one call to the next.
"""

import threading

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2.kernel import _all_pairs_barrier, _rows_wait

# The barrier semaphore belongs to a collective id, and the fused kernel's
# barriers count on kernel.COLLECTIVE_ID's. The exchange's own entry barrier
# takes another, so a shard still in this barrier and a faster one already in
# the kernel's never add into one semaphore.
COLLECTIVE_ID = 1

_SMEM = pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM)
_HBM = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
_SEMAPHORE = pl.BlockSpec(memory_space=pltpu.MemorySpace.SEMAPHORE)


def _build_start(ep, t_local, send_rows, row_shape, dtype):
    """The issuing pallas_call for one geometry."""

    def kernel(sizes_sm, offsets_sm, rank_sm, send_hbm, recv_hbm, send_sem, recv_sem):
        # Peers write into this shard's receive buffer, which is this call's
        # output and so exists from this call on: nobody sends until every
        # shard has entered.
        _all_pairs_barrier(ep)
        me = rank_sm[0, 0]
        # Step k sends to shard me + k, so step 0 is this shard's own run and
        # no destination takes every source's first DMA at once. A zero-length
        # DMA must never issue, so every copy is predicated on its length.
        for k in range(ep):
            d = lax.rem(me + k, jnp.int32(ep))
            rows = sizes_sm[d]
            src = send_hbm.at[pl.ds(offsets_sm[d], rows)]
            dst = recv_hbm.at[pl.ds(me * t_local, rows)]
            if k == 0:
                # Its own run needs no fabric; the copy signals the send
                # semaphore, which the wait counts it against.
                @pl.when(rows > 0)
                def _(src=src, dst=dst):
                    pltpu.make_async_copy(src, dst, send_sem).start()

            else:

                @pl.when(rows > 0)
                def _(src=src, dst=dst, d=d):
                    pltpu.make_async_remote_copy(
                        src_ref=src,
                        dst_ref=dst,
                        send_sem=send_sem,
                        recv_sem=recv_sem,
                        device_id=(d,),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((ep * t_local, *row_shape), dtype),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA(()),
        ),
        in_specs=[_SMEM, _SMEM, _SMEM, _HBM],
        out_specs=(_HBM, _SEMAPHORE, _SEMAPHORE),
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=f"fused_ep_dispatch_start_ep{ep}_t{t_local}_s{send_rows}",
    )


def _build_wait(ep, t_local, send_rows, row_shape, dtype):
    """The waiting pallas_call for one geometry."""

    def kernel(expected_sm, send_hbm, recv_in, send_sem, recv_sem, recv_hbm):
        # recv_in and recv_hbm are one buffer (aliased); the waits count
        # bytes, so each is sized by the rows it owes and the geometry of the
        # buffer those rows live in.
        del recv_in
        sent = expected_sm[0]
        arriving = expected_sm[1]

        @pl.when(sent > 0)
        def _():
            _rows_wait(send_sem, send_hbm, sent)

        @pl.when(arriving > 0)
        def _():
            _rows_wait(recv_sem, recv_hbm, arriving)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((ep * t_local, *row_shape), dtype),
        in_specs=[_SMEM, _HBM, _HBM, _SEMAPHORE, _SEMAPHORE],
        out_specs=_HBM,
        input_output_aliases={2: 0},
        name=f"fused_ep_dispatch_wait_ep{ep}_t{t_local}_s{send_rows}",
    )


# Pallas keys its kernel-to-jaxpr cache on the kernel function itself, so a
# build per call would re-trace both kernels for every layer.
_BUILDS = {}
_BUILDS_LOCK = threading.Lock()


def _builds(ep, t_local, send):
    key = (ep, t_local, send.shape[0], tuple(send.shape[1:]), jnp.dtype(send.dtype))
    built = _BUILDS.get(key)
    if built is None:
        with _BUILDS_LOCK:
            built = _BUILDS.get(key)
            if built is None:
                built = (_build_start(*key), _build_wait(*key))
                _BUILDS[key] = built
    return built


def exchange_start(send, sizes, offsets, rank, *, t_local):
    """Issue the exchange; returns (receive buffer, send sem, recv sem).

    send [S, ...] is this shard's rows, destination-major: shard d's run is
    sizes[d] rows from offsets[d]. Every run lands in shard d's receive
    buffer [ep * t_local, ...] at rows [me * t_local, me * t_local +
    sizes[d]), so no run may exceed t_local rows. rank [1, 1] i32 is this
    shard's mesh index. Called inside the layer's shard_map. The buffer's
    other rows are never written: rows nobody sends are rows nobody reads.
    """
    ep = sizes.shape[0]
    start, _ = _builds(ep, t_local, send)
    return start(
        sizes.astype(jnp.int32),
        offsets.astype(jnp.int32),
        rank.astype(jnp.int32),
        send,
    )


def exchange_wait(recv, send, send_sem, recv_sem, expected, *, t_local):
    """Wait for exchange_start's DMAs; returns the receive buffer.

    expected [2] i32 is (rows this shard sent, its own run included; rows due
    here from the other shards). Both must be exact: a short count returns
    before the rows land, a long one never returns. `send` is taken so that
    its buffer outlives the DMAs reading it.
    """
    ep = recv.shape[0] // t_local
    _, wait = _builds(ep, t_local, send)
    return wait(expected.astype(jnp.int32), send, recv, send_sem, recv_sem)
