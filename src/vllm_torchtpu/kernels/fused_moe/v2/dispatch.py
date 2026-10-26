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

With `payload`, the same issue also hands every shard this shard's routing
payload -- the flat [width] row the layer otherwise all-gathers -- one remote
DMA per peer into row me of each peer's [ep, width] buffer, ahead of the
rows, and drains those sends before issuing a row; `payload_wait` takes
the table back once every row of it (its own copy included) has landed. At
DP32 the SparseCore all-gather of that 213 KB table was a 32-step ring that
left 91 us a layer exposed in front of the routing plan, longer than the
rows' own exchange; pushed directly it is one hop.

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


def _build_start(ep, t_local, send_rows, row_shape, dtype, width):
    """The issuing pallas_call for one geometry; width > 0 adds the payload."""

    def kernel(sizes_sm, offsets_sm, rank_sm, send_hbm, *refs):
        if width:
            packed_hbm, recv_hbm, pbuf_hbm = refs[:3]
            # psend_sem is scratch: the payload's sends are drained here.
            send_sem, recv_sem, precv_sem, psend_sem = refs[3:]
        else:
            recv_hbm, send_sem, recv_sem = refs
        # Peers write into this shard's receive buffers, which are this call's
        # outputs and so exist from this call on: nobody sends until every
        # shard has entered.
        _all_pairs_barrier(ep)
        me = rank_sm[0, 0]
        # Step k sends to shard me + k, so step 0 is this shard's own copy and
        # no destination takes every source's first DMA at once. The payload
        # goes first: every peer's routing plan waits for it, and a DMA queue
        # retires in order.
        if width:
            for k in range(ep):
                d = lax.rem(me + k, jnp.int32(ep))
                dst = pbuf_hbm.at[pl.ds(me, 1)]
                if k == 0:
                    # Its own row counts as an arrival: the payload wait must
                    # cover it, or the plan can read row me before it lands.
                    pltpu.make_async_copy(packed_hbm, dst, precv_sem).start()
                else:
                    pltpu.make_async_remote_copy(
                        src_ref=packed_hbm,
                        dst_ref=dst,
                        send_sem=psend_sem,
                        recv_sem=precv_sem,
                        device_id=(d,),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()
            # The payload has left before the first row is issued: its source
            # is free when this call returns, and no more than one batch of
            # remote DMAs is ever in flight from here.
            _rows_wait(psend_sem, pbuf_hbm, ep - 1)
        # A zero-length DMA must never issue, so every row copy is predicated
        # on its length.
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

    out_shape = [jax.ShapeDtypeStruct((ep * t_local, *row_shape), dtype)]
    out_specs = [_HBM]
    if width:
        out_shape.append(jax.ShapeDtypeStruct(_table_shape(ep, width), jnp.int32))
        out_specs.append(_HBM)
    sems = 3 if width else 2
    return pl.pallas_call(
        kernel,
        out_shape=(*out_shape, *(pltpu.SemaphoreType.DMA(()) for _ in range(sems))),
        in_specs=[_SMEM, _SMEM, _SMEM, _HBM] + ([_HBM] if width else []),
        out_specs=(*out_specs, *(_SEMAPHORE for _ in range(sems))),
        scratch_shapes=[pltpu.SemaphoreType.DMA] if width else [],
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=f"fused_ep_dispatch_start_ep{ep}_t{t_local}_s{send_rows}_p{width}",
    )


def _padded_width(width):
    """The payload width rounded up to whole (8, 128) i32 tiles."""
    return -(-width // 1024) * 1024


def _table_shape(ep, width):
    """The payload table as [ep, tiles, 8, 128]: one source per leading index
    and every source a whole number of (8, 128) tiles, so every DMA and wait
    slices an untiled dimension of tile-exact blocks. A 2-D [ep, width] table
    is (8, 128)-tiled on the source axis, and Mosaic refuses a slice of
    ep - 1 sources along it (seen at the 2048-row bucket, ep 32)."""
    return (ep, _padded_width(width) // 1024, 8, 128)


def _build_payload_wait(ep, width):
    """Wait for the other shards' payload rows; returns the payload table."""

    def kernel(pbuf_in, precv_sem, pbuf_hbm):
        del pbuf_in  # aliased
        # Every shard's row, this one's own copy included.
        _rows_wait(precv_sem, pbuf_hbm, ep)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(_table_shape(ep, width), jnp.int32),
        in_specs=[_HBM, _SEMAPHORE],
        out_specs=_HBM,
        input_output_aliases={0: 0},
        name=f"fused_ep_dispatch_payload_wait_ep{ep}_p{width}",
    )


def _build_wait(ep, t_local, send_rows, row_shape, dtype, width):
    """The waiting pallas_call for one geometry (the payload, if any, was
    drained by the issue and waited for by payload_wait)."""
    del width

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


def _builds(ep, t_local, send, width):
    key = (
        ep,
        t_local,
        send.shape[0],
        tuple(send.shape[1:]),
        jnp.dtype(send.dtype),
        width,
    )
    built = _BUILDS.get(key)
    if built is None:
        with _BUILDS_LOCK:
            built = _BUILDS.get(key)
            if built is None:
                built = (_build_start(*key), _build_wait(*key[:-1], 0))
                _BUILDS[key] = built
    return built


def exchange_start(send, sizes, offsets, rank, *, t_local, payload=None):
    """Issue the exchange; returns (receive buffer, send sem, recv sem), or
    with `payload` (receive buffer, payload table, send sem, recv sem,
    payload recv sem).

    send [S, ...] is this shard's rows, destination-major: shard d's run is
    sizes[d] rows from offsets[d]. Every run lands in shard d's receive
    buffer [ep * t_local, ...] at rows [me * t_local, me * t_local +
    sizes[d]), so no run may exceed t_local rows. rank [1, 1] i32 is this
    shard's mesh index. Called inside the layer's shard_map. The buffer's
    other rows are never written: rows nobody sends are rows nobody reads.
    payload [width] i32, if given, lands at row me of every shard's [ep,
    width] payload table (payload_wait).
    """
    ep = sizes.shape[0]
    width = 0 if payload is None else payload.shape[-1]
    start, _ = _builds(ep, t_local, send, width)
    extra = () if payload is None else (_payload_block(payload),)
    return start(
        sizes.astype(jnp.int32),
        offsets.astype(jnp.int32),
        rank.astype(jnp.int32),
        send,
        *extra,
    )


def _payload_block(payload):
    """This shard's [width] payload as its [1, tiles, 8, 128] table block."""
    width = payload.shape[-1]
    padded = jnp.pad(payload.astype(jnp.int32), (0, _padded_width(width) - width))
    return padded.reshape(1, -1, 8, 128)


_PAYLOAD_WAITS = {}


def payload_wait(pbuf, precv_sem, *, width):
    """Wait for every shard's payload row; returns the [ep, width] table."""
    ep = pbuf.shape[0]
    wait = _PAYLOAD_WAITS.get((ep, width))
    if wait is None:
        with _BUILDS_LOCK:
            wait = _PAYLOAD_WAITS.setdefault(
                (ep, width), _build_payload_wait(ep, width)
            )
    return wait(pbuf, precv_sem).reshape(ep, -1)[:, :width]


def exchange_wait(recv, send, send_sem, recv_sem, expected, *, t_local):
    """Wait for exchange_start's row DMAs; returns the receive buffer.

    expected [2] i32 is (rows this shard sent, its own run included; rows due
    here from the other shards). Both must be exact: a short count returns
    before the rows land, a long one never returns. `send` is taken so that
    its buffer outlives the DMAs reading it.
    """
    ep = recv.shape[0] // t_local
    _, wait = _builds(ep, t_local, send, 0)
    return wait(expected.astype(jnp.int32), send, recv, send_sem, recv_sem)
