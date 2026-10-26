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

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.fused_moe.v2.kernel import _all_pairs_barrier, _rows_wait

# The barrier semaphore belongs to a collective id, and the fused kernel's
# barriers count on kernel.COLLECTIVE_ID's. The exchange's own entry barrier
# takes another, so a shard still in this barrier and a faster one already in
# the kernel's never add into one semaphore.
COLLECTIVE_ID = 1


def issue_config():
    """(barrier rounds, destination order, where the payload sends drain).

    Read where the kernels are built, so the build cache and the kernel's name
    carry it and two settings cannot share a compiled program.
    """
    return (
        max(1, int(envs.MOE_FUSED_EP_DISPATCH_BARRIER_ROUNDS)),
        envs.MOE_FUSED_EP_DISPATCH_ORDER,
        envs.MOE_FUSED_EP_DISPATCH_PAYLOAD_DRAIN,
    )


_SMEM = pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM)
_HBM = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
_SEMAPHORE = pl.BlockSpec(memory_space=pltpu.MemorySpace.SEMAPHORE)


def _build_start(ep, t_local, send_rows, row_shape, dtype, width, cfg, groups=0):
    """The issuing pallas_call for one geometry; width > 0 adds the payload.

    With `groups`, `sizes` and `offsets` are [ep, groups] and each
    destination takes one DMA per expert group rather than one in all, in
    group-major order: every destination's group 0 is on the wire before any
    group 1, which is the order the receivers wait in. Run (d, g) signals
    `recv_sems[g]` at the destination, so a receiver can wait for one group
    without counting a faster source's later groups. This shard's own run
    rides the arrival semaphore too (a local copy signals on completion, so
    the receiver's own wait is also what frees the send buffer's rows).
    """
    rounds, order, drain = cfg
    # "stagger" walks me, me + 1, ...; "farthest" starts half the mesh away,
    # which at DP32 is across the z links. Either way the own-destination step
    # is known at trace time, so the local copy needs no predicate.
    skew = 0 if order == "stagger" else ep // 2
    own_k = (ep - skew) % ep

    def kernel(sizes_sm, offsets_sm, rank_sm, send_hbm, *refs):
        # Inputs, then outputs, then scratch: the payload's send semaphore is
        # last either way, as a scratch where the issue drains it itself and
        # as an output where the payload wait does.
        it = iter(refs)
        packed_hbm = next(it) if width else None
        recv_hbm = next(it)
        pbuf_hbm = next(it) if width else None
        if groups:
            # The send buffer, aliased in and straight back out, written by
            # nobody: see the lifetime note on the output list below.
            next(it)
        send_sem, recv_sem = next(it), next(it)
        precv_sem = next(it) if width else None
        psend_sem = next(it) if width else None
        # Peers write into this shard's receive buffers, which are this call's
        # outputs and so exist from this call on: nobody sends until every
        # shard has entered.
        _all_pairs_barrier(ep, rounds)
        me = rank_sm[0, 0]
        # Step k sends to shard me + skew + k, so no destination takes every
        # source's first DMA at once. The payload goes first: every peer's
        # routing plan waits for it, and a DMA queue retires in order.
        if width:
            for k in range(ep):
                d = lax.rem(me + skew + k, jnp.int32(ep))
                dst = pbuf_hbm.at[pl.ds(me, 1)]
                if k == own_k:
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
            if drain == "issue":
                # The payload has left before the first row is issued: its
                # source is free when this call returns, and no more than one
                # batch of remote DMAs is ever in flight from here.
                _rows_wait(psend_sem, pbuf_hbm, ep - 1)
        # A zero-length DMA must never issue, so every row copy is predicated
        # on its length.
        if groups:
            _issue_grouped(
                ep, t_local, groups, skew, own_k, sizes_sm, offsets_sm, me,
                send_hbm, recv_hbm, send_sem, recv_sem,
            )
            return
        for k in range(ep):
            d = lax.rem(me + skew + k, jnp.int32(ep))
            rows = sizes_sm[d]
            src = send_hbm.at[pl.ds(offsets_sm[d], rows)]
            dst = recv_hbm.at[pl.ds(me * t_local, rows)]
            if k == own_k:
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
    aliases = {}
    if groups:
        # The send buffer comes straight back out, aliased. Taking it as an
        # operand of whatever waits for the sends is not enough to keep it
        # where the DMAs are reading it: with the grouped dispatch nothing
        # waits for them until after the expert kernel, and over a live range
        # that long XLA will rematerialize or copy the 4.4 MB `jnp.take` that
        # produced it and recycle the address underneath the in-flight
        # transfers. Measured at the served expert dims: 1128 of 1152 output
        # rows wrong, identically on every repeat, and clean as soon as the
        # sends are waited for before the kernel. A donated custom-call
        # result cannot be rematerialized, so this states the requirement
        # where XLA can see it.
        aliases[3] = len(out_shape)
        out_shape.append(jax.ShapeDtypeStruct((send_rows, *row_shape), dtype))
        out_specs.append(_HBM)
    # send, receive, the payload's receive where there is a payload, and the
    # payload's send where the payload wait drains it. The arrival semaphore
    # is per group, so that a wait for one group cannot be satisfied by
    # another group's bytes.
    sem_shapes = [(), (groups,) if groups else ()]
    if width:
        sem_shapes.append(())
        if drain == "wait":
            sem_shapes.append(())
    scratch = [pltpu.SemaphoreType.DMA] if width and drain == "issue" else []
    return pl.pallas_call(
        kernel,
        out_shape=(*out_shape, *(pltpu.SemaphoreType.DMA(s) for s in sem_shapes)),
        in_specs=[_SMEM, _SMEM, _SMEM, _HBM] + ([_HBM] if width else []),
        out_specs=(*out_specs, *(_SEMAPHORE for _ in sem_shapes)),
        scratch_shapes=scratch,
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=(
            f"fused_ep_dispatch_start_ep{ep}_t{t_local}_s{send_rows}_p{width}"
            f"_g{groups}_b{rounds}_{order}_{drain}"
        ),
    )


def _issue_grouped(
    ep, t_local, groups, skew, own_k, sizes_sm, offsets_sm, me, send_hbm,
    recv_hbm, send_sem, recv_sems,
):
    """One DMA per (destination, group), group-major, destinations staggered.

    At most `ep * groups` descriptors leave one call, and they are issued
    group by group so that no receiver's group g waits behind another
    receiver's group g + 1.
    """
    for g in range(groups):
        for k in range(ep):
            d = lax.rem(me + skew + k, jnp.int32(ep))
            rows = sizes_sm[d, g]
            src = send_hbm.at[pl.ds(offsets_sm[d, g], rows)]
            # The destination keeps this shard's fixed block; inside it the
            # group runs sit at the prefix sums of what this shard sends it,
            # which is this run's offset less the destination's first.
            dst = recv_hbm.at[
                pl.ds(me * t_local + offsets_sm[d, g] - offsets_sm[d, 0], rows)
            ]
            if k == own_k:

                @pl.when(rows > 0)
                def _(src=src, dst=dst, g=g):
                    pltpu.make_async_copy(src, dst, recv_sems.at[g]).start()

            else:

                @pl.when(rows > 0)
                def _(src=src, dst=dst, d=d, g=g):
                    pltpu.make_async_remote_copy(
                        src_ref=src,
                        dst_ref=dst,
                        send_sem=send_sem,
                        recv_sem=recv_sems.at[g],
                        device_id=(d,),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()


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


def _build_payload_wait(ep, width, drain):
    """Wait for the other shards' payload rows; returns the payload table.

    With `drain == "wait"` this also drains the payload's own sends, which the
    issue then does not: the rows go on the wire that much earlier, and the
    payload's source buffer is held alive by being an operand here. The table
    has the same byte geometry as the source block, so one ref sizes both
    waits.
    """

    def kernel(pbuf_in, *refs):
        if drain == "wait":
            src_in, precv_sem, psend_sem, pbuf_hbm = refs
            del src_in  # held alive for the sends reading it
        else:
            precv_sem, pbuf_hbm = refs
        del pbuf_in  # aliased
        # Every shard's row, this one's own copy included.
        _rows_wait(precv_sem, pbuf_hbm, ep)
        if drain == "wait":
            _rows_wait(psend_sem, pbuf_hbm, ep - 1)

    in_specs = [_HBM] + ([_HBM] if drain == "wait" else [])
    in_specs += [_SEMAPHORE] * (2 if drain == "wait" else 1)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(_table_shape(ep, width), jnp.int32),
        in_specs=in_specs,
        out_specs=_HBM,
        input_output_aliases={0: 0},
        name=f"fused_ep_dispatch_payload_wait_ep{ep}_p{width}_{drain}",
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


def _builds(ep, t_local, send, width, groups=0):
    key = (
        ep,
        t_local,
        send.shape[0],
        tuple(send.shape[1:]),
        jnp.dtype(send.dtype),
        width,
        issue_config(),
        groups,
    )
    built = _BUILDS.get(key)
    if built is None:
        with _BUILDS_LOCK:
            built = _BUILDS.get(key)
            if built is None:
                built = (_build_start(*key), _build_wait(*key[:-3], 0))
                _BUILDS[key] = built
    return built


def exchange_start(send, sizes, offsets, rank, *, t_local, payload=None, groups=0):
    """Issue the exchange; returns (receive buffer, send sem, recv sem), or
    with `payload` (receive buffer, payload table, send sem, recv sem,
    payload recv sem [, payload send sem]). The last is returned only where
    `MOE_FUSED_EP_DISPATCH_PAYLOAD_DRAIN=wait` leaves the payload's sends for
    `payload_wait` to drain.

    With `groups`, `sizes` and `offsets` are [ep, groups], the send buffer
    comes back out aliased after the receive buffer (hand THAT to
    `send_drain`, not the operand) and the returned arrival semaphore is a
    [groups] array: run (d, g) signals element g at the destination, which is
    what lets a receiver wait for one expert group's rows without a faster
    source's later rows satisfying the wait.

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
    start, _ = _builds(ep, t_local, send, width, groups)
    block = None if payload is None else _payload_block(payload)
    extra = () if block is None else (block,)
    out = start(
        sizes.astype(jnp.int32),
        offsets.astype(jnp.int32),
        rank.astype(jnp.int32),
        send,
        *extra,
    )
    if width and issue_config()[2] == "wait":
        # The source block goes back to the caller so `payload_wait`, which
        # drains the sends reading it, can hold the very same buffer alive.
        return (*out, block)
    return out


def _payload_block(payload):
    """This shard's [width] payload as its [1, tiles, 8, 128] table block."""
    width = payload.shape[-1]
    padded = jnp.pad(payload.astype(jnp.int32), (0, _padded_width(width) - width))
    return padded.reshape(1, -1, 8, 128)


_PAYLOAD_WAITS = {}


def payload_wait(pbuf, precv_sem, *, width, psend_sem=None, src=None):
    """Wait for every shard's payload row; returns the [ep, width] table.

    `psend_sem` and `src` are `exchange_start`'s last two results where the
    issue left the payload's sends undrained; both must be given together.
    """
    if (psend_sem is None) != (src is None):
        raise ValueError("payload_wait takes psend_sem and src together")
    drain = "wait" if psend_sem is not None else "issue"
    ep = pbuf.shape[0]
    wait = _PAYLOAD_WAITS.get((ep, width, drain))
    if wait is None:
        with _BUILDS_LOCK:
            wait = _PAYLOAD_WAITS.setdefault(
                (ep, width, drain), _build_payload_wait(ep, width, drain)
            )
    args = (pbuf, precv_sem) if drain == "issue" else (pbuf, src, precv_sem, psend_sem)
    return wait(*args).reshape(ep, -1)[:, :width]


_SEND_DRAINS = {}


def send_drain(keep, send, send_sem, expected):
    """Wait out the exchange's own sends, behind whatever `keep` depends on.

    The in-kernel dispatch has no `exchange_wait`: the receiver's waits moved
    into the expert kernel, and its own run rides the arrival semaphore. What
    is left is the SENDER's obligation -- the send buffer must outlive the
    DMAs reading it -- and this is where it is discharged. `keep` is the
    expert kernel's arrival buffer, taken and returned unchanged (aliased):
    it is what orders this call after the kernel and keeps the combine, which
    reads the result, from being scheduled in front of it.

    `expected` [1] i32 is the rows this shard sends to OTHER shards. It must
    be exact: short returns under in-flight DMAs, long never returns.
    """
    key = (keep.shape, jnp.dtype(keep.dtype), send.shape, jnp.dtype(send.dtype))
    built = _SEND_DRAINS.get(key)
    if built is None:
        with _BUILDS_LOCK:
            built = _SEND_DRAINS.setdefault(key, _build_send_drain(*key))
    return built(expected.astype(jnp.int32), keep, send, send_sem)


def _build_send_drain(keep_shape, keep_dtype, send_shape, send_dtype):
    del send_shape, send_dtype

    def kernel(expected_sm, keep_in, send_hbm, send_sem, keep_hbm):
        del keep_in, keep_hbm  # aliased; this call only waits
        sent = expected_sm[0]

        @pl.when(sent > 0)
        def _():
            _rows_wait(send_sem, send_hbm, sent)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(keep_shape, keep_dtype),
        in_specs=[_SMEM, _HBM, _HBM, _SEMAPHORE],
        out_specs=_HBM,
        input_output_aliases={1: 0},
        name="fused_ep_dispatch_send_drain",
    )


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
