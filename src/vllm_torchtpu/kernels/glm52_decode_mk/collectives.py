# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""In-kernel collectives of the megakernel (all barrier-free after the one entry barrier).

* `all_reduce`: main's moe_tp_decode all-reduce protocol (all_reduce.py `_exchange` @ 9ffdb1eb4: reduce-scatter of
  R column chunks, f32 sum of the R bf16 slots in rank order rounded to bf16, all-gather; per-sender receive
  semaphores; rank r's i-th copy to (r + i) % R). Copied rather than imported so the fault variants (a dropped
  partial) and the chunk-width fallback live next to it; the protocol and its numerics are unchanged.
* `exchange`: glm5_mk_dcp's one-round all-peer exchange (kernel.py `exchange` @ d41e70f6d, :896-920): q head
  all-gather and the partial + lse all-to-all.
"""

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16, F32, I32 = jnp.bfloat16, jnp.float32, jnp.int32
MESH = pl.DeviceIdType.MESH


def ar_scratch(R, M, H):
    """Scratch of one all-reduce slot set: recv [R, M, H/R] bf16, red [M, H/R] bf16, 4 x DMA((R,)), DMA((2,))."""
    cw = H // R
    return (
        [pltpu.VMEM((R, M, cw), BF16), pltpu.VMEM((M, cw), BF16)]
        + [pltpu.SemaphoreType.DMA((R,)) for _ in range(4)]
        + [pltpu.SemaphoreType.DMA((2,))]
    )


AR_NAMES = ("recv", "red", "rs_send", "rs_recv", "ag_send", "ag_recv", "loc")


def _cols(ref, t, size):
    return ref.at[:, pl.ds(pl.multiple_of(t * size, size), size)]


def _send_all(R, me, src_of, dst_of, send_sem, recv_sem):
    sends = []
    for step in range(1, R):
        t = lax.rem(me + step, R)
        c = pltpu.make_async_remote_copy(
            src_of(t),
            dst_of(me),
            send_sem.at[step],
            recv_sem.at[me],
            device_id=(t,),
            device_id_type=MESH,
        )
        c.start()
        sends.append(c)
    return sends


def _recv_all(R, me, dst_of, recv_sem):
    for step in range(1, R):
        t = lax.rem(me + (R - step), R)
        pltpu.make_async_copy(dst_of(t), dst_of(t), recv_sem.at[t]).wait_recv()


def all_reduce(R, me, buf, rows, cols, s, drop_last=False, early_read=False, mids=()):
    """In-place all-reduce of buf ([rows, cols] bf16 VMEM). s: dict(zip(AR_NAMES, scratch)).
    drop_last (FAULT control `missing_rank_partial`): rank R-1's partial is left out of the f32 sum.
    early_read (FAULT control `f_ar_earlyread`, a delay-exposed race): the reduce-scatter slots are summed BEFORE
    their receives are waited, so a late peer's slot still holds the previous layer's partial (the barrier-free
    slot reuse this protocol must never expose); the waits follow, keeping the semaphores balanced.
    mids (stage-B L2 `gearly`): up to two callables run while the reduce-scatter (mids[0]) and the all-gather
    (mids[1]) are in flight, i.e. in the TC's idle wait windows; they must not touch buf or the slot sets."""
    if R == 1:
        if drop_last:
            buf[...] = jnp.zeros(buf.shape, buf.dtype)
        return
    cw = cols // R

    def slot(q):
        return s["recv"].at[q, pl.ds(0, rows), pl.ds(0, cw)]

    def chunk(t):
        return _cols(buf, t, cw)

    red = s["red"].at[pl.ds(0, rows), pl.ds(0, cw)]
    own = pltpu.make_async_copy(chunk(me), slot(me), s["loc"].at[0])
    own.start()
    rs = _send_all(R, me, chunk, slot, s["rs_send"], s["rs_recv"])
    if len(mids) > 0 and mids[0] is not None:
        mids[0]()
    own.wait()
    if not early_read:
        _recv_all(R, me, slot, s["rs_recv"])
    acc = s["recv"][0, pl.ds(0, rows), pl.ds(0, cw)].astype(F32)
    for q in range(1, R):
        v = s["recv"][q, pl.ds(0, rows), pl.ds(0, cw)].astype(F32)
        if drop_last and q == R - 1:
            v = jnp.zeros_like(v)
        acc = acc + v
    red[...] = acc.astype(red.dtype)
    if early_read:  # the sum is materialised before the receives are waited
        _recv_all(R, me, slot, s["rs_recv"])
    own_ag = pltpu.make_async_copy(red, chunk(me), s["loc"].at[1])
    own_ag.start()
    ag = _send_all(R, me, lambda t: red, chunk, s["ag_send"], s["ag_recv"])
    if len(mids) > 1 and mids[1] is not None:
        mids[1]()
    own_ag.wait()
    _recv_all(R, me, chunk, s["ag_recv"])
    for c in (*rs, *ag):
        c.wait_send()


def row_switch(n_dyn, options, fn):
    """LV (live-row collectives): run fn(n) for the one static row count n in `options` equal to the runtime n_dyn
    (DMA shapes are static, so every option is its own issue site; exactly one fires, rank-uniformly)."""
    for n in options:

        @pl.when(n_dyn == n)
        def _(n=n):
            fn(n)


def all_reduce_lv(R, me, buf, rows, cols, s, nlb, rb, drop_last=False, mids=()):
    """LV form of `all_reduce`: the same protocol (reduce-scatter of R column chunks, f32 rank-order sum, all-gather;
    the same semaphores and peer order), moving only the first nlb * rb rows (nlb dynamic in 1..rows // rb, the same on
    every rank). Rows >= nlb * rb of buf are neither sent nor overwritten (the caller zeroes rows >= T, so buf ends
    bit-identical to `all_reduce`'s result); slot rows past them hold stale bytes and are masked out of `red`."""
    if R == 1:
        if drop_last:
            buf[...] = jnp.zeros(buf.shape, buf.dtype)
        return
    cw = cols // R
    opts = tuple(k * rb for k in range(1, rows // rb + 1))
    n_dyn = nlb * rb

    def slot(q, n):
        return s["recv"].at[q, pl.ds(0, n), pl.ds(0, cw)]

    def chunk(t, n):
        return buf.at[pl.ds(0, n), pl.ds(pl.multiple_of(t * cw, cw), cw)]

    def red(n):
        return s["red"].at[pl.ds(0, n), pl.ds(0, cw)]

    def rs_start(n):
        pltpu.make_async_copy(chunk(me, n), slot(me, n), s["loc"].at[0]).start()
        _send_all(R, me, lambda t: chunk(t, n), lambda q: slot(q, n), s["rs_send"], s["rs_recv"])

    def rs_wait(n):
        pltpu.make_async_copy(chunk(me, n), slot(me, n), s["loc"].at[0]).wait()
        _recv_all(R, me, lambda q: slot(q, n), s["rs_recv"])

    def ag_start(n):
        pltpu.make_async_copy(red(n), chunk(me, n), s["loc"].at[1]).start()
        _send_all(R, me, lambda t: red(n), lambda q: chunk(q, n), s["ag_send"], s["ag_recv"])

    def ag_wait(n):
        pltpu.make_async_copy(red(n), chunk(me, n), s["loc"].at[1]).wait()
        _recv_all(R, me, lambda q: chunk(q, n), s["ag_recv"])
        for step in range(1, R):
            t = lax.rem(me + step, R)
            for src, dst, ss, rs_ in (
                (chunk(t, n), slot(me, n), s["rs_send"], s["rs_recv"]),
                (red(n), chunk(me, n), s["ag_send"], s["ag_recv"]),
            ):
                pltpu.make_async_remote_copy(
                    src, dst, ss.at[step], rs_.at[me], device_id=(t,), device_id_type=MESH
                ).wait_send()

    row_switch(nlb, tuple(range(1, len(opts) + 1)), lambda k: rs_start(k * rb))
    if len(mids) > 0 and mids[0] is not None:
        mids[0]()
    row_switch(nlb, tuple(range(1, len(opts) + 1)), lambda k: rs_wait(k * rb))
    acc = s["recv"][0, pl.ds(0, rows), pl.ds(0, cw)].astype(F32)
    for q in range(1, R):
        v = s["recv"][q, pl.ds(0, rows), pl.ds(0, cw)].astype(F32)
        if drop_last and q == R - 1:
            v = jnp.zeros_like(v)
        acc = acc + v
    live = lax.broadcasted_iota(jnp.int32, (rows, cw), 0) < n_dyn
    s["red"][pl.ds(0, rows), pl.ds(0, cw)] = jnp.where(live, acc, 0.0).astype(BF16)
    row_switch(nlb, tuple(range(1, len(opts) + 1)), lambda k: ag_start(k * rb))
    if len(mids) > 1 and mids[1] is not None:
        mids[1]()
    row_switch(nlb, tuple(range(1, len(opts) + 1)), lambda k: ag_wait(k * rb))


def entry_barrier(R):
    """The one barrier of the call: every rank signals every rank, then waits for R signals."""
    if R == 1:
        return
    b = pltpu.get_barrier_semaphore()
    for d in range(R):
        pl.semaphore_signal(b, inc=1, device_id=(jnp.int32(d),), device_id_type=MESH)
    pl.semaphore_wait(b, R)


def exchange(R, me, src_fn, dst_fn, send_sem, recv_sem, wait=True, scope=None):
    """Start src_fn(d) -> peer d's dst_fn(me) for every peer d != me; finish() waits the R-1 receives and sends
    (byte-counting semaphores; every exchange carries the same bytes per peer)."""
    if R == 1:
        return None if wait else (lambda: None)
    for d in range(R):

        @pl.when(me != d)
        def _(d=d):
            pltpu.make_async_remote_copy(
                src_ref=src_fn(d),
                dst_ref=dst_fn(me),
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(jnp.int32(d),),
                device_id_type=MESH,
            ).start()

    def finish():
        desc = pltpu.make_async_remote_copy(
            src_ref=src_fn(0),
            dst_ref=dst_fn(0),
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(me,),
            device_id_type=MESH,
        )

        def waits():
            for _ in range(R - 1):
                desc.wait_recv()
            for _ in range(R - 1):
                desc.wait_send()

        if scope is not None:
            with scope("X_wait"):
                waits()
        else:
            waits()

    if not wait:
        return finish
    finish()


def start_all(R, me, src_fn, dst_fn, send_sem, recv_sem):
    """Start src_fn(d) -> peer d's dst_fn(me) for every peer d != me (no wait). L4 C1 / C2: the chunked A2A and the
    lse tiles start pieces of one exchange at different times on one (send, recv) semaphore pair."""
    if R == 1:
        return
    for d in range(R):

        @pl.when(me != d)
        def _(d=d):
            pltpu.make_async_remote_copy(
                src_ref=src_fn(d),
                dst_ref=dst_fn(me),
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(jnp.int32(d),),
                device_id_type=MESH,
            ).start()


def wait_pieces(R, me, pieces, send_sem, recv_sem, scope=None):
    """Wait every piece started by start_all on (send_sem, recv_sem): pieces = [(src_ref, dst_ref, n_starts)], each
    started n_starts times towards every peer. Byte-counting semaphores: all receives are waited before any send, and
    nothing is read before every wait returned, so the arrival order of pieces does not matter."""
    if R == 1:
        return

    def waits():
        descs = [
            (
                pltpu.make_async_remote_copy(
                    src_ref=s_,
                    dst_ref=d_,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(me,),
                    device_id_type=MESH,
                ),
                n,
            )
            for s_, d_, n in pieces
        ]
        for kind in ("recv", "send"):
            for desc, n in descs:
                tot = n * (R - 1)
                fn = desc.wait_recv if kind == "recv" else desc.wait_send
                if isinstance(tot, int) and tot <= 16:
                    for _ in range(tot):
                        fn()
                else:

                    def body(i, c, fn=fn):
                        fn()
                        return c

                    lax.fori_loop(0, tot, body, 0)

    if scope is not None:
        with scope("X_wait"):
            waits()
    else:
        waits()


def spin(n, ref):
    """F-DELAY: a data-dependent busy loop of n iterations on a (8, 128) f32 VMEM ref (csq_proto spin, :538)."""
    ref[...] = lax.fori_loop(
        0, n, lambda it, a: a * 0.9990234375 + 0.0009765625, ref[...]
    )
