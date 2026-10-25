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
"""In-kernel remote-DMA all-reduce for the DSv4 decode megakernel.

Every rank contributes an f32 partial [rows, hidden]; payloads travel as bf16
and are summed in f32 in a rank order that does not depend on the receiving
rank, so every rank ends with bit-identical results (the replicated
activations the next layer reads must agree). Modes:

  flat       one round: the whole partial to every peer.
  rsag       reduce-scatter of lane chunks, then all-gather of chunk sums.
  pair       on-chip pair (me ^ 1) first, then pair sums to the other pairs.
  butterfly  recursive doubling, log2(R) rounds of the whole partial.
  hrsag      rsag inside groups of `group` ranks (a host), chunk sums
             exchanged across groups, then all-gather inside the group.

Peers are addressed as me ^ off on a mesh in ascending device-id order, so
R must be a power of two. Adapted from the Qwen3.5 megakernel all-reduce.
"""

import dataclasses
import math

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16 = jnp.bfloat16
F32 = jnp.float32
AR_MODES = ("flat", "rsag", "pair", "butterfly", "hrsag")


@dataclasses.dataclass(frozen=True)
class AllReduceSpec:
    """Static shape of one all-reduce site: R ranks, a [rows, hidden] payload,
    `slots` independent buffers (alternate them between consecutive calls)."""

    R: int
    rows: int
    hidden: int
    mode: str = "flat"
    slots: int = 2
    group: int = 8  # hrsag: ranks per group (contiguous mesh indices)

    def __post_init__(self):
        if self.mode not in AR_MODES:
            raise ValueError(f"mode must be one of {AR_MODES}; got {self.mode!r}")
        if self.R < 2 or self.R & (self.R - 1):
            raise ValueError(f"R must be a power of two >= 2; got {self.R}")
        if self.rows % 16:
            raise ValueError(
                f"rows must be a multiple of 16 (bf16 tile); got {self.rows}"
            )
        if self.mode == "rsag" and (self.hidden // self.R) % 128:
            raise ValueError(
                f"rsag needs hidden/R a multiple of 128; got {self.hidden}/{self.R}"
            )

    @property
    def n_rounds(self):
        if self.mode == "flat":
            return 1
        if self.mode == "butterfly":
            return self.R.bit_length() - 1
        return 3 if self.mode == "hrsag" else 2

    @property
    def chunk_ranks(self):
        return {"rsag": self.R, "hrsag": self.group}.get(self.mode, 1)

    @property
    def chunk(self):
        return self.hidden // self.chunk_ranks

    def scratch(self):
        """name -> scratch shape, all names prefixed `ar_`."""
        R, S, M, H = self.R, self.slots, self.rows, self.hidden
        nr = self.n_rounds
        vm = lambda *shape: pltpu.VMEM(shape, BF16)  # noqa: E731
        sem = pltpu.SemaphoreType.DMA((nr, S, R))
        if self.mode == "flat":
            bufs = {"ar_send": vm(S, M, H), "ar_recv": vm(S, R, M, H)}
        elif self.mode == "rsag":
            C = self.chunk
            bufs = {
                "ar_send": vm(S, R, M, C),
                "ar_recv": vm(S, R, M, C),
                "ar_agsend": vm(S, M, C),
                "ar_agrecv": vm(S, R, M, C),
            }
        elif self.mode == "hrsag":
            G, C = self.group, self.chunk
            bufs = {
                "ar_send": vm(S, G, M, C),
                "ar_recv": vm(S, G, M, C),
                "ar_xsend": vm(S, M, C),
                "ar_xrecv": vm(S, R // G, M, C),
                "ar_agsend": vm(S, M, C),
                "ar_agrecv": vm(S, G, M, C),
            }
        elif self.mode == "pair":
            bufs = {
                "ar_send": vm(S, M, H),
                "ar_recv": vm(S, 2, M, H),
                "ar_agsend": vm(S, M, H),
                "ar_agrecv": vm(S, R // 2, M, H),
            }
        else:
            bufs = {"ar_send": vm(nr, S, M, H), "ar_recv": vm(nr, S, 2, M, H)}
        return {**bufs, "ar_ssem": sem, "ar_rsem": sem}

    def vmem_bytes(self):
        n = 0
        for name, v in self.scratch().items():
            if not name.endswith("sem"):
                n += math.prod(v.shape) * jnp.dtype(v.dtype).itemsize
        return n


def barrier(R, me):
    """All R ranks have entered the kernel (their VMEM scratch exists)."""
    sem = pltpu.get_barrier_semaphore()
    for off in range(1, R):
        pl.semaphore_signal(
            sem, 1, device_id=(me ^ off,), device_id_type=pl.DeviceIdType.MESH
        )
    pl.semaphore_wait(sem, R - 1)


LIVE_BLOCK = 16  # rows per transfer chunk of a live-row all-reduce (bf16 tile)


class AllReduce:
    """The all-reduce of one AllReduceSpec over kernel scratch `scr` (a dict
    holding at least the spec's `ar_*` names); `me` is this rank's mesh index.

    live_blocks (hrsag reduce_into only; traced int32, the same on every
    rank): only rows [0, live_blocks * 16) are exchanged and summed, as
    16-row DMAs; the destination's rows past them are zeroed."""

    def __init__(self, spec: AllReduceSpec, scr, me, live_blocks=None):
        self.spec, self.scr, self.me = spec, scr, me
        self.live = live_blocks
        if live_blocks is not None:
            assert spec.mode == "hrsag", "live rows: hrsag only"

    def _copy(self, src, dst, rnd, slot, peer_off):
        s = self.scr
        return pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=dst,
            send_sem=s["ar_ssem"].at[rnd, slot, peer_off],
            recv_sem=s["ar_rsem"].at[rnd, slot, peer_off],
            device_id=(self.me ^ peer_off,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _offs(self, rnd):
        """Peer offsets of round `rnd` as (first, step, count)."""
        sp = self.spec
        R = sp.R
        if sp.mode == "butterfly":
            return 1 << rnd, 1, 1
        if sp.mode == "pair":
            return (1, 1, 1) if rnd == 0 else (2, 2, R // 2 - 1)
        if sp.mode == "hrsag":
            G = sp.group
            return (G, G, R // G - 1) if rnd == 1 else (1, 1, G - 1)
        return 1, 1, R - 1

    def _desc(self, slot, rnd, off, blk=None):
        """Round `rnd`'s send to peer me ^ off (`off` may be traced). A peer's
        send to this rank uses the same offset, hence the same semaphore and
        byte count, so the same descriptor also waits for the receive.
        blk: only the 16-row block `blk` (traced) of the payload."""
        sp, s, me = self.spec, self.scr, self.me
        if sp.mode == "flat":
            src, dst = s["ar_send"].at[slot], s["ar_recv"].at[slot, me]
        elif sp.mode == "rsag":
            src, dst = (
                (s["ar_send"].at[slot, me ^ off], s["ar_recv"].at[slot, me])
                if rnd == 0
                else (s["ar_agsend"].at[slot], s["ar_agrecv"].at[slot, me])
            )
        elif sp.mode == "hrsag":
            G = sp.group
            lme = lax.rem(me, G)
            src, dst = (
                (s["ar_send"].at[slot, lme ^ off], s["ar_recv"].at[slot, lme]),
                (s["ar_xsend"].at[slot], s["ar_xrecv"].at[slot, me // G]),
                (s["ar_agsend"].at[slot], s["ar_agrecv"].at[slot, lme]),
            )[rnd]
        elif sp.mode == "pair":
            src, dst = (
                (s["ar_send"].at[slot], s["ar_recv"].at[slot, lax.rem(me, 2)])
                if rnd == 0
                else (s["ar_agsend"].at[slot], s["ar_agrecv"].at[slot, me // 2])
            )
        else:
            bit = lax.rem(me // (1 << rnd), 2)
            src, dst = s["ar_send"].at[rnd, slot], s["ar_recv"].at[rnd, slot, bit]
        if blk is not None:
            rows = pl.ds(pl.multiple_of(blk * LIVE_BLOCK, LIVE_BLOCK), LIVE_BLOCK)
            src, dst = src.at[rows], dst.at[rows]
        return self._copy(src, dst, rnd, slot, off)

    def copies(self, slot, rnd):
        first, step, count = self._offs(rnd)
        return [self._desc(slot, rnd, first + step * k) for k in range(count)]

    def _each_peer(self, slot, rnd, fn):
        """fn(descriptor) for every peer of the round, in a rolled loop (one
        descriptor site of code whatever the peer count)."""
        first, step, count = self._offs(rnd)

        def peer(off):
            if self.live is None:
                fn(self._desc(slot, rnd, off))
                return
            lax.fori_loop(
                0, self.live, lambda b, c: (fn(self._desc(slot, rnd, off, b)), c)[1], 0
            )

        if count == 1:
            peer(first)
            return

        def body(k, c):
            peer(first + step * k)
            return c

        lax.fori_loop(0, count, body, 0)

    def _round(self, slot, rnd):
        self._each_peer(slot, rnd, lambda d: d.start())
        self._each_peer(slot, rnd, lambda d: d.wait_recv())

    def wait_sends(self, slot):
        """Retire this slot's sends: before its send buffers are rewritten and
        before the kernel exits."""
        for rnd in range(self.spec.n_rounds):
            self._each_peer(slot, rnd, lambda d: d.wait_send())

    def reduce_into(self, src, dst, slot, row_block=16):
        """hrsag of src[0:rows] (f32 ref) into dst[0:rows] (any float dtype),
        rolled over row blocks so code does not grow with rows; the same
        arithmetic as __call__."""
        sp, s, me = self.spec, self.scr, self.me
        if sp.mode != "hrsag":
            dst[0 : sp.rows] = self(src[0 : sp.rows], slot).astype(dst.dtype)
            return
        G, C, R = sp.group, sp.chunk, sp.R
        lme, host = lax.rem(me, G), me // G
        nblk = sp.rows // row_block
        live = nblk
        if self.live is not None:
            assert row_block == LIVE_BLOCK
            live = self.live

        def rows(r):
            return pl.ds(pl.multiple_of(r * row_block, row_block), row_block)

        def over_rows(fn):
            if self.live is None and nblk == 1:
                fn(0)
            else:
                lax.fori_loop(0, live, lambda r, c: (fn(r), c)[1], 0)

        def pack(r):
            for j in range(G):
                s["ar_send"][slot, j, rows(r)] = src[
                    rows(r), j * C : (j + 1) * C
                ].astype(BF16)
            s["ar_recv"][slot, lme, rows(r)] = s["ar_send"][slot, lme, rows(r)]

        def local_sum(r):
            tot = s["ar_recv"][slot, 0, rows(r)].astype(F32)
            for j in range(1, G):
                tot = tot + s["ar_recv"][slot, j, rows(r)].astype(F32)
            s["ar_xsend"][slot, rows(r)] = tot.astype(BF16)
            s["ar_xrecv"][slot, host, rows(r)] = s["ar_xsend"][slot, rows(r)]

        def host_sum(r):
            tot = s["ar_xrecv"][slot, 0, rows(r)].astype(F32)
            for h in range(1, R // G):
                tot = tot + s["ar_xrecv"][slot, h, rows(r)].astype(F32)
            s["ar_agsend"][slot, rows(r)] = tot.astype(BF16)
            s["ar_agrecv"][slot, lme, rows(r)] = s["ar_agsend"][slot, rows(r)]

        def unpack(r):
            for j in range(G):
                dst[rows(r), j * C : (j + 1) * C] = (
                    s["ar_agrecv"][slot, j, rows(r)].astype(F32).astype(dst.dtype)
                )

        over_rows(pack)
        self._round(slot, 0)
        over_rows(local_sum)
        if R > G:
            self._round(slot, 1)
        over_rows(host_sum)
        self._round(slot, 2)
        over_rows(unpack)
        if self.live is not None:  # rows past the live ones: zero, not stale

            def zero(r, c):
                dst[rows(r)] = jnp.zeros((row_block, dst.shape[1]), dst.dtype)
                return c

            lax.fori_loop(live, nblk, zero, 0)

    def __call__(self, x, slot):
        """x f32 [rows, hidden] partial -> f32 [rows, hidden] sum over ranks."""
        sp, s, me = self.spec, self.scr, self.me
        assert self.live is None, "live rows: use reduce_into"
        R = sp.R
        if sp.mode == "flat":
            s["ar_send"][slot] = x.astype(BF16)
            s["ar_recv"][slot, me] = s["ar_send"][slot]
            self._round(slot, 0)
            return _ordered_sum(s["ar_recv"], slot, R)
        if sp.mode == "rsag":
            C = sp.chunk
            for j in range(R):
                s["ar_send"][slot, j] = x[:, j * C : (j + 1) * C].astype(BF16)
            s["ar_recv"][slot, me] = s["ar_send"][slot, me]
            self._round(slot, 0)
            s["ar_agsend"][slot] = _ordered_sum(s["ar_recv"], slot, R).astype(BF16)
            s["ar_agrecv"][slot, me] = s["ar_agsend"][slot]
            self._round(slot, 1)
            return jnp.concatenate(
                [s["ar_agrecv"][slot, j] for j in range(R)], axis=-1
            ).astype(F32)
        if sp.mode == "hrsag":
            G, C = sp.group, sp.chunk
            lme = lax.rem(me, G)
            for j in range(G):
                s["ar_send"][slot, j] = x[:, j * C : (j + 1) * C].astype(BF16)
            s["ar_recv"][slot, lme] = s["ar_send"][slot, lme]
            self._round(slot, 0)
            s["ar_xsend"][slot] = _ordered_sum(s["ar_recv"], slot, G).astype(BF16)
            s["ar_xrecv"][slot, me // G] = s["ar_xsend"][slot]
            if R > G:
                self._round(slot, 1)
            s["ar_agsend"][slot] = _ordered_sum(s["ar_xrecv"], slot, R // G).astype(
                BF16
            )
            s["ar_agrecv"][slot, lme] = s["ar_agsend"][slot]
            self._round(slot, 2)
            return jnp.concatenate(
                [s["ar_agrecv"][slot, j] for j in range(G)], axis=-1
            ).astype(F32)
        if sp.mode == "pair":
            s["ar_send"][slot] = x.astype(BF16)
            s["ar_recv"][slot, lax.rem(me, 2)] = s["ar_send"][slot]
            self._round(slot, 0)
            s["ar_agsend"][slot] = _ordered_sum(s["ar_recv"], slot, 2).astype(BF16)
            s["ar_agrecv"][slot, me // 2] = s["ar_agsend"][slot]
            self._round(slot, 1)
            return _ordered_sum(s["ar_agrecv"], slot, R // 2)
        val = x.astype(BF16)
        tot = None
        for rnd in range(sp.n_rounds):
            if rnd:
                val = tot.astype(BF16)
            s["ar_send"][rnd, slot] = val
            s["ar_recv"][rnd, slot, lax.rem(me // (1 << rnd), 2)] = s["ar_send"][
                rnd, slot
            ]
            self._round(slot, rnd)
            tot = s["ar_recv"][rnd, slot, 0].astype(F32) + s["ar_recv"][
                rnd, slot, 1
            ].astype(F32)
        return tot


def _ordered_sum(ref, slot, n):
    """Sum ref[slot, 0..n-1] in index order (the same order on every rank)."""
    tot = ref[slot, 0].astype(F32)
    for j in range(1, n):
        tot = tot + ref[slot, j].astype(F32)
    return tot


def reference_allreduce(partials, mode, R, group=8):
    """Plain-JAX model of AllReduce: partials f32 [R, rows, hidden] -> the
    f32 [rows, hidden] every rank returns, with the same bf16 roundings."""
    b = lambda v: v.astype(BF16).astype(F32)  # noqa: E731
    if mode == "flat":
        tot = b(partials[0])
        for j in range(1, R):
            tot = tot + b(partials[j])
        return tot
    if mode == "rsag":
        C = partials.shape[-1] // R
        outs = []
        for c in range(R):
            tot = b(partials[0, :, c * C : (c + 1) * C])
            for j in range(1, R):
                tot = tot + b(partials[j, :, c * C : (c + 1) * C])
            outs.append(b(tot))
        return jnp.concatenate(outs, axis=-1)
    if mode == "hrsag":
        G = group
        C = partials.shape[-1] // G
        outs = []
        for c in range(G):
            hosts = []
            for h in range(R // G):
                tot = b(partials[h * G, :, c * C : (c + 1) * C])
                for j in range(1, G):
                    tot = tot + b(partials[h * G + j, :, c * C : (c + 1) * C])
                hosts.append(b(tot))
            tot = hosts[0]
            for v in hosts[1:]:
                tot = tot + v
            outs.append(b(tot))
        return jnp.concatenate(outs, axis=-1)
    if mode == "pair":
        pairs = [b(b(partials[2 * q]) + b(partials[2 * q + 1])) for q in range(R // 2)]
        tot = pairs[0]
        for p in pairs[1:]:
            tot = tot + p
        return tot
    vals = [b(p) for p in partials]
    while len(vals) > 1:
        sums = [vals[2 * i] + vals[2 * i + 1] for i in range(len(vals) // 2)]
        vals = [b(v) for v in sums] if len(sums) > 1 else sums
    return vals[0]


COLLECTIVE_ID = 0


def allreduce_call(mesh, spec: AllReduceSpec, *, n_iter=1, interpret=None):
    """Standalone test/bench wrapper: x f32 [R * rows, hidden] sharded on the
    mesh axis -> the same shape, every shard holding x_{n_iter} where
    x_{i+1} = AllReduce(x_i) / R. Slots alternate between iterations."""
    import jax
    from jax.sharding import PartitionSpec as P

    (ax,) = mesh.axis_names
    names = list(spec.scratch())

    def kernel(x_ref, o_ref, *scratch):
        scr = dict(zip(names, scratch))
        me = lax.axis_index(ax)
        barrier(spec.R, me)
        ar = AllReduce(spec, scr, me)
        x = x_ref[...]
        for i in range(n_iter):
            slot = i % spec.slots
            if i >= spec.slots:
                ar.wait_sends(slot)
            x = ar(x, slot) * (1.0 / spec.R)
        for slot in range(min(n_iter, spec.slots)):
            ar.wait_sends(slot)
        o_ref[...] = x

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((spec.rows, spec.hidden), F32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[spec.scratch()[k] for k in names],
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=100 * 2**20 // 2 + 12 * 2**20
        ),
        interpret=interpret if interpret is not None else False,
        name=f"dsv4_allreduce_{spec.mode}",
    )
    return jax.jit(
        jax.shard_map(call, mesh=mesh, in_specs=P(ax), out_specs=P(ax), check_vma=False)
    )


def alltoall_call(mesh, rows, width, *, n_iter=1, interpret=None):
    """Standalone test/bench of an in-kernel all-to-all: x bf16 [R * R * rows,
    width] sharded on the mesh axis; shard r's block j ([rows, width]) goes to
    shard j's block r. Iterated n_iter times (x <- result)."""
    import jax
    from jax.sharding import PartitionSpec as P

    (ax,) = mesh.axis_names
    R = mesh.shape[ax]

    def kernel(x_ref, o_ref, recv, ssem, rsem):
        me = lax.axis_index(ax)
        barrier(R, me)

        def copies(src, it):
            return [
                pltpu.make_async_remote_copy(
                    src_ref=src.at[me ^ o],
                    dst_ref=recv.at[it % 2, me],
                    send_sem=ssem.at[it % 2, o],
                    recv_sem=rsem.at[it % 2, o],
                    device_id=(me ^ o,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                for o in range(1, R)
            ]

        src = x_ref
        for it in range(n_iter):
            if it >= 2:
                for d in copies(recv.at[it % 2], it):
                    d.wait_send()
            descs = copies(src, it)
            for d in descs:
                d.start()
            recv[it % 2, me] = src[me]
            for d in descs:
                d.wait_recv()
            src = recv.at[it % 2]
        for it in range(max(n_iter - 2, 0), n_iter):
            for d in copies(x_ref if it == 0 else recv.at[(it - 1) % 2], it):
                d.wait_send()
        o_ref[...] = src[...]

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((R, rows, width), BF16),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[
            pltpu.VMEM((2, R, rows, width), BF16),
            pltpu.SemaphoreType.DMA((2, R)),
            pltpu.SemaphoreType.DMA((2, R)),
        ],
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        interpret=interpret if interpret is not None else False,
        name="dsv4_alltoall",
    )

    def local(x):
        return call(x.reshape(R, rows, width)).reshape(R * rows, width)

    return jax.jit(
        jax.shard_map(
            local, mesh=mesh, in_specs=P(ax), out_specs=P(ax), check_vma=False
        )
    )
