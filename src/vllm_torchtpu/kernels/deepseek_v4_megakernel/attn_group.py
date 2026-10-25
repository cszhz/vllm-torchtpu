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
"""The CSA top-k of one request split across the P ranks of an attention group.

Each rank scores its own range of compressed columns (rank order = column
order) and takes its local per-row top-k. Exact global selection without
moving any column lists:

1. all-gather each rank's local top-k keys per row (monotone int keys,
   ``tc_topk.monotone_key``, INT_MIN for missing entries);
2. every rank computes, identically, each row's global threshold G = the k-th
   largest gathered key, and the tie quota k - #(keys > G), handed out in
   stock tie order (lower column first = lower rank first);
3. each rank keeps its columns with key > G plus its share of the ties (in
   column order) and compacts them into the union format; the union of the
   ranks' unions, concatenated in rank order, is the single-rank union.

The global top-k is inside the union of the local top-k sets, and a rank
holding more than k keys >= G has its own list full of them, so its share
never exceeds the ties in its list.
"""

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4_megakernel import tc_topk

I32 = jnp.int32
INT_MIN, INT_MAX = tc_topk.INT_MIN, tc_topk.INT_MAX


class Group:
    """This rank's place in its attention group: P ranks at mesh indices
    base .. base + P - 1, this one is jj."""

    def __init__(self, me, P):
        self.P, self.jj = P, me % P
        self.base = me - self.jj

    def peer(self, i):
        return self.base + i


def all_gather(group, src, recv, ready, send_sem, recv_sem, *, local=True):
    """recv[i] <- src of group rank i, for every i (recv[jj] is a local copy,
    skipped with local=False when src is recv[jj] itself).
    `ready` (REGULAR) makes it safe to reuse recv: every rank first tells its
    peers its recv is free, and sends only once all have."""
    P, jj = group.P, group.jj
    for k in range(1, P):
        pl.semaphore_signal(ready, 1, device_id=(group.peer((jj + k) % P),),
                            device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(ready, P - 1)
    sends = []
    for k in range(1, P):
        cp = pltpu.make_async_remote_copy(src, recv.at[jj], send_sem, recv_sem,
                                          device_id=(group.peer((jj + k) % P),),
                                          device_id_type=pl.DeviceIdType.MESH)
        cp.start()
        sends.append(cp)
    if local:
        cp = pltpu.make_async_copy(src, recv.at[jj], send_sem)
        cp.start()
        cp.wait()
    for cp in sends:
        cp.wait_send()
    for cp in sends:  # P - 1 arrivals of the same size
        cp.wait_recv()


def global_cut(keys, k):
    """keys: int32 [8, n] (row t: all ranks' local top-k keys, rank-major;
    INT_MIN = missing). Per row, int32 [8, 1]: G = the k-th largest key, the
    number of keys > G, and 1 where fewer than k keys are eligible (> key(-inf):
    then every rank's list holds all its eligible keys, and all are taken)."""
    neg = jnp.int32(tc_topk.KEY_NEG_INF)
    n_elig = jnp.sum((keys > neg).astype(I32), axis=1, keepdims=True)

    def count_ge(x):
        return jnp.sum((keys >= x).astype(I32), axis=1, keepdims=True)

    # largest x with count(keys >= x) >= k, by bisection on the int32 key space
    # (lo always qualifies; hi never does); 33 halvings close the int32 range
    lo = jnp.full((8, 1), INT_MIN, I32)
    hi = jnp.full((8, 1), INT_MAX, I32)

    def step(_, c):
        lo, hi = c
        mid = (lo >> 1) + (hi >> 1) + (lo & hi & 1)  # floor((lo + hi) / 2), no overflow
        ok = (lo < hi) & (count_ge(mid + 1) >= k)  # does mid + 1 still qualify?
        return jnp.where(ok, mid + 1, lo), jnp.where(ok, hi, mid)

    lo, hi = lax.fori_loop(0, 33, step, (lo, hi))
    g = lo
    n_gt = jnp.sum((keys > g).astype(I32), axis=1, keepdims=True)
    return g, n_gt, (n_elig < k).astype(I32)
