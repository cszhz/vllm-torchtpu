# SPDX-License-Identifier: Apache-2.0
"""defer_restore + the folded TP sum against restore + the separate TP sum.

Eight local TPU devices in a permuted mesh order. The fused EP layer's routed
rows are deferred, then summed-and-gathered with a TP-partial block by
small_all_reduce_op.group_fold; every result must be bit-identical to the
layer's own restore, the separate small all-reduce, and the runner's bf16
scale and add.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives.all_reduce import (
    small_all_reduce,
)
from vllm_torchtpu.kernels.fused_moe.v2 import AXIS, WeightFormat, fused_ep_moe_v2

pytestmark = pytest.mark.multichip

_TP4 = ((7, 1, 4, 2), (6, 0, 5, 3))
_TP8 = ((3, 6, 0, 1, 7, 2, 4, 5),)


@pytest.fixture(scope="module")
def mesh():
    devices = jax.local_devices()
    if len(devices) < 8 or devices[0].platform != "tpu":
        pytest.skip("requires eight local TPU devices")
    order = (3, 0, 6, 1, 7, 4, 2, 5)
    return Mesh(np.asarray([devices[i] for i in order]), (AXIS,))


def _inputs(mesh, groups, rows, hidden):
    ep = mesh.shape[AXIS]
    owner = [0] * ep
    for i, group in enumerate(groups):
        for device in group:
            owner[device] = i
    rank = jax.device_put(
        np.arange(ep, dtype=np.int32).reshape(ep, 1), NamedSharding(mesh, P(AXIS))
    )

    def init(rank_l):
        me = rank_l[0, 0]
        group_id = jnp.asarray(owner, dtype=jnp.int32)[me]
        tk = jax.random.split(jax.random.fold_in(jax.random.key(123), group_id), 2)
        wk = jax.random.split(jax.random.fold_in(jax.random.key(456), me), 3)
        x = (jax.random.normal(tk[0], (rows, hidden)) * 0.2).astype(jnp.bfloat16)
        gate = jax.random.normal(tk[1], (rows, 2 * ep))
        shared = jax.random.normal(wk[2], (rows, hidden)).astype(jnp.bfloat16)

        def weights(key, k, n):
            raw = jax.random.normal(key, (2, k, n)) / jnp.sqrt(float(k))
            dtype = jnp.float8_e4m3fn
            scale = jnp.maximum(
                jnp.max(jnp.abs(raw), axis=1, keepdims=True)
                / float(jnp.finfo(dtype).max),
                1e-12,
            )
            return (raw / scale).astype(dtype), scale[:, 0, :]

        w1, s1 = weights(wk[0], hidden, 1024)
        w2, s2 = weights(wk[1], 512, hidden)
        return x, w1, w2, s1, s2, gate, shared

    out = jax.jit(
        jax.shard_map(
            init, mesh=mesh, in_specs=P(AXIS), out_specs=(P(AXIS),) * 7, check_vma=False
        )
    )(rank)
    return rank, out[:6], out[6]


def _tables(mesh, groups):
    ep = mesh.shape[AXIS]
    slot, peers = [0] * ep, [None] * ep
    for group in groups:
        for i, m in enumerate(group):
            slot[m], peers[m] = i, group
    sh = NamedSharding(mesh, P(AXIS))
    return (
        jax.device_put(np.asarray(slot, np.int32), sh),
        jax.device_put(np.asarray(peers, np.int32).reshape(-1), sh),
    )


@pytest.mark.parametrize(
    "groups,rows,hidden",
    [
        pytest.param(_TP8, 8, 1024, id="tp8-r8"),
        pytest.param(_TP8, 24, 1024, id="tp8-r24-padded"),
        pytest.param(_TP8, 3, 1024, id="tp8-r3-empty-members"),
        pytest.param(_TP4, 32, 512, id="tp4-r32"),
        pytest.param(_TP4, 7, 512, id="tp4-r7-tail"),
        pytest.param(_TP8, 128, 1024, id="tp8-r128-xla"),
    ],
)
@pytest.mark.parametrize(
    "scale", [None, 2.5, 2.446], ids=["pair", "add2.5", "add2.446"]
)
def test_fold_matches_restore_and_separate_sum(mesh, groups, rows, hidden, scale):
    from vllm_torchtpu.layers.adapter.small_all_reduce_op import bf16_scale, group_fold

    rank, inputs, shared = _inputs(mesh, groups, rows, hidden)
    width = len(groups[0])

    def layer(defer):
        def run(rank_arg, *args):
            return fused_ep_moe_v2(
                *args,
                topk=4,
                renormalize=True,
                mesh=mesh,
                capacity=128,
                weight_format=WeightFormat.FP8,
                rank=rank_arg,
                sharded_plan=True,
                token_replica_groups=groups,
                pallas_restore=True,
                decode_combine=True,
                defer_restore=defer,
            )

        return jax.block_until_ready(jax.jit(run)(rank, *inputs))

    restored, deferred = layer(False), layer(True)
    slot, peers = _tables(mesh, groups)
    cap = 64
    sm = lambda body, n: jax.jit(  # noqa: E731
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS),) * 4,
            out_specs=(P(AXIS),) * n if n > 1 else P(AXIS),
            check_vma=False,
        )
    )

    def separate(x, routed, r, p):
        if rows > cap:
            total = jax.lax.psum(x, AXIS, axis_index_groups=[list(g) for g in groups])
        else:
            total = small_all_reduce(x, num_cores=width, axis=AXIS, rank=r, peers=p)
        return total, routed

    total, routed = sm(separate, 2)(shared, restored, slot, peers)
    s = None if scale is None else bf16_scale(scale)
    folded = sm(group_fold(groups, cap, AXIS, s), 1 if s is not None else 2)(
        shared, deferred, slot, peers
    )
    bits = lambda a: np.asarray(a).view(np.uint16)  # noqa: E731
    if scale is None:
        np.testing.assert_array_equal(bits(folded[0]), bits(total))
        np.testing.assert_array_equal(bits(folded[1]), bits(routed))
        return
    # What the runner's `routed *= scale; shared + routed` computes on TPU:
    # the scalar rounded to bf16, the product rounded, the sum rounded
    # (probe_scaled_add_rounding.py).
    f32 = lambda a: np.asarray(a).astype(np.float32)  # noqa: E731
    bf = lambda a: np.asarray(a, np.float32).astype(jnp.bfloat16)  # noqa: E731
    want = bf(f32(total) + f32(bf(f32(routed) * np.float32(s))))
    np.testing.assert_array_equal(bits(folded), bits(want))
