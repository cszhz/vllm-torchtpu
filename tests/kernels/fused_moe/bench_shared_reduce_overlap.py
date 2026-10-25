# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B023, E501
"""Fused EP MoE layer + the shared expert's TP sum, prefill shapes, 8 TPU cores.

    A  fused_ep_moe_v2 (restore / local rows), then XLA psum (psum_scatter)
       of the shared partial -- what the runner does today;
    B  fused_ep_moe_v2(reduce_order_after=True): the Pallas reduce-scatter's
       transfers overlap the kernel.
Checks B's sum against the float32 group-order reference (bit-exact) and
reports its max abs / rel difference to XLA's sum; times A and B per layer.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2 import AXIS, WeightFormat, fused_ep_moe_v2

H = 7168
IT = 5


def tm(f, *a):
    jax.block_until_ready(f(*a))
    b = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*a))
        b = min(b, time.perf_counter() - t0)
    return b / IT * 1e6


def main():
    devs = jax.local_devices()
    ep = len(devs)
    mesh = Mesh(np.asarray(devs), (AXIS,))
    sh = NamedSharding(mesh, P(AXIS))
    g_local, inter, topk = 24, 3072, 6
    e_total = g_local * ep
    for width, rows in [(4, 512), (4, 256), (8, 2048), (8, 1024)]:
        groups = tuple(tuple(range(g, g + width)) for g in range(0, ep, width))
        owner = [i // width for i in range(ep)]
        rank = jax.device_put(np.arange(ep, dtype=np.int32).reshape(ep, 1), sh)

        def init(rank_l):
            me = rank_l[0, 0]
            gid = jnp.asarray(owner, jnp.int32)[me]
            tk = jax.random.split(jax.random.fold_in(jax.random.key(1), gid), 2)
            wk = jax.random.split(jax.random.fold_in(jax.random.key(2), me), 3)
            x = (jax.random.normal(tk[0], (rows, H)) * 0.2).astype(jnp.bfloat16)
            gate = jax.random.normal(tk[1], (rows, e_total))
            shared = jax.random.normal(wk[2], (rows, H)).astype(jnp.bfloat16)

            def weights(key, k, n):
                raw = jax.random.normal(key, (g_local, k, n)) / jnp.sqrt(float(k))
                scale = jnp.maximum(
                    jnp.max(jnp.abs(raw), axis=1, keepdims=True) / 448.0, 1e-12
                )
                return (raw / scale).astype(jnp.float8_e4m3fn), scale[:, 0, :]

            w1, s1 = weights(wk[0], H, 2 * inter)
            w2, s2 = weights(wk[1], inter, H)
            return x, w1, w2, s1, s2, gate, shared

        x, w1, w2, s1, s2, gate, shared = jax.jit(
            jax.shard_map(
                init,
                mesh=mesh,
                in_specs=P(AXIS),
                out_specs=(P(AXIS),) * 7,
                check_vma=False,
            )
        )(rank)
        ig = [list(g) for g in groups]
        for local in (False, True):

            def layer(xa, w1a, w2a, s1a, s2a, ga, sa, ra, reduce):
                return fused_ep_moe_v2(
                    xa,
                    w1a,
                    w2a,
                    s1a,
                    s2a,
                    ga,
                    topk=topk,
                    renormalize=True,
                    mesh=mesh,
                    capacity=128,
                    weight_format=WeightFormat.FP8,
                    rank=ra,
                    sharded_plan=True,
                    token_replica_groups=groups,
                    order_after=sa,
                    restore_rows=not local,
                    reduce_order_after=reduce,
                )

            def xla_sum(sa):
                def body(b):
                    if local:
                        return lax.psum_scatter(
                            b,
                            AXIS,
                            scatter_dimension=0,
                            tiled=True,
                            axis_index_groups=ig,
                        )
                    return lax.psum(b, AXIS, axis_index_groups=ig)

                return jax.shard_map(
                    body,
                    mesh=mesh,
                    in_specs=P(AXIS),
                    out_specs=P(AXIS),
                    check_vma=False,
                )(sa)

            def A(*a):
                def body(i, carry):
                    routed = layer(*a[:6], carry, a[7], False)
                    return (
                        carry
                        + (xla_sum(carry).sum() + routed.sum()).astype(carry.dtype) * 0
                    )

                return lax.fori_loop(0, IT, body, a[6])

            def B(*a):
                def body(i, carry):
                    routed, red = layer(*a[:6], carry, a[7], True)
                    return carry + (red.sum() + routed.sum()).astype(carry.dtype) * 0

                return lax.fori_loop(0, IT, body, a[6])

            args = (x, w1, w2, s1, s2, gate, shared, rank)
            ta = tm(jax.jit(A), *args)
            tb = tm(jax.jit(B), *args)
            t_layer = tm(
                jax.jit(
                    lambda *a: lax.fori_loop(
                        0,
                        IT,
                        lambda i, c: (
                            c + layer(*a[:6], c, a[7], False).sum().astype(c.dtype) * 0
                        ),
                        a[6],
                    )
                ),
                *args,
            )
            # numerics
            routed_b, red = jax.jit(lambda *a: layer(*a[:6], a[6], a[7], True))(*args)
            routed_a = jax.jit(lambda *a: layer(*a[:6], a[6], a[7], False))(*args)
            ref_x = jax.jit(xla_sum)(shared)
            sn = np.asarray(shared.astype(jnp.float32)).reshape(ep, rows, H)
            want = []
            for m in range(ep):
                g = groups[owner[m]]
                acc = np.zeros((rows, H), np.float32)
                for q in g:
                    acc = acc + sn[q]
                full = np.asarray(
                    jnp.asarray(acc).astype(jnp.bfloat16).astype(jnp.float32)
                )
                slot = g.index(m)
                r = rows // width
                want.append(full[slot * r : (slot + 1) * r] if local else full)
            want = np.concatenate(want)
            got = np.asarray(red.astype(jnp.float32))
            xs = np.asarray(ref_x.astype(jnp.float32))
            d = np.abs(got - xs)
            rel = np.linalg.norm(got - xs) / np.linalg.norm(xs)
            same_routed = bool(
                np.array_equal(
                    np.asarray(routed_a).view(np.uint16),
                    np.asarray(routed_b).view(np.uint16),
                )
            )
            print(
                f"TP{width} T{rows} {'local' if local else 'full'}: layer {t_layer:7.1f} A(layer+XLA sum) {ta:7.1f} B(overlap) {tb:7.1f} us"
                f" | saved {ta - tb:6.1f} us | sum==f32 group order {bool(np.array_equal(got, want))}"
                f" | vs XLA max abs {d.max():.3g} rel L2 {rel:.3g} | routed bit-exact {same_routed}",
                flush=True,
            )


if __name__ == "__main__":
    main()
