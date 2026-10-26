# SPDX-License-Identifier: Apache-2.0
"""The ragged all-to-all dispatch's two plans against a model of the exchange.

The layer sends each wire row to the shards whose experts it selected and
reads the received rows through a remapped token index (layer.py,
`_all_to_all_send_plan` / `_all_to_all_recv_plan`). These tests run the two
plans for every shard of a random routing on the host and perform the
all-to-all in numpy by the collective's own rule -- shard s's run for shard d
lands at s's block of d's buffer -- so a disagreement between what a sender
packs and what a receiver expects shows here, with no device.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import layer

pytestmark = pytest.mark.cpu_test


def _routing(rng, *, ep, t_local, topk, e_total):
    """[ep, t_local, topk] distinct slot ids per row; row 0 of every shard
    routes nowhere (every slot the sentinel e_total), as the layer's padding
    rows do; shard 1's row 1 selects two slots of one shard."""
    sel = np.stack(
        [
            np.stack(
                [rng.choice(e_total, size=topk, replace=False) for _ in range(t_local)]
            )
            for _ in range(ep)
        ]
    ).astype(np.int32)
    sel[:, 0, :] = e_total
    g_local = e_total // ep
    sel[1, 1, 0], sel[1, 1, 1] = (
        2 * g_local,
        2 * g_local + 1 if g_local > 1 else 2 * g_local,
    )
    return sel


@pytest.mark.parametrize(
    "ep,t_local,topk,g_local",
    [(8, 32, 4, 4), (32, 144, 6, 13), (4, 16, 6, 3)],
    ids=["dp8", "dp32-c768-13-slots", "ep4-topk-over-ep"],
)
def test_the_plans_agree_with_a_simulated_exchange(ep, t_local, topk, g_local):
    rng = np.random.default_rng(ep * 1000 + t_local)
    e_total = ep * g_local
    sel = _routing(rng, ep=ep, t_local=t_local, topk=topk, e_total=e_total)
    pairs_g = jnp.asarray(sel.reshape(ep, t_local * topk))
    sends = [
        tuple(
            np.asarray(a)
            for a in layer._all_to_all_send_plan(
                jnp.asarray(sel[s].reshape(-1)),
                g_local=g_local,
                ep=ep,
                t_local=t_local,
                topk=topk,
            )
        )
        for s in range(ep)
    ]
    recvs = [
        tuple(
            np.asarray(a)
            for a in layer._all_to_all_recv_plan(
                pairs_g, d, g_local=g_local, ep=ep, t_local=t_local, topk=topk
            )
        )
        for d in range(ep)
    ]
    goes = np.any(
        sel[:, :, :, None] // g_local == np.arange(ep), axis=2
    )  # [src, row, dst]
    send_sizes = np.stack([s[1] for s in sends])  # [src, dst]
    recv_sizes = np.stack([r[0] for r in recvs])  # [dst, src]
    # A row is sent once per destination however many of its slots are there.
    np.testing.assert_array_equal(send_sizes, goes.sum(axis=1))
    # The collective's precondition: what d expects from s is what s sends d.
    np.testing.assert_array_equal(recv_sizes, send_sizes.T)
    for s in range(ep):
        send_rows, sizes, offsets = sends[s]
        np.testing.assert_array_equal(offsets, np.cumsum(sizes) - sizes)
        assert send_rows.shape == (t_local * min(topk, ep),)
        for d in range(ep):
            run = send_rows[offsets[d] : offsets[d] + sizes[d]]
            # Each run is that destination's rows in row order.
            np.testing.assert_array_equal(run, np.flatnonzero(goes[s, :, d]))
    # The exchange itself: token (s, i) carries the value s * t_local + i.
    T = ep * t_local
    buffers = np.full((ep, T), -1)
    for s in range(ep):
        send_rows, sizes, offsets = sends[s]
        for d in range(ep):
            run = send_rows[offsets[d] : offsets[d] + sizes[d]]
            buffers[d, s * t_local : s * t_local + sizes[d]] = s * t_local + run
    for d in range(ep):
        _, token_index = recvs[d]
        assert (
            token_index.shape == (T,)
            and token_index.min() >= 0
            and token_index.max() < T
        )
        for s in range(ep):
            for i in np.flatnonzero(goes[s, :, d]):
                assert buffers[d, token_index[s * t_local + i]] == s * t_local + i
    # Fewer rows than the gather's T per shard, and never more than the bound.
    assert recv_sizes.sum(axis=1).max() < T
    assert send_sizes.sum(axis=1).max() <= t_local * min(topk, ep)


def test_the_all_to_all_dispatch_is_refused_off_the_fused_stage():
    import jax
    from jax.sharding import Mesh

    from vllm_torchtpu.kernels.fused_moe.v2.layer import fused_ep_moe_v2

    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), axis_names=("x",))
    x = jnp.zeros((8, 256), jnp.bfloat16)
    w1 = jnp.zeros((4, 256, 512), jnp.bfloat16)
    w2 = jnp.zeros((4, 256, 256), jnp.bfloat16)
    gating = jnp.zeros((8, 4), jnp.float32)
    with pytest.raises(ValueError, match="fused stage"):
        fused_ep_moe_v2(
            x,
            w1,
            w2,
            None,
            None,
            gating,
            topk=2,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            weight_format=layer.WeightFormat.BF16,
            stage="route",
            dispatch="all_to_all",
        )
    with pytest.raises(ValueError, match="dispatch must be one of"):
        fused_ep_moe_v2(
            x,
            w1,
            w2,
            None,
            None,
            gating,
            topk=2,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            weight_format=layer.WeightFormat.BF16,
            dispatch="broadcast",
        )
