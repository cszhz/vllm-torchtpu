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

import re

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import host, layer

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


@pytest.mark.parametrize("form", ["compare", "scatter"])
@pytest.mark.parametrize(
    "ep,t_local,topk,g_local",
    [(8, 32, 4, 4), (32, 144, 6, 13), (4, 16, 6, 3)],
    ids=["dp8", "dp32-c768-13-slots", "ep4-topk-over-ep"],
)
def test_the_plans_agree_with_a_simulated_exchange(
    ep, t_local, topk, g_local, form, monkeypatch
):
    """Both forms of the send plan: the compare-and-count the decode shapes
    take and the scatter a prefill-sized shard falls back to."""
    if form == "scatter":
        monkeypatch.setattr(layer, "SEND_PLAN_COMPARE_MAX", 0)
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


@pytest.mark.parametrize("form", ["compare", "scatter"])
@pytest.mark.parametrize("groups", [1, 2, 4, 13])
@pytest.mark.parametrize(
    "ep,t_local,topk,g_local",
    [(8, 32, 4, 4), (32, 144, 6, 13), (4, 16, 6, 3)],
    ids=["dp8", "dp32-c768-13-slots", "ep4-topk-over-ep"],
)
def test_the_grouped_plans_agree_with_a_simulated_exchange(
    ep, t_local, topk, g_local, groups, form, monkeypatch
):
    """The in-kernel dispatch's plans: same rows and same bytes as the
    ungrouped pair, ordered (destination, group, row)-major, and a row that
    several of a destination's groups want delivered with the lowest of
    them."""
    if groups > g_local:
        pytest.skip("a group is a nonempty run of local slots")
    if form == "scatter":
        monkeypatch.setattr(layer, "SEND_PLAN_COMPARE_MAX", 0)
    rng = np.random.default_rng(ep * 1000 + t_local + groups)
    e_total = ep * g_local
    sel = _routing(rng, ep=ep, t_local=t_local, topk=topk, e_total=e_total)
    pairs_g = jnp.asarray(sel.reshape(ep, t_local * topk))
    kw = dict(g_local=g_local, ep=ep, t_local=t_local, topk=topk, groups=groups)
    sends = [
        tuple(
            np.asarray(a)
            for a in layer._all_to_all_send_plan_grouped(
                jnp.asarray(sel[s].reshape(-1)), **kw
            )
        )
        for s in range(ep)
    ]
    recvs = [
        tuple(
            np.asarray(a)
            for a in layer._all_to_all_recv_plan_grouped(pairs_g, d, **kw)
        )
        for d in range(ep)
    ]
    spg = -(-g_local // groups)
    dest = sel // g_local
    grp = np.minimum((sel % g_local) // spg, groups - 1)
    # The group a (row, destination) pair travels in: the lowest of the
    # destination's groups that selected it, `groups` where it goes nowhere.
    gmin = np.where(dest[:, :, :, None] == np.arange(ep), grp[:, :, :, None], groups)
    gmin = gmin.min(axis=2)  # [src, row, dst]
    goes = gmin < groups
    send_sizes = np.stack([s[1] for s in sends])  # [src, dst, g]
    recv_sizes = np.stack([r[0] for r in recvs])  # [dst, src, g]
    np.testing.assert_array_equal(
        send_sizes, (gmin[:, :, :, None] == np.arange(groups)).sum(axis=1)
    )
    # The collective's precondition, per group: what d expects from s is
    # what s sends d, run for run.
    np.testing.assert_array_equal(recv_sizes, send_sizes.transpose(1, 0, 2))
    # Same bytes as the ungrouped plan: a row still goes to a destination
    # once, however many of its groups wanted it.
    np.testing.assert_array_equal(send_sizes.sum(axis=2), goes.sum(axis=1))
    for s in range(ep):
        send_rows, sizes, offsets = sends[s]
        np.testing.assert_array_equal(
            offsets.reshape(-1), np.cumsum(sizes) - sizes.reshape(-1)
        )
        for d in range(ep):
            for g in range(groups):
                lo = offsets[d, g]
                run = send_rows[lo : lo + sizes[d, g]]
                np.testing.assert_array_equal(run, np.flatnonzero(gmin[s, :, d] == g))
    # The exchange itself: token (s, i) carries the value s * t_local + i,
    # and source s's whole contribution still lands in block s.
    T = ep * t_local
    buffers = np.full((ep, T), -1)
    for s in range(ep):
        send_rows, sizes, offsets = sends[s]
        base = offsets[:, 0]
        for d in range(ep):
            n = sizes[d].sum()
            run = send_rows[base[d] : base[d] + n]
            buffers[d, s * t_local : s * t_local + n] = s * t_local + run
    for d in range(ep):
        _, token_index = recvs[d]
        assert token_index.shape == (T,)
        assert token_index.min() >= 0 and token_index.max() < T
        for s in range(ep):
            for i in np.flatnonzero(goes[s, :, d]):
                assert buffers[d, token_index[s * t_local + i]] == s * t_local + i


@pytest.mark.parametrize(
    "ep,t_local,topk,g_local",
    [(8, 32, 4, 4), (32, 144, 6, 13)],
    ids=["dp8", "dp32-c768-13-slots"],
)
@pytest.mark.parametrize("groups", [1, 2, 4, 13])
def test_a_group_is_waited_for_before_any_of_its_rows_are_read(
    ep, t_local, topk, g_local, groups
):
    """The kernel's wait schedule is sound: the visits come out group-major,
    so each group is waited for exactly once, and every arriving row is
    waited for by the group that is visited no later than the expert reading
    it."""
    if groups > g_local:
        pytest.skip("a group is a nonempty run of local slots")
    rng = np.random.default_rng(7 * ep + groups)
    e_total = ep * g_local
    sel = _routing(rng, ep=ep, t_local=t_local, topk=topk, e_total=e_total)
    pairs_g = jnp.asarray(sel.reshape(ep, t_local * topk))
    spg = -(-g_local // groups)
    for me in range(ep):
        recv_sizes, _ = layer._all_to_all_recv_plan_grouped(
            pairs_g, me, g_local=g_local, ep=ep, t_local=t_local, topk=topk,
            groups=groups,
        )
        arriving = np.asarray(recv_sizes).sum(axis=0)  # [groups]
        rows = np.zeros(g_local, np.int32)
        for e in range(g_local):
            rows[e] = (sel[:, :, :] == me * g_local + e).any(axis=2).sum()
        visit, n_visit = host.expert_visit_list(
            jnp.asarray(rows), g_local, groups=groups
        )
        visit, n_visit = np.asarray(visit), int(np.asarray(n_visit)[0])
        visited = visit[:n_visit]
        # Every nonempty expert once, group-major.
        np.testing.assert_array_equal(sorted(visited), np.flatnonzero(rows))
        np.testing.assert_array_equal(
            np.minimum(visited // spg, groups - 1),
            sorted(np.minimum(visited // spg, groups - 1)),
        )
        # A group with rows on the wire has a visit to wait at, and the
        # kernel's "first visit of this group" predicate fires once per group.
        fires = np.minimum(visited // spg, groups - 1)
        for g in range(groups):
            if arriving[g]:
                assert (fires == g).sum() >= 1
        assert len(set(fires)) == len(
            [g for g in range(groups) if (fires == g).any()]
        )


@pytest.mark.parametrize("dispatch", ["all_to_all", "pallas_all_to_all"])
def test_the_all_to_all_dispatch_is_refused_off_the_fused_stage(dispatch):
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
            dispatch=dispatch,
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


@pytest.mark.parametrize(
    "issue_env",
    [
        {},
        {"MOE_FUSED_EP_DISPATCH_BARRIER_ROUNDS": "1"},
        {"MOE_FUSED_EP_DISPATCH_ORDER": "farthest"},
        {"MOE_FUSED_EP_DISPATCH_PAYLOAD_DRAIN": "wait"},
    ],
    ids=["shipped", "barrier1", "farthest", "drain-in-wait"],
)
@pytest.mark.parametrize("with_payload", [False, True], ids=["rows", "rows+payload"])
def test_the_pallas_exchange_lowers_for_tpu_from_a_host(
    with_payload, issue_env, monkeypatch
):
    """The issue/wait pair hands DMA semaphores from one pallas_call to the
    next. Lowered for TPU from this host -- the way torch_tpu's jax.export
    bridge sees it, for an 8-way mesh of v7x cores -- every wait must take its
    semaphores straight from the issue, and no call may be marked
    side-effecting: the bridge's SPMD partitioner refuses a side-effecting
    custom call that carries no sharding of its own. With the routing payload
    on the same issue there is a third call, the payload's wait."""
    import jax
    from jax import export, lax
    from jax.sharding import AbstractMesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.fused_moe.v2.dispatch import (
        exchange_start,
        exchange_wait,
        payload_wait,
    )

    for name, value in issue_env.items():
        monkeypatch.setenv(name, value)
    drain_in_wait = issue_env.get("MOE_FUSED_EP_DISPATCH_PAYLOAD_DRAIN") == "wait"

    ep, t_local, sent, width = 8, 144, 864, 1664
    mesh = AbstractMesh(
        (ep,),
        ("d",),
        abstract_device=jax.sharding.AbstractDevice(
            device_kind="TPU7x", num_cores=1, platform="tpu"
        ),
    )

    def body(x_l, sizes_l, rank_l, packed_l):
        sizes = sizes_l[0]
        offsets = jnp.cumsum(sizes) - sizes
        if not with_payload:
            recv, send_sem, recv_sem = exchange_start(
                x_l, sizes, offsets, rank_l, t_local=t_local
            )
            recv, rank_l = lax.optimization_barrier((recv, rank_l))
            expected = jnp.stack([jnp.sum(sizes), jnp.sum(sizes) - sizes[rank_l[0, 0]]])
            return exchange_wait(
                recv, x_l, send_sem, recv_sem, expected, t_local=t_local
            ), packed_l
        packed = packed_l[0]
        recv, pbuf, send_sem, recv_sem, precv_sem, *undrained = exchange_start(
            x_l, sizes, offsets, rank_l, t_local=t_local, payload=packed
        )
        recv, rank_l = lax.optimization_barrier((recv, rank_l))
        psend_sem, src = undrained if undrained else (None, None)
        table = payload_wait(pbuf, precv_sem, width=width, psend_sem=psend_sem, src=src)
        expected = jnp.stack([jnp.sum(sizes), jnp.sum(sizes) - sizes[rank_l[0, 0]]])
        rows = exchange_wait(recv, x_l, send_sem, recv_sem, expected, t_local=t_local)
        return rows, table[rank_l[0, 0]][None]

    shard = NamedSharding(mesh, P("d"))
    specs = (
        jax.ShapeDtypeStruct((ep * sent, 40, 128), jnp.float8_e4m3fn, sharding=shard),
        jax.ShapeDtypeStruct((ep, ep), jnp.int32, sharding=shard),
        jax.ShapeDtypeStruct((ep, 1), jnp.int32, sharding=shard),
        jax.ShapeDtypeStruct((ep, width), jnp.int32, sharding=shard),
    )
    fn = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(P("d"),) * 4,
        out_specs=(P("d"), P("d")),
        check_vma=False,
    )
    with jax.sharding.use_abstract_mesh(mesh):
        module = export.export(jax.jit(fn), platforms=["tpu"])(*specs).mlir_module()
    calls = [line for line in module.splitlines() if "tpu_custom_call" in line]
    assert len(calls) == (3 if with_payload else 2)
    assert not any("has_side_effect = true" in line for line in calls)
    # The issue returns its buffers then its semaphores; each wait takes its
    # semaphores straight from it.
    start = re.match(r"\s*(%[\w.]+):(\d+) = ", calls[0])
    assert start is not None, calls[0]
    name, results = start.group(1), int(start.group(2))
    if not with_payload:
        assert results == 3
        assert f"{name}#1, {name}#2)" in calls[1], calls[1]
    elif not drain_in_wait:
        # rows, payload table, rows send/recv sems, payload recv sem; the
        # payload's send sem is the issue's own scratch (drained there).
        assert results == 5
        assert f"{name}#1, {name}#4)" in calls[1], calls[1]  # payload wait
        assert f"{name}#2, {name}#3)" in calls[2], calls[2]  # rows wait
    else:
        # One more: the payload's send semaphore, which the payload wait
        # drains, holding the source block alive as its second operand.
        assert results == 6
        assert f"{name}#1, " in calls[1] and f"{name}#4, {name}#5)" in calls[1]
        assert f"{name}#2, {name}#3)" in calls[2], calls[2]  # rows wait
