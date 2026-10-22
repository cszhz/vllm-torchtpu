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
"""Qwen3.8 segment kernel vs the composed pure-JAX segment reference (CPU)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen35, qwen38
from vllm_torchtpu.megakernel.models.qwen38_pack import pack_weights, shard_step_inputs
from vllm_torchtpu.megakernel.models.qwen38_reference import segment_reference
from vllm_torchtpu.megakernel.ops import BF16, F32, rup

_TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
G = qwen38.TINY


def _hc(key: jax.Array, n: int) -> dict[str, jax.Array]:
    W = G.NHC * G.H
    k = jax.random.split(key, 4)
    return {
        "norm": jax.random.normal(k[0], (n, W), F32) * 0.05,
        "down": (jax.random.normal(k[1], (n, W, G.LOWRANK), F32) * 0.03).astype(BF16),
        "up": (jax.random.normal(k[2], (n, G.LOWRANK, W), F32) * 0.1).astype(BF16),
        "inj": (jax.random.normal(k[3], (n, W, G.NHC), F32) * 0.3).astype(BF16),
    }


def _hc_weights(name: str, p: dict[str, Any], R: int, mix: bool) -> dict[str, Any]:
    n, W = p["down"].shape[0], G.NHC * G.H

    def rep(x):
        return jnp.broadcast_to(x[None], (R, *x.shape))

    if not mix:
        hinj = jnp.zeros((n, W, 128), BF16).at[:, :, : G.NHC].set(p["inj"])
        return {f"{name}_hinj": rep(hinj)}
    n_down, k_up = rup(G.LOWRANK + G.NHC, 128), rup(G.LOWRANK, 128)
    hdn = jnp.zeros((n, W, n_down), BF16)
    hdn = hdn.at[:, :, : G.LOWRANK].set(p["down"])
    hdn = hdn.at[:, :, G.LOWRANK : G.LOWRANK + G.NHC].set(p["inj"])
    hup = jnp.zeros((n, k_up, W), BF16).at[:, : G.LOWRANK].set(p["up"])
    out = {f"{name}_hdn": rep(hdn)}
    for b in range(G.NHC):
        out[f"{name}_hup{b}"] = rep(hup[..., b * G.H : (b + 1) * G.H])
    return out


def _rows(arr: jax.Array, rows: tuple[int, ...], L: int) -> jax.Array:
    """Scatter per-rank `[R, n, w]` smalls into kernel-layer rows of `[R, L, w]`."""
    out = jnp.zeros((arr.shape[0], L, arr.shape[-1]), F32)
    return out.at[:, list(rows)].set(arr.astype(F32))


def _run(T: int, key: int = 40):
    L, R, B = qwen38.SEGMENT_LAYERS, 2, 2 * T
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=_TILES)
    gi = G.inner(R, L)
    spec = qwen38.segment_spec(G, R, L=L)
    keys = jax.random.split(jax.random.PRNGKey(key), 6)
    pos, _, w_l, st_l = qwen35.random_logical_inputs(keys[0], gi, ex)
    w_l = {**w_l, "e2_s": w_l["e2_s"] * 2000.0, "sh2_s": w_l["sh2_s"] * 400.0,
           "lin_out_s": w_l["lin_out_s"] * 50.0}
    hc = {"qsa_out": _hc(keys[1], 1), "gdn": _hc(keys[2], L - 1), "moe": _hc(keys[3], L)}
    K = G.NH // R * G.HD
    attn = (jax.random.normal(keys[4], (R, B, K), F32) * 0.5).astype(BF16)
    o_proj = (jax.random.normal(keys[5], (R, K, G.H), F32) * 0.05).astype(BF16)
    x = jax.random.normal(jax.random.PRNGKey(key + 1), (B, G.NHC * G.H), F32).astype(BF16)

    ref, ref_gdn = segment_reference(G, ex, x, hc, w_l, st_l, attn, o_proj, L=L)

    weights = {
        "o_proj": o_proj[:, None],
        "lin_in": w_l["lin_in"],
        "lin_out": w_l["lin_out"],
        "router": jnp.broadcast_to(w_l["router"][None], (R, *w_l["router"].shape)),
        "sh13": w_l["sh13"],
        "sh2": w_l["sh2"],
        "e13": tuple(w_l["e13"][:, l] for l in range(L)),
        "e2": tuple(w_l["e2"][:, l] for l in range(L)),
        "es": jnp.concatenate([w_l["e13_s"], w_l["e2_s"]], axis=-1)[:, :, :, None, :],
        **_hc_weights("qsa_out", hc["qsa_out"], R, mix=False),
        **_hc_weights("gdn", hc["gdn"], R, mix=True),
        **_hc_weights("moe", hc["moe"], R, mix=True),
    }
    gdn_rows = tuple(range(1, L))
    smalls: dict[str, Any] = {
        "qsa_out_hcn": _rows(jnp.broadcast_to(hc["qsa_out"]["norm"][None], (R, 1, G.NHC * G.H)), (0,), L),
        "gdn_hcn": _rows(jnp.broadcast_to(hc["gdn"]["norm"][None], (R, L - 1, G.NHC * G.H)), gdn_rows, L),
        "moe_hcn": hc["moe"]["norm"],
        "in_s": _rows(w_l["lin_in_s"], gdn_rows, L),
        "out_s": _rows(w_l["lin_out_s"], gdn_rows, L),
        **{k: _rows(w_l[k], gdn_rows, L) for k in ("conv", "alog", "dtb", "gnorm")},
        **{k: w_l[k] for k in ("post_norm", "sh_gate", "sh13_s", "sh2_s", "ecopy") if k in w_l},
    }
    lay = qwen35.GDNBlock.lin_in_layout(gi, ex)
    hr = rup(3 + T, 8)
    states = {
        "conv": jnp.pad(st_l["conv"], ((0, 0), (0, 0), (0, 0), (0, hr - 3), (0, 0))).astype(F32)
        if T > 1 else None,
        "rec": st_l["rec"].reshape(R, L - 1, ex.NS * lay.vh, lay.KD, lay.VD).astype(BF16),
    }
    if T == 1:
        cr = rup(3 * B, 8)
        c3 = jnp.swapaxes(st_l["conv"], 2, 3).reshape(R, L - 1, 3 * B, lay.QKV)
        states["conv"] = jnp.pad(c3, ((0, 0), (0, 0), (0, cr - 3 * B), (0, 0))).astype(F32)

    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    w_mk = pack_weights(spec, ex, weights, smalls)
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, pos, x, w_mk, states)
    attn_mk = jnp.pad(attn, ((0, 0), (0, ex.M - B), (0, 0))).reshape(R * ex.M, K)
    attn_mk = jax.device_put(attn_mk, NamedSharding(mesh, P("tp")))
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, attn_mk)
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32)
    return res, np.asarray(ref, np.float32), np.asarray(x, np.float32)


def _assert_within_budget(res: np.ndarray, ref: np.ndarray) -> None:
    """Eight BF16-rounded sublayers with BF16 all-reduce payloads: bound the error
    distribution (mean, 99.9th percentile, max) rather than every element."""
    scale = float(np.abs(ref).mean())
    for r in range(res.shape[0]):
        d = np.abs(res[r] - ref)
        assert d.mean() < 1e-2 * scale, (r, d.mean(), scale)
        assert np.quantile(d, 0.999) < 5e-2 * scale, (r, np.quantile(d, 0.999), scale)
        assert d.max() < 1e-1 * scale, (r, d.max(), scale)


@pytest.mark.parametrize("T", [1, 4])
def test_qwen38_segment_kernel_matches_reference(T: int):
    res, ref, x = _run(T)
    # Every sublayer must move the residual, or the comparison is vacuous.
    assert float(np.std(ref - x)) > 0.05
    _assert_within_budget(res, ref)


def test_qwen38_segment_budget_catches_a_dropped_layer(monkeypatch):
    """Positive control: the budget rejects a kernel whose last MoE sublayer is skipped."""
    from vllm_torchtpu.megakernel.blocks.hc import HCSublayer

    orig = HCSublayer._combine

    def drop_last_moe(self, ctx, block):
        if self.name != "moe":
            return orig(self, ctx, block)
        import jax.numpy as jnp
        from jax import lax

        keep = ctx.stream.sp.block_ord(ctx._hc_layer, "moe") < qwen38.SEGMENT_LAYERS - 1
        return lax.cond(keep, lambda: orig(self, ctx, block), lambda: None)

    orig_emit = HCSublayer.emit

    def emit(self, ctx, l, slot):
        ctx._hc_layer = l
        return orig_emit(self, ctx, l, slot)

    monkeypatch.setattr(HCSublayer, "_combine", drop_last_moe)
    monkeypatch.setattr(HCSublayer, "emit", emit)
    res, ref, _ = _run(1)
    with pytest.raises(AssertionError):
        _assert_within_budget(res, ref)
