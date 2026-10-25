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
"""Qwen3.8-shaped (`KH=16 < R=32`, `IS=2048 < 128*R`) hybrid composition at 32 ranks."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import pytest
import torch
from jax.sharding import Mesh

from vllm_torchtpu.megakernel import kernel, planner
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.model_def import find_model, get_model
from vllm_torchtpu.megakernel.models import (
    qwen35,
    qwen35_profile,
    qwen38,
    qwen38_profile,
)
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES
from vllm_torchtpu.megakernel.ops import F32, I32
from vllm_torchtpu.megakernel_runtime import weight_collector

R = 32
TOL = 3e-2
CTX = 2048
POS0 = 1800
_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_CHILD_TIMEOUT_S = 540

QWEN38_FULL_HF_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 8192,
        "num_hidden_layers": 92,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 23,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 128,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 2048,
        "shared_expert_intermediate_size": 2048,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "rope_parameters": {
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
        },
    },
}


def _tiny_qwen38_geometry() -> qwen35.Qwen35Geometry:
    """Tiny Qwen3.8-shaped geometry (`KH=16 < R=32`, `IS=2048 < 128*R`) at R=32."""
    return qwen35.Qwen35Geometry(
        L=4,
        att_every=4,
        H=1024,
        NH=32,
        NKV=4,
        HD=128,
        ROT=32,
        KH=16,
        VH=32,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=2048,
        moe_tp=False,
    )


def _tiny_qwen38_r8_geometry() -> qwen35.Qwen35Geometry:
    """8-rank geometry with identical replication factors (`kh_rep=2`, `sh_rep=2`)."""
    return qwen35.Qwen35Geometry(
        L=4,
        att_every=4,
        H=1024,
        NH=8,
        NKV=2,
        HD=128,
        ROT=32,
        KH=4,
        VH=8,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=512,
        moe_tp=False,
    )


def _rms(x) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.square(x.astype(F32)))))


def _report(name: str, got, exp) -> None:
    err = jnp.abs(got.astype(F32) - exp.astype(F32))
    print(
        f"{name}: max_err={float(err.max()):.3e} rms_err={_rms(err):.3e} "
        f"rms_ref={_rms(exp):.3e}"
    )


def _check(name: str, got, exp, controls) -> None:
    _report(name, got, exp)
    assert jnp.allclose(got, exp, atol=TOL, rtol=TOL), name
    for label, bad in controls:
        _report(f"{name} vs {label}", bad, exp)
        assert not jnp.allclose(bad, exp, atol=TOL, rtol=TOL), f"{name}: {label}"


def _swap(x, a: int, b: int):
    return x.at[a].set(x[b]).at[b].set(x[a])


def test_qwen38_full_geometry_and_shared_expert_replication_at_r32():
    """Verify full Qwen3.8 geometry (`L=92, H=8192, KH=16, IS=2048`) plans at `R=32`."""
    assert find_model(QWEN38_FULL_HF_CONFIG) is qwen38.Qwen38
    assert find_model(qwen38_profile.DEFAULT_HF_CONFIG) is qwen38.Qwen38
    assert find_model(qwen35_profile.DEFAULT_HF_CONFIG) is qwen35.Qwen35
    assert get_model("qwen3_8") is qwen38.Qwen38
    assert qwen38.Qwen38.default_profile().name == "qwen38-baseline"
    assert qwen38.Qwen38.get_profile("tp32-dcp8").name == "qwen38-tp32-dcp8"

    g = qwen38.Qwen38.geometry(QWEN38_FULL_HF_CONFIG, moe_tp=False)
    ex = ExecSpec(B=4, T=1, R=R, ctx=CTX)
    g.check(ex)
    assert g.kh_rep(R) == 2
    assert g.kh(R) == 1
    assert g.vh(R) == 4
    assert g.sh_rep(R) == 2
    assert g.ISR(R) == 128
    assert g.ER(R) == 16

    ms = g.to_model_spec(R=R)
    assert ms.name == "qwen3_8"
    assert ms.layer_type.name == "qwen38_layer"
    p = planner.plan(ms, ex, fit_vmem=False)
    assert p.module_params["gdn"].kh == 1
    assert p.module_params["gdn"].vh == 4
    assert p.module_params["moe"].ISR == 128
    assert p.module_params["moe"].sh_rep == 2
    assert ms.cache_backend is not None
    play = ms.cache_backend.layout
    assert play.kh_per_rank(R) == 1
    assert play.vh_per_rank(R) == 4
    assert play.qkv_per_rank(R) == 768
    assert play.rec_pages_per_slot == 4
    play.validate(ex)

    # Verify pack_router_and_shared_layer slices by sh_rank = tp_rank // 2.
    class _FakeMLP:
        pass

    mlp = _FakeMLP()
    g_tiny = _tiny_qwen38_geometry()
    ex_tiny = ExecSpec(B=2, T=1, R=R, ctx=CTX)
    p_tiny = planner.plan(g_tiny.to_model_spec(), ex_tiny, fit_vmem=False)
    torch.manual_seed(0)
    mlp._fused_gate_gu_fp8_weight = torch.nn.Parameter(
        torch.randn(g_tiny.H, g_tiny.E + 2 * g_tiny.IS).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    mlp._fused_gate_gu_fp8_scale = torch.nn.Parameter(
        torch.rand(g_tiny.E + 2 * g_tiny.IS, dtype=torch.float32) * 0.02 + 0.01,
        requires_grad=False,
    )
    mlp._fused_se_down_fp8_weight = torch.nn.Parameter(
        torch.randn(g_tiny.IS, g_tiny.H).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    mlp._fused_se_down_fp8_scale = torch.nn.Parameter(
        torch.rand(g_tiny.H, dtype=torch.float32) * 0.02 + 0.01,
        requires_grad=False,
    )
    mlp._fused_se_gate_weight = torch.nn.Parameter(
        torch.randn(g_tiny.H, dtype=torch.float32),
        requires_grad=False,
    )
    rs0 = weight_collector.pack_router_and_shared_layer(
        mlp,
        g_tiny,
        ex_tiny,
        tp_rank=0,
        tk_router=p_tiny.tk["router"],
        tk_sh13=p_tiny.tk["sh13"],
        tk_sh2=p_tiny.tk["sh2"],
    )
    rs1 = weight_collector.pack_router_and_shared_layer(
        mlp,
        g_tiny,
        ex_tiny,
        tp_rank=1,
        tk_router=p_tiny.tk["router"],
        tk_sh13=p_tiny.tk["sh13"],
        tk_sh2=p_tiny.tk["sh2"],
    )
    rs2 = weight_collector.pack_router_and_shared_layer(
        mlp,
        g_tiny,
        ex_tiny,
        tp_rank=2,
        tk_router=p_tiny.tk["router"],
        tk_sh13=p_tiny.tk["sh13"],
        tk_sh2=p_tiny.tk["sh2"],
    )
    assert torch.equal(rs0["sh13"].view(torch.uint8), rs1["sh13"].view(torch.uint8))
    assert torch.equal(rs0["sh2"].view(torch.uint8), rs1["sh2"].view(torch.uint8))
    assert not torch.equal(rs0["sh13"].view(torch.uint8), rs2["sh13"].view(torch.uint8))

    # Also verify 32-rank weight splitting, packing, and reference step on tiny Qwen3.8.
    k_in, k_w = jax.random.split(jax.random.PRNGKey(3800))
    _, hid_l, _, st_l = qwen35.random_logical_inputs(k_in, g_tiny, ex_tiny)
    w_l = qwen35.split_hf_weights(
        qwen35.random_hf_weights(k_w, g_tiny), g_tiny, ex_tiny
    )
    pos = POS0 + jnp.arange(ex_tiny.NS, dtype=I32)
    qk_cols = 2 * g_tiny.kh(R) * g_tiny.KD
    assert jnp.array_equal(
        w_l["lin_in"][0, :, :, :qk_cols], w_l["lin_in"][1, :, :, :qk_cols]
    )
    assert not jnp.array_equal(
        w_l["lin_in"][0, :, :, :qk_cols], w_l["lin_in"][2, :, :, :qk_cols]
    )
    assert jnp.array_equal(w_l["sh13"][0], w_l["sh13"][1])
    assert not jnp.array_equal(w_l["sh13"][0], w_l["sh13"][2])
    ref_hid, ref_st = qwen35.reference_step(g_tiny, ex_tiny, pos, hid_l, w_l, st_l)
    _, _, w_mk, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g_tiny, ex_tiny, mesh=None
    )
    assert jnp.array_equal(w_mk["sh13"][0], w_mk["sh13"][1])
    assert not jnp.array_equal(w_mk["sh13"][0], w_mk["sh13"][2])
    conv_swap_01_qk = (
        ref_st["conv"]
        .at[:, :, :, :qk_cols]
        .set(_swap(ref_st["conv"][:, :, :, :qk_cols], 0, 1))
    )
    assert jnp.array_equal(
        conv_swap_01_qk[:, :, : 3 * ex_tiny.B, :qk_cols],
        ref_st["conv"][:, :, : 3 * ex_tiny.B, :qk_cols],
    )
    assert not jnp.allclose(
        _swap(ref_st["conv"], 0, 2), ref_st["conv"], atol=TOL, rtol=TOL
    )
    assert not jnp.allclose(hid_l, ref_hid, atol=TOL, rtol=TOL)


def test_qwen38_sub_rank_replication_step_in_process():
    """Run `make_step` on 8 CPU devices with `kh_rep=2` and `sh_rep=2`."""
    r8 = 8
    devs = jax.devices("cpu")
    assert len(devs) >= r8, f"Requires {r8} CPU devices, found {len(devs)}"
    g = _tiny_qwen38_r8_geometry()
    assert g.kh_rep(r8) == 2 and g.sh_rep(r8) == 2
    tiles = dataclasses.replace(
        TINY_TILES, bank=256 << 10, hier=3, mb8=2, inplace=False
    ).with_modules({"gqa": {"split": True}})
    ex = ExecSpec(B=2, T=1, R=r8, ctx=1024, tiles=tiles, interpret=True)

    k_in, k_w = jax.random.split(jax.random.PRNGKey(3808))
    _, hid_l, _, st_l = qwen35.random_logical_inputs(k_in, g, ex)
    w_l = qwen35.split_hf_weights(qwen35.random_hf_weights(k_w, g), g, ex)
    pos = 900 + jnp.arange(ex.NS, dtype=I32)
    ref_hid, ref_st = qwen35.reference_step(g, ex, pos, hid_l, w_l, st_l)

    mesh = Mesh(devs[:r8], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh
    )
    step, _ = qwen35.make_step(mesh, g, ex, donate_states=True)
    st_in = {
        k: jax.device_get(kernel.unfold(st_mk[k], r8))
        for k in ("conv", "rec", "kc", "vc")
    }
    out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)

    hid_r = kernel.unfold(out_hid, r8)[:, : ex.B]
    for r in range(1, r8):
        assert jnp.array_equal(hid_r[r], hid_r[0]), f"hidden differs on rank {r}"
    _check("hidden", hid_r[0], ref_hid, [("input hidden", hid_l)])

    qk_cols = 2 * g.kh(r8) * g.KD
    conv_got = kernel.unfold(out_st["conv"], r8)
    conv_swap_01_qk = conv_got.at[:, :, :, :qk_cols].set(
        _swap(conv_got[:, :, :, :qk_cols], 0, 1)
    )
    assert jnp.array_equal(
        conv_swap_01_qk[:, :, : 3 * ex.B, :qk_cols],
        conv_got[:, :, : 3 * ex.B, :qk_cols],
    )
    _check(
        "conv",
        conv_got,
        ref_st["conv"],
        [
            ("ranks 0 and 2 swapped", _swap(conv_got, 0, 2)),
            ("input state", st_in["conv"]),
        ],
    )
    for k, a, b in (("rec", 0, 1), ("kc", 0, 4), ("vc", 0, 4)):
        got = kernel.unfold(out_st[k], r8)
        _check(
            k,
            got,
            ref_st[k],
            [
                (f"ranks {a} and {b} swapped", _swap(got, a, b)),
                ("input state", st_in[k]),
            ],
        )


def _step_qwen38_32_ranks_in_process() -> None:
    devs = jax.devices("cpu")
    assert len(devs) >= R, f"Requires {R} CPU devices, found {len(devs)}"
    g = _tiny_qwen38_geometry()
    tiles = dataclasses.replace(
        TINY_TILES, bank=256 << 10, hier=3, mb8=2, inplace=False
    ).with_modules({"gqa": {"split": True}})
    ex = ExecSpec(B=2, T=1, R=R, ctx=256, tiles=tiles, interpret=True)

    k_in, k_w = jax.random.split(jax.random.PRNGKey(3801))
    _, hid_l, _, st_l = qwen35.random_logical_inputs(k_in, g, ex)
    w_l = qwen35.split_hf_weights(qwen35.random_hf_weights(k_w, g), g, ex)
    pos = 100 + jnp.arange(ex.NS, dtype=I32)

    # Ranks 0 and 1 share key/query head 0 and shared-expert slice 0; rank 2 does not.
    qk_cols = 2 * g.kh(R) * g.KD
    assert jnp.array_equal(
        w_l["lin_in"][0, :, :, :qk_cols], w_l["lin_in"][1, :, :, :qk_cols]
    )
    assert not jnp.array_equal(
        w_l["lin_in"][0, :, :, :qk_cols], w_l["lin_in"][2, :, :, :qk_cols]
    )
    assert jnp.array_equal(w_l["sh13"][0], w_l["sh13"][1])
    assert not jnp.array_equal(w_l["sh13"][0], w_l["sh13"][2])

    ref_hid, ref_st = qwen35.reference_step(g, ex, pos, hid_l, w_l, st_l)

    mesh = Mesh(devs[:R], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh
    )
    step, _ = qwen35.make_step(mesh, g, ex, donate_states=True)
    st_in = {
        k: jax.device_get(kernel.unfold(st_mk[k], R))
        for k in ("conv", "rec", "kc", "vc")
    }
    out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)

    hid_r = kernel.unfold(out_hid, R)[:, : ex.B]
    for r in range(1, R):
        assert jnp.array_equal(hid_r[r], hid_r[0]), f"hidden differs on rank {r}"
    _check("hidden", hid_r[0], ref_hid, [("input hidden", hid_l)])

    conv_got = kernel.unfold(out_st["conv"], R)
    # Positive control: swapping Q/K conv state between replica ranks 0 and 1 is exact.
    conv_swap_01_qk = conv_got.at[:, :, :, :qk_cols].set(
        _swap(conv_got[:, :, :, :qk_cols], 0, 1)
    )
    assert jnp.array_equal(
        conv_swap_01_qk[:, :, : 3 * ex.B, :qk_cols],
        conv_got[:, :, : 3 * ex.B, :qk_cols],
    )
    # Negative control: swapping Q/K conv state between ranks 0 and 2 fails.
    conv_swap_02 = _swap(conv_got, 0, 2)
    _check(
        "conv",
        conv_got,
        ref_st["conv"],
        [("ranks 0 and 2 swapped", conv_swap_02), ("input state", st_in["conv"])],
    )

    for k, a, b in (("rec", 0, 1), ("kc", 0, 8), ("vc", 0, 8)):
        got = kernel.unfold(out_st[k], R)
        _check(
            k,
            got,
            ref_st[k],
            [
                (f"ranks {a} and {b} swapped", _swap(got, a, b)),
                ("input state", st_in[k]),
            ],
        )
    print("32-rank step completed successfully with L=4, att_every=4, CTX=256")


@pytest.mark.slow
def test_qwen38_decode_with_32_ranks():
    flags = [
        f
        for f in os.environ.get("XLA_FLAGS", "").split()
        if not f.startswith(_DEVICE_COUNT_FLAG)
    ]
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        MALLOC_ARENA_MAX="2",
        XLA_FLAGS=" ".join([*flags, f"{_DEVICE_COUNT_FLAG}={R}"]),
    )
    t0 = time.perf_counter()
    out = subprocess.run(
        [sys.executable or "python3", __file__, "qwen38_r32"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
    )
    child_time = time.perf_counter() - t0
    print(f"test_qwen38_decode_with_32_ranks child wall time: {child_time:.2f}s")
    print(out.stdout)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]


def test_moe_block_with_nvfp4_routed_experts_in_process():
    """Verify MoEModule expert_quant='nvfp4' numerical dequant and step parity."""
    from dataclasses import replace as dc_replace

    from vllm_torchtpu.megakernel.modules.moe import MoEModule
    from vllm_torchtpu.megakernel.modules.moe.experts import unpack_nvfp4_tile
    from vllm_torchtpu.megakernel.modules.moe.reference import dequant_nvfp4_weight

    mb_default = MoEModule(prefix="mlp")
    mb_bound = MoEModule.bind(prefix="mlp")
    assert mb_default.options == mb_bound.options
    mb_nv = MoEModule(prefix="mlp", expert_quant="nvfp4")
    assert mb_nv.options["expert_quant"] == "nvfp4"
    g_from_hf = qwen38.Qwen38.derive_geometry(
        {**QWEN38_FULL_HF_CONFIG, "quantization_config": {"quant_algo": "NVFP4"}}
    )
    assert g_from_hf.expert_quant == "nvfp4"
    with pytest.raises(ValueError, match="unpack_nvfp4_tile requires"):
        unpack_nvfp4_tile(jnp.zeros((4, 256), dtype=jnp.uint8), 128)

    g_base = dc_replace(_tiny_qwen38_geometry(), I=256)  # e2 tile K = I // 2 >= 128
    ex8 = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        tiles=dc_replace(TINY_TILES, bank=256 << 10, hier=0, mb8=1, inplace=False),
        interpret=True,
    )
    g_nv = dc_replace(g_base, expert_quant="nvfp4")
    _, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.PRNGKey(91), g_nv, ex8
    )
    assert w_l["e13"].dtype == jnp.uint8
    n_log = 2 * g_nv.IR(ex8.R)
    w_tile = w_l["e13"][0, 0, 0]
    unp_got = unpack_nvfp4_tile(w_tile, n_log, interpret=True).astype(F32)
    deq_ref = dequant_nvfp4_weight(w_tile, n_log)
    w_bad = w_tile.at[:, n_log:].set(jnp.uint8(0))
    deq_bad = dequant_nvfp4_weight(w_bad, n_log)
    _check(
        "nvfp4 unpack vs dequant_ref",
        unp_got,
        deq_ref,
        [("zeroed block scale", deq_bad)],
    )

    w_f32_eq = dict(w_l)
    w_f32_eq["e13"] = jax.vmap(
        jax.vmap(jax.vmap(lambda t: dequant_nvfp4_weight(t, n_log)))
    )(w_l["e13"])
    w_f32_eq["e2"] = jax.vmap(
        jax.vmap(jax.vmap(lambda t: dequant_nvfp4_weight(t, g_nv.H)))
    )(w_l["e2"])
    pos = POS0 + jnp.arange(ex8.NS, dtype=I32)
    ref_hid_nv, _ = qwen35.reference_step(g_nv, ex8, pos, hid_l, w_l, st_l)
    assert ref_hid_nv.shape == (ex8.B, g_nv.H)
    ref_hid_f32, _ = qwen35.reference_step(g_base, ex8, pos, hid_l, w_f32_eq, st_l)
    w_bad_s = dict(w_l, e13_s=jnp.zeros_like(w_l["e13_s"]))
    ref_hid_bad, _ = qwen35.reference_step(g_nv, ex8, pos, hid_l, w_bad_s, st_l)
    _check(
        "nvfp4 reference_step vs f32 dequant control",
        ref_hid_nv,
        ref_hid_f32,
        [("zeroed e13_s", ref_hid_bad), ("input hidden", hid_l)],
    )
    devs = jax.devices("cpu")
    if len(devs) >= ex8.R:
        mesh = Mesh(devs[: ex8.R], ("tp",))
        pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
            pos, hid_l, w_l, st_l, g_nv, ex8, mesh=mesh
        )
        step_nv, _ = qwen35.make_step(mesh, g_nv, ex8, donate_states=True)
        out_hid_nv, _ = step_nv(pos_mk, hid_mk, w_mk, st_mk)
        hid_nv_r = kernel.unfold(out_hid_nv, ex8.R)[:, : ex8.B]
        _check(
            "nvfp4 make_step vs reference_step",
            hid_nv_r[0],
            ref_hid_nv,
            [("zeroed e13_s", ref_hid_bad), ("input hidden", hid_l)],
        )


def test_moe_nvfp4_rolled_multi_tile_interpret():
    """G1: Multi-tile rolled NVFP4 MoE (T13=2, tk_raw=128) in interpret mode."""
    import types
    from dataclasses import replace as dc_replace

    from vllm_torchtpu.megakernel import planner
    from vllm_torchtpu.megakernel.modules.moe import experts as moe_experts
    from vllm_torchtpu.megakernel.modules.moe.experts import _mm_etake_w
    from vllm_torchtpu.megakernel.modules.moe.reference import dequant_nvfp4_weight
    from vllm_torchtpu.megakernel.ops import BF16

    class _Ref:
        def __init__(self, arr):
            self.arr = arr

        @staticmethod
        def _idx(k):
            if isinstance(k, tuple):
                return tuple(_Ref._idx(i) for i in k)
            if hasattr(k, "start") and hasattr(k, "size") and not isinstance(k, slice):
                return slice(int(k.start), int(k.start) + int(k.size))
            return k

        def __getitem__(self, k):
            return self.arr[self._idx(k)]

        def __setitem__(self, k, v):
            self.arr = self.arr.at[self._idx(k)].set(v)

    g_rolled = dc_replace(
        _tiny_qwen38_geometry(),
        H=512,
        I=1024,
        moe_tp=False,
        expert_quant="nvfp4",
    )
    ex8 = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        tiles=dc_replace(
            TINY_TILES, bank=256 << 10, ebank=320 << 10, hier=0, mb8=1, inplace=False
        ),
        interpret=True,
    )
    mspec = qwen35.qwen35_spec(g_rolled, R=ex8.R)
    p = planner.plan(mspec, ex8)
    assert p.nt["e13"] == 2 and p.tk["e13"] == 128
    assert not p.module_params["moe"].unroll

    _, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.PRNGKey(191), g_rolled, ex8
    )

    # Direct check of _mm_etake_w 64-row loop (tk_raw=128 > 64) and single-slice
    # branch (tk_raw=64) in interpret mode (fails before D2 when interpret= is
    # not forwarded to unpack_nvfp4_tile).
    n_log = 2 * g_rolled.IR(ex8.R)
    w_tile_128 = w_l["e13"][0, 0, 0, :128]
    x_act = jax.random.normal(jax.random.PRNGKey(192), (ex8.MB, 256), dtype=BF16)
    acc_128 = _Ref(jnp.zeros((ex8.MB, n_log), dtype=F32))
    acc_64 = _Ref(jnp.zeros((ex8.MB, n_log), dtype=F32))
    m_stub = types.SimpleNamespace(
        mp=types.SimpleNamespace(expert_quant="nvfp4"),
        ex=ex8,
        scr={},
        stream=types.SimpleNamespace(
            mm=lambda a, b: jnp.matmul(a.astype(F32), b.astype(F32)),
        ),
    )
    orig_etake = moe_experts.etake
    moe_experts.etake = lambda m, kind, u: _Ref(
        w_tile_128 if u == 0 else w_tile_128[:64]
    )
    try:
        with jax.disable_jit():
            _mm_etake_w(m_stub, "e13", 0, _Ref(x_act), 0, 256, n_log, acc_128)
            _mm_etake_w(m_stub, "e13", 1, _Ref(x_act), 0, 128, n_log, acc_64)
    finally:
        moe_experts.etake = orig_etake
    ref_128 = jnp.matmul(x_act.astype(F32), dequant_nvfp4_weight(w_tile_128, n_log))
    ref_64 = jnp.matmul(
        x_act[:, :128].astype(F32), dequant_nvfp4_weight(w_tile_128[:64], n_log)
    )
    _check(
        "_mm_etake_w tk_raw=128 vs dequant_ref",
        acc_128.arr,
        ref_128,
        [("half-tile control", ref_64)],
    )
    _check(
        "_mm_etake_w tk_raw=64 vs dequant_ref",
        acc_64.arr,
        ref_64,
        [("full-tile control", ref_128)],
    )

    pos = POS0 + jnp.arange(ex8.NS, dtype=I32)
    ref_hid_nv, _ = qwen35.reference_step(g_rolled, ex8, pos, hid_l, w_l, st_l)
    w_bad_s = dict(w_l, e13_s=jnp.zeros_like(w_l["e13_s"]))
    ref_hid_bad, _ = qwen35.reference_step(g_rolled, ex8, pos, hid_l, w_bad_s, st_l)

    devs = jax.devices("cpu")
    assert len(devs) >= ex8.R
    mesh = Mesh(devs[: ex8.R], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g_rolled, ex8, mesh=mesh
    )
    step_nv, _ = qwen35.make_step(mesh, g_rolled, ex8, donate_states=True)
    out_hid_nv, _ = step_nv(pos_mk, hid_mk, w_mk, st_mk)
    hid_nv_r = kernel.unfold(out_hid_nv, ex8.R)[:, : ex8.B]
    _check(
        "nvfp4 rolled multi-tile make_step vs reference_step",
        hid_nv_r[0],
        ref_hid_nv,
        [("zeroed e13_s", ref_hid_bad), ("input hidden", hid_l)],
    )


def test_pack_nvfp4_rejects_stock_prepared_layout():
    """G2: pack_expert_views_layer rejects stock requant_unpack_kmajor layout."""
    import importlib.util
    import os
    from dataclasses import replace as dc_replace
    from pathlib import Path

    import numpy as np
    import torch

    spec = importlib.util.find_spec("libtpu")
    if spec and spec.submodule_search_locations:
        os.environ.setdefault(
            "TPU_LIBRARY_PATH",
            os.path.join(spec.submodule_search_locations[0], "libtpu.so"),
        )
    import vllm_torchtpu

    gmm_path = (
        Path(vllm_torchtpu.__file__).resolve().parent
        / "layers"
        / "core"
        / "fused_moe_gmm.py"
    )
    gmm_spec = importlib.util.spec_from_file_location(
        "vllm_torchtpu_fused_moe_gmm", gmm_path
    )
    assert gmm_spec is not None and gmm_spec.loader is not None
    gmm_mod = importlib.util.module_from_spec(gmm_spec)
    gmm_spec.loader.exec_module(gmm_mod)
    requant_unpack_kmajor = gmm_mod.requant_unpack_kmajor
    from vllm_torchtpu.megakernel.errors import MegakernelConfigError
    from vllm_torchtpu.megakernel_runtime.weight_collector import (
        pack_expert_views_layer,
    )

    g_nv = dc_replace(
        _tiny_qwen38_geometry(), H=256, I=256, moe_tp=False, expert_quant="nvfp4"
    )
    ex8 = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        tiles=dc_replace(TINY_TILES, bank=256 << 10, hier=0, mb8=1, inplace=False),
        interpret=True,
    )
    ew = g_nv.EW(ex8.R)  # 2
    ir = g_nv.IR(ex8.R)  # 256
    H = g_nv.H  # 256

    rng = np.random.default_rng(77)
    w13_raw = rng.integers(0, 256, size=(ew, 2 * ir, H // 2), dtype=np.uint8)
    s13_raw = rng.integers(1, 60, size=(ew, 2 * ir, H // 16), dtype=np.uint8)
    w2_raw = rng.integers(0, 256, size=(ew, H, ir // 2), dtype=np.uint8)
    s2_raw = rng.integers(1, 60, size=(ew, H, ir // 16), dtype=np.uint8)

    w13_stock, s13_stock = requant_unpack_kmajor(
        jnp.asarray(w13_raw),
        jnp.asarray(s13_raw).view(jnp.float8_e4m3fn).astype(jnp.float32),
        256,
    )
    assert tuple(w13_stock.shape) == (ew, H, 2 * ir)
    assert tuple(s13_stock.shape) == (ew, H // 256, 1, 2 * ir)

    class _Experts:
        pass

    class _MLP:
        pass

    # 1. Stock-prepared tensor shapes (uint8 [ew, H, ir], 4D f32 scale)
    exp_stock = _Experts()
    exp_stock.w13_weight = torch.zeros((ew, H, ir), dtype=torch.uint8)
    exp_stock.w13_weight_scale = torch.from_numpy(np.asarray(s13_stock).copy())
    exp_stock.w13_weight_scale_2 = None
    exp_stock.w2_weight = torch.from_numpy(w2_raw)
    exp_stock.w2_weight_scale = torch.from_numpy(s2_raw)
    exp_stock.w2_weight_scale_2 = None
    mlp_stock = _MLP()
    mlp_stock.experts = exp_stock

    with pytest.raises(
        MegakernelConfigError,
        match="stock process_weights_after_loading already requantized; see unit B",
    ):
        pack_expert_views_layer(mlp_stock, g_nv, ex8, tk_e13=128, tk_e2=128)

    # 2. Raw weights/scales present, but w13_weight_scale_2 deleted (None)
    exp_no_gscale = _Experts()
    exp_no_gscale.w13_weight = torch.from_numpy(w13_raw)
    exp_no_gscale.w13_weight_scale = torch.from_numpy(s13_raw)
    exp_no_gscale.w13_weight_scale_2 = None
    exp_no_gscale.w2_weight = torch.from_numpy(w2_raw)
    exp_no_gscale.w2_weight_scale = torch.from_numpy(s2_raw)
    exp_no_gscale.w2_weight_scale_2 = None
    mlp_no_gscale = _MLP()
    mlp_no_gscale.experts = exp_no_gscale

    with pytest.raises(
        MegakernelConfigError,
        match="stock process_weights_after_loading already requantized; see unit B",
    ):
        pack_expert_views_layer(mlp_no_gscale, g_nv, ex8, tk_e13=128, tk_e2=128)


if __name__ == "__main__":
    if sys.argv[1] == "qwen38_r32":
        _step_qwen38_32_ranks_in_process()


def test_replicated_reference_matches_unreplicated_reference():
    """kh_rep = sh_rep = 2 at R=8 keeps the R=4 (unreplicated) layer output."""
    g = _tiny_qwen38_r8_geometry()
    assert (g.kh_rep(8), g.sh_rep(8)) == (2, 2)
    assert (g.kh_rep(4), g.sh_rep(4)) == (1, 1)
    hf = qwen35.random_hf_weights(jax.random.PRNGKey(3808), g)
    tiles = dataclasses.replace(
        TINY_TILES, bank=256 << 10, hier=0, mb8=2, inplace=False
    )
    delta, w_by_r, args_by_r = {}, {}, {}
    for r in (4, 8):
        ex = ExecSpec(B=2, T=1, R=r, ctx=1024, tiles=tiles, interpret=True)
        _, hid, _, st = qwen35.random_logical_inputs(jax.random.PRNGKey(7), g, ex)
        st0 = jax.tree.map(jnp.zeros_like, st)
        w = qwen35.split_hf_weights(hf, g, ex)
        pos = 900 + jnp.arange(ex.NS, dtype=I32)
        out, _ = qwen35.reference_step(g, ex, pos, hid, w, st0)
        delta[r] = out.astype(F32) - hid.astype(F32)
        w_by_r[r], args_by_r[r] = w, (g, ex, pos, hid)
    # Two arms x 2 all-reduces per layer x L layers of bf16 partial rounding.
    n_ar = 2 * g.L
    tol = 2 * n_ar * 2.0**-8 * float(jnp.max(jnp.abs(delta[4])))
    _report("replicated R=8 vs unreplicated R=4", delta[8], delta[4])
    assert float(jnp.max(jnp.abs(delta[8] - delta[4]))) <= tol
    # Negative control: drop the 1/sh_rep compensation (shared expert counted twice).
    g8, ex8, pos8, hid8 = args_by_r[8]
    w_bad = dict(w_by_r[8], sh2_s=w_by_r[8]["sh2_s"] * 2.0)
    st8 = jax.tree.map(
        jnp.zeros_like, qwen35.random_logical_inputs(jax.random.PRNGKey(7), g8, ex8)[3]
    )
    out_bad, _ = qwen35.reference_step(g8, ex8, pos8, hid8, w_bad, st8)
    d_bad = out_bad.astype(F32) - hid8.astype(F32)
    assert float(jnp.max(jnp.abs(d_bad - delta[4]))) > tol


def test_unpack_nvfp4_tile_fp8_matches_bf16():
    """Normalized FP8 tile unpack avoids saturation on large e4m3 scales."""
    import numpy as np
    from jax import lax

    from vllm_torchtpu.megakernel.modules.moe.experts import (
        unpack_nvfp4_tile,
        unpack_nvfp4_tile_fp8,
    )

    rng = np.random.default_rng(123)
    k_logical, n_logical = 128, 64
    k_half = k_logical // 2
    w_u8 = jnp.asarray(rng.integers(0, 256, size=(k_half, n_logical), dtype=np.uint8))
    # Include large e4m3 block scales where unnormalized w4 * bscale > 448 saturates.
    s_f32 = jnp.asarray(
        rng.uniform(0.5, 240.0, size=(k_logical // 16, n_logical)), dtype=jnp.float32
    )
    s_u8 = lax.bitcast_convert_type(s_f32.astype(jnp.float8_e4m3fn), jnp.uint8).reshape(
        k_half, n_logical // 8
    )
    packed_tile = jnp.concatenate([w_u8, s_u8], axis=-1)

    w_bf16 = unpack_nvfp4_tile(packed_tile, n_logical, interpret=True).astype(
        jnp.float32
    )
    w_fp8, w_s = unpack_nvfp4_tile_fp8(packed_tile, n_logical, interpret=True)
    assert w_fp8.dtype == jnp.float8_e4m3fn
    w_recon = w_fp8.astype(jnp.float32) * w_s

    # One e4m3 rounding step (2^-3 = 0.125 relative spacing) on w_bf16 * bscale_norm.
    assert not bool(jnp.any(jnp.isnan(w_recon)))
    rel_err = float(jnp.max(jnp.abs(w_recon - w_bf16)) / jnp.max(jnp.abs(w_bf16)))
    assert rel_err <= 0.125
    # Negative control: unnormalized cast to FP8 overflows/saturates above 448.0.
    w_unnorm = w_bf16.astype(jnp.float8_e4m3fn).astype(jnp.float32)
    w_unnorm_sat = jnp.where(jnp.isnan(w_unnorm), jnp.sign(w_bf16) * 448.0, w_unnorm)
    sat_err = float(jnp.max(jnp.abs(w_unnorm_sat - w_bf16)) / jnp.max(jnp.abs(w_bf16)))
    assert sat_err > 0.125
