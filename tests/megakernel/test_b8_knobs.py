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
"""Regression and contract tests for generic module/kernel generalizations."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.megakernel as mk


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


class _SynthBaseModel(mk.ModelDef):
    """Compact synthetic hybrid `ModelDef` supporting `R=2` and `R=8`."""

    name = "b8_synth_base"
    hf_architectures = ()
    hf_model_type = ()
    supported_tp_sizes = (2, 8)
    default_hf_config = {
        "n_vocab": 256,
        "n_layers": 2,
        "d_model": 256,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_head": 256,
        "rotary_frac": 0.25,
        "theta": 10000.0,
        "norm_eps": 1e-6,
        "ssm_k_heads": 2,
        "ssm_v_heads": 4,
        "ssm_k_dim": 128,
        "ssm_v_dim": 128,
        "ssm_conv_kernel": 4,
        "n_routed_experts": 128,
        "top_k_experts": 2,
        "d_ff_routed": 256,
        "d_ff_shared": 256,
    }
    geometry = mk.FromHF(
        hidden="d_model",
        layers="n_layers",
        vocab="n_vocab",
        eps="norm_eps",
        rope_theta="theta",
        rope_partial=("rotary_frac", 0.25),
    )
    attn = mk.GQA.bind(
        prefix="self_attn",
        heads="n_heads",
        kv_heads="n_kv_heads",
        head_dim="d_head",
        qk_norm=True,
        output_gate=True,
    )
    linear = mk.GDN.bind(
        prefix="linear_attn",
        k_heads="ssm_k_heads",
        v_heads="ssm_v_heads",
        k_dim="ssm_k_dim",
        v_dim="ssm_v_dim",
        conv_kernel="ssm_conv_kernel",
    )
    moe = mk.MoE.bind(
        prefix="mlp",
        experts="n_routed_experts",
        top_k="top_k_experts",
        inter="d_ff_routed",
        shared_inter="d_ff_shared",
        shared_gate=True,
    )
    schedule = mk.Schedule(
        mixer=mk.Every(n=2, last=attn, rest=linear),
        ffn=moe,
        pre_norm=mk.Norm.bind("input_layernorm", zero_centered=True),
        post_norm=mk.Norm.bind("post_attention_layernorm", zero_centered=True),
    )
    embed = mk.Embedding("model.embed_tokens")
    final_norm = mk.Norm.bind("model.norm", zero_centered=True)
    lm_head = mk.LMHead("lm_head")


class _SigmoidGDNModel(_SynthBaseModel):
    """Synthetic hybrid model binding GDN gated-norm activation to `sigmoid`."""

    name = "b8_sigmoid_gdn"
    linear = mk.GDN.bind(
        prefix="linear_attn",
        k_heads="ssm_k_heads",
        v_heads="ssm_v_heads",
        k_dim="ssm_k_dim",
        v_dim="ssm_v_dim",
        conv_kernel="ssm_conv_kernel",
        gate="sigmoid",
    )
    schedule = mk.Schedule(
        mixer=mk.Every(n=2, last=_SynthBaseModel.attn, rest=linear),
        ffn=_SynthBaseModel.moe,
        pre_norm=_SynthBaseModel.schedule.pre_norm,
        post_norm=_SynthBaseModel.schedule.post_norm,
    )


class _HFBoundGateGDNModel(_SynthBaseModel):
    """Synthetic hybrid model reading GDN gate activation from an HF config key."""

    name = "b8_hf_gate_gdn"
    linear = mk.GDN.bind(
        prefix="linear_attn",
        k_heads="ssm_k_heads",
        v_heads="ssm_v_heads",
        k_dim="ssm_k_dim",
        v_dim="ssm_v_dim",
        conv_kernel="ssm_conv_kernel",
        gate=("linear_gate_act", "silu"),
    )
    schedule = mk.Schedule(
        mixer=mk.Every(n=2, last=_SynthBaseModel.attn, rest=linear),
        ffn=_SynthBaseModel.moe,
        pre_norm=_SynthBaseModel.schedule.pre_norm,
        post_norm=_SynthBaseModel.schedule.post_norm,
    )


def test_gdn_gate_binding_and_validation() -> None:
    g_default = _SynthBaseModel.derive_geometry()
    assert g_default.gdn_gate == "silu"

    g_sig = _SigmoidGDNModel.derive_geometry()
    assert g_sig.gdn_gate == "sigmoid"

    g_hf_default = _HFBoundGateGDNModel.derive_geometry()
    assert g_hf_default.gdn_gate == "silu"

    cfg_sig = {**_SynthBaseModel.default_hf_config, "linear_gate_act": "sigmoid"}
    g_hf_sig = _HFBoundGateGDNModel.derive_geometry(cfg_sig)
    assert g_hf_sig.gdn_gate == "sigmoid"
    shrunk_cfg = _HFBoundGateGDNModel.shrink_hf_config(cfg_sig, periods=1, R=2)
    assert shrunk_cfg["linear_gate_act"] == "sigmoid"

    with pytest.raises(mk.MegakernelConfigError, match="gate"):
        mk.GDNModule(gate="relu")

    with pytest.raises(mk.MegakernelConfigError, match="gate"):
        _HFBoundGateGDNModel.derive_geometry(
            {**_SynthBaseModel.default_hf_config, "linear_gate_act": "relu"}
        )


def test_gdn_sigmoid_kernel_matches_reference_all_paths() -> None:
    # B_per_T=8 at R=2, T=1 -> B=16 (WIDE=True -> GCH=True chunked path)
    report = mk.verify_model(
        _SigmoidGDNModel,
        R=2,
        B_per_T=8,
        T_values=(1,),
        hf=False,
        approx_profiles=False,
    )
    report.assert_passed()

    # Unchunked GCH=False (B=4, split PallasWriter) and fused pool I/O path
    R, ctx = 2, 256
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = _SigmoidGDNModel.derive_geometry()
    spec = _SigmoidGDNModel.to_model_spec(g, R=R)
    tiles = mk.Tiles(
        bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32
    ).with_modules(
        {
            "gqa": {"kv_blk": 256},
            "gdn": {"conv_stage": False, "gdn_head_major": True},
        }
    )

    ex = mk.ExecSpec(
        R=R,
        B=4,
        T=1,
        ctx=ctx,
        tiles=tiles,
        interpret=True,
        cache=mk.CacheExec(paged_kv=True, num_pages=128, pool_gather=False),
    )
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(
        jax.random.key(11), g, ex, model_def=_SigmoidGDNModel
    )
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, _ = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    play = _SigmoidGDNModel.pool_layout(g, R=R)
    pt, ckpt, roff, _ = mk.pool.allocate_test_layout(play, ex)
    base_pools = [
        np.asarray(r).copy()
        for r in mk.pool.states_to_pool(
            play, ex, st_l, pt, ckpt[..., :1], jnp.maximum(roff, 0)
        )
    ]
    step_s, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex, donate=False
    )
    step_f, _ = mk.PallasWriter(fused_pool_io=True).make_step(
        mesh, spec, ex, donate=True
    )
    out_s, _ = step_s(
        pos,
        hid_mk,
        w_mk,
        tuple(jax.device_put(r.copy(), tp) for r in base_pools),
        pt,
        ckpt,
        roff,
    )
    out_f, _ = step_f(
        pos,
        hid_mk,
        w_mk,
        tuple(jax.device_put(r.copy(), tp) for r in base_pools),
        pt,
        ckpt,
        roff,
    )
    np.testing.assert_allclose(
        np.asarray(out_f[:4], dtype=np.float32),
        np.asarray(out_s[:4], dtype=np.float32),
        atol=1e-2,
        rtol=1e-2,
    )

    # Confirm sigmoid output matches kernel and differs from silu on identical inputs
    g_silu = _SynthBaseModel.derive_geometry()
    ref_sig, _ = _SigmoidGDNModel.reference_step(
        g, ex, pos, hid_l, w_l, st_l, paged_kv=True, pool_io=True
    )
    ref_silu, _ = _SynthBaseModel.reference_step(
        g_silu, ex, pos, hid_l, w_l, st_l, paged_kv=True, pool_io=True
    )
    assert (
        float(
            jnp.max(
                jnp.abs(out_f[:4].astype(jnp.float32) - ref_sig.astype(jnp.float32))
            )
        )
        <= 0.032
    )
    diff = float(
        jnp.max(jnp.abs(ref_sig.astype(jnp.float32) - ref_silu.astype(jnp.float32)))
    )
    assert diff > 1e-4


class _H640Model(_SynthBaseModel):
    """Synthetic hybrid model with `H=640` (non-power-of-two multiple of 128)."""

    name = "b8_h640"
    default_hf_config = {
        **_SynthBaseModel.default_hf_config,
        "d_model": 640,
    }

    @classmethod
    def shrink(
        cls,
        hf_config_or_geom=None,
        *,
        periods: int = 2,
        R: int = 8,
        ex: mk.ExecSpec | None = None,
    ) -> mk.ModelGeometry:
        shrunk = dataclasses.replace(
            super().shrink(hf_config_or_geom, periods=periods, R=R, ex=ex),
            H=640,
        )
        object.__setattr__(shrunk, "_model_def", cls)
        return shrunk


def test_moe_col_chunk_h2560() -> None:
    assert mk.col_chunk(4096) == 1024
    assert mk.col_chunk(2048) == 1024
    assert mk.col_chunk(2560) == 640
    assert mk.col_chunk(1920) == 640
    assert mk.col_chunk(1536) == 768
    assert mk.col_chunk(640) == 640

    # H=2560 passes ModelGeometry.check and derives CH=640 in MoEParams
    cfg_2560 = {**_SynthBaseModel.default_hf_config, "d_model": 2560}
    g_2560 = _SynthBaseModel.derive_geometry(cfg_2560)
    ex_2560 = mk.ExecSpec(R=2, B=4, T=1, ctx=2048)
    g_2560.check(ex_2560)
    pl_2560 = mk.planner.plan(_SynthBaseModel.to_model_spec(g_2560, R=2), ex_2560)
    mp_2560 = pl_2560.module_params["moe"]
    assert mp_2560.CH == 640
    assert g_2560.H % mp_2560.CH == 0

    # End-to-end kernel vs reference parity with CH=640 (H=640 < 1024, 1024 % 640 != 0)
    report = mk.verify_model(
        _H640Model,
        R=2,
        B_per_T=2,
        T_values=(1,),
        hf=False,
        approx_profiles=False,
    )
    assert report.geometry.H == 640
    report.assert_passed()


class _Bf16DenseModel(_SynthBaseModel):
    """Synthetic hybrid model with bf16 dense projections and fp8 routed experts."""

    name = "b8_bf16_dense"
    attn = mk.GQA.bind(
        prefix="self_attn",
        heads="n_heads",
        kv_heads="n_kv_heads",
        head_dim="d_head",
        qk_norm=True,
        output_gate=True,
        weight_dtypes={"att_in": "bf16", "att_out": "bf16"},
    )
    linear = mk.GDN.bind(
        prefix="linear_attn",
        k_heads="ssm_k_heads",
        v_heads="ssm_v_heads",
        k_dim="ssm_k_dim",
        v_dim="ssm_v_dim",
        conv_kernel="ssm_conv_kernel",
        weight_dtypes={"lin_in": "bf16", "lin_out": "bf16"},
    )
    moe = mk.MoE.bind(
        prefix="mlp",
        experts="n_routed_experts",
        top_k="top_k_experts",
        inter="d_ff_routed",
        shared_inter="d_ff_shared",
        shared_gate=True,
        weight_dtypes={"sh13": "bf16", "sh2": "bf16", "e13": "fp8_channel"},
    )
    schedule = mk.Schedule(
        mixer=mk.Every(n=2, last=attn, rest=linear),
        ffn=moe,
        pre_norm=_SynthBaseModel.schedule.pre_norm,
        post_norm=_SynthBaseModel.schedule.post_norm,
    )


def test_per_weight_dtype_policy() -> None:
    with pytest.raises(mk.MegakernelConfigError, match="unknown weight"):
        mk.GQAModule(weight_dtypes={"bad": "bf16"})
    with pytest.raises(mk.MegakernelConfigError, match="must be 'fp8' or 'bf16'"):
        mk.GDNModule(weight_dtypes={"lin_in": "int8"})
    with pytest.raises(mk.MegakernelConfigError, match="share the same weight dtype"):
        mk.MoEModule(weight_dtypes={"e13": "bf16", "e2": "fp8"})

    g = _Bf16DenseModel.shrink(periods=1, R=2)
    ex = mk.ExecSpec(
        R=2,
        B=4,
        T=1,
        ctx=256,
        tiles=mk.Tiles(
            bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32
        ).with_modules({"gqa": {"kv_blk": 256}}),
        interpret=True,
        cache=mk.CacheExec(paged_kv=True, num_pages=128, pool_gather=False),
    )
    spec = _Bf16DenseModel.to_model_spec(g, R=2)
    pl = mk.planner.plan(spec, ex)

    # Scale smalls are dropped when dense projections are bf16 passthrough
    for dropped in ("in_s", "out_s", "sh13_s", "sh2_s"):
        assert dropped not in pl.smalls_layout
    for retained in (
        "in_norm",
        "post_norm",
        "qn",
        "kn",
        "conv",
        "alog",
        "dtb",
        "gnorm",
        "sh_gate",
    ):
        assert retained in pl.smalls_layout
    for k_bf16 in ("att_in", "att_out", "lin_in", "lin_out", "sh13", "sh2"):
        assert jnp.dtype(pl.weight_decls[k_bf16].dtype) == jnp.dtype(jnp.bfloat16)
    for k_fp8 in ("e13", "e2"):
        assert jnp.dtype(pl.weight_decls[k_fp8].dtype) == jnp.dtype(jnp.float8_e4m3fn)

    report = mk.verify_model(
        _Bf16DenseModel,
        R=2,
        B_per_T=2,
        T_values=(1, 4),
        hf=False,
        approx_profiles=False,
    )
    report.assert_passed()


def test_merged_tile_operand_layout() -> None:
    # Direct unit test of unpack_local_weight for rank-3, rank-4, and rank-5 shapes
    x3 = jnp.arange(2 * 128 * 64, dtype=jnp.bfloat16).reshape(2 * 128, 64)
    u3 = mk.kernel.unpack_local_weight(
        x3, (2, 128, 64), name="per_layer", allow_expert_reshape=True
    )
    assert u3.shape == (2, 128, 64)
    u3_b = mk.kernel.unpack_local_weight(
        x3[None], (2, 128, 64), name="per_layer", allow_expert_reshape=True
    )
    assert u3_b.shape == (2, 128, 64)

    x4 = jnp.arange(3 * 2 * 128 * 64, dtype=jnp.bfloat16).reshape(3, 2 * 128, 64)
    u4 = mk.kernel.unpack_local_weight(
        x4, (3, 2, 128, 64), name="stacked", allow_expert_reshape=True
    )
    assert u4.shape == (3, 2, 128, 64)

    x_es = jnp.arange(3 * 4 * 64, dtype=jnp.float32).reshape(3, 4, 64)
    u_es = mk.kernel.unpack_local_weight(
        x_es[None], (3, 4, 1, 64), name="es", allow_expert_reshape=True
    )
    assert u_es.shape == (3, 4, 1, 64)

    # End-to-end step parity: merged_tiles=True vs merged_tiles=False (default)
    R, ctx = 2, 256
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = _SynthBaseModel.derive_geometry()
    spec = _SynthBaseModel.to_model_spec(g, R=R)
    ex = mk.ExecSpec(
        R=R,
        B=4,
        T=1,
        ctx=ctx,
        tiles=mk.Tiles(
            bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32
        ).with_modules({"gqa": {"kv_blk": 256}}),
        interpret=True,
        cache=mk.CacheExec(paged_kv=True, num_pages=128, pool_gather=False),
    )
    pl = mk.planner.plan(spec, ex)
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(
        jax.random.key(19), g, ex, model_def=_SynthBaseModel
    )
    _, hid_mk, w_tiled, st_mk = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    w_merged_host = mk.pack_logical_weights(
        _SynthBaseModel, ex, w_l, g=g, merged_tiles=True
    )
    # Default pack_logical_weights keeps tiled layout; merged_tiles=True merges nt*tk
    for k in pl.static_kinds:
        nt_k, tk_k, n_k = pl.nt[k], pl.tk[k], pl.kinds[k][1]
        assert w_tiled[k].shape[-3:] == (nt_k, tk_k, n_k)
        assert w_merged_host[k].shape[-2:] == (nt_k * tk_k, n_k)
        assert w_merged_host[k].ndim == w_tiled[k].ndim - 1
    assert w_tiled["es"].ndim == 5
    assert w_merged_host["es"].ndim == 4

    # load_weights with merged_tiles=True also produces rank-3 stacked static operands
    src = mk.MappingWeightSource(w_l)
    lw_default = mk.load_weights(_SynthBaseModel, ex, src, g=g, tp_rank=0)
    lw_merged = mk.load_weights(
        _SynthBaseModel, ex, src, g=g, tp_rank=0, merged_tiles=True
    )
    for k in pl.static_kinds:
        assert lw_default[k].ndim == 4
        assert lw_merged[k].ndim == 3
    assert lw_default["es"].ndim == 4
    assert lw_merged["es"].ndim == 3

    w_merged = {
        k: (
            tuple(jax.device_put(x, tp) for x in v)
            if isinstance(v, tuple)
            else jax.device_put(v, tp)
        )
        for k, v in w_merged_host.items()
    }

    # DirectHBM make_step bit-exact between tiled and merged operands
    step_d, _ = mk.make_step(mesh, spec, ex, donate_states=False)
    out_d_tiled, _ = step_d(pos, hid_mk, w_tiled, st_mk)
    out_d_merged, _ = step_d(pos, hid_mk, w_merged, st_mk)
    np.testing.assert_array_equal(np.asarray(out_d_merged), np.asarray(out_d_tiled))

    # PallasWriter make_pooled_step and make_flat_pooled_step bit-exact
    play = _SynthBaseModel.pool_layout(g, R=R)
    pt, ckpt, roff, _ = mk.pool.allocate_test_layout(play, ex)
    base_pools = [
        np.asarray(r).copy()
        for r in mk.pool.states_to_pool(
            play, ex, st_l, pt, ckpt[..., :1], jnp.maximum(roff, 0)
        )
    ]
    step_p, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex, donate=False
    )
    out_p_tiled, _ = step_p(
        pos,
        hid_mk,
        w_tiled,
        tuple(jax.device_put(r.copy(), tp) for r in base_pools),
        pt,
        ckpt,
        roff,
    )
    out_p_merged, _ = step_p(
        pos,
        hid_mk,
        w_merged,
        tuple(jax.device_put(r.copy(), tp) for r in base_pools),
        pt,
        ckpt,
        roff,
    )
    np.testing.assert_array_equal(np.asarray(out_p_merged), np.asarray(out_p_tiled))

    flat_step, _, _, _ = mk.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=False, fused_pool_io=False
    )
    w_flat_tiled = []
    w_flat_merged = []
    for k in pl.w_names:
        vt, vm = w_tiled[k], w_merged[k]
        w_flat_tiled.extend(vt if isinstance(vt, tuple) else (vt,))
        w_flat_merged.extend(vm if isinstance(vm, tuple) else (vm,))
    pools_sh = tuple(jax.device_put(r.copy(), tp) for r in base_pools)
    out_f_tiled = flat_step(pos, hid_mk, *w_flat_tiled, *pools_sh, pt, ckpt, roff)
    out_f_merged = flat_step(pos, hid_mk, *w_flat_merged, *pools_sh, pt, ckpt, roff)
    np.testing.assert_array_equal(
        np.asarray(out_f_merged[0]), np.asarray(out_f_tiled[0])
    )


def test_configurable_collective_id(monkeypatch: pytest.MonkeyPatch) -> None:
    assert mk.Tiles().collective_id == 0
    assert mk.ExecSpec().resolved_collective_id == 0
    assert mk.ExecSpec(tiles=mk.Tiles(collective_id=47)).resolved_collective_id == 47
    assert (
        mk.ExecSpec(
            tiles=mk.Tiles(collective_id=12), collective_id=47
        ).resolved_collective_id
        == 47
    )
    assert mk.exec_spec.tile_numerics()["collective_id"] == "exact"

    with pytest.raises(ValueError, match="collective_id"):
        mk.Tiles(collective_id=-1)
    with pytest.raises(ValueError, match="collective_id"):
        mk.ExecSpec(collective_id=-1)

    g = _SynthBaseModel.derive_geometry()
    spec = _SynthBaseModel.to_model_spec(g, R=2)
    base_tiles = mk.Tiles(
        bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32
    ).with_modules({"gqa": {"kv_blk": 256}})
    seen: list[int | None] = []
    real_cp = mk.kernel.pltpu.CompilerParams

    def spy_cp(*args, **kwargs):
        seen.append(kwargs.get("collective_id"))
        return real_cp(*args, **kwargs)

    monkeypatch.setattr(mk.kernel.pltpu, "CompilerParams", spy_cp)

    _, info_def = mk.build(spec, mk.ExecSpec(R=2, B=4, T=1, ctx=256, tiles=base_tiles))
    assert info_def["collective_id"] == 0
    assert seen[-1] == 0

    _, info_tiles = mk.build(
        spec,
        mk.ExecSpec(
            R=2,
            B=4,
            T=1,
            ctx=256,
            tiles=dataclasses.replace(base_tiles, collective_id=47),
        ),
    )
    assert info_tiles["collective_id"] == 47
    assert seen[-1] == 47

    _, info_ex = mk.build(
        spec,
        mk.ExecSpec(R=2, B=4, T=1, ctx=256, tiles=base_tiles, collective_id=99),
    )
    assert info_ex["collective_id"] == 99
    assert seen[-1] == 99


def test_comm_params_derives_pc_hpc_pr() -> None:
    ex = mk.ExecSpec(R=8, B=4, T=1, ctx=2048)
    tiles = mk.Tiles()

    cp_4096 = mk.comm.comm_params(ex, tiles, 4096)
    assert (cp_4096.PC, cp_4096.HPC, cp_4096.PR) == (1024, 4, 16)

    cp_2560 = mk.comm.comm_params(ex, tiles, 2560)
    assert (cp_2560.PC, cp_2560.HPC, cp_2560.PR) == (640, 4, 16)

    # Custom secondary ring width (e.g. HCSublayer with width=128)
    cp_hc = mk.comm.comm_params(ex, tiles, 128)
    assert (cp_hc.PC, cp_hc.HPC, cp_hc.PR) == (128, 1, 16)
    scr = mk.comm.ring_scratch(cp_hc)
    assert "send" in scr and "recv" in scr and "ar_ssem" in scr and "ar_rsem" in scr
