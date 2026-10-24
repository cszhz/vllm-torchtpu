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

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import (
    qwen35,
)
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES
from vllm_torchtpu.megakernel.ops import F32, I32

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


def test_gdn_block_with_kda_key_decay_and_gate_in_process():
    """Verify GDNModule KDA per-key decay and sigmoid gate numerical parity."""
    from dataclasses import replace as dc_replace

    from vllm_torchtpu.megakernel.modules.gdn import GDNModule
    from vllm_torchtpu.megakernel.modules.gdn.reference import gdn_reference

    gb_default = GDNModule(prefix="linear_attn")
    gb_bound = GDNModule.bind(prefix="linear_attn")
    assert dict(gb_default.options) == dict(gb_bound.options)
    gb_kda = GDNModule(
        prefix="linear_attn",
        decay_type="key",
        decay_law="bounded_sigmoid",
        gate_type="sigmoid",
        decay_bound=5.0,
    )
    assert gb_kda.options["decay_type"] == "key"

    g_base = _tiny_qwen38_geometry()
    ex8 = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        tiles=dc_replace(TINY_TILES, bank=512 << 10, hier=0, mb8=1, inplace=False),
        interpret=True,
    )

    class _KDAGeom(qwen35.Qwen35Geometry):
        decay_type: str = "key"
        decay_law: str = "bounded_sigmoid"
        gate_type: str = "sigmoid"
        decay_bound: float = 5.0

    class _HeadBSigGeom(qwen35.Qwen35Geometry):
        decay_type: str = "head"
        decay_law: str = "bounded_sigmoid"
        gate_type: str = "sigmoid"
        decay_bound: float = 5.0

    base_fields = {
        f.name: getattr(g_base, f.name) for f in g_base.__dataclass_fields__.values()
    }
    g_kda = _KDAGeom(**base_fields)
    g_head_bsig = _HeadBSigGeom(**base_fields)
    lay_kda = GDNModule.lin_in_layout(g_kda, ex8)
    assert lay_kda.vh * lay_kda.KD == lay_kda.A_DIM
    assert gb_kda.weights(g_base, ex8)[0].N == lay_kda.NLI
    assert gb_kda.smalls(g_base, ex8)[4].width == lay_kda.A_DIM
    lay_sc = GDNModule.lin_in_layout(g_base, ex8)
    _, hid_l, w_sc, st_l = qwen35.random_logical_inputs(
        jax.random.PRNGKey(92), g_base, ex8
    )
    out_sc, _ = gdn_reference(w_sc, hid_l, st_l, g=g_base, ex=ex8, gdn_idx=0)
    out_head_bsig, _ = gdn_reference(
        w_sc, hid_l, st_l, g=g_head_bsig, ex=ex8, gdn_idx=0
    )

    a_off = lay_sc.A_OFF
    vh = lay_sc.vh
    kd = lay_kda.KD
    w_in_prefix = w_sc["lin_in"][..., :a_off]
    w_in_a_rep = jnp.repeat(w_sc["lin_in"][..., a_off : a_off + vh], kd, axis=-1)
    w_in_cat = jnp.concatenate([w_in_prefix, w_in_a_rep], axis=-1)
    pad_cols = lay_kda.NLI - w_in_cat.shape[-1]
    s_prefix = w_sc["lin_in_s"][..., :a_off]
    s_a_rep = jnp.repeat(w_sc["lin_in_s"][..., a_off : a_off + vh], kd, axis=-1)
    s_cat = jnp.concatenate([s_prefix, s_a_rep], axis=-1)
    w_kda = dict(w_sc)
    w_kda["lin_in"] = jnp.pad(w_in_cat, ((0, 0), (0, 0), (0, 0), (0, pad_cols)))
    w_kda["lin_in_s"] = jnp.pad(
        s_cat, ((0, 0), (0, 0), (0, pad_cols)), constant_values=0.01
    )
    w_kda["alog"] = jnp.repeat(w_sc["alog"], kd, axis=-1)
    w_kda["dtb"] = jnp.repeat(w_sc["dtb"], kd, axis=-1)
    out_kda, _ = gdn_reference(w_kda, hid_l, st_l, g=g_kda, ex=ex8, gdn_idx=0)
    w_bad_kda = dict(w_kda, alog=w_kda["alog"] + 3.0)
    out_bad_kda, _ = gdn_reference(w_bad_kda, hid_l, st_l, g=g_kda, ex=ex8, gdn_idx=0)
    _check(
        "kda gdn_reference vs head bounded_sigmoid",
        out_kda,
        out_head_bsig,
        [("softplus/silu control", out_sc), ("perturbed alog", out_bad_kda)],
    )
    devs = jax.devices("cpu")
    if len(devs) >= ex8.R:
        pos = POS0 + jnp.arange(ex8.NS, dtype=I32)
        ref_hid_kda, _ = qwen35.reference_step(g_kda, ex8, pos, hid_l, w_kda, st_l)
        ref_hid_bad_kda, _ = qwen35.reference_step(
            g_kda, ex8, pos, hid_l, w_bad_kda, st_l
        )
        mesh = Mesh(devs[: ex8.R], ("tp",))
        pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
            pos, hid_l, w_kda, st_l, g_kda, ex8, mesh=mesh
        )
        step_kda, _ = qwen35.make_step(mesh, g_kda, ex8, donate_states=True)
        out_hid_kda, _ = step_kda(pos_mk, hid_mk, w_mk, st_mk)
        hid_kda_r = kernel.unfold(out_hid_kda, ex8.R)[:, : ex8.B]
        _check(
            "kda make_step vs reference_step",
            hid_kda_r[0],
            ref_hid_kda,
            [("perturbed alog", ref_hid_bad_kda), ("input hidden", hid_l)],
        )


def _kimi_k3_kda_oracle_gdn_layer(
    w_kda: dict[str, jax.Array],
    hid_l: jax.Array,
    st_l: dict[str, jax.Array],
    g_kda: object,
    ex: ExecSpec,
    *,
    gdn_idx: int = 0,
    wrong_axis: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Float64 transcription of origin/main kimi_k3/kda.py::kda_step on 1 GDN layer."""
    import numpy as np

    from vllm_torchtpu.megakernel.modules.gdn import GDNModule

    R = ex.R
    B = ex.B
    H = g_kda.H
    kh = g_kda.kh(R)
    vh = g_kda.vh(R)
    KD = g_kda.KD
    VD = g_kda.VD
    lay = GDNModule.lin_in_layout(g_kda, ex)
    QKV, Z, BA, A_OFF = lay.QKV, lay.Z, lay.BA, lay.A_OFF
    a_dim = vh * KD
    expand = vh // kh

    # gdn_reference receives x directly (input norm is applied by Norm module)
    norm_x = np.asarray(hid_l[:B].astype(jnp.bfloat16), dtype=np.float32).astype(
        np.float64
    )

    lin_in_all = np.asarray(w_kda["lin_in"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    lin_in_s_all = np.asarray(w_kda["lin_in_s"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    conv_w_all = np.asarray(w_kda["conv"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    alog_all = np.asarray(w_kda["alog"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    dtb_all = np.asarray(w_kda["dtb"][:, gdn_idx], dtype=np.float64)
    gnorm_all = np.asarray(w_kda["gnorm"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    lin_out_all = np.asarray(w_kda["lin_out"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    lin_out_s_all = np.asarray(w_kda["lin_out_s"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    conv_st_all = np.asarray(st_l["conv"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )
    rec_st_all = np.asarray(st_l["rec"][:, gdn_idx], dtype=np.float32).astype(
        np.float64
    )

    decay_law = str(getattr(g_kda, "decay_law", "softplus"))
    gate_type = str(getattr(g_kda, "gate_type", "silu"))
    decay_bound = float(getattr(g_kda, "decay_bound", 5.0))

    rank_partials = []
    rec_next_ranks = []
    for r in range(R):
        w_in_r = lin_in_all[r].reshape(-1, lay.NLI)
        ybuf = (norm_x @ w_in_r) * lin_in_s_all[r][None, :]
        # Causal depthwise 4-tap conv1d + SiLU on QKV
        c_st = conv_st_all[r]  # [B, 3, QKV]
        cw = conv_w_all[r]  # [4 * QKV]
        taps = [cw[t * QKV : (t + 1) * QKV] for t in range(4)]
        conv_out = (
            c_st[:, 0, :] * taps[0]
            + c_st[:, 1, :] * taps[1]
            + c_st[:, 2, :] * taps[2]
            + ybuf[:, :QKV] * taps[3]
        )
        u = conv_out / (1.0 + np.exp(-conv_out))

        alog = alog_all[r, :a_dim]
        dtb = dtb_all[r, :a_dim]
        gnw = gnorm_all[r, :VD]
        mix_rows = []
        rec_b_list = []
        for b in range(B):
            st_cur = (
                rec_st_all[r, b].copy()
                if rec_st_all.ndim == 5
                else rec_st_all[r, b * vh : (b + 1) * vh].copy()
            )  # [vh, KD, VD]
            u_b = u[b]
            q = u_b[: kh * KD].reshape(kh, KD)
            k = u_b[kh * KD : 2 * kh * KD].reshape(kh, KD)
            v = u_b[2 * kh * KD : QKV].reshape(vh, VD)
            q_den = np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + g_kda.gdn_l2_eps)
            k_den = np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + g_kda.gdn_l2_eps)
            q = q * (KD**-0.5) / q_den
            k = k / k_den
            qc = np.repeat(q, expand, axis=0)[:, :, None]
            kc = np.repeat(k, expand, axis=0)[:, :, None]
            beta = 1.0 / (1.0 + np.exp(-ybuf[b, BA : BA + vh]))
            a_val = ybuf[b, A_OFF : A_OFF + a_dim]
            if decay_law == "bounded_sigmoid":
                sig_arg = np.exp(alog) * (a_val + dtb)
                decay_flat = -decay_bound * (1.0 / (1.0 + np.exp(-sig_arg)))
            else:
                decay_flat = -np.exp(alog) * np.log1p(np.exp(a_val + dtb))
            if wrong_axis:
                # Deliberate axis bug: broadcast KD decay across first KD cols of VD
                eg_bad = np.ones((vh, 1, VD), dtype=np.float64)
                eg_bad[:, 0, :KD] = np.exp(decay_flat.reshape(vh, KD))
                st_cur = st_cur * eg_bad
            else:
                eg = np.exp(decay_flat.reshape(vh, KD))[:, :, None]
                st_cur = st_cur * eg
            pred = np.sum(st_cur * kc, axis=1)
            delta = (v - pred) * beta[:, None]
            st_cur = st_cur + kc * delta[:, None, :]
            rec_b_list.append(st_cur)
            out_h = np.sum(st_cur * qc, axis=1)
            val = out_h / np.sqrt(
                np.mean(out_h * out_h, axis=-1, keepdims=True) + g_kda.eps
            )
            val = gnw[None, :] * val
            zz = ybuf[b, QKV : QKV + Z].reshape(vh, VD)
            gate_val = (
                (1.0 / (1.0 + np.exp(-zz)))
                if gate_type == "sigmoid"
                else (zz / (1.0 + np.exp(-zz)))
            )
            mix_rows.append((val * gate_val).reshape(Z))
        mix = np.stack(mix_rows, axis=0)
        w_out_r = lin_out_all[r].reshape(Z, H)
        yo = (mix @ w_out_r) * lin_out_s_all[r][None, :]
        rank_partials.append(yo)
        rec_next_ranks.append(np.concatenate(rec_b_list, axis=0))

    y_sum = np.sum(rank_partials, axis=0)
    return (
        jnp.asarray(y_sum, dtype=F32),
        jnp.asarray(np.stack(rec_next_ranks, axis=0), dtype=F32),
    )


def test_gdn_kda_step_matches_kimi_kda_step():
    """E1-E4: Rectangular KD=128 != VD=256 KDA step matches Kimi K3 oracle."""
    from dataclasses import replace as dc_replace

    import numpy as np

    from vllm_torchtpu.megakernel.errors import MegakernelConfigError
    from vllm_torchtpu.megakernel.modules.gdn import GDNModule
    from vllm_torchtpu.megakernel.modules.gdn.decls import gdn_hf_params
    from vllm_torchtpu.megakernel.modules.gdn.reference import gdn_reference

    # E4: validate_gdn_options rejects decay_bound <= 0
    with pytest.raises(MegakernelConfigError, match="decay_bound > 0"):
        GDNModule(decay_bound=0.0)
    with pytest.raises(MegakernelConfigError, match="decay_bound > 0"):
        GDNModule(decay_bound=-3.0)

    class _RectKDAGeom(qwen35.Qwen35Geometry):
        decay_type: str = "key"
        decay_law: str = "bounded_sigmoid"
        gate_type: str = "sigmoid"
        decay_bound: float = 5.0

    g_base = dc_replace(_tiny_qwen38_geometry(), KD=128, VD=256)
    base_fields = {
        f.name: getattr(g_base, f.name) for f in g_base.__dataclass_fields__.values()
    }
    g_kda = _RectKDAGeom(**base_fields)
    assert g_kda.KD == 128 and g_kda.VD == 256 and g_kda.KD != g_kda.VD

    # E2: explicit option vs geometry default preservation
    mod_default = GDNModule(prefix="linear_attn")
    assert mod_default.resolved_options(g_kda)["decay_type"] == "key"
    mod_explicit_head = GDNModule(prefix="linear_attn", decay_type="head")
    assert mod_explicit_head.resolved_options(g_kda)["decay_type"] == "head"

    b_default = GDNModule.bind(prefix="linear_attn")
    m_from_default_bind = b_default.resolve(QWEN38_FULL_HF_CONFIG).to_module()
    assert m_from_default_bind.resolved_options(g_kda)["decay_type"] == "key"

    b_explicit_head = GDNModule.bind(prefix="linear_attn", decay_type="head")
    m_from_explicit_bind = b_explicit_head.resolve(QWEN38_FULL_HF_CONFIG).to_module()
    assert m_from_explicit_bind.resolved_options(g_kda)["decay_type"] == "head"

    # E3: gdn_hf_params rejects decay_type="key"
    with pytest.raises(MegakernelConfigError, match="decay_type='key'"):
        gdn_hf_params("model.layers.0.linear_attn", g_kda)

    # E1: Rectangular KD=128 != VD=256 with independent random per-key a cols & dtb
    ex8 = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        tiles=dc_replace(TINY_TILES, bank=512 << 10, hier=0, mb8=1, inplace=False),
        interpret=True,
    )
    lay_sc = GDNModule.lin_in_layout(g_base, ex8)
    lay_kda = GDNModule.lin_in_layout(g_kda, ex8)
    assert lay_kda.vh * 128 == lay_kda.A_DIM

    _, hid_l, w_sc, st_l = qwen35.random_logical_inputs(
        jax.random.PRNGKey(2026), g_base, ex8
    )
    a_off = lay_sc.A_OFF
    a_dim = lay_kda.A_DIM
    rng = np.random.default_rng(2026)

    # Construct independent random per-key a columns (NOT repeated head scalars)
    w_in_prefix = np.asarray(w_sc["lin_in"][..., :a_off])
    R_cnt, L_cnt, H_dim, _ = w_in_prefix.shape
    a_rand_fp8 = jnp.asarray(
        rng.standard_normal((R_cnt, L_cnt, H_dim, a_dim), dtype=np.float32) * 0.5,
        dtype=w_sc["lin_in"].dtype,
    )
    pad_cols = lay_kda.NLI - (a_off + a_dim)
    w_in_kda = jnp.concatenate(
        [
            w_sc["lin_in"][..., :a_off],
            a_rand_fp8,
            jnp.zeros((R_cnt, L_cnt, H_dim, pad_cols), dtype=w_sc["lin_in"].dtype),
        ],
        axis=-1,
    )
    s_a_rand = jnp.asarray(
        np.abs(rng.standard_normal((R_cnt, L_cnt, a_dim), dtype=np.float32)) * 0.2
        + 0.05,
        dtype=F32,
    )
    s_in_kda = jnp.concatenate(
        [
            w_sc["lin_in_s"][..., :a_off],
            s_a_rand,
            jnp.ones((R_cnt, L_cnt, pad_cols), dtype=F32) * 0.01,
        ],
        axis=-1,
    )
    alog_kda = jnp.asarray(
        rng.uniform(-0.5, 1.5, size=(R_cnt, L_cnt, a_dim)).astype(np.float32),
        dtype=F32,
    )
    dtb_kda = jnp.asarray(
        rng.standard_normal((R_cnt, L_cnt, a_dim), dtype=np.float32) * 1.5,
        dtype=F32,
    )

    w_kda = dict(w_sc)
    w_kda["lin_in"] = w_in_kda
    w_kda["lin_in_s"] = s_in_kda
    w_kda["alog"] = alog_kda
    w_kda["dtb"] = dtb_kda

    out_ref, st_ref = gdn_reference(w_kda, hid_l, st_l, g=g_kda, ex=ex8, gdn_idx=0)
    gb_kda = GDNModule(
        prefix="linear_attn",
        decay_type="key",
        decay_law="bounded_sigmoid",
        gate_type="sigmoid",
        decay_bound=5.0,
    )
    out_mod_ref, _ = gb_kda.reference(w_kda, hid_l, st_l, g=g_base, ex=ex8, gdn_idx=0)
    np.testing.assert_allclose(
        np.asarray(out_mod_ref, dtype=np.float32),
        np.asarray(out_ref, dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )
    out_oracle, st_oracle = _kimi_k3_kda_oracle_gdn_layer(
        w_kda, hid_l, st_l, g_kda, ex8, gdn_idx=0, wrong_axis=False
    )
    out_wrong_axis, _ = _kimi_k3_kda_oracle_gdn_layer(
        w_kda, hid_l, st_l, g_kda, ex8, gdn_idx=0, wrong_axis=True
    )
    _check(
        "rectangular KD=128 != VD=256 gdn_reference vs float64 Kimi K3 oracle",
        out_ref,
        out_oracle,
        [("wrong VD-axis decay broadcast", out_wrong_axis)],
    )
    rec_ref_out = (
        st_ref["rec_out"][:, 0]
        if "rec_out" in st_ref
        else (st_ref["rec"][:, 0] if st_ref["rec"].ndim == 5 else st_ref["rec"])
    )
    np.testing.assert_allclose(
        np.asarray(rec_ref_out, dtype=np.float32),
        np.asarray(st_oracle, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

    # Full 8-rank Pallas make_step vs reference_step on rectangular KD=128 != VD=256
    devs = jax.devices("cpu")
    assert len(devs) >= ex8.R
    pos = POS0 + jnp.arange(ex8.NS, dtype=I32)
    ref_hid_kda, _ = qwen35.reference_step(g_kda, ex8, pos, hid_l, w_kda, st_l)
    w_bad_kda = dict(w_kda, dtb=w_kda["dtb"] + 4.0)
    ref_hid_bad_kda, _ = qwen35.reference_step(g_kda, ex8, pos, hid_l, w_bad_kda, st_l)
    mesh = Mesh(devs[: ex8.R], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_kda, st_l, g_kda, ex8, mesh=mesh
    )
    step_kda, _ = qwen35.make_step(mesh, g_kda, ex8, donate_states=True)
    out_hid_kda, _ = step_kda(pos_mk, hid_mk, w_mk, st_mk)
    hid_kda_r = kernel.unfold(out_hid_kda, ex8.R)[:, : ex8.B]
    _check(
        "rectangular KD=128 != VD=256 make_step vs reference_step",
        hid_kda_r[0],
        ref_hid_kda,
        [("perturbed dtb", ref_hid_bad_kda), ("input hidden", hid_l)],
    )


def test_gdn_options_are_plain_values_with_one_resolver():
    """Requested options stay plain values; explicit > geometry > default."""
    from vllm_torchtpu.megakernel.modules.gdn import GDNModule

    b = GDNModule.bind(prefix="linear_attn")
    assert all(v is None for v in b.options.values())
    m = GDNModule(prefix="linear_attn", decay_type="key")
    assert type(m.options["decay_type"]) is str and m.options["decay_law"] is None
    assert m.resolved_options(None) == {
        "decay_type": "key",
        "decay_law": "softplus",
        "gate_type": "silu",
        "decay_bound": 5.0,
    }
