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
"""Unit tests for threshold flag emission, target-envs validation, and CDK ceilings."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline.cli import main as cli_main
from vllm_torchtpu.roofline.solver import ParetoPoint, PoolMeshPlan, ServingPlan

QWEN35_CFG = {
    "text_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
    }
}


def test_emit_threshold_is_band_lower_edge(tmp_path: Path) -> None:
    geom = rf.from_hf_config(QWEN35_CFG)
    chip = rf.CHIPS["tpu7x"]
    topo = rf.TOPOLOGIES["2x2x1"]
    dt = rf.Dtypes(w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0)
    scen = rf.SCENARIO_PRESETS["coding-agent"]
    splan = rf.solve_serving_plan(geom, dt, chip, scen, mode="mixed", max_chips=4)
    par = rf.Parallel(tp=8, moe="ep", ep=8)
    bplan = rf.recommend_bucket_regimes(
        geom,
        dt,
        chip,
        topo,
        par,
        scenario=scen,
        spec_window=3,
        acceptance_length=3.39,
    )
    assert bplan.fused_ep_min_tokens is not None and bplan.fused_ep_min_tokens > 1

    f_envs = tmp_path / "envs_with_step_knob.py"
    f_envs.write_text(
        "environment_variables = {\n"
        "    'TPU_MOE_SEQUENCE_PARALLEL': lambda: False,\n"
        "    'USE_MOE_FUSED_EP_KERNEL': lambda: False,\n"
        "    'MOE_FUSED_EP_KERNEL_MIN_TOKENS': lambda: 1024,\n"
        "    'TPU_MOE_FUSED_EP_STEP_MIN_TOKENS': lambda: 0,\n"
        "}\n",
        encoding="utf-8",
    )

    out = rf.emit_engine_args(
        splan, bplan, engine="vllm", dtypes=dt, target_envs=f_envs
    )
    edge = str(bplan.fused_ep_min_tokens)
    assert out["env"]["USE_MOE_FUSED_EP_KERNEL"] == "1"
    assert out["env"]["MOE_FUSED_EP_KERNEL_MIN_TOKENS"] == edge
    assert out["env"]["TPU_MOE_FUSED_EP_STEP_MIN_TOKENS"] == edge
    assert out["env"]["TPU_MOE_SEQUENCE_PARALLEL"] == "0"
    assert "TPU_MOE_SEQUENCE_PARALLEL" in out["deferred"]
    assert out["refused_flags"] == {}


def test_emit_refuses_flag_absent_from_target_envs(tmp_path: Path) -> None:
    geom = rf.from_hf_config(QWEN35_CFG)
    chip = rf.CHIPS["tpu7x"]
    topo = rf.TOPOLOGIES["2x2x1"]
    dt = rf.Dtypes(w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0)
    scen = rf.SCENARIO_PRESETS["coding-agent"]
    splan = rf.solve_serving_plan(geom, dt, chip, scen, mode="mixed", max_chips=4)
    par = rf.Parallel(tp=8, moe="ep", ep=8)
    bplan = rf.recommend_bucket_regimes(
        geom,
        dt,
        chip,
        topo,
        par,
        scenario=scen,
        spec_window=3,
        acceptance_length=3.39,
    )

    out = rf.emit_engine_args(splan, bplan, engine="vllm", dtypes=dt, target_envs=None)
    assert "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS" not in out["env"]
    assert "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS" in out["refused_flags"]
    assert "absent from" in out["refused_flags"]["TPU_MOE_FUSED_EP_STEP_MIN_TOKENS"]
    assert out["env"]["USE_MOE_FUSED_EP_KERNEL"] == "1"

    cfg_path = tmp_path / "q35_fp8.json"
    cfg_path.write_text(
        json.dumps({**QWEN35_CFG, "quantization_config": {"quant_method": "fp8"}}),
        encoding="utf-8",
    )
    rc = cli_main(
        [
            "emit",
            "--model",
            str(cfg_path),
            "--scenario",
            "coding-agent",
            "--max-chips",
            "4",
            "--strict",
            "--json",
        ]
    )
    assert rc == 1


def test_emit_records_no_sglang_equivalent() -> None:
    geom = rf.from_hf_config(QWEN35_CFG)
    chip = rf.CHIPS["tpu7x"]
    topo = rf.TOPOLOGIES["2x2x1"]
    dt = rf.Dtypes(w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0)
    scen = rf.SCENARIO_PRESETS["chat-8k1k"]
    splan = rf.solve_serving_plan(geom, dt, chip, scen, mode="mixed", max_chips=4)
    par = rf.Parallel(tp=8, moe="ep", ep=8)
    bplan = rf.recommend_bucket_regimes(
        geom, dt, chip, topo, par, context=8192, engine="sglang"
    )
    out = rf.emit_engine_args(splan, bplan, engine="sglang", dtypes=dt)
    assert out["env"] == {}
    assert "sglang" in out
    assert out["sglang"]["TPU_MOE_FUSED_EP_STEP_MIN_TOKENS"] == "no SGLang equivalent"
    assert out["sglang"]["MOE_FUSED_EP_KERNEL_MIN_TOKENS"] == "no SGLang equivalent"


def test_emit_vllm_sglang_flags_and_cdk_ceiling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    geom = rf.from_hf_config(QWEN35_CFG)
    chip = rf.CHIPS["tpu7x"]
    topo = rf.TOPOLOGIES["2x2x1"]
    dt = rf.Dtypes(w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0)
    scen = rf.SCENARIO_PRESETS["chat-8k1k"]
    splan = rf.solve_serving_plan(geom, dt, chip, scen, mode="mixed", max_chips=4)
    par = rf.Parallel(tp=8, moe="ep", ep=8)
    bplan = rf.recommend_bucket_regimes(geom, dt, chip, topo, par, context=8192)

    vllm_out = rf.emit_engine_args(splan, bplan, engine="vllm", dtypes=dt)
    assert vllm_out["engine"] == "vllm"
    assert "--tensor-parallel-size" in vllm_out["cli_args"]
    assert "--enable-expert-parallel" in vllm_out["cli_args"]
    # Startup SP stays off when bucket 1 is replicated; fused EP turns on at
    # its band edge.
    assert vllm_out["env"]["TPU_MOE_SEQUENCE_PARALLEL"] == "0"
    assert vllm_out["env"]["USE_MOE_FUSED_EP_KERNEL"] == "1"
    assert vllm_out["env"]["MOE_FUSED_EP_KERNEL_MIN_TOKENS"] == str(
        bplan.fused_ep_min_tokens
    )
    assert "TPU_MEGAKERNEL_MAX_TOKENS" not in vllm_out["env"]
    assert "TPU_MOE_SP_MIN_TOKENS" not in vllm_out["env"]
    assert len(vllm_out["bucket_bands"]) >= 2
    assert "serving_plan" in vllm_out

    hi_bplan = rf.recommend_bucket_regimes(
        geom, dt, chip, topo, par, buckets=(1024, 2048), context=8192
    )
    hi_out = rf.emit_engine_args(splan, hi_bplan, engine="vllm", dtypes=dt)
    assert hi_out["env"]["TPU_MOE_SEQUENCE_PARALLEL"] == "1"
    assert hi_out["env"]["USE_MOE_FUSED_EP_KERNEL"] == "1"
    assert hi_out["env"]["MOE_FUSED_EP_KERNEL_MIN_TOKENS"] == "1024"

    sglang_bplan = rf.recommend_bucket_regimes(
        geom, dt, chip, topo, par, context=8192, engine="sglang"
    )
    sgl_out = rf.emit_engine_args(splan, sglang_bplan, engine="sglang", dtypes=dt)
    assert sgl_out["engine"] == "sglang"
    assert "--tp-size" in sgl_out["cli_args"]
    assert "--ep-size" in sgl_out["cli_args"]
    assert "--enable-ep-moe" not in sgl_out["cli_args"]
    assert "fp8_e4m3" in sgl_out["cli_args"]
    assert sgl_out["env"] == {}

    dp_pool = PoolMeshPlan(
        role="decode",
        topology="2x2x1",
        instances=1,
        chips_per_instance=4,
        tp=4,
        dp=2,
        ep=8,
        pp=1,
        pcp=1,
        kv_split=1,
        mesh_axes=(2, 2, 1, 1),
        hbm_used_gb=60.0,
        hbm_limit_gb=87.5,
    )
    dp_pt = ParetoPoint(
        mode="mixed",
        concurrency=32,
        total_chips=4,
        prefill_pool=None,
        decode_pool=dp_pool,
        p90_interactivity_tok_s=40.0,
        token_throughput_per_chip=5000.0,
        output_throughput_per_chip=500.0,
        p90_ttft_ms=250.0,
        p90_tpot_ms=25.0,
        p_mixed=0.1,
        binding="hbm",
        slo_met=True,
    )
    dp_splan = ServingPlan(
        model_layers=geom.num_layers,
        chip="tpu7x",
        scenario=scen,
        mode="mixed",
        recommended=dp_pt,
        frontier=(dp_pt,),
        warnings=(),
    )
    dp_par = rf.Parallel(tp=4, dp=2, moe="ep", ep=8, moe_dispatch="ag_ep")
    dp_bplan = rf.recommend_bucket_regimes(
        geom, dt, chip, topo, dp_par, buckets=(16, 128), context=8192, engine="sglang"
    )
    assert all(not r.megakernel and r.moe_dispatch == "ag_ep" for r in dp_bplan.rows)
    dp_sgl = rf.emit_engine_args(dp_splan, dp_bplan, engine="sglang", dtypes=dt)
    assert "--enable-dp-attention" in dp_sgl["cli_args"]
    tp_idx = dp_sgl["cli_args"].index("--tp-size")
    dp_idx = dp_sgl["cli_args"].index("--dp-size")
    ep_idx = dp_sgl["cli_args"].index("--ep-size")
    assert dp_sgl["cli_args"][tp_idx + 1] == "8"
    assert dp_sgl["cli_args"][dp_idx + 1] == "2"
    assert dp_sgl["cli_args"][ep_idx + 1] == "8"

    dp_vllm_geom = replace(geom, top_k=24)
    dp_vllm_bplan = rf.recommend_bucket_regimes(
        dp_vllm_geom, dt, chip, topo, dp_par, buckets=(16, 64, 128), context=8192
    )
    assert all(r.moe_dispatch == "ag_ep" for r in dp_vllm_bplan.rows)
    assert dp_vllm_bplan.fused_ep_min_tokens is None
    dp_vllm_out = rf.emit_engine_args(dp_splan, dp_vllm_bplan, engine="vllm", dtypes=dt)
    assert dp_vllm_out["env"]["USE_MOE_FUSED_EP_KERNEL"] == "0"
    assert "MOE_FUSED_EP_KERNEL_MIN_TOKENS" not in dp_vllm_out["env"]

    e128_geom = replace(geom, num_experts=128)
    e128_bplan = rf.recommend_bucket_regimes(
        e128_geom, dt, chip, topo, par, buckets=(2048,), context=8192, engine="sglang"
    )
    assert e128_bplan.rows[0].moe_dispatch == "fused_rs"

    ceil_rows = rf.emit_cdk_ceiling_rows(splan)
    assert ceil_rows
    first_keys = tuple(ceil_rows[0].keys())
    assert first_keys[:2] == ("p90_interactivity_tok_s", "token_throughput_per_chip")
    for r in ceil_rows:
        assert r["output_throughput_per_chip"] > 0.0
        assert r["p90_interactivity_tok_s"] == pytest.approx(1000.0 / r["p90_tpot_ms"])
        assert "p90_ttft_ms" in r and "p90_tpot_ms" in r and "binding_unit" in r

    cfg_path = tmp_path / "q35.json"
    q35_fp8 = {**QWEN35_CFG, "quantization_config": {"quant_method": "fp8"}}
    cfg_path.write_text(json.dumps(q35_fp8), encoding="utf-8")
    for sub_argv in (
        ["buckets", "--model", str(cfg_path), "--tp", "8", "--moe", "ep", "--json"],
        [
            "emit",
            "--model",
            str(cfg_path),
            "--engine",
            "vllm",
            "--max-chips",
            "4",
            "--json",
        ],
        [
            "emit",
            "--model",
            str(cfg_path),
            "--engine",
            "sglang",
            "--max-chips",
            "4",
            "--json",
        ],
        ["ceiling", "--model", str(cfg_path), "--max-chips", "4", "--json"],
    ):
        assert cli_main(sub_argv) == 0
        assert json.loads(capsys.readouterr().out)
