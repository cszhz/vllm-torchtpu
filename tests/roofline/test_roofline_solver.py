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
"""Unit tests for mesh/prefill-decode solver and CLI solve subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_torchtpu.roofline import cli

QWEN35_CFG: dict[str, object] = {
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


def test_solver_qwen35_mixed_vs_1p1d_and_qwen38_crossover(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import vllm_torchtpu.roofline as rf

    q35_geom = rf.from_hf_config(QWEN35_CFG)
    chip = rf.CHIPS["tpu7x"]
    scen = rf.SCENARIO_PRESETS["chat-8k1k"]
    mixed_plan = rf.solve_serving_plan(
        q35_geom, rf.Dtypes(kv=1.0), chip, scen, mode="mixed", max_chips=8
    )
    disagg_plan = rf.solve_serving_plan(
        q35_geom, rf.Dtypes(kv=1.0), chip, scen, mode="disagg", max_chips=8
    )
    mixed_c16 = [p for p in mixed_plan.frontier if p.concurrency >= 16]
    assert mixed_c16 and any(p.p_mixed >= 0.10 for p in mixed_c16)
    assert disagg_plan.recommended is not None
    assert disagg_plan.recommended.p_mixed == 0.0

    # coding-agent (accept_len=3.4) with 1 MTP layer prices k=3 window.
    ca_scen = rf.build_scenario("coding-agent", concurrencies=(1,))
    ca_plan = rf.solve_serving_plan(
        q35_geom, rf.Dtypes(kv=1.0), chip, ca_scen, mode="disagg", max_chips=8
    )
    assert ca_plan.recommended is not None
    rec_d = ca_plan.recommended.decode_pool
    sf_k3 = rf.spec_cycle_floor(
        q35_geom,
        chip,
        rf.Parallel(
            tp=rec_d.tp,
            dp=rec_d.dp,
            moe="ep" if rec_d.ep > 1 else "tp_slice",
            ep=rec_d.ep,
            moe_dispatch="ag_ep"
            if (rec_d.ep > 1 and (rec_d.dp > 1 or rec_d.ep >= 16))
            else "allreduce",
            kv_split=rec_d.kv_split,
            mesh_axes=rec_d.mesh_axes,
        ),
        max(1, (1 + rec_d.dp - 1) // rec_d.dp),
        ca_scen.isl + ca_scen.osl // 2,
        3,
        rf.Dtypes(kv=1.0),
        topology=rf.TOPOLOGIES[rec_d.topology],
    )
    assert ca_plan.recommended.p90_tpot_ms == pytest.approx(
        sf_k3.serial_floor_s * 1e3 / ca_scen.accept_len, rel=1e-4
    )

    # DP > 1 MoE decode candidates price cross-DP EP collectives even under allreduce.
    from dataclasses import replace as dc_replace

    from vllm_torchtpu.roofline.solver import _candidate_decode_parallels

    dp_cands = [
        p
        for p in _candidate_decode_parallels(q35_geom, rf.TOPOLOGIES["2x2x1"])
        if p.dp > 1
    ]
    assert dp_cands
    for par_dp in dp_cands:
        assert par_dp.moe == "ep" and par_dp.ep == 8 and par_dp.moe_dispatch == "ag_ep"
        sf_dp = rf.step_floor(
            q35_geom,
            chip,
            par_dp,
            rf.StepShape(batch=4, context=1024),
            topology=rf.TOPOLOGIES["2x2x1"],
        )
        sf_dp_ar = rf.step_floor(
            q35_geom,
            chip,
            dc_replace(par_dp, moe_dispatch="allreduce"),
            rf.StepShape(batch=4, context=1024),
            topology=rf.TOPOLOGIES["2x2x1"],
        )
        assert sf_dp.chain_s > 0.0
        assert sf_dp_ar.chain_s == pytest.approx(sf_dp.chain_s)

    # Mixed prefill work scales as 1/dp (chunk_tok per replica).
    dp2_geom = rf.from_hf_config(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "intermediate_size": 4096,
            "vocab_size": 32768,
        }
    )
    dp2_scen = rf.TrafficScenario(name="custom", isl=4096, osl=512, concurrencies=(8,))
    dp2_plan = rf.solve_serving_plan(
        dp2_geom, rf.Dtypes(kv=1.0), chip, dp2_scen, mode="mixed", max_chips=1
    )
    assert dp2_plan.recommended is not None and dp2_plan.recommended.decode_pool.dp == 2
    par_dp2 = rf.Parallel(tp=1, dp=2, mesh_axes=(1, 1, 1, 1))
    sf_d_dp2 = rf.step_floor(
        dp2_geom,
        chip,
        par_dp2,
        rf.StepShape(batch=4, context=4352),
        rf.Dtypes(kv=1.0),
        topology=rf.TOPOLOGIES["1x1x1"],
    )
    sf_m_dp2 = rf.step_floor(
        dp2_geom,
        chip,
        par_dp2,
        rf.StepShape(batch=4, context=4352, prefill_tokens=2048),
        rf.Dtypes(kv=1.0),
        topology=rf.TOPOLOGIES["1x1x1"],
    )
    exp_p_mix = (4 * 2 * sf_m_dp2.serial_floor_s) / (
        4 * 2 * sf_m_dp2.serial_floor_s + 512 * sf_d_dp2.serial_floor_s
    )
    assert dp2_plan.recommended.p_mixed == pytest.approx(exp_p_mix, rel=1e-5)

    # PP > 1 prefill TTFT includes stage fill for n_chunks=1 and n_chunks=4.
    pp_geom = rf.from_hf_config(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "intermediate_size": 4096,
            "vocab_size": 32768,
        }
    )
    for p_chunk, n_chk in ((2048, 1), (256, 4)):
        chk_scen = rf.TrafficScenario(
            name="custom", isl=1024, osl=256, concurrencies=(1,), prefill_chunk=p_chunk
        )
        chk_disagg = rf.solve_serving_plan(
            pp_geom, rf.Dtypes(kv=1.0), chip, chk_scen, mode="disagg", max_chips=8
        )
        pp_pts = [
            p
            for p in chk_disagg.frontier
            if p.prefill_pool is not None and p.prefill_pool.pp > 1
        ]
        assert pp_pts
        p0 = pp_pts[0]
        assert p0.prefill_pool is not None
        sf_st = rf.step_floor(
            pp_geom,
            chip,
            rf.Parallel(
                tp=p0.prefill_pool.tp,
                pp=p0.prefill_pool.pp,
                pcp=p0.prefill_pool.pcp,
                moe="ep" if p0.prefill_pool.ep > 1 else "tp_slice",
                ep=p0.prefill_pool.ep,
                mesh_axes=p0.prefill_pool.mesh_axes,
            ),
            rf.StepShape(
                batch=0, rows_per_seq=0, context=0, state_writes=0, prefill_tokens=1024
            ),
            rf.Dtypes(kv=1.0),
            topology=rf.TOPOLOGIES[p0.prefill_pool.topology],
        )
        bubble_mult = 1.0 + float(p0.prefill_pool.pp - 1) / float(n_chk)
        assert p0.p90_ttft_ms == pytest.approx(
            sf_st.serial_floor_s * bubble_mult * 1e3, rel=1e-4
        )

    hybrid_2t_cfg = {
        "text_config": {
            "hidden_size": 8192,
            "num_hidden_layers": 92,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 23,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "attn_output_gate": True,
            "linear_num_key_heads": 32,
            "linear_num_value_heads": 128,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 2048,
            "shared_expert_intermediate_size": 2048,
            "vocab_size": 248320,
        }
    }
    hybrid_2t_geom = rf.from_hf_config(hybrid_2t_cfg)
    hybrid_2t_dt = rf.Dtypes(
        w_dense=1.0, w_expert=0.5625, w_aux=1.0, kv=1.0, state=2.0, lm_head=2.0
    )
    topo_32 = rf.TOPOLOGIES["2x2x4"]
    par_tp32 = rf.Parallel(
        tp=32, dp=1, moe="ep", ep=32, moe_dispatch="ag_ep", kv_split=8
    )
    par_tp8_dp4 = rf.Parallel(
        tp=8, dp=4, moe="ep", ep=32, moe_dispatch="ag_ep", kv_split=2
    )
    sf_tp32_c4 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp32,
        rf.StepShape(batch=4, context=131072),
        hybrid_2t_dt,
        topology=topo_32,
    )
    sf_dp4_c4 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp8_dp4,
        rf.StepShape(batch=1, context=131072),
        hybrid_2t_dt,
        topology=topo_32,
    )
    assert sf_tp32_c4.serial_floor_s < sf_dp4_c4.serial_floor_s

    sf_tp32_c256 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp32,
        rf.StepShape(batch=256, context=8192),
        hybrid_2t_dt,
        topology=topo_32,
    )
    sf_dp4_c256 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp8_dp4,
        rf.StepShape(batch=64, context=8192),
        hybrid_2t_dt,
        topology=topo_32,
    )
    assert sf_dp4_c256.serial_floor_s < sf_tp32_c256.serial_floor_s

    gpt_oss_cfg = {
        "hidden_size": 2880,
        "num_hidden_layers": 36,
        "layer_types": ["sliding_attention", "full_attention"] * 18,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "sliding_window": 128,
        "num_local_experts": 128,
        "num_experts_per_tok": 4,
        "intermediate_size": 2880,
        "vocab_size": 201088,
        "quantization_config": {
            "quant_method": "mxfp4",
            "modules_to_not_convert": ["self_attn", "lm_head"],
        },
    }
    oss_geom = rf.from_hf_config(gpt_oss_cfg)
    oss_dt = rf.dtypes_from_hf_config(gpt_oss_cfg)
    oss_disagg = rf.solve_serving_plan(
        oss_geom, oss_dt, chip, scen, mode="disagg", max_chips=7
    )
    assert oss_disagg.recommended is not None
    assert oss_disagg.recommended.total_chips <= 7
    assert any(
        p.prefill_pool is not None
        and (
            p.prefill_pool.instances != p.decode_pool.instances
            or p.prefill_pool.topology != p.decode_pool.topology
        )
        for p in oss_disagg.frontier
    )

    infeasible_scen = rf.TrafficScenario(
        name="custom", isl=8192, osl=1024, concurrencies=(1, 4), tpot_slo_ms=0.05
    )
    inf_plan = rf.solve_serving_plan(
        q35_geom, rf.Dtypes(kv=1.0), chip, infeasible_scen, mode="mixed", max_chips=4
    )
    assert any("INFEASIBLE_SLO" in w for w in inf_plan.warnings)

    cfg_path = tmp_path / "q35_config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc = cli.main(
        [
            "solve",
            "--model",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--scenario",
            "chat-8k1k",
            "--mode",
            "auto",
            "--max-chips",
            "8",
            "--slo-tier",
            "interactive",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended"] is not None
