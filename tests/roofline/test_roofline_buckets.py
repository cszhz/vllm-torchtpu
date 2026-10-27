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
"""Unit tests for scenario-driven bucket step construction and regime scoring."""

from __future__ import annotations

from dataclasses import replace

import pytest

import vllm_torchtpu.roofline as rf

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


def test_bucket_step_uses_scenario_mix() -> None:
    coding = rf.SCENARIO_PRESETS["coding-agent"]
    chat = rf.SCENARIO_PRESETS["chat-8k1k"]
    k = 3

    for t_tok in (*range(1, k + 2), *rf.DEFAULT_VLLM_BUCKETS):
        sh = rf.bucket_step_shape(coding, k, t_tok, acceptance_length=3.39)
        assert sh.rows_per_seq == k + 1
        assert sh.batch * (k + 1) + sh.prefill_tokens == t_tok
        assert sh.prefix_hit_rate == pytest.approx(coding.prefix_hit_rate)
        assert sh.context == int(round(coding.isl + coding.osl / 2.0))
        if t_tok < k + 1:
            assert sh.batch == 0
            assert sh.prefill_tokens == t_tok

    sh_coding_256 = rf.bucket_step_shape(coding, k, 256, acceptance_length=3.39)
    sh_chat_256 = rf.bucket_step_shape(chat, k, 256, acceptance_length=1.0)
    assert sh_coding_256.prefill_tokens < sh_chat_256.prefill_tokens
    assert sh_coding_256.batch > sh_chat_256.batch

    geom = rf.from_hf_config(QWEN35_CFG)
    chip = rf.CHIPS["tpu7x"]
    topo = rf.TOPOLOGIES["2x2x1"]
    dt = rf.Dtypes(w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0)
    par = rf.Parallel(tp=8, moe="ep", ep=8)
    plan_coding = rf.recommend_bucket_regimes(
        geom,
        dt,
        chip,
        topo,
        par,
        scenario=coding,
        spec_window=k,
        acceptance_length=3.39,
    )
    plan_chat = rf.recommend_bucket_regimes(
        geom,
        dt,
        chip,
        topo,
        par,
        scenario=chat,
        spec_window=0,
        acceptance_length=1.0,
    )
    assert plan_coding.rows[-1].serial_floor_ms != pytest.approx(
        plan_chat.rows[-1].serial_floor_ms
    )


def test_bucket_regimes_sp_shared_expert_and_megakernel_cap() -> None:
    geom = rf.from_hf_config(QWEN35_CFG)
    chip = rf.CHIPS["tpu7x"]
    topo = rf.TOPOLOGIES["2x2x1"]
    dt = rf.Dtypes(w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0)
    par = rf.Parallel(tp=8, moe="ep", ep=8)

    sf_rep_1 = rf.step_floor(
        geom,
        chip,
        replace(par, residual="replicated"),
        rf.StepShape(batch=1, context=8192),
        dt,
        topology=topo,
    )
    sf_sp_1 = rf.step_floor(
        geom,
        chip,
        replace(par, residual="sp"),
        rf.StepShape(batch=1, context=8192),
        dt,
        topology=topo,
    )
    moe_rep = next(m for m in sf_rep_1.modules if m.name == "moe")
    moe_sp = next(m for m in sf_sp_1.modules if m.name == "moe")
    assert (moe_sp.hbm_bytes - moe_rep.hbm_bytes) == pytest.approx(
        660_602_880.0, rel=1e-6
    )
    assert sf_rep_1.serial_floor_s < sf_sp_1.serial_floor_s

    buckets = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    bplan = rf.recommend_bucket_regimes(
        geom, dt, chip, topo, par, buckets=buckets, context=8192
    )
    assert len(bplan.rows) == len(buckets)
    assert 2 <= len(bplan.bands) <= 3

    # Megakernel is measure-only in roofline_auto; small bucket stays
    # replicated + allreduce.
    row_1 = bplan.rows[0]
    assert row_1.bucket_tokens == 1
    assert row_1.residual == "replicated"
    assert row_1.moe_dispatch == "allreduce"
    assert row_1.megakernel is False
    assert row_1.scratchpad_bytes <= chip.vmem_bytes
    assert row_1.source == "roofline_auto"

    row_2048 = bplan.rows[-1]
    assert row_2048.bucket_tokens == 2048
    assert row_2048.megakernel is False
    assert row_2048.scratchpad_bytes > chip.vmem_bytes
    assert row_2048.residual == "sp"
    assert row_2048.moe_dispatch == "a2a_push"
    assert bplan.sp_min_tokens is not None and bplan.sp_min_tokens > 1
    assert bplan.fused_ep_min_tokens is not None and bplan.fused_ep_min_tokens > 1
    assert bplan.megakernel_max_tokens == 0

    sp_seq = [1 if r.residual == "sp" else 0 for r in bplan.rows]
    ep_seq = [1 if r.moe_dispatch != "allreduce" else 0 for r in bplan.rows]
    assert sp_seq == sorted(sp_seq)
    assert ep_seq == sorted(ep_seq)

    def_plan = rf.recommend_bucket_regimes(geom, dt, chip, topo, buckets=(1, 64, 512))
    assert all(r.residual == "replicated" for r in def_plan.rows)
    with pytest.raises(ValueError, match="moe='ep'"):
        rf.validate_zero_reshard_regimes(
            geom,
            dt,
            chip,
            rf.Parallel(tp=8, moe="tp_slice"),
            [
                {
                    "min_tokens": 1,
                    "max_tokens": 16,
                    "residual": "sp",
                    "moe_dispatch": "allreduce",
                    "megakernel": False,
                }
            ],
        )

    ov_plan = rf.recommend_bucket_regimes(
        geom,
        dt,
        chip,
        topo,
        par,
        buckets=(1, 16, 64, 512),
        context=8192,
        overrides="1..16:replicated,allreduce,mk=1;64..512:sp,a2a_push,mk=0",
    )
    assert all(r.source == "explicit_override" for r in ov_plan.rows)
    assert (
        ov_plan.rows[0].residual == "replicated" and ov_plan.rows[0].megakernel is True
    )
    assert ov_plan.megakernel_max_tokens == 16
    assert (
        ov_plan.rows[-1].residual == "sp"
        and ov_plan.rows[-1].moe_dispatch == "a2a_push"
    )

    with pytest.raises(ValueError, match="reshard"):
        rf.validate_zero_reshard_regimes(
            geom,
            dt,
            chip,
            par,
            [
                {
                    "min_tokens": 1,
                    "max_tokens": 16,
                    "tp": 4,
                    "residual": "replicated",
                    "moe_dispatch": "allreduce",
                    "megakernel": False,
                },
                {
                    "min_tokens": 32,
                    "max_tokens": 512,
                    "tp": 8,
                    "residual": "sp",
                    "moe_dispatch": "a2a_push",
                    "megakernel": False,
                },
            ],
        )
    with pytest.raises(ValueError, match="dtype"):
        rf.validate_zero_reshard_regimes(
            geom,
            dt,
            chip,
            par,
            [
                {
                    "min_tokens": 1,
                    "max_tokens": 16,
                    "w_expert": 0.5625,
                    "residual": "replicated",
                    "moe_dispatch": "allreduce",
                    "megakernel": False,
                },
            ],
        )
    with pytest.raises(ValueError, match="tp > 1"):
        rf.validate_zero_reshard_regimes(
            geom,
            dt,
            chip,
            rf.Parallel(tp=1, moe="tp_slice"),
            [
                {
                    "min_tokens": 1,
                    "max_tokens": 16,
                    "residual": "sp",
                    "moe_dispatch": "allreduce",
                    "megakernel": False,
                }
            ],
        )
    with pytest.raises(ValueError, match="scratchpad"):
        rf.validate_zero_reshard_regimes(
            geom,
            dt,
            chip,
            par,
            [
                {
                    "min_tokens": 512,
                    "max_tokens": 2048,
                    "residual": "replicated",
                    "moe_dispatch": "allreduce",
                    "megakernel": True,
                }
            ],
        )
