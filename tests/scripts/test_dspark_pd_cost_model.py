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
"""Unit tests for the remote DSpark P/D cost model."""

import importlib.util
import pathlib
import sys

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "benchmarking"
    / "dspark_pd_cost_model.py"
)
_spec = importlib.util.spec_from_file_location("dspark_pd_cost_model", _SCRIPT)
dspark_pd_cost_model = importlib.util.module_from_spec(_spec)
sys.modules["dspark_pd_cost_model"] = dspark_pd_cost_model
_spec.loader.exec_module(dspark_pd_cost_model)


def _inputs(**overrides):
    values = {
        "context_update_ms": 1.0,
        "draft_generate_ms": 2.0,
        "verify_ms": 4.0,
        "batch_size": 2,
        "proposal_tokens_per_request": 7,
        "feedback_tokens_per_request": 7,
        "feedback_layer_count": 5,
        "hidden_size": 16,
        "feedback_dtype_bytes": 2,
        "token_id_bytes": 4,
        "metadata_bytes_per_request": 16,
        "p2d_latency_us": 0.0,
        "p2d_bandwidth_gbps": 8.0,
        "d2p_latency_us": 0.0,
        "d2p_bandwidth_gbps": 8.0,
        "mean_accepted_tokens": 3.5,
    }
    values.update(overrides)
    return dspark_pd_cost_model.Inputs(**values)


def test_evaluate_derives_payloads_and_pipeline_balance():
    result = dspark_pd_cost_model.evaluate(_inputs())

    assert result["payload_bytes"] == {
        "p_to_d": 2 * 7 * 4 + 2 * 16,
        "d_to_p": 2 * 7 * 5 * 16 * 2 + 2 * 16,
    }
    assert result["stage_ms"]["p"] == pytest.approx(3.000088)
    assert result["stage_ms"]["d"] == pytest.approx(4.002272)
    assert result["stage_ms"]["pipeline_cycle"] == result["stage_ms"]["d"]
    assert result["balance"]["bottleneck"] == "d"
    assert result["performance"]["accepted_tokens_per_microbatch"] == 7


def test_same_request_latency_is_not_reported_as_pipeline_latency():
    result = dspark_pd_cost_model.evaluate(_inputs())

    assert result["assumptions"]["same_request_rounds_overlap"] is False
    assert result["stage_ms"]["split_single_request_round"] == pytest.approx(
        result["stage_ms"]["p"] + result["stage_ms"]["d"]
    )
    assert (
        result["performance"]["split_single_request_tpot_ms"]
        > result["performance"]["pipeline_tpot_ms_per_request"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("batch_size", 0), ("d2p_bandwidth_gbps", 0.0), ("verify_ms", -0.1)],
)
def test_evaluate_rejects_invalid_inputs(field, value):
    with pytest.raises(ValueError, match=field):
        dspark_pd_cost_model.evaluate(_inputs(**{field: value}))
