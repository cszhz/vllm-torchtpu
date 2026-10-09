# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the last-stage, forward-only MTP diagnostic."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.qwen3_5_mtp_prefill import Qwen3_5MoeMTPPrefill
from vllm_torchtpu.spec_decode.prefill_only import (
    validate_prefill_only_config,
    validate_prefill_span,
)


def config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            tensor_parallel_size=1,
            prefill_context_parallel_size=8,
            decode_context_parallel_size=1,
            data_parallel_size=1,
            nnodes=2,
        ),
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            enable_chunked_prefill=False,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        kv_transfer_config=None,
    )


def test_two_host_pp2_pcp8_admitted():
    validate_prefill_only_config(config())


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 2),
        ("data_parallel_size", 2),
        ("decode_context_parallel_size", 2),
    ],
)
def test_unsupported_parallelism_rejected(field, value):
    c = config()
    setattr(c.parallel_config, field, value)
    with pytest.raises(NotImplementedError, match="TP=DCP=DP=1"):
        validate_prefill_only_config(c)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("scheduler_config", "async_scheduling", True),
        ("scheduler_config", "enable_chunked_prefill", True),
        ("cache_config", "enable_prefix_caching", True),
    ],
)
def test_rejects_unimplemented_lifecycles(section, field, value):
    c = config()
    setattr(getattr(c, section), field, value)
    with pytest.raises(NotImplementedError):
        validate_prefill_only_config(c)


@pytest.mark.parametrize(
    "computed,scheduled,prompt",
    [
        (0, 3, 4),
        (1, 3, 4),
        (4, 1, 4),
        (0, 5, 4),
    ],
)
def test_rejects_partial_prompt_cache_hit_or_decode(computed, scheduled, prompt):
    with pytest.raises(NotImplementedError, match="complete fresh prompt"):
        validate_prefill_span(computed, scheduled, prompt)


def test_complete_prompt_span():
    validate_prefill_span(0, 4, 4)


class Layer(torch.nn.Module):
    use_attn_reduce_scatter_for_moe = False

    def forward(self, positions, hidden_states, residual):
        assert residual is None
        self.positions = positions
        return hidden_states * 2, hidden_states


class Norm(torch.nn.Module):
    def forward(self, x, residual):
        return x + residual, None


def make_head():
    head = Qwen3_5MoeMTPPrefill.__new__(Qwen3_5MoeMTPPrefill)
    torch.nn.Module.__init__(head)
    model = torch.nn.Module()
    model.embed_tokens = torch.nn.Embedding(7, 3)
    model.embed_input_ids = model.embed_tokens
    model.pre_fc_norm_embedding = torch.nn.Identity()
    model.pre_fc_norm_hidden = torch.nn.Identity()
    model.fc = torch.nn.Linear(6, 3, bias=False)
    model.layers = torch.nn.ModuleList([Layer()])
    model.num_mtp_layers = 1
    model.norm = Norm()
    head.model = model
    return head


def test_whole_head_projects_every_prompt_row_without_first_rank_branch():
    head = make_head()
    ids = torch.tensor([1, 2, 3, 4])
    positions = torch.arange(4)
    hidden = torch.randn(4, 3)
    expected = head.model.fc(torch.cat([head.model.embed_tokens(ids), hidden], -1)) * 3
    result = Qwen3_5MoeMTPPrefill.forward(head, ids, positions, hidden)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(head.model.layers[0].positions, positions)
    assert result.shape == (4, 3)


def test_prefill_head_rejects_loop_step():
    head = make_head()
    with pytest.raises(NotImplementedError, match="step-0"):
        Qwen3_5MoeMTPPrefill.forward(
            head,
            torch.tensor([1]),
            torch.tensor([0]),
            torch.zeros(1, 3),
            spec_step_idx=1,
        )


@pytest.mark.parametrize("partial", [False, True])
def test_prefill_executes_head_without_producing_draft_tokens(partial):
    import numpy as np

    from vllm_torchtpu.layers.core.sequence_layout import (
        PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
        AllSequenceLayoutPlanner,
        SequenceLayoutDescriptor,
        SequenceLayoutKind,
        SequenceLayoutPlan,
    )
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    head = make_head()
    proposer = Eagle3Proposer.__new__(Eagle3Proposer)
    proposer.draft_model = head
    hidden = torch.randn(4, 3)
    ids = torch.tensor([2, 3, 4, 5])
    positions = torch.arange(4)
    chunk = SimpleNamespace(
        hidden_states=hidden,
        query_start_loc_np=np.array([0, 4]),
        sequence_layout_plan=AllSequenceLayoutPlanner().prepare_dummy(
            num_tokens=4,
            num_reqs=1,
            kv_cache_initialized=True,
        ),
    )
    if partial:
        # This worker owns logical rows 0, 2, 4; its fourth row is padding.
        chunk.query_start_loc_np = np.array([0, 5])
        chunk.sequence_layout_plan = SequenceLayoutPlan(
            descriptor=SequenceLayoutDescriptor(
                kind=SequenceLayoutKind.PARTIAL,
                protocol=PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
            ),
            token_slice=slice(0, 4),
            global_num_tokens=5,
            global_padded_num_tokens=8,
            local_num_tokens=3,
            local_padded_num_tokens=4,
            _packed_to_request_major_token_indices=np.array(
                [0, 2, 4, -1, 1, 3, -1, -1]
            ),
            _request_major_to_packed_token_indices=np.array([0, 4, 1, 5, 2]),
        )
    proposer.draft_chunks = [chunk]
    proposer._prepare_draft_inputs = lambda *a, **kw: (
        ids,
        positions,
        torch.tensor([3]),
        np.array([0]),
    )

    def forward(chunk, input_ids, positions, target_hidden_states, **kwargs):
        assert kwargs["step_idx"] == 0
        assert kwargs["seq_lens_delta"] == 0
        output = Qwen3_5MoeMTPPrefill.forward(
            head,
            input_ids,
            positions,
            target_hidden_states,
        )
        return output, output

    proposer._forward_draft = forward
    proposer.prefill_only(SimpleNamespace(), torch.tensor([5]))
    assert len(proposer.prefill_snapshots) == 1
    snapshot = proposer.prefill_snapshots[0]
    valid_rows = 3 if partial else 4
    logical = torch.tensor([0, 2, 4]) if partial else positions
    torch.testing.assert_close(snapshot["input_ids"], ids[:valid_rows])
    torch.testing.assert_close(snapshot["logical_token_indices"], logical)
    torch.testing.assert_close(snapshot["target_hidden"], hidden[:valid_rows])
    expected = Qwen3_5MoeMTPPrefill.forward(head, ids, positions, hidden)
    torch.testing.assert_close(snapshot["mtp_hidden"], expected[:valid_rows])


@pytest.mark.parametrize("last", [False, True])
def test_only_final_stage_initializes_mtp(monkeypatch, last):
    from vllm_torchtpu.runner import tpu_runner

    monkeypatch.setenv("TPU_MTP_PREFILL_ONLY", "1")
    monkeypatch.setattr(tpu_runner, "pp_rank_flags", lambda _: (not last, last))
    created = []

    def make_proposer(runner, config):
        created.append(runner)
        return "draft"

    monkeypatch.setattr(tpu_runner, "Eagle3Proposer", make_proposer)
    runner = SimpleNamespace(parallel_config=object(), vllm_config=object())
    tpu_runner.TPUModelRunner._init_speculative_decoding(runner)
    assert len(created) == int(last)
    assert runner.drafter == ("draft" if last else None)
    assert runner.rejection_sampler is None
