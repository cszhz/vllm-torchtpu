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
"""Unit tests for TPU_DRAFT_STEP0_IN_LOOP and TPU_FOLD_SPEC_BOOKKEEPING."""

import contextlib
import types
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.spec_decode.mtp_fused_loop import (
    _MtpFullDraftLoop,
    _MtpFusedLoop,
)


class _DummyMtpDraft(nn.Module):
    """Deterministic toy MTP draft model for testing step-0 + loop fusion."""

    def __init__(self, hidden_size: int = 4, vocab_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seen_step_metadata: list[
            tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.forward_context import get_forward_context

        try:
            ctx = get_forward_context()
            attn_md = getattr(ctx, "attn_metadata", None)
            if isinstance(attn_md, dict) and "layer.0" in attn_md:
                md0 = attn_md["layer.0"]
                self.seen_step_metadata.append(
                    (
                        spec_step_idx,
                        md0.seq_lens.clone(),
                        md0.query_start_loc.clone(),
                        md0.request_distribution.clone(),
                    )
                )
        except AssertionError:
            pass
        pos_1d = positions[0] if positions.ndim == 2 else positions
        step_offset = (
            input_ids.to(torch.float32) * 3.0
            + pos_1d.to(torch.float32) * 2.0
            + float(spec_step_idx + 1)
        ).unsqueeze(-1)
        new_hidden = hidden_states + step_offset
        return new_hidden, new_hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = hidden_states.sum(dim=-1).to(torch.int64).abs() % self.vocab_size
        logits = torch.zeros(
            (hidden_states.shape[0], self.vocab_size), dtype=torch.float32
        )
        logits.scatter_(1, scores.unsqueeze(1), 1.0)
        return logits


def test_draft_step0_in_loop_env_default_false() -> None:
    """TPU_DRAFT_STEP0_IN_LOOP defaults to False."""
    assert envs.TPU_DRAFT_STEP0_IN_LOOP is False


def test_mtp_full_draft_loop_matches_separate_step0_and_fused_loop() -> None:
    """Verify `_MtpFullDraftLoop` produces bit-identical `[p, K]` draft tokens
    to running step 0 forward + gather + step 0 propose + `_MtpFusedLoop`
    (steps 1..K-1) separately, and restores attention metadata on exit."""
    draft_model = _DummyMtpDraft(hidden_size=4, vocab_size=32)
    num_reqs = 2
    spec_k = 3
    query_len = 1 + spec_k
    total_tokens = num_reqs * query_len

    step0_input_ids = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23], dtype=torch.int32)
    step0_positions = torch.tensor([5, 6, 7, 8, 15, 16, 17, 18], dtype=torch.int32)
    target_hidden = torch.arange(total_tokens * 4, dtype=torch.float32).reshape(
        total_tokens, 4
    )
    # Req 0 accepted 2 drafts (slot 2), req 1 accepted 1 draft (slot 5)
    step0_gather_indices = torch.tensor([2, 5], dtype=torch.int64)
    step0_seq_lens = torch.tensor([9, 19], dtype=torch.int32)
    step0_qsl = torch.tensor([0, 4, 8], dtype=torch.int32)
    step0_rd = torch.tensor([0, 0, 2], dtype=torch.int32)
    num_rejected_dev = torch.tensor([1, 2], dtype=torch.int32)
    loop_qsl = torch.tensor([0, 1, 2], dtype=torch.int32)
    loop_rd = torch.tensor([2, 2, 2], dtype=torch.int32)

    orig_sl = torch.tensor([99, 99], dtype=torch.int32)
    orig_qsl = torch.tensor([0, 0, 0], dtype=torch.int32)
    orig_rd = torch.tensor([0, 0, 0], dtype=torch.int32)
    md_obj = types.SimpleNamespace(
        seq_lens=orig_sl,
        query_start_loc=orig_qsl,
        request_distribution=orig_rd,
    )
    fake_ctx = types.SimpleNamespace(attn_metadata={"layer.0": md_obj})

    full_loop = _MtpFullDraftLoop(
        vllm_config=cast(Any, types.SimpleNamespace()),
        prefix="test_full",
        mtp=draft_model,
        num_steps=spec_k - 1,
        advance_draft_positions=True,
    )
    with patch("vllm.forward_context.get_forward_context", return_value=fake_ctx):
        full_drafts = full_loop(
            step0_input_ids=step0_input_ids,
            step0_positions=step0_positions,
            target_hidden_states=target_hidden,
            step0_gather_indices=step0_gather_indices,
            step0_seq_lens=step0_seq_lens.clone(),
            step0_query_start_loc=step0_qsl,
            step0_request_distribution=step0_rd,
            num_rejected_dev=num_rejected_dev,
            loop_query_start_loc=loop_qsl,
            loop_request_distribution=loop_rd,
        )

    assert md_obj.seq_lens is orig_sl
    assert md_obj.query_start_loc is orig_qsl
    assert md_obj.request_distribution is orig_rd
    assert len(draft_model.seen_step_metadata) == spec_k
    assert torch.equal(draft_model.seen_step_metadata[0][1], step0_seq_lens)
    assert torch.equal(draft_model.seen_step_metadata[0][2], step0_qsl)
    assert torch.equal(draft_model.seen_step_metadata[0][3], step0_rd)
    assert torch.equal(
        draft_model.seen_step_metadata[1][1],
        step0_seq_lens + 1 - num_rejected_dev,
    )
    assert torch.equal(draft_model.seen_step_metadata[1][2], loop_qsl)
    assert torch.equal(draft_model.seen_step_metadata[1][3], loop_rd)
    assert torch.equal(
        draft_model.seen_step_metadata[2][1],
        step0_seq_lens + 2 - num_rejected_dev,
    )

    # Reference path: step 0 forward -> gather -> propose tok0 -> _MtpFusedLoop
    last0, hid0 = draft_model(step0_input_ids, step0_positions, target_hidden)
    last_carry = torch.index_select(last0, 0, step0_gather_indices)
    hid_carry = torch.index_select(hid0, 0, step0_gather_indices)
    pos_carry = torch.index_select(step0_positions, 0, step0_gather_indices)
    tok0 = draft_model.compute_logits(last_carry).argmax(dim=-1).to(torch.int32)

    fused_loop = _MtpFusedLoop(
        vllm_config=cast(Any, types.SimpleNamespace()),
        prefix="test_fused",
        mtp=draft_model,
        num_steps=spec_k - 1,
        advance_draft_positions=True,
    )
    ref_seq_lens_step1 = step0_seq_lens + 1 - num_rejected_dev
    loop_toks = fused_loop(
        input_ids=tok0,
        positions=pos_carry + 1,
        hidden_states=hid_carry,
        seq_lens=ref_seq_lens_step1,
    )
    ref_drafts = torch.stack([tok0, *loop_toks], dim=1)
    assert isinstance(full_drafts, torch.Tensor)
    assert torch.equal(full_drafts, ref_drafts)


class _FakeWarmCompiledProgram:
    def __init__(self, output_tensor: torch.Tensor) -> None:
        self.first_compile = True
        self.call_count = 0
        self.seen_seq_lens: list[torch.Tensor] = []
        self._output = output_tensor

    def __call__(
        self,
        _step0_input_ids: torch.Tensor,
        _step0_positions: torch.Tensor,
        _target_hidden_states: torch.Tensor,
        _step0_gather_indices: torch.Tensor,
        step0_seq_lens: torch.Tensor,
        *_rest: Any,
    ) -> torch.Tensor:
        self.call_count += 1
        self.seen_seq_lens.append(step0_seq_lens.clone())
        step0_seq_lens.add_(100)
        return self._output


def test_run_mtp_full_draft_loop_executes_once_when_warm_even_if_first_compile_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_torchtpu.spec_decode.eagle3 as eagle3_mod
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    monkeypatch.setattr(
        eagle3_mod, "set_model_tag", lambda *_a, **_kw: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        eagle3_mod, "set_forward_context", lambda *_a, **_kw: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        eagle3_mod,
        "set_vllm_model_wrapper_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )

    proposer = object.__new__(Eagle3Proposer)
    expected_out = torch.tensor([[11, 12, 13]], dtype=torch.int32)
    fake_program = _FakeWarmCompiledProgram(expected_out)
    proposer._mtp_full_draft_loop = cast(Any, fake_program)
    proposer._mtp_full_draft_loop_warm = False
    proposer._zero_rej_dev_cache = {}
    proposer.draft_vllm_config = cast(Any, types.SimpleNamespace())
    proposer.runner = cast(
        Any,
        types.SimpleNamespace(
            device=torch.device("cpu"),
            mesh=None,
            num_tokens_paddings=[16, 32],
            _dp_num_tokens_across_dp=lambda _n: None,
        ),
    )
    md_obj = types.SimpleNamespace(
        seq_lens=torch.tensor([10, 0, 0, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 4, 4, 4], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    proposer._build_draft_attn_metadata = cast(Any, lambda **_kw: {"layer.0": md_obj})
    chunk = cast(Any, types.SimpleNamespace(num_reqs=1))

    # First call when cold runs warm-up + actual call (2 invocations),
    # and preserves step0_seq_lens.
    out1 = proposer._run_mtp_full_draft_loop(
        chunk=chunk,
        step0_input_ids=torch.zeros(4, dtype=torch.int32),
        step0_positions=torch.zeros(4, dtype=torch.int32),
        target_hidden_states=torch.zeros((4, 8), dtype=torch.float32),
        step0_gather_indices=torch.zeros(4, dtype=torch.int64),
        num_rejected_np=None,
        num_rejected_dev=torch.zeros(4, dtype=torch.int32),
        loop_query_start_loc=torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
        loop_request_distribution=torch.tensor([1, 1, 1], dtype=torch.int32),
    )
    assert torch.equal(cast(torch.Tensor, out1), expected_out)
    assert fake_program.call_count == 2
    assert proposer._mtp_full_draft_loop_warm is True
    assert torch.equal(
        fake_program.seen_seq_lens[0],
        torch.tensor([10, 0, 0, 0], dtype=torch.int32),
    )
    assert torch.equal(
        fake_program.seen_seq_lens[1],
        torch.tensor([10, 0, 0, 0], dtype=torch.int32),
    )

    # Reset seq_lens for step 2: even though fake_program.first_compile is STILL True,
    # _mtp_full_draft_loop_warm is True so program must only run ONCE!
    md_obj.seq_lens = torch.tensor([14, 0, 0, 0], dtype=torch.int32)
    out2 = proposer._run_mtp_full_draft_loop(
        chunk=chunk,
        step0_input_ids=torch.zeros(4, dtype=torch.int32),
        step0_positions=torch.zeros(4, dtype=torch.int32),
        target_hidden_states=torch.zeros((4, 8), dtype=torch.float32),
        step0_gather_indices=torch.zeros(4, dtype=torch.int64),
        num_rejected_np=None,
        num_rejected_dev=torch.zeros(4, dtype=torch.int32),
        loop_query_start_loc=torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
        loop_request_distribution=torch.tensor([1, 1, 1], dtype=torch.int32),
    )
    assert torch.equal(cast(torch.Tensor, out2), expected_out)
    assert fake_program.call_count == 3


def test_prepare_draft_inputs_repeats_last_token_index_to_loop_bucket() -> None:
    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    proposer = object.__new__(Eagle3Proposer)
    proposer._loop_bucket = cast(Any, lambda _nr: 16)
    proposer._draft_seed_input_ids = cast(Any, lambda ids, _lti, _nt: ids)
    proposer.runner = cast(
        Any,
        types.SimpleNamespace(
            device=torch.device("cpu"),
            input_batch=types.SimpleNamespace(req_ids=["r0", "r1", "r2", "r3"]),
        ),
    )

    num_reqs = 4
    padded_width = 8
    # 4 requests x 4 tokens = 16 tokens
    chunk_input_ids = torch.arange(100, 116, dtype=torch.int32)
    chunk_positions = torch.arange(16, dtype=torch.int64)
    query_start_loc_np = np.array([0, 4, 8, 12, 16], dtype=np.int32)
    # Next tokens from rejection sampler padded to width=8 (< p=16).
    next_tokens_dev = torch.full((padded_width, 4), -1, dtype=torch.int32)
    # Req 0: 2 (idx 1), Req 1: 4 (idx 7), Req 2: 1 (idx 8), Req 3: 3 (idx 14).
    next_tokens_dev[0, :2] = torch.tensor([201, 202], dtype=torch.int32)
    next_tokens_dev[1, :4] = torch.tensor([301, 302, 303, 304], dtype=torch.int32)
    next_tokens_dev[2, :1] = torch.tensor([401], dtype=torch.int32)
    next_tokens_dev[3, :3] = torch.tensor([501, 502, 503], dtype=torch.int32)

    layout_plan = types.SimpleNamespace(
        kind=SequenceLayoutKind.ALL,
        localize_token_tensor_and_gather_indices=lambda ids, gi, num_valid_gathers: (
            ids,
            gi,
        ),
    )
    chunk = cast(
        Any,
        types.SimpleNamespace(
            num_reqs=num_reqs,
            start_index=0,
            req_ids=["r0", "r1", "r2", "r3"],
            input_ids=chunk_input_ids,
            position_ids=chunk_positions,
            query_start_loc_np=query_start_loc_np,
            draft_lengths=torch.tensor([3, 3, 3, 3, 0, 0, 0, 0], dtype=torch.int32),
            attn_ctx=types.SimpleNamespace(
                query_start_loc=torch.tensor(
                    [0, 4, 8, 12, 16, 16, 16, 16, 16], dtype=torch.int32
                )
            ),
            sequence_layout_plan=layout_plan,
            _precomputed_draft_prep=None,
        ),
    )
    _draft_ids, _pos, gather_indices, _num_rej = proposer._prepare_draft_inputs(
        chunk=chunk,
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=None,
        next_tokens_device=next_tokens_dev,
        pad_gather_with_last=True,
    )
    assert gather_indices.shape == (16,)
    assert gather_indices[:4].tolist() == [1, 7, 8, 14]
    # Padded entries [4..15] must repeat the last valid request's index (14), NOT 0!
    assert gather_indices[4:].tolist() == [14] * 12


_FAKE_FORWARD_CTX = types.SimpleNamespace(attn_metadata=None)


@contextlib.contextmanager
def _fake_set_forward_context(attn_metadata: Any, *_args: Any, **_kwargs: Any) -> Any:
    prev_md = _FAKE_FORWARD_CTX.attn_metadata
    _FAKE_FORWARD_CTX.attn_metadata = attn_metadata
    try:
        yield _FAKE_FORWARD_CTX
    finally:
        _FAKE_FORWARD_CTX.attn_metadata = prev_md


def _fake_get_forward_context() -> Any:
    return _FAKE_FORWARD_CTX


class _FirstCallGatherWidthSpecializer(nn.Module):
    """Simulates PiecewiseBackend.create_concrete_args static specialization:
    records `step0_gather_indices.shape[0]` on the very first call (warmup at
    B=1) and slices `step0_gather_indices[:specialized_width]` on all later
    calls (e.g. decode at B=8)."""

    def __init__(self, inner_loop: _MtpFullDraftLoop) -> None:
        super().__init__()
        self.inner_loop = inner_loop
        self.specialized_gather_width: int | None = None

    def forward(
        self,
        step0_input_ids: torch.Tensor,
        step0_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        step0_gather_indices: torch.Tensor,
        step0_seq_lens: torch.Tensor,
        step0_query_start_loc: torch.Tensor,
        step0_request_distribution: torch.Tensor,
        num_rejected_dev: torch.Tensor,
        loop_query_start_loc: torch.Tensor,
        loop_request_distribution: torch.Tensor,
    ) -> torch.Tensor:
        cur_width = int(step0_gather_indices.shape[0])
        if self.specialized_gather_width is None:
            self.specialized_gather_width = cur_width
        effective_gather = step0_gather_indices[: self.specialized_gather_width]
        return self.inner_loop(
            step0_input_ids=step0_input_ids,
            step0_positions=step0_positions,
            target_hidden_states=target_hidden_states,
            step0_gather_indices=effective_gather,
            step0_seq_lens=step0_seq_lens,
            step0_query_start_loc=step0_query_start_loc,
            step0_request_distribution=step0_request_distribution,
            num_rejected_dev=num_rejected_dev,
            loop_query_start_loc=loop_query_start_loc,
            loop_request_distribution=loop_request_distribution,
        )


def _make_b8_test_proposer(
    draft_model: _DummyMtpDraft,
    max_num_reqs: int,
    spec_k: int,
    step0_in_loop: bool,
) -> tuple[Any, _FirstCallGatherWidthSpecializer | None]:
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    proposer = object.__new__(Eagle3Proposer)
    proposer.speculative_config = cast(
        Any,
        types.SimpleNamespace(
            method="mtp",
            num_speculative_tokens=spec_k,
        ),
    )
    proposer.draft_vllm_config = cast(
        Any,
        types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                data_parallel_size=1,
                use_sequence_parallel_moe=False,
                is_moe_model=False,
                pipeline_parallel_size=1,
            ),
            compilation_config=types.SimpleNamespace(
                static_forward_context={},
            ),
            model_config=None,
            speculative_config=None,
            kv_transfer_config=None,
        ),
    )
    proposer.draft_model = draft_model
    proposer._share_mtp_indices = False
    proposer._draft_kv_shared = False
    proposer._draft_replicated = False
    proposer._draft_has_moe_cache = False
    proposer._draft_attn_layer_names = {"layer.0"}
    proposer._draft_md_cache = {}
    proposer._zero_rej_dev_cache = {}
    proposer._loop_qsl_reqdist_cache = {}
    proposer._mtp_loop_model = None
    proposer._mtp_loop_model_warm = True
    proposer._draft_moe_ctx = contextlib.nullcontext
    proposer.runner = cast(
        Any,
        types.SimpleNamespace(
            device=torch.device("cpu"),
            mesh=None,
            uses_mrope=False,
            max_num_reqs=max_num_reqs,
            num_reqs_max_model_len=max_num_reqs,
            num_reqs_most_model_len=max_num_reqs,
            num_tokens_paddings=[4, 8, 16, 32, 64],
            _dp_lockstep_enabled=lambda: False,
            _dp_num_tokens_across_dp=lambda _num_tok: None,
            _has_mamba_state=False,
            _unified_kv_layout=False,
            requests={},
            input_batch=types.SimpleNamespace(
                num_reqs=max_num_reqs,
                req_ids=[f"req_{req_i}" for req_i in range(max_num_reqs)],
            ),
        ),
    )
    fused_loop = _MtpFusedLoop(
        vllm_config=cast(Any, types.SimpleNamespace()),
        prefix="b8_fused",
        mtp=draft_model,
        num_steps=spec_k - 1,
        advance_draft_positions=True,
    )
    proposer._mtp_fused_loop = cast(Any, fused_loop)
    proposer._mtp_fused_loop_warm = True
    specializer: _FirstCallGatherWidthSpecializer | None = None
    if step0_in_loop:
        full_loop = _MtpFullDraftLoop(
            vllm_config=cast(Any, types.SimpleNamespace()),
            prefix="b8_full",
            mtp=draft_model,
            num_steps=spec_k - 1,
            advance_draft_positions=True,
        )
        specializer = _FirstCallGatherWidthSpecializer(full_loop)
        proposer._mtp_full_draft_loop = cast(Any, specializer)
        proposer._mtp_full_draft_loop_warm = False
    else:
        proposer._mtp_full_draft_loop = None
        proposer._mtp_full_draft_loop_warm = False
    return proposer, specializer


def _build_b8_chunk(
    num_reqs: int,
    max_num_reqs: int,
    token_counts_per_req: tuple[int, ...],
    padded_total_tokens: int,
    hidden_size: int,
    spec_k: int,
) -> Any:
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind

    qsl_list = [0]
    for count in token_counts_per_req[:num_reqs]:
        qsl_list.append(qsl_list[-1] + count)
    qsl_np = np.array(qsl_list, dtype=np.int32)
    padded_qsl_list = list(qsl_list) + [qsl_list[-1]] * (max_num_reqs - num_reqs)
    qsl_dev = torch.tensor(padded_qsl_list, dtype=torch.int32)

    seq_lens_list = [
        20 + req_i * 7 + token_counts_per_req[req_i] if req_i < num_reqs else 0
        for req_i in range(max_num_reqs)
    ]
    seq_lens_dev = torch.tensor(seq_lens_list, dtype=torch.int32)
    req_dist_dev = torch.tensor([0, 0, num_reqs], dtype=torch.int32)

    chunk_input_ids = torch.zeros(padded_total_tokens, dtype=torch.int32)
    chunk_positions = torch.zeros(padded_total_tokens, dtype=torch.int32)
    for req_i in range(num_reqs):
        tok_start = qsl_list[req_i]
        tok_count = token_counts_per_req[req_i]
        base_pos = 20 + req_i * 7
        for offset in range(tok_count):
            chunk_input_ids[tok_start + offset] = 100 + req_i * 10 + offset
            chunk_positions[tok_start + offset] = base_pos + offset

    target_hidden = (
        torch.arange(padded_total_tokens * hidden_size, dtype=torch.float32).reshape(
            padded_total_tokens, hidden_size
        )
        * 0.25
    )
    draft_lengths = torch.zeros(max_num_reqs, dtype=torch.int32)
    for req_i in range(num_reqs):
        draft_lengths[req_i] = max(0, token_counts_per_req[req_i] - 1)

    target_attn_md = AttentionMetadata(
        input_positions=cast(Any, chunk_positions),
        block_tables=cast(Any, torch.zeros((max_num_reqs, 4), dtype=torch.int32)),
        seq_lens=cast(Any, seq_lens_dev),
        query_start_loc=cast(Any, qsl_dev),
        request_distribution=cast(Any, req_dist_dev),
    )
    layout_plan = types.SimpleNamespace(
        kind=SequenceLayoutKind.ALL,
        localize_token_tensor_and_gather_indices=lambda ids, gi, num_valid_gathers: (
            ids,
            gi,
        ),
        aggregate_request_aligned_tensor=lambda tensor, _mask: tensor,
    )
    return types.SimpleNamespace(
        num_reqs=num_reqs,
        start_index=0,
        req_ids=[f"req_{req_i}" for req_i in range(num_reqs)],
        input_ids=chunk_input_ids,
        position_ids=chunk_positions,
        hidden_states=target_hidden,
        aux_hidden_states=[],
        query_start_loc_np=qsl_np,
        draft_lengths=draft_lengths,
        attn_metadata={"layer.0": target_attn_md},
        attn_ctx=types.SimpleNamespace(
            use_max_model_len=True,
            position_ids_override=None,
            seq_lens=seq_lens_dev,
            query_start_loc=qsl_dev,
            request_distribution=req_dist_dev,
        ),
        sequence_layout_plan=layout_plan,
        _precomputed_draft_prep=None,
    )


@pytest.mark.parametrize(
    ("token_counts_per_req", "rejected_per_req"),
    [
        ((4, 4, 4, 4, 4, 4, 4, 4), (0, 1, 2, 3, 0, 2, 1, 3)),
        ((4, 1, 4, 2, 4, 3, 4, 4), (1, 0, 2, 1, 3, 2, 0, 1)),
    ],
)
def test_propose_multi_request_b8_step0_in_loop_matches_separate_after_b1_warmup(
    monkeypatch: pytest.MonkeyPatch,
    token_counts_per_req: tuple[int, ...],
    rejected_per_req: tuple[int, ...],
) -> None:
    """Regression test for multi-request (`B = 8 > 4`) draft acceptance:
    when `_MtpFullDraftLoop` is first warmed/specialized at `num_reqs = 1`
    with `num_tokens_paddings = [4, 8, 16, 32, 64]`, `_loop_bucket` must keep
    `step0_gather_indices` at `max_loop_bucket = 8` (not `4`), so subsequent
    `num_reqs = 8` calls produce bit-identical `[8, K]` draft tokens to
    `TPU_DRAFT_STEP0_IN_LOOP=0` for both uniform (`4` tokens/req) and mixed
    token counts across sync and async (`next_tokens_per_chunk`) paths."""
    import vllm.forward_context as vllm_fwd_ctx

    import vllm_torchtpu.spec_decode.eagle3 as eagle3_mod
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    monkeypatch.setattr(
        eagle3_mod, "set_model_tag", lambda *_a, **_kw: contextlib.nullcontext()
    )
    monkeypatch.setattr(eagle3_mod, "set_forward_context", _fake_set_forward_context)
    monkeypatch.setattr(vllm_fwd_ctx, "get_forward_context", _fake_get_forward_context)
    monkeypatch.setattr(
        eagle3_mod,
        "set_vllm_model_wrapper_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    for method_name in (
        "_draft_seed_input_ids",
        "_draft_gather_carries",
        "_draft_propose_token",
    ):
        wrapped_fn = vars(Eagle3Proposer)[method_name].__wrapped__.__wrapped__
        monkeypatch.setattr(Eagle3Proposer, method_name, wrapped_fn)

    max_num_reqs = 8
    spec_k = 3
    hidden_size = 4
    draft_model = _DummyMtpDraft(hidden_size=hidden_size, vocab_size=64)

    proposer_ref, _ = _make_b8_test_proposer(
        draft_model=draft_model,
        max_num_reqs=max_num_reqs,
        spec_k=spec_k,
        step0_in_loop=False,
    )
    proposer_test, specializer = _make_b8_test_proposer(
        draft_model=draft_model,
        max_num_reqs=max_num_reqs,
        spec_k=spec_k,
        step0_in_loop=True,
    )
    assert specializer is not None

    # 1. Warmup proposer_test at B=1 (4 tokens), just like _warmup_spec_decode.
    monkeypatch.setattr(envs, "TPU_DRAFT_STEP0_IN_LOOP", True)
    proposer_test.runner.input_batch.num_reqs = 1
    warmup_chunk = _build_b8_chunk(
        num_reqs=1,
        max_num_reqs=max_num_reqs,
        token_counts_per_req=(4,),
        padded_total_tokens=4,
        hidden_size=hidden_size,
        spec_k=spec_k,
    )
    proposer_test.draft_chunks = [warmup_chunk]
    warmup_out = proposer_test.propose(
        sampled_token_ids=[[15]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=np.array([0], dtype=np.int32),
        scheduler_output=None,
        return_device=True,
    )
    assert isinstance(warmup_out, torch.Tensor)
    assert warmup_out.shape == (1, spec_k)
    # PiecewiseBackend specialization width captured at B=1 must be 8, NOT 4!
    assert specializer.specialized_gather_width == 8

    # 2. Run B=8 (num_reqs=8 > 4) on sync path: compare step0_in_loop=False vs True.
    proposer_ref.runner.input_batch.num_reqs = max_num_reqs
    proposer_test.runner.input_batch.num_reqs = max_num_reqs
    sampled_ids = [[30 + req_i] for req_i in range(max_num_reqs)]
    num_rej_np = np.array(rejected_per_req, dtype=np.int32)

    chunk_ref_sync = _build_b8_chunk(
        num_reqs=max_num_reqs,
        max_num_reqs=max_num_reqs,
        token_counts_per_req=token_counts_per_req,
        padded_total_tokens=32,
        hidden_size=hidden_size,
        spec_k=spec_k,
    )
    chunk_test_sync = _build_b8_chunk(
        num_reqs=max_num_reqs,
        max_num_reqs=max_num_reqs,
        token_counts_per_req=token_counts_per_req,
        padded_total_tokens=32,
        hidden_size=hidden_size,
        spec_k=spec_k,
    )
    proposer_ref.draft_chunks = [chunk_ref_sync]
    proposer_test.draft_chunks = [chunk_test_sync]

    monkeypatch.setattr(envs, "TPU_DRAFT_STEP0_IN_LOOP", False)
    ref_sync = proposer_ref.propose(
        sampled_token_ids=sampled_ids,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=num_rej_np,
        scheduler_output=None,
        return_device=False,
    )
    monkeypatch.setattr(envs, "TPU_DRAFT_STEP0_IN_LOOP", True)
    test_sync = proposer_test.propose(
        sampled_token_ids=sampled_ids,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=num_rej_np,
        scheduler_output=None,
        return_device=False,
    )
    assert isinstance(ref_sync, list) and len(ref_sync) == max_num_reqs
    assert test_sync == ref_sync

    # 3. Run B=8 on async path (`next_tokens_per_chunk`, `return_device=True`).
    next_tokens_dev = torch.full((max_num_reqs, spec_k + 1), -1, dtype=torch.int32)
    for req_i in range(max_num_reqs):
        draft_cnt = max(0, token_counts_per_req[req_i] - 1)
        acc_cnt = max(0, draft_cnt - rejected_per_req[req_i])
        for col_i in range(acc_cnt):
            next_tokens_dev[req_i, col_i] = 200 + req_i * 10 + col_i
        next_tokens_dev[req_i, acc_cnt] = 30 + req_i

    chunk_ref_async = _build_b8_chunk(
        num_reqs=max_num_reqs,
        max_num_reqs=max_num_reqs,
        token_counts_per_req=token_counts_per_req,
        padded_total_tokens=32,
        hidden_size=hidden_size,
        spec_k=spec_k,
    )
    chunk_test_async = _build_b8_chunk(
        num_reqs=max_num_reqs,
        max_num_reqs=max_num_reqs,
        token_counts_per_req=token_counts_per_req,
        padded_total_tokens=32,
        hidden_size=hidden_size,
        spec_k=spec_k,
    )
    proposer_ref.draft_chunks = [chunk_ref_async]
    proposer_test.draft_chunks = [chunk_test_async]

    monkeypatch.setattr(envs, "TPU_DRAFT_STEP0_IN_LOOP", False)
    ref_async = proposer_ref.propose(
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=None,
        return_device=True,
        next_tokens_per_chunk=[next_tokens_dev],
    )
    monkeypatch.setattr(envs, "TPU_DRAFT_STEP0_IN_LOOP", True)
    test_async = proposer_test.propose(
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=None,
        return_device=True,
        next_tokens_per_chunk=[next_tokens_dev],
    )
    assert isinstance(ref_async, torch.Tensor)
    assert isinstance(test_async, torch.Tensor)
    assert ref_async.shape == (max_num_reqs, spec_k)
    assert test_async.shape == (max_num_reqs, spec_k)
    assert torch.equal(test_async, ref_async)
    assert test_async.tolist() == ref_sync
