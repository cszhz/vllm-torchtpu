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
from vllm_torchtpu.runner.static_decode_buffers import (
    INVALID_TOKEN_ID,
    assemble_spec_next_tokens,
    compute_num_rejected,
    extract_draft_token_ids,
    fold_draft_loop_spec_bookkeeping,
    fold_verify_and_draft_prep,
    subtract_and_substitute_draft_token_ids,
)
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


def test_fold_spec_bookkeeping_env_default_false() -> None:
    """TPU_FOLD_SPEC_BOOKKEEPING defaults to False."""
    assert envs.TPU_FOLD_SPEC_BOOKKEEPING is False


@pytest.mark.parametrize(
    "accepted_counts",
    [(0, 3, 1, 2), (3, 3, 3, 3), (0, 0, 0, 0), (2, 1, 0, 3)],
)
def test_fold_verify_and_draft_prep_matches_unfolded(
    accepted_counts: tuple[int, int, int, int],
) -> None:
    """Verify `fold_verify_and_draft_prep` produces bit-identical tensors to
    the separate `_draft_async_prepare_inputs` + `compute_num_rejected` +
    mamba read-offset scatter paths across 0..K accepted draft tokens."""
    num_reqs = 4
    batch_bucket = 8
    knr = 8
    loop_bucket = 16
    step0_pad_len = 16
    max_num_reqs = 8
    spec_k = 3
    query_len = 1 + spec_k
    total_tokens = batch_bucket * query_len

    next_tokens = torch.full(
        (batch_bucket, spec_k + 1), INVALID_TOKEN_ID, dtype=torch.int32
    )
    for req_idx, num_acc in enumerate(accepted_counts):
        for col_idx in range(num_acc + 1):
            next_tokens[req_idx, col_idx] = 1000 + req_idx * 10 + col_idx

    draft_lengths = torch.zeros(batch_bucket, dtype=torch.int32)
    draft_lengths[:num_reqs] = spec_k

    qsl_np = np.minimum(np.arange(knr + 1, dtype=np.int32), num_reqs) * query_len
    query_start_loc = torch.from_numpy(qsl_np)

    pad_req_idx_np = np.minimum(np.arange(loop_bucket, dtype=np.int64), num_reqs - 1)
    pad_req_idx = torch.from_numpy(pad_req_idx_np)

    chunk_input_ids = torch.arange(100, 100 + total_tokens, dtype=torch.int32)
    base_positions_per_req = np.array([20, 40, 60, 80, 0, 0, 0, 0], dtype=np.int32)
    positions_np = np.repeat(base_positions_per_req, query_len) + np.tile(
        np.arange(query_len, dtype=np.int32), batch_bucket
    )
    positions = torch.from_numpy(positions_np)
    base_seq_lens = torch.from_numpy(base_positions_per_req + query_len)

    read_offsets_folded = torch.zeros(32, dtype=torch.int32)
    read_offsets_ref = torch.zeros(32, dtype=torch.int32)
    stacked_groups = torch.tensor(
        [[2, 5, 8, 11, 0, 0, 0, 0], [3, 6, 9, 12, 0, 0, 0, 0]],
        dtype=torch.int32,
    )

    (
        draft_input_ids,
        gather_indices,
        num_rejected_knr,
        step0_gather_indices,
        pos_step1,
        seq_lens_step1,
        bonus_full,
        spec_num_rejected,
    ) = fold_verify_and_draft_prep(
        next_tokens=next_tokens,
        chunk_input_ids=chunk_input_ids,
        draft_lengths=draft_lengths,
        query_start_loc=query_start_loc,
        pad_req_idx=pad_req_idx,
        positions=positions,
        base_seq_lens=base_seq_lens,
        step0_pad_len=step0_pad_len,
        advance_draft_positions=True,
        max_num_reqs=max_num_reqs,
        read_offsets=read_offsets_folded,
        stacked_groups=stacked_groups,
    )

    expected_rejected = torch.tensor(
        [spec_k - acc for acc in accepted_counts], dtype=torch.int32
    )
    ref_rejected = compute_num_rejected(
        next_tokens[:num_reqs], draft_lengths[:num_reqs]
    )
    assert torch.equal(ref_rejected, expected_rejected)
    assert torch.equal(num_rejected_knr[:num_reqs], expected_rejected)
    assert torch.equal(spec_num_rejected[:num_reqs], expected_rejected)
    assert int(num_rejected_knr[num_reqs:].abs().sum()) == 0

    for req_idx, num_acc in enumerate(accepted_counts):
        expected_last_tok_idx = (req_idx + 1) * query_len - 1 - (spec_k - num_acc)
        assert int(gather_indices[req_idx]) == expected_last_tok_idx
        expected_bonus = 1000 + req_idx * 10 + num_acc
        assert int(bonus_full[req_idx, 0]) == expected_bonus
        assert int(draft_input_ids[expected_last_tok_idx]) == expected_bonus
        expected_pos_step1 = int(base_positions_per_req[req_idx]) + num_acc + 1
        assert int(pos_step1[req_idx]) == expected_pos_step1
        expected_sl_step1 = int(base_positions_per_req[req_idx]) + num_acc + 1 + 1
        assert int(seq_lens_step1[req_idx]) == expected_sl_step1

    assert step0_gather_indices.shape[0] == step0_pad_len
    for req_idx, num_acc in enumerate(accepted_counts):
        slot_g0 = int(stacked_groups[0, req_idx])
        slot_g1 = int(stacked_groups[1, req_idx])
        read_offsets_ref[slot_g0] = num_acc
        read_offsets_ref[slot_g1] = num_acc
    assert torch.equal(read_offsets_folded, read_offsets_ref)


def test_fold_draft_loop_epilogue_matches_next_step_subtraction_and_substitution() -> (
    None
):
    """Verify `fold_draft_loop_spec_bookkeeping` produces the exact same
    `src_padded`, `next_seq_lens`, `next_positions`, `next_input_ids`, and
    `next_draft_token_ids` as the unfolded assemble + next-step
    `subtract_and_substitute_draft_token_ids` pipeline."""
    num_reqs = 4
    batch_bucket = 8
    knr = 8
    loop_bucket = 16
    max_num_reqs = 8
    spec_k = 3
    query_len = 1 + spec_k
    padded_total_tokens = batch_bucket * query_len
    assemble_bound = max_num_reqs * query_len

    accepted_counts = (2, 0, 3, 1)
    next_tokens = torch.full(
        (batch_bucket, spec_k + 1), INVALID_TOKEN_ID, dtype=torch.int32
    )
    for req_idx, num_acc in enumerate(accepted_counts):
        for col_idx in range(num_acc + 1):
            next_tokens[req_idx, col_idx] = 500 + req_idx * 10 + col_idx

    draft_lengths = torch.zeros(batch_bucket, dtype=torch.int32)
    draft_lengths[:num_reqs] = spec_k
    qsl_np = np.minimum(np.arange(knr + 1, dtype=np.int32), num_reqs) * query_len
    query_start_loc = torch.from_numpy(qsl_np)
    pad_req_idx = torch.from_numpy(
        np.minimum(np.arange(loop_bucket, dtype=np.int64), num_reqs - 1)
    )
    chunk_input_ids = torch.arange(1, 1 + padded_total_tokens, dtype=torch.int32)
    base_computed = np.array([12, 28, 44, 60, 0, 0, 0, 0], dtype=np.int32)
    positions_step_t = torch.from_numpy(
        np.repeat(base_computed, query_len)
        + np.tile(np.arange(query_len, dtype=np.int32), batch_bucket)
    )
    seq_lens_step_t = torch.from_numpy(base_computed + query_len)

    (
        _draft_in,
        _gather_idx,
        _num_rej_knr,
        _step0_gather,
        pos_step1,
        seq_lens_step1,
        bonus_full,
        spec_num_rejected,
    ) = fold_verify_and_draft_prep(
        next_tokens=next_tokens,
        chunk_input_ids=chunk_input_ids,
        draft_lengths=draft_lengths,
        query_start_loc=query_start_loc,
        pad_req_idx=pad_req_idx,
        positions=positions_step_t,
        base_seq_lens=seq_lens_step_t,
        step0_pad_len=loop_bucket,
        advance_draft_positions=True,
        max_num_reqs=max_num_reqs,
    )

    seq_lens_end = seq_lens_step1 + (spec_k - 2)
    drafts = torch.arange(2000, 2000 + loop_bucket * spec_k, dtype=torch.int32).reshape(
        loop_bucket, spec_k
    )

    req_expand_np = np.zeros(padded_total_tokens, dtype=np.int64)
    offsets_np = np.zeros(padded_total_tokens, dtype=np.int32)
    valid_tok_np = np.zeros(padded_total_tokens, dtype=np.int32)
    for req_idx in range(num_reqs):
        tok_start = req_idx * query_len
        tok_end = tok_start + query_len
        req_expand_np[tok_start:tok_end] = req_idx
        offsets_np[tok_start:tok_end] = np.arange(query_len, dtype=np.int32)
    valid_tok_np[: num_reqs * query_len] = 1
    valid_req_np = np.zeros(knr, dtype=np.int32)
    valid_req_np[:num_reqs] = 1
    valid_draft_np = np.zeros((batch_bucket, spec_k), dtype=np.int32)
    valid_draft_np[:num_reqs, :] = 1

    (
        drafts_out,
        src_padded,
        next_seq_lens,
        next_positions,
        next_input_ids,
        next_draft_token_ids,
    ) = fold_draft_loop_spec_bookkeeping(
        drafts=drafts,
        pos_step1=pos_step1,
        seq_lens_end=seq_lens_end,
        bonus_tokens=bonus_full,
        assemble_bound=assemble_bound,
        num_loop_steps=spec_k - 1,
        pos_req_expand_idx=torch.from_numpy(req_expand_np),
        pos_step1_offsets=torch.from_numpy(offsets_np),
        valid_token_mask=torch.from_numpy(valid_tok_np),
        valid_req_mask=torch.from_numpy(valid_req_np),
        valid_draft_mask=torch.from_numpy(valid_draft_np),
    )
    assert torch.equal(drafts_out, drafts)

    ref_src_unpadded = assemble_spec_next_tokens(next_tokens, drafts, num_reqs)
    assert torch.equal(src_padded[: num_reqs * query_len], ref_src_unpadded)

    optimistic_computed_next = base_computed.copy()
    optimistic_computed_next[:num_reqs] += query_len
    optimistic_seq_lens = torch.zeros(knr, dtype=torch.int32)
    optimistic_seq_lens[:num_reqs] = torch.from_numpy(
        optimistic_computed_next[:num_reqs] + query_len
    )
    optimistic_positions = torch.zeros(padded_total_tokens, dtype=torch.int32)
    for req_idx in range(num_reqs):
        for offset in range(query_len):
            optimistic_positions[req_idx * query_len + offset] = (
                int(optimistic_computed_next[req_idx]) + offset
            )
    optimistic_input_ids = torch.zeros(padded_total_tokens, dtype=torch.int32)

    seq_sub_idx = torch.full((knr,), -1, dtype=torch.int32)
    seq_sub_idx[:num_reqs] = torch.arange(num_reqs, dtype=torch.int32)
    pos_sub_idx = torch.full((padded_total_tokens,), -1, dtype=torch.int32)
    for req_idx in range(num_reqs):
        pos_sub_idx[req_idx * query_len : (req_idx + 1) * query_len] = req_idx

    cur_in_idx = torch.arange(padded_total_tokens, dtype=torch.int64)
    pre_nt_idx = torch.full((padded_total_tokens,), -1, dtype=torch.int64)
    pre_nt_idx[: num_reqs * query_len] = torch.arange(
        num_reqs * query_len, dtype=torch.int64
    )
    substitution_indices = torch.stack([cur_in_idx, pre_nt_idx], dim=0)

    logits_indices = torch.arange(padded_total_tokens, dtype=torch.int32)
    target_logits_list: list[int] = []
    for req_idx in range(batch_bucket):
        for offset in range(spec_k):
            if req_idx < num_reqs:
                target_logits_list.append(req_idx * query_len + offset)
            else:
                target_logits_list.append(num_reqs * query_len)
    target_logits_indices = torch.tensor(target_logits_list, dtype=torch.int32)

    (
        ref_next_seq_lens,
        ref_next_positions,
        ref_next_input_ids,
        ref_next_draft_token_ids,
    ) = subtract_and_substitute_draft_token_ids(
        seq_lens=optimistic_seq_lens,
        positions=optimistic_positions,
        num_rejected=spec_num_rejected,
        seq_lens_subtract_indices=seq_sub_idx,
        positions_subtract_indices=pos_sub_idx,
        input_ids=optimistic_input_ids,
        substitution_indices=substitution_indices,
        next_tokens_tpu=src_padded,
        logits_indices=logits_indices,
        target_logits_indices=target_logits_indices,
    )

    assert torch.equal(next_seq_lens, ref_next_seq_lens)
    assert torch.equal(next_positions, ref_next_positions)
    assert torch.equal(next_input_ids, ref_next_input_ids)
    assert torch.equal(next_draft_token_ids, ref_next_draft_token_ids)


def test_fold_draft_loop_epilogue_src_only_fallback() -> None:
    """When steady-state next-step masks are omitted (e.g. MRoPE or non-zero
    start_index chunk), `fold_draft_loop_spec_bookkeeping` still folds
    `src_padded` assembly and returns a 2-tuple `(drafts, src_padded)`."""
    drafts = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)
    pos_step1 = torch.tensor([5, 9], dtype=torch.int32)
    seq_lens_end = torch.tensor([6, 10], dtype=torch.int32)
    bonus_tokens = torch.tensor([[10], [20]], dtype=torch.int32)
    out = fold_draft_loop_spec_bookkeeping(
        drafts=drafts,
        pos_step1=pos_step1,
        seq_lens_end=seq_lens_end,
        bonus_tokens=bonus_tokens,
        assemble_bound=12,
        num_loop_steps=2,
    )
    assert len(out) == 2
    assert torch.equal(out[0], drafts)
    expected_src = torch.tensor(
        [10, 11, 12, 13, 20, 21, 22, 23, 0, 0, 0, 0], dtype=torch.int32
    )
    assert torch.equal(out[1], expected_src)


def test_fold_draft_loop_epilogue_mrope_2d_positions_and_step1_seq_lens() -> None:
    """Verify `fold_draft_loop_spec_bookkeeping` supports 2D MRoPE positions
    `(3, p) -> (3, T)` and `seq_lens_is_step1=True` (used when
    `TPU_DRAFT_STEP0_IN_LOOP=1` runs alongside `TPU_FOLD_SPEC_BOOKKEEPING=1`)."""
    num_reqs = 2
    batch_bucket = 4
    loop_bucket = 4
    spec_k = 3
    query_len = 1 + spec_k
    padded_total_tokens = batch_bucket * query_len
    assemble_bound = batch_bucket * query_len

    drafts = torch.tensor(
        [[101, 102, 103], [201, 202, 203], [0, 0, 0], [0, 0, 0]],
        dtype=torch.int32,
    )
    pos_1d = np.array([15, 32, 32, 32], dtype=np.int32)
    pos_step1_2d = torch.from_numpy(np.stack([pos_1d, pos_1d, pos_1d], axis=0))
    seq_lens_step1 = torch.tensor([16, 33, 0, 0], dtype=torch.int32)
    bonus_tokens = torch.tensor([[100], [200], [-1], [-1]], dtype=torch.int32)

    req_expand_np = np.zeros(padded_total_tokens, dtype=np.int64)
    offsets_np = np.zeros(padded_total_tokens, dtype=np.int32)
    valid_tok_np = np.zeros(padded_total_tokens, dtype=np.int32)
    for req_idx in range(num_reqs):
        tok_start = req_idx * query_len
        tok_end = tok_start + query_len
        req_expand_np[tok_start:tok_end] = min(req_idx, loop_bucket - 1)
        offsets_np[tok_start:tok_end] = np.arange(query_len, dtype=np.int32)
    valid_tok_np[: num_reqs * query_len] = 1
    valid_req_np = np.array([1, 1, 0, 0], dtype=np.int32)
    valid_draft_np = np.zeros((batch_bucket, spec_k), dtype=np.int32)
    valid_draft_np[:num_reqs, :] = 1

    (
        drafts_out,
        src_padded,
        next_seq_lens,
        next_positions,
        next_input_ids,
        next_draft_token_ids,
    ) = fold_draft_loop_spec_bookkeeping(
        drafts=drafts,
        pos_step1=pos_step1_2d,
        seq_lens_end=seq_lens_step1,
        bonus_tokens=bonus_tokens,
        assemble_bound=assemble_bound,
        num_loop_steps=spec_k - 1,
        pos_req_expand_idx=torch.from_numpy(req_expand_np),
        pos_step1_offsets=torch.from_numpy(offsets_np),
        valid_token_mask=torch.from_numpy(valid_tok_np),
        valid_req_mask=torch.from_numpy(valid_req_np),
        valid_draft_mask=torch.from_numpy(valid_draft_np),
        seq_lens_is_step1=True,
    )

    assert torch.equal(drafts_out, drafts)
    assert next_positions.shape == (3, padded_total_tokens)
    expected_pos_row = torch.tensor(
        [15, 16, 17, 18, 32, 33, 34, 35, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.int32,
    )
    for row_idx in range(3):
        assert torch.equal(next_positions[row_idx], expected_pos_row)
    expected_next_seq_lens = torch.tensor(
        [16 + spec_k, 33 + spec_k, 0, 0], dtype=torch.int32
    )
    assert torch.equal(next_seq_lens, expected_next_seq_lens)
    expected_next_input_ids = torch.tensor(
        [100, 101, 102, 103, 200, 201, 202, 203, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.int32,
    )
    assert torch.equal(next_input_ids, expected_next_input_ids)
    expected_next_drafts = torch.tensor(
        [101, 102, 103, 201, 202, 203, 0, 0, 0, 0, 0, 0],
        dtype=torch.int32,
    )
    assert torch.equal(next_draft_token_ids, expected_next_drafts)
    assert torch.equal(src_padded[:8], expected_next_input_ids[:8])


def test_fold_draft_loop_epilogue_warmup_mismatched_num_reqs_and_token_paddings() -> (
    None
):
    """Verify 1D `valid_draft_mask` of length `padded_total_tokens` produces
    `next_draft_token_ids` of shape `(4,)` when `num_reqs = 1`,
    `padded_num_reqs = 8`, `spec_k = 3`, and `padded_total_tokens = 4`."""
    num_reqs = 1
    padded_num_reqs = 8
    knr = 8
    spec_k = 3
    query_len = 1 + spec_k
    padded_total_tokens = 4
    assemble_bound = padded_num_reqs * query_len

    drafts = torch.tensor(
        [[11, 12, 13], [99, 98, 97], [0, 0, 0], [0, 0, 0]],
        dtype=torch.int32,
    )
    pos_1d = np.array([20, 20, 20, 20], dtype=np.int32)
    pos_step1_2d = torch.from_numpy(np.stack([pos_1d, pos_1d, pos_1d], axis=0))
    seq_lens_end = torch.tensor([22, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    bonus_tokens = torch.tensor(
        [[10], [-1], [-1], [-1], [-1], [-1], [-1], [-1]], dtype=torch.int32
    )

    req_expand_np = np.zeros(padded_total_tokens, dtype=np.int64)
    offsets_np = np.arange(padded_total_tokens, dtype=np.int32)
    valid_tok_np = np.ones(padded_total_tokens, dtype=np.int32)
    valid_req_np = np.zeros(knr, dtype=np.int32)
    valid_req_np[:num_reqs] = 1
    valid_draft_np = np.zeros(padded_total_tokens, dtype=np.int32)
    valid_draft_np[: min(num_reqs * spec_k, padded_total_tokens)] = 1

    (
        drafts_out,
        src_padded,
        next_seq_lens,
        next_positions,
        next_input_ids,
        next_draft_token_ids,
    ) = fold_draft_loop_spec_bookkeeping(
        drafts=drafts,
        pos_step1=pos_step1_2d,
        seq_lens_end=seq_lens_end,
        bonus_tokens=bonus_tokens,
        assemble_bound=assemble_bound,
        num_loop_steps=spec_k - 1,
        pos_req_expand_idx=torch.from_numpy(req_expand_np),
        pos_step1_offsets=torch.from_numpy(offsets_np),
        valid_token_mask=torch.from_numpy(valid_tok_np),
        valid_req_mask=torch.from_numpy(valid_req_np),
        valid_draft_mask=torch.from_numpy(valid_draft_np),
        seq_lens_is_step1=False,
    )

    assert torch.equal(drafts_out, drafts)
    assert src_padded.shape == (assemble_bound,)
    assert next_seq_lens.shape == (knr,)
    assert next_positions.shape == (3, padded_total_tokens)
    assert next_input_ids.shape == (padded_total_tokens,)
    assert next_draft_token_ids.shape == (padded_total_tokens,)
    assert torch.equal(
        next_draft_token_ids, torch.tensor([11, 12, 13, 0], dtype=torch.int32)
    )

    target_logits_argmax = torch.tensor([11, 12, 99, 10], dtype=torch.int32)
    accepted = next_draft_token_ids == target_logits_argmax
    assert torch.equal(
        accepted, torch.tensor([True, True, False, False], dtype=torch.bool)
    )

    logits_indices = torch.arange(padded_total_tokens, dtype=torch.int32)
    target_logits_indices = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
    ref_extracted = extract_draft_token_ids(
        next_input_ids, logits_indices, target_logits_indices
    )
    assert torch.equal(
        next_draft_token_ids[: num_reqs * spec_k],
        ref_extracted[: num_reqs * spec_k],
    )


def test_sample_spec_verify_chunk_non_greedy_skips_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_sample_spec_verify_chunk` does not run greedy fold prep when
    `all_greedy=False` (e.g. temperature > 0) even with
    `TPU_FOLD_SPEC_BOOKKEEPING=True`."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)
    draft_chunk = types.SimpleNamespace(
        num_reqs=1,
        input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        position_ids=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        attn_ctx=types.SimpleNamespace(
            seq_lens=torch.tensor([14], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        ),
        sequence_layout_plan=None,
    )
    expected_out = torch.tensor([[5, 6, -1, -1]], dtype=torch.int32)
    build_called = False
    non_greedy_branch_entered = False

    def _fail_if_called(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal build_called
        build_called = True
        raise AssertionError("Should not build greedy fold prep when all_greedy=False")

    def _build_padded_params(
        cur_start: int, cur_end: int, bonus_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal non_greedy_branch_entered
        non_greedy_branch_entered = True
        n = cur_end - cur_start
        dev = bonus_logits.device
        return (
            torch.full((n,), 0.8, dtype=torch.float32, device=dev),
            torch.full((n,), -1, dtype=torch.int32, device=dev),
            torch.full((n,), 1.0, dtype=torch.float32, device=dev),
        )

    class _FakeRejectionSampler:
        synthetic_mode = False
        synthetic_conditional_rates = None

        def __call__(self, *_a: Any, **_kw: Any) -> torch.Tensor:
            return expected_out

    monkeypatch.setattr(
        TPUModelRunner, "_build_fold_verify_and_draft_prep_inputs", _fail_if_called
    )
    fake_runner = types.SimpleNamespace(
        rejection_sampler=_FakeRejectionSampler(),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=types.SimpleNamespace(async_scheduling=True),
        _is_async_drafter=True,
        input_batch=types.SimpleNamespace(top_k_cpu=np.array([0], dtype=np.int32)),
        spec_gather_bonus_and_target_logits=lambda logits, b_idx, t_idx: (
            logits[b_idx],
            logits[t_idx],
        ),
        _build_padded_sampling_params=_build_padded_params,
        _sampler_topk_candidates=lambda *_a, **_kw: 0,
        _sampler_applies_topk_topp=lambda *_a, **_kw: False,
        sample_from_logits_func=lambda *_a, **_kw: torch.tensor([6], dtype=torch.int32),
        spec_prepare_non_greedy_rejection_inputs=lambda *_a, **_kw: (
            torch.zeros((3, 8), dtype=torch.float32),
            torch.zeros((3, 8), dtype=torch.float32),
        ),
    )
    md = types.SimpleNamespace(
        draft_token_ids=torch.tensor([2, 3, 4], dtype=torch.int32),
        draft_lengths=torch.tensor([3], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        segment_ids=torch.tensor([0, 0, 0], dtype=torch.int32),
        group_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    logits = torch.zeros((4, 8), dtype=torch.float32)
    out = TPUModelRunner._sample_spec_verify_chunk(
        fake_runner,
        logits=logits,
        md=md,
        grammar_output=None,
        scheduler_output=cast(Any, None),
        cur_start_idx=0,
        cur_end_idx=1,
        all_greedy=False,
        sampling_generator=None,
        top_tokens=False,
        draft_chunk=draft_chunk,
        mamba_indices=None,
        can_defer_mamba_offsets=True,
    )
    assert not build_called
    assert non_greedy_branch_entered
    assert getattr(draft_chunk, "_precomputed_draft_prep", None) is None
    assert torch.equal(out, expected_out)


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
    chunk = cast(Any, types.SimpleNamespace(num_reqs=1, _precomputed_draft_prep=None))

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
    chunk_input_ids = torch.arange(100, 116, dtype=torch.int32)
    chunk_positions = torch.arange(16, dtype=torch.int64)
    query_start_loc_np = np.array([0, 4, 8, 12, 16], dtype=np.int32)
    next_tokens_dev = torch.full((padded_width, 4), -1, dtype=torch.int32)
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
    assert gather_indices[4:].tolist() == [14] * 12


def test_real_eagle3_proposer_fold_verify_attribute_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instantiate real Eagle3Proposer and run fold-verify builder."""
    import vllm_torchtpu.envs as tpu_envs
    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    monkeypatch.setattr(tpu_envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)

    fake_vllm_config = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
            draft_tensor_parallel_size=8,
            draft_model_config=types.SimpleNamespace(
                max_model_len=2048,
                get_hidden_size=lambda: 64,
                hf_text_config=types.SimpleNamespace(model_type="qwen3_5_moe_mtp"),
            ),
            draft_parallel_config=types.SimpleNamespace(tensor_parallel_size=8),
        ),
        model_config=types.SimpleNamespace(
            dtype=torch.bfloat16,
            max_model_len=2048,
            hf_text_config=types.SimpleNamespace(model_type="qwen3_5_moe"),
        ),
        cache_config=types.SimpleNamespace(block_size=16),
        scheduler_config=types.SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=64,
            async_scheduling=True,
        ),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=8),
        compilation_config=types.SimpleNamespace(),
    )
    fake_runner_for_init = types.SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_model_len=2048,
        max_num_reqs=8,
        max_num_tokens=64,
        num_tokens_paddings=[16, 32, 64],
        _dp_lockstep_enabled=lambda: False,
    )

    real_proposer = Eagle3Proposer(
        vllm_config=cast(Any, fake_vllm_config),
        runner=cast(Any, fake_runner_for_init),
    )

    # Assert every attribute read by C1/C2/C3 on Eagle3Proposer exists after __init__
    required_proposer_attrs = (
        "_async_pad_req_idx_cache",
        "_zero_rej_dev_cache",
        "_loop_qsl_reqdist_cache",
        "_mtp_full_draft_loop",
        "_mtp_full_draft_loop_warm",
        "_mtp_fused_loop",
        "_mtp_fused_loop_warm",
        "_last_step0_gather_indices",
        "_last_prep_step1",
        "_share_mtp_indices",
        "_draft_kv_shared",
    )
    proposer_vars = vars(real_proposer)
    for attr_name in required_proposer_attrs:
        assert attr_name in proposer_vars, f"Missing {attr_name} on Eagle3Proposer"
    assert isinstance(real_proposer._async_pad_req_idx_cache, dict)

    # Run _build_fold_verify_and_draft_prep_inputs with the real Eagle3Proposer
    draft_chunk = types.SimpleNamespace(
        num_reqs=2,
        start_index=0,
        req_ids=["r0", "r1"],
        input_ids=torch.arange(100, 116, dtype=torch.int32),
        position_ids=torch.arange(16, dtype=torch.int64),
        query_start_loc_np=np.array([0, 4, 8], dtype=np.int32),
        sequence_layout_plan=types.SimpleNamespace(kind=SequenceLayoutKind.ALL),
        attn_ctx=types.SimpleNamespace(
            use_max_model_len=False,
            seq_lens_cpu=np.array([20, 30, 0, 0, 0, 0, 0, 0], dtype=np.int32),
            seq_lens=torch.tensor([20, 30, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
            query_start_loc=torch.tensor(
                [0, 4, 8, 8, 8, 8, 8, 8, 8], dtype=torch.int32
            ),
        ),
        _precomputed_draft_prep=None,
    )
    runner_ns = types.SimpleNamespace(
        device=torch.device("cpu"),
        drafter=real_proposer,
        _is_async_drafter=True,
        max_num_reqs=8,
        uses_mrope=False,
        requests={},
        input_batch=types.SimpleNamespace(req_ids=["r0", "r1"]),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=types.SimpleNamespace(async_scheduling=True),
        num_reqs_max_model_len=8,
        num_reqs_most_model_len=8,
        mamba_slot_read_offsets=None,
        _mamba_state_pos=None,
        _last_step_used_static_decode=False,
        _folded_steady_state_prep=None,
        _fold_epilogue_mask_cache={},
    )

    fold_inputs = TPUModelRunner._build_fold_verify_and_draft_prep_inputs(
        cast(Any, runner_ns),
        draft_chunk=cast(Any, draft_chunk),
        mamba_indices=None,
        can_defer_mamba_offsets=False,
    )
    assert fold_inputs is not None
    assert (16, 2, "cpu") in real_proposer._async_pad_req_idx_cache


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
    proposer._draft_tp_matches_target = True
    proposer._draft_has_moe_cache = False
    proposer._draft_attn_layer_names = {"layer.0"}
    proposer._draft_md_cache = {}
    proposer._zero_rej_dev_cache = {}
    proposer._loop_qsl_reqdist_cache = {}
    proposer._async_pad_req_idx_cache = {}
    proposer._last_step0_gather_indices = None
    proposer._last_prep_step1 = None
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
    assert specializer.specialized_gather_width == 8

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


def test_fold_spec_bookkeeping_b8_with_step0_in_loop_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify C2 + C3 (`TPU_DRAFT_STEP0_IN_LOOP=True` and
    `TPU_FOLD_SPEC_BOOKKEEPING=True`) at `B = 8 > 4` after `B = 1` warmup
    produces bit-identical `[8, 3]` draft tokens, `src_padded` (all 8 requests,
    32 tokens), `next_seq_lens`, `next_positions`, `next_input_ids`, and
    `next_draft_token_ids` as the unfolded baseline."""
    import vllm.forward_context as vllm_fwd_ctx

    import vllm_torchtpu.spec_decode.eagle3 as eagle3_mod
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner, _PrecomputedDraftPrep
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
    query_len = 1 + spec_k
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

    # Warmup proposer_test at B=1 first.
    monkeypatch.setattr(envs, "TPU_DRAFT_STEP0_IN_LOOP", True)
    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)
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
    _ = proposer_test.propose(
        sampled_token_ids=[[15]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=np.array([0], dtype=np.int32),
        scheduler_output=None,
        return_device=True,
    )
    assert specializer.specialized_gather_width == 8

    # Now run B=8 with fold_verify_and_draft_prep + propose + epilogue.
    token_counts = (4, 4, 4, 4, 4, 4, 4, 4)
    rejected_counts = (0, 1, 2, 3, 1, 0, 2, 3)
    next_tokens_dev = torch.full(
        (max_num_reqs, spec_k + 1), INVALID_TOKEN_ID, dtype=torch.int32
    )
    for req_i in range(max_num_reqs):
        acc_cnt = spec_k - rejected_counts[req_i]
        for col_i in range(acc_cnt):
            next_tokens_dev[req_i, col_i] = 200 + req_i * 10 + col_i
        next_tokens_dev[req_i, acc_cnt] = 30 + req_i

    chunk_ref = _build_b8_chunk(
        num_reqs=max_num_reqs,
        max_num_reqs=max_num_reqs,
        token_counts_per_req=token_counts,
        padded_total_tokens=32,
        hidden_size=hidden_size,
        spec_k=spec_k,
    )
    chunk_test = _build_b8_chunk(
        num_reqs=max_num_reqs,
        max_num_reqs=max_num_reqs,
        token_counts_per_req=token_counts,
        padded_total_tokens=32,
        hidden_size=hidden_size,
        spec_k=spec_k,
    )

    runner_ns = types.SimpleNamespace(
        device=torch.device("cpu"),
        drafter=proposer_test,
        _is_async_drafter=True,
        max_num_reqs=max_num_reqs,
        uses_mrope=False,
        requests={},
        input_batch=types.SimpleNamespace(
            req_ids=[f"req_{req_i}" for req_i in range(max_num_reqs)]
        ),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=spec_k),
        scheduler_config=types.SimpleNamespace(async_scheduling=True),
        num_reqs_max_model_len=max_num_reqs,
        num_reqs_most_model_len=max_num_reqs,
        mamba_slot_read_offsets=None,
        _mamba_state_pos=None,
        _last_step_used_static_decode=False,
        _folded_steady_state_prep=None,
        _fold_epilogue_mask_cache={},
    )
    fold_inputs = TPUModelRunner._build_fold_verify_and_draft_prep_inputs(
        cast(Any, runner_ns),
        draft_chunk=cast(Any, chunk_test),
        mamba_indices=None,
        can_defer_mamba_offsets=False,
    )
    assert fold_inputs is not None
    (
        f_chunk_ids,
        f_qsl,
        f_pad_req_idx,
        f_positions,
        f_base_sl,
        f_step0_pad_len,
        f_adv_pos,
        f_max_reqs,
        _f_ro,
        _f_sg,
        f_epilogue_tensors,
        f_steady_key,
    ) = fold_inputs
    assert f_step0_pad_len == 8
    assert f_pad_req_idx.shape == (8,)

    prep_out = fold_verify_and_draft_prep(
        next_tokens=next_tokens_dev,
        chunk_input_ids=f_chunk_ids,
        draft_lengths=chunk_test.draft_lengths,
        query_start_loc=f_qsl,
        pad_req_idx=f_pad_req_idx,
        positions=f_positions,
        base_seq_lens=f_base_sl,
        step0_pad_len=f_step0_pad_len,
        advance_draft_positions=f_adv_pos,
        max_num_reqs=f_max_reqs,
    )
    chunk_test._precomputed_draft_prep = _PrecomputedDraftPrep(
        draft_input_ids=prep_out[0],
        gather_indices=prep_out[1],
        num_rejected_knr=prep_out[2],
        step0_gather_indices=prep_out[3],
        pos_step1=prep_out[4],
        seq_lens_step1=prep_out[5],
        bonus_full=prep_out[6],
        spec_num_rejected=prep_out[7],
        mamba_offsets_updated=False,
        pos_req_expand_idx=f_epilogue_tensors[0],
        pos_step1_offsets=f_epilogue_tensors[1],
        valid_token_mask=f_epilogue_tensors[2],
        valid_req_mask=f_epilogue_tensors[3],
        valid_draft_mask=f_epilogue_tensors[4],
        steady_state_key=f_steady_key,
    )

    proposer_ref.runner.input_batch.num_reqs = max_num_reqs
    proposer_test.runner.input_batch.num_reqs = max_num_reqs
    proposer_ref.draft_chunks = [chunk_ref]
    proposer_test.draft_chunks = [chunk_test]

    monkeypatch.setattr(envs, "TPU_DRAFT_STEP0_IN_LOOP", False)
    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", False)
    drafts_ref = proposer_ref.propose(
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=None,
        return_device=True,
        next_tokens_per_chunk=[next_tokens_dev],
    )

    monkeypatch.setattr(envs, "TPU_DRAFT_STEP0_IN_LOOP", True)
    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)
    drafts_test = proposer_test.propose(
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=None,
        return_device=True,
        next_tokens_per_chunk=[next_tokens_dev],
    )
    assert isinstance(drafts_ref, torch.Tensor)
    assert isinstance(drafts_test, torch.Tensor)
    assert torch.equal(drafts_test, drafts_ref)

    prep_after = chunk_test._precomputed_draft_prep
    assert prep_after is not None
    assert prep_after.src_padded is not None
    ref_src = assemble_spec_next_tokens(next_tokens_dev, drafts_ref, max_num_reqs)
    assert torch.equal(prep_after.src_padded[: max_num_reqs * query_len], ref_src)
    assert prep_after.next_input_ids is not None
    assert torch.equal(prep_after.next_input_ids[: max_num_reqs * query_len], ref_src)
