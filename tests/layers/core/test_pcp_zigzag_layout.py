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

from types import SimpleNamespace
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.core.pcp_sequence_layout import (
    PCP_INTERLEAVED_SEQUENCE_LAYOUT_DESCRIPTOR,
    PCP_ZIGZAG_SEQUENCE_LAYOUT_DESCRIPTOR,
    PcpSequenceLayoutEligibility,
    PcpSequenceLayoutMode,
    PcpSequenceLayoutPlanner,
    apply_pcp_interleaved_token_order,
    apply_pcp_zigzag_token_order,
    pcp_interleaved_page_table,
    pcp_interleaved_seq_arrays,
    pcp_interleaved_token_layout,
    pcp_zigzag_page_table,
    pcp_zigzag_seq_arrays,
    pcp_zigzag_token_layout,
    prepare_pcp_interleaved_sequence_layout,
    prepare_pcp_zigzag_sequence_layout,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    PCP_INTERLEAVED_SEQUENCE_LAYOUT_PROTOCOL,
    PCP_ZIGZAG_SEQUENCE_LAYOUT_PROTOCOL,
    is_pcp_interleaved_sequence_layout_descriptor,
    is_pcp_zigzag_sequence_layout_descriptor,
)


def _input_batch(*, computed, prompt):
    req_ids = [f"req{i}" for i in range(len(computed))]
    return SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={req_id: idx for idx, req_id in enumerate(req_ids)},
        num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
        num_prompt_tokens=np.asarray(prompt, dtype=np.int32),
    )


def _scheduler_output(scheduled):
    return SimpleNamespace(
        num_scheduled_tokens={
            f"req{i}": int(tokens) for i, tokens in enumerate(scheduled)
        }
    )


def test_pcp_zigzag_token_layout_single_request():
    pcp_size = 2
    counts = [256]
    chunk, off, s_live = pcp_zigzag_token_layout(counts, pcp_size)

    assert chunk == [64]
    assert off == [0]
    assert s_live == 128


def test_pcp_zigzag_token_layout_multi_request():
    pcp_size = 4
    counts = [200, 300]
    chunk, off, s_live = pcp_zigzag_token_layout(counts, pcp_size, block_size=16)

    # two_p = 8, quantum = 128.
    # req 0: ceil(200 / 128) * 16 = 32
    # req 1: ceil(300 / 128) * 16 = 48
    assert chunk == [32, 48]
    assert off == [0, 64]
    assert s_live == 160


def test_pcp_zigzag_seq_arrays():
    pcp_size = 2
    two_p = 4
    n_off = 4
    chunk = [32]
    off = [0]
    counts = [100]  # chunk is 32, so 4 chunks = 128 total capacity

    cu_q_lens, q_pos_offsets, pcp_write = pcp_zigzag_seq_arrays(
        chunk, off, pcp_size, n_off, counts
    )

    assert cu_q_lens.shape == (pcp_size, n_off + 1)
    assert q_pos_offsets.shape == (pcp_size, n_off)
    assert pcp_write.shape == (pcp_size, n_off)

    # For rank 0: owns chunk 0 (head) and chunk 3 (tail)
    # Head chunk offset is 0 * 32 = 0
    assert q_pos_offsets[0, 0] == 0
    # Tail chunk offset is 3 * 32 = 96
    assert q_pos_offsets[0, 1] == 96

    # For rank 1: owns chunk 1 (head) and chunk 2 (tail)
    # Head chunk offset is 1 * 32 = 32
    assert q_pos_offsets[1, 0] == 32
    # Tail chunk offset is 2 * 32 = 64
    assert q_pos_offsets[1, 1] == 64

    # Writer flags: tail chunk has real tokens (100 - 3*32 = 4 > 0), so tail writes
    assert pcp_write[0, 1] == 1
    assert pcp_write[0, 0] == 0


def test_pcp_zigzag_seq_arrays_writer_fallback():
    pcp_size = 2
    two_p = 4
    n_off = 4
    chunk = [32]
    off = [0]
    # Total tokens is only 64, so chunks 2 and 3 have 0 real tokens
    counts = [64]

    cu_q_lens, q_pos_offsets, pcp_write = pcp_zigzag_seq_arrays(
        chunk, off, pcp_size, n_off, counts
    )

    # Rank 0 owns chunk 0 (head) and chunk 3 (tail).
    # Chunk 3 has start offset 3*32 = 96 >= 64, so 0 real tokens.
    # Therefore, rank 0's writer flag should fall back to chunk 0 (head)!
    assert pcp_write[0, 1] == 0
    assert pcp_write[0, 0] == 1


def test_pcp_zigzag_page_table():
    pcp_size = 2
    block_size = 16
    n_off = 4
    chunk = [32]  # pages_per_chunk = 32 // 16 = 2

    table = pcp_zigzag_page_table(chunk, pcp_size, block_size, n_off)
    assert table.shape == (n_off, 2 * pcp_size * 2)
    # Chunk 0 (rank 0 head): pages 0, 1
    assert np.array_equal(table[0, 0:2], [0, 1])
    # Chunk 1 (rank 1 head): pages 4, 5
    assert np.array_equal(table[0, 2:4], [4, 5])
    # Chunk 2 (rank 1 tail): pages 6, 7
    assert np.array_equal(table[0, 4:6], [6, 7])
    # Chunk 3 (rank 0 tail): pages 2, 3
    assert np.array_equal(table[0, 6:8], [2, 3])


def test_apply_pcp_zigzag_token_order():
    pcp_size = 2
    counts = [128]
    chunk, off, s_live = pcp_zigzag_token_layout(counts, pcp_size)
    s_pad = 64
    padded_total_num_tokens = s_pad * pcp_size  # 128

    input_ids = np.arange(128, dtype=np.int32)
    positions_np = np.arange(128, dtype=np.int32)

    req_to_packed, packed_to_req = apply_pcp_zigzag_token_order(
        input_ids, positions_np, counts, chunk, off, pcp_size, padded_total_num_tokens
    )

    # Verify that the permutation is invertible
    assert np.array_equal(packed_to_req[req_to_packed], np.arange(128))

    # Verify rank 0 slot contains chunk 0 (head, 0..31) and chunk 3 (tail, 96..127)
    rank0_tokens = input_ids[:s_pad]
    assert np.array_equal(rank0_tokens[0:32], np.arange(0, 32))
    assert np.array_equal(rank0_tokens[32:64], np.arange(96, 128))

    # Verify rank 1 slot contains chunk 1 (head, 32..63) and chunk 2 (tail, 64..95)
    rank1_tokens = input_ids[s_pad : 2 * s_pad]
    assert np.array_equal(rank1_tokens[0:32], np.arange(32, 64))
    assert np.array_equal(rank1_tokens[32:64], np.arange(64, 96))


def test_pcp_zigzag_eligibility():
    eligibility = PcpSequenceLayoutEligibility(
        pcp_size=4,
        interleave_size=16,
        is_zigzag=True,
        block_size=16,
    )
    assert eligibility.enabled
    assert eligibility.is_zigzag

    decision = eligibility.evaluate_runner_chunk(
        input_batch=_input_batch(computed=[0], prompt=[256]),
        scheduler_output=_scheduler_output([256]),
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=np.array([256], dtype=np.int32),
        num_tokens_paddings=[128, 256, 512],
        max_num_tokens=512,
    )

    assert decision.mode is PcpSequenceLayoutMode.ZIGZAG
    assert decision.local_required_tokens == 256 // 4  # 64 per rank


def test_pcp_zigzag_planner_prepare_dummy():
    eligibility = PcpSequenceLayoutEligibility(
        pcp_size=2,
        interleave_size=16,
        is_zigzag=True,
        block_size=16,
    )
    planner = PcpSequenceLayoutPlanner(eligibility)

    plan = planner.prepare_dummy(
        num_tokens=128,
        num_reqs=2,
        kv_cache_initialized=True,
    )

    assert is_pcp_zigzag_sequence_layout_descriptor(plan.descriptor)
    assert plan.descriptor.protocol == PCP_ZIGZAG_SEQUENCE_LAYOUT_PROTOCOL
    assert plan.pcp_metadata is not None
    assert plan.pcp_metadata.cache_pages == 0
    assert plan.pcp_metadata.query_start_loc.shape[0] == 2


def test_pcp_zigzag_decode_single_request():
    eligibility = PcpSequenceLayoutEligibility(
        pcp_size=4,
        interleave_size=16,
        is_zigzag=True,
        block_size=16,
    )
    decision = eligibility.evaluate_runner_chunk(
        input_batch=_input_batch(computed=[256], prompt=[256]),
        scheduler_output=_scheduler_output([1]),
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=np.array([1], dtype=np.int32),
        num_tokens_paddings=[128, 256, 512],
        max_num_tokens=512,
    )
    assert decision.mode is PcpSequenceLayoutMode.ZIGZAG
    assert decision.local_required_tokens == 1
    assert decision.local_padded_tokens == 128


def test_pcp_zigzag_decode_multi_request():
    eligibility = PcpSequenceLayoutEligibility(
        pcp_size=4,
        interleave_size=16,
        is_zigzag=True,
        block_size=16,
    )
    decision = eligibility.evaluate_runner_chunk(
        input_batch=_input_batch(computed=[256, 512], prompt=[256, 512]),
        scheduler_output=_scheduler_output([1, 1]),
        start_index=0,
        num_reqs=2,
        num_scheduled_tokens_per_req=np.array([1, 1], dtype=np.int32),
        num_tokens_paddings=[128, 256, 512],
        max_num_tokens=512,
    )
    assert decision.mode is PcpSequenceLayoutMode.ZIGZAG


def test_pcp_zigzag_rejects_mixed_decode_prefill():
    eligibility = PcpSequenceLayoutEligibility(
        pcp_size=4,
        interleave_size=16,
        is_zigzag=True,
        block_size=16,
    )
    with pytest.raises(NotImplementedError, match="mixed decode and prefill"):
        eligibility.evaluate_runner_chunk(
            input_batch=_input_batch(computed=[256, 0], prompt=[256, 256]),
            scheduler_output=_scheduler_output([1, 256]),
            start_index=0,
            num_reqs=2,
            num_scheduled_tokens_per_req=np.array([1, 256], dtype=np.int32),
            num_tokens_paddings=[128, 256, 512],
            max_num_tokens=512,
        )


def test_pcp_zigzag_seq_arrays_multi_request():
    pcp_size = 2
    n_off = 4
    counts = [1, 1]
    chunk, off, _ = pcp_zigzag_token_layout(counts, pcp_size, block_size=16)
    cu_q_lens, q_pos_offsets, pcp_write = pcp_zigzag_seq_arrays(
        chunk, off, pcp_size, n_off, counts
    )
    # Req 0 tail is empty (counts[0]=1 <= 16), but intermediate cu_q_lens[r, 2]
    # must advance to off[1] = 32 socu_q_lens stays monotonic!
    assert np.all(np.diff(cu_q_lens, axis=1) >= 0)
    assert cu_q_lens[0, 2] == 32
    assert cu_q_lens[0, 3] == 48
    assert cu_q_lens[0, 4] == 48


def test_pcp_zigzag_page_table_multi_request():
    pcp_size = 2
    block_size = 16
    n_off = 4
    counts = [64, 64]
    chunk, off, s_live = pcp_zigzag_token_layout(counts, pcp_size, block_size=block_size)
    s_pad = 128
    table = pcp_zigzag_page_table(
        chunk, pcp_size, block_size, n_off, off=off, s_pad=s_pad
    )
    # Static width: two_p * (s_pad // 2 // block_size) = 4 * 4 = 16
    assert table.shape == (n_off, 16)
    # Req 0 and Req 1 should map to disjoint page sets in the all-gathered buffer
    req0_pages = set(table[0, :4].tolist())
    req1_pages = set(table[2, :4].tolist())
    assert req0_pages.isdisjoint(req1_pages)


def test_pcp_interleaved_token_layout():
    pcp_size = 2
    block_size = 128
    counts = [300, 600]
    chunk, off, s_live = pcp_interleaved_token_layout(
        counts, pcp_size, block_size=block_size
    )
    # quantum = 2 * 128 = 256
    # req 0: ceil(300 / 256) * 128 = 256
    # req 1: ceil(600 / 256) * 128 = 384
    assert chunk == [256, 384]
    assert off == [0, 256]
    assert s_live == 640


def test_pcp_interleaved_seq_arrays():
    pcp_size = 2
    block_size = 128
    n_off = 4
    counts = [300, 600]
    chunk, off, _ = pcp_interleaved_token_layout(
        counts, pcp_size, block_size=block_size
    )
    cu_q_lens, q_pos_offsets, pcp_write = pcp_interleaved_seq_arrays(
        chunk, off, pcp_size, n_off, counts=counts, block_size=block_size
    )
    assert cu_q_lens.shape == (pcp_size, n_off + 1)
    assert q_pos_offsets.shape == (pcp_size, n_off)
    assert pcp_write.shape == (pcp_size, n_off)

    for r in range(pcp_size):
        assert np.array_equal(cu_q_lens[r], [0, 256, 640, 640, 640])
        assert q_pos_offsets[r, 0] == r * block_size
        assert q_pos_offsets[r, 1] == r * block_size
        assert q_pos_offsets[r, 2] == 0
        assert np.array_equal(pcp_write[r], [True, True, False, False])


def test_pcp_interleaved_page_table():
    pcp_size = 2
    block_size = 128
    n_off = 2
    counts = [512, 256]
    chunk, off, s_live = pcp_interleaved_token_layout(
        counts, pcp_size, block_size=block_size
    )
    s_pad = 512
    table = pcp_interleaved_page_table(
        chunk, pcp_size, block_size, n_off, off=off, s_pad=s_pad
    )
    # max_req_pages = 512 // 128 = 4, width = 2 * 4 = 8
    assert table.shape == (n_off, 8)
    # Req 0 has c_0 = 256 (req_pages = 2, 4 logical pages):
    # p=0 (rank 0, lp 0): (0*512 + 0)//128 + 0 = 0
    # p=1 (rank 1, lp 0): (1*512 + 0)//128 + 0 = 4
    # p=2 (rank 0, lp 1): (0*512 + 0)//128 + 1 = 1
    # p=3 (rank 1, lp 1): (1*512 + 0)//128 + 1 = 5
    assert np.array_equal(table[0, :4], [0, 4, 1, 5])
    # Req 1 has c_1 = 128 (req_pages = 1, 2 logical pages, off[1] = 256):
    # p=0 (rank 0, lp 0): (0*512 + 256)//128 + 0 = 2
    # p=1 (rank 1, lp 0): (1*512 + 256)//128 + 0 = 6
    assert np.array_equal(table[1, :2], [2, 6])


def test_apply_pcp_interleaved_token_order():
    pcp_size = 2
    block_size = 128
    counts = [384]  # 3 pages of 128: p=0 (rank 0, lp 0), p=1 (rank 1, lp 0), p=2 (rank 0, lp 1)
    chunk, off, s_live = pcp_interleaved_token_layout(
        counts, pcp_size, block_size=block_size
    )
    assert chunk == [256]
    s_pad = 256
    padded_total_num_tokens = s_pad * pcp_size  # 512

    input_ids = np.zeros(padded_total_num_tokens, dtype=np.int32)
    input_ids[:384] = np.arange(1, 385, dtype=np.int32)
    positions_np = np.zeros(padded_total_num_tokens, dtype=np.int32)
    positions_np[:384] = np.arange(384, dtype=np.int32)

    rm_to_packed, packed_to_rm = apply_pcp_interleaved_token_order(
        input_ids,
        positions_np,
        counts,
        chunk,
        off,
        pcp_size,
        padded_total_num_tokens,
        block_size=block_size,
    )

    assert np.array_equal(packed_to_rm[rm_to_packed[:384]], np.arange(384))
    # Rank 0 local buffer [0:256] holds page 0 (tokens 0..127) and page 2 (tokens 256..383)
    assert np.array_equal(positions_np[0:128], np.arange(0, 128))
    assert np.array_equal(positions_np[128:256], np.arange(256, 384))
    # Rank 1 local buffer [256:512] holds page 1 (tokens 128..255) and padding (0)
    assert np.array_equal(positions_np[256:384], np.arange(128, 256))
    assert np.all(positions_np[384:512] == 0)


def test_pcp_interleaved_eligibility_and_prepare_dummy():
    eligibility = PcpSequenceLayoutEligibility(
        pcp_size=2,
        interleave_size=1,
        is_interleaved=True,
        block_size=128,
    )
    assert eligibility.enabled
    assert eligibility.is_interleaved

    decision = eligibility.evaluate_runner_chunk(
        input_batch=_input_batch(computed=[0], prompt=[128]),
        scheduler_output=_scheduler_output([128]),
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=np.array([128], dtype=np.int32),
        num_tokens_paddings=[128, 256, 512],
        max_num_tokens=512,
    )
    assert decision.mode is PcpSequenceLayoutMode.INTERLEAVED
    # Prefill buckets must be >= 2 * block_size (256)
    assert decision.local_required_tokens == 128
    assert decision.local_padded_tokens == 256

    planner = PcpSequenceLayoutPlanner(eligibility)
    plan = planner.prepare_dummy(
        num_tokens=256,
        num_reqs=2,
        kv_cache_initialized=True,
    )
    assert is_pcp_interleaved_sequence_layout_descriptor(plan.descriptor)
    assert plan.descriptor.protocol == PCP_INTERLEAVED_SEQUENCE_LAYOUT_PROTOCOL
    assert plan.pcp_metadata is not None
    assert plan.pcp_metadata.is_interleaved is True
    assert plan.pcp_metadata.gdn_packed_to_rm is not None
    assert plan.pcp_metadata.query_start_loc.shape == (2, 3)


