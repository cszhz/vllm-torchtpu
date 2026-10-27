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
"""Host-side derivation of the GDN prefill checkpoint (offset, block) rows."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.kernels.gdn.v3.config import PREFILL_CKPT_ALIGNMENT
from vllm_torchtpu.layers.core import attention_metadata
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
    mamba_prefill_checkpoint_rows,
)

_BLOCK = 3072


def _rows(seq_lens, query_lens, block_table, *, checkpoint_alignment=None):
    seq_lens = np.asarray(seq_lens, dtype=np.int32)
    query_lens = np.asarray(query_lens, dtype=np.int32)
    query_start_loc = np.concatenate(
        [[0], np.cumsum(query_lens, dtype=np.int32)]
    ).astype(np.int32)
    return mamba_prefill_checkpoint_rows(
        seq_lens,
        query_start_loc,
        torch.as_tensor(block_table, dtype=torch.int32),
        hash_block_size=_BLOCK,
        mamba_block_size=_BLOCK,
        cache_hit_alignment_tokens=_BLOCK,
        checkpoint_alignment=checkpoint_alignment or PREFILL_CKPT_ALIGNMENT,
        drop_eagle_block=False,
    )


def test_full_prompt_checkpoints_at_last_block_boundary():
    # 9455-token prompt in one chunk: state@9216 -> manager column
    # cdiv(9455, 3072) - 2 = 2, the running state goes to column 3.
    offsets, blocks = _rows([9455], [9455], [[0, 0, 42, 43]])
    np.testing.assert_array_equal(offsets, [9216])
    np.testing.assert_array_equal(blocks, [42])


def test_resumed_final_chunk_uses_query_relative_offset():
    # Same prompt, final chunk [6144, 9455): the checkpoint is 3072 tokens
    # into the chunk, still column 2.
    offsets, blocks = _rows([9455], [9455 - 6144], [[0, 0, 42, 43]])
    np.testing.assert_array_equal(offsets, [3072])
    np.testing.assert_array_equal(blocks, [42])


def test_block_multiple_prompt_checkpoints_one_block_below_the_end():
    # Upstream leaves the last token to recompute on resend, so a 9216-token
    # prompt's cacheable boundary is 6144 (column 1); column 2 is running.
    offsets, blocks = _rows([9216], [9216], [[0, 41, 42, 0]])
    np.testing.assert_array_equal(offsets, [6144])
    np.testing.assert_array_equal(blocks, [41])


def test_chunk_not_reaching_prefill_end_has_no_allocated_column():
    # A block-aligned interior chunk [0, 6144) would nominally checkpoint at
    # 3072 (column 0), but the manager only reserves the checkpoint block
    # for the chunk reaching prefill end; column 0 is the null block here,
    # so the row must come back as "none" rather than target block 0.
    offsets, blocks = _rows([6144], [6144], [[0, 42, 0, 0]])
    np.testing.assert_array_equal(offsets, [0])
    np.testing.assert_array_equal(blocks, [0])


def test_short_or_unaligned_queries_have_no_checkpoint():
    offsets, blocks = _rows(
        [3000, 9455, 1],
        [3000, 9455 - 6000, 1],
        [[0, 0, 0, 0], [0, 0, 42, 43], [44, 0, 0, 0]],
    )
    # 3000 tokens: no boundary at least one hash block in; 6000-start:
    # not hash aligned; decode: nothing.
    np.testing.assert_array_equal(offsets, [0, 0, 0])
    np.testing.assert_array_equal(blocks, [0, 0, 0])


def test_second_request_behind_an_unaligned_chunk_is_accepted():
    # The second request starts at flat row 9455; offsets are query-relative
    # and the kernels tile each checkpointed request on its own grid, so no
    # batch-flat alignment is required.
    seq_lens = [9455, 9455]
    query_lens = [9455, 9455]
    table = [[0, 0, 42, 43], [0, 0, 52, 53]]
    offsets, blocks = _rows(seq_lens, query_lens, table)
    np.testing.assert_array_equal(offsets, [9216, 9216])
    np.testing.assert_array_equal(blocks, [42, 52])


def test_offset_off_the_alignment_grid_is_a_loud_error(monkeypatch):
    # Last-resort guard: upstream's own validity rule already declines such
    # checkpoints, so this only trips if the rules diverge; it must never
    # degrade into a silently skipped write.
    monkeypatch.setattr(
        attention_metadata,
        "compute_mamba_prefill_checkpoints",
        lambda *a, **k: ([9000], [2]),
    )
    with pytest.raises(ValueError, match="multiples of"):
        _rows([9455], [9455], [[0, 0, 42, 43]])


def _runner_mock(block_table: torch.Tensor):
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner.most_model_len = None
    runner.max_num_reqs = block_table.shape[0]
    runner.position_ids = torch.zeros((8,), dtype=torch.int32)
    runner._unfenced_h2d_sources = []
    runner._free_h2d_sources = {}
    runner._registered_h2d_sources = set()
    runner.attn_groups = []
    runner._unified_kv_layout = True
    bt = MagicMock()
    bt.max_num_blocks_per_req = block_table.shape[1]
    bt.get_cpu_tensor.return_value = block_table
    runner.input_batch.block_table = [bt]
    return runner


def _mamba_builder(runner, *, num_prefill_checkpoint_blocks: int):
    spec = MambaSpec(
        block_size=_BLOCK,
        shapes=[(4, 128), (8, 64, 32)],
        dtypes=[torch.bfloat16, torch.float32],
        page_size_padded=16384,
        num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
        prefill_checkpoint_alignment=(
            PREFILL_CKPT_ALIGNMENT if num_prefill_checkpoint_blocks else None
        ),
    )
    vllm_config = MagicMock()
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.cache_config.hash_block_size = _BLOCK
    vllm_config.cache_config.cache_hit_alignment_tokens = _BLOCK
    vllm_config.speculative_config = None
    return AttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["gdn.0"],
        vllm_config=vllm_config,
        device=runner.device,
        runner=runner,
        kv_cache_group_id=0,
    )


def _ctx(seq_lens, query_lens, *, host: bool = True, **kwargs):
    seq_lens = np.asarray(seq_lens, dtype=np.int32)
    qsl = np.concatenate([[0], np.cumsum(query_lens, dtype=np.int32)]).astype(np.int32)
    num_reqs = seq_lens.size
    return AttentionMetadataBuilderContext(
        num_reqs=num_reqs,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.from_numpy(seq_lens),
        query_start_loc=torch.from_numpy(qsl),
        request_distribution=torch.tensor([0, 0, num_reqs], dtype=torch.int32),
        seq_lens_cpu=seq_lens if host else None,
        query_start_loc_cpu=qsl if host else None,
        **kwargs,
    )


def _common(num_reqs):
    cm = MagicMock()
    cm.num_reqs = num_reqs
    return cm


def test_builder_emits_checkpoint_tensors_from_host_arrays():
    table = torch.tensor([[0, 0, 42, 43], [51, 0, 0, 0]], dtype=torch.int32)
    runner = _runner_mock(table)
    builder = _mamba_builder(runner, num_prefill_checkpoint_blocks=1)
    runner._attn_metadata_builder_ctx = _ctx([9455, 1], [9455, 1])

    meta = builder.build(common_prefix_len=0, common_attn_metadata=_common(2))

    assert meta.mamba_prefill_ckpt_offsets is not None
    assert meta.mamba_prefill_ckpt_blocks is not None
    np.testing.assert_array_equal(meta.mamba_prefill_ckpt_offsets.numpy(), [9216, 0])
    np.testing.assert_array_equal(meta.mamba_prefill_ckpt_blocks.numpy(), [42, 0])
    assert meta.mamba_prefill_ckpt_offsets.dtype == torch.int32
    # The running state still comes from the positional column.
    np.testing.assert_array_equal(meta.mamba_state_indices.numpy(), [43, 51])


def test_builder_dummy_walk_uploads_zero_checkpoints():
    table = torch.tensor([[0, 0, 42, 43]], dtype=torch.int32)
    runner = _runner_mock(table)
    builder = _mamba_builder(runner, num_prefill_checkpoint_blocks=1)
    runner._attn_metadata_builder_ctx = _ctx(
        [9455],
        [9455],
        host=False,
        position_ids_override=torch.zeros((8,), dtype=torch.int32),
    )

    meta = builder.build(common_prefix_len=0, common_attn_metadata=_common(1))

    np.testing.assert_array_equal(meta.mamba_prefill_ckpt_offsets.numpy(), [0])
    np.testing.assert_array_equal(meta.mamba_prefill_ckpt_blocks.numpy(), [0])


def test_builder_real_walk_without_host_arrays_fails_loudly():
    table = torch.tensor([[0, 0, 42, 43]], dtype=torch.int32)
    runner = _runner_mock(table)
    builder = _mamba_builder(runner, num_prefill_checkpoint_blocks=1)
    runner._attn_metadata_builder_ctx = _ctx([9455], [9455], host=False)

    with pytest.raises(NotImplementedError, match="host seq_lens"):
        builder.build(common_prefix_len=0, common_attn_metadata=_common(1))


def test_builder_without_declared_blocks_emits_none():
    table = torch.tensor([[0, 0, 42, 43]], dtype=torch.int32)
    runner = _runner_mock(table)
    builder = _mamba_builder(runner, num_prefill_checkpoint_blocks=0)
    runner._attn_metadata_builder_ctx = _ctx([9455], [9455])

    meta = builder.build(common_prefix_len=0, common_attn_metadata=_common(1))

    assert meta.mamba_prefill_ckpt_offsets is None
    assert meta.mamba_prefill_ckpt_blocks is None
