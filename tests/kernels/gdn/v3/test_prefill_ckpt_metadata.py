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
"""CPU tests for the GDN v3 prefill-checkpoint metadata and write-back plumbing.

The kernel exports a second state on the interior tile whose last token is
the request's vLLM prefill checkpoint offset. These tests pin the tile
derivation (per-sequence and PCP stage tiling), the compact-update scatter
and the static geometry checks; the DMA path itself needs a TPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3 import (
    config,
    memory_ref,
    metadata,
    pcp_metadata,
    pcp_wrapper,
)


def _cfg(batch_size: int, tile_size: int) -> config.GDNConfig:
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.float32.dtype,
            act_out=jnp.bfloat16.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        batch_size=batch_size,
        dim_size=384,
        kernel_size=4,
        tile_size=tile_size,
        num_kq_heads=1,
        num_v_heads=1,
        kq_head_dim=128,
        v_head_dim=128,
    )


def _per_seq_inputs(query_lens, computed, *, max_num_seqs):
    query_lens = np.asarray(query_lens, dtype=np.int32)
    computed = np.asarray(computed, dtype=np.int32)
    query_start = np.zeros((max_num_seqs + 1,), dtype=np.int32)
    query_start[1 : query_lens.size + 1] = np.cumsum(query_lens, dtype=np.int32)
    query_start[query_lens.size + 1 :] = query_start[query_lens.size]
    seq_lens = np.zeros((max_num_seqs,), dtype=np.int32)
    seq_lens[: query_lens.size] = query_lens + computed
    state_indices = np.arange(1, max_num_seqs + 1, dtype=np.int32)
    read_offsets = np.zeros((max_num_seqs,), dtype=np.int32)
    return tuple(
        jnp.asarray(x) for x in (seq_lens, query_start, state_indices, read_offsets)
    )


def _per_seq_metadata(cfg, query_lens, computed, offsets, blocks, *, start_seq=0):
    max_num_seqs = 4
    seq_lens, query_start, state_indices, read_offsets = _per_seq_inputs(
        query_lens, computed, max_num_seqs=max_num_seqs
    )
    ckpt_offsets = np.zeros((max_num_seqs,), dtype=np.int32)
    ckpt_blocks = np.zeros((max_num_seqs,), dtype=np.int32)
    ckpt_offsets[: len(offsets)] = offsets
    ckpt_blocks[: len(blocks)] = blocks
    return metadata.compute_per_seq_metadata(
        cfg,
        seq_lens,
        query_start,
        state_indices,
        read_offsets,
        jnp.asarray(start_seq, dtype=jnp.int32),
        jnp.asarray(len(query_lens), dtype=jnp.int32),
        prefill_ckpt_offsets=jnp.asarray(ckpt_offsets),
        prefill_ckpt_blocks=jnp.asarray(ckpt_blocks),
    )


def _tile_table(md):
    n = int(md.num_tiles)
    is_ckpt = (
        np.zeros((n,), dtype=bool)
        if md.p_id_is_ckpt_tile is None
        else np.asarray(md.p_id_is_ckpt_tile.data[:n])
    )
    return (
        np.asarray(md.p_id_to_s_idx.data[:n]),
        np.asarray(md.p_id_to_r_base.data[:n]),
        np.asarray(md.p_id_to_r_size.data[:n]),
        np.asarray(md.p_id_is_last_tile.data[:n]),
        is_ckpt,
    )


def test_per_seq_ckpt_tile_is_the_tile_ending_on_the_offset():
    cfg = _cfg(batch_size=1024, tile_size=64)
    md = _per_seq_metadata(
        cfg, query_lens=[300, 200], computed=[0, 0], offsets=[128, 0], blocks=[7, 0]
    )
    s_idx, r_base, r_size, is_last, is_ckpt = _tile_table(md)

    # Sequence 0 spans tiles [0, 5): the checkpoint tile is the one whose
    # last token is row 128, i.e. tile 1 (rows 64..128).
    assert int(md.num_tiles) == 5 + 4
    expected = np.zeros_like(is_ckpt)
    expected[1] = True
    np.testing.assert_array_equal(is_ckpt, expected)
    assert r_base[1] + r_size[1] == 128
    assert s_idx[1] == 0
    # A checkpoint tile is never the sequence's last tile.
    assert not np.any(is_ckpt & is_last)
    # Sequence 1 (offset 0) never flags a tile.
    assert not np.any(is_ckpt[s_idx == 1])
    np.testing.assert_array_equal(
        np.asarray(md.s_idx_to_prefill_ckpt_block), [7, 0, 0, 0]
    )


def test_per_seq_ckpt_tile_uses_query_relative_offset_of_a_resumed_chunk():
    cfg = _cfg(batch_size=1024, tile_size=64)
    # Request resumed at absolute token 6144 with a 320-token final chunk and
    # a checkpoint 192 tokens into that chunk; the preceding request pushes
    # its query start to row 100, which must not shift the match.
    md = _per_seq_metadata(
        cfg,
        query_lens=[100, 320],
        computed=[0, 6144],
        offsets=[0, 192],
        blocks=[0, 9],
    )
    s_idx, r_base, r_size, is_last, is_ckpt = _tile_table(md)
    flagged = np.nonzero(is_ckpt)[0]
    assert flagged.tolist() == [2 + 2]  # 2 tiles of seq 0, then tile 2 of seq 1
    assert s_idx[flagged[0]] == 1
    assert r_base[flagged[0]] + r_size[flagged[0]] == 100 + 192
    assert not np.any(is_ckpt & is_last)


def test_per_seq_ckpt_rolls_with_start_seq():
    cfg = _cfg(batch_size=1024, tile_size=64)
    # start_seq=1: sequence 0 is a decode handled by the batched kernel, so
    # the per-sequence arrays are rolled and sequence 1 becomes slot 0.
    md = _per_seq_metadata(
        cfg,
        query_lens=[1, 256, 128],
        computed=[10, 0, 0],
        offsets=[0, 128, 64],
        blocks=[0, 11, 12],
        start_seq=1,
    )
    s_idx, r_base, r_size, is_last, is_ckpt = _tile_table(md)
    assert int(md.num_tiles) == 4 + 2
    np.testing.assert_array_equal(is_ckpt, [False, True, False, False, True, False])
    np.testing.assert_array_equal(
        np.asarray(md.s_idx_to_prefill_ckpt_block)[:3], [11, 12, 0]
    )
    assert not np.any(is_ckpt & is_last)


def test_per_seq_offset_off_the_tile_grid_flags_no_tile():
    cfg = _cfg(batch_size=1024, tile_size=64)
    md = _per_seq_metadata(
        cfg, query_lens=[300], computed=[0], offsets=[100], blocks=[7]
    )
    _, _, _, _, is_ckpt = _tile_table(md)
    assert not np.any(is_ckpt)


def test_per_seq_without_checkpoints_keeps_metadata_fields_none():
    cfg = _cfg(batch_size=256, tile_size=64)
    seq_lens, query_start, state_indices, read_offsets = _per_seq_inputs(
        [128], [0], max_num_seqs=4
    )
    md = metadata.compute_per_seq_metadata(
        cfg,
        seq_lens,
        query_start,
        state_indices,
        read_offsets,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    assert md.p_id_is_ckpt_tile is None
    assert md.s_idx_to_prefill_ckpt_block is None


def test_metadata_ref_requires_both_checkpoint_arrays():
    cfg = _cfg(batch_size=256, tile_size=64)
    zeros = jnp.zeros((256,), dtype=jnp.int32)
    flags = jnp.zeros((256,), dtype=jnp.bool_)
    with pytest.raises(ValueError, match="set together"):
        memory_ref.MetadataRef.create(
            cfgs=cfg,
            num_tiles=jnp.asarray(1),
            p_id_to_s_idx=zeros,
            p_id_to_r_base=zeros,
            p_id_to_r_size=zeros,
            p_id_is_first_tile=flags,
            p_id_is_last_tile=flags,
            s_idx_has_initial_state=flags[:4],
            s_idx_to_state_indices=zeros[:4],
            s_idx_to_read_offset=zeros[:4],
            p_id_is_ckpt_tile=flags,
        )


def test_pcp_stage_tiles_flag_the_tile_ending_on_the_batch_flat_ckpt_row():
    cfg = _cfg(batch_size=128, tile_size=4)
    max_num_seqs = 4
    seq_lens, query_start, state_indices, _ = _per_seq_inputs(
        [40, 20], [0, 0], max_num_seqs=max_num_seqs
    )
    # Request 0 checkpoints 16 tokens in (flat row 16, a rank-slot boundary
    # with comm_chunk_size=4); request 1 checkpoints 8 tokens in, flat row
    # 48 (its query starts at 40, a slot boundary, so row 48 is one too).
    offsets = jnp.asarray([16, 8, 0, 0], dtype=jnp.int32)
    blocks = jnp.asarray([5, 6, 0, 0], dtype=jnp.int32)
    md, _ = pcp_metadata.compute_pcp_stage_metadata(
        cfg,
        seq_lens,
        query_start,
        state_indices,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(2, dtype=jnp.int32),
        pcp_size=4,
        comm_chunk_size=4,
        projection_token_block_size=16,
        num_qkv_out_blocks=6,
        num_projection_out_blocks=8,
        prefill_ckpt_offsets=offsets,
        prefill_ckpt_blocks=blocks,
    )
    s_idx, r_base, r_size, is_last, is_ckpt = _tile_table(md)
    flagged = np.nonzero(is_ckpt)[0]
    assert len(flagged) == 2
    ends = (r_base + r_size)[flagged]
    np.testing.assert_array_equal(ends, [16, 48])
    np.testing.assert_array_equal(s_idx[flagged], [0, 1])
    assert not np.any(is_ckpt & is_last)
    np.testing.assert_array_equal(np.asarray(md.s_idx_to_prefill_ckpt_block), blocks)


def test_pcp_stage_metadata_without_checkpoints_is_unchanged():
    cfg = _cfg(batch_size=128, tile_size=4)
    seq_lens, query_start, state_indices, _ = _per_seq_inputs(
        [40, 20], [0, 0], max_num_seqs=4
    )
    md, _ = pcp_metadata.compute_pcp_stage_metadata(
        cfg,
        seq_lens,
        query_start,
        state_indices,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(2, dtype=jnp.int32),
        pcp_size=4,
        comm_chunk_size=4,
        projection_token_block_size=16,
        num_qkv_out_blocks=6,
        num_projection_out_blocks=8,
    )
    assert md.p_id_is_ckpt_tile is None
    assert md.s_idx_to_prefill_ckpt_block is None


def _pcp_metadata(cfg, query_lens, offsets, blocks, *, pcp_size, comm_chunk_size):
    max_num_seqs = 4
    seq_lens, query_start, state_indices, _ = _per_seq_inputs(
        query_lens, [0] * len(query_lens), max_num_seqs=max_num_seqs
    )
    kwargs = {}
    if offsets is not None:
        ckpt_offsets = np.zeros((max_num_seqs,), dtype=np.int32)
        ckpt_blocks = np.zeros((max_num_seqs,), dtype=np.int32)
        ckpt_offsets[: len(offsets)] = offsets
        ckpt_blocks[: len(blocks)] = blocks
        kwargs = dict(
            prefill_ckpt_offsets=jnp.asarray(ckpt_offsets),
            prefill_ckpt_blocks=jnp.asarray(ckpt_blocks),
        )
    md, stages = pcp_metadata.compute_pcp_stage_metadata(
        cfg,
        seq_lens,
        query_start,
        state_indices,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(len(query_lens), dtype=jnp.int32),
        pcp_size=pcp_size,
        comm_chunk_size=comm_chunk_size,
        projection_token_block_size=16,
        num_qkv_out_blocks=6,
        num_projection_out_blocks=8,
        **kwargs,
    )
    return md, stages, np.asarray(query_start)


def _assert_tiles_cover_requests(md, query_start, num_reqs, *, tile_size):
    """Tiles of each request partition its rows in order, within tile size."""
    s_idx, r_base, r_size, is_last, _ = _tile_table(md)
    is_first = np.asarray(md.p_id_is_first_tile.data[: int(md.num_tiles)])
    assert np.all(r_size >= 1) and np.all(r_size <= tile_size)
    for req in range(num_reqs):
        rows = np.nonzero(s_idx == req)[0]
        assert r_base[rows[0]] == query_start[req]
        np.testing.assert_array_equal(r_base[rows[1:]], (r_base + r_size)[rows[:-1]])
        assert (r_base + r_size)[rows[-1]] == query_start[req + 1]
        np.testing.assert_array_equal(is_first[rows], np.arange(len(rows)) == 0)
        np.testing.assert_array_equal(
            is_last[rows], np.arange(len(rows)) == len(rows) - 1
        )


def _tiles(md):
    n = int(md.num_tiles)
    return [
        (int(a), int(b), int(c))
        for a, b, c in zip(
            np.asarray(md.p_id_to_s_idx.data[:n]),
            np.asarray(md.p_id_to_r_base.data[:n]),
            np.asarray(md.p_id_to_r_size.data[:n]),
        )
    ]


def test_pcp_splits_only_the_tile_straddling_the_checkpoint_row():
    # tile 8, slots of 16 rows over 4 ranks. Request 1 starts at flat row 13
    # and checkpoints 24 tokens in (flat row 37), inside rank 2's slot
    # [32, 48) of round 0 where the slot-aligned tiles are [32, 40), [40, 48).
    cfg = _cfg(batch_size=256, tile_size=8)
    plain, _, query_start = _pcp_metadata(
        cfg, [13, 60], None, None, pcp_size=4, comm_chunk_size=16
    )
    split, _, _ = _pcp_metadata(
        cfg, [13, 60], [0, 24], [0, 9], pcp_size=4, comm_chunk_size=16
    )
    _assert_tiles_cover_requests(split, query_start, 2, tile_size=8)
    plain_tiles = _tiles(plain)
    split_tiles = _tiles(split)
    assert (1, 32, 8) in plain_tiles
    # Exactly that tile became [32, 37) + [37, 40); everything else is the
    # pre-existing schedule, in order.
    expected = []
    for tile in plain_tiles:
        if tile == (1, 32, 8):
            expected += [(1, 32, 5), (1, 37, 3)]
        else:
            expected.append(tile)
    assert split_tiles == expected
    assert int(split.num_tiles) == int(plain.num_tiles) + 1
    s_idx, r_base, r_size, is_last, is_ckpt = _tile_table(split)
    flagged = np.nonzero(is_ckpt)[0]
    assert len(flagged) == 1
    assert s_idx[flagged[0]] == 1
    assert (r_base + r_size)[flagged[0]] == 37
    assert not np.any(is_ckpt & is_last)
    # Only the second half of the split tile starts off the slot-aligned
    # grid (the aligned fast path covers every other tile).
    unaligned = [t for t in split_tiles if t[1] % 8 != 0 and t[1] != 13]
    assert unaligned == [(1, 37, 3)]


def test_pcp_checkpoint_on_a_natural_tile_end_adds_no_tile():
    cfg = _cfg(batch_size=256, tile_size=8)
    plain, _, _ = _pcp_metadata(
        cfg, [16, 60], None, None, pcp_size=4, comm_chunk_size=16
    )
    # Request 1 starts at 16 (slot boundary); checkpoint 24 in -> row 40,
    # already a tile end of the slot-aligned grid.
    same, _, query_start = _pcp_metadata(
        cfg, [16, 60], [0, 24], [0, 9], pcp_size=4, comm_chunk_size=16
    )
    assert _tiles(same) == _tiles(plain)
    _, r_base, r_size, _, is_ckpt = _tile_table(same)
    assert (r_base + r_size)[np.nonzero(is_ckpt)[0]].tolist() == [40]


def test_pcp_schedule_unchanged_without_checkpointed_requests():
    cfg = _cfg(batch_size=256, tile_size=8)
    plain, _, query_start = _pcp_metadata(
        cfg, [13, 60], None, None, pcp_size=4, comm_chunk_size=16
    )
    gated, _, _ = _pcp_metadata(
        cfg, [13, 60], [0, 0], [0, 0], pcp_size=4, comm_chunk_size=16
    )
    assert _tiles(gated) == _tiles(plain)
    _assert_tiles_cover_requests(plain, query_start, 2, tile_size=8)


def test_pcp_split_tiles_fit_the_schedule_capacity():
    cfg = _cfg(batch_size=256, tile_size=8)
    md, _, query_start = _pcp_metadata(
        cfg, [3, 250], [0, 248], [0, 9], pcp_size=4, comm_chunk_size=16
    )
    _assert_tiles_cover_requests(md, query_start, 2, tile_size=8)
    assert int(md.num_tiles) <= md.p_id_to_s_idx.data.shape[0]
    _, r_base, r_size, _, is_ckpt = _tile_table(md)
    assert (r_base + r_size)[np.nonzero(is_ckpt)[0]].tolist() == [3 + 248]


@pytest.mark.parametrize(
    ("tile_size", "ok"),
    [(64, True), (128, True), (256, True), (96, False), (512, False)],
)
def test_pcp_prefill_ckpt_tiling_geometry(tile_size, ok):
    if ok:
        pcp_wrapper._validate_prefill_ckpt_tiling(tile_size=tile_size)
    else:
        with pytest.raises(ValueError):
            pcp_wrapper._validate_prefill_ckpt_tiling(tile_size=tile_size)


def test_receive_group_stride_covers_rephased_tile_starts():
    aligned = pcp_wrapper._qkv_receive_group_stride(512, 64)
    rephased = pcp_wrapper._qkv_receive_group_stride(
        512, 64, request_relative_tiles=True
    )
    # A tile may start on the slot's last row and loads cdiv(7 + 64, 8)
    # groups from there.
    assert rephased == 511 // 8 + 9
    assert rephased >= aligned
    assert pcp_wrapper._qkv_receive_group_stride(16, 4) == (
        pcp_wrapper._qkv_receive_group_stride(16, 4, request_relative_tiles=False)
    )


def test_compact_scatter_writes_checkpoint_slot_only_for_offset_requests():
    pool = jnp.zeros((8, 4, 2), dtype=jnp.float32)
    region = config.StateRegion(
        kb0=0,
        nblocks=1,
        row0=0,
        nrows=4,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=4,
    )
    # Two sequences, two slots each: slot 0 = final state, slot 1 = checkpoint.
    updates = jnp.stack(
        [
            jnp.stack([jnp.full((1, 4, 2), 10.0), jnp.full((1, 4, 2), 11.0)]),
            jnp.stack([jnp.full((1, 4, 2), 20.0), jnp.full((1, 4, 2), 21.0)]),
        ]
    )
    actual = pcp_wrapper._scatter_compact_state_updates(
        pool,
        updates,
        jnp.asarray([3, 5], dtype=jnp.int32),
        jnp.asarray([0, 300, 420], dtype=jnp.int32),
        jnp.asarray(2, dtype=jnp.int32),
        state_stride=1,
        region=region,
        whole_block_dma=False,
        # Sequence 1 carries a non-zero block but no offset: its checkpoint
        # slot holds unwritten bytes and must stay out of the pool.
        prefill_ckpt_offsets=jnp.asarray([256, 0], dtype=jnp.int32),
        prefill_ckpt_blocks=jnp.asarray([2, 6], dtype=jnp.int32),
    )
    actual = np.asarray(actual)
    assert np.all(actual[3] == 10.0)
    assert np.all(actual[2] == 11.0)
    assert np.all(actual[5] == 20.0)
    assert np.all(actual[6] == 0.0)
    assert np.all(actual[[0, 1, 4, 7]] == 0.0)


def test_compact_scatter_rejects_single_slot_updates_with_checkpoints():
    pool = jnp.zeros((4, 4, 2), dtype=jnp.float32)
    region = config.StateRegion(
        kb0=0,
        nblocks=1,
        row0=0,
        nrows=4,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=4,
    )
    with pytest.raises(ValueError, match="second compact update slot"):
        pcp_wrapper._scatter_compact_state_updates(
            pool,
            jnp.zeros((1, 1, 1, 4, 2), dtype=jnp.float32),
            jnp.asarray([1], dtype=jnp.int32),
            jnp.asarray([0, 8], dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            state_stride=1,
            region=region,
            whole_block_dma=False,
            prefill_ckpt_offsets=jnp.asarray([4], dtype=jnp.int32),
            prefill_ckpt_blocks=jnp.asarray([2], dtype=jnp.int32),
        )


def test_compact_update_shape_adds_checkpoint_slot():
    pool = jax.ShapeDtypeStruct((8, 256, 1, 4, 128), jnp.float8_e4m3fn)
    region = config.StateRegion(
        kb0=0,
        nblocks=2,
        row0=0,
        nrows=16,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=16,
    )
    shape = pcp_wrapper._compact_state_updates_shape(
        pool, 3, region, whole_block_dma=False, num_slots=2
    )
    assert shape.shape == (3, 2, 2, 16, 1, 4, 128)
