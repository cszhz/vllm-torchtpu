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
"""Tests for PCP (Prefill Context Parallelism) batched RPA attention adapter."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.kernels.experimental.batched_rpa import configs
from vllm_torchtpu.layers.adapter import attention, cp_attention
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    PCPMetadata,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


@pytest.fixture
def cpu_vllm_config_context():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config), set_vllm_model_wrapper_context(
        mesh=None, vllm_config=config
    ):
        yield


def test_make_cu_cache():
    # 2 virtual sequences per request, e.g. [0, 128, 256] -> merged request [0, 256, 256]
    cu = torch.tensor([0, 128, 256], dtype=torch.int32)
    cu_cache = cp_attention._make_cu_cache(cu, num_seqs=2)
    assert cu_cache.tolist() == [0, 256, 256]


def test_per_request_helpers():
    num_seqs = 4
    # 1D seq_lens
    seq_lens = torch.tensor([100, 100, 200, 200], dtype=torch.int32)
    req_lens = cp_attention._per_request_1d(seq_lens, num_seqs)
    assert req_lens.shape == (num_seqs,)
    assert req_lens.tolist() == [100, 200, 0, 0]

    # 2D block_tables flattened
    bt = torch.tensor(
        [
            [1, 2],
            [1, 2],
            [3, 4],
            [3, 4],
        ],
        dtype=torch.int32,
    ).reshape(-1)
    req_bt = cp_attention._per_request_pages(bt, num_seqs)
    req_bt_2d = req_bt.reshape(num_seqs, 2)
    assert req_bt_2d[0].tolist() == [1, 2]
    assert req_bt_2d[1].tolist() == [3, 4]
    assert req_bt_2d[2].tolist() == [0, 0]
    assert req_bt_2d[3].tolist() == [0, 0]


def test_lse_weighted_combine():
    T, H, D = 4, 2, 64
    out_a = torch.ones((T, H, D), dtype=torch.float32)
    lse_a = torch.zeros((T, H), dtype=torch.float32)  # exp(0) = 1

    out_b = torch.full((T, H, D), 3.0, dtype=torch.float32)
    lse_b = torch.zeros((T, H), dtype=torch.float32)  # exp(0) = 1

    # Equal weights -> average = (1 + 3) / 2 = 2.0
    combined_out, combined_lse = cp_attention.lse_weighted_combine(
        out_a, lse_a, out_b, lse_b
    )
    assert torch.allclose(combined_out, torch.full((T, H, D), 2.0))
    # log(exp(0) + exp(0)) = log(2)
    assert torch.allclose(combined_lse, torch.full((T, H), float(np.log(2.0))))


def test_build_pcp_kernels_mock():
    with patch(
        "vllm_torchtpu.layers.adapter.cp_attention.get_or_create_pcp_mesh",
        return_value=None,
    ), patch(
        "torch_tpu._internal.pallas.jax_op"
    ) as mock_jax_op:
        mock_op = MagicMock()
        mock_jax_op.return_value = mock_op

        cache_k, new_k = cp_attention.build_pcp_kernels(
            sliding_window=None,
            sm_scale=1.0,
            logits_soft_cap=None,
            q_scale=None,
            k_scale=None,
            v_scale=None,
            cp_group_size=2,
            cp_rank=0,
            use_causal_mask=True,
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        )
        assert callable(cache_k)
        assert callable(new_k)


def test_pcp_forward_dispatch(cpu_vllm_config_context):
    backend = attention.PallasBatchedRPAAttentionBackendImpl(
        num_heads=4,
        head_size=128,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bfloat16",
    )

    query = torch.ones(8, 4, 128)
    key = torch.ones(8, 2, 128)
    value = torch.ones(8, 2, 128)
    kv_cache = torch.zeros(4, 16, 2, 1, 128)

    md = AttentionMetadata(
        input_positions=torch.arange(8, dtype=torch.int32),
        block_tables=torch.zeros(2, 4, dtype=torch.int32),
        seq_lens=torch.tensor([8, 8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
        pcp=PCPMetadata(
            query_start_loc=torch.tensor([[0, 4, 8], [0, 4, 8]], dtype=torch.int32),
            kv_cache_lens=torch.tensor([0, 0], dtype=torch.int32),
            q_pos_offsets=torch.tensor([[0, 4], [2, 6]], dtype=torch.int32),
            cache_pages=0,
            new_kv_page_indices=torch.zeros(8, dtype=torch.int32),
            update_kv_cache=torch.tensor([[False, True], [False, True]]),
        ),
    )

    layer = MagicMock()
    layer._k_scale_float = None
    layer._v_scale_float = None

    expected_out = torch.ones(8, 4, 128)
    with patch.object(backend, "_run_pcp_forward", return_value=expected_out) as mock_pcp:
        out = backend.forward(layer, query, key, value, kv_cache, md)
        mock_pcp.assert_called_once()
        assert torch.allclose(out, expected_out)


def test_forward_with_pcp_interleaved_prefill():
    query = torch.ones(256, 4, 128)
    key = torch.ones(256, 2, 128)
    value = torch.ones(256, 2, 128)
    kv_cache = torch.zeros(16, 128, 2, 1, 128)

    pcp_meta = SimpleNamespace(
        query_start_loc=torch.tensor(
            [[0, 128, 256, 256], [0, 128, 256, 256]], dtype=torch.int32
        ),
        kv_cache_lens=torch.tensor([128, 0, 0], dtype=torch.int32),
        q_pos_offsets=torch.tensor(
            [[0, 0, 0], [128, 128, 0]], dtype=torch.int32
        ),
        cache_pages=2,
        new_kv_page_indices=torch.arange(6, dtype=torch.int32),
        update_kv_cache=torch.tensor(
            [[1, 1, 0], [1, 1, 0]], dtype=torch.int32
        ),
        is_decode=False,
        is_interleaved=True,
    )
    md = SimpleNamespace(
        seq_lens=torch.tensor([384, 256, 0], dtype=torch.int32),
        block_tables=torch.tensor(
            [[1, 2], [3, 4], [0, 0]], dtype=torch.int32
        ).reshape(-1),
        query_start_loc=torch.tensor([0, 128, 256, 256], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
        pcp=pcp_meta,
    )

    layer = MagicMock()
    layer._k_scale_float = None
    layer._v_scale_float = None

    cache_calls = []
    new_calls = []

    def fake_cache_kernel(
        kv_c, q, k, v, seq_lens, block_tables, cu_q_lens, dist, kv_new_lens
    ):
        cache_calls.append(
            (
                seq_lens.clone(),
                block_tables.clone(),
                cu_q_lens.clone(),
                dist.clone(),
                kv_new_lens.clone(),
            )
        )
        return torch.ones_like(q), torch.zeros((q.shape[0], q.shape[1]))

    def fake_new_kernel(
        kv_c,
        q,
        k,
        v,
        seq_lens,
        block_tables,
        cu_q_lens,
        dist,
        kv_new_lens,
        q_positions,
        update_kv_cache,
        new_kv_pages,
    ):
        new_calls.append(
            (
                seq_lens.clone(),
                block_tables.clone(),
                cu_q_lens.clone(),
                dist.clone(),
                kv_new_lens.clone(),
                q_positions.clone(),
                update_kv_cache.clone(),
            )
        )
        return torch.full_like(q, 3.0), torch.zeros((q.shape[0], q.shape[1]))

    with patch(
        "vllm_torchtpu.layers.adapter.cp_attention.build_pcp_kernels",
        return_value=(fake_cache_kernel, fake_new_kernel),
    ) as mock_build, patch(
        "vllm_torchtpu.layers.adapter.cp_attention.get_pcp_group",
        return_value=None,
    ), patch(
        "vllm_torchtpu.layers.adapter.cp_attention.all_gather_equal_tokens",
        side_effect=lambda x, dim=0: torch.cat([x, x], dim=dim),
    ), patch(
        "vllm_torchtpu.layers.adapter.cp_attention.synchronize_tensors"
    ):
        out = cp_attention.forward_with_pcp(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=md,
            sliding_window=None,
            sm_scale=1.0,
            logits_soft_cap=None,
            kv_cache_quantized_dtype=None,
            pcp_world_size=2,
            pcp_rank=1,
            cp_interleaved=True,
        )

    assert mock_build.call_args.kwargs["cp_interleaved"] is True
    assert len(cache_calls) == 1
    c_seq, c_bt, c_cu, c_dist, c_new = cache_calls[0]
    assert c_seq.tolist() == [384, 256, 0]
    assert c_bt.tolist() == [1, 2, 3, 4, 0, 0]
    assert c_cu.tolist() == [0, 128, 256, 256]
    assert c_dist.tolist() == [0, 0, 2]
    assert c_new.tolist() == [256, 256, 0]

    assert len(new_calls) == 1
    n_seq, n_bt, n_cu, n_dist, n_new, n_qpos, n_upd = new_calls[0]
    assert n_seq.tolist() == [384, 256, 0]
    assert n_bt.tolist() == [1, 2, 3, 4, 0, 0]
    assert n_cu.tolist() == [0, 128, 256, 256]
    assert n_dist.tolist() == [0, 0, 2]
    assert n_new.tolist() == [256, 256, 0]
    assert n_qpos.tolist() == [256, 128, 0]
    assert n_upd.tolist() == [1, 1, 0]
    assert torch.allclose(out, torch.full_like(query, 2.0))


def test_forward_with_pcp_interleaved_decode():
    query = torch.ones(4, 4, 128)
    key = torch.ones(4, 2, 128)
    value = torch.ones(4, 2, 128)
    kv_cache = torch.zeros(16, 128, 2, 1, 128)

    pcp_meta = SimpleNamespace(
        query_start_loc=torch.tensor(
            [[0, 1, 2, 2, 2], [0, 1, 2, 2, 2]], dtype=torch.int32
        ),
        kv_cache_lens=torch.tensor([256, 256, 0, 0], dtype=torch.int32),
        q_pos_offsets=torch.tensor(
            [[0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int32
        ),
        cache_pages=2,
        new_kv_page_indices=torch.empty(0, dtype=torch.int32),
        update_kv_cache=torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32
        ),
        is_decode=True,
        is_interleaved=True,
    )
    md = SimpleNamespace(
        seq_lens=torch.tensor([257, 385, 0, 0], dtype=torch.int32),
        block_tables=torch.tensor(
            [[1, 2], [3, 4], [0, 0], [0, 0]], dtype=torch.int32
        ).reshape(-1),
        query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        pcp=pcp_meta,
    )

    layer = MagicMock()
    layer._k_scale_float = None
    layer._v_scale_float = None

    cache_calls = []
    new_calls = []

    def fake_cache_kernel(
        kv_c, q, k, v, seq_lens, block_tables, cu_q_lens, dist, kv_new_lens
    ):
        cache_calls.append(
            (
                seq_lens.clone(),
                block_tables.clone(),
                cu_q_lens.clone(),
                dist.clone(),
                kv_new_lens.clone(),
            )
        )
        return torch.ones_like(q), torch.zeros((q.shape[0], q.shape[1]))

    def fake_new_kernel(
        kv_c,
        q,
        k,
        v,
        seq_lens,
        block_tables,
        cu_q_lens,
        dist,
        kv_new_lens,
        q_positions,
        update_kv_cache,
        new_kv_pages,
    ):
        new_calls.append(
            (
                seq_lens.clone(),
                block_tables.clone(),
                cu_q_lens.clone(),
                dist.clone(),
                kv_new_lens.clone(),
                q_positions.clone(),
                update_kv_cache.clone(),
            )
        )
        return torch.full_like(q, 3.0), torch.zeros((q.shape[0], q.shape[1]))

    with patch(
        "vllm_torchtpu.layers.adapter.cp_attention.build_pcp_kernels",
        return_value=(fake_cache_kernel, fake_new_kernel),
    ) as mock_build, patch(
        "vllm_torchtpu.layers.adapter.cp_attention.get_pcp_group",
        return_value=None,
    ), patch(
        "vllm_torchtpu.layers.adapter.cp_attention.synchronize_tensors"
    ):
        out = cp_attention.forward_with_pcp(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=md,
            sliding_window=None,
            sm_scale=1.0,
            logits_soft_cap=None,
            kv_cache_quantized_dtype=None,
            pcp_world_size=2,
            pcp_rank=0,
        )

    assert mock_build.call_args.kwargs["cp_interleaved"] is True
    assert len(cache_calls) == 1
    c_seq, c_bt, c_cu, c_dist, c_new = cache_calls[0]
    assert c_seq.tolist() == [257, 385, 0, 0]
    assert c_bt.tolist() == [1, 2, 3, 4, 0, 0, 0, 0]
    assert c_cu.tolist() == [0, 1, 2, 2, 2]
    assert c_dist.tolist() == [2, 2, 2]
    assert c_new.tolist() == [1, 129, 0, 0]

    assert len(new_calls) == 1
    n_seq, n_bt, n_cu, n_dist, n_new, n_qpos, n_upd = new_calls[0]
    assert n_seq.tolist() == [257, 385, 0, 0]
    assert n_bt.tolist() == [1, 2, 3, 4, 0, 0, 0, 0]
    assert n_cu.tolist() == [0, 1, 2, 2, 2]
    assert n_dist.tolist() == [2, 2, 2]
    assert n_new.tolist() == [1, 129, 0, 0]
    assert n_qpos.tolist() == [256, 256, 0, 0]
    assert n_upd.tolist() == [1, 0, 0, 0]
    assert torch.allclose(out[:2], torch.full((2, 4, 128), 2.0))
    assert torch.allclose(out[2:], torch.zeros((2, 4, 128)))


def test_interleaved_schedule_and_mask():
    import jax.numpy as jnp
    from vllm_torchtpu.kernels.experimental.batched_rpa import (
        kernel as rpa_kernel,
        schedule as rpa_schedule,
        schedule_cp,
    )

    mock_tpu = MagicMock()
    mock_tpu.generation = 7
    mock_tpu.chip_version = "v7x"
    mock_tpu.num_lanes = 128
    mock_tpu.mxu_column_size = 128
    mock_tpu.vmem_capacity_bytes = 64 * 1024 * 1024
    mock_tpu.fp8_ops_per_second = 2
    mock_tpu.bf16_ops_per_second = 1

    model_cfg = configs.ModelConfigs(
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
        mask_value=-1e9,
    )
    serve = configs.ServingConfigs(
        num_seqs=4,
        page_size=128,
        total_q_tokens=256,
        num_page_indices=32,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
        cp_group_size=2,
        attention_scope=configs.AttentionScope.NEW_TOKENS_ONLY,
        cp_interleaved=True,
    )
    block_cfg = configs.BlockSizes(
        bq_sz=256,
        bq_c_sz=256,
        bkv_sz=128,
        batch_size=1,
    )
    with patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_tpu):
        cfgs = configs.RpaConfigs(
            block=block_cfg,
            model=model_cfg,
            serve=serve,
            mode=configs.RpaCase.MIXED,
            vmem_limit_bytes=64 * 1024 * 1024,
        )

    computer = schedule_cp.CPMetadataComputer(
        schedule=None,
        schedule_hbm_ref=None,
        dma_sem=None,
        cfgs=cfgs,
        extra_refs=([jnp.int32(1)],),
    )
    computer.k_loop = lambda k_idx, carry, **kw: rpa_schedule.LoopCarry(
        carry.hbm_offset, carry.count + 1
    )

    out_carry = computer.q_loop(
        0,
        rpa_schedule.LoopCarry(jnp.int32(0), jnp.int32(0)),
        s_idx=jnp.int32(0),
        q_start=jnp.int32(0),
        q_end=jnp.int32(256),
        q_position=jnp.int32(128),
        global_kv_cache_len=jnp.int32(0),
        kv_new_len=jnp.int32(512),
        num_k=jnp.int32(4),
        num_q=jnp.int32(1),
        update_kv_cache=jnp.bool_(True),
    )
    # Verify all 4 KV tiles (0, 1, 2, 3) were scheduled for rank 1's tile 0
    assert int(out_carry.count) == 4

    step_meta = rpa_kernel.ComputeMetadata(
        causal_offset=[jnp.int32(256 - 128)],  # k_id (256) - q_position (128)
        is_valid=[jnp.bool_(True)],
        q_sz=[jnp.int32(256)],
        k_id=[jnp.int32(256)],
        total_kv_len=[jnp.int32(512)],
        local_k_start=[jnp.int32(0)],
        local_k_end=None,
        q_local_start=[jnp.int32(0)],
    )
    n_q_per_kv = cfgs.aligned_num_q_heads_per_kv_head
    masks = rpa_kernel.generate_mask(
        (1, 2, 256 * n_q_per_kv, 128),
        bq_start=0,
        step_meta=step_meta,
        cfgs=cfgs,
    )
    mask_b = np.asarray(masks[0])[0, ::n_q_per_kv, :]  # shape (256, 128)
    assert not mask_b[:128, :].any()
    assert mask_b[128:, :].all()

