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
"""Unit tests for VllmMambaMixer2 custom op and JAX Mamba-2 core."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh

from vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op import VllmMambaMixer2
from vllm_torchtpu.layers.core.mamba_mixer2 import (
    run_jax_mamba_mixer,
)
from vllm_torchtpu.layers.core.ssd_scan import _segsum_jax, ssd_minimal_discrete_hybrid
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


def _mesh(tp_size: int = 1):
    return SimpleNamespace(
        shape={"attn_dp": 1, "expert": 1, "model": tp_size},
        axis_names=("attn_dp", "expert", "model"),
    )


def _vllm_config(*, dp_size: int = 1):
    return SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=dp_size))


def _granite_mamba_layer(
    prefix: str = "model.layers.0.mamba",
    *,
    tp_size: int = 1,
    n_groups: int = 4,
    bias: bool = True,
):
    layer = VllmMambaMixer2.__new__(VllmMambaMixer2)
    torch.nn.Module.__init__(layer)
    layer.prefix = prefix
    layer.tp_size = tp_size
    layer.num_heads = 32
    layer.head_dim = 32
    layer.ssm_state_size = 64
    layer.conv_kernel_size = 4
    layer.n_groups = n_groups
    layer.intermediate_size = layer.num_heads * layer.head_dim
    layer.groups_ssm_state_size = layer.n_groups * layer.ssm_state_size
    layer.conv_dim = layer.intermediate_size + 2 * layer.groups_ssm_state_size
    layer.tped_intermediate_size = layer.intermediate_size // layer.tp_size
    layer.tped_conv_size = layer.conv_dim // layer.tp_size
    layer.tped_dt_size = layer.num_heads // layer.tp_size

    layer.conv1d = SimpleNamespace(
        weight=torch.randn(layer.tped_conv_size, 1, layer.conv_kernel_size),
        bias=torch.randn(layer.tped_conv_size) if bias else None,
    )
    layer.A = torch.nn.Parameter(-torch.ones(layer.tped_dt_size))
    layer.D = torch.nn.Parameter(torch.ones(layer.tped_dt_size))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(layer.tped_dt_size))
    layer.norm = SimpleNamespace(
        weight=torch.ones(layer.tped_intermediate_size),
        variance_epsilon=1e-5,
        n_groups=n_groups,
    )
    return layer


class TestVllmMambaMixer2:
    def test_init_builds_mamba_op(self):
        mock_op = MagicMock()
        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "MambaMixer2.__init__",
                return_value=None,
            ),
            patch.object(
                VllmMambaMixer2, "_build_mamba_op", return_value=mock_op
            ) as mock_build,
        ):
            layer = VllmMambaMixer2()

        mock_build.assert_called_once_with()
        assert layer.mamba_op is mock_op

    def test_build_mamba_op_registers_pallas_jax_op_and_fake(self):
        layer0 = _granite_mamba_layer("model.layers.0.mamba")
        layer1 = _granite_mamba_layer("model.layers.1.mamba")
        fake_jax_op = MagicMock()

        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op.pallas.jax_op",
                return_value=fake_jax_op,
            ) as mock_jax_op,
        ):
            layer0._build_mamba_op()
            layer1._build_mamba_op()

        assert mock_jax_op.call_count == 2
        assert (
            mock_jax_op.call_args_list[0].args[0]
            == "pallas::mamba_mixer2_model_layers_0_mamba"
        )
        assert (
            mock_jax_op.call_args_list[1].args[0]
            == "pallas::mamba_mixer2_model_layers_1_mamba"
        )
        assert mock_jax_op.call_args_list[0].kwargs["donate_argnums"] == (1, 2)
        assert fake_jax_op.register_fake.call_count == 2

        fake_fn = fake_jax_op.register_fake.call_args_list[0].args[0]
        proj = torch.zeros(
            (
                16,
                layer0.tped_intermediate_size
                + layer0.tped_conv_size
                + layer0.tped_dt_size,
            )
        )
        conv_state = torch.zeros(
            (8, layer0.conv_kernel_size - 1, layer0.tped_conv_size)
        )
        rec_state = torch.zeros(
            (8, layer0.tped_dt_size, layer0.head_dim, layer0.ssm_state_size)
        )
        fake_conv, fake_rec, fake_out = fake_fn(proj, conv_state, rec_state)
        assert fake_conv.shape == conv_state.shape
        assert fake_rec.shape == rec_state.shape
        assert fake_out.shape == (16, layer0.tped_intermediate_size)

    def test_forward_warmup_with_unallocated_kv_cache(self):
        layer = _granite_mamba_layer()
        layer.kv_cache = (torch.tensor([]), torch.tensor([]))
        layer.mamba_op = MagicMock()

        total_proj_dim = (
            layer.tped_intermediate_size + layer.tped_conv_size + layer.tped_dt_size
        )
        layer.in_proj = MagicMock(
            side_effect=lambda x: (torch.ones(x.size(0), total_proj_dim), None)
        )
        layer.norm = MagicMock(side_effect=lambda x, gate: x)
        layer.out_proj = MagicMock(
            side_effect=lambda x: (torch.full((x.size(0), 512), 2.0), None)
        )

        hidden_states = torch.randn(8, 512)
        out = layer.forward(hidden_states)

        layer.mamba_op.assert_not_called()
        layer.norm.assert_called_once()
        assert out.shape == (8, 512)

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op.get_forward_context"
    )
    def test_forward_executes_mamba_op_and_updates_caches_in_place(
        self, mock_get_forward_context
    ):
        layer = _granite_mamba_layer()
        num_tokens = 12
        max_seqs = 4
        conv_state = torch.zeros(
            (max_seqs, layer.conv_kernel_size - 1, layer.tped_conv_size)
        )
        rec_state = torch.zeros(
            (max_seqs, layer.tped_dt_size, layer.head_dim, layer.ssm_state_size)
        )
        layer.kv_cache = (conv_state, rec_state)

        new_conv = torch.ones_like(conv_state) * 3.0
        new_rec = torch.ones_like(rec_state) * 5.0
        expected_ssm_out = torch.ones((num_tokens, layer.tped_intermediate_size)) * 7.0

        fake_jax_op = MagicMock(return_value=(new_conv, new_rec, expected_ssm_out))
        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op.pallas.jax_op",
                return_value=fake_jax_op,
            ),
        ):
            layer.mamba_op = layer._build_mamba_op()

            total_proj_dim = (
                layer.tped_intermediate_size + layer.tped_conv_size + layer.tped_dt_size
            )
            layer.in_proj = MagicMock(
                side_effect=lambda x: (
                    torch.ones(x.size(0), total_proj_dim),
                    None,
                )
            )
            layer.out_proj = MagicMock(side_effect=lambda x: (x[:, :512], None))

            attn_metadata = SimpleNamespace(
                mamba_state_indices=torch.tensor([0, 1], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 6, 12], dtype=torch.int32),
                request_distribution=torch.tensor([0, 2, 2], dtype=torch.int32),
                seq_lens=torch.tensor([6, 6], dtype=torch.int32),
                block_tables=None,
            )
            mock_fc = MagicMock()
            mock_fc.attn_metadata = {layer.prefix: attn_metadata}
            mock_get_forward_context.return_value = mock_fc

            out = layer.forward(torch.randn(num_tokens, 512))

        assert torch.allclose(conv_state, new_conv)
        assert torch.allclose(rec_state, new_rec)
        assert torch.allclose(out, torch.ones(num_tokens, 512) * 7.0)

    def test_run_jax_mamba_mixer_e2e_pipeline(self):
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        num_tokens = 16
        max_seqs = 4
        num_heads = 4
        head_dim = 8
        n_groups = 2
        d_state = 8
        kernel_size = 4
        chunk_size = 8
        tped_intermediate_size = num_heads * head_dim
        tped_conv_size = tped_intermediate_size + 2 * n_groups * d_state
        tped_dt_size = num_heads

        j_projected_states = jnp.ones(
            (
                num_tokens,
                tped_intermediate_size + tped_conv_size + tped_dt_size,
            ),
            dtype=jnp.float32,
        )
        conv_state = jnp.zeros(
            (max_seqs, kernel_size - 1, tped_conv_size), dtype=jnp.float32
        )
        recurrent_state = jnp.zeros(
            (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        j_conv_weight = jnp.ones((tped_conv_size, 1, kernel_size), dtype=jnp.float32)
        j_conv_bias = jnp.zeros((tped_conv_size,), dtype=jnp.float32)
        j_A = -jnp.ones((num_heads,), dtype=jnp.float32)
        j_D = jnp.ones((num_heads,), dtype=jnp.float32)
        j_dt_bias = jnp.zeros((num_heads,), dtype=jnp.float32)
        j_norm_weight = jnp.ones((tped_intermediate_size,), dtype=jnp.float32)
        state_indices = jnp.array([0, 1], dtype=jnp.int32)
        query_start_loc = jnp.array([0, 8, 16], dtype=jnp.int32)
        distribution = jnp.array([0, 2, 2], dtype=jnp.int32)
        seq_lens = jnp.array([8, 8], dtype=jnp.int32)

        def _fake_conv1d(x, c_state, weight, bias, *args, **kwargs):
            return jax.nn.silu(x), c_state + 1.0

        with patch(
            "vllm_torchtpu.layers.core.mamba_mixer2.causal_conv1d.ragged_causal_conv1d",
            side_effect=_fake_conv1d,
        ):
            (new_conv, new_rec), y_out = run_jax_mamba_mixer(
                j_projected_states,
                conv_state,
                recurrent_state,
                j_conv_weight,
                j_conv_bias,
                j_A,
                j_D,
                j_dt_bias,
                j_norm_weight,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                mesh=mesh,
                tped_intermediate_size=tped_intermediate_size,
                tped_conv_size=tped_conv_size,
                tped_dt_size=tped_dt_size,
                num_heads=num_heads,
                head_dim=head_dim,
                n_groups=n_groups,
                d_state=d_state,
                kernel_size=kernel_size,
                chunk_size=chunk_size,
                shard_groups=True,
                rms_norm_eps=1e-5,
                apply_rms_norm=True,
            )

        assert new_conv.shape == conv_state.shape
        assert new_rec.shape == recurrent_state.shape
        assert y_out.shape == (num_tokens, tped_intermediate_size)
        assert not jnp.any(jnp.isnan(y_out))
        assert jnp.any(new_rec[0] != 0.0)
        assert jnp.any(new_rec[1] != 0.0)
        assert jnp.all(new_rec[2] == 0.0)

    def test_ssd_minimal_discrete_hybrid_matches_jax_reference(self):
        batch, seqlen, block_len, n_heads, d_head, d_state = 2, 16, 4, 4, 8, 8
        n_chunks_per_seq = seqlen // block_len
        n_chunks = batch * n_chunks_per_seq

        rng = jax.random.key(42)
        k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

        X_c = jax.random.normal(k1, (n_chunks, n_heads, block_len, d_head))
        A_c = -jnp.exp(jax.random.normal(k2, (n_chunks, n_heads, block_len)))
        B_c = jax.random.normal(k3, (n_chunks, n_heads, block_len, d_state))
        C_c = jax.random.normal(k4, (n_chunks, n_heads, block_len, d_state))
        init_states_per_batch = jax.random.normal(k5, (batch, n_heads, d_head, d_state))

        initial_states = jnp.zeros((n_chunks, n_heads, d_head, d_state))
        reset_mask = np.zeros((n_chunks,), dtype=bool)
        for b in range(batch):
            initial_states = initial_states.at[b * n_chunks_per_seq].set(
                init_states_per_batch[b]
            )
            reset_mask[b * n_chunks_per_seq] = True
        reset_mask = jnp.array(reset_mask)
        last_chunk_indices = jnp.array(
            [b * n_chunks_per_seq + n_chunks_per_seq - 1 for b in range(batch)],
            dtype=jnp.int32,
        )

        # Reference math
        A_cumsum = jnp.cumsum(A_c, axis=-1)
        L = jnp.exp(_segsum_jax(A_c))
        Y_diag = jnp.einsum("chln,chsn,chls,chsp->chlp", C_c, B_c, L, X_c)
        decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
        states_c_ref = jnp.einsum("chln,chlp,chl->chpn", B_c, X_c, decay_states)
        decay_chunk = jnp.exp(A_cumsum[..., -1])

        def scan_body(carry, args):
            s_c, d_c, init_s, r_c = args
            inc = jnp.where(r_c, init_s, carry)
            nxt = inc * d_c[:, None, None] + s_c
            return nxt, (inc, nxt)

        _, (states_inc_ref, states_out_ref) = jax.lax.scan(
            scan_body,
            jnp.zeros((n_heads, d_head, d_state), dtype=X_c.dtype),
            (states_c_ref, decay_chunk, initial_states, reset_mask),
        )
        Y_off = jnp.einsum(
            "chln,chpn,chl->chlp", C_c, states_inc_ref, jnp.exp(A_cumsum)
        )
        y_ref = Y_diag + Y_off
        final_state_ref = states_out_ref[last_chunk_indices]

        y_hybrid, final_state_hybrid = ssd_minimal_discrete_hybrid(
            X_c,
            A_c,
            B_c,
            C_c,
            last_chunk_indices,
            initial_states,
            reset_mask,
        )

        np.testing.assert_allclose(y_hybrid, y_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            final_state_hybrid, final_state_ref, rtol=1e-4, atol=1e-4
        )
