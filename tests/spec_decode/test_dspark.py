# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_torchtpu.spec_decode.dspark import DSparkProposer


class _FakeHFConfig:
    """to_dict() reflects attribute writes, like a transformers config."""

    def __init__(self, d):
        self._d = dict(d)

    def to_dict(self):
        out = dict(self._d)
        if hasattr(self, "dflash_config"):
            out["dflash_config"] = self.dflash_config
        return out


def _make_proposer(hf_dict, num_speculative_tokens=4, **spec_fields):
    # The vLLM 0.29 drafting fields at their defaults; tests override them.
    fields = dict(
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
        dspark_draft_topk=None,
    )
    fields.update(spec_fields)
    speculative_config = SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        draft_tensor_parallel_size=1,
        draft_model_config=SimpleNamespace(hf_config=_FakeHFConfig(hf_dict)),
        **fields,
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        model_config=SimpleNamespace(max_model_len=2048),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "model.layers.0.self_attn": mock.MagicMock(),
            },
            fast_moe_cold_start=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
            is_moe_model=False,
            use_sequence_parallel_moe=False,
        ),
    )
    runner = mock.MagicMock()
    runner.device = torch.device("cpu")
    return DSparkProposer(runner=runner, vllm_config=vllm_config)


_LAYER_IDS = {"dspark_target_layer_ids": [1, 15, 30]}


def test_noise_resolution_and_dense_block():
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "dspark_block_size": 4,
            **_LAYER_IDS,
        }
    )
    assert p.mask_token_id == 7
    assert p.sample_from_anchor
    assert p._query_block_size(4) == 4
    # propose() sizes the query block via this property, so the dense
    # override must reach it (K slots, not DFlash's 1 + K).
    assert p.block_size == 4


def test_bonus_anchor_keeps_dflash_layout():
    p = _make_proposer(
        {
            "mask_token_id": 5,
            "sample_from_anchor": False,
            **_LAYER_IDS,
        }
    )
    assert not p.sample_from_anchor
    assert p._query_block_size(4) == 5
    assert p.block_size == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_adaptive_verification", True),
        ("dspark_draft_topk", 8),
    ],
)
def test_rejects_unsupported_drafting_options(field, value):
    with pytest.raises(ValueError, match="does not support"):
        _make_proposer(
            {
                "dspark_noise_token_id": 7,
                "dspark_block_size": 4,
                **_LAYER_IDS,
            },
            **{field: value},
        )


def test_k_need_not_match_checkpoint_block():
    # Upstream DSparkSpeculator drafts any K from a checkpoint trained at one
    # block length; the query block follows K.
    p = _make_proposer(
        {"dspark_noise_token_id": 7, "dspark_block_size": 5, **_LAYER_IDS},
        num_speculative_tokens=6,
    )
    assert p.block_size == 6


def test_probabilistic_requires_synthetic_rejection():
    hf = {"dspark_noise_token_id": 7, "vocab_size": 32, **_LAYER_IDS}
    with pytest.raises(ValueError, match="synthetic"):
        _make_proposer(
            hf,
            draft_sample_method="probabilistic",
            rejection_sample_method="standard",
        )
    p = _make_proposer(
        hf,
        draft_sample_method="probabilistic",
        rejection_sample_method="synthetic",
    )
    assert p._probabilistic
    assert p._draft_vocab_size == 32


def test_probabilistic_sample_inputs_shapes():
    hf = {"dspark_noise_token_id": 7, "vocab_size": 32, **_LAYER_IDS}
    p = _make_proposer(
        hf,
        num_speculative_tokens=4,
        draft_sample_method="probabilistic",
        rejection_sample_method="synthetic",
    )
    p.runner.input_batch.temperature_cpu_tensor = torch.tensor([0.0, 0.7, 1.0])
    temps, noise = p._draft_sample_inputs(4, start_index=1, num_reqs=2)
    assert temps.tolist() == pytest.approx([0.7, 1.0, 0.0, 0.0])
    assert noise.shape == (4, 4, 32)
    dummy_temps, dummy_noise = p._draft_sample_inputs(4)
    assert dummy_temps.tolist() == [0.0] * 4
    assert dummy_noise.shape == (4, 4, 32)
    assert _make_proposer(hf)._draft_sample_inputs(4) == ()


def test_sample_draft_greedy_rows_and_gumbel_distribution():
    torch.manual_seed(0)
    logits = torch.log(torch.tensor([0.1, 0.2, 0.7]))
    n = 20000
    rows = logits.expand(n, 3)
    temps = torch.ones(n)
    temps[:10] = 0.0
    tokens = DSparkProposer._sample_draft(rows, temps, torch.rand(n, 3))
    assert (tokens[:10] == 2).all()
    freq = torch.bincount(tokens[10:], minlength=3).float() / (n - 10)
    assert freq.tolist() == pytest.approx([0.1, 0.2, 0.7], abs=0.015)
    assert (DSparkProposer._sample_draft(rows, None, None) == 2).all()


def test_missing_noise_token():
    with pytest.raises(ValueError, match="noise/mask token"):
        _make_proposer({})


def test_missing_target_layer_ids():
    with pytest.raises(ValueError, match="target layer ids"):
        _make_proposer({"dspark_noise_token_id": 7})


def test_target_layer_id_resolution_chain():
    # Aux ids (target + 1) take precedence and are shifted back down.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "eagle_aux_hidden_state_layer_ids": [2, 16, 31],
            "dspark_target_layer_ids": [9, 9, 9],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [1, 15, 30]

    # Dense-checkpoint spelling.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "target_layer_ids": [3, 7],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [3, 7]

    # An explicit dflash_config wins unchanged.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "dflash_config": {"target_layer_ids": [4, 8]},
            "dspark_target_layer_ids": [9, 9],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [4, 8]


def test_no_forced_embedding_sharing():
    # A reduced-vocab DSpark draft owns its lm_head; load_model must honor
    # has_own_embed_tokens / has_own_lm_head instead of force-sharing.
    p = _make_proposer({"dspark_noise_token_id": 7, **_LAYER_IDS})
    assert p._force_share_target_embeddings is False


def test_megakernel_idle_draft_replays_the_busy_sampling_collectives():
    """An idle DP engine runs the draft megakernel idle, then the draft
    sampling graph at the busy engines' lockstep bucket: under a TP-sharded
    draft the sampling issues TP collectives, which pair across DP engines."""
    p = _make_proposer(
        {"dspark_noise_token_id": 7, "dspark_block_size": 4, **_LAYER_IDS}
    )
    calls = []
    draft = SimpleNamespace(
        logits=False,
        spec=SimpleNamespace(hidden_size=16, vocab_size=32),
        idle=lambda bucket: calls.append(("idle", bucket)),
    )
    p.runner._dsv4_mk = SimpleNamespace(draft=draft, last_run="t16")
    p.runner._dp_step_max_reqs = 2
    p.runner.num_tokens_paddings = [8, 16, 32]

    def sample(hidden, ids, block_size, *rest):
        calls.append(("sample", tuple(hidden.shape), ids.dtype, block_size, rest))
        return torch.zeros(1), hidden

    p._sample_from_hidden = sample
    rows = p._dp_draft_bucket()
    assert rows == p._get_padded_len(2 * p.block_size)
    with mock.patch("vllm_torchtpu.utils.synchronize_tensors"):
        p._draft_replicated = False
        p.run_dp_dummy_draft(1)
        assert calls == [
            ("idle", "t16"),
            ("sample", (rows, 16), torch.int32, p.block_size, ()),
        ]
        # a replicated draft samples rank-locally: nothing to pair
        calls.clear()
        p._draft_replicated = True
        p.run_dp_dummy_draft(1)
        assert calls == [("idle", "t16")]
