# SPDX-License-Identifier: Apache-2.0
"""MTP filtering must prevent payload reads and preserve loaded values."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from vllm.config import LoadConfig
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP
from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMTP

from vllm_torchtpu.model_loader_patches import _should_skip_weight_tpu
from vllm_torchtpu.models.vllm.qwen3_5_mtp_prefill import Qwen3_5MoeMTPPrefill
from vllm_torchtpu.spec_decode import qwen3_next_mtp_loader as mtp_loader

pytestmark = pytest.mark.cpu_test


@pytest.fixture(params=[Qwen3NextMTP, Qwen3_5MoeMTP])
def checkpoint(tmp_path, request):
    model_cls = request.param
    embedding_name = (
        "model.language_model.embed_tokens.weight"
        if model_cls is Qwen3_5MoeMTP
        else "model.embed_tokens.weight"
    )
    shards = {
        "main.safetensors": {"model.layers.0.weight": torch.ones(2, 2)},
        "vocab.safetensors": {
            embedding_name: torch.arange(6.0).reshape(3, 2),
            "model.layers.1.weight": torch.ones(2, 2),
        },
        "head.safetensors": {"lm_head.weight": torch.arange(6.0).reshape(3, 2)},
        "mtp.safetensors": {
            "mtp.fc.weight": torch.tensor([[0.2, 0.3], [0.7, 0.9]]),
            "mtp.experts.0.weight_packed": torch.tensor([1, 2], dtype=torch.uint8),
            "mtp.experts.1.weight_packed": torch.tensor([3, 4], dtype=torch.uint8),
            "mtp.experts.1.input_scale": torch.tensor([0.5]),
        },
    }
    weight_map = {}
    for filename, tensors in shards.items():
        save_file(tensors, str(tmp_path / filename))
        weight_map.update(dict.fromkeys(tensors, filename))
    (tmp_path / SAFE_WEIGHTS_INDEX_NAME).write_text(
        json.dumps({"weight_map": weight_map})
    )
    config = SimpleNamespace(model=str(tmp_path), revision=None)
    tensors = {
        name: tensor for shard in shards.values() for name, tensor in shard.items()
    }
    return config, tensors, model_cls, embedding_name


@pytest.mark.parametrize("local_expert_ids", [None, {0}])
@pytest.mark.parametrize("strategy", [None, "lazy"])
def test_mtp_reads_only_needed_payloads_and_preserves_forward(
    checkpoint, monkeypatch, local_expert_ids, strategy
):
    config, tensors, model_cls, embedding_name = checkpoint
    loader = mtp_loader.Qwen3NextMTPLoader(
        LoadConfig(safetensors_load_strategy=strategy, use_tqdm_on_load=False)
    )
    loader.local_expert_ids = local_expert_ids
    monkeypatch.setattr(weight_utils, "should_skip_weight", _should_skip_weight_tpu)
    opened, read = [], []
    real_safe_open = weight_utils.safe_open

    class TrackingSafeOpen:
        def __init__(self, path, **kwargs):
            opened.append(path.rsplit("/", 1)[-1])
            self.inner = real_safe_open(path, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def keys(self):
            return self.inner.keys()

        def get_tensor(self, name):
            read.append(name)
            return self.inner.get_tensor(name)

    monkeypatch.setattr(weight_utils, "safe_open", TrackingSafeOpen)
    weights = dict(loader.get_all_weights(config, torch.nn.Module()))
    expected = {
        embedding_name,
        "lm_head.weight",
        "mtp.fc.weight",
        "mtp.experts.0.weight_packed",
        "mtp.experts.1.input_scale",
    }
    if local_expert_ids is None:
        expected.add("mtp.experts.1.weight_packed")
    assert set(opened) == {"vocab.safetensors", "head.safetensors", "mtp.safetensors"}
    assert set(read) == set(weights) == expected
    assert len(read) == len(expected)
    for name, tensor in weights.items():
        torch.testing.assert_close(tensor, tensors[name], rtol=0, atol=0)

    draft = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(draft)
    draft.model = torch.nn.Module()
    draft.model.embed_tokens = torch.nn.Embedding(3, 2)
    draft.model.fc = torch.nn.Linear(2, 2, bias=False)
    draft.model.experts = torch.nn.ModuleDict()
    for name, tensor in weights.items():
        if name.startswith("mtp.experts."):
            _, _, expert, param = name.split(".")
            if expert not in draft.model.experts:
                draft.model.experts[expert] = torch.nn.Module()
            draft.model.experts[expert].register_parameter(
                param, torch.nn.Parameter(torch.empty_like(tensor), requires_grad=False)
            )
    draft.lm_head = torch.nn.Linear(2, 3, bias=False)
    draft.load_weights(weights.items())
    token_ids = torch.tensor([0, 2])
    actual = draft.lm_head(draft.model.fc(draft.model.embed_tokens(token_ids)))
    expected_output = torch.nn.functional.linear(
        torch.nn.functional.linear(
            torch.nn.functional.embedding(token_ids, tensors[embedding_name]),
            tensors["mtp.fc.weight"],
        ),
        tensors["lm_head.weight"],
    )
    torch.testing.assert_close(actual, expected_output, rtol=0, atol=0)


def test_mtp_still_validates_full_checkpoint_index(checkpoint):
    config, _, _, _ = checkpoint
    (Path(config.model) / "main.safetensors").unlink()
    loader = mtp_loader.Qwen3NextMTPLoader(LoadConfig(safetensors_load_strategy="lazy"))
    with pytest.raises(FileNotFoundError, match="main.safetensors"):
        list(loader.get_all_weights(config, torch.nn.Module()))


@pytest.mark.parametrize(
    "model_cls,strategy,indexed,optimized",
    [
        (Qwen3NextMTP, "lazy", True, True),
        (Qwen3NextMTP, None, True, True),
        (Qwen3_5MoeMTP, "lazy", True, True),
        (Qwen3_5MoeMTP, None, True, True),
        (Qwen3_5MoeMTPPrefill, "lazy", True, True),
        (torch.nn.Module, "lazy", True, False),
        (Qwen3NextMTP, "eager", True, False),
        (Qwen3_5MoeMTP, "prefetch", True, False),
        (Qwen3NextMTP, "lazy", False, False),
    ],
)
def test_mtp_loader_selection_is_scoped(
    checkpoint, monkeypatch, model_cls, strategy, indexed, optimized
):
    config, _, _, _ = checkpoint
    monkeypatch.setattr(mtp_loader.os.path, "isfile", lambda _path: indexed)
    monkeypatch.setattr(
        mtp_loader, "get_model_architecture", lambda _cfg: (model_cls, "")
    )
    loader = DefaultModelLoader(LoadConfig(safetensors_load_strategy=strategy))
    selected = mtp_loader.maybe_use_qwen3_next_mtp_loader(loader, config)
    if optimized:
        assert isinstance(selected, mtp_loader.Qwen3NextMTPLoader)
    else:
        assert selected is loader
