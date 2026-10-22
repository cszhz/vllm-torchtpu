# SPDX-License-Identifier: Apache-2.0
"""Snapshot contents, sampling and readback ownership are diagnostic contracts."""

import pickle
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner.hidden_nan_flags import (
    HiddenNaNDetector,
    ModelHiddenNaNSnapshot,
    install_hidden_nan_detector,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def detector(tmp_path, device="cpu", capacity=8):
    return HiddenNaNDetector(
        tuple(
            ModelHiddenNaNSnapshot(
                role,
                layers,
                torch.tensor([-1, 0, 0, 0], dtype=torch.int32, device=device),
                torch.zeros(capacity * 128, device=device),
                torch.zeros(capacity, dtype=torch.int32, device=device),
                torch.zeros(capacity, dtype=torch.int32, device=device),
                torch.zeros(1, dtype=torch.int32, device=device),
                128,
            )
            for role, layers in (("target", (7, 11, 19)), ("draft", (60,)))
        ),
        tmp_path,
    )


def record_on_cpu(
    state, saved_hidden, saved_tokens, armed, hidden, tokens, rows, slot, rounds
):
    if (
        armed[0]
        and state[0] < 0
        and (rounds[1] == 0 or rounds[0] == rounds[1])
        and (~torch.isfinite(hidden)).any()
    ):
        saved_hidden[: hidden.numel()].copy_(hidden.reshape(-1))
        saved_tokens[: tokens.shape[0]].copy_(tokens)
        state[:] = torch.tensor([slot, int(rounds[0]), hidden.shape[0], int(rows[0])])
        rounds[1] = rounds[0]
    return hidden


def test_unarmed_warmup_submits_no_readback(tmp_path):
    capture = detector(tmp_path)
    with (
        patch.object(torch.Tensor, "to") as copy,
        patch("vllm_torchtpu.runner.hidden_nan_flags.TpuEvent") as event,
    ):
        assert capture.readback() is None
    copy.assert_not_called()
    event.assert_not_called()


@pytest.mark.parametrize("tp,pcp", [(2, 1), (1, 8)])
def test_async_multirank_flags_require_output_aggregation(monkeypatch, tp, pcp):
    runner = object.__new__(TPUModelRunner)
    runner.parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
    )
    runner.is_pooling_model = False
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=None, parallel_config=runner.parallel_config
    )
    monkeypatch.setattr(envs, "TPU_NAN_LOG_LEVEL", envs.NanLogLevel.DETAIL)
    with pytest.raises(ValueError, match="all-rank output consumption"):
        runner.capture_model()


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("tuple_output", [False, True])
@pytest.mark.parametrize(
    "container,layer_ids,slots",
    [("list", (0, 1), [0, 1]), ("dict", (7, 19), [1, 0])],
)
def test_layer_hooks_check_full_hidden_once_and_preserve_output(
    tmp_path, monkeypatch, multimodal, tuple_output, container, layer_ids, slots
):
    layer_inputs = []
    checked_outputs = []

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

        def forward(self, hidden):
            layer_inputs.append(hidden)
            for operation in self.inner:
                hidden = operation(hidden)
            result = hidden + 1
            return (result, None) if tuple_output else result

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = (
                torch.nn.ModuleList([Layer(), Layer()])
                if container == "list"
                else torch.nn.ModuleDict({"19": Layer(), "7": Layer()})
            )

        def forward(self, input_ids):
            hidden = input_ids[:, None].expand(-1, 128).float().clone()
            hidden[1, 0] = float("nan")
            for layer in self.layers.children():
                output = layer(hidden)
                hidden = output[0] if tuple_output else output
            return hidden

    class MultiModalModel(torch.nn.Module):
        supports_multimodal = True

        def __init__(self):
            super().__init__()
            self.vision_layers = torch.nn.ModuleList([Layer()])
            self.vision_embedding = torch.nn.Embedding(256, 128)
            self.language_model = Model()

        def get_language_model(self):
            return self.language_model

        def forward(self, hidden):
            self.vision_embedding(torch.ones((2, 3), dtype=torch.int64))
            return self.language_model(hidden)

    model = MultiModalModel() if multimodal else Model()
    config = SimpleNamespace(get_hidden_size=lambda: 128, dtype=torch.float32)
    capture = install_hidden_nan_detector(
        [("target", model, config, 8)], torch.device("cpu"), dump_dir=tmp_path
    )
    assert capture.model_diagnostics[0].layer_ids == layer_ids
    assert capture.model_diagnostics[0].hidden_shape == (8, 128)
    capture.arm()
    capture.begin_round(["r"])
    seen_slots = []

    def inspect(
        state, saved_hidden, saved_tokens, armed, hidden, tokens, rows, slot, rounds
    ):
        seen_slots.append(slot)
        checked = record_on_cpu(
            state, saved_hidden, saved_tokens, armed, hidden, tokens, rows, slot, rounds
        ).clone()
        checked_outputs.append(checked)
        return checked

    monkeypatch.setattr(
        "vllm_torchtpu.runner.hidden_nan_flags.record_layer_nan", inspect
    )
    tokens = torch.tensor([15, 27], dtype=torch.int32)
    result = model(tokens)
    assert layer_inputs[1] is checked_outputs[0]
    assert result is checked_outputs[1]
    hidden = tokens[:, None].expand(-1, 128).float().clone()
    hidden[1, 0] = float("nan")
    torch.testing.assert_close(result, hidden + 2, equal_nan=True)
    snapshot = capture.model_diagnostics[0]
    assert snapshot.layer_ids[int(snapshot.state[0])] == layer_ids[slots[0]]
    torch.testing.assert_close(
        snapshot.hidden.reshape(snapshot.hidden_shape)[:2], hidden + 1, equal_nan=True
    )
    assert snapshot.input_ids[:2].tolist() == [15, 27]
    assert seen_slots == slots


@pytest.mark.parametrize("replicated_embedding", [False, True])
def test_unbound_mtp_forward_captures_new_embedding_tokens(
    tmp_path, monkeypatch, replicated_embedding
):
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    class Embedding(VocabParallelEmbedding):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, input_ids):
            return input_ids[:, None].expand(-1, 128).float().clone()

    class LMHead(ParallelLMHead):
        def __init__(self):
            torch.nn.Module.__init__(self)

    class Model(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embed_tokens = embedding
            self.embed_positions = torch.nn.Embedding(128, 128)
            self.lm_head = LMHead()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, input_ids):
            hidden = self.embed_tokens(input_ids)
            hidden = hidden + self.embed_positions(torch.arange(input_ids.numel()))
            hidden[1, 0] = float("nan")
            return self.layers[0](hidden)

    embedding = torch.nn.Embedding(256, 128) if replicated_embedding else Embedding()
    target, model = Model(embedding), Model(embedding)
    config = SimpleNamespace(get_hidden_size=lambda: 128, dtype=torch.float32)
    capture = install_hidden_nan_detector(
        [("target", target, config, 2), ("draft", model, config, 8)],
        torch.device("cpu"),
        dump_dir=tmp_path,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.hidden_nan_flags.record_layer_nan", record_on_cpu
    )
    target(torch.tensor([15, 27], dtype=torch.int32))
    capture.arm()
    capture.begin_round(["step1"])
    type(model).forward(model, torch.tensor([39, 42, 81, 99], dtype=torch.int32))
    assert capture.model_diagnostics[0].hidden_shape[0] == 2
    assert capture.model_diagnostics[1].hidden_shape[0] == 8
    assert capture.model_diagnostics[1].input_ids[:4].tolist() == [39, 42, 81, 99]


def test_idle_forward_between_rounds_does_not_spend_nan_sample(tmp_path):
    capture = detector(tmp_path)
    capture.arm()
    assert capture.armed_tpu.tolist() == [False]
    capture.begin_round(["real"])
    with (
        patch.object(torch.Tensor, "to", lambda tensor, *a, **kw: tensor.clone()),
        patch("vllm_torchtpu.runner.hidden_nan_flags.TpuEvent"),
    ):
        capture.readback()
    diagnostic = capture.model_diagnostics[0]
    record_on_cpu(
        diagnostic.state,
        diagnostic.hidden,
        diagnostic.input_ids,
        capture.armed_tpu,
        torch.full((1, 128), float("nan")),
        torch.tensor([42]),
        torch.tensor([1]),
        0,
        capture.round_state,
    )
    assert diagnostic.state.tolist() == [-1, 0, 0, 0]


def test_native_round_counter_has_bounded_hbm(tmp_path, device):
    """Long runs must not retain a new scalar buffer for every round number."""
    if device.type != "tpu":
        pytest.skip("Requires native TorchTPU scalar caching and HBM accounting")
    import jax
    from torch_tpu._internal import sync

    capture = detector(tmp_path, device)
    capture.arm()
    for _ in range(100):
        capture.begin_round(["clean"])
        readback = capture.readback()
        assert readback is not None
        readback.consume()
    sync.synchronize(tensors=None, wait=True)
    before = jax.devices()[0].memory_stats()["bytes_in_use"]
    for _ in range(1100):
        capture.begin_round(["clean"])
        readback = capture.readback()
        assert readback is not None
        readback.consume()
    sync.synchronize(tensors=None, wait=True)
    after = jax.devices()[0].memory_stats()["bytes_in_use"]
    assert after - before < 64 * 1024, (before, after)
    assert capture.round_state.cpu().tolist() == [1200, 0]


@pytest.mark.parametrize("capacity", [2, 8])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("consume_before_pause", [False, True])
def test_native_donated_snapshot_survives_delayed_readback(
    tmp_path, device, capacity, bad_value, consume_before_pause
):
    """Paused forwards must not overwrite the one donated snapshot per role."""
    if device.type != "tpu":
        pytest.skip("Requires native TorchTPU buffer donation and D2H")
    from vllm_torchtpu.layers.adapter.custom_ops.hidden_nan_flags_op import (
        record_layer_nan,
    )

    capture = detector(tmp_path, device, capacity)
    diagnostic = capture.model_diagnostics[0]

    def record(hidden, tokens):
        rows = torch.scalar_tensor(tokens.shape[0], dtype=torch.int32, device=device)
        return record_layer_nan(
            diagnostic.state,
            diagnostic.hidden,
            diagnostic.input_ids,
            capture.armed_tpu,
            hidden,
            tokens,
            rows.reshape(1),
            1,
            capture.round_state,
        )

    compiled = torch.compile(record, backend="tpu", fullgraph=True, dynamic=False)
    good = torch.ones((2, 128), device=device)
    bad = good.clone()
    bad[1, 0] = bad_value
    tokens = torch.tensor([15, 27], dtype=torch.int32, device=device)
    compiled(good, tokens)
    capture.arm()
    capture.begin_round(["first"])
    compiled(bad, tokens)
    first = capture.readback()
    assert first is not None
    if consume_before_pause:
        first.consume()
    for _ in range(20):
        capture.begin_round(["paused"])
        compiled(bad * 9, tokens + 9)
        later = capture.readback()
        if later is not None:
            later.consume()
    if not consume_before_pause:
        first.consume()
    assert capture._dump_future is not None
    capture._dump_future.result(timeout=30)
    capture.begin_round(["second"])
    compiled(bad * 3, tokens + 2)
    second = capture.readback()
    assert second is not None
    second.consume()
    assert capture._dump_future is not None
    capture._dump_future.result(timeout=30)
    snapshots = []
    for path in tmp_path.glob("*.pkl"):
        with path.open("rb") as stream:
            snapshots.append(pickle.load(stream))
    snapshots.sort(key=lambda value: value["round"])
    assert len(snapshots) == 2
    for snapshot, round_id, offset, value in zip(snapshots, (1, 22), (0, 2), (1, 3)):
        assert snapshot["round"] == round_id
        assert snapshot["input_ids"].tolist() == [15 + offset, 27 + offset]
        assert snapshot["hidden_states"][0, 0] == value
        torch.testing.assert_close(
            snapshot["hidden_states"][1, 0], torch.tensor(bad_value), equal_nan=True
        )
