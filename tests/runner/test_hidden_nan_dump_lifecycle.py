# SPDX-License-Identifier: Apache-2.0
"""Bounded asynchronous dumps must survive slow disk and later bad rounds."""

import pickle
from threading import Event

import pytest
import torch

from vllm_torchtpu.runner.hidden_nan_flags import (
    HiddenNaNDetector,
    ModelHiddenNaNSnapshot,
)


@pytest.fixture
def capture(tmp_path, monkeypatch):
    class CPUEvent:
        def record(self):
            pass

        def synchronize(self):
            pass

    monkeypatch.setattr("vllm_torchtpu.runner.hidden_nan_flags.TpuEvent", CPUEvent)
    original_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        result = original_to(tensor, *args, **kwargs)
        return result.clone() if args and args[0] == "cpu" else result

    monkeypatch.setattr(torch.Tensor, "to", to)
    detector = HiddenNaNDetector(
        tuple(
            ModelHiddenNaNSnapshot(
                role,
                (7, 11),
                torch.tensor([-1, 0, 0, 0], dtype=torch.int32),
                torch.zeros(8 * 128),
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                128,
            )
            for role in ("target", "draft")
        ),
        tmp_path,
    )
    detector.arm()
    yield detector
    detector._dump_executor.shutdown(wait=True)


def test_clean_rounds_only_copy_small_states(capture, monkeypatch, tmp_path):
    copied_shapes = []
    original_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        copied_shapes.append(tuple(tensor.shape))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    for _ in range(3):
        capture.begin_round(["clean"])
        readback = capture.readback()
        assert readback is not None
        readback.consume()
    assert copied_shapes == [(4,)] * 6
    assert not list(tmp_path.iterdir())


def test_slow_pickle_does_not_block_consumer_or_allow_snapshot_reuse(
    capture, tmp_path, monkeypatch
):
    entered, release = Event(), Event()
    real_dump = pickle.dump

    def slow_dump(value, stream, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release the blocked disk writer"
        real_dump(value, stream, **kwargs)

    monkeypatch.setattr(pickle, "dump", slow_dump)
    capture.begin_round(["first"])
    diagnostic = capture.model_diagnostics[0]
    diagnostic.state[:] = torch.tensor([1, 1, 2, 2])
    diagnostic.hidden[128] = float("nan")
    diagnostic.input_ids[:2] = torch.tensor([27, 39])
    readback = capture.readback()
    assert readback is not None
    try:
        readback.consume()
        assert entered.wait(2), "snapshot was not sent to the pickle worker"
        for _ in range(24):
            capture.begin_round(["later"])
            # The first-hit state is the device-side write guard. Resetting it
            # while the writer owns the snapshot permits a subsequent overwrite.
            assert diagnostic.state[0].item() == 1
            assert capture.readback() is None
            readback.consume()  # A duplicate consumer cannot enqueue a second job.
        assert not list(tmp_path.glob("*.pkl"))
    finally:
        release.set()
    assert capture._dump_future is not None
    capture._dump_future.result(timeout=5)
    capture.begin_round(["resumed"])
    assert diagnostic.state[0].item() == -1
    files = list(tmp_path.glob("*.pkl"))
    assert len(files) == 1
    with files[0].open("rb") as stream:
        snapshot = pickle.load(stream)
    assert snapshot["round"] == 1
    assert snapshot["layer_id"] == 11
    assert snapshot["req_ids"] == ["first"]
    assert snapshot["input_ids"].tolist() == [27, 39]
    assert torch.isnan(snapshot["hidden_states"][1, 0])


def test_completed_dump_still_skips_twenty_full_rounds(capture, tmp_path):
    capture.begin_round(["first"])
    for diagnostic in capture.model_diagnostics:
        diagnostic.state[:] = torch.tensor([0, 1, 2, 2])
    readback = capture.readback()
    assert readback is not None
    readback.consume()
    assert capture._dump_future is not None
    capture._dump_future.result(timeout=5)
    for _ in range(20):
        capture.begin_round(["cooldown"])
        assert all(
            diagnostic.state[0].item() == 0 for diagnostic in capture.model_diagnostics
        )
    capture.begin_round(["resume"])
    assert all(
        diagnostic.state[0].item() == -1 for diagnostic in capture.model_diagnostics
    )
    # A readback retained across resume must not export the reused buffers again.
    readback.consume()
    snapshots = []
    for path in tmp_path.glob("*.pkl"):
        with path.open("rb") as stream:
            snapshots.append(pickle.load(stream))
    assert sorted(snapshot["role"] for snapshot in snapshots) == ["draft", "target"]


@pytest.mark.parametrize("roles", [("target",), ("target", "draft")])
def test_only_ten_dump_rounds_are_exported(capture, tmp_path, monkeypatch, roles):
    for _ in range(2):
        capture.begin_round(["clean"])
        readback = capture.readback()
        assert readback is not None
        readback.consume()

    for index in range(10):
        capture.begin_round([f"bad-{index}"])
        for diagnostic in capture.model_diagnostics:
            if diagnostic.role in roles:
                diagnostic.state[:] = torch.tensor([0, capture._round_id, 2, 2])
                diagnostic.hidden[0] = float("nan")
        readback = capture.readback()
        assert readback is not None
        readback.consume()
        readback.consume()
        assert capture._dump_future is not None
        capture._dump_future.result(timeout=5)
        for _ in range(20):
            capture.begin_round(["cooldown"])

    assert len(list(tmp_path.glob("*.pkl"))) == 10 * len(roles)

    def unexpected_copy(*args, **kwargs):
        pytest.fail("Exhausted dump budget must not submit another D2H copy")

    monkeypatch.setattr(torch.Tensor, "to", unexpected_copy)
    capture.begin_round(["over-budget"])
    assert capture.readback() is None
    assert capture.armed_tpu.tolist() == [False]

    capture.arm()
    capture.begin_round(["rearmed"])
    assert capture.readback() is None
    assert capture.armed_tpu.tolist() == [False]
