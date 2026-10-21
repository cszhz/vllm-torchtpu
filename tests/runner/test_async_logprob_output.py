# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for TPU output ordering and asynchronous copy lifetime."""
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput

from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncTPUCopyState, AsyncTPUModelRunnerOutput)


@pytest.fixture
def delayed_device(monkeypatch):
    # Only the hardware copy/event boundary is substituted. Destinations
    # remain unfilled until the recorded event completes; all host work is real.
    pending = []

    class DeviceTensor:
        device = SimpleNamespace(type="tpu")

        def __init__(self, data, dtype):
            self.data = torch.tensor(data, dtype=dtype)

        def to(self, device, *, non_blocking):
            assert device == "cpu" and non_blocking
            destination = torch.full_like(self.data, -99)
            pending.append((weakref.ref(self), destination))
            return destination

    class Event:

        def record(self):
            self.jobs = tuple(pending)

        def synchronize(self):
            for source, destination in self.jobs:
                tensor = source()
                assert tensor is not None, "source freed before D2H completed"
                destination.copy_(tensor.data)

    monkeypatch.setattr(torch,
                        "tpu",
                        SimpleNamespace(Event=Event),
                        raising=False)
    return DeviceTensor


def output(copy_state, discarded=()):
    model_output = ModelRunnerOutput(req_ids=["a", "b", "c"],
                                     req_id_to_index={
                                         "a": 0,
                                         "b": 1,
                                         "c": 2
                                     },
                                     sampled_token_ids=[],
                                     logprobs=[],
                                     prompt_logprobs_dict={},
                                     pooler_output=[])
    return AsyncTPUModelRunnerOutput(model_output, copy_state, list(discarded))


def test_async_logprobs_wait_trim_and_preserve_request_order(delayed_device):
    tensor = delayed_device
    tokens = [
        tensor([[11], [99], [99]], torch.int64),
        tensor([[22], [23]], torch.int64)
    ]
    logprobs = [
        LogprobsTensors(
            tensor([[11, 41], [999, 999], [999, 999]], torch.int32),
            tensor([[-1., -2.], [-99., -99.], [-99., -99.]], torch.float32),
            tensor([3, 99, 99], torch.int32)),
        LogprobsTensors(tensor([[22, 42], [23, 43]], torch.int32),
                        tensor([[-3., -4.], [-5., -6.]], torch.float32),
                        tensor([4, 5], torch.int32)),
    ]
    state = AsyncTPUCopyState.from_device_chunks(tokens, [1, 2], logprobs)
    result = output(state, discarded=[1])
    del tokens, logprobs
    gc.collect()
    actual = result.get_output()
    assert actual.sampled_token_ids == [[11], [], [23]]
    # Discarded partial-prefill tokens do not remove or reorder logprob rows.
    assert actual.logprobs.logprob_token_ids.tolist() == [[11, 41], [22, 42],
                                                          [23, 43]]
    assert actual.logprobs.logprobs.tolist() == [[-1., -2.], [-3., -4.],
                                                 [-5., -6.]]
    assert actual.logprobs.sampled_token_ranks.tolist() == [3, 4, 5]
    assert actual.logprobs.cu_num_generated_tokens is None
    assert result.get_output().sampled_token_ids == actual.sampled_token_ids


def test_token_only_copy_preserves_existing_empty_logprobs(delayed_device):
    tensor = delayed_device([[17], [99]], torch.int64)
    state = AsyncTPUCopyState.from_device_chunks([tensor], [1])
    del tensor
    gc.collect()
    actual = output(state).get_output()
    assert actual.sampled_token_ids == [[17]]
    assert actual.logprobs == []
