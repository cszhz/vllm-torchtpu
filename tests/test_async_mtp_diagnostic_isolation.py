# SPDX-License-Identifier: Apache-2.0
"""Async low acceptance must not submit diagnostic TPU work on output threads."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.outputs import ModelRunnerOutput

from vllm_torchtpu.core.mtp_zero_acceptance import is_mtp_diagnostic_enabled
from vllm_torchtpu.runner.mtp_diagnostic_output import maybe_wrap_mtp_output
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
)

pytestmark = pytest.mark.cpu_test


class ReadyEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


@pytest.mark.parametrize("env_switch", [None, "0", "1"])
def test_async_low_acceptance_preserves_sampled_output_without_tensor_capture(
    monkeypatch, env_switch
):
    if env_switch is None:
        monkeypatch.delenv("TPU_MTP_ZERO_ACCEPTANCE_LOG", raising=False)
    else:
        monkeypatch.setenv("TPU_MTP_ZERO_ACCEPTANCE_LOG", env_switch)
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=ReadyEvent), raising=False)
    config = SimpleNamespace(
        speculative_config=object(),
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )
    original = AsyncTPUModelRunnerOutput(
        ModelRunnerOutput(req_ids=["request-a"], req_id_to_index={"request-a": 0}),
        AsyncTPUCopyState.from_device_chunks([torch.tensor([[77, -1, -1, -1]])], [1]),
        [],
    )
    metadata = SimpleNamespace(
        draft_lengths_cpu=np.array([3]), draft_token_ids=torch.tensor([1, 2, 3])
    )
    assert is_mtp_diagnostic_enabled(config) is (env_switch != "0")
    output = maybe_wrap_mtp_output(original, config, ["request-a"], [metadata], [1])
    assert output is original, (
        "async output must not retain or read TPU diagnostic tensors"
    )
    assert output.get_output().sampled_token_ids == [[77]]
