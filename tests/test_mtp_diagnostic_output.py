# SPDX-License-Identifier: Apache-2.0
"""Verified draft snapshots must stay aligned across async batch changes."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.outputs import ModelRunnerOutput

from vllm_torchtpu.runner.mtp_diagnostic_output import maybe_wrap_mtp_output
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
)

pytestmark = pytest.mark.cpu_test


class CopyEvent:
    def __init__(self):
        self.recorded = False
        self.waited = False

    def record(self):
        self.recorded = True

    def synchronize(self):
        assert self.recorded
        self.waited = True


@pytest.fixture
def output(monkeypatch):
    monkeypatch.setenv("TPU_MTP_ZERO_ACCEPTANCE_LOG", "1")
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=CopyEvent), raising=False)
    base = ModelRunnerOutput(
        req_ids=["a", "b", "c", "d", "e"],
        req_id_to_index={rid: i for i, rid in enumerate("abcde")},
        prompt_logprobs_dict={"a": None},
    )
    copies = AsyncTPUCopyState.from_device_chunks(
        [torch.tensor([[99, -1], [98, -1], [97, -1], [96, -1], [95, -1], [0, 0]])],
        [5],
    )
    return AsyncTPUModelRunnerOutput(base, copies, [2]), base


def test_verified_drafts_survive_padding_reordering_and_transport(output):
    original, base = output
    config = SimpleNamespace(speculative_config=object(), kv_transfer_config=None)
    first = SimpleNamespace(
        draft_lengths_cpu=np.array([2, 1, 0]),
        draft_token_ids=torch.tensor([101, 102, 201, 999, 999, 999]),
    )
    last = SimpleNamespace(
        draft_lengths_cpu=np.array([1]), draft_token_ids=torch.tensor([501, 999])
    )
    req_ids = base.req_ids.copy()
    wrapped = maybe_wrap_mtp_output(
        original, config, req_ids, [first, None, last], [3, 1, 1]
    )
    # The next execute_model may reorder requests and replace cached metadata.
    req_ids.reverse()
    first.draft_lengths_cpu[:] = 0
    first.draft_token_ids = torch.tensor([901, 902, 903])
    assert not wrapped.draft_copy.completed
    result = wrapped.get_output()
    assert wrapped.draft_copy.completed
    assert result.verified_draft_token_ids == {
        "a": [101, 102],
        "b": [201],
        "c": [],
        "d": [],
        "e": [501],
    }
    assert result.sampled_token_ids == [[99], [98], [], [96], [95]]
    assert result.req_id_to_index is base.req_id_to_index
    assert result.prompt_logprobs_dict is base.prompt_logprobs_dict
    assert (
        pickle.loads(pickle.dumps(result)).verified_draft_token_ids
        == result.verified_draft_token_ids
    )


@pytest.mark.parametrize(
    "enabled,speculative,consumer",
    [(False, True, None), (True, False, None), (True, True, False)],
)
def test_inactive_output_preserves_identity(
    output, monkeypatch, enabled, speculative, consumer
):
    original, base = output
    monkeypatch.setenv("TPU_MTP_ZERO_ACCEPTANCE_LOG", str(int(enabled)))
    config = SimpleNamespace(
        speculative_config=object() if speculative else None,
        kv_transfer_config=None
        if consumer is None
        else SimpleNamespace(is_kv_consumer=consumer),
    )
    assert maybe_wrap_mtp_output(original, config, base.req_ids, [], []) is original


def test_bootstrap_without_drafts_preserves_identity(output):
    original, base = output
    config = SimpleNamespace(speculative_config=object(), kv_transfer_config=None)
    assert (
        maybe_wrap_mtp_output(original, config, base.req_ids, [None], [5]) is original
    )
