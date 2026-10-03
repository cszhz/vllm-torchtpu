# SPDX-License-Identifier: Apache-2.0
"""Protect asynchronous store snapshots, chunk coverage and padding semantics."""

import numpy as np
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_torchtpu.distributed.kv_transfer import kv_save_nan_check
from vllm_torchtpu.distributed.kv_transfer.kv_save_nan_check import (
    KVSaveNanCheck,
    SupportsKVSaveNanCheck,
    begin_kv_save_nan_check,
    wait_for_save_with_nan_check,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    SequenceLayoutPlan,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def events(monkeypatch):
    events = []

    class Event:
        ready = False
        synchronized = False

        def __init__(self):
            events.append(self)

        def record(self):
            assert not self.synchronized

        def query(self):
            return self.ready

        def synchronize(self):
            assert self.ready, "The store consumer must wait for copy readiness"
            self.synchronized = True

    monkeypatch.setattr(kv_save_nan_check, "TpuEvent", Event)
    return events


def plan(size, valid, token_slice=None, mapping=None):
    return SequenceLayoutPlan(
        descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
        token_slice=token_slice or slice(0, size),
        global_num_tokens=valid,
        global_padded_num_tokens=size,
        local_num_tokens=valid,
        local_padded_num_tokens=size,
        _packed_to_request_major_token_indices=mapping,
    )


def test_earlier_chunk_and_request_mapping_survive_reused_host_buffers(events):
    check = KVSaveNanCheck()
    query = np.array([0, 2, 3])
    hidden = torch.tensor([[0.0], [float("nan")], [1.0], [float("nan")]])
    check.record(hidden, ("bad", "good"), query, plan(4, 3))
    query[:] = 0
    check.record(torch.ones(4, 1), ("later",), np.array([0, 3]), plan(4, 3))
    assert not check.is_ready()
    assert not any(event.synchronized for event in events)
    for event in events:
        event.ready = True
    assert check.is_ready()
    assert check.bad_request_ids() == {"bad"}
    assert all(event.synchronized for event in events)
    assert check.bad_request_ids() == {"bad"}


@pytest.mark.parametrize("bad_row,expected", [(0, {"b"}), (1, set()), (2, {"a"})])
def test_pcp_repacked_tokens_ignore_padding(events, bad_row, expected):
    check = KVSaveNanCheck()
    mapping = np.array([0, 2, -1, 1, 3, 4, -1, -1])
    hidden = torch.ones(4, 2)
    hidden[bad_row, 0] = float("nan")
    check.record(
        hidden, ("a", "b"), np.array([0, 2, 5]), plan(8, 5, slice(1, 5), mapping)
    )
    events[0].ready = True
    assert check.bad_request_ids() == expected


def test_steps_hold_independent_copy_events_and_verdicts(events):
    bad, good = KVSaveNanCheck(), KVSaveNanCheck()
    for check, value in [(bad, float("nan")), (good, 0.0)]:
        check.record(torch.tensor([[value]]), ("same",), np.array([0, 1]), plan(1, 1))
    events[1].ready = True
    assert good.is_ready() and not bad.is_ready()
    assert good.bad_request_ids() == set()
    events[0].ready = True
    assert bad.bad_request_ids() == {"same"}


def test_multi_connector_only_passes_snapshot_to_opted_in_stores():
    class Plain:
        calls = 0

        def wait_for_save(self):
            self.calls += 1

    class Checking(SupportsKVSaveNanCheck):
        needs_kv_save_nan_check = False
        snapshot = None
        calls = 0

        def wait_for_save(self, nan_check=None):
            self.snapshot = nan_check
            self.calls += 1

    plain, checking, disabled = Plain(), Checking(), Checking()
    multi = object.__new__(MultiConnector)
    multi._connectors = [plain, checking, disabled]
    assert begin_kv_save_nan_check(multi) is None
    checking.needs_kv_save_nan_check = True
    check = begin_kv_save_nan_check(multi)
    assert check is not None
    wait_for_save_with_nan_check(multi, check)
    assert plain.calls == 1 and checking.snapshot is check
    assert disabled.calls == 1 and disabled.snapshot is None
