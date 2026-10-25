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

import socket
import threading

import pytest
import torch

import vllm_torchtpu.spec_decode.remote_dspark as remote_dspark_module
from vllm_torchtpu.spec_decode.remote_dspark import (
    DraftProposal,
    PackedTensorRows,
    RemoteDeviceTensorRows,
    RemoteDraftPipeline,
    RemoteDraftProtocolError,
    RemoteDSparkTransport,
    RequestPhase,
    VerifyFeedback,
    complete_remote_dspark_handoff,
    is_remote_draft_proposal_rank,
    is_remote_dspark_request,
    mark_remote_dspark_request,
    publish_remote_dspark_handoff,
    remote_dspark_handoff_path,
    take_remote_dspark_handoff,
)


def test_remote_dspark_request_marker() -> None:
    params = mark_remote_dspark_request({"do_remote_decode": True})

    assert is_remote_dspark_request(params)
    assert not is_remote_dspark_request({"do_remote_decode": True})


def test_remote_dspark_handoff_is_atomic_and_consumed_once(tmp_path) -> None:
    request_id = "request/with unsafe path characters"
    params = {"uuid": 7, "num_tokens": 31}

    publish_remote_dspark_handoff(request_id, params, tmp_path)

    path = remote_dspark_handoff_path(request_id, tmp_path)
    assert path.parent == tmp_path
    assert "/" not in path.name
    assert take_remote_dspark_handoff(request_id, tmp_path) == params
    assert take_remote_dspark_handoff(request_id, tmp_path) is None


def test_remote_dspark_handoff_waits_for_prefill_completion(tmp_path) -> None:
    params = {"uuid": 7, "num_tokens": 31}
    publish_remote_dspark_handoff("request", params, tmp_path, ready=False)
    assert take_remote_dspark_handoff("request", tmp_path) is None

    complete_remote_dspark_handoff("request", tmp_path)
    assert take_remote_dspark_handoff("request", tmp_path) == params


def test_remote_dspark_handoff_maps_internal_id_to_proxy_id(tmp_path) -> None:
    params = {"uuid": 7, "req_id": "cmpl-external-0-internal"}
    publish_remote_dspark_handoff(
        params["req_id"],
        params,
        tmp_path,
        ready=False,
        control_request_id="external",
    )
    assert take_remote_dspark_handoff("external", tmp_path) is None

    complete_remote_dspark_handoff(params["req_id"], tmp_path)
    assert take_remote_dspark_handoff("external", tmp_path) == params
    assert take_remote_dspark_handoff(params["req_id"], tmp_path) is None


def test_transport_proposal_feedback_round_trip():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    producer: list[RemoteDSparkTransport] = []

    def accept_producer() -> None:
        producer.append(
            RemoteDSparkTransport(
                role="producer",
                producer_host="127.0.0.1",
                port=port,
                rank=0,
                timeout_s=5,
                device=torch.device("cpu"),
            )
        )

    accept_thread = threading.Thread(target=accept_producer)
    accept_thread.start()
    consumer = RemoteDSparkTransport(
        role="consumer",
        producer_host="127.0.0.1",
        port=port,
        rank=0,
        timeout_s=5,
        device=torch.device("cpu"),
    )
    accept_thread.join(timeout=5)
    assert not accept_thread.is_alive()
    producer[0].send_proposal(["req-a"], [[11, 12]])
    assert consumer.recv_proposal() == (["req-a"], [[11, 12]])

    consumer.send_feedback({"request_ids": ["req-a"]})
    assert producer[0].recv_feedback()["request_ids"] == ["req-a"]


def test_transport_accepts_bootstrap_feedback_before_first_proposal():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    producer: list[RemoteDSparkTransport] = []

    def accept_producer() -> None:
        producer.append(
            RemoteDSparkTransport(
                role="producer",
                producer_host="127.0.0.1",
                port=port,
                rank=0,
                timeout_s=5,
                device=torch.device("cpu"),
            )
        )

    accept_thread = threading.Thread(target=accept_producer)
    accept_thread.start()
    consumer = RemoteDSparkTransport(
        role="consumer",
        producer_host="127.0.0.1",
        port=port,
        rank=0,
        timeout_s=5,
        device=torch.device("cpu"),
    )
    accept_thread.join(timeout=5)
    assert not accept_thread.is_alive()

    consumer.send_feedback({"request_ids": ["req-a"]})
    assert producer[0].recv_feedback()["request_ids"] == ["req-a"]
    producer[0].send_proposal(["req-a"], [[11, 12]])
    assert consumer.recv_proposal() == (["req-a"], [[11, 12]])


def test_packed_tensor_rows_round_trip():
    tensor = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)

    packed = PackedTensorRows.from_tensor(tensor, live_rows=2)

    assert packed.tensor.shape == (2, 4)
    assert packed.transferred_bytes == 16
    assert packed.padded_bytes == 48
    restored = packed.restore(torch.device("cpu"), padding_value=-1)
    torch.testing.assert_close(restored[:2], tensor[:2])
    torch.testing.assert_close(
        restored[2:], torch.full((4, 4), -1.0, dtype=torch.bfloat16)
    )


def test_packed_tensor_rows_bundles_layers_behind_one_transfer():
    first = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    second = first + 100

    packed = PackedTensorRows.from_tensors([first, second], live_rows=2)

    assert packed.tensor.shape == (2, 2, 4)
    assert packed.transferred_bytes == 32
    assert packed.padded_bytes == 96
    restored = packed.restore_tensors(torch.device("cpu"))
    assert len(restored) == 2
    torch.testing.assert_close(restored[0][:2], first[:2])
    torch.testing.assert_close(restored[1][:2], second[:2])
    padding = torch.zeros((4, 4), dtype=torch.bfloat16)
    torch.testing.assert_close(restored[0][2:], padding)
    torch.testing.assert_close(restored[1][2:], padding)


def test_packed_tensor_rows_rejects_invalid_live_count():
    tensor = torch.zeros((2, 4))
    with pytest.raises(ValueError, match="outside"):
        PackedTensorRows.from_tensor(tensor, live_rows=3)


def test_tpu_sync_feedback_reuses_native_channel(monkeypatch):
    class _FakeWeightSynchronizer:
        def __init__(
            self,
            tensors: tuple[torch.Tensor, ...],
            endpoint: str | None = None,
        ) -> None:
            self.tensors = tensors
            self.endpoint = endpoint
            self.bind_count = 0
            self.h2d_count = 0
            synchronizers.append(self)

        def bind_weights(self, device_tensors: list[list[torch.Tensor]]) -> None:
            self.tensors = tuple(layer[0] for layer in device_tensors)
            self.bind_count += 1

        def push_weights(self, peers: list[str]) -> None:
            assert len(peers) == 1
            receiver = receivers[peers[0]]
            for source, destination in zip(self.tensors, receiver.tensors, strict=True):
                destination.copy_(source)

        def h2d(self) -> None:
            self.h2d_count += 1

        def get_local_endpoints(self) -> list[dict[str, str]]:
            assert self.endpoint is not None
            return [{"endpoint": self.endpoint}]

    receivers: dict[str, _FakeWeightSynchronizer] = {}
    synchronizers: list[_FakeWeightSynchronizer] = []

    def new_sender(
        tensors: tuple[torch.Tensor, ...], rank: int
    ) -> _FakeWeightSynchronizer:
        assert rank == 0
        return _FakeWeightSynchronizer(tensors)

    def new_receiver(
        tensors: tuple[torch.Tensor, ...], rank: int, bind_ip: str
    ) -> _FakeWeightSynchronizer:
        assert rank == 0
        assert bind_ip == "127.0.0.1"
        endpoint = f"fake://receiver-{len(receivers)}"
        receiver = _FakeWeightSynchronizer(tensors, endpoint)
        receivers[endpoint] = receiver
        return receiver

    monkeypatch.setattr(remote_dspark_module, "_new_sender_synchronizer", new_sender)
    monkeypatch.setattr(
        remote_dspark_module, "_new_receiver_synchronizer", new_receiver
    )

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    producer: list[RemoteDSparkTransport] = []

    def accept_producer() -> None:
        producer.append(
            RemoteDSparkTransport(
                role="producer",
                producer_host="127.0.0.1",
                port=port,
                rank=0,
                timeout_s=5,
                device=torch.device("cpu"),
            )
        )

    accept_thread = threading.Thread(target=accept_producer)
    accept_thread.start()
    consumer = RemoteDSparkTransport(
        role="consumer",
        producer_host="127.0.0.1",
        port=port,
        rank=0,
        timeout_s=5,
        device=torch.device("cpu"),
    )
    accept_thread.join(timeout=5)
    assert not accept_thread.is_alive()

    def receive_one(feedback: list[dict]) -> None:
        feedback.append(producer[0].recv_feedback())

    for offset in (0, 100):
        feedback: list[dict] = []
        receive_thread = threading.Thread(target=receive_one, args=(feedback,))
        receive_thread.start()
        first = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4) + offset
        second = first + 50
        transfer = consumer.send_tensor_rows(
            channel=0,
            tensors=[first, second],
            live_rows=2,
        )
        consumer.send_feedback(
            {"request_ids": ["req-a"], "aux_hidden_states": [transfer]}
        )
        receive_thread.join(timeout=5)
        assert not receive_thread.is_alive()
        received = feedback[0]["aux_hidden_states"][0]
        assert isinstance(received, RemoteDeviceTensorRows)
        restored = received.restore_tensors(torch.device("cpu"))
        torch.testing.assert_close(restored[0], first)
        torch.testing.assert_close(restored[1], second)
        assert received.live_rows == 2

    assert len(synchronizers) == 2
    sender = next(sync for sync in synchronizers if sync.endpoint is None)
    receiver = next(sync for sync in synchronizers if sync.endpoint is not None)
    assert sender.bind_count == 1
    assert receiver.h2d_count == 2


def test_only_output_rank_owns_remote_proposals():
    assert is_remote_draft_proposal_rank(0, 1)
    assert not is_remote_draft_proposal_rank(1, 1)
    assert not is_remote_draft_proposal_rank(7, 1)

    with pytest.raises(ValueError, match="requires PP=1"):
        is_remote_draft_proposal_rank(0, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        is_remote_draft_proposal_rank(-1, 1)


def test_feedback_channel_does_not_require_a_rank_local_proposal():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1] - 1

    producer: list[RemoteDSparkTransport] = []

    def accept_producer() -> None:
        producer.append(
            RemoteDSparkTransport(
                role="producer",
                producer_host="127.0.0.1",
                port=port,
                rank=1,
                timeout_s=5,
                device=torch.device("cpu"),
            )
        )

    accept_thread = threading.Thread(target=accept_producer)
    accept_thread.start()
    consumer = RemoteDSparkTransport(
        role="consumer",
        producer_host="127.0.0.1",
        port=port,
        rank=1,
        timeout_s=5,
        device=torch.device("cpu"),
    )
    accept_thread.join(timeout=5)
    assert not accept_thread.is_alive()

    for round_id in range(2):
        consumer.send_feedback({"request_ids": ["req-a"]})
        feedback = producer[0].recv_feedback()
        assert feedback["round_id"] == round_id


def test_next_proposal_can_abandon_terminal_round_without_feedback():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    producer: list[RemoteDSparkTransport] = []

    def accept_producer() -> None:
        producer.append(
            RemoteDSparkTransport(
                role="producer",
                producer_host="127.0.0.1",
                port=port,
                rank=0,
                timeout_s=5,
                device=torch.device("cpu"),
            )
        )

    accept_thread = threading.Thread(target=accept_producer)
    accept_thread.start()
    consumer = RemoteDSparkTransport(
        role="consumer",
        producer_host="127.0.0.1",
        port=port,
        rank=0,
        timeout_s=5,
        device=torch.device("cpu"),
    )
    accept_thread.join(timeout=5)

    producer[0].send_proposal(["finished"], [[11]])
    assert consumer.recv_proposal() == (["finished"], [[11]])
    producer[0].send_proposal(["next"], [[12]])
    assert consumer.recv_proposal() == (["next"], [[12]])
    consumer.send_feedback({"request_ids": ["next"]})
    assert producer[0].recv_feedback()["round_id"] == 1


def _proposal(request_id: str, round_id: int, kv_version: int) -> DraftProposal:
    return DraftProposal(request_id, round_id, kv_version, (11, 12, 13, 14, 15))


def _feedback(
    request_id: str,
    round_id: int,
    kv_version: int,
    *,
    accepted_length: int = 3,
    finished: bool = False,
) -> VerifyFeedback:
    return VerifyFeedback(
        request_id=request_id,
        round_id=round_id,
        proposal_kv_version=kv_version,
        next_kv_version=kv_version + accepted_length + 1,
        accepted_length=accepted_length,
        correction_token_id=99,
        hidden_state_bytes=accepted_length * 4096 * 2,
        finished=finished,
    )


def test_feedback_advances_round_and_committed_kv_version():
    pipeline = RemoteDraftPipeline(max_inflight_proposals=2)
    pipeline.register("req-a", kv_version=1024)

    work = pipeline.claim_draft_batch(1)
    assert work[0].round_id == 0
    assert pipeline.publish_proposal(_proposal("req-a", 0, 1024))
    assert pipeline.take_verify_batch(1)[0].token_ids == (11, 12, 13, 14, 15)
    assert pipeline.submit_feedback(_feedback("req-a", 0, 1024))

    next_work = pipeline.claim_draft_batch(1)
    assert next_work[0].round_id == 1
    assert next_work[0].kv_version == 1028


def test_duplicate_retries_are_idempotent():
    pipeline = RemoteDraftPipeline(max_inflight_proposals=1)
    pipeline.register("req-a", kv_version=8)
    pipeline.claim_draft_batch(1)
    proposal = _proposal("req-a", 0, 8)
    assert pipeline.publish_proposal(proposal)
    assert not pipeline.publish_proposal(proposal)
    feedback = _feedback("req-a", 0, 8)
    assert pipeline.submit_feedback(feedback)
    assert not pipeline.submit_feedback(feedback)


def test_stale_feedback_is_rejected():
    pipeline = RemoteDraftPipeline(max_inflight_proposals=1)
    pipeline.register("req-a", kv_version=8)
    pipeline.claim_draft_batch(1)
    pipeline.publish_proposal(_proposal("req-a", 0, 8))

    with pytest.raises(RemoteDraftProtocolError, match="expected round=0"):
        pipeline.submit_feedback(_feedback("req-a", 1, 8))


def test_feedback_cannot_accept_beyond_proposal():
    pipeline = RemoteDraftPipeline(max_inflight_proposals=1)
    pipeline.register("req-a", kv_version=8)
    pipeline.claim_draft_batch(1)
    pipeline.publish_proposal(_proposal("req-a", 0, 8))

    with pytest.raises(RemoteDraftProtocolError, match="accepted 6 tokens"):
        pipeline.submit_feedback(_feedback("req-a", 0, 8, accepted_length=6))


def test_backpressure_bounds_unverified_proposals():
    pipeline = RemoteDraftPipeline(max_inflight_proposals=1)
    pipeline.register("req-a", kv_version=0)
    pipeline.register("req-b", kv_version=0)

    first = pipeline.claim_draft_batch(2)
    assert [work.request_id for work in first] == ["req-a"]
    assert pipeline.claim_draft_batch(1) == ()
    pipeline.publish_proposal(_proposal("req-a", 0, 0))
    assert pipeline.claim_draft_batch(1) == ()
    pipeline.submit_feedback(_feedback("req-a", 0, 0))
    assert pipeline.claim_draft_batch(1)[0].request_id == "req-b"


def test_cancel_removes_queued_proposal_and_releases_capacity():
    pipeline = RemoteDraftPipeline(max_inflight_proposals=1)
    pipeline.register("req-a", kv_version=0)
    pipeline.register("req-b", kv_version=0)
    pipeline.claim_draft_batch(1)
    pipeline.publish_proposal(_proposal("req-a", 0, 0))

    pipeline.cancel("req-a")

    assert pipeline.phase("req-a") == RequestPhase.CANCELLED
    assert pipeline.take_verify_batch(1) == ()
    assert pipeline.claim_draft_batch(1)[0].request_id == "req-b"


def test_finished_feedback_does_not_requeue_request():
    pipeline = RemoteDraftPipeline(max_inflight_proposals=1)
    pipeline.register("req-a", kv_version=0)
    pipeline.claim_draft_batch(1)
    pipeline.publish_proposal(_proposal("req-a", 0, 0))
    pipeline.submit_feedback(_feedback("req-a", 0, 0, finished=True))

    assert pipeline.phase("req-a") == RequestPhase.FINISHED
    assert pipeline.claim_draft_batch(1) == ()
