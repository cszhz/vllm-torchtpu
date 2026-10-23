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

import pytest

from vllm_torchtpu.spec_decode.remote_dspark import (
    DraftProposal,
    RemoteDraftPipeline,
    RemoteDraftProtocolError,
    RequestPhase,
    VerifyFeedback,
)


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
