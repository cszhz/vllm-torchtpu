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
"""CPU readiness metadata for independently scheduled remote Draft batches.

This channel never carries auxiliary tensors or executes TPU operations. D must
publish its actual layout before entering the auxiliary transfer setup handshake.
"""

from __future__ import annotations

import dataclasses
import json
import time
from enum import StrEnum

import zmq
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.spec_decode.remote_dspark import (
    DraftProposal,
    DraftWork,
    RemoteDraftPipeline,
    RemoteDraftProposalContext,
    RemoteDraftProtocolError,
    VerifyFeedback,
)

_MAX_PLAN_BYTES = 1 << 16
GENERATION_KEY = "_remote_dspark_generation"


class RemoteQueryKind(StrEnum):
    BOOTSTRAP = "bootstrap"
    VERIFY = "verify"


@dataclasses.dataclass(frozen=True)
class RemoteDraftQuery:
    request_id: str
    generation: int
    round_id: int
    position: int
    num_tokens: int
    kind: RemoteQueryKind

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise RemoteDraftProtocolError("readiness request ID must be nonempty")
        if (
            any(
                type(value) is not int or value < 0
                for value in (self.generation, self.round_id, self.position)
            )
            or type(self.num_tokens) is not int
            or self.num_tokens <= 0
        ):
            raise RemoteDraftProtocolError("readiness counters must be nonnegative")
        if not isinstance(self.kind, RemoteQueryKind):
            raise RemoteDraftProtocolError("readiness query kind must be explicit")
        if self.kind == RemoteQueryKind.BOOTSTRAP:
            if self.round_id != 0 or self.num_tokens != 1:
                raise RemoteDraftProtocolError(
                    "bootstrap needs one token in round zero"
                )
        elif self.round_id == 0:
            raise RemoteDraftProtocolError("verify needs a positive request round")


@dataclasses.dataclass(frozen=True)
class RemoteDraftTerminal:
    request_id: str
    generation: int
    status: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or type(self.generation) is not int
            or self.generation < 0
            or type(self.status) is not int
        ):
            raise RemoteDraftProtocolError("terminal needs a request ID and generation")
        if not RequestStatus.is_finished(RequestStatus(self.status)):
            raise RemoteDraftProtocolError(
                "terminal event needs an upstream finished status"
            )


@dataclasses.dataclass(frozen=True)
class RemoteDraftPlan:
    sequence: int
    queries: tuple[RemoteDraftQuery, ...]
    chunk_ends: tuple[int, ...]
    terminals: tuple[RemoteDraftTerminal, ...] = ()

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise RemoteDraftProtocolError("readiness sequence must be nonnegative")
        ids = [query.request_id for query in self.queries]
        terminal_ids = [terminal.request_id for terminal in self.terminals]
        if ids and terminal_ids:
            raise RemoteDraftProtocolError(
                "terminal events need a separate control frame"
            )
        if (
            not (ids or terminal_ids)
            or len(ids) != len(set(ids))
            or len(terminal_ids) != len(set(terminal_ids))
            or set(ids).intersection(terminal_ids)
        ):
            raise RemoteDraftProtocolError("readiness requires distinct request IDs")
        ends = (0, *self.chunk_ends)
        if (
            (bool(ids) != bool(self.chunk_ends))
            or (self.chunk_ends and self.chunk_ends[-1] != len(ids))
            or any(type(end) is not int for end in self.chunk_ends)
            or any(left >= right for left, right in zip(ends, ends[1:]))
        ):
            raise RemoteDraftProtocolError(
                "readiness chunks must partition request rows"
            )

    def validate_layout(
        self,
        request_ids: list[str],
        positions: list[int],
        lengths: list[int],
        chunk_ends: list[int],
    ) -> None:
        expected_positions = [
            position
            for query in self.queries
            for position in range(query.position, query.position + query.num_tokens)
        ]
        if (
            request_ids != [query.request_id for query in self.queries]
            or lengths != [query.num_tokens for query in self.queries]
            or positions != expected_positions
            or tuple(chunk_ends) != self.chunk_ends
        ):
            raise RemoteDraftProtocolError(
                "remote Draft post-reorder query/chunk layout differs from D plan"
            )


@dataclasses.dataclass
class RemoteDraftSchedulerOutput(SchedulerOutput):
    # None explicitly denotes ordinary Target prefill (or a zero-token step).
    remote_draft_plan: RemoteDraftPlan | None = None


@dataclasses.dataclass
class RemoteDraftConsumerOutput(SchedulerOutput):
    """Marks consumer execution whose actual layout must be published to P."""

    remote_draft_terminals: tuple[RemoteDraftTerminal, ...] = ()


class RemoteDraftConsumerLifecycle:
    """Adapt actual D query/results to the existing request proposal lifecycle."""

    def __init__(self, max_requests: int) -> None:
        self._pipeline = RemoteDraftPipeline(max_requests)
        self._generations: dict[str, int] = {}
        self._work: dict[str, DraftWork] = {}
        self._pending: tuple[str, ...] = ()

    def begin(self, plan: RemoteDraftPlan, token_rows: list[list[int]]) -> None:
        if self._pending or len(token_rows) != len(plan.queries):
            raise RemoteDraftProtocolError("D lifecycle has pending or missing results")
        for query, tokens in zip(plan.queries, token_rows, strict=True):
            if not tokens or any(
                type(token) is not int or token < 0 for token in tokens
            ):
                raise RemoteDraftProtocolError("D feedback needs committed token IDs")
            if query.kind == RemoteQueryKind.BOOTSTRAP:
                if query.request_id in self._generations or len(tokens) != 1:
                    raise RemoteDraftProtocolError("duplicate or invalid D bootstrap")
                if len(self._generations) >= self._pipeline.max_inflight_proposals:
                    raise RemoteDraftProtocolError(
                        "D lifecycle exceeds live sequence capacity"
                    )
                self._pipeline.register(query.request_id, kv_version=query.position + 1)
                self._generations[query.request_id] = query.generation
            else:
                work = self._work.get(query.request_id)
                if (
                    self._generations.get(query.request_id) != query.generation
                    or work is None
                    or work.round_id + 1 != query.round_id
                    or work.kv_version != query.position
                    or len(tokens) > query.num_tokens
                ):
                    raise RemoteDraftProtocolError(
                        "D verify round/generation/span mismatch"
                    )
                self._pipeline.submit_feedback(
                    VerifyFeedback(
                        query.request_id,
                        work.round_id,
                        work.kv_version,
                        query.position + len(tokens),
                        len(tokens) - 1,
                        tokens[-1],
                        0,
                    )
                )
        ids = tuple(query.request_id for query in plan.queries)
        work = self._pipeline.claim_draft_batch(len(ids), request_ids=ids)
        self._work.update((item.request_id, item) for item in work)
        self._pending = ids

    def receive(self, request_ids: list[str], token_rows: list[list[int]]) -> None:
        if (
            len(request_ids) != len(set(request_ids))
            or set(request_ids) != set(self._pending)
            or len(token_rows) != len(request_ids)
        ):
            raise RemoteDraftProtocolError("proposal differs from pending D execution")
        for request_id, tokens in zip(request_ids, token_rows, strict=True):
            work = self._work[request_id]
            self._pipeline.publish_proposal(
                DraftProposal(request_id, work.round_id, work.kv_version, tuple(tokens))
            )
        self._pipeline.take_verify_batch(len(request_ids))
        self._pending = ()

    def proposal_contexts(self) -> tuple[RemoteDraftProposalContext, ...]:
        if not self._pending:
            raise RemoteDraftProtocolError("D has no pending proposal work")
        return tuple(
            RemoteDraftProposalContext(
                request_id,
                self._generations[request_id],
                self._work[request_id].round_id,
                self._work[request_id].kv_version,
            )
            for request_id in self._pending
        )

    def finish(self, terminals: tuple[RemoteDraftTerminal, ...]) -> None:
        for terminal in terminals:
            if terminal.request_id not in self._generations:
                # An upstream abort may precede the first Target execution.
                continue
            if (
                self._generations.get(terminal.request_id) != terminal.generation
                or terminal.request_id in self._pending
            ):
                raise RemoteDraftProtocolError("terminal differs from D lifecycle")
            self._pipeline.cancel(terminal.request_id)
            self._pipeline.retire(terminal.request_id)
            self._work.pop(terminal.request_id, None)
            del self._generations[terminal.request_id]


def proposal_contexts_for_plan(
    plan: RemoteDraftPlan, token_rows: list[list[int]]
) -> tuple[RemoteDraftProposalContext, ...]:
    if len(token_rows) != len(plan.queries):
        raise RemoteDraftProtocolError("feedback token rows differ from execution plan")
    contexts = []
    for query, tokens in zip(plan.queries, token_rows, strict=True):
        if (
            not tokens
            or len(tokens) > query.num_tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise RemoteDraftProtocolError(
                "feedback has an invalid committed token span"
            )
        contexts.append(
            RemoteDraftProposalContext(
                query.request_id,
                query.generation,
                query.round_id,
                query.position + len(tokens),
            )
        )
    return tuple(contexts)


def encode_plan(rank: int, plan: RemoteDraftPlan) -> bytes:
    payload = json.dumps({"rank": rank, "plan": dataclasses.asdict(plan)}).encode()
    if len(payload) > _MAX_PLAN_BYTES:
        raise RemoteDraftProtocolError("remote Draft readiness metadata exceeds 64 KiB")
    return payload


def decode_plan(payload: bytes) -> tuple[int, RemoteDraftPlan]:
    if len(payload) > _MAX_PLAN_BYTES:
        raise RemoteDraftProtocolError("remote Draft readiness metadata exceeds 64 KiB")
    try:
        envelope = json.loads(payload)
        if set(envelope) != {"rank", "plan"}:
            raise ValueError("unexpected envelope fields")
        rank, raw = envelope["rank"], envelope["plan"]
        if type(rank) is not int or rank < 0:
            raise ValueError("invalid rank")
        if set(raw) != {"sequence", "queries", "chunk_ends", "terminals"}:
            raise ValueError("unexpected plan fields")
        queries = []
        for query in raw["queries"]:
            if set(query) != {
                "request_id",
                "generation",
                "round_id",
                "position",
                "num_tokens",
                "kind",
            }:
                raise ValueError("unexpected query fields")
            queries.append(
                RemoteDraftQuery(
                    request_id=query["request_id"],
                    generation=query["generation"],
                    round_id=query["round_id"],
                    position=query["position"],
                    num_tokens=query["num_tokens"],
                    kind=RemoteQueryKind(query["kind"]),
                )
            )
        terminals = tuple(
            RemoteDraftTerminal(**terminal) for terminal in raw["terminals"]
        )
        return rank, RemoteDraftPlan(
            raw["sequence"], tuple(queries), tuple(raw["chunk_ends"]), terminals
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RemoteDraftProtocolError(
            "invalid remote Draft readiness metadata"
        ) from error


class RemoteDraftPlanReceiver:
    """Nonblocking scheduler-side receiver with one TP-agreed pending batch.

    ZMQ delivers complete frames, so a partial TCP write cannot block schedule().
    Only the scheduler thread mutates readiness/sequence state.
    """

    def __init__(
        self, endpoint: str, rank_count: int, timeout_s: float | None = None
    ) -> None:
        if rank_count <= 0:
            raise ValueError("readiness needs at least one worker rank")
        self.rank_count = rank_count
        self._timeout_s = (
            envs.VLLM_TPU_REMOTE_DSPARK_TIMEOUT_S if timeout_s is None else timeout_s
        )
        if self._timeout_s <= 0:
            raise ValueError("readiness rank timeout must be positive")
        self._first_rank_at: float | None = None
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.setsockopt(zmq.RCVHWM, rank_count)
        self._socket.setsockopt(zmq.MAXMSGSIZE, _MAX_PLAN_BYTES)
        self._socket.setsockopt(zmq.LINGER, 0)
        try:
            self._socket.bind(endpoint)
        except zmq.ZMQError:
            self.close()
            raise
        self.endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self._sequence = 0
        self._pending: dict[int, RemoteDraftPlan] = {}

    def poll(self) -> RemoteDraftPlan | None:
        for _ in range(self.rank_count - len(self._pending)):
            try:
                payload = self._socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break
            rank, plan = decode_plan(payload)
            if rank >= self.rank_count or rank in self._pending:
                raise RemoteDraftProtocolError("unexpected or duplicate readiness rank")
            if plan.sequence != self._sequence:
                raise RemoteDraftProtocolError("stale or skipped readiness sequence")
            if self._pending and plan != next(iter(self._pending.values())):
                raise RemoteDraftProtocolError(
                    "worker ranks disagree on D execution plan"
                )
            if not self._pending:
                self._first_rank_at = time.monotonic()
            self._pending[rank] = plan
        if len(self._pending) == self.rank_count:
            return next(iter(self._pending.values()))
        if (
            self._first_rank_at is not None
            and time.monotonic() - self._first_rank_at >= self._timeout_s
        ):
            missing = sorted(set(range(self.rank_count)) - self._pending.keys())
            raise RemoteDraftProtocolError(
                f"remote Draft readiness sequence {self._sequence} timed out "
                f"waiting for worker ranks {missing}"
            )
        return None

    def consume(self, plan: RemoteDraftPlan) -> None:
        if len(self._pending) != self.rank_count or any(
            pending != plan for pending in self._pending.values()
        ):
            raise RemoteDraftProtocolError("cannot consume an unagreed execution plan")
        # Dispatch acknowledgement, not a native auxiliary-consumption fence.
        self._pending.clear()
        self._first_rank_at = None
        self._sequence += 1

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


class RemoteDraftPlanSender:
    """Rank-local CPU publisher; send completion is not an execution ACK."""

    def __init__(self, endpoint: str, rank: int, timeout_s: float) -> None:
        self.rank = rank
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.SNDTIMEO, max(1, int(timeout_s * 1000)))
        self._socket.setsockopt(zmq.IMMEDIATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)

    def send(self, plan: RemoteDraftPlan) -> None:
        self._socket.send(encode_plan(self.rank, plan))

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()
