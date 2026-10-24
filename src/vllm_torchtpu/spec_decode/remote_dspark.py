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
"""Protocol and rank-local transport for a remote DSpark P/D pipeline."""

from __future__ import annotations

import collections
import dataclasses
import pickle
import socket
import struct
import time
from enum import StrEnum
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_FRAME_HEADER = struct.Struct("!Q")
_MAX_FRAME_BYTES = 1 << 30


def is_remote_draft_proposal_rank(rank: int, pipeline_parallel_size: int) -> bool:
    """Return whether this worker owns the engine-level proposal channel.

    vLLM invokes ``take_draft_token_ids`` on every model worker, but consumes
    only the output worker's reply. Remote proposals therefore belong to that
    one worker; blocking every TP rank on a proposal can prevent the remaining
    ranks from entering the next target-model collective.
    """
    if pipeline_parallel_size != 1:
        raise ValueError("remote DSpark proposal routing currently requires PP=1")
    if rank < 0:
        raise ValueError("remote DSpark rank must be nonnegative")
    return rank == 0


class RemoteDraftProtocolError(RuntimeError):
    """A remote draft message violates request ordering or KV ownership."""


@dataclasses.dataclass(frozen=True)
class PackedTensorRows:
    """A tensor transferred with only its live leading-dimension rows."""

    tensor: torch.Tensor
    padded_rows: int

    def __post_init__(self) -> None:
        if self.tensor.device.type != "cpu":
            raise ValueError("packed remote tensor must reside on CPU")
        if self.padded_rows < self.tensor.shape[0]:
            raise ValueError(
                "packed remote tensor row count exceeds its padded row count"
            )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, live_rows: int) -> PackedTensorRows:
        if live_rows < 0 or live_rows > tensor.shape[0]:
            raise ValueError(
                f"live row count {live_rows} is outside [0, {tensor.shape[0]}]"
            )
        host_tensor = tensor.detach().cpu()
        return cls(host_tensor[:live_rows].clone(), tensor.shape[0])

    @classmethod
    def from_tensors(
        cls, tensors: list[torch.Tensor], live_rows: int
    ) -> PackedTensorRows:
        """Pack same-shaped tensors behind one device-to-host transfer."""
        if not tensors:
            raise ValueError("remote tensor bundle must not be empty")
        first = tensors[0]
        signature = (first.shape, first.dtype, first.device)
        if any(
            (tensor.shape, tensor.dtype, tensor.device) != signature
            for tensor in tensors[1:]
        ):
            raise ValueError("remote tensor bundle members must have one signature")
        return cls.from_tensor(torch.stack(tensors, dim=1), live_rows)

    @property
    def transferred_bytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()

    @property
    def padded_bytes(self) -> int:
        if self.tensor.shape[0] == 0:
            return 0
        return self.transferred_bytes * self.padded_rows // self.tensor.shape[0]

    def restore(self, device: torch.device, *, padding_value: int = 0) -> torch.Tensor:
        tensor = self.tensor
        missing_rows = self.padded_rows - tensor.shape[0]
        if missing_rows:
            padding = torch.full(
                (missing_rows, *tensor.shape[1:]),
                padding_value,
                dtype=tensor.dtype,
                device="cpu",
            )
            tensor = torch.cat((tensor, padding), dim=0)
        return tensor.to(device, non_blocking=True)

    def restore_tensors(self, device: torch.device) -> list[torch.Tensor]:
        """Restore one transfer and expose its layer dimension as views."""
        if self.tensor.ndim < 2:
            raise ValueError("remote tensor bundle has no layer dimension")
        return list(self.restore(device).unbind(dim=1))


class RemoteDSparkTransport:
    """One full-duplex TCP stream between matching P/D tensor-parallel ranks.

    The producer listens and the consumer connects. Messages alternate between
    P-to-D proposals and D-to-P feedback, so a single blocking stream provides
    backpressure without a background queue. Payloads contain trusted local
    experiment data only; pickle is intentionally not exposed on a public
    listener.
    """

    def __init__(
        self,
        *,
        role: str,
        producer_host: str,
        port: int,
        rank: int,
        timeout_s: float,
    ) -> None:
        if role not in ("producer", "consumer"):
            raise ValueError(f"remote DSpark role must be producer or consumer: {role}")
        if not producer_host:
            raise ValueError("remote DSpark producer host must not be empty")
        if port <= 0 or port + rank > 65535:
            raise ValueError(f"invalid remote DSpark port/rank: {port}+{rank}")
        if timeout_s <= 0:
            raise ValueError("remote DSpark timeout must be positive")

        self.role = role
        self.rank = rank
        self._round_id = 0
        self._proposal_outstanding = False
        address = (producer_host, port + rank)
        if role == "producer":
            stream = self._accept(address, timeout_s)
        else:
            stream = self._connect(address, timeout_s)
        stream.settimeout(timeout_s)
        stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._stream = stream
        self._exchange_handshake()
        logger.info(
            "Remote DSpark transport connected: role=%s rank=%d peer=%s:%d",
            role,
            rank,
            producer_host,
            port + rank,
        )

    @staticmethod
    def _accept(address: tuple[str, int], timeout_s: float) -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", address[1]))
        listener.listen(1)
        listener.settimeout(timeout_s)
        try:
            stream, _ = listener.accept()
            return stream
        finally:
            listener.close()

    @staticmethod
    def _connect(address: tuple[str, int], timeout_s: float) -> socket.socket:
        deadline = time.monotonic() + timeout_s
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                stream.settimeout(min(5.0, max(0.1, deadline - time.monotonic())))
                stream.connect(address)
                return stream
            except OSError as exc:
                last_error = exc
                stream.close()
                time.sleep(0.2)
        raise TimeoutError(
            f"timed out connecting remote DSpark rank to {address[0]}:{address[1]}"
        ) from last_error

    def _exchange_handshake(self) -> None:
        local = {"kind": "handshake", "role": self.role, "rank": self.rank}
        if self.role == "producer":
            self._send(local)
            peer = self._recv()
        else:
            peer = self._recv()
            self._send(local)
        expected_role = "consumer" if self.role == "producer" else "producer"
        if peer != {"kind": "handshake", "role": expected_role, "rank": self.rank}:
            raise RemoteDraftProtocolError(
                f"remote DSpark handshake mismatch on rank {self.rank}: {peer!r}"
            )

    def send_proposal(self, request_ids: list[str], token_ids: list[list[int]]) -> None:
        if self.role != "producer":
            raise RemoteDraftProtocolError("only the producer sends proposals")
        if self._proposal_outstanding:
            if self.rank == 0:
                logger.info(
                    "Remote DSpark abandoning terminal proposal: round=%d",
                    self._round_id,
                )
            self._round_id += 1
        if self.rank == 0:
            logger.info(
                "Remote DSpark proposal send: round=%d requests=%d lengths=%s",
                self._round_id,
                len(request_ids),
                [len(tokens) for tokens in token_ids],
            )
        self._send(
            {
                "kind": "proposal",
                "round_id": self._round_id,
                "request_ids": request_ids,
                "token_ids": token_ids,
            }
        )
        self._proposal_outstanding = True

    def recv_proposal(self) -> tuple[list[str], list[list[int]]]:
        if self.role != "consumer":
            raise RemoteDraftProtocolError("only the consumer receives proposals")
        message = self._recv()
        round_id = message.get("round_id")
        if (
            message.get("kind") != "proposal"
            or not isinstance(round_id, int)
            or round_id not in (self._round_id, self._round_id + 1)
        ):
            raise RemoteDraftProtocolError(
                f"remote DSpark rank {self.rank} expected proposal round "
                f"{self._round_id} or {self._round_id + 1}, got "
                f"{message.get('kind')} round {round_id}"
            )
        if round_id == self._round_id + 1:
            if self.rank == 0:
                logger.info(
                    "Remote DSpark consumer skipped terminal feedback: round=%d",
                    self._round_id,
                )
            self._round_id = round_id
        if self.rank == 0:
            logger.info(
                "Remote DSpark proposal receive: round=%d requests=%d",
                self._round_id,
                len(message["request_ids"]),
            )
        return message["request_ids"], message["token_ids"]

    def send_feedback(self, payload: dict[str, Any]) -> None:
        if self.role != "consumer":
            raise RemoteDraftProtocolError("only the consumer sends feedback")
        if self.rank == 0:
            logger.info(
                "Remote DSpark feedback send: round=%d requests=%d",
                self._round_id,
                len(payload.get("request_ids", ())),
            )
        self._send({"kind": "feedback", "round_id": self._round_id, **payload})
        self._round_id += 1

    def recv_feedback(self) -> dict[str, Any]:
        if self.role != "producer":
            raise RemoteDraftProtocolError("only the producer receives feedback")
        message = self._recv_round("feedback")
        if self.rank == 0:
            logger.info(
                "Remote DSpark feedback receive: round=%d requests=%d",
                self._round_id,
                len(message.get("request_ids", ())),
            )
        self._proposal_outstanding = False
        self._round_id += 1
        return message

    def _recv_round(self, kind: str) -> dict[str, Any]:
        message = self._recv()
        if message.get("kind") != kind or message.get("round_id") != self._round_id:
            raise RemoteDraftProtocolError(
                f"remote DSpark rank {self.rank} expected {kind} round "
                f"{self._round_id}, got {message.get('kind')} round "
                f"{message.get('round_id')}"
            )
        return message

    def _send(self, message: dict[str, Any]) -> None:
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > _MAX_FRAME_BYTES:
            raise RemoteDraftProtocolError(
                f"remote DSpark frame is too large: {len(payload)} bytes"
            )
        self._stream.sendall(_FRAME_HEADER.pack(len(payload)))
        self._stream.sendall(payload)

    def _recv(self) -> dict[str, Any]:
        header = self._recv_exact(_FRAME_HEADER.size)
        (size,) = _FRAME_HEADER.unpack(header)
        if size > _MAX_FRAME_BYTES:
            raise RemoteDraftProtocolError(
                f"remote DSpark peer declared an oversized frame: {size} bytes"
            )
        message = pickle.loads(self._recv_exact(size))  # noqa: S301
        if not isinstance(message, dict):
            raise RemoteDraftProtocolError(
                f"remote DSpark frame must be a dict, got {type(message).__name__}"
            )
        return message

    def _recv_exact(self, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._stream.recv(size - len(chunks))
            if not chunk:
                raise ConnectionError(
                    f"remote DSpark peer closed rank {self.rank} connection"
                )
            chunks.extend(chunk)
        return bytes(chunks)


class RequestPhase(StrEnum):
    READY_TO_DRAFT = "ready_to_draft"
    DRAFTING = "drafting"
    WAITING_FOR_VERIFY = "waiting_for_verify"
    CANCELLED = "cancelled"
    FINISHED = "finished"


@dataclasses.dataclass(frozen=True)
class DraftWork:
    request_id: str
    round_id: int
    kv_version: int


@dataclasses.dataclass(frozen=True)
class DraftProposal:
    request_id: str
    round_id: int
    kv_version: int
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.round_id < 0 or self.kv_version < 0:
            raise ValueError("round_id and kv_version must be nonnegative")
        if not self.token_ids:
            raise ValueError("a draft proposal must contain at least one token")


@dataclasses.dataclass(frozen=True)
class VerifyFeedback:
    request_id: str
    round_id: int
    proposal_kv_version: int
    next_kv_version: int
    accepted_length: int
    correction_token_id: int
    hidden_state_bytes: int
    finished: bool = False

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        nonnegative = (
            self.round_id,
            self.proposal_kv_version,
            self.next_kv_version,
            self.accepted_length,
            self.hidden_state_bytes,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("feedback counters and sizes must be nonnegative")


@dataclasses.dataclass
class _RequestState:
    round_id: int
    kv_version: int
    phase: RequestPhase = RequestPhase.READY_TO_DRAFT
    proposal: DraftProposal | None = None
    last_feedback: VerifyFeedback | None = None


class RemoteDraftPipeline:
    """Ordered per-request protocol with bounded proposal backpressure.

    P claims ready work, publishes candidates, and later applies D feedback.
    D takes published proposals and submits exactly one feedback for each
    round. Duplicate proposal/feedback retries are idempotent; stale or
    conflicting messages fail loudly.
    """

    def __init__(self, max_inflight_proposals: int):
        if max_inflight_proposals <= 0:
            raise ValueError("max_inflight_proposals must be positive")
        self.max_inflight_proposals = max_inflight_proposals
        self._requests: dict[str, _RequestState] = {}
        self._draft_ready: collections.deque[str] = collections.deque()
        self._verify_ready: collections.deque[DraftProposal] = collections.deque()
        self._reserved_slots = 0

    @property
    def inflight_proposals(self) -> int:
        return self._reserved_slots

    def phase(self, request_id: str) -> RequestPhase:
        return self._state(request_id).phase

    def register(self, request_id: str, *, kv_version: int) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if kv_version < 0:
            raise ValueError("kv_version must be nonnegative")
        if request_id in self._requests:
            raise RemoteDraftProtocolError(
                f"request {request_id!r} is already registered"
            )
        self._requests[request_id] = _RequestState(round_id=0, kv_version=kv_version)
        self._draft_ready.append(request_id)

    def claim_draft_batch(self, limit: int) -> tuple[DraftWork, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        available = self.max_inflight_proposals - self._reserved_slots
        count = min(limit, available, len(self._draft_ready))
        work: list[DraftWork] = []
        for _ in range(count):
            request_id = self._draft_ready.popleft()
            state = self._state(request_id)
            self._expect_phase(request_id, state, RequestPhase.READY_TO_DRAFT)
            state.phase = RequestPhase.DRAFTING
            self._reserved_slots += 1
            work.append(DraftWork(request_id, state.round_id, state.kv_version))
        return tuple(work)

    def publish_proposal(self, proposal: DraftProposal) -> bool:
        state = self._state(proposal.request_id)
        if (
            state.proposal == proposal
            and state.phase == RequestPhase.WAITING_FOR_VERIFY
        ):
            return False
        self._expect_phase(proposal.request_id, state, RequestPhase.DRAFTING)
        self._expect_round_and_version(
            proposal.request_id,
            state,
            proposal.round_id,
            proposal.kv_version,
        )
        state.proposal = proposal
        state.phase = RequestPhase.WAITING_FOR_VERIFY
        self._verify_ready.append(proposal)
        return True

    def take_verify_batch(self, limit: int) -> tuple[DraftProposal, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        proposals: list[DraftProposal] = []
        for _ in range(min(limit, len(self._verify_ready))):
            proposals.append(self._verify_ready.popleft())
        return tuple(proposals)

    def submit_feedback(self, feedback: VerifyFeedback) -> bool:
        state = self._state(feedback.request_id)
        if state.last_feedback == feedback and state.round_id == feedback.round_id + 1:
            return False
        self._expect_phase(feedback.request_id, state, RequestPhase.WAITING_FOR_VERIFY)
        proposal = state.proposal
        assert proposal is not None
        self._expect_round_and_version(
            feedback.request_id,
            state,
            feedback.round_id,
            feedback.proposal_kv_version,
        )
        if feedback.accepted_length > len(proposal.token_ids):
            raise RemoteDraftProtocolError(
                f"request {feedback.request_id!r} round {feedback.round_id} accepted "
                f"{feedback.accepted_length} tokens from a "
                f"{len(proposal.token_ids)}-token "
                "proposal"
            )
        expected_next_version = state.kv_version + feedback.accepted_length + 1
        if feedback.next_kv_version != expected_next_version:
            raise RemoteDraftProtocolError(
                f"request {feedback.request_id!r} round {feedback.round_id} next "
                f"KV version {feedback.next_kv_version} != {expected_next_version}"
            )

        state.last_feedback = feedback
        state.proposal = None
        state.round_id += 1
        state.kv_version = feedback.next_kv_version
        self._reserved_slots -= 1
        if feedback.finished:
            state.phase = RequestPhase.FINISHED
        else:
            state.phase = RequestPhase.READY_TO_DRAFT
            self._draft_ready.append(feedback.request_id)
        return True

    def cancel(self, request_id: str) -> None:
        state = self._state(request_id)
        if state.phase in (RequestPhase.CANCELLED, RequestPhase.FINISHED):
            return
        if state.phase in (RequestPhase.DRAFTING, RequestPhase.WAITING_FOR_VERIFY):
            self._reserved_slots -= 1
        state.phase = RequestPhase.CANCELLED
        state.proposal = None
        self._draft_ready = collections.deque(
            queued for queued in self._draft_ready if queued != request_id
        )
        self._verify_ready = collections.deque(
            proposal
            for proposal in self._verify_ready
            if proposal.request_id != request_id
        )

    def _state(self, request_id: str) -> _RequestState:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise RemoteDraftProtocolError(
                f"request {request_id!r} is not registered"
            ) from exc

    @staticmethod
    def _expect_phase(
        request_id: str, state: _RequestState, expected: RequestPhase
    ) -> None:
        if state.phase != expected:
            raise RemoteDraftProtocolError(
                f"request {request_id!r} is {state.phase}, expected {expected}"
            )

    @staticmethod
    def _expect_round_and_version(
        request_id: str,
        state: _RequestState,
        round_id: int,
        kv_version: int,
    ) -> None:
        if round_id != state.round_id or kv_version != state.kv_version:
            raise RemoteDraftProtocolError(
                f"request {request_id!r} expected round={state.round_id}, "
                f"kv_version={state.kv_version}; got round={round_id}, "
                f"kv_version={kv_version}"
            )
