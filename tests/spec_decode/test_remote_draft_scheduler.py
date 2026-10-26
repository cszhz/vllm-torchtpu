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
"""Real CPU scheduler and ZMQ control integration; no model forward or TPU."""

import copy
import json
import pickle
import queue
import socket
import time
from concurrent.futures import Future, ThreadPoolExecutor

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core.remote_draft_scheduler import (
    RemoteDraftConsumerScheduler,
    RemoteDraftScheduler,
)
from vllm_torchtpu.spec_decode.remote_draft_control import (
    GENERATION_KEY,
    RemoteDraftConsumerLifecycle,
    RemoteDraftPlan,
    RemoteDraftPlanReceiver,
    RemoteDraftPlanSender,
    RemoteDraftQuery,
    RemoteDraftSchedulerOutput,
    RemoteDraftTerminal,
    RemoteQueryKind,
    decode_plan,
    encode_plan,
    proposal_contexts_for_plan,
)
from vllm_torchtpu.spec_decode.remote_dspark import (
    RemoteDraftProposalContext,
    RemoteDraftProtocolError,
    RemoteDSparkTransport,
    complete_remote_dspark_handoff,
    publish_remote_dspark_handoff,
    take_remote_dspark_handoff,
)


@pytest.fixture
def transports():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    options = dict(
        producer_host="127.0.0.1",
        port=port,
        rank=0,
        timeout_s=3,
        device=torch.device("cpu"),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(RemoteDSparkTransport, role="producer", **options)
        consumer = RemoteDSparkTransport(role="consumer", **options)
        producer = future.result(timeout=3)
    try:
        yield producer, consumer
    finally:
        producer.close()
        consumer.close()


def test_transport_close_wakes_peer_and_is_idempotent(transports):
    producer, consumer = transports
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(consumer.recv_proposal)
        producer.close()
        producer.close()
        with pytest.raises(ConnectionError, match="peer closed"):
            pending.result(timeout=2)
    consumer.close()
    consumer.close()


@pytest.mark.parametrize("state", [PauseState.PAUSED_ALL, PauseState.PAUSED_NEW])
def test_remote_pause_rejected_without_consuming_plan_or_releasing_kv(runtime, state):
    scheduler, receiver, senders = runtime
    request = add(scheduler, "a")
    complete(scheduler, scheduler.schedule())
    blocks = scheduler.kv_cache_manager.get_block_ids("a")
    publish(receiver, senders, bootstrap())
    with pytest.raises(NotImplementedError, match="coordinated P/D draining"):
        scheduler.set_pause_state(state)
    assert scheduler.pause_state == PauseState.UNPAUSED
    assert request.num_computed_tokens == 3
    assert scheduler.kv_cache_manager.get_block_ids("a") == blocks
    assert scheduler.schedule().remote_draft_plan == bootstrap()


@pytest.mark.parametrize("bad_field", [None, "generation", "round_id", "kv_version"])
def test_planned_proposal_wire_validates_request_context(transports, bad_field):
    producer, consumer = transports
    lifecycle = RemoteDraftConsumerLifecycle(1)
    plan = bootstrap()
    lifecycle.begin(plan, [[20]])
    consumer.send_feedback({"execution_plan": plan})
    feedback = producer.recv_feedback()
    assert feedback["execution_plan"] == plan
    contexts = proposal_contexts_for_plan(plan, [[20]])
    assert contexts == lifecycle.proposal_contexts()
    if bad_field is not None:
        values = vars(contexts[0]).copy()
        values[bad_field] += 1
        contexts = (RemoteDraftProposalContext(**values),)
    producer.send_proposal(["a"], [[11, 12]], contexts=contexts)
    if bad_field is not None:
        with pytest.raises(RemoteDraftProtocolError, match="generation/round/frontier"):
            consumer.recv_proposal(contexts=lifecycle.proposal_contexts())
    else:
        ids, tokens = consumer.recv_proposal(contexts=lifecycle.proposal_contexts())
        lifecycle.receive(ids, tokens)
        assert lifecycle._pipeline.inflight_proposals == 1


def test_planned_batches_keep_global_rounds_with_partial_request_finish(transports):
    producer, consumer = transports
    lifecycle = RemoteDraftConsumerLifecycle(2)
    plans = [bootstrap(), bootstrap(1, "b", position=5)]
    for plan in plans:
        lifecycle.begin(plan, [[20]])
        consumer.send_feedback({"execution_plan": plan})
        assert producer.recv_feedback()["execution_plan"] == plan
        contexts = proposal_contexts_for_plan(plan, [[20]])
        ids = [context.request_id for context in contexts]
        producer.send_proposal(ids, [[11, 12]], contexts=contexts)
        with pytest.raises(RemoteDraftProtocolError, match="feedback first"):
            producer.send_proposal(ids, [[11, 12]], contexts=contexts)
        received = consumer.recv_proposal(contexts=lifecycle.proposal_contexts())
        lifecycle.receive(*received)
    lifecycle.finish(
        (RemoteDraftTerminal("a", 7, int(RequestStatus.FINISHED_ABORTED)),)
    )
    plan = RemoteDraftPlan(
        2,
        (RemoteDraftQuery("b", 7, 1, 6, 3, RemoteQueryKind.VERIFY),),
        (1,),
    )
    lifecycle.begin(plan, [[21, 22]])
    consumer.send_feedback({"execution_plan": plan})
    producer.recv_feedback()
    producer.send_proposal(
        ["b"], [[30, 31]], contexts=proposal_contexts_for_plan(plan, [[21, 22]])
    )
    lifecycle.receive(*consumer.recv_proposal(contexts=lifecycle.proposal_contexts()))
    assert producer._round_id == consumer._round_id == 3


@pytest.fixture
def runtime(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 64,
                "word_embed_proj_dim": 64,
                "ffn_dim": 128,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "max_position_embeddings": 256,
                "vocab_size": 256,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }
        )
    )
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(model),
            dtype="float16",
            max_model_len=128,
            skip_tokenizer_init=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=8,
            max_model_len=128,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=False,
        ),
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=False,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    cache = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    config.cache_config.num_gpu_blocks = 32
    register_all_kvcache_specs(config)
    init_none_hash(sha256)
    receiver = RemoteDraftPlanReceiver("tcp://127.0.0.1:*", 2)
    senders = [RemoteDraftPlanSender(receiver.endpoint, rank, 2) for rank in range(2)]
    scheduler = RemoteDraftScheduler(
        vllm_config=config,
        kv_cache_config=cache,
        block_size=16,
        structured_output_manager=StructuredOutputManager(config),
        plan_receiver=receiver,
    )
    try:
        yield scheduler, receiver, senders
    finally:
        for sender in senders:
            sender.close()
        scheduler.shutdown()


def add(scheduler, request_id, max_tokens=16, prompt_length=3, kv_transfer_params=None):
    params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=True,
        extra_args={
            "kv_transfer_params": {
                "_remote_dspark": True,
                GENERATION_KEY: 7,
                **(kv_transfer_params or {}),
            }
        },
    )
    params.update_from_generation_config({}, 2)
    request = Request(
        request_id,
        [1] * prompt_length,
        params,
        None,
        block_hasher=get_request_block_hasher(16, sha256),
    )
    scheduler.add_request(request)
    return request


def complete(scheduler, output, tokens=None, connector_output=None):
    ids = list(output.num_scheduled_tokens)
    return scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(ids)},
            sampled_token_ids=tokens if tokens is not None else [[10] for _ in ids],
            kv_connector_output=connector_output,
        ),
    )


def transferred(consumer, request):
    request.kv_transfer_params.update({"req_id": request.request_id, "uuid": 7})
    # Model only the completed external-load boundary; the allocator and all
    # subsequent scheduling/accounting are the actual upstream implementations.
    assert (
        consumer.kv_cache_manager.allocate_slots(request, request.num_prompt_tokens)
        is not None
    )
    request.num_computed_tokens = request.num_prompt_tokens - 1


def bootstrap(sequence=0, request_id="a", generation=7, position=3):
    return RemoteDraftPlan(
        sequence,
        (
            RemoteDraftQuery(
                request_id,
                generation,
                0,
                position,
                1,
                RemoteQueryKind.BOOTSTRAP,
            ),
        ),
        (1,),
    )


def await_plan(receiver):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        plan = receiver.poll()
        if plan is not None:
            return plan
    pytest.fail("readiness did not arrive before control deadline")


def publish(receiver, senders, plan):
    for sender in reversed(senders):
        sender.send(plan)
    assert await_plan(receiver) == plan


def test_rank_agreed_plan_drives_real_scheduler_and_typed_worker_metadata(runtime):
    scheduler, receiver, senders = runtime
    a, b = add(scheduler, "a"), add(scheduler, "b")
    complete(scheduler, scheduler.schedule())
    blocks = scheduler.kv_cache_manager.get_block_ids("b")
    plan = bootstrap()
    senders[1].send(plan)
    output = scheduler.schedule()
    assert not output.num_scheduled_tokens and output.remote_draft_plan is None
    complete(scheduler, output)
    senders[0].send(plan)
    assert await_plan(receiver) == plan
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"a": 1}
    assert output.remote_draft_plan == plan
    restored = pickle.loads(pickle.dumps(output))
    assert isinstance(restored, RemoteDraftSchedulerOutput)
    assert restored.remote_draft_plan == plan
    complete(scheduler, output)
    assert (a.num_computed_tokens, b.num_computed_tokens) == (4, 3)
    assert scheduler.kv_cache_manager.get_block_ids("b") == blocks
    publish(receiver, senders, bootstrap(1, "b"))
    complete(scheduler, scheduler.schedule())
    assert (a.num_computed_tokens, b.num_computed_tokens) == (4, 4)


def test_control_wait_does_not_block_target_prefill(runtime):
    scheduler, receiver, senders = runtime
    a = add(scheduler, "a")
    complete(scheduler, scheduler.schedule())
    c = add(scheduler, "c")
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"c": 3}
    assert output.remote_draft_plan is None
    complete(scheduler, output)
    assert a.num_computed_tokens == 3 and c.num_computed_tokens == 3
    publish(receiver, senders, bootstrap())
    output = scheduler.schedule()
    assert output.remote_draft_plan == bootstrap()
    complete(scheduler, output)


@pytest.mark.parametrize("k", [2, 5])
@pytest.mark.parametrize("accepted", range(6))
def test_mixed_spans_use_real_speculative_rejection_accounting(runtime, accepted, k):
    if accepted > k:
        pytest.skip("acceptance exceeds draft window")
    scheduler, receiver, senders = runtime
    a, b = add(scheduler, "a"), add(scheduler, "b")
    complete(scheduler, scheduler.schedule())
    a.spec_token_ids = list(range(11, 11 + k))
    scheduler.num_spec_tokens = k
    plan = RemoteDraftPlan(
        0,
        (
            RemoteDraftQuery("b", 7, 0, 3, 1, RemoteQueryKind.BOOTSTRAP),
            RemoteDraftQuery("a", 7, 1, 3, k + 1, RemoteQueryKind.VERIFY),
        ),
        (1, 2),
    )
    publish(receiver, senders, plan)
    output = scheduler.schedule()
    assert list(output.num_scheduled_tokens.items()) == [("b", 1), ("a", k + 1)]
    complete(scheduler, output, [[30], list(range(20, 21 + accepted))])
    assert (a.num_computed_tokens, b.num_computed_tokens) == (4 + accepted, 4)
    assert (a.num_in_flight_tokens, b.num_in_flight_tokens) == (0, 0)


@pytest.mark.parametrize("generation,position", [(8, 3), (7, 4)])
def test_invalid_query_never_charges_scheduler(runtime, generation, position):
    scheduler, receiver, senders = runtime
    a = add(scheduler, "a")
    complete(scheduler, scheduler.schedule())
    publish(receiver, senders, bootstrap(generation=generation, position=position))
    with pytest.raises(RemoteDraftProtocolError):
        scheduler.schedule()
    assert a.num_computed_tokens == 3 and a.num_in_flight_tokens == 0


def test_rank_disagreement_fails_before_scheduler_accounting(runtime):
    scheduler, receiver, senders = runtime
    a = add(scheduler, "a")
    complete(scheduler, scheduler.schedule())
    senders[0].send(bootstrap())
    senders[1].send(bootstrap(position=4))
    with pytest.raises(RemoteDraftProtocolError, match="ranks disagree"):
        await_plan(receiver)
    assert a.num_computed_tokens == 3 and a.num_in_flight_tokens == 0


def test_missing_rank_times_out_without_blocking_scheduler_poll(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(
        "vllm_torchtpu.spec_decode.remote_draft_control.time.monotonic",
        lambda: clock[0],
    )
    receiver = RemoteDraftPlanReceiver("tcp://127.0.0.1:*", 2, timeout_s=3)
    sender = RemoteDraftPlanSender(receiver.endpoint, 0, 2)
    try:
        # No announced batch means no deadline: ordinary prefill may run idle.
        clock[0] = 200.0
        assert receiver.poll() is None
        sender.send(bootstrap())
        deadline = time.perf_counter() + 2
        while receiver._first_rank_at is None and time.perf_counter() < deadline:
            assert receiver.poll() is None
        assert receiver._first_rank_at == 200.0
        clock[0] = 202.9
        assert receiver.poll() is None
        clock[0] = 203.0
        with pytest.raises(RemoteDraftProtocolError, match=r"worker ranks \[1\]"):
            receiver.poll()
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize("request_id,status", [(17, 4), ("a", True), ("a", "4")])
def test_terminal_rejects_noncanonical_identity_or_status(request_id, status):
    with pytest.raises(RemoteDraftProtocolError):
        RemoteDraftTerminal(request_id, 7, status)


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_live_consumer_lifecycle_mixed_rounds_and_partial_terminal(accepted):
    lifecycle = RemoteDraftConsumerLifecycle(2)
    lifecycle.begin(bootstrap(position=3), [[20]])
    lifecycle.receive(["a"], [[11, 12]])
    mixed = RemoteDraftPlan(
        1,
        (
            RemoteDraftQuery("b", 7, 0, 5, 1, RemoteQueryKind.BOOTSTRAP),
            RemoteDraftQuery("a", 7, 1, 4, 3, RemoteQueryKind.VERIFY),
        ),
        (2,),
    )
    lifecycle.begin(mixed, [[30], list(range(21, 22 + accepted))])
    # Transport proposal order need not match D's post-RPA order.
    lifecycle.receive(["a", "b"], [[31, 32], [41, 42]])
    lifecycle.finish(
        (RemoteDraftTerminal("a", 7, int(RequestStatus.FINISHED_ABORTED)),)
    )
    plan = RemoteDraftPlan(
        2,
        (RemoteDraftQuery("b", 7, 1, 6, 3, RemoteQueryKind.VERIFY),),
        (1,),
    )
    lifecycle.begin(plan, [[50]])
    lifecycle.receive(["b"], [[51, 52]])
    assert lifecycle._pipeline.inflight_proposals == 1
    assert lifecycle._work["b"].round_id == 1


@pytest.mark.parametrize(
    "generation,round_id,position", [(8, 1, 4), (7, 2, 4), (7, 1, 5)]
)
def test_consumer_lifecycle_rejects_stale_verify(generation, round_id, position):
    lifecycle = RemoteDraftConsumerLifecycle(1)
    lifecycle.begin(bootstrap(), [[20]])
    lifecycle.receive(["a"], [[11, 12]])
    plan = RemoteDraftPlan(
        1,
        (
            RemoteDraftQuery(
                "a", generation, round_id, position, 3, RemoteQueryKind.VERIFY
            ),
        ),
        (1,),
    )
    with pytest.raises(RemoteDraftProtocolError, match="round/generation/span"):
        lifecycle.begin(plan, [[30]])


def test_consumer_lifecycle_rejects_duplicate_or_foreign_pending_proposal():
    lifecycle = RemoteDraftConsumerLifecycle(1)
    lifecycle.begin(bootstrap(), [[20]])
    with pytest.raises(RemoteDraftProtocolError, match="pending"):
        lifecycle.begin(bootstrap(), [[20]])
    with pytest.raises(RemoteDraftProtocolError, match="pending D execution"):
        lifecycle.receive(["foreign"], [[11, 12]])
    lifecycle.receive(["a"], [[11, 12]])
    with pytest.raises(RemoteDraftProtocolError, match="pending D execution"):
        lifecycle.receive(["a"], [[11, 12]])


def test_lifecycle_reclaims_terminal_state_and_reused_source_rejects_old_wire(
    transports,
):
    producer, consumer = transports
    lifecycle = RemoteDraftConsumerLifecycle(1)
    old_context = None
    for generation in range(1, 17):
        plan = bootstrap(generation=generation)
        lifecycle.begin(plan, [[20]])
        consumer.send_feedback({"execution_plan": plan})
        producer.recv_feedback()
        contexts = proposal_contexts_for_plan(plan, [[20]])
        producer.send_proposal(["a"], [[11, 12]], contexts=contexts)
        lifecycle.receive(
            *consumer.recv_proposal(contexts=lifecycle.proposal_contexts())
        )
        lifecycle.finish(
            (
                RemoteDraftTerminal(
                    "a", generation, int(RequestStatus.FINISHED_LENGTH_CAPPED)
                ),
            )
        )
        assert lifecycle._work == lifecycle._generations == {}
        assert lifecycle._pipeline._requests == {}
        assert lifecycle._pipeline.inflight_proposals == 0
        old_context = contexts
    plan = bootstrap(generation=17)
    lifecycle.begin(plan, [[20]])
    consumer.send_feedback({"execution_plan": plan})
    producer.recv_feedback()
    producer.send_proposal(["a"], [[99]], contexts=old_context)
    with pytest.raises(RemoteDraftProtocolError, match="generation/round/frontier"):
        consumer.recv_proposal(contexts=lifecycle.proposal_contexts())


def test_typed_layout_and_decode_reject_wrong_position_and_unknown_fields():
    plan = bootstrap()
    assert decode_plan(encode_plan(0, plan)) == (0, plan)
    plan.validate_layout(["a"], [3], [1], [1])
    with pytest.raises(RemoteDraftProtocolError, match="layout differs"):
        plan.validate_layout(["a"], [4], [1], [1])
    raw = json.loads(encode_plan(0, plan))
    raw["plan"]["extra"] = True
    with pytest.raises(RemoteDraftProtocolError, match="invalid remote Draft"):
        decode_plan(json.dumps(raw).encode())


@pytest.mark.parametrize(
    "status",
    [
        RequestStatus.FINISHED_ABORTED,
        RequestStatus.FINISHED_STOPPED,
        RequestStatus.FINISHED_LENGTH_CAPPED,
    ],
)
def test_authoritative_d_terminal_releases_only_matching_p_request(runtime, status):
    scheduler, receiver, senders = runtime
    a, b = add(scheduler, "a"), add(scheduler, "b")
    complete(scheduler, scheduler.schedule())
    consumer = RemoteDraftConsumerScheduler(
        vllm_config=scheduler.vllm_config,
        kv_cache_config=scheduler.kv_cache_config,
        structured_output_manager=scheduler.structured_output_manager,
        block_size=16,
    )
    try:
        d_a = add(consumer, "d-a")
        transferred(consumer, d_a)
        d_a.kv_transfer_params.update({"req_id": "a", "uuid": 7})
        complete(consumer, consumer.schedule())
        # This is the actual scheduler stop/abort path, not a sample-time guess.
        consumer.finish_requests(["d-a"], status)
        output = consumer.schedule()
        assert output.remote_draft_terminals == (
            RemoteDraftTerminal("a", 7, int(status)),
        )
        assert not consumer.schedule().remote_draft_terminals
        b_blocks = scheduler.kv_cache_manager.get_block_ids("b")
        plan = RemoteDraftPlan(0, (), (), output.remote_draft_terminals)
        assert decode_plan(encode_plan(0, plan)) == (0, plan)
        publish(receiver, senders, plan)
        output = scheduler.schedule()
        assert "a" in output.finished_req_ids and output.remote_draft_plan is None
        assert a.status == status and "a" not in scheduler.requests
        assert b.num_computed_tokens == 3 and b.num_in_flight_tokens == 0
        assert scheduler.kv_cache_manager.get_block_ids("b") == b_blocks
        client_outputs = complete(scheduler, output)
        terminal_output = client_outputs[a.client_index].outputs
        assert len(terminal_output) == 1
        assert terminal_output[0].request_id == "a"
        assert terminal_output[0].finish_reason == a.get_finished_reason()
        assert terminal_output[0].new_token_ids == []
        assert not complete(scheduler, scheduler.schedule())
        publish(receiver, senders, bootstrap(1, "b"))
        complete(scheduler, scheduler.schedule())
        assert b.num_computed_tokens == 4
    finally:
        consumer.shutdown()


def test_terminal_for_old_generation_does_not_finish_reused_request(runtime):
    scheduler, receiver, senders = runtime
    a = add(scheduler, "a")
    complete(scheduler, scheduler.schedule())
    plan = RemoteDraftPlan(
        0,
        (),
        (),
        (
            RemoteDraftTerminal(
                "a",
                6,
                int(RequestStatus.FINISHED_ABORTED),
            ),
        ),
    )
    publish(receiver, senders, plan)
    with pytest.raises(RemoteDraftProtocolError, match="stale request generation"):
        scheduler.schedule()
    assert "a" in scheduler.requests and not a.is_finished()


def test_sampling_length_stop_is_published_by_actual_d_scheduler(runtime):
    scheduler, _, _ = runtime
    consumer = RemoteDraftConsumerScheduler(
        vllm_config=scheduler.vllm_config,
        kv_cache_config=scheduler.kv_cache_config,
        structured_output_manager=scheduler.structured_output_manager,
        block_size=16,
    )
    try:
        request = add(consumer, "d-a", max_tokens=1)
        transferred(consumer, request)
        request.kv_transfer_params.update({"req_id": "a", "uuid": 7})
        complete(consumer, consumer.schedule())
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
        output = consumer.schedule()
        assert output.remote_draft_terminals == (
            RemoteDraftTerminal("a", 7, int(RequestStatus.FINISHED_LENGTH_CAPPED)),
        )
    finally:
        consumer.shutdown()


@pytest.mark.parametrize("reason", ["length", "eos", "stop_token"])
def test_producer_keeps_remote_kv_until_d_terminal_despite_local_stop(runtime, reason):
    scheduler, receiver, senders = runtime
    request = add(scheduler, "a", max_tokens=1 if reason == "length" else 16)
    params = request.sampling_params
    if reason == "eos":
        params.ignore_eos = False
        params.update_from_generation_config({}, 2)
    elif reason == "stop_token":
        params.stop_token_ids = [2]
    initial = scheduler.schedule()
    complete(scheduler, initial, [[2]])
    blocks = scheduler.kv_cache_manager.get_block_ids("a")
    assert request.status == RequestStatus.RUNNING
    assert "a" in scheduler.requests
    publish(receiver, senders, bootstrap())
    draft = scheduler.schedule()
    complete(scheduler, draft, [[2]])
    assert request.status == RequestStatus.RUNNING
    assert list(request.output_token_ids) == [2, 2]
    assert request.max_tokens == (1 if reason == "length" else 16)
    assert scheduler.kv_cache_manager.get_block_ids("a") == blocks
    status = (
        RequestStatus.FINISHED_LENGTH_CAPPED
        if reason == "length"
        else RequestStatus.FINISHED_STOPPED
    )
    publish(
        receiver,
        senders,
        RemoteDraftPlan(1, (), (), (RemoteDraftTerminal("a", 7, int(status)),)),
    )
    finished = scheduler.schedule()
    assert finished.finished_req_ids == {"a"}
    assert not finished.num_scheduled_tokens
    assert request.status == status


@pytest.mark.parametrize("selection", ["single", "iterator", "all"])
def test_direct_p_abort_rejected_before_request_or_kv_mutation(runtime, selection):
    scheduler, receiver, senders = runtime
    a, b = add(scheduler, "a"), add(scheduler, "b")
    complete(scheduler, scheduler.schedule())
    blocks = {
        name: scheduler.kv_cache_manager.get_block_ids(name) for name in ("a", "b")
    }
    counters = [(r.num_computed_tokens, r.num_in_flight_tokens) for r in (a, b)]
    ids = (
        "a"
        if selection == "single"
        else iter(["b", "a"])
        if selection == "iterator"
        else None
    )
    with pytest.raises(NotImplementedError, match="cancel remote requests through D"):
        scheduler.finish_requests(ids, RequestStatus.FINISHED_ABORTED)
    assert scheduler.running == [a, b]
    assert all(r.status == RequestStatus.RUNNING for r in (a, b))
    assert [(r.num_computed_tokens, r.num_in_flight_tokens) for r in (a, b)] == counters
    for name in ("a", "b"):
        assert scheduler.kv_cache_manager.get_block_ids(name) == blocks[name]
    # The authoritative D terminal remains allowed after rejecting direct P abort.
    publish(
        receiver,
        senders,
        RemoteDraftPlan(
            0,
            (),
            (),
            (RemoteDraftTerminal("a", 7, int(RequestStatus.FINISHED_ABORTED)),),
        ),
    )
    output = scheduler.schedule()
    assert output.finished_req_ids == {"a"}
    assert b.status == RequestStatus.RUNNING
    assert scheduler.kv_cache_manager.get_block_ids("b") == blocks["b"]


def _raiden_consumer(runtime, monkeypatch, inline, failure_policy="fail"):
    from vllm_torchtpu.distributed.kv_transfer import tpu_connector

    producer, _, _ = runtime
    config = copy.copy(producer.vllm_config)
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="TPURaidenConnector",
        kv_connector_module_path="vllm_torchtpu.distributed.kv_transfer.tpu_connector",
        kv_role="kv_consumer",
        kv_load_failure_policy=failure_policy,
    )
    monkeypatch.setattr(tpu_connector.tpu_envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(tpu_connector.dist_utils, "get_kv_ips", lambda: "127.0.0.1")
    monkeypatch.setattr(tpu_connector.dist_utils, "get_kv_ports", lambda: 9100)
    monkeypatch.setattr(tpu_connector.dist_utils, "get_side_channel_port", lambda: 9600)
    monkeypatch.setattr(
        tpu_connector.dist_utils, "get_raiden_inline_load", lambda: inline
    )
    return RemoteDraftConsumerScheduler(
        vllm_config=config,
        kv_cache_config=producer.kv_cache_config,
        structured_output_manager=producer.structured_output_manager,
        block_size=16,
    )


def test_real_raiden_consumer_rejects_recompute_failure_policy(runtime, monkeypatch):
    with pytest.raises(NotImplementedError, match="kv_load_failure_policy='fail'"):
        _raiden_consumer(runtime, monkeypatch, False, "recompute")


def _remote_transfer_params(source_id="a"):
    return {
        "do_remote_prefill": True,
        "req_id": source_id,
        "uuid": 7,
        "num_tokens": 3,
        "src_controller_address": "127.0.0.1:27000",
        "src_job_name": "prefill",
        "src_engine_id": "producer-engine",
        "src_data_replica_idx": 0,
        "src_parallelism": 1,
        "dst_tp_size": 1,
        "dst_dcp_size": 1,
    }


def test_real_raiden_partial_admission_preserves_pending_b_and_executes_ready_a(
    runtime, monkeypatch
):
    consumer = _raiden_consumer(runtime, monkeypatch, False)
    try:
        consumer.num_spec_tokens = 2
        requests = [
            add(
                consumer,
                f"d-{source}",
                prompt_length=4,
                kv_transfer_params=_remote_transfer_params(source),
            )
            for source in ("a", "b")
        ]
        loading = consumer.schedule()
        assert not loading.num_scheduled_tokens
        assert set(loading.kv_connector_metadata.reqs_to_load) == {"d-a", "d-b"}
        b_blocks = consumer.kv_cache_manager.get_block_ids("d-b")
        complete(
            consumer,
            loading,
            connector_output=KVConnectorOutput(finished_recving={"d-a"}),
        )
        bootstrap_a = consumer.schedule()
        assert bootstrap_a.num_scheduled_tokens == {"d-a": 1}
        assert bootstrap_a.scheduled_new_reqs[0].num_computed_tokens == 3
        assert requests[1].status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert requests[1].num_computed_tokens == 3
        assert consumer.kv_cache_manager.get_block_ids("d-b") == b_blocks
        complete(
            consumer,
            bootstrap_a,
            connector_output=KVConnectorOutput(finished_recving={"d-b"}),
        )
        requests[0].spec_token_ids = [11, 12]
        mixed = consumer.schedule()
        assert mixed.num_scheduled_tokens == {"d-a": 3, "d-b": 1}
        assert mixed.scheduled_spec_decode_tokens == {"d-a": [11, 12]}
        assert mixed.scheduled_new_reqs[0].num_computed_tokens == 3
        assert mixed.scheduled_cached_reqs.num_computed_tokens == [4]
        assert consumer.kv_cache_manager.get_block_ids("d-b") == b_blocks
    finally:
        consumer.shutdown()


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_real_raiden_admission_waits_for_kv_or_fails_without_target_replay(
    runtime, monkeypatch, inline, failed
):
    consumer = _raiden_consumer(runtime, monkeypatch, inline)
    try:
        consumer.num_spec_tokens = 2
        request = add(
            consumer,
            "d-a",
            prompt_length=4,
            kv_transfer_params=_remote_transfer_params(),
        )
        output = consumer.schedule()
        load = output.kv_connector_metadata.reqs_to_load["d-a"]
        assert (load.source_req_id, load.uuid, load.num_tokens) == ("a", 7, 3)
        assert request.num_computed_tokens == 3 + int(inline)
        if not inline:
            assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            assert not output.num_scheduled_tokens
            complete(consumer, output)
            output = consumer.schedule()
            assert not output.num_scheduled_tokens
            assert request.num_computed_tokens == 3
        else:
            assert output.num_scheduled_tokens == {"d-a": 1}
            assert not output.scheduled_spec_decode_tokens
            assert output.scheduled_new_reqs[0].num_computed_tokens == 3
        ids = list(output.num_scheduled_tokens)
        blocks = consumer.kv_cache_manager.get_block_ids("d-a")[0]
        receipt = KVConnectorOutput(
            finished_recving=set() if inline or failed else {"d-a"},
            failed_recving={"d-a"} if failed and not inline else set(),
            invalid_block_ids={blocks[0]} if failed and inline else set(),
        )
        # Explicit worker receipt/output; actual connector admission and
        # scheduler load/finish accounting execute unchanged, without DMA.
        consumer.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={req_id: index for index, req_id in enumerate(ids)},
                sampled_token_ids=[[20] for _ in ids],
                kv_connector_output=receipt,
            ),
        )
        following = consumer.schedule()
        if failed:
            assert request.status == RequestStatus.FINISHED_ERROR
            assert not following.num_scheduled_tokens
            assert following.remote_draft_terminals == (
                RemoteDraftTerminal("a", 7, int(RequestStatus.FINISHED_ERROR)),
            )
        elif not inline:
            assert following.num_scheduled_tokens == {"d-a": 1}
            assert request.num_computed_tokens == 4
            assert following.scheduled_new_reqs[0].num_computed_tokens == 3
            assert not following.scheduled_spec_decode_tokens
    finally:
        consumer.shutdown()


@pytest.mark.parametrize(
    "scheduler_cls", [RemoteDraftScheduler, RemoteDraftConsumerScheduler]
)
@pytest.mark.parametrize(
    "field", ["prefill_context_parallel_size", "decode_context_parallel_size", "nnodes"]
)
def test_planned_scheduler_rejects_context_parallel_or_multinode(
    runtime, scheduler_cls, field
):
    producer, _, _ = runtime
    config = copy.copy(producer.vllm_config)
    config.parallel_config = copy.copy(config.parallel_config)
    setattr(config.parallel_config, field, 2)
    with pytest.raises(NotImplementedError, match="sync TP-only"):
        scheduler_cls(
            vllm_config=config,
            kv_cache_config=producer.kv_cache_config,
            structured_output_manager=producer.structured_output_manager,
            block_size=16,
        )


@pytest.mark.parametrize("role", ["producer", "consumer"])
def test_kv_exhaustion_cannot_reset_remote_request_for_target_replay(runtime, role):
    producer, receiver, senders = runtime
    scheduler = producer
    if role == "consumer":
        scheduler = RemoteDraftConsumerScheduler(
            vllm_config=producer.vllm_config,
            kv_cache_config=producer.kv_cache_config,
            structured_output_manager=producer.structured_output_manager,
            block_size=16,
        )
        scheduler.num_spec_tokens = 2
    retained = []
    pool = scheduler.kv_cache_manager.block_pool
    try:
        request = add(scheduler, "a", prompt_length=16)
        if role == "producer":
            complete(scheduler, scheduler.schedule(), [[]])
            complete(scheduler, scheduler.schedule())
            publish(receiver, senders, bootstrap(position=16))
        else:
            transferred(scheduler, request)
            complete(scheduler, scheduler.schedule())
            request.spec_token_ids = [11, 12]
        position = request.num_computed_tokens
        blocks = scheduler.kv_cache_manager.get_block_ids("a")
        retained = pool.get_new_blocks(pool.get_num_free_blocks())
        with pytest.raises(NotImplementedError, match="must not replay Target"):
            scheduler.schedule()
        assert request.num_computed_tokens == position
        assert request.num_in_flight_tokens == request.num_preemptions == 0
        assert request.status == RequestStatus.RUNNING
        assert scheduler.kv_cache_manager.get_block_ids("a") == blocks
    finally:
        pool.free_blocks(retained)
        if role == "consumer":
            scheduler.shutdown()


def test_real_engine_flushes_zero_token_terminal_without_waiting_for_another_proposal(
    runtime,
):
    producer, receiver, senders = runtime
    add(producer, "a")
    complete(producer, producer.schedule())
    consumer = RemoteDraftConsumerScheduler(
        vllm_config=producer.vllm_config,
        kv_cache_config=producer.kv_cache_config,
        structured_output_manager=producer.structured_output_manager,
        block_size=16,
    )
    events = []

    class CPUExecutor:
        # Explicit model/worker stand-in; EngineCore and scheduler run unchanged.
        def execute_model(self, output, non_block):
            assert non_block
            events.append(("execute", output.total_num_scheduled_tokens))
            if output.remote_draft_terminals:
                publish(
                    receiver,
                    senders,
                    RemoteDraftPlan(0, (), (), output.remote_draft_terminals),
                )
            ids = list(output.num_scheduled_tokens)
            future = Future()
            future.set_result(
                ModelRunnerOutput(
                    req_ids=ids,
                    req_id_to_index={req_id: i for i, req_id in enumerate(ids)},
                    sampled_token_ids=[[20] for _ in ids],
                )
            )
            return future

        def take_draft_token_ids(self):
            events.append(("take_draft", 1))
            return None

    core = object.__new__(EngineCore)
    core.scheduler, core.model_executor = consumer, CPUExecutor()
    core.vllm_config, core.log_stats = producer.vllm_config, False
    core.check_for_draft_tokens, core.async_scheduling = True, False
    core.aborts_queue, core._weight_version = queue.Queue(), None
    try:
        request = add(consumer, "d-a", max_tokens=1)
        transferred(consumer, request)
        request.kv_transfer_params["req_id"] = "a"
        _, executed = core.step()
        core.post_step(executed)
        assert request.is_finished() and consumer.has_requests()
        _, executed = core.step()
        core.post_step(executed)
        assert events == [("execute", 1), ("take_draft", 1), ("execute", 0)]
        assert not consumer.has_requests()
        p_output = producer.schedule()
        assert "a" in p_output.finished_req_ids
        assert p_output.total_num_scheduled_tokens == 0
    finally:
        consumer.shutdown()


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_consumer_bootstrap_padding_removed_before_accounting(runtime, accepted):
    scheduler, _, _ = runtime
    consumer = RemoteDraftConsumerScheduler(
        vllm_config=scheduler.vllm_config,
        kv_cache_config=scheduler.kv_cache_config,
        structured_output_manager=scheduler.structured_output_manager,
        block_size=16,
    )
    try:
        consumer.num_spec_tokens = 2
        a = add(consumer, "d-a")
        transferred(consumer, a)
        output = consumer.schedule()
        assert output.num_scheduled_tokens == {"d-a": 1}
        assert not output.scheduled_spec_decode_tokens
        assert (a.num_computed_tokens, a.num_in_flight_tokens) == (3, 1)
        complete(consumer, output)
        assert (a.num_computed_tokens, a.num_in_flight_tokens) == (3, 0)
        a.spec_token_ids = [11, 12]
        b = add(consumer, "d-b")
        transferred(consumer, b)
        output = consumer.schedule()
        assert output.num_scheduled_tokens == {"d-a": 3, "d-b": 1}
        assert output.scheduled_spec_decode_tokens == {"d-a": [11, 12]}
        assert (b.num_computed_tokens, b.num_in_flight_tokens) == (3, 1)
        complete(consumer, output, [list(range(20, 21 + accepted)), [30]])
        assert (a.num_computed_tokens, b.num_computed_tokens) == (4 + accepted, 3)
        assert (a.num_in_flight_tokens, b.num_in_flight_tokens) == (0, 0)
    finally:
        consumer.shutdown()


def test_consumer_bootstrap_from_published_raiden_handoff(runtime, tmp_path):
    scheduler, _, _ = runtime
    consumer = RemoteDraftConsumerScheduler(
        vllm_config=scheduler.vllm_config,
        kv_cache_config=scheduler.kv_cache_config,
        structured_output_manager=scheduler.structured_output_manager,
        block_size=16,
    )
    try:
        consumer.num_spec_tokens = 1
        params = {"req_id": "p-a", "uuid": 7, "num_tokens": 7}
        publish_remote_dspark_handoff("p-a", params, tmp_path, ready=False)
        complete_remote_dspark_handoff("p-a", tmp_path)
        # Match the proxy's JSON handoff; do not use add(), which marks params.
        sampling = SamplingParams(
            max_tokens=8,
            ignore_eos=True,
            extra_args={
                "kv_transfer_params": take_remote_dspark_handoff("p-a", tmp_path)
            },
        )
        sampling.update_from_generation_config({}, 2)
        request = Request(
            "d-a",
            [1] * 8,
            sampling,
            None,
            block_hasher=get_request_block_hasher(16, sha256),
        )
        consumer.add_request(request)
        transferred(consumer, request)
        output = consumer.schedule()
        assert output.num_scheduled_tokens == {"d-a": 1}
        assert not output.scheduled_spec_decode_tokens
        assert (request.num_computed_tokens, request.num_in_flight_tokens) == (8, 1)
        assert params == {"req_id": "p-a", "uuid": 7, "num_tokens": 7}
    finally:
        consumer.shutdown()


def test_consumer_refuses_missing_remote_prefix_before_accounting(runtime):
    scheduler, _, _ = runtime
    consumer = RemoteDraftConsumerScheduler(
        vllm_config=scheduler.vllm_config,
        kv_cache_config=scheduler.kv_cache_config,
        structured_output_manager=scheduler.structured_output_manager,
        block_size=16,
    )
    try:
        request = add(consumer, "d-a")
        with pytest.raises(RemoteDraftProtocolError, match="refusing Target prefill"):
            consumer.schedule()
        assert request.num_computed_tokens == request.num_in_flight_tokens == 0
    finally:
        consumer.shutdown()


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_p_and_d_positions_align_across_mixed_bootstrap_verify(runtime, accepted):
    producer, receiver, senders = runtime
    consumer = RemoteDraftConsumerScheduler(
        vllm_config=producer.vllm_config,
        kv_cache_config=producer.kv_cache_config,
        structured_output_manager=producer.structured_output_manager,
        block_size=16,
    )
    try:
        consumer.num_spec_tokens = producer.num_spec_tokens = 2
        p_a, p_b = add(producer, "a"), add(producer, "b")
        complete(producer, producer.schedule())
        # The actual proxy sends P the prompt without its final token. D's
        # bootstrap query position therefore equals P's completed prefix.
        d_a = add(consumer, "d-a", prompt_length=4)
        transferred(consumer, d_a)
        d_output = consumer.schedule()
        assert d_output.num_scheduled_tokens == {"d-a": 1}
        lifecycle = RemoteDraftConsumerLifecycle(2)
        lifecycle.begin(bootstrap(), [[20]])
        publish(receiver, senders, bootstrap())
        p_output = producer.schedule()
        complete(producer, p_output, [[20]])
        complete(consumer, d_output, [[20]])
        assert p_a.num_computed_tokens == d_a.num_computed_tokens == 4
        lifecycle.receive(["a"], [[11, 12]])

        p_a.spec_token_ids = d_a.spec_token_ids = [11, 12]
        d_b = add(consumer, "d-b", prompt_length=4)
        transferred(consumer, d_b)
        d_output = consumer.schedule()
        assert d_output.num_scheduled_tokens == {"d-a": 3, "d-b": 1}
        # Worker RPA places B's one-token query before A's verify window.
        plan = RemoteDraftPlan(
            1,
            (
                RemoteDraftQuery("b", 7, 0, 3, 1, RemoteQueryKind.BOOTSTRAP),
                RemoteDraftQuery("a", 7, 1, 4, 3, RemoteQueryKind.VERIFY),
            ),
            (2,),
        )
        accepted_tokens = list(range(30, 31 + accepted))
        lifecycle.begin(plan, [[40], accepted_tokens])
        publish(receiver, senders, plan)
        p_output = producer.schedule()
        assert p_output.remote_draft_plan == plan
        complete(producer, p_output, [[40], accepted_tokens])
        complete(consumer, d_output, [accepted_tokens, [40]])
        assert p_a.num_computed_tokens == d_a.num_computed_tokens == 5 + accepted
        assert p_b.num_computed_tokens == d_b.num_computed_tokens == 4
        lifecycle.receive(["a", "b"], [[50, 51], [60, 61]])
        assert lifecycle._work["a"].kv_version == d_a.num_computed_tokens
        assert lifecycle._work["b"].kv_version == d_b.num_computed_tokens
        assert all(
            request.num_in_flight_tokens == 0 for request in (p_a, p_b, d_a, d_b)
        )
    finally:
        consumer.shutdown()
