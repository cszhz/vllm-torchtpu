# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the last-stage, forward-only MTP diagnostic."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.qwen3_5_mtp_prefill import (
    Qwen3_5MoeMTPPrefill,
    Qwen3NextMTPPrefill,
)
from vllm_torchtpu.spec_decode.prefill_only import (
    validate_prefill_only_config,
    validate_prefill_span,
)


def config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            tensor_parallel_size=1,
            prefill_context_parallel_size=8,
            decode_context_parallel_size=1,
            data_parallel_size=1,
            nnodes=2,
        ),
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            enable_chunked_prefill=False,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        kv_transfer_config=None,
    )


def test_two_host_pp2_pcp8_admitted():
    validate_prefill_only_config(config())


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 2),
        ("data_parallel_size", 2),
        ("decode_context_parallel_size", 2),
    ],
)
def test_unsupported_parallelism_rejected(field, value):
    c = config()
    setattr(c.parallel_config, field, value)
    with pytest.raises(NotImplementedError, match="TP=DCP=DP=1"):
        validate_prefill_only_config(c)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("scheduler_config", "async_scheduling", True),
    ],
)
def test_rejects_unimplemented_lifecycles(section, field, value):
    c = config()
    setattr(getattr(c, section), field, value)
    with pytest.raises(NotImplementedError):
        validate_prefill_only_config(c)


@pytest.mark.parametrize(
    "computed,scheduled,prompt",
    [
        (-1, 3, 4),
        (1, 0, 4),
        (4, 1, 4),
        (0, 5, 4),
    ],
)
def test_rejects_invalid_span_or_decode(computed, scheduled, prompt):
    with pytest.raises(NotImplementedError, match="span within the prompt"):
        validate_prefill_span(computed, scheduled, prompt)


@pytest.mark.parametrize("computed,scheduled", [(0, 4), (0, 2), (2, 2), (1, 2)])
def test_prefill_spans(computed, scheduled):
    validate_prefill_span(computed, scheduled, 4)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_chunking_and_prefix_caching_are_orthogonal(chunked, cached):
    c = config()
    c.scheduler_config.enable_chunked_prefill = chunked
    c.cache_config.enable_prefix_caching = cached
    validate_prefill_only_config(c)


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_kv_transfer_admits_only_pipeline_producer(monkeypatch, role):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(envs, "TPU_RAIDEN_QWEN35_ADMISSION", True)
    monkeypatch.setattr(envs, "TPU_RAIDEN_POOL_TAGS_PER_LAYER", True)
    c = config()
    c.kv_transfer_config = SimpleNamespace(
        kv_connector="TPURaidenConnector", kv_role=role
    )
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setenv("TPU_RAIDEN_QWEN35_ADMISSION", "1")
    monkeypatch.setenv("TPU_RAIDEN_POOL_TAGS_PER_LAYER", "1")
    if role == "kv_producer":
        validate_prefill_only_config(c)
    else:
        with pytest.raises(NotImplementedError, match="pipeline Raiden"):
            validate_prefill_only_config(c)


def test_handoff_uses_known_prompt_token_without_mutating_sample():
    from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
        SendMeta,
        TPUConnectorMetadata,
    )
    from vllm_torchtpu.spec_decode.prefill_only import handoff_seed_tokens

    meta = TPUConnectorMetadata(
        reqs_to_send={
            "finishing": SendMeta(1, [9], 30.0, next_prompt_token_id=0),
        }
    )
    sampled = torch.tensor([123, 456])
    result = handoff_seed_tokens(sampled, ["partial", "finishing"], meta)
    assert result.tolist() == [123, 0]
    assert sampled.tolist() == [123, 456]
    assert handoff_seed_tokens(sampled, [], None) is sampled


class Layer(torch.nn.Module):
    use_attn_reduce_scatter_for_moe = False

    def forward(self, positions, hidden_states, residual):
        assert residual is None
        self.positions = positions
        return hidden_states * 2, hidden_states


class Norm(torch.nn.Module):
    def forward(self, x, residual):
        return x + residual, None


def make_head(head_class=Qwen3_5MoeMTPPrefill):
    head = head_class.__new__(head_class)
    torch.nn.Module.__init__(head)
    model = torch.nn.Module()
    model.embed_tokens = torch.nn.Embedding(7, 3)
    model.embed_input_ids = model.embed_tokens
    model.pre_fc_norm_embedding = torch.nn.Identity()
    model.pre_fc_norm_hidden = torch.nn.Identity()
    model.fc = torch.nn.Linear(6, 3, bias=False)
    model.layers = torch.nn.ModuleList([Layer()])
    model.num_mtp_layers = 1
    model.norm = Norm()
    head.model = model
    return head


@pytest.mark.parametrize("head_class", [Qwen3_5MoeMTPPrefill, Qwen3NextMTPPrefill])
def test_whole_head_projects_every_prompt_row_without_first_rank_branch(head_class):
    from vllm.model_executor.models.interfaces import supports_pp

    assert supports_pp(head_class)
    head = make_head(head_class)
    ids = torch.tensor([1, 2, 3, 4])
    positions = torch.arange(4)
    hidden = torch.randn(4, 3)
    expected = head.model.fc(torch.cat([head.model.embed_tokens(ids), hidden], -1)) * 3
    result = head.forward(ids, positions, hidden)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(head.model.layers[0].positions, positions)
    assert result.shape == (4, 3)


def test_prefill_head_rejects_loop_step():
    head = make_head()
    with pytest.raises(NotImplementedError, match="step-0"):
        Qwen3_5MoeMTPPrefill.forward(
            head,
            torch.tensor([1]),
            torch.tensor([0]),
            torch.zeros(1, 3),
            spec_step_idx=1,
        )


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_prefill_executes_head_without_producing_draft_tokens(
    partial, capture, monkeypatch
):
    import numpy as np

    from vllm_torchtpu.layers.core.sequence_layout import (
        PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
        AllSequenceLayoutPlanner,
        SequenceLayoutDescriptor,
        SequenceLayoutKind,
        SequenceLayoutPlan,
    )
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    head = make_head()
    proposer = Eagle3Proposer.__new__(Eagle3Proposer)
    proposer.draft_model = head
    proposer.capture_prefill_snapshots = capture
    hidden = torch.randn(4, 3)
    ids = torch.tensor([2, 3, 4, 5])
    positions = torch.arange(4)
    chunk = SimpleNamespace(
        hidden_states=hidden,
        query_start_loc_np=np.array([0, 4]),
        sequence_layout_plan=AllSequenceLayoutPlanner().prepare_dummy(
            num_tokens=4,
            num_reqs=1,
            kv_cache_initialized=True,
        ),
    )
    if partial:
        # This worker owns logical rows 0, 2, 4; its fourth row is padding.
        chunk.query_start_loc_np = np.array([0, 5])
        chunk.sequence_layout_plan = SequenceLayoutPlan(
            descriptor=SequenceLayoutDescriptor(
                kind=SequenceLayoutKind.PARTIAL,
                protocol=PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
            ),
            token_slice=slice(0, 4),
            global_num_tokens=5,
            global_padded_num_tokens=8,
            local_num_tokens=3,
            local_padded_num_tokens=4,
            _packed_to_request_major_token_indices=np.array(
                [0, 2, 4, -1, 1, 3, -1, -1]
            ),
            _request_major_to_packed_token_indices=np.array([0, 4, 1, 5, 2]),
        )
    proposer.draft_chunks = [chunk]
    proposer._prepare_draft_inputs = lambda *a, **kw: (
        ids,
        positions,
        torch.tensor([3]),
        np.array([0]),
    )

    def forward(chunk, input_ids, positions, target_hidden_states, **kwargs):
        assert kwargs["step_idx"] == 0
        assert kwargs["seq_lens_delta"] == 0
        output = Qwen3_5MoeMTPPrefill.forward(
            head,
            input_ids,
            positions,
            target_hidden_states,
        )
        return output, output

    proposer._forward_draft = forward
    if not capture:

        def unexpected_copy(*args, **kwargs):
            raise AssertionError("Serving MTP prefill must not capture CPU tensors")

        monkeypatch.setattr(torch.Tensor, "cpu", unexpected_copy)
    proposer.prefill_only(SimpleNamespace(), torch.tensor([5]))
    if not capture:
        assert proposer.prefill_snapshots == []
        return
    assert len(proposer.prefill_snapshots) == 1
    snapshot = proposer.prefill_snapshots[0]
    valid_rows = 3 if partial else 4
    logical = torch.tensor([0, 2, 4]) if partial else positions
    torch.testing.assert_close(snapshot["input_ids"], ids[:valid_rows])
    torch.testing.assert_close(snapshot["logical_token_indices"], logical)
    torch.testing.assert_close(snapshot["target_hidden"], hidden[:valid_rows])
    expected = Qwen3_5MoeMTPPrefill.forward(head, ids, positions, hidden)
    torch.testing.assert_close(snapshot["mtp_hidden"], expected[:valid_rows])


@pytest.mark.parametrize("last", [False, True])
def test_only_final_stage_initializes_mtp(monkeypatch, last):
    from vllm_torchtpu.runner import tpu_runner

    monkeypatch.setenv("TPU_MTP_PREFILL_ONLY", "1")
    monkeypatch.setattr(tpu_runner, "pp_rank_flags", lambda _: (not last, last))
    created = []

    def make_proposer(runner, config):
        created.append(runner)
        return "draft"

    monkeypatch.setattr(tpu_runner, "Eagle3Proposer", make_proposer)
    runner = SimpleNamespace(parallel_config=object(), vllm_config=object())
    tpu_runner.TPUModelRunner._init_speculative_decoding(runner)
    assert len(created) == int(last)
    assert runner.drafter == ("draft" if last else None)
    assert runner.rejection_sampler is None


@pytest.mark.parametrize("enabled", [False, True])
def test_cache_group_marker_excludes_target_mamba(monkeypatch, enabled):
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.sched.scheduler import Scheduler

    import vllm_torchtpu

    monkeypatch.setattr(Scheduler, "_mamba_block_aligned_split", lambda *a: 511)
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        MambaManager,
    )

    retained = []

    def tail(manager, request, num_tokens):
        if num_tokens == request.num_prompt_tokens // 128 * 128:
            retained.append(num_tokens)
            return num_tokens
        return None

    for manager in (FullAttentionManager, MambaManager):
        monkeypatch.setattr(manager, "_cache_partial_tail_block", tail)

    groups = [
        SimpleNamespace(
            layer_names=["model.layers.0.linear_attn"], is_eagle_group=False
        ),
        SimpleNamespace(
            layer_names=["model.layers.3.self_attn.attn"], is_eagle_group=False
        ),
        SimpleNamespace(
            layer_names=["mtp.layers.0.self_attn.attn"], is_eagle_group=False
        ),
    ]
    monkeypatch.setenv("TPU_MTP_PREFILL_ONLY", str(int(enabled)))
    monkeypatch.setattr(kv_cache_utils, "get_kv_cache_groups", lambda *a: groups)
    vllm_torchtpu._patch_vllm_mtp_prefill_cache_groups()
    wrapped = kv_cache_utils.get_kv_cache_groups
    vllm_torchtpu._patch_vllm_mtp_prefill_cache_groups()
    assert kv_cache_utils.get_kv_cache_groups is wrapped
    assert [g.is_eagle_group for g in wrapped(config(), {})] == [False, False, enabled]

    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True), hash_block_size=128
    )
    request = SimpleNamespace(num_prompt_tokens=511, num_computed_tokens=0)
    assert Scheduler._mamba_block_aligned_split(scheduler, request, 511) == (
        256 if enabled else 511
    )
    request.num_computed_tokens = 256
    assert Scheduler._mamba_block_aligned_split(scheduler, request, 255) == 511
    manager = SimpleNamespace(block_pool=SimpleNamespace(hash_block_size=128))
    MambaManager._cache_partial_tail_block(manager, request, 256)
    assert retained == ([256] if enabled else [])
    assert request.num_prompt_tokens == 511


def test_intermediate_chunks_forward_discard_indices():
    """Prefill must override meaningless intermediate samples with prompt tokens."""
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    proposer = Eagle3Proposer.__new__(Eagle3Proposer)
    proposer.draft_model = make_head()
    proposer.draft_chunks = [SimpleNamespace(hidden_states=torch.zeros(2, 3))]
    seen = []

    class StopAfterPrepare(Exception):
        pass

    def prepare(chunk, sampled, discarded, rejected, scheduler, **kwargs):
        seen.append(discarded)
        raise StopAfterPrepare

    proposer._prepare_draft_inputs = prepare
    with pytest.raises(StopAfterPrepare):
        proposer.prefill_only(SimpleNamespace(), torch.tensor([5, 6]), [0])
    assert seen == [[0]]


def test_chunk_seed_uses_known_prompt_token_and_final_sample():
    import numpy as np

    from vllm_torchtpu.layers.core.sequence_layout import AllSequenceLayoutPlanner
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    proposer = Eagle3Proposer.__new__(Eagle3Proposer)
    # Chunk starts at batch row 1: request a is partial, b reaches its end.
    proposer.runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(req_ids=["outside", "a", "b"]),
        requests={
            "a": SimpleNamespace(
                num_computed_tokens=2,
                get_token_id=lambda position: [1, 2, 3, 4, 5][position],
            ),
            "b": SimpleNamespace(num_computed_tokens=0),
        },
    )
    proposer._loop_bucket = lambda n: n

    def seed(ids, indices, next_ids):
        shifted = torch.cat((ids[1:], ids[-1:]))
        shifted[indices] = next_ids.to(ids.dtype)
        return shifted

    proposer._draft_seed_input_ids = seed
    chunk = SimpleNamespace(
        start_index=1,
        num_reqs=2,
        query_start_loc_np=np.array([0, 2, 4]),
        input_ids=torch.tensor([3, 4, 6, 7]),
        position_ids=torch.tensor([2, 3, 0, 1]),
        sequence_layout_plan=AllSequenceLayoutPlanner().prepare_dummy(
            num_tokens=4, num_reqs=2, kv_cache_initialized=True
        ),
    )
    result, positions, _, _ = proposer._prepare_draft_inputs(
        chunk,
        [],
        [1],
        None,
        SimpleNamespace(num_scheduled_tokens={"a": 2, "b": 2}),
        device_seed=torch.tensor([99, 88, 8]),
    )
    torch.testing.assert_close(result, torch.tensor([4, 5, 7, 8]))
    torch.testing.assert_close(positions, chunk.position_ids)


@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("prompt_tokens", [15, 16])
def test_real_hybrid_cache_reuses_mtp_and_rejects_changed_lookahead(
    monkeypatch, prompt_tokens, block_size
):
    import inspect

    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        MambaManager,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.request import Request

    import vllm_torchtpu

    monkeypatch.setenv("TPU_MTP_PREFILL_ONLY", "1")
    monkeypatch.setattr(kv_cache_utils, "get_kv_cache_groups", lambda *a: [])
    monkeypatch.setattr(Scheduler, "_mamba_block_aligned_split", lambda *a: 0)
    for cls in (FullAttentionManager, MambaManager):
        monkeypatch.setattr(
            cls,
            "_cache_partial_tail_block",
            inspect.unwrap(cls._cache_partial_tail_block),
        )
    vllm_torchtpu._patch_vllm_mtp_prefill_cache_groups()
    kv_cache_utils.init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=4, dtype=torch.float32
    )
    mamba = MambaSpec(
        block_size=block_size,
        shapes=((4, 4),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    cache = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["target.mamba"], mamba),
            KVCacheGroupSpec(["target.attention"], attention),
            KVCacheGroupSpec(["mtp.attention"], attention, is_eagle_group=True),
        ],
    )
    manager = KVCacheManager(
        cache,
        max_model_len=32,
        max_in_flight_tokens=16,
        scheduler_block_size=block_size,
        hash_block_size=4,
        enable_caching=True,
        use_eagle=True,
        num_prefill_lookahead=1,
    )

    def request(name, ids):
        return Request(
            name,
            ids,
            SamplingParams(max_tokens=1),
            None,
            block_hasher=kv_cache_utils.get_request_block_hasher(4, sha256),
        )

    ids = list(range(prompt_tokens))
    cold = request("cold", ids)
    for end in (4, 8, 12, prompt_tokens):
        manager.coordinator.new_step_starts()
        assert manager.allocate_slots(cold, end - cold.num_computed_tokens) is not None
        cold.num_computed_tokens = end
        # Model completion releases the CoW endpoints retained by the scheduler.
        _, retained = manager.take_kv_cache_block_copies()
        manager.block_pool.free_blocks(retained)
    manager.free(cold)
    manager.coordinator.new_step_starts()
    assert manager.get_computed_blocks(request("warm", ids))[1] == 8
    changed = ids.copy()
    changed[8] = 99
    # KV at MTP row 7 depends on token 8, outside the otherwise matching prefix.
    assert manager.get_computed_blocks(request("changed", changed))[1] < 8
    assert manager.reset_prefix_cache()
    assert manager.get_computed_blocks(request("reset", ids))[1] == 0


def test_next_mtp_registered_before_config_resolution(monkeypatch):
    from vllm.model_executor.models import ModelRegistry

    from vllm_torchtpu.models.vllm import register_models

    registered = {}
    monkeypatch.setattr(ModelRegistry, "register_model", registered.__setitem__)
    monkeypatch.setenv("TPU_MTP_PREFILL_ONLY", "1")
    register_models()
    assert registered["Qwen3NextMTP"].endswith(":Qwen3NextMTPPrefill")
    registered.clear()
    monkeypatch.setenv("TPU_MTP_PREFILL_ONLY", "0")
    register_models()
    assert registered["Qwen3NextMTP"].endswith(":Qwen3NextMTPPrefill")


@pytest.mark.parametrize("block_size, first_bucket", [(384, 1024), (9216, 512)])
def test_dynamic_pp_buckets_reach_copied_mtp_compilation_config(
    block_size, first_bucket
):
    from unittest.mock import MagicMock

    from vllm_torchtpu.core.pp_chunk_scheduler import SCHEDULER_CLS
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    drafter = MagicMock(spec=Eagle3Proposer)
    drafter.draft_vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(compile_sizes=[4096])
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=4),
            scheduler_config=SimpleNamespace(scheduler_cls=SCHEDULER_CLS),
            compilation_config=SimpleNamespace(compile_sizes=[4096]),
        ),
        _pp_chunk_granularity=None,
        _has_mamba_state=True,
        _pcp_enabled=True,
        cache_config=SimpleNamespace(block_size=block_size, mamba_cache_mode="align"),
        max_num_tokens=4096,
        num_tokens_paddings=[4096],
        drafter=drafter,
    )
    TPUModelRunner._add_pipeline_chunk_buckets(runner)
    assert first_bucket in runner.num_tokens_paddings
    assert drafter.draft_vllm_config.compilation_config.compile_sizes == (
        runner.num_tokens_paddings
    )
    assert drafter.draft_vllm_config.compilation_config.compile_sizes is not (
        runner.vllm_config.compilation_config.compile_sizes
    )


def test_multi_connector_handoff_uses_raiden_child():
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorMetadata,
    )

    from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
        SendMeta,
        TPUConnectorMetadata,
    )
    from vllm_torchtpu.spec_decode.prefill_only import handoff_seed_tokens

    raiden = TPUConnectorMetadata(
        reqs_to_send={
            "request": SendMeta(1, [9], 30.0, next_prompt_token_id=42),
        }
    )
    metadata = MultiKVConnectorMetadata((raiden, object()))
    sampled = torch.tensor([123])
    assert handoff_seed_tokens(sampled, ["request"], metadata).tolist() == [42]
    assert sampled.tolist() == [123]
    with pytest.raises(ValueError, match="exactly one Raiden"):
        handoff_seed_tokens(sampled, ["request"], MultiKVConnectorMetadata((object(),)))


@pytest.mark.parametrize(
    "pp,role,method,enabled",
    [
        (4, "kv_producer", "mtp", True),
        (4, "kv_consumer", "mtp", False),
        (1, "kv_producer", "mtp", False),
        (4, "kv_producer", "eagle3", False),
    ],
)
def test_pp_producer_mtp_auto_enabled(monkeypatch, pp, role, method, enabled):
    from vllm_torchtpu import env_override, envs

    monkeypatch.delenv("TPU_MTP_PREFILL_ONLY", raising=False)
    c = config()
    c.parallel_config.pipeline_parallel_size = pp
    c.speculative_config.method = method
    c.kv_transfer_config = SimpleNamespace(kv_role=role)
    env_override.configure_mtp_prefill_mode(c)
    assert envs.TPU_MTP_PREFILL_ONLY is enabled


@pytest.mark.parametrize("last", [False, True])
def test_pp_mtp_warmup_all_stages_cover_pcp_buckets(monkeypatch, last):
    from contextlib import nullcontext

    from vllm_torchtpu.runner import tpu_runner

    monkeypatch.setenv("SPEC_WARMUP", "1")
    monkeypatch.setattr(tpu_runner, "_suspend_kv_transfer_group", nullcontext)
    monkeypatch.setattr(tpu_runner, "synchronize_device", lambda: None)
    seen = []
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=8),
        num_tokens_paddings=[512, 1024, 1536, 4096],
        max_model_len=65536,
        _pp_is_last=last,
        _precompile_timed=lambda _: nullcontext(),
        _warmup_one_pcp_mtp_prefill=lambda n: seen.append(n) or True,
    )
    tpu_runner.TPUModelRunner._warmup_pp_mtp_prefill(runner)
    assert seen == [4096, 8192, 12288, 32768]

    runner._warmup_one_pcp_mtp_prefill = lambda n: False
    with pytest.raises(RuntimeError, match="refusing to serve"):
        tpu_runner.TPUModelRunner._warmup_pp_mtp_prefill(runner)


def test_non_prefill_mtp_delegates_upstream_forward(monkeypatch):
    from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMTP

    head = make_head(Qwen3NextMTPPrefill)
    head._prefill_only = False
    seen = []

    def forward(self, **kwargs):
        seen.append(kwargs)
        return kwargs["hidden_states"]

    monkeypatch.setattr(Qwen3NextMTP, "forward", forward)
    hidden = torch.zeros(2, 3)
    result = head.forward(
        torch.tensor([1, 2]), torch.tensor([0, 1]), hidden, spec_step_idx=2
    )
    assert result is hidden
    assert seen[0]["spec_step_idx"] == 2


def test_pp_prefill_warmup_drains_handoff_before_sampling(monkeypatch):
    from vllm_torchtpu.runner import tpu_runner

    monkeypatch.setenv("TPU_MTP_PREFILL_ONLY", "1")
    seen = []
    runner = SimpleNamespace(
        max_model_len=65536,
        kv_cache_config=SimpleNamespace(
            num_blocks=128,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=384))
            ],
        ),
        _pp_wave=SimpleNamespace(settle=lambda: seen.append("settle")),
        execute_model=lambda so: seen.append(
            ("execute", so.total_num_scheduled_tokens)
        ),
        sample_tokens=lambda _: seen.append("sample"),
        _warmup_spec_decode_cleanup=lambda rid: seen.append("cleanup") or True,
    )
    assert tpu_runner.TPUModelRunner._warmup_one_pcp_mtp_prefill(runner, 4096)
    assert seen == [("execute", 4096), "settle", "sample", "cleanup"]

    def fail(_):
        raise RuntimeError("test sampling failure")

    runner.sample_tokens = fail
    seen.clear()
    assert not tpu_runner.TPUModelRunner._warmup_one_pcp_mtp_prefill(runner, 4096)
    assert seen[-1] == "cleanup"
