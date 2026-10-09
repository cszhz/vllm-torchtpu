# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the last-stage, forward-only MTP diagnostic."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.qwen3_5_mtp_prefill import Qwen3_5MoeMTPPrefill
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


def test_kv_transfer_still_rejected():
    c = config()
    c.kv_transfer_config = object()
    with pytest.raises(NotImplementedError, match="does not transfer KV"):
        validate_prefill_only_config(c)


class Layer(torch.nn.Module):
    use_attn_reduce_scatter_for_moe = False

    def forward(self, positions, hidden_states, residual):
        assert residual is None
        self.positions = positions
        return hidden_states * 2, hidden_states


class Norm(torch.nn.Module):
    def forward(self, x, residual):
        return x + residual, None


def make_head():
    head = Qwen3_5MoeMTPPrefill.__new__(Qwen3_5MoeMTPPrefill)
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


def test_whole_head_projects_every_prompt_row_without_first_rank_branch():
    head = make_head()
    ids = torch.tensor([1, 2, 3, 4])
    positions = torch.arange(4)
    hidden = torch.randn(4, 3)
    expected = head.model.fc(torch.cat([head.model.embed_tokens(ids), hidden], -1)) * 3
    result = Qwen3_5MoeMTPPrefill.forward(head, ids, positions, hidden)
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


@pytest.mark.parametrize("partial", [False, True])
def test_prefill_executes_head_without_producing_draft_tokens(partial):
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
    proposer.prefill_only(SimpleNamespace(), torch.tensor([5]))
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


def test_real_hybrid_cache_reuses_mtp_and_rejects_changed_lookahead(monkeypatch):
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
        block_size=16, num_kv_heads=1, head_size=4, dtype=torch.float32
    )
    mamba = MambaSpec(
        block_size=16,
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
        scheduler_block_size=16,
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

    ids = list(range(15))
    cold = request("cold", ids)
    for end in (4, 8, 12, 15):
        manager.coordinator.new_step_starts()
        assert manager.allocate_slots(cold, end - cold.num_computed_tokens) is not None
        cold.num_computed_tokens = end
        manager.cache_blocks(cold, end)
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
