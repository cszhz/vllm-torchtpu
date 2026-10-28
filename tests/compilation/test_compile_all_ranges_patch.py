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
"""How TPU_PARALLEL_PRECOMPILE splits a subgraph's ranges among a host's ranks.

Each range has one owner on the host, which compiles it; the other ranks
compile their own ranges first, then wait for and replay the owners' artifacts.
The biggest range takes far longer than the others, so startup is bounded by
when it starts: local rank 0 must start on it, not reach it after the smallest.
"""

from types import SimpleNamespace

import pytest
from vllm.compilation import piecewise_backend
from vllm.config.utils import Range
from vllm.distributed import parallel_state

from vllm_torchtpu import _patch_vllm_compile_all_ranges
from vllm_torchtpu.compilation import tpu_compiler

SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


@pytest.fixture
def compile_all_ranges(monkeypatch):
    monkeypatch.setenv("TPU_PARALLEL_PRECOMPILE", "1")
    monkeypatch.setenv("TPU_PRECOMPILE_BIGGEST_FIRST", "1")
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "1")
    # ranks_per_host() also reads LOCAL_WORLD_SIZE; keep a launcher's value
    # (torchrun, CI) from changing which rank owns which range.
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("TPU_LOCAL_RANK_OFFSET", raising=False)
    monkeypatch.setattr(
        piecewise_backend, "create_concrete_args", lambda graph, size: [size]
    )
    # Restored on teardown, so the patch does not leak into other tests.
    monkeypatch.setattr(
        piecewise_backend.PiecewiseBackend,
        "compile_all_ranges",
        piecewise_backend.PiecewiseBackend.compile_all_ranges,
    )
    _patch_vllm_compile_all_ranges()
    return piecewise_backend.PiecewiseBackend.compile_all_ranges


def _host(monkeypatch, ranks_per_host):
    """A world of two hosts with `ranks_per_host` ranks each, or an unknown
    one for None."""
    if ranks_per_host is None:

        def get_world_group():
            raise AssertionError("world group is not initialized")

    else:

        def get_world_group():
            return SimpleNamespace(world_size=2 * ranks_per_host)

    monkeypatch.setattr(parallel_state, "get_world_group", get_world_group)
    monkeypatch.setattr(parallel_state, "get_node_count", lambda: 2)


def _compile(
    compile_all_ranges,
    monkeypatch,
    local_rank,
    ranks_per_host=8,
    disable_cache=False,
    dp_size=1,
    dp_size_local=1,
):
    """(range end, peer wait in seconds) per compile, in order."""
    _host(monkeypatch, ranks_per_host)
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    calls = []

    def compile(graph, args, *rest, compile_range, **kwargs):
        calls.append(
            (compile_range.end, getattr(tpu_compiler._peer_wait, "timeout_s", 0.0))
        )
        return lambda *a: a

    # In vLLM's order: smallest range first.
    backend = SimpleNamespace(
        graph=object(),
        range_entries={
            Range(s, s): piecewise_backend.RangeEntry(compile_range=Range(s, s))
            for s in SIZES
        },
        vllm_backend=SimpleNamespace(
            compiler_manager=SimpleNamespace(
                compile=compile, disable_cache=disable_cache
            ),
            inductor_config={},
            is_encoder=False,
        ),
        compilation_config=None,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=dp_size, data_parallel_size_local=dp_size_local
            )
        ),
        piecewise_compile_index=0,
        total_piecewise_compiles=1,
        _log_compile_start=lambda compile_range: None,
    )
    compile_all_ranges(backend)
    return calls


def _owned(calls):
    return [end for end, wait in calls if wait == 0.0]


def test_local_rank_0_starts_on_the_biggest_range(compile_all_ranges, monkeypatch):
    assert _compile(compile_all_ranges, monkeypatch, 0)[0][0] == 4096


def test_each_range_has_one_owner_per_host(compile_all_ranges, monkeypatch):
    owned = [_owned(_compile(compile_all_ranges, monkeypatch, r)) for r in range(8)]
    # Dealt in a snake: the rank with the smallest first range gets the 9th.
    assert owned == [[4096], [2048], [1024], [512], [256], [128], [64], [32, 16]]


def test_other_ranges_wait_for_their_owner(compile_all_ranges, monkeypatch):
    calls = _compile(compile_all_ranges, monkeypatch, 0)
    assert calls == [(4096, 0.0)] + [
        (end, tpu_compiler.PEER_WAIT_S)
        for end in [32, 64, 128, 256, 512, 1024, 2048, 16]
    ]


def test_every_rank_compiles_every_range(compile_all_ranges, monkeypatch):
    for rank in range(8):
        calls = _compile(compile_all_ranges, monkeypatch, rank)
        assert sorted(end for end, _ in calls) == SIZES


def test_a_smaller_host_splits_the_ranges_among_its_ranks(
    compile_all_ranges, monkeypatch
):
    owned = [
        _owned(_compile(compile_all_ranges, monkeypatch, r, ranks_per_host=4))
        for r in range(4)
    ]
    assert owned == [[4096, 32, 16], [2048, 64], [1024, 128], [512, 256]]


def test_no_wait_when_the_host_size_is_unknown(compile_all_ranges, monkeypatch):
    calls = _compile(compile_all_ranges, monkeypatch, 1, ranks_per_host=None)
    assert all(wait == 0.0 for _, wait in calls)
    assert sorted(end for end, _ in calls) == SIZES


def test_biggest_first_off_deals_in_vllm_order(compile_all_ranges, monkeypatch):
    monkeypatch.setenv("TPU_PRECOMPILE_BIGGEST_FIRST", "0")
    owned = [_owned(_compile(compile_all_ranges, monkeypatch, r)) for r in range(8)]
    # vLLM's order, smallest first: local rank 0 starts on the smallest range.
    assert owned == [[16], [32], [64], [128], [256], [512], [1024], [2048, 4096]]
    for rank in range(8):
        calls = _compile(compile_all_ranges, monkeypatch, rank)
        assert sorted(end for end, _ in calls) == SIZES


@pytest.mark.parametrize(
    "biggest_first, firsts",
    [
        ("1", [4096, 2048, 1024, 512, 256, 128, 64, 32]),
        ("0", SIZES[:8]),
        (None, SIZES[:8]),
    ],
)
@pytest.mark.parametrize("peer_replay", ["0", None])
def test_peer_replay_off_compiles_every_range_without_waiting(
    compile_all_ranges, monkeypatch, peer_replay, biggest_first, firsts
):
    """TPU_PRECOMPILE_PEER_REPLAY off, explicitly or by default (None): no
    owners and no waits; each local rank starts on a different range and
    compiles every range itself."""
    for name, value in [
        ("TPU_PRECOMPILE_PEER_REPLAY", peer_replay),
        ("TPU_PRECOMPILE_BIGGEST_FIRST", biggest_first),
    ]:
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)
    runs = [_compile(compile_all_ranges, monkeypatch, r) for r in range(8)]
    assert [calls[0][0] for calls in runs] == firsts
    for calls in runs:
        assert all(wait == 0.0 for _, wait in calls)
        assert sorted(end for end, _ in calls) == SIZES


def _host_owners(compile_all_ranges, monkeypatch, local_ranks, **kwargs):
    """Every range a rank compiled without waiting, for each rank present."""
    return {
        r: _owned(_compile(compile_all_ranges, monkeypatch, r, **kwargs))
        for r in local_ranks
    }


def _assert_one_owner_each(owners):
    """The ranks present own every range exactly once between them, so no
    rank waits on a slot no process holds."""
    owned = [end for ends in owners.values() for end in ends]
    assert sorted(owned) == SIZES


def _assert_opted_out(compile_all_ranges, monkeypatch, local_ranks, **kwargs):
    for r in local_ranks:
        calls = _compile(compile_all_ranges, monkeypatch, r, **kwargs)
        assert all(wait == 0.0 for _, wait in calls)
        assert sorted(end for end, _ in calls) == SIZES


def test_dp_ranks_use_local_world_size_for_host_splitting(
    compile_all_ranges, monkeypatch
):
    """Single-host DP8 x TP1 (+EP): each DP engine is its own world group of
    one rank, so the host's 8 engines split the ranges between them."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    owners = _host_owners(
        compile_all_ranges,
        monkeypatch,
        range(8),
        ranks_per_host=1,
        dp_size=8,
        dp_size_local=8,
    )
    assert list(owners.values()) == [
        [4096],
        [2048],
        [1024],
        [512],
        [256],
        [128],
        [64],
        [32, 16],
    ]


@pytest.mark.parametrize(
    "local_world, dp_size_local",
    [
        (16, 8),  # multiprocess: LOCAL_WORLD_SIZE is the global slice width
        (8, 16),  # Ray without --data-parallel-size-local
    ],
)
def test_multi_host_dp_counts_only_this_hosts_engines(
    compile_all_ranges, monkeypatch, local_world, dp_size_local
):
    """Multi-host DP16 x TP1 + EP, 8 engines per host: whichever of
    LOCAL_WORLD_SIZE and data_parallel_size_local overstates the host, the 8
    engines present own every range."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local_world))
    owners = _host_owners(
        compile_all_ranges,
        monkeypatch,
        range(8),
        ranks_per_host=1,
        dp_size=16,
        dp_size_local=dp_size_local,
    )
    _assert_one_owner_each(owners)


def test_dense_dp_engines_compile_alone(compile_all_ranges, monkeypatch):
    """Dense DP (no EP): the replicas are independent engines with
    data_parallel_size collapsed to 1, all bound to LOCAL_RANK=0 even when
    LOCAL_WORLD_SIZE=8. None of them waits on slots 1..7."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    _assert_opted_out(
        compile_all_ranges, monkeypatch, [0] * 8, ranks_per_host=1, dp_size=1
    )


def test_multi_rank_engines_do_not_split_across_dp_groups(
    compile_all_ranges, monkeypatch
):
    """DP2 x TP4: each engine's 4 ranks split the ranges among themselves,
    whatever LOCAL_WORLD_SIZE says."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    _assert_one_owner_each(
        _host_owners(compile_all_ranges, monkeypatch, range(4), ranks_per_host=4)
    )


def test_second_tp_group_maps_onto_its_engines_slots(compile_all_ranges, monkeypatch):
    """DP2 x TP4 + EP: the second engine's ranks have LOCAL_RANK 4..7 but
    slots 0..3 in their own engine, so they own its ranges (they used to own
    none and replay nothing)."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    first = _host_owners(compile_all_ranges, monkeypatch, range(4), ranks_per_host=4)
    second = _host_owners(
        compile_all_ranges, monkeypatch, range(4, 8), ranks_per_host=4
    )
    _assert_one_owner_each(second)
    assert list(second.values()) == list(first.values())


def test_local_rank_offset_is_taken_off(compile_all_ranges, monkeypatch):
    """TPU_LOCAL_RANK_OFFSET=4 with 4 single-rank DP engines: the binding
    sets LOCAL_RANK 4..7 and LOCAL_WORLD_SIZE 8; the engines are slots 0..3."""
    monkeypatch.setenv("TPU_LOCAL_RANK_OFFSET", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    owners = _host_owners(
        compile_all_ranges,
        monkeypatch,
        range(4, 8),
        ranks_per_host=1,
        dp_size=4,
        dp_size_local=4,
    )
    _assert_one_owner_each(owners)


def test_a_slot_outside_the_host_opts_out(compile_all_ranges, monkeypatch):
    """A rank whose slot falls outside the host count waits on no one."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    _assert_opted_out(
        compile_all_ranges,
        monkeypatch,
        [9],
        ranks_per_host=1,
        dp_size=16,
        dp_size_local=8,
    )


def test_peer_replay_is_ignored_with_the_compile_cache_disabled(
    compile_all_ranges, monkeypatch
):
    """Peers hand ranges over through vLLM's compile cache, so with the cache
    disabled TPU_PRECOMPILE_PEER_REPLAY=1 is an invalid combination: it warns
    and falls back to every rank compiling every range, without waiting."""
    import vllm_torchtpu

    warnings = []

    class _Logger:
        def warning_once(self, msg, *args, **kwargs):
            warnings.append(msg)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(vllm_torchtpu, "logger", _Logger())
    runs = [
        _compile(compile_all_ranges, monkeypatch, r, disable_cache=True)
        for r in range(8)
    ]
    for calls in runs:
        assert all(wait == 0.0 for _, wait in calls)
        assert sorted(end for end, _ in calls) == SIZES
    assert warnings and "VLLM_DISABLE_COMPILE_CACHE" in warnings[0]
