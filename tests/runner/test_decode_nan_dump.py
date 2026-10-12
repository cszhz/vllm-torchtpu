# SPDX-License-Identifier: Apache-2.0
"""Protect dump cadence, cache ownership, and complete batch serialization."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

pytestmark = pytest.mark.cpu_test


def make_runner():
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=2, dtype=torch.float32
    )
    caches = {
        name: SimpleNamespace(kv_cache=torch.arange(64.0).reshape(4, 4, 2, 2))
        for name in ("target.attn", "draft.attn")
    }
    return SimpleNamespace(
        requests={
            "a": SimpleNamespace(
                prompt_token_ids=[1, 2], output_token_ids=[3], block_ids=([1],)
            ),
            "b": SimpleNamespace(
                prompt_token_ids=[4, 5], output_token_ids=[6], block_ids=([2, 3],)
            ),
        },
        input_batch=SimpleNamespace(
            req_ids=["a", "b"],
            num_computed_tokens_cpu=np.array([3, 3]),
            num_prompt_tokens=np.array([2, 2]),
        ),
        parallel_config=SimpleNamespace(rank=1, data_parallel_index=0),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=caches)
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=list(caches), kv_cache_spec=spec)
            ]
        ),
        _unified_kv_layout=True,
    )


def record(dumper, runner):
    dumper.record_forward(
        runner,
        0,
        2,
        {"input_ids": torch.tensor([3, 6, 0]), "positions": torch.tensor([2, 2, 0])},
        {
            name: SimpleNamespace(seq_lens=torch.tensor([3, 3, 0]))
            for name in ("target.attn", "draft.attn")
        },
    )


def load_one(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def test_zero_twenty_forty_cadence_is_independent_for_target_and_draft(tmp_path):
    from vllm_torchtpu.runner.decode_nan_dump import DecodeNaNDumper

    runner = make_runner()
    target = DecodeNaNDumper("target", {"target.attn"}, tmp_path)
    draft = DecodeNaNDumper("draft", {"draft.attn"}, tmp_path)
    bad = torch.tensor([[1.0], [float("nan")], [float("nan")]])
    for _ in range(41):
        record(target, runner)
        target.check(runner, torch.ones(3, 1), np.array([0, 1]))
        record(target, runner)
        target.check(runner, bad, np.array([0, 1]))
    record(draft, runner)
    draft.check(runner, bad, np.array([0, 1]))
    reports = [load_one(p) for p in tmp_path.glob("*/manifest.pkl")]
    assert sorted(
        r["anomaly_index"] for r in reports if r["model_role"] == "target"
    ) == [0, 20, 40]
    assert [r["anomaly_index"] for r in reports if r["model_role"] == "draft"] == [0]


def test_pickle_contains_entire_batch_but_only_the_owning_models_cache(tmp_path):
    from vllm_torchtpu.runner.decode_nan_dump import DecodeNaNDumper

    runner = make_runner()
    dumper = DecodeNaNDumper("draft", {"draft.attn"}, tmp_path)
    record(dumper, runner)
    dumper.check(runner, torch.tensor([[1.0], [float("nan")]]), np.array([0, 1]))
    (folder,) = tmp_path.iterdir()
    report = load_one(folder / "manifest.pkl")
    assert report["request_ids"] == ["a", "b"]
    assert report["bad_request_ids"] == ["b"]
    assert report["cache_phase"] == "after_forward_before_logits"
    assert report["gdn_input_state_available"] is False
    inputs = load_one(folder / "inputs.pkl")
    assert torch.equal(inputs["forward"]["input_ids"], torch.tensor([3, 6, 0]))
    assert inputs["requests"]["a"]["prompt_token_ids"] == [1, 2]
    assert inputs["requests"]["b"]["output_token_ids"] == [6]
    assert [entry["layer_name"] for entry in report["layers"]] == ["draft.attn"]
    with (folder / report["layers"][0]["path"]).open("rb") as stream:
        header = pickle.load(stream)
        blocks = [pickle.load(stream) for _ in range(header["num_blocks"])]
        assert stream.read() == b""
    assert [block["block_id"] for block in blocks] == [1, 2, 3]
    for block in blocks:
        expected = runner.vllm_config.compilation_config.static_forward_context[
            "draft.attn"
        ].kv_cache[block["block_id"] : block["block_id"] + 1]
        assert torch.equal(block["tensor_bytes"][0].view(torch.float32), expected)
    runner.vllm_config.compilation_config.static_forward_context[
        "draft.attn"
    ].kv_cache.zero_()
    assert blocks[-1]["tensor_bytes"][0].view(torch.float32).sum() > 0


def test_prefill_padding_and_warmup_do_not_advance_anomaly_counter(tmp_path):
    from vllm_torchtpu.runner.decode_nan_dump import DecodeNaNDumper

    runner = make_runner()
    runner.input_batch.num_computed_tokens_cpu[1] = 1
    dumper = DecodeNaNDumper("target", {"target.attn"}, tmp_path)
    record(dumper, runner)
    dumper.check(
        runner, torch.tensor([[1.0], [float("nan")], [float("nan")]]), np.array([0, 1])
    )
    with dumper.pause():
        record(dumper, runner)
        dumper.check(runner, torch.full((3, 1), float("nan")), np.array([0, 1]))
    assert not list(tmp_path.iterdir())
    record(dumper, runner)
    dumper.check(runner, torch.tensor([[float("nan")], [1.0], [1.0]]), np.array([0, 1]))
    report = load_one(next(tmp_path.glob("*/manifest.pkl")))
    assert report["anomaly_index"] == 0
    assert report["bad_request_ids"] == ["a"]


def test_selected_verify_rows_preserve_request_ownership(tmp_path):
    from vllm_torchtpu.runner.decode_nan_dump import DecodeNaNDumper

    runner = make_runner()
    dumper = DecodeNaNDumper("target", {"target.attn"}, tmp_path)
    record(dumper, runner)
    hidden = torch.tensor([[float("nan")], [1.0], [1.0], [float("nan")]])
    # The unselected NaN at row 0 and padded selection are not a trigger.
    indices = torch.tensor([1, 2, 3, 0])
    dumper.check(runner, hidden, np.array([0, 0, 1]), indices)
    report = load_one(next(tmp_path.glob("*/manifest.pkl")))
    assert report["bad_request_ids"] == ["b"]


def test_manager_blocks_expand_to_all_kernel_pages(tmp_path):
    from vllm_torchtpu.runner.decode_nan_dump import DecodeNaNDumper

    runner = make_runner()
    context = runner.vllm_config.compilation_config.static_forward_context
    context["target.attn"].kv_cache = torch.arange(128.0).reshape(8, 4, 2, 2)
    runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec = FullAttentionSpec(
        block_size=8, num_kv_heads=1, head_size=2, dtype=torch.float32
    )
    dumper = DecodeNaNDumper("target", {"target.attn"}, tmp_path)
    record(dumper, runner)
    dumper.check(runner, torch.full((2, 1), float("nan")), np.array([0, 1]))
    (folder,) = tmp_path.iterdir()
    report = load_one(folder / "manifest.pkl")
    with (folder / report["layers"][0]["path"]).open("rb") as stream:
        header = pickle.load(stream)
        blocks = [pickle.load(stream) for _ in range(header["num_blocks"])]
    assert header["kernel_pages_per_block"] == 2
    assert torch.equal(
        blocks[-1]["tensor_bytes"][0].view(torch.float32),
        torch.arange(96.0, 128.0).reshape(2, 4, 2, 2),
    )


def test_unified_gdn_pickle_preserves_fp8_carrier_bits(tmp_path):
    from vllm_torchtpu.runner.decode_nan_dump import DecodeNaNDumper

    runner = make_runner()
    name = "target.gdn"
    carrier = (
        torch.arange(128, dtype=torch.uint8)
        .reshape(8, 4, 2, 2)
        .view(torch.float8_e4m3fn)
    )
    runner.vllm_config.compilation_config.static_forward_context = {
        name: SimpleNamespace(kv_cache=[carrier])
    }
    runner.kv_cache_config.kv_cache_groups = [
        SimpleNamespace(
            layer_names=[name],
            kv_cache_spec=MambaSpec(
                block_size=1,
                shapes=((4,),),
                dtypes=(torch.bfloat16,),
                page_size_padded=32,
            ),
        )
    ]
    dumper = DecodeNaNDumper("target", {name}, tmp_path)
    dumper.record_forward(runner, 0, 2, {}, {name: SimpleNamespace()})
    dumper.check(runner, torch.full((2, 1), float("nan")), np.array([0, 1]))
    (folder,) = tmp_path.iterdir()
    report = load_one(folder / "manifest.pkl")
    with (folder / report["layers"][0]["path"]).open("rb") as stream:
        header = pickle.load(stream)
        blocks = [pickle.load(stream) for _ in range(header["num_blocks"])]
    restored = blocks[-1]["tensor_bytes"][0]
    assert header["tensor_dtypes"] == [torch.float8_e4m3fn]
    assert restored.dtype == torch.uint8
    assert torch.equal(
        restored.view(torch.uint8),
        torch.arange(96, 128, dtype=torch.uint8).reshape(2, 4, 2, 2),
    )
