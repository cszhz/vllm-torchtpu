# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One candidate-reuse indexer layer, without model weights or a serving engine.

Synthetic per-query candidates preserve the serving geometry but are not a
capture of model activations. Measure preparation and scoring separately, then
measure their compiled composition. Membership alternatives are diagnostic only.
"""

import argparse
import functools
import json
import statistics
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8_candidate_cache import (
    PreparedCandidates,
    candidate_membership,
    gather_candidate_keys,
    prepare_candidate_metadata,
    prepare_candidates,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PackedFP8Layout,
    quantize_index_rows,
    read_fp8_records,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import indexer_impl
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_impl import _score_prepared


def pack_fixture(keys, layout):
    """Initialize pages without the incremental writer's SMEM index limit."""
    values, scales, _ = quantize_index_rows(keys)
    bits = jax.lax.bitcast_convert_type(values, jnp.uint8).astype(jnp.uint32)
    bits = bits.reshape(-1, 32, 4)
    words = bits[..., 0] | (bits[..., 1] << 8)
    words |= (bits[..., 2] << 16) | (bits[..., 3] << 24)
    records = jnp.concatenate(
        (
            words.astype(jnp.int32),
            jax.lax.bitcast_convert_type(scales, jnp.int32)[:, None],
        ),
        axis=1,
    )
    pages = records.reshape(layout.pages, layout.page_size, 33).transpose(0, 2, 1)
    pages = pages.reshape(layout.pages, -1)
    return jnp.pad(
        pages, ((0, 0), (0, layout.page_rows * 128 - pages.shape[1]))
    ).reshape(-1, 128)


def membership(candidates, canonical, *, method):
    ordered = jnp.sort(candidates, axis=1)
    slots = jax.vmap(lambda a, v: jnp.searchsorted(a, v, method=method))(
        ordered, canonical
    )
    found = jnp.take_along_axis(
        ordered, jnp.minimum(slots, candidates.shape[1] - 1), axis=1
    )
    return (canonical >= 0) & (found == canonical)


def measure(name, fn, inputs, root, repeats, dimensions):
    started = time.perf_counter()
    lowered = jax.jit(fn).lower(*inputs)
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started
    hlo = root / "rank-0/compiler/hlo"
    hlo.mkdir(parents=True, exist_ok=True)
    (hlo / f"{name}.txt").write_text(executable.as_text())
    for _ in range(3):
        jax.block_until_ready(executable(*inputs))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        jax.block_until_ready(executable(*inputs))
        samples.append((time.perf_counter_ns() - started) / 1e6)
    metric = dict(
        dimensions,
        variant=name,
        latency_ms=statistics.median(samples),
        sample_statistics={"min_ms": min(samples), "max_ms": max(samples)},
        samples_ms=samples,
        compile_time_s=compile_seconds,
    )
    with (root / "rank-0/benchmark/metrics.jsonl").open("a") as out:
        out.write(json.dumps(metric) + "\n")
    print(json.dumps(metric), flush=True)
    profile = root / "rank-0/profiling/xprof" / name
    profile.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(profile)):
        for _ in range(5):
            jax.block_until_ready(executable(*inputs))
    return executable(*inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--queries-per-request", type=int, default=6)
    parser.add_argument("--width", type=int, default=98816)
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.7)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--gather-sweep", action="store_true")
    parser.add_argument(
        "--stage", choices=("all", "prepare", "gather", "score"), default="all"
    )
    args = parser.parse_args()
    if not 0 <= args.overlap <= 1 or args.width % args.page_size:
        raise ValueError("Use overlap in [0,1] and a page-aligned width")
    root = args.out
    (root / "rank-0/benchmark").mkdir(parents=True, exist_ok=True)
    dimensions = dict(
        requests=args.requests,
        queries_per_request=args.queries_per_request,
        tokens=args.requests * args.queries_per_request,
        seq_len=args.width,
        candidate_blocks=args.candidates,
        block_size=args.block,
        page_size=args.page_size,
        overlap=args.overlap,
        topk=args.topk,
    )
    (root / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                workflow="operator-optimization",
                operator_family="indexer",
                operator_name="paired_candidate_layer",
                dimensions=dimensions,
                source_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                input_kind="synthetic, fixed seed; not captured model activations",
                timing="compiled warm runs; host dispatch plus device completion",
            ),
            indent=2,
        )
    )
    rng = np.random.default_rng(826)
    tokens = dimensions["tokens"]
    pages_per_request = args.width // args.page_size
    layout = PackedFP8Layout(
        args.requests * pages_per_request, page_size=args.page_size
    )
    pages = jnp.asarray(rng.permutation(layout.pages).reshape(args.requests, -1))
    owners = jnp.repeat(
        jnp.arange(args.requests, dtype=jnp.int32), args.queries_per_request
    )
    candidate_rows = []
    for _ in range(args.requests):
        common = rng.choice(
            args.width // args.block, int(args.candidates * args.overlap), replace=False
        )
        for _ in range(args.queries_per_request):
            extra = rng.choice(
                args.width // args.block, args.candidates - len(common), replace=False
            )
            candidate_rows.append(np.concatenate((common, extra)))
    candidates = jnp.asarray(np.array(candidate_rows), jnp.int32)
    metadata = measure(
        "prepare_metadata",
        functools.partial(
            prepare_candidate_metadata,
            layout=layout,
            block=args.block,
            width=args.width,
            grouped=True,
        ),
        (pages, candidates, owners),
        root,
        args.repeats,
        dimensions,
    )
    expected = np.stack(
        [
            np.isin(row, original) & (row >= 0)
            for row, original in zip(
                np.asarray(metadata.block_ids), np.asarray(candidates), strict=True
            )
        ]
    )
    np.testing.assert_array_equal(np.asarray(metadata.membership), expected)
    host_ids = np.asarray(metadata.block_ids)
    host_physical = np.asarray(metadata.physical)
    host_candidates = np.asarray(candidates)
    host_pages = np.asarray(pages)
    pair_size = 2 if args.queries_per_request % 2 == 0 else 1
    group = 0
    fragment = min(args.block, args.page_size)
    for request in range(args.requests):
        for first in range(0, args.queries_per_request, pair_size):
            row = request * args.queries_per_request + first
            union = np.unique(host_candidates[row : row + pair_size])
            for query in range(row, row + pair_size):
                np.testing.assert_array_equal(
                    host_ids[query][host_ids[query] >= 0], union
                )
            logical = (
                union[:, None] * args.block
                + np.arange(max(1, args.block // args.page_size)) * fragment
            ).reshape(-1)
            expected_physical = (
                host_pages[request, logical // args.page_size]
                * (args.page_size // fragment)
                + logical % args.page_size // fragment
            )
            np.testing.assert_array_equal(
                host_physical[group][host_physical[group] >= 0], expected_physical
            )
            group += 1
    if args.stage == "prepare":
        (root / "correctness.json").write_text(
            json.dumps(
                {"membership_numpy": "passed", "union_and_physical_numpy": "passed"}
            )
        )
        return
    # Generate on device to keep host memory bounded. Split at a page boundary.
    keys = jax.random.normal(
        jax.random.key(826), (layout.pages * layout.page_size, 128), jnp.bfloat16
    )
    pool = jax.jit(functools.partial(pack_fixture, layout=layout))(keys)
    banks = tuple(jnp.split(pool, 2))
    jax.block_until_ready(banks)
    probes = jnp.asarray(
        [0, args.page_size - 1, args.page_size, keys.shape[0] // 2, keys.shape[0] - 1],
        jnp.int32,
    )
    values, scales, _ = quantize_index_rows(keys[probes])
    np.testing.assert_array_equal(
        np.asarray(read_fp8_records(banks, probes, layout)),
        np.asarray((values.astype(jnp.float32) * scales[:, None]).astype(jnp.bfloat16)),
    )
    del pool, keys
    q = jax.random.normal(jax.random.key(827), (tokens, 32, 128)).astype(
        jnp.float8_e4m3fn
    )
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (tokens, 32)), jnp.float32)
    lengths = jnp.asarray(
        args.width
        - np.tile(np.arange(args.queries_per_request - 1, -1, -1), args.requests),
        jnp.int32,
    )
    prepare = functools.partial(
        prepare_candidates,
        layout=layout,
        block=args.block,
        width=args.width,
        grouped=True,
    )
    if args.stage == "all":
        prepared = measure(
            "prepare",
            prepare,
            (banks, pages, candidates, owners),
            root,
            args.repeats,
            dimensions,
        )
        expected = np.stack(
            [
                np.isin(row, original) & (row >= 0)
                for row, original in zip(
                    np.asarray(prepared.block_ids), np.asarray(candidates), strict=True
                )
            ]
        )
        np.testing.assert_array_equal(np.asarray(prepared.membership), expected)
        for method in ("scan", "sort"):
            actual = measure(
                "membership_" + method,
                functools.partial(membership, method=method),
                (candidates, prepared.block_ids),
                root,
                args.repeats,
                dimensions,
            )
            np.testing.assert_array_equal(np.asarray(actual), expected)
        actual = measure(
            "membership_histogram",
            functools.partial(candidate_membership, blocks=args.width // args.block),
            (candidates, prepared.block_ids),
            root,
            args.repeats,
            dimensions,
        )
        np.testing.assert_array_equal(np.asarray(actual), expected)
    offsets = np.asarray(metadata.offsets)
    physical = metadata.physical
    gathered = measure(
        "gather",
        functools.partial(
            gather_candidate_keys,
            block=min(args.block, args.page_size),
            layout=layout,
        ),
        (banks, physical, metadata.offsets),
        root,
        args.repeats,
        dimensions,
    )
    # Only valid blocks are initialized; skipped padding is never consumed.
    per_page = 128 // min(args.block, args.page_size)
    valid = np.asarray(physical >= 0).reshape(-1, 1, per_page)
    valid = np.broadcast_to(valid, (valid.shape[0], 33, per_page)).reshape(-1)
    if args.stage == "all":
        np.testing.assert_array_equal(
            np.asarray(gathered).reshape(-1, min(args.block, args.page_size))[valid],
            np.asarray(prepared.keys).reshape(-1, min(args.block, args.page_size))[
                valid
            ],
        )
    else:
        # Read original cache records directly; no second gather as an oracle.
        slots = np.flatnonzero(np.asarray(physical).reshape(-1) >= 0)[::97]
        fragment = min(args.block, args.page_size)
        target = (slots[:, None] * fragment + np.arange(fragment)).reshape(-1)
        source = (
            np.asarray(physical).reshape(-1)[slots, None] * fragment
            + np.arange(fragment)
        ).reshape(-1)
        np.testing.assert_array_equal(
            np.asarray(
                read_fp8_records(
                    (gathered,),
                    jnp.asarray(target),
                    PackedFP8Layout(gathered.shape[0] // 33),
                )
            ),
            np.asarray(read_fp8_records(banks, jnp.asarray(source), layout)),
        )
        prepared = PreparedCandidates(gathered, *metadata[1:])
    if args.gather_sweep:
        from jax.experimental.pallas import tpu_sc as plsc

        from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
            tuned_block_sizes,
        )

        info = plsc.get_sparse_core_info()
        key = (min(args.block, args.page_size), len(banks))
        whole_pages = key in tuned_block_sizes.PAGED_MQA_WHOLE_PAGE_GATHER_CONFIG
        tuning = (
            tuned_block_sizes.PAGED_MQA_WHOLE_PAGE_GATHER_CONFIG
            if whole_pages
            else tuned_block_sizes.PAGED_MQA_GATHER_CONFIG
        )
        original = tuning[key]
        expected_keys = np.asarray(gathered).reshape(-1, key[0])[valid]
        for group in (32, 64, 96, 128, 192, 256):
            for buffers in (1, 2, 3, 4):
                words = (
                    layout.page_rows * 128
                    + layout.page_rows * 128 // key[0]
                    + len(banks)
                    if whole_pages
                    else 33 * (key[0] + len(banks))
                )
                scratch = buffers * group * words * 4
                if scratch > info.vmem_capacity_bytes * 0.9:
                    continue
                tuning[key] = (group, buffers)
                result = measure(
                    f"gather_g{group}_b{buffers}",
                    functools.partial(
                        gather_candidate_keys, block=key[0], layout=layout
                    ),
                    (banks, physical, metadata.offsets),
                    root,
                    args.repeats,
                    dimensions,
                )
                np.testing.assert_array_equal(
                    np.asarray(result).reshape(-1, key[0])[valid], expected_keys
                )
        tuning[key] = original
    live_blocks = int(np.count_nonzero(np.asarray(physical) >= 0))
    (root / "gather-traffic.json").write_text(
        json.dumps(
            {
                "valid_blocks": live_blocks,
                "valid_input_bytes": live_blocks
                * 33
                * min(args.block, args.page_size)
                * 4,
                "whole_page_input_bytes": live_blocks * layout.page_rows * 128 * 4,
                "allocated_output_bytes": prepared.keys.size * 4,
                "active_group_capacity_bytes": int(np.count_nonzero(np.diff(offsets)))
                * physical.shape[1]
                * min(args.block, args.page_size)
                * 33
                * 4,
            },
            indent=2,
        )
    )
    if args.stage == "gather":
        (root / "correctness.json").write_text(
            json.dumps(
                {
                    "membership_numpy": "passed",
                    "sampled_gather_readback": "passed",
                }
            )
        )
        return
    score = functools.partial(
        _score_prepared,
        topk=args.topk,
        block=args.block,
        unordered=True,
        width=args.width,
        grouped=True,
    )
    if args.stage == "score":

        def logits(
            q,
            weights,
            lengths,
            keys,
            canonical,
            membership,
            offsets,
            slots,
            output_slots,
        ):
            compact_lengths = jnp.sum(
                jnp.where(
                    canonical >= 0,
                    jnp.clip(lengths[:, None] - canonical * args.block, 0, args.block),
                    0,
                ),
                axis=1,
            )
            return indexer_impl.fp8_gathered_mqa_logits(
                q,
                keys,
                weights.astype(jnp.float32),
                compact_lengths,
                membership,
                query_start_loc=offsets,
                candidate_slots=slots,
                output_slots=output_slots,
                membership_block_size=args.block,
                max_model_len=canonical.shape[1] * args.block,
            )

        raw_scores = measure(
            "logits",
            logits,
            (
                q,
                weights,
                lengths,
                prepared.keys,
                prepared.block_ids,
                prepared.membership,
                prepared.offsets,
                prepared.candidate_slots,
                prepared.output_slots,
            ),
            root,
            args.repeats,
            dimensions,
        )
        masked_scores = measure(
            "membership_mask",
            lambda scores, allowed: jnp.where(
                jnp.repeat(allowed, args.block, axis=1), scores, -jnp.inf
            ),
            (raw_scores, prepared.candidates >= 0),
            root,
            args.repeats,
            dimensions,
        )
        split_selected = measure(
            "topk_and_token_mapping",
            functools.partial(
                indexer_impl.streaming_topk.select_candidate_tokens,
                k=min(args.topk, args.width, prepared.candidates.shape[1] * args.block),
                block_size=args.block,
                unordered=True,
                sorted_indices=False,
            ),
            (masked_scores, prepared.candidates),
            root,
            args.repeats,
            dimensions,
        )
    selected = measure(
        "score_topk",
        score,
        (q, weights, lengths, prepared),
        root,
        args.repeats,
        dimensions,
    )

    if args.stage == "score":
        np.testing.assert_array_equal(
            np.sort(np.asarray(split_selected), axis=1),
            np.sort(np.asarray(selected), axis=1),
        )
        (root / "correctness.json").write_text(
            json.dumps(
                {
                    "split_vs_combined_score_topk": "passed",
                }
            )
        )
        return

    def layer(q, weights, lengths, banks, pages, candidates, owners):
        return score(q, weights, lengths, prepare(banks, pages, candidates, owners))

    combined = measure(
        "layer",
        layer,
        (q, weights, lengths, banks, pages, candidates, owners),
        root,
        args.repeats,
        dimensions,
    )
    # Unordered selection may reorder equal-scoring keys; compare sets.
    np.testing.assert_array_equal(
        np.sort(np.asarray(selected), axis=1), np.sort(np.asarray(combined), axis=1)
    )
    (root / "correctness.json").write_text(
        json.dumps(
            {
                "membership_numpy": "passed",
                "membership_scan_vs_sort": "passed",
                "active_gather": "passed",
                "split_vs_combined_topk": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with jax.default_device(jax.devices("tpu")[0]):
        main()
