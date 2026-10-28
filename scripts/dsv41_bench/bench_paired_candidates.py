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
    candidate_membership,
    gather_candidate_keys,
    prepare_candidates,
    resolve_candidate_blocks,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PackedFP8Layout,
    quantize_index_rows,
    read_fp8_records,
)
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
    # Replay the real gather with resolved physical IDs from the same union.
    offsets = np.asarray(prepared.offsets)
    reps = np.minimum(offsets[:-1], tokens - 1)
    group_ids = prepared.block_ids[reps]
    group_ids = jnp.where(jnp.asarray(np.diff(offsets) > 0)[:, None], group_ids, -1)
    physical = resolve_candidate_blocks(
        group_ids,
        pages[owners[reps]],
        sparse_block_kv=args.block,
        cache_pages=layout.pages,
        page_size=layout.page_size,
    )
    gathered = measure(
        "gather",
        functools.partial(
            gather_candidate_keys,
            block=min(args.block, args.page_size),
            layout=layout,
        ),
        (banks, physical, prepared.offsets),
        root,
        args.repeats,
        dimensions,
    )
    # Only valid blocks are initialized; skipped padding is never consumed.
    per_page = 128 // min(args.block, args.page_size)
    valid = np.asarray(physical >= 0).reshape(-1, 1, per_page)
    valid = np.broadcast_to(valid, (valid.shape[0], 33, per_page)).reshape(-1)
    np.testing.assert_array_equal(
        np.asarray(gathered).reshape(-1, min(args.block, args.page_size))[valid],
        np.asarray(prepared.keys).reshape(-1, min(args.block, args.page_size))[valid],
    )
    score = functools.partial(
        _score_prepared,
        topk=args.topk,
        block=args.block,
        unordered=True,
        width=args.width,
        grouped=True,
    )
    selected = measure(
        "score_topk",
        score,
        (q, weights, lengths, prepared),
        root,
        args.repeats,
        dimensions,
    )

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
