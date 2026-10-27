# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-free A/B of the decode Indexer stage schedule (L20 -> L24..L36).

One jit replays a steady DSpark decode step on one TPU core: the candidate
source (L20) scores the full packed FP8 cache and publishes 2048 blocks, three
intermediate layers each write the shared cache owner (SWA publication), and
four consumers score the gathered candidate keys. Calibrated matmul chains
stand in for the remaining TensorCore work, with the trace's data
dependencies: each Indexer reads the previous layer's output, attention waits
for its selection, and every layer writes the owner the gather reads.

Variants change only where the candidate gather runs and whether the prepared
consumer branches on short contexts. All variants must select identical IDs.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8_candidate_cache import (
    PreparedCandidates,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PAGE_RECORDS,
    PAGE_ROWS,
    PackedFP8Layout,
    pack_fp8_records,
    quantize_index_rows,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import fp8_indexer

REQUESTS, QUERIES, TOPK, BLOCK, CANDIDATES = 48, 6, 512, 8, 2048
WIDTH = 98816  # max_model_len // compress_ratio for the shared L20 cache
SWA_ROWS = 1024  # owner rows rewritten by each layer's SWA publication
# (early gather, branch-free prepared select)
VARIANTS = {
    "baseline": (False, False),
    "early_gather": (True, False),
    "no_cond": (False, True),
    "early_gather_no_cond": (True, True),
}
# Microseconds of TensorCore work between Indexer events, from exp-gcznvam0td.
GAPS_US = {"after_select": 30, "layer": 3400, "middle": 3550}


def fixture(seed):
    key = jax.random.split(jax.random.key(seed), 4)
    tokens = REQUESTS * QUERIES
    pages = REQUESTS * -(-WIDTH // PAGE_RECORDS)
    keys = jax.random.normal(key[0], (pages * PAGE_RECORDS, 128), jnp.bfloat16)
    records = pack_fp8_records(keys).reshape(pages, PAGE_RECORDS, PAGE_ROWS)
    words = records.transpose(0, 2, 1).reshape(pages * PAGE_ROWS, PAGE_RECORDS)
    split = pages // 2 * PAGE_ROWS
    banks = (
        words[:split],
        jnp.pad(words[split:], ((0, SWA_ROWS), (0, 0))),
    )
    rng = np.random.default_rng(seed)
    table = jnp.asarray(rng.permutation(pages).reshape(REQUESTS, -1), jnp.int32)
    q, scales, _ = quantize_index_rows(
        jax.random.normal(key[1], (tokens, 32, 128), jnp.bfloat16)
    )
    weights = jax.random.normal(key[2], (tokens, 32)) * 0.02 * scales
    lengths = jnp.tile(WIDTH - QUERIES + 1 + jnp.arange(QUERIES), REQUESTS)
    requests = jnp.repeat(jnp.arange(REQUESTS), QUERIES)
    inputs = (q, weights, lengths.astype(jnp.int32), requests.astype(jnp.int32))
    return banks, table, inputs, PackedFP8Layout(pages)


def tensor_work(h, weight, steps):
    return jax.lax.fori_loop(0, steps, lambda _, x: jnp.tanh(x @ weight), h)


def calibrate(h, weight, steps=400, repeats=5):
    run = jax.jit(tensor_work, static_argnums=2)
    run(h, weight, steps).block_until_ready()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run(h, weight, steps).block_until_ready()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) / steps * 1e6


def select_prepared(q, weights, lengths, prepared, branch_free):
    candidates = PreparedCandidates(*prepared)
    lengths = jnp.where(candidates.valid_queries, lengths, 0)
    ids = jnp.arange(min(TOPK, WIDTH), dtype=jnp.int32)
    short = jnp.where(ids[None, :] < lengths[:, None], ids[None, :], -1)

    def scored():
        return fp8_indexer._score_prepared(
            q, weights, lengths, candidates, TOPK, BLOCK, True, WIDTH, grouped=True
        )

    small = jnp.max(lengths) <= TOPK
    if branch_free:
        return jnp.where(small, short, scored())
    return jax.lax.cond(small, lambda: short, scored)


def stage(h, banks, table, inputs, weight, *, variant, layout, steps):
    early, branch_free = VARIANTS[variant]
    q, weights, lengths, requests = inputs
    layout_args = (layout.pages, layout.page_stride, layout.page_offset, PAGE_RECORDS)

    def tie(x, h):
        return jax.lax.optimization_barrier((x, h))[0]

    def publish_swa(banks, h):
        # Each layer rewrites part of the shared owner from its own output.
        rows = jax.lax.bitcast_convert_type(
            h[:SWA_ROWS, :256].reshape(SWA_ROWS, 128, 2), jnp.int32
        )
        low, high = banks
        start = high.shape[0] - SWA_ROWS
        return low, jax.lax.dynamic_update_slice(high, rows, (start, 0))

    def prepare(candidates):
        return fp8_indexer._prepare(
            banks, table, candidates, requests, layout_args, BLOCK, WIDTH
        )

    selected, candidates = fp8_indexer.select_indexer_jax(
        tie(q, h),
        banks,
        table,
        weights,
        lengths,
        jnp.full((q.shape[0], CANDIDATES), -1, jnp.int32),
        requests,
        jnp.max(lengths).reshape(1),
        layout_args,
        TOPK,
        BLOCK,
        True,
        False,
        True,
        WIDTH,
    )
    prepared = prepare(candidates) if early else None
    outputs = [selected]
    h = tensor_work(h, weight, steps["after_select"])
    h = tie(h, selected)
    for _ in range(3):
        h = tensor_work(h, weight, steps["middle"] // 3)
        banks = publish_swa(banks, h)
    for layer in range(4):
        if prepared is None:
            prepared = prepare(candidates)
        selected = select_prepared(tie(q, h), weights, lengths, prepared, branch_free)
        outputs.append(selected)
        h = tensor_work(h, weight, steps["after_select"])
        h = tie(h, selected)
        if layer < 3:
            h = tensor_work(h, weight, steps["layer"])
            banks = publish_swa(banks, h)
    return h, banks, jnp.stack(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    banks, table, inputs, layout = fixture(args.seed)
    h = jax.random.normal(jax.random.key(1), (1024, 2048), jnp.bfloat16)
    weight = jax.random.normal(jax.random.key(2), (2048, 2048), jnp.bfloat16) * 0.02
    step_us = calibrate(h, weight)
    steps = {name: max(1, round(us / step_us)) for name, us in GAPS_US.items()}
    # Same TensorCore work with no Indexer: the floor every variant approaches.
    total = steps["after_select"] * 5 + steps["middle"] // 3 * 3 + steps["layer"] * 3
    floor = jax.jit(lambda h: tensor_work(h, weight, total))

    def timed(fn, *operands, donate=False):
        samples, result = [], None
        for index in range(args.repeats + 3):
            start = time.perf_counter()
            result = fn(*operands)
            jax.block_until_ready(result)
            if index >= 3:
                samples.append((time.perf_counter() - start) * 1e3)
            if donate:
                operands = (operands[0], result[1], *operands[2:])
        return samples, result

    floor_ms, _ = timed(floor, h)
    rows, reference = [], None
    for variant in args.variants:
        run = jax.jit(
            lambda h, banks, table, inputs, weight, variant=variant: stage(
                h,
                banks,
                table,
                inputs,
                weight,
                variant=variant,
                layout=layout,
                steps=steps,
            ),
            donate_argnums=1,
        )
        start = time.perf_counter()
        compiled = run.lower(h, banks, table, inputs, weight).compile()
        compile_s = time.perf_counter() - start
        samples, result = timed(compiled, h, banks, table, inputs, weight, donate=True)
        banks = result[1]
        selected = np.asarray(result[2])
        reference = selected if reference is None else reference
        if args.profile:
            with jax.profiler.trace(str(args.out / "profile" / variant)):
                for _ in range(3):
                    result = compiled(h, banks, table, inputs, weight)
                    banks = result[1]
                    jax.block_until_ready(result)
        row = {
            "variant": variant,
            "stage_ms": statistics.median(samples),
            "stage_p10_ms": float(np.percentile(samples, 10)),
            "stage_p90_ms": float(np.percentile(samples, 90)),
            "indexer_overhead_ms": statistics.median(samples)
            - statistics.median(floor_ms),
            "compile_s": compile_s,
            "matches_baseline": bool(np.array_equal(selected, reference)),
        }
        rows.append(row)
        print("STAGE", json.dumps(row), flush=True)
    summary = {
        "jax": jax.__version__,
        "device": str(jax.devices()[0].device_kind),
        "step_us": step_us,
        "steps": steps,
        "floor_ms": statistics.median(floor_ms),
        "variants": rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)
    if not all(row["matches_baseline"] for row in rows):
        raise SystemExit("variants selected different indices")


if __name__ == "__main__":
    main()
