# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B the Index-K write (produce) path for packed-FP8 prefill.

The published-K producer turns the compressor's pre-RoPE ``wk(latent)`` output
into packed FP8 Index-K records. Two implementations are compared on identical
inputs:

* ``baseline`` -- the eager chain: ``k_norm`` (weighted RMSNorm -> bf16), a
  standalone RoPE kernel, ``quantize_index_rows`` + ``pack_fp8_records``, then
  the packed-page scatter. This is what ``_produce_index_k`` ran before the
  fused producer, and it materializes the whole D128 key tensor in HBM between
  each stage.
* ``fused`` -- the ``knorm_rope_quant_pack`` kernel does norm -> RoPE -> E4M3
  quant -> bit-pack in one pass, emitting the ``[tokens, 33]`` record directly
  for the same scatter. No intermediate RoPE/quant HBM round trips.

Both paths share the identical scatter (``write_packed_records``), so the A/B
isolates exactly the fused-vs-staged produce cost -- the prefill regression
this kernel targets. Correctness is the gate: the two banks must come out
byte-for-byte identical, and the fused records must match the eager pack.

Run with the installed serving torch-tpu/JAX/libtpu, same as the other
``bench_*`` scripts. Pure JAX; no torch.distributed harness.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.rope import knorm_rope_quant_pack
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PAGE_RECORDS,
    PackedFP8Layout,
    pack_fp8_records,
    write_packed_records,
)

HEAD_DIM = 128  # index_head_dim
ROTARY_DIM = 64  # qk_rope_head_dim
EPS = 1e-6


def _make_cos_sin_cache(max_position: int, rotary_dim: int, seed: int) -> np.ndarray:
    """The ``[cos | sin]`` layout DeepSeek-V4 builds, packed side by side."""
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.0, 2.0 * np.pi, (max_position, rotary_dim // 2)).astype(
        np.float32
    )
    return np.concatenate([np.cos(freqs), np.sin(freqs)], axis=-1).astype(np.float32)


def _rope_ref(x, positions, cos_sin_cache):
    """GPT-J interleaved RoPE over the trailing ``rotary_dim`` channels (fp32)."""
    rotary_dim = cos_sin_cache.shape[1]
    xf = x.astype(jnp.float32)
    cos, sin = jnp.split(cos_sin_cache[positions].astype(jnp.float32), 2, axis=-1)
    cos = jnp.repeat(cos, 2, axis=-1)
    sin = jnp.repeat(sin, 2, axis=-1)
    x_pass, x_rot = xf[..., :-rotary_dim], xf[..., -rotary_dim:]
    rotated = jnp.stack([-x_rot[..., 1::2], x_rot[..., 0::2]], axis=-1).reshape(
        x_rot.shape
    )
    out_rot = x_rot * cos + rotated * sin
    return jnp.concatenate([x_pass, out_rot], axis=-1)


def _eager_pack(latent, positions, cos_sin_cache, weight):
    """The staged chain: weighted RMSNorm -> bf16 -> RoPE -> bf16 -> pack."""
    xf = latent.astype(jnp.float32)
    rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + EPS)
    keys = (xf * rms * weight.astype(jnp.float32)).astype(jnp.bfloat16)
    keys = _rope_ref(keys, positions, cos_sin_cache).astype(jnp.bfloat16)
    return pack_fp8_records(keys)  # [tokens, 33] int32


def _fused_pack(latent, positions, cos_sin_cache, weight):
    """The fused producer: one kernel to the [tokens, 33] record."""
    words, scales = knorm_rope_quant_pack(
        latent, positions, cos_sin_cache, weight, eps=EPS
    )
    return jnp.concatenate(
        (words, jax.lax.bitcast_convert_type(scales, jnp.int32)[:, None]), axis=1
    )


def _make_banks(num_pages, page_size):
    rows = PackedFP8Layout(num_pages, page_size=page_size).page_rows * num_pages
    return (jnp.zeros((rows, PAGE_RECORDS), jnp.int32),)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[512, 2048, 8192],
        help="Prefill token counts to sweep (keys written per step).",
    )
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--max-position", type=int, default=65536)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["baseline", "fused"],
        default=["baseline", "fused"],
    )
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assert jax.default_backend() == "tpu"
    assert args.modes[0] == "baseline"
    assert args.page_size % 16 == 0 and PAGE_RECORDS % args.page_size == 0

    runtime = {
        name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "libtpu")
    }
    print("RUNTIME", json.dumps(runtime), flush=True)

    packers = {"baseline": _eager_pack, "fused": _fused_pack}
    metrics = []
    for tokens in args.tokens:
        # One contiguous destination page run; slots are the logical records.
        num_pages = (tokens + args.page_size - 1) // args.page_size + 4
        rng = np.random.default_rng(42)
        latent = jnp.asarray(
            rng.normal(size=(tokens, HEAD_DIM)).astype(np.float32) * 0.25,
            jnp.bfloat16,
        )
        weight = jnp.asarray(
            rng.normal(size=(HEAD_DIM,)).astype(np.float32) * 0.1, jnp.bfloat16
        )
        positions = jnp.asarray(
            rng.integers(0, args.max_position, size=(tokens,)).astype(np.int32)
        )
        slots = jnp.asarray(np.arange(tokens, dtype=np.int32))
        cos_sin_cache = jnp.asarray(
            _make_cos_sin_cache(args.max_position, ROTARY_DIM, 0)
        )
        layout = PackedFP8Layout(num_pages, page_size=args.page_size)
        layout_args = (
            layout.pages,
            layout.page_stride,
            layout.page_offset,
            layout.page_size,
        )

        def produce(
            latent,
            positions,
            weight,
            slots,
            banks,
            *,
            packer,
            cos_sin_cache=cos_sin_cache,
            layout_args=layout_args,
        ):
            values = packer(latent, positions, cos_sin_cache, weight)
            return write_packed_records(
                banks, slots, values, PackedFP8Layout(*layout_args)
            )

        compiled, outputs, compile_seconds = {}, {}, {}
        for mode in args.modes:
            fn = jax.jit(
                lambda la, po, we, sl, ba, packer=packers[mode]: produce(
                    la, po, we, sl, ba, packer=packer
                )
            )
            compiled[mode] = fn

        inputs = (latent, positions, weight, slots)
        for mode in args.modes:
            banks = _make_banks(num_pages, args.page_size)
            started = time.perf_counter()
            result = jax.block_until_ready(compiled[mode](*inputs, banks))
            compile_seconds[mode] = time.perf_counter() - started
            outputs[mode] = [np.asarray(b) for b in result]
        # Correctness gate: fused banks must be byte-for-byte the eager banks.
        for mode in args.modes[1:]:
            for got, ref in zip(outputs[mode], outputs["baseline"]):
                np.testing.assert_array_equal(got, ref)
        print("VERIFIED", tokens, flush=True)

        samples = {mode: [] for mode in args.modes}
        banks_by_mode = {m: _make_banks(num_pages, args.page_size) for m in args.modes}
        for iteration in range(args.repeats):
            offset = iteration % len(args.modes)
            order = args.modes[offset:] + args.modes[:offset]
            for mode in order:
                started = time.perf_counter_ns()
                jax.block_until_ready(compiled[mode](*inputs, banks_by_mode[mode]))
                samples[mode].append((time.perf_counter_ns() - started) / 1e6)
        for mode in args.modes:
            metric = dict(
                variant=mode,
                runtime=runtime,
                timing="synchronized_host_latency",
                tokens=tokens,
                page_size=args.page_size,
                compile_seconds=compile_seconds[mode],
                latency_ms=float(np.median(samples[mode])),
                samples_ms=samples[mode],
                correct=True,
            )
            metrics.append(metric)
            print(
                json.dumps({k: v for k, v in metric.items() if k != "samples_ms"}),
                flush=True,
            )
            with (args.out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metric) + "\n")
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if args.profile:
            for mode in args.modes:
                with jax.profiler.trace(str(args.out / "xprof" / f"{mode}-{tokens}")):
                    for _ in range(5):
                        jax.block_until_ready(
                            compiled[mode](*inputs, banks_by_mode[mode])
                        )

    base = {m["tokens"]: m for m in metrics if m["variant"] == "baseline"}
    for m in metrics:
        if m["variant"] == "fused":
            b = base[m["tokens"]]["latency_ms"]
            speedup = b / m["latency_ms"] if m["latency_ms"] else float("nan")
            print(
                f"SPEEDUP tokens={m['tokens']} baseline={b:.4f}ms "
                f"fused={m['latency_ms']:.4f}ms x{speedup:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
