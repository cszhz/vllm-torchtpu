"""Benchmark one Qwen3.5 dense-attention chunk on eight independent TPU replicas.

This measures the batched RPA operator, including metadata generation and KV
writeback. It does not load model weights or benchmark the full model.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

MODEL_CONFIG_URL = (
    "https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8/raw/main/config.json"
)
HISTORY_LENGTHS = tuple(1024 * value for value in (4, 8, 16, 32, 64, 128, 256, 512))
PAGE_SIZE = 256
NEW_TOKENS = 4096
KV_DTYPE = jnp.dtype("float8_e4m3fn")
Q_DTYPE = jnp.dtype("bfloat16")
LAYOUT = configs.KVLayout.SEQ_ALONG_LANE


def load_config(path: Path | None) -> tuple[dict, str]:
    if path is None:
        with urllib.request.urlopen(MODEL_CONFIG_URL, timeout=30) as response:
            raw = response.read()
    else:
        raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def benchmark_case(
    history: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    warmups: int,
    samples: int,
) -> dict:
    pages = (history + NEW_TOKENS + PAGE_SIZE - 1) // PAGE_SIZE
    cache_shape = wrapper.get_kv_cache_shape(
        pages, PAGE_SIZE, kv_heads, head_dim, KV_DTYPE, kv_layout=LAYOUT
    )
    n_devices = jax.local_device_count()

    @jax.pmap
    def initialize(_):
        return (
            jnp.ones((NEW_TOKENS, heads, head_dim), Q_DTYPE),
            jnp.ones((NEW_TOKENS, kv_heads, head_dim), KV_DTYPE),
            jnp.ones((NEW_TOKENS, kv_heads, head_dim), KV_DTYPE),
            jnp.zeros(cache_shape, KV_DTYPE),
            jnp.array([history + NEW_TOKENS], jnp.int32),
            jnp.arange(pages, dtype=jnp.int32),
            jnp.array([0, NEW_TOKENS], jnp.int32),
            jnp.array([0, 0, 1], jnp.int32),
        )

    @jax.pmap
    def run(q, k, v, cache, lengths, page_indices, cu_q_lens, distribution):
        return wrapper.ragged_paged_attention(
            q,
            k,
            v,
            cache,
            lengths,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=head_dim**-0.5,
            kv_layout=LAYOUT,
        )

    args = list(initialize(jnp.arange(n_devices, dtype=jnp.int32)))
    jax.block_until_ready(args)
    compile_start = time.perf_counter()
    output, args[3] = run(*args)
    jax.block_until_ready((output, args[3]))
    compile_seconds = time.perf_counter() - compile_start
    for _ in range(warmups):
        output, args[3] = run(*args)
        jax.block_until_ready((output, args[3]))
    durations = []
    for _ in range(samples):
        start = time.perf_counter()
        output, args[3] = run(*args)
        jax.block_until_ready((output, args[3]))
        durations.append((time.perf_counter() - start) * 1000)
    return {
        "history_tokens": history,
        "new_tokens": NEW_TOKENS,
        "pages_per_replica": pages,
        "kv_cache_mib_per_replica": round(
            KV_DTYPE.itemsize * math.prod(cache_shape) / 1024**2, 2
        ),
        "compile_and_first_run_s": round(compile_seconds, 3),
        "samples_ms": [round(value, 3) for value in durations],
        "median_ms": round(statistics.median(durations), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    config, digest = load_config(args.config)
    text_config = config["text_config"]
    heads = text_config["num_attention_heads"]
    kv_heads = text_config["num_key_value_heads"]
    head_dim = text_config["head_dim"]
    if jax.local_device_count() != 8:
        raise RuntimeError("This benchmark requires exactly eight local TPU devices")
    result = {
        "model": "Qwen/Qwen3.5-397B-A17B-FP8",
        "config_sha256": digest,
        "model_max_position_embeddings": text_config["max_position_embeddings"],
        "geometry": {"q_heads": heads, "kv_heads": kv_heads, "head_dim": head_dim},
        "kv_dtype": str(KV_DTYPE),
        "q_dtype": str(Q_DTYPE),
        "kv_layout": "HND",
        "page_size": PAGE_SIZE,
        "dp_replicas": 8,
        "active_requests_per_replica": 1,
        "warmups": args.warmups,
        "samples": args.samples,
        "device_kind": jax.devices()[0].device_kind,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for history in HISTORY_LENGTHS:
        try:
            case = benchmark_case(
                history,
                heads,
                kv_heads,
                head_dim,
                args.warmups,
                args.samples,
            )
            print(f"{history // 1024}K: {case['median_ms']:.3f} ms", flush=True)
        except Exception as exc:
            case = {"history_tokens": history, "error": repr(exc)}
            print(f"{history // 1024}K: {case['error']}", flush=True)
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
