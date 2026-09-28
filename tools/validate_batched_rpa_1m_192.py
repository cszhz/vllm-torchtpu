"""Smoke-test one 1 Mi-token request in a 192-slot batched RPA bucket."""

import argparse
import time

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-tokens", type=int, default=4096)
    parser.add_argument("--replicas", type=int, choices=(1, 8), default=1)
    parser.add_argument("--history-tokens", type=int, default=1024 * 1024)
    parser.add_argument("--max-num-seqs", type=int, default=192)
    parser.add_argument("--force-legacy", action="store_true")
    args = parser.parse_args()

    history = args.history_tokens
    page_size = 256
    max_num_seqs = args.max_num_seqs
    pages_per_seq = (history + args.new_tokens + page_size - 1) // page_size
    dtype = jnp.dtype("float8_e4m3fn")
    cache_shape = wrapper.get_kv_cache_shape(
        pages_per_seq,
        page_size,
        2,
        256,
        dtype,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )

    def initialize(_):
        q = jnp.ones((args.new_tokens, 32, 256), jnp.bfloat16)
        k = jnp.ones((args.new_tokens, 2, 256), dtype)
        v = jnp.ones((args.new_tokens, 2, 256), dtype)
        cache = jnp.zeros(cache_shape, dtype)
        kv_lens = (
            jnp.zeros((max_num_seqs,), jnp.int32).at[0].set(history + args.new_tokens)
        )
        page_indices = jnp.zeros((max_num_seqs, pages_per_seq), jnp.int32)
        page_indices = (
            page_indices.at[0].set(jnp.arange(pages_per_seq, dtype=jnp.int32)).ravel()
        )
        cu_q_lens = jnp.full((max_num_seqs + 1,), args.new_tokens, jnp.int32)
        cu_q_lens = cu_q_lens.at[0].set(0)
        distribution = jnp.array([0, 0, max_num_seqs], jnp.int32)
        return q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution

    def run(q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution):
        return wrapper.ragged_paged_attention(
            q,
            k,
            v,
            cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=256**-0.5,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            stream_page_indices=False if args.force_legacy else None,
        )

    if args.replicas == 8:
        if jax.local_device_count() != 8:
            raise RuntimeError("Eight local TPU devices are required")
        inputs = jax.pmap(initialize)(jnp.arange(8, dtype=jnp.int32))
        execute = jax.pmap(run)
    else:
        inputs = initialize(None)
        execute = run

    jax.block_until_ready(inputs)
    start = time.perf_counter()
    out, new_cache = execute(*inputs)
    jax.block_until_ready((out, new_cache))
    finite = bool(jnp.all(jnp.isfinite(out)))
    first_new_page = (
        new_cache[:, history // page_size]
        if args.replicas == 8
        else new_cache[history // page_size]
    )
    written = bool(jnp.all(first_new_page == jnp.asarray(1, dtype)))
    if not finite or not written:
        raise AssertionError(f"Invalid result: {finite=}, {written=}")
    print(
        f"PASS: {history} history + {args.new_tokens} new tokens, "
        f"{max_num_seqs} slots per replica, {args.replicas} replicas, "
        f"{pages_per_seq} pages/slot; "
        f"compile and first run {time.perf_counter() - start:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
