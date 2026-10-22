# SPDX-License-Identifier: Apache-2.0
"""Private read-only Torch bridges for scheduled-cache NaN diagnostics."""

import functools
import itertools
import math

import jax
import jax.numpy as jnp
from torch_tpu._internal import pallas

from vllm_torchtpu.block_major_pool import kernel_view
from vllm_torchtpu.kernels.kv_cache_nan_flags import (
    attention_cache_has_nan,
    dense_state_has_nan,
    pooled_state_has_nan,
)
from vllm_torchtpu.layers.adapter.attention import _reshape_packed_kv_cache
from vllm_torchtpu.layers.core.gdn_attention import _build_v3_pool_state_plan

_op_ids = itertools.count()


def inspect_attention_cache(
    cache,
    tables,
    seq_lens,
    starts,
    *,
    native_shape,
    num_kv_heads,
    head_size,
    seq_along_lane=False,
    block_major=False,
):
    """Normalize only a selected page; never transpose/copy the entire pool."""
    if block_major:
        cache = kernel_view(cache)
    page_size = native_shape[-1] if seq_along_lane else native_shape[1]
    page_stride = (
        math.prod(cache.shape[1:]) // math.prod(native_shape[1:])
        if cache.ndim == 4
        else 1
    )

    def read_page(index):
        if cache.ndim == 4:
            raw = cache[index // page_stride][None]
            packed_shape = (1, raw.shape[1], 1, native_shape[3], raw.shape[-1])
            pages = _reshape_packed_kv_cache(
                raw.reshape(packed_shape), (page_stride, *native_shape[1:])
            )
            page = pages[index % page_stride]
        else:
            page = cache[index]
        if seq_along_lane:
            page = page.transpose(3, 0, 1, 2).reshape(page_size, page.shape[0], -1)
        else:
            page = page.reshape(page_size, -1, page.shape[-1])
        # hd64 stores one head's K and V together in its 128-wide lane.
        heads, width = (
            (num_kv_heads, 128) if head_size == 64 else (2 * num_kv_heads, head_size)
        )
        return page[:, :heads, :width]

    return attention_cache_has_nan(
        cache,
        tables * page_stride,
        seq_lens,
        starts,
        page_size=page_size,
        read_page=read_page,
    )


@functools.cache
def attention_nan_op(
    native_shape, num_kv_heads, head_size, seq_along_lane, block_major
):
    def inspect(
        cache: jax.Array,
        tables: jax.Array,
        seq_lens: jax.Array,
        starts: jax.Array,
        armed: jax.Array,
    ) -> jax.Array:
        return jax.lax.cond(
            armed[0],
            lambda: inspect_attention_cache(
                cache,
                tables,
                seq_lens,
                starts,
                native_shape=native_shape,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                seq_along_lane=seq_along_lane,
                block_major=block_major,
            ),
            lambda: jnp.array(False),
        )

    return pallas.jax_op(f"pallas::kv_cache_nan_attention_{next(_op_ids)}", inspect)


@functools.cache
def dense_state_nan_op():
    def inspect(
        conv: jax.Array,
        ssm: jax.Array,
        slots: jax.Array,
        seq_lens: jax.Array,
        starts: jax.Array,
        armed: jax.Array,
        offsets: jax.Array | None,
    ) -> jax.Array:
        return jax.lax.cond(
            armed[0],
            lambda: dense_state_has_nan((conv, ssm), slots, seq_lens, starts, offsets),
            lambda: jnp.array(False),
        )

    return pallas.jax_op("pallas::kv_cache_nan_dense_state", inspect)


@functools.cache
def pooled_state_nan_op(
    conv_dim, n_v, d_k, d_v, kernel_size, block_tokens, ssm_dtype, block_major
):
    def inspect(
        cache: jax.Array,
        slots: jax.Array,
        seq_lens: jax.Array,
        starts: jax.Array,
        armed: jax.Array,
        offsets: jax.Array | None,
        ckpts: jax.Array | None,
    ) -> jax.Array:
        pool = kernel_view(cache) if block_major else cache
        plan = _build_v3_pool_state_plan(
            pool,
            conv_dim=conv_dim,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            pool_block_tokens=block_tokens,
            # Row permutations preserve any-NaN, including the live-row set.
            qk_pair_layout=False,
            recurrent_state_dtype=ssm_dtype,
        )
        return jax.lax.cond(
            armed[0],
            lambda: pooled_state_has_nan(
                pool, slots, seq_lens, starts, offsets, ckpts, plan=plan
            ),
            lambda: jnp.array(False),
        )

    return pallas.jax_op(f"pallas::kv_cache_nan_pooled_state_{next(_op_ids)}", inspect)
