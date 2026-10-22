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
"""DP/EP MoE dispatch of tokens that were routed on the rank that owns them.

The naive DP/EP dispatch all-gathers every rank's bf16 hidden states and router
logits, and every rank then runs the same top-k over all gathered tokens. Top-k,
its weights and the routed kernel's lhs quantization are all row-local, so each
rank can compute them for its own tokens first and gather only the result: one
all-gather of the rows the expert kernel reads plus their expert ids and
weights.

A dispatched row is the hidden size in the hidden dtype, then `META_LANES` lanes
of metadata in the same dtype: the routed kernel's lhs block scales from lane 0
(zero when the batch is not quantized), then each route's expert id as its low
and high byte, then each route's weight. Small integers are exact in bf16, so
the ids survive the hidden dtype.
"""

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.megablox.moe_routed_ffn import (
    LHS_SCALE_LANES,
    quantize_rows,
)

# Move a routed-kernel batch as quantized rows; False moves bf16 rows always.
QUANTIZE_ROUTED_ROWS = True
META_LANES = LHS_SCALE_LANES
# Metadata lanes reserved for the lhs block scales; the routes follow.
SCALE_LANES = 32
# An expert id travels as two values below this, each exact in bf16.
_ID_BASE = 256


def max_dispatch_topk() -> int:
    return (META_LANES - SCALE_LANES) // 3


def _meta(
    scales: jax.Array | None,  # [T, blocks]
    topk_ids: jax.Array,  # int32[T, topk]: global expert ids
    topk_weights: jax.Array,  # [T, topk]
    dtype: jnp.dtype,
) -> jax.Array:
    num_tokens, topk = topk_ids.shape
    if topk > max_dispatch_topk():
        raise ValueError(f"cannot dispatch top-{topk} routes in {META_LANES} lanes")
    ids = topk_ids.astype(jnp.int32)
    if scales is None:
        scales = jnp.zeros((num_tokens, 0), dtype)
    parts = [
        scales.astype(dtype),
        jnp.zeros((num_tokens, SCALE_LANES - scales.shape[1]), dtype),
        (ids % _ID_BASE).astype(dtype),
        (ids // _ID_BASE).astype(dtype),
        topk_weights.astype(dtype),
        jnp.zeros((num_tokens, META_LANES - SCALE_LANES - 3 * topk), dtype),
    ]
    return jnp.concatenate(parts, axis=1)


def split_dispatched(
    rows: jax.Array, hidden: int, topk: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A dispatched batch's hidden rows, int32 expert ids and routing weights."""
    meta = rows[:, hidden:]
    lanes = [
        meta[:, SCALE_LANES + i * topk : SCALE_LANES + (i + 1) * topk] for i in range(3)
    ]
    low, high, weights = lanes
    ids = high.astype(jnp.int32) * _ID_BASE + low.astype(jnp.int32)
    return rows[:, :hidden], ids, weights


def _pack(parts: list[jax.Array], bits: int) -> jax.Array:
    """Equal [T, N] arrays of `bits`-bit values as int32 words: part i in bits
    [i * bits, (i + 1) * bits) of word [t, n]. Lane-aligned, so the packing
    and unpacking are elementwise."""
    uint = jnp.uint8 if bits == 8 else jnp.uint16
    word = jnp.zeros(parts[0].shape, jnp.uint32)
    for i, part in enumerate(parts):
        word |= lax.bitcast_convert_type(part, uint).astype(jnp.uint32) << (i * bits)
    return lax.bitcast_convert_type(word, jnp.int32)


def _unpack(words: jax.Array, count: int, bits: int, dtype: jnp.dtype) -> jax.Array:
    """`_pack`'s parts, concatenated along the last axis."""
    uint = jnp.uint8 if bits == 8 else jnp.uint16
    words = lax.bitcast_convert_type(words, jnp.uint32)
    parts = [
        lax.bitcast_convert_type((words >> (i * bits)).astype(uint), dtype)
        for i in range(count)
    ]
    return jnp.concatenate(parts, axis=1)


def _pack_words(x_q: jax.Array, meta: jax.Array) -> jax.Array:
    """[T, H] 8-bit rows and [T, META_LANES] 16-bit metadata as int32 words."""
    rows = jnp.split(x_q, 4, axis=1)
    return jnp.concatenate(
        [_pack(rows, 8), _pack(jnp.split(meta, 2, axis=1), 16)], axis=1
    )


def _unpack_words(
    words: jax.Array, hidden: int, q_dtype: jnp.dtype, dtype: jnp.dtype
) -> jax.Array:
    x_q = _unpack(words[:, : hidden // 4], 4, 8, q_dtype)
    meta = _unpack(words[:, hidden // 4 :], 2, 16, dtype)
    return jnp.concatenate([x_q.astype(dtype), meta], axis=1)


def dispatch_local_tokens(
    hidden_states: jax.Array,  # [T, H]: this rank's tokens
    topk_ids: jax.Array,  # int32[T, topk]: their global expert ids
    topk_weights: jax.Array,  # [T, topk]
    *,
    lhs_quant: tuple[jnp.dtype, int] | None,
    axis_name: str,
) -> jax.Array:
    """Every rank's dispatched rows, rank-major: [T * ranks, H + META_LANES].

    The body of a shard_map over `axis_name`. With `lhs_quant`, `quantize_rows`'s
    (q_dtype, block), the rows are its values and scales, moved as 8-bit words,
    for `moe_routed_ffn(..., prequantized=True)`; otherwise the bf16 rows.
    """
    dtype = hidden_states.dtype
    if lhs_quant is None:
        meta = _meta(None, topk_ids, topk_weights, dtype)
        rows = jnp.concatenate([hidden_states, meta], axis=1)
        return lax.all_gather(rows, axis_name, axis=0, tiled=True)
    q_dtype, block = lhs_quant
    if jax.dtypes.itemsize_bits(q_dtype) != 8 or jax.dtypes.itemsize_bits(dtype) != 16:
        raise ValueError(f"cannot pack {q_dtype} rows with {dtype} metadata")
    x_q, scales = quantize_rows(hidden_states, q_dtype=q_dtype, block=block)
    words = _pack_words(x_q, _meta(scales, topk_ids, topk_weights, dtype))
    words = lax.all_gather(words, axis_name, axis=0, tiled=True)
    return _unpack_words(words, hidden_states.shape[1], q_dtype, dtype)
