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
"""Word-level views of the served DeepSeek-V4 attention caches.

The megakernel attention reads every cache as 32-bit words with 128 lanes,
the same bytes the serving kernels see through their uint8 views:

  bf16 latent token (SWA, HCA)  2 rows: word ``w`` of row ``s`` holds bf16
      ``256 s + w`` in its low half and bf16 ``256 s + 128 + w`` in its high
      half (the byte order `mla_swa.prepare_kv_inputs` writes).
  CSA NoPE token                1 row: byte ``b`` of word ``w`` is record byte
      ``128 b + w``; bytes 0..447 are e4m3 values of dims 0..447 and bytes
      448..511 the e8m0 scale of dims ``d % 64`` (`sparse_mla._dequant_dsv4_fp8`).
  CSA RoPE                      4 tokens per row, see `csa_cache_layout`.
  indexer K, seq-along-lane     per page ``[groups, page]`` words: byte ``i``
      of word ``(g, t)`` is e4m3 key dim ``4 g + i`` of token ``t``; group
      ``head_dim // 4`` holds the e8m0 scale in byte 0.

The functions below are pure jnp, so the kernels and the plain-JAX reference
share one decoder.
"""

import jax
import jax.numpy as jnp
from jax import lax

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
NOPE_SCALE_PERIOD = 64
LANES = 128
ROPE_WORDS = 32

_HI16 = -65536  # 0xFFFF0000 as int32


def _f32(words: jax.Array) -> jax.Array:
    return lax.bitcast_convert_type(words, jnp.float32)


def bf16_tokens_from_words(even: jax.Array, odd: jax.Array, axis: int = -1) -> jax.Array:
    """int32 ``[n, 128]`` rows 0 and 1 of n tokens -> bf16 ``[n, 512]``.

    With ``axis=0`` the inputs are transposed (``[128, n]``) and so is the result.
    """
    parts = (even << 16, even & _HI16, odd << 16, odd & _HI16)
    return jnp.concatenate([_f32(p) for p in parts], axis=axis).astype(jnp.bfloat16)


def bf16_tokens_to_words(x: jax.Array) -> jax.Array:
    """bf16 ``[..., 512]`` -> int32 ``[..., 2, 128]`` (inverse of the above)."""
    u = lax.bitcast_convert_type(x, jnp.uint16).astype(jnp.uint32)
    u = u.reshape(*x.shape[:-1], 2, 2, LANES)
    return lax.bitcast_convert_type(u[..., 0, :] | (u[..., 1, :] << 16), jnp.int32)


def _byte(words: jax.Array, b: int) -> jax.Array:
    return (words >> (8 * b)) & 0xFF


def e4m3_from_byte(byte: jax.Array) -> jax.Array:
    """int32 holding an e4m3 code in its low byte -> f32 value."""
    return lax.bitcast_convert_type(byte.astype(jnp.uint8), jnp.float8_e4m3fn).astype(
        jnp.float32
    )


def e8m0_from_byte(byte: jax.Array) -> jax.Array:
    """int32 holding an e8m0 code in its low byte -> f32 ``2^(code - 127)``.

    Exact for every code the quantizers write (1..254).
    """
    return _f32(byte << 23)


def nope_tokens_from_words(words: jax.Array) -> jax.Array:
    """int32 ``[n, 128]`` CSA NoPE rows -> bf16 ``[n, 448]`` dequantized dims.

    Every product is exact in bf16 (3-bit mantissa times a power of two), so
    this matches `sparse_mla._dequant_dsv4_fp8` bit for bit.
    """
    scale_lanes = _byte(words, 3)[..., NOPE_SCALE_PERIOD:]
    scale = e8m0_from_byte(jnp.concatenate([scale_lanes, scale_lanes], axis=-1))
    blocks = [e4m3_from_byte(_byte(words, b)) * scale for b in range(4)]
    return jnp.concatenate(blocks, axis=-1)[..., :NOPE_DIM].astype(jnp.bfloat16)


def rope_tokens_from_words(words: jax.Array) -> jax.Array:
    """int32 ``[n, 32]`` per-token RoPE words -> bf16 ``[n, 64]``."""
    return jnp.concatenate([_f32(words << 16), _f32(words & _HI16)], axis=-1).astype(
        jnp.bfloat16
    )


def monotone_key(scores: jax.Array) -> jax.Array:
    """f32 -> int32 key with the same order (``-0.0 < +0.0``), as the SparseCore
    top-k compares."""
    bits = lax.bitcast_convert_type(scores, jnp.int32)
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)
