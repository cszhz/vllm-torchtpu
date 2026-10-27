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

import math
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# bf16 tiles are (16, 128); keep token blocks sublane-aligned.
SUBLANE = 16

# Explicit scoped-VMEM budget, repo convention (mla/v1, deepseek_v4 and
# fused_moe use the same constant). Never rely on the backend default —
# it varies by Mosaic version.
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

# Tokens whose post/comb gates the kernels' epilogue computes together,
# tokens on lanes. Wider groups amortize the epilogue's serial Sinkhorn
# chain over more lanes (deviceless tpu7x LLO at T=1024: 8x565 bundles
# per call at 128, 4x972 at 256).
GATE_GROUP_TOKENS = 256


def round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def select_token_block(
    num_tokens: int,
    token_block_size: int,
    *,
    vmem_need: Callable[[int], int] | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[int, int]:
    """Grid token block, and the token count padded to a multiple of it.

    ``vmem_need(tb)`` estimates a kernel's VMEM footprint for a block of
    ``tb`` tokens. When supplied, the block halves until it fits within
    ``vmem_limit_bytes`` — degrading to a smaller block instead of a
    compile-time VMEM OOM — with ``SUBLANE`` as the floor.

    Returns (token_block, padded_tokens).
    """
    tb = min(token_block_size, round_up(num_tokens, SUBLANE))
    if vmem_need is not None:
        while tb > SUBLANE and vmem_need(tb) > vmem_limit_bytes:
            tb //= 2
    return tb, round_up(num_tokens, tb)


def pad_to(padded_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
    """Zero-pad the leading (token) axis of each array to ``padded_tokens``."""
    pad = padded_tokens - arrays[0].shape[0]
    if pad == 0:
        return arrays
    return tuple(jnp.pad(a, ((0, pad), (0, 0))) for a in arrays)


def trim_to(num_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
    """Inverse of ``pad_to``: drop the rows the padding added."""
    if arrays[0].shape[0] == num_tokens:
        return arrays
    return tuple(a[:num_tokens] for a in arrays)


def split_fn3(fn: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """3-chunk bf16 split of fn (8 mantissa bits each = f32's 24).

    Computed in XLA outside the kernel; the chunks are what stays
    VMEM-resident. Chunks MUST be built with reduce_precision, not dtype
    round-trips: XLA's excess-precision simplification folds
    f32->bf16->f32 into the identity, which silently zeroes the mid/lo
    chunks.
    """
    fn_hi = jax.lax.reduce_precision(fn, 8, 7)
    rem = fn - fn_hi
    fn_mid = jax.lax.reduce_precision(rem, 8, 7)
    fn_lo = jax.lax.reduce_precision(rem - fn_mid, 8, 7)
    return cast(
        tuple[jax.Array, jax.Array, jax.Array],
        tuple(t.astype(jnp.bfloat16) for t in (fn_hi, fn_mid, fn_lo)),
    )


def stack_fn_chunks(fn: jax.Array, gemm_precision: str = "highest") -> jax.Array:
    """bf16 chunks of fn stacked along the output axis, so one MXU pass
    against the stack does the work of one pass per chunk."""
    chunks = split_fn3(fn)
    if gemm_precision == "default":
        chunks = chunks[:1]
    return jnp.concatenate(chunks, axis=0)


def stack_fn_chunks_into(fn_ref, out_ref, lanes: int = 1024) -> None:
    """In-kernel ``stack_fn_chunks``: f32 fn -> stacked bf16 chunks in VMEM.

    f32 -> bf16 casts round to nearest even, as ``split_fn3``'s
    reduce_precision does, and each remainder is exact, so the chunks match.
    """
    rows, cols = fn_ref.shape
    n_chunks = out_ref.shape[0] // rows
    lanes = min(lanes, cols)
    assert cols % lanes == 0, (cols, lanes)

    def body(c, carry):
        sl = pl.ds(pl.multiple_of(c * lanes, lanes), lanes)
        rem = fn_ref[:, sl]
        for k in range(n_chunks):
            part = rem.astype(jnp.bfloat16)
            out_ref[k * rows : (k + 1) * rows, sl] = part
            rem = rem - part.astype(jnp.float32)
        return carry

    jax.lax.fori_loop(0, cols // lanes, body, 0)


def dot_stacked_chunks(x: jax.Array, fn_stack: jax.Array, n_out: int) -> jax.Array:
    """x @ fn.T from one dot against ``stack_fn_chunks``: hi + mid + lo."""
    dn = (((1,), (1,)), ((), ()))
    acc3 = jax.lax.dot_general(x, fn_stack, dn, preferred_element_type=jnp.float32)
    acc = acc3[:, :n_out]
    for c in range(1, fn_stack.shape[0] // n_out):
        acc = acc + acc3[:, c * n_out : (c + 1) * n_out]
    return acc


def sqrsum_on_mxu(x: jax.Array, x_rows: jax.Array) -> jax.Array:
    """sum(x * x, -1) as the diagonal of the Gram block x @ x_rows.T.

    ``x_rows`` holds the same rows as ``x`` (e.g. the VMEM block ``x`` was
    stored to). bf16 products are exact and the MXU accumulates in f32, so
    only the summation order differs from a VPU sum of squares.
    """
    dn = (((1,), (1,)), ((), ()))
    gram = jax.lax.dot_general(x, x_rows, dn, preferred_element_type=jnp.float32)
    row = jax.lax.broadcasted_iota(jnp.int32, gram.shape, 0)
    col = jax.lax.broadcasted_iota(jnp.int32, gram.shape, 1)
    return jnp.sum(jnp.where(row == col, gram, 0.0), axis=-1, keepdims=True)


def mhc_pre_gates(
    mixes: jax.Array,
    sqrsum: jax.Array,
    hc_mult: int,
    hidden_size: int,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gating / softmax / Sinkhorn on the tiny (T, hc_mult3) mix logits.

    The XLA reference for the gates the kernels compute in-block
    (``gate_epilogue``).
    Returns (pre_mix (T, M), post_mix (T, M), comb_mix (T, M, M)).
    """
    num_tokens = mixes.shape[0]

    mixes = mixes * jax.lax.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].reshape(
        num_tokens, hc_mult, hc_mult
    ) * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
    comb_mix = jax.nn.softmax(comb_logits, axis=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=-1, keepdims=True) + hc_sinkhorn_eps
        )
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
        )
    return pre_mix, post_mix, comb_mix


def gate_group_blocks(num_blocks: int, token_block: int) -> int:
    """Grid blocks per gate group; divides ``num_blocks`` so groups tile the grid."""
    return math.gcd(num_blocks, max(1, GATE_GROUP_TOKENS // token_block))


def gate_epilogue(
    logits_t: jax.Array,
    *,
    hc_mult: int,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array]:
    """In-kernel post/comb half of ``mhc_pre_gates``, tokens on lanes.

    ``logits_t`` is (hc_mult3, G): the gate logits (scaled mixes times
    hc_scale plus hc_base), one token per lane. Same op order as
    ``mhc_pre_gates``; comb row ``i`` is the (hc_mult, G) slab of rows
    ``i * hc_mult + j``, so the softmax and row sums run down sublanes and
    the column sums add the slabs.

    Returns post (hc_mult, G) and comb (hc_mult * hc_mult, G), row-major (i, j).
    """
    m = hc_mult
    post = jax.nn.sigmoid(logits_t[m : 2 * m]) * hc_post_mult_value

    rows = []
    for i in range(m):
        lo = 2 * m + i * m
        rows.append(jax.nn.softmax(logits_t[lo : lo + m], axis=0) + hc_sinkhorn_eps)

    def col_norm(rows):
        col = rows[0]
        for r in rows[1:]:
            col = col + r
        col = col + hc_sinkhorn_eps
        return [r / col for r in rows]

    def row_norm(rows):
        return [r / (jnp.sum(r, axis=0, keepdims=True) + hc_sinkhorn_eps) for r in rows]

    rows = col_norm(rows)
    for _ in range(sinkhorn_repeat - 1):
        rows = col_norm(row_norm(rows))
    return post, jnp.concatenate(rows, axis=0)


def stage_mixes_and_gates(
    mixes: jax.Array,
    mix_scr,
    post_out_ref,
    comb_out_ref,
    sc_ref,
    hb_ref,
    *,
    gate_blocks: int,
    hc_mult: int,
    **gate_kwargs,
) -> None:
    """Stage this block's gate logits; the group's last block writes its gates.

    ``mix_scr`` is a (G, hc_mult3) f32 scratch, G = gate_blocks * tb; the
    gate outputs are (G, hc_mult) / (G, hc_mult * hc_mult) blocks indexed
    by group, so they are written back once per group. The logits are
    formed tokens-on-sublanes so hc_base is read as the (1, hc_mult3) row
    it already is; a (hc_mult3, 1) column costs an XLA relayout per call.
    """
    tb = mixes.shape[0]
    m = hc_mult
    col = jax.lax.broadcasted_iota(jnp.int32, (1, mixes.shape[1]), 1)
    scale = jnp.where(
        col < m, sc_ref[0, 0], jnp.where(col < 2 * m, sc_ref[0, 1], sc_ref[0, 2])
    )
    j = pl.program_id(0) % gate_blocks
    mix_scr[pl.ds(pl.multiple_of(j * tb, tb), tb), :] = mixes * scale + hb_ref[...]

    @pl.when(j == gate_blocks - 1)
    def _():
        post, comb = gate_epilogue(mix_scr[...].T, hc_mult=m, **gate_kwargs)
        post_out_ref[...] = post.T
        comb_out_ref[...] = comb.T
