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
# tokens on lanes. The epilogue's serial Sinkhorn chain costs about the
# same for any group up to 1024 tokens (one dense vreg per gate plane), so
# wider groups run it fewer times (deviceless tpu7x LLO, seam at T=1024:
# 1 x 1258 bundles per call vs 4 x 1134 for the sublane-slab epilogue at 256).
GATE_GROUP_TOKENS = 1024

# Token rows per step of the collapse loop (``collapse_rows``): one step's
# streams fit in vregs, where a whole-block collapse in one basic block is
# scheduled loads-first and spills them.
COLLAPSE_LOOP_ROWS = 16


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

    When ``out_ref`` is narrower than fn, fn's column blocks of its width
    are summed in f32 first (left to right): ``x @ chunks.T`` is then
    ``concat([x] * n) @ fn.T`` up to f32 rounding of the summed weights.
    """
    rows = fn_ref.shape[0]
    cols = out_ref.shape[1]
    n_sum = fn_ref.shape[1] // cols
    assert fn_ref.shape[1] == n_sum * cols, (fn_ref.shape, out_ref.shape)
    n_chunks = out_ref.shape[0] // rows
    lanes = min(lanes, cols)
    assert cols % lanes == 0, (cols, lanes)

    def body(c, carry):
        sl = pl.ds(pl.multiple_of(c * lanes, lanes), lanes)
        rem = fn_ref[:, sl]
        for s in range(1, n_sum):
            start = pl.multiple_of(s * cols + c * lanes, lanes)
            rem = rem + fn_ref[:, pl.ds(start, lanes)]
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
    (``gate_epilogue_planes``).
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
    """Grid blocks per gate group; divides ``num_blocks`` so groups tile the grid.

    Blocks group only when ``token_block`` is lane-aligned, so each block's
    logits land at a 128-aligned lane offset of the staging scratch.
    """
    if token_block % 128:
        return 1
    return math.gcd(num_blocks, max(1, GATE_GROUP_TOKENS // token_block))


def gate_epilogue_planes(
    planes: list[jax.Array],
    *,
    hc_mult: int,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[list[jax.Array], list[jax.Array]]:
    """In-kernel post/comb half of ``mhc_pre_gates``, one plane per logit.

    ``planes[c]`` holds gate logit column ``c`` (scaled mixes times hc_scale
    plus hc_base) for a group of tokens as a dense (sublanes, lanes) array,
    so the softmax and the Sinkhorn row/column sums are adds of whole
    planes, with no cross-sublane reductions. Same ops as
    ``mhc_pre_gates``; comb entry (i, j) is plane ``2 * hc_mult + i * hc_mult + j``.

    Returns post (hc_mult planes) and comb (hc_mult * hc_mult planes,
    row-major (i, j)).
    """
    m = hc_mult
    post = [jax.nn.sigmoid(planes[m + q]) * hc_post_mult_value for q in range(m)]

    def plane_sum(xs):
        s = xs[0]
        for x in xs[1:]:
            s = s + x
        return s

    rows = []
    for i in range(m):
        logits = planes[2 * m + i * m : 2 * m + (i + 1) * m]
        mx = logits[0]
        for x in logits[1:]:
            mx = jnp.maximum(mx, x)
        e = [jnp.exp(x - mx) for x in logits]
        s = plane_sum(e)
        rows.append([x / s + hc_sinkhorn_eps for x in e])

    def col_norm(rows):
        cols = [plane_sum([r[j] for r in rows]) + hc_sinkhorn_eps for j in range(m)]
        return [[r[j] / cols[j] for j in range(m)] for r in rows]

    def row_norm(rows):
        return [[x / (plane_sum(r) + hc_sinkhorn_eps) for x in r] for r in rows]

    rows = col_norm(rows)
    for _ in range(sinkhorn_repeat - 1):
        rows = col_norm(row_norm(rows))
    return post, [x for r in rows for x in r]


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

    ``mix_scr`` is a (hc_mult3, G) f32 scratch, G = gate_blocks * tb, tokens
    on lanes; the gate outputs are (G, hc_mult) / (G, hc_mult * hc_mult)
    blocks indexed by group, so they are written back once per group. The
    logits are formed tokens-on-sublanes so hc_base is read as the
    (1, hc_mult3) row it already is (a (hc_mult3, 1) column costs an XLA
    relayout per call), then transposed into the scratch.
    """
    tb = mixes.shape[0]
    m = hc_mult
    col = jax.lax.broadcasted_iota(jnp.int32, (1, mixes.shape[1]), 1)
    scale = jnp.where(
        col < m, sc_ref[0, 0], jnp.where(col < 2 * m, sc_ref[0, 1], sc_ref[0, 2])
    )
    j = pl.program_id(0) % gate_blocks
    logits = mixes * scale + hb_ref[...]
    mix_scr[:, pl.ds(pl.multiple_of(j * tb, tb), tb)] = logits.T

    @pl.when(j == gate_blocks - 1)
    def _():
        # Plane c gets token s * lanes + l at (s, l).
        group = mix_scr.shape[1]
        lanes = group if group <= 128 else math.gcd(group, 128)
        n_sub = group // lanes
        blocks_t = [mix_scr[:, b * lanes : (b + 1) * lanes] for b in range(n_sub)]
        planes = [
            jnp.concatenate([blk[c : c + 1] for blk in blocks_t], axis=0)
            for c in range(mixes.shape[1])
        ]
        post, comb = gate_epilogue_planes(planes, hc_mult=m, **gate_kwargs)
        for b in range(n_sub):
            rows = pl.ds(b * lanes, lanes)
            post_out_ref[rows, :] = jnp.concatenate([q[b : b + 1] for q in post], 0).T
            comb_out_ref[rows, :] = jnp.concatenate([q[b : b + 1] for q in comb], 0).T


def replicate_lanes(cols: jax.Array) -> jax.Array:
    """(rows, n) -> (rows, n * 128): column k repeated over lane tile k."""
    rows, n = cols.shape
    return jnp.concatenate(
        [jnp.broadcast_to(cols[:, k : k + 1], (rows, 128)) for k in range(n)], axis=1
    )


def collapse_rows(streams_ref, weights_ref, out_ref, *, hc_mult: int) -> None:
    """out = sum_k w_k * streams[:, k*h:(k+1)*h] in f32, k left to right.

    ``weights_ref`` holds ``replicate_lanes(w)``, so each weight is a plain
    vreg load in the loop rather than a lane broadcast.
    """
    tb, h = out_ref.shape
    rows = min(COLLAPSE_LOOP_ROWS, tb)
    assert tb % rows == 0, (tb, rows)

    def body(t, carry):
        r = pl.ds(pl.multiple_of(t * rows, rows), rows)
        w = [weights_ref[r, k * 128 : (k + 1) * 128] for k in range(hc_mult)]
        for c in range(0, h, 128):
            lay = w[0] * streams_ref[r, c : c + 128].astype(jnp.float32)
            for k in range(1, hc_mult):
                g = streams_ref[r, k * h + c : k * h + c + 128].astype(jnp.float32)
                lay = lay + w[k] * g
            out_ref[r, c : c + 128] = lay.astype(out_ref.dtype)
        return carry

    jax.lax.fori_loop(0, tb // rows, body, 0)
