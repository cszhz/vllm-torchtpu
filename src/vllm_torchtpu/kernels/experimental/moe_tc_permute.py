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
"""TensorCore row permutation for the whole-expert MoE path, one DMA per row.

Both kernels move rows between HBM and VMEM with one DMA descriptor per row.
A row of a 2-D bf16 array in the default (16, 128) tiling shares its 32-bit
words with its neighbour, so a single row cannot be a DMA operand; the
kernels therefore run on the row-contiguous view ``[rows, hidden // 128,
128]``, where each row is a whole number of (16, 128) tiles. The reshape to
and from that view is a relayout copy that XLA runs outside the kernel.

``tc_row_gather`` is the dispatch gather (``x[indices]``) and ``tc_combine``
the weighted reduction of the expert outputs back to tokens.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
ROW_TILE = 256
TOKEN_TILE = 64


def _align(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _row_view(x: jax.Array) -> jax.Array:
    rows, hidden = x.shape
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} must be a multiple of {LANES}")
    return x.reshape(rows, hidden // LANES, LANES)


def _gather_kernel(idx_hbm, x_hbm, out_ref, idx_smem, sems, *, row_tile, fast):
    tile = pl.program_id(0)
    idx_copy = pltpu.make_async_copy(
        idx_hbm.at[pl.ds(pl.multiple_of(tile * row_tile, row_tile), row_tile)],
        idx_smem,
        sems.at[0],
    )
    idx_copy.start()
    idx_copy.wait()

    def issue(i, carry):
        row = idx_smem[i]
        pltpu.make_async_copy(
            x_hbm.at[pl.ds(row, 1)], out_ref.at[pl.ds(i, 1)], sems.at[1]
        ).start()
        return carry

    lax.fori_loop(0, row_tile, issue, 0, unroll=8 if fast else 1)

    if fast:
        # One wait for the tile: the semaphore counts bytes, and the whole
        # block is the sum of the row copies.
        pltpu.make_async_copy(x_hbm.at[pl.ds(0, row_tile)], out_ref, sems.at[1]).wait()
        return

    def wait(i, carry):
        pltpu.make_async_copy(
            x_hbm.at[pl.ds(0, 1)], out_ref.at[pl.ds(0, 1)], sems.at[1]
        ).wait()
        return carry

    lax.fori_loop(0, row_tile, wait, 0)


@functools.partial(jax.jit, static_argnames=("row_tile", "fast"))
def tc_row_gather(
    x: jax.Array, indices: jax.Array, *, row_tile: int = ROW_TILE, fast: bool = True
) -> jax.Array:
    """Returns ``x[indices]`` gathered on the TensorCore, one DMA per row.

    ``fast`` unrolls the issue loop and waits once per tile; off, it waits
    once per row."""
    num_rows = indices.shape[0]
    padded_rows = _align(num_rows, row_tile)
    idx = jnp.pad(indices.astype(jnp.int32), (0, padded_rows - num_rows))
    x3 = _row_view(x)
    sub, lanes = x3.shape[1:]
    out3 = pl.pallas_call(
        functools.partial(_gather_kernel, row_tile=row_tile, fast=fast),
        out_shape=jax.ShapeDtypeStruct((padded_rows, sub, lanes), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(padded_rows // row_tile,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((row_tile, sub, lanes), lambda t: (t, 0, 0)),
            scratch_shapes=[
                pltpu.SMEM((row_tile,), jnp.int32),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name="tc_row_gather" if fast else "tc_row_gather_slow",
    )(idx, x3)
    return out3.reshape(padded_rows, x.shape[1])[:num_rows]


def _combine_kernel(
    slot_hbm,
    w_hbm,
    y_hbm,
    out_ref,
    slot_smem,
    w_smem,
    buf,
    sems,
    *,
    token_tile,
    topk,
    fast,
):
    tile = pl.program_id(0)
    routes = token_tile * topk
    base = pl.multiple_of(tile * routes, routes)
    slot_copy = pltpu.make_async_copy(
        slot_hbm.at[pl.ds(base, routes)], slot_smem, sems.at[0]
    )
    w_copy = pltpu.make_async_copy(w_hbm.at[pl.ds(base, routes)], w_smem, sems.at[1])
    slot_copy.start()
    w_copy.start()
    slot_copy.wait()
    w_copy.wait()

    # Route (token, k) lands at buf[k * token_tile + token], so the k-th
    # contribution of every token of the tile is one contiguous block.
    def issue(i, carry):
        token = i // topk
        k = i - token * topk
        pltpu.make_async_copy(
            y_hbm.at[pl.ds(slot_smem[i], 1)],
            buf.at[pl.ds(k * token_tile + token, 1)],
            sems.at[2],
        ).start()
        return carry

    lax.fori_loop(0, routes, issue, 0, unroll=8 if fast else 1)

    if fast:
        pltpu.make_async_copy(y_hbm.at[pl.ds(0, routes)], buf, sems.at[2]).wait()
    else:

        def wait(i, carry):
            pltpu.make_async_copy(
                y_hbm.at[pl.ds(0, 1)], buf.at[pl.ds(0, 1)], sems.at[2]
            ).wait()
            return carry

        lax.fori_loop(0, routes, wait, 0)

    def reduce_token(token, carry):
        acc = buf[token].astype(jnp.float32) * w_smem[token * topk]
        for k in range(1, topk):
            acc = acc + buf[k * token_tile + token].astype(jnp.float32) * w_smem[
                token * topk + k
            ]
        out_ref[token] = acc.astype(out_ref.dtype)
        return carry

    lax.fori_loop(0, token_tile, reduce_token, 0)


@functools.partial(jax.jit, static_argnames=("topk", "num_tokens", "token_tile", "fast"))
def tc_combine(
    y: jax.Array,
    revert_indices: jax.Array,
    topk_weights: jax.Array,
    valid_mask: jax.Array,
    *,
    topk: int,
    num_tokens: int,
    token_tile: int = TOKEN_TILE,
    fast: bool = True,
) -> jax.Array:
    """Weighted sum of each token's ``topk`` expert output rows.

    ``y`` holds the expert outputs in sorted (expert-major) row order,
    ``revert_indices[t * topk + k]`` the row of token ``t``'s ``k``-th route,
    ``topk_weights`` and ``valid_mask`` its weight and validity. The sum runs
    in f32 in route order, as sc_ragged_gather_reduce_v3 does.
    """
    padded_tokens = _align(num_tokens, token_tile)
    pad = (padded_tokens - num_tokens) * topk
    slots = jnp.pad(revert_indices.astype(jnp.int32), (0, pad))
    weights = jnp.where(valid_mask, topk_weights.astype(jnp.float32), 0.0)
    weights = jnp.pad(weights, (0, pad))
    y3 = _row_view(y)
    sub, lanes = y3.shape[1:]
    out3 = pl.pallas_call(
        functools.partial(_combine_kernel, token_tile=token_tile, topk=topk, fast=fast),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, sub, lanes), y.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(padded_tokens // token_tile,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((token_tile, sub, lanes), lambda t: (t, 0, 0)),
            scratch_shapes=[
                pltpu.SMEM((token_tile * topk,), jnp.int32),
                pltpu.SMEM((token_tile * topk,), jnp.float32),
                pltpu.VMEM((token_tile * topk, sub, lanes), y.dtype),
                pltpu.SemaphoreType.DMA((3,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name="tc_combine" if fast else "tc_combine_slow",
    )(slots, weights, y3)
    return out3.reshape(padded_tokens, y.shape[1])[:num_tokens]
