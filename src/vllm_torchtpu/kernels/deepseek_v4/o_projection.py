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

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import vmem_budget
from vllm_torchtpu.kernels.deepseek_v4.rope import (
    LANE,
    CosSinRef,
    _cos_sin_lanes,
    _rotate_gptj,
)
from vllm_torchtpu.layers.core.quantization import FP8_E4M3_MAX

DEFAULT_VMEM_LIMIT_BYTES = vmem_budget.LEGACY_LIMIT_BYTES

SUBLANE = 8


def _largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of ``x`` that is <= ``cap``."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


def _cos_sin_body(
    cos_sin_ref: jax.Ref,
    out_ref: jax.Ref,
    *,
    rotary_dim: int,
    inverse: bool,
    out_dtype: jnp.dtype,
) -> None:
    """Expand a ``(tile_n, rotary_dim)`` block to ``(tile_n, 2 * LANE)``."""
    cos, sin = _cos_sin_lanes(cos_sin_ref[...], rotary_dim=rotary_dim, inverse=inverse)
    out_ref[...] = jnp.concatenate([cos, sin], axis=-1).astype(out_dtype)


def _cos_sin_kernel(
    # scalar prefetch
    positions_ref: jax.Ref,
    # HBM input
    cos_sin_cache_hbm_ref: jax.Ref,
    # HBM output
    out_hbm_ref: jax.Ref,
    *,
    num_tiles: int,
    tile_n: int,
    rotary_dim: int,
    inverse: bool,
    cos_sin_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
) -> None:
    cos_sin_spec = pl.BlockSpec(
        block_shape=(tile_n, rotary_dim),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0),
    )
    out_spec = pl.BlockSpec(
        block_shape=(tile_n, 2 * LANE),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0),
    )
    # The per-token row gather; one buffer of lookahead hides its DMAs.
    cos_sin_alloc = CosSinRef.create(
        spec=cos_sin_spec, dtype=cos_sin_dtype, buffer_count=2, tile_n=tile_n
    )

    pipeline = pltpu.emit_pipeline(
        functools.partial(
            _cos_sin_body,
            rotary_dim=rotary_dim,
            inverse=inverse,
            out_dtype=out_dtype,
        ),
        grid=(num_tiles,),
        in_specs=[cos_sin_spec],
        out_specs=[out_spec],
    )
    allocations = (
        cos_sin_alloc,
        pltpu.BufferedRef.output(out_spec, out_dtype, buffer_count=2),
    )

    @pl.with_scoped(allocations=allocations)
    def _run(allocations):
        pipeline(
            (cos_sin_cache_hbm_ref, positions_ref),
            out_hbm_ref,
            allocations=allocations,
        )

    _run()


@functools.partial(
    jax.jit,
    static_argnames=("inverse", "out_dtype", "name"),
)
def gather_cos_sin(
    positions: jax.Array,  # [T] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    inverse: bool = False,
    out_dtype: jnp.dtype = jnp.float32,
    name: str = "gather_cos_sin",
) -> jax.Array:
    """Builds ``wo_a_projection``'s ``[T, 2 * LANE]`` cos/sin table.

    Gathers ``cos_sin_cache[positions]`` based on the token positions, expands
    each row to ``[2 * LANE]``
    """
    assert positions.ndim == 1
    assert cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    rotary_dim = cos_sin_cache.shape[1]
    assert rotary_dim % 2 == 0 and rotary_dim <= LANE

    tile_n = _largest_divisor(num_tokens, cap=128)

    return pl.pallas_call(
        functools.partial(
            _cos_sin_kernel,
            num_tiles=num_tokens // tile_n,
            tile_n=tile_n,
            rotary_dim=rotary_dim,
            inverse=inverse,
            cos_sin_dtype=cos_sin_cache.dtype,
            out_dtype=out_dtype,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, 2 * LANE), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_budget.dsv4_limit_bytes(num_tokens),
            disable_bounds_checks=True,
        ),
        name=name,
    )(positions, cos_sin_cache)


def _rope_heads(x: jax.Array, cos_sin: jax.Array, *, head_dim: int) -> jax.Array:
    """apply RoPE to the last rope dim of head dim of ``x``."""
    cos = cos_sin[:, :LANE]
    sin = cos_sin[:, LANE:]
    lo = head_dim - LANE
    tail = x[:, :, lo:].astype(jnp.float32)
    roped = tail * cos[:, None, :] + _rotate_gptj(tail) * sin[:, None, :]
    return jnp.concatenate([x[:, :, :lo], roped.astype(x.dtype)], axis=-1)


def _kernel(
    # inputs
    x_ref: jax.Ref,  # VMEM (tile_t, H, head_dim) bf16
    w_ref: jax.Ref,  # VMEM (D, tile_r) fp8
    scale_ref: jax.Ref,  # VMEM (1, tile_r) f32
    cos_sin_ref: jax.Ref,  # VMEM (tile_t, 2 * LANE) f32
    # output
    out_ref: jax.Ref,  # VMEM (tile_t, tile_r) out_dtype
    *,
    tile_t: int,
    num_sub_t: int,
    quantize_activations: bool,
    head_dim: int,
    groups_per_step: int = 1,
) -> None:
    cos_sin = cos_sin_ref[...]
    sub_t = tile_t // num_sub_t
    out_dtype = x_ref.dtype
    heads = x_ref.shape[1] // groups_per_step
    cols = w_ref.shape[1] // groups_per_step
    for j in range(groups_per_step):
        _group(
            x_ref.at[:, pl.ds(j * heads, heads)],
            w_ref.at[:, pl.ds(j * cols, cols)],
            scale_ref.at[:, pl.ds(j * cols, cols)],
            cos_sin,
            out_ref.at[:, pl.ds(j * cols, cols)],
            sub_t=sub_t,
            num_sub_t=num_sub_t,
            quantize_activations=quantize_activations,
            head_dim=head_dim,
            out_dtype=out_dtype,
        )


def _group(
    x_ref,
    w_ref,
    scale_ref,
    cos_sin,
    out_ref,
    *,
    sub_t,
    num_sub_t,
    quantize_activations,
    head_dim,
    out_dtype,
):
    """One group's heads against its wo_a column block."""
    rhs = w_ref[...]
    scale = scale_ref[...]
    for s in range(num_sub_t):
        # Subchunk the tile to allow more Instruction level parallelism
        # opportunities to be exploited by the LLO compiler.
        rows = slice(s * sub_t, (s + 1) * sub_t)
        x = _rope_heads(x_ref[rows], cos_sin[rows], head_dim=head_dim)
        x = x.reshape(sub_t, -1)
        inv = None
        if quantize_activations:
            amax = jnp.max(jnp.abs(x), axis=1, keepdims=True)
            inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(
                jnp.bfloat16
            )
            lhs = (x * inv).astype(jnp.float8_e4m3fn)
        else:
            lhs = x

        partial = jax.lax.dot_general(
            lhs,
            rhs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if quantize_activations:
            assert inv is not None
            partial = partial * (1.0 / inv.astype(jnp.float32))
        out_ref[rows] = (partial * scale).astype(out_dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "tile_t",
        "tile_r",
        "sub_t",
        "quantize_activations",
        "name",
        "groups_per_step",
    ),
)
def wo_a_projection(
    x: jax.Array,  # [T, G * H, head_dim] bf16
    wo_a: jax.Array,  # [D, G * R] fp8
    wo_a_scale: jax.Array,  # [G * R] float32
    cos_sin: jax.Array,  # [T, 2 * LANE] float32
    *,
    tile_t: int | None = None,
    tile_r: int | None = None,
    sub_t: int | None = None,
    quantize_activations: bool = True,
    name: str = "wo_a_projection",
    groups_per_step: int | None = None,
) -> jax.Array:
    """Do DSv4 wo_a projection.

    T: num_tokens
    G: num_groups
    H: num_heads_per_group
    D: H * head_dim
    R: o_lora_dim

    Equivalent to
    x = rope(x)
    return ``einsum("tgd,dgr->tgr", x.view(t, g, d), wo_a.view(d, g, r),
    preferred_element_type=f32) * wo_a_scale.view(g, r).to(bf16)``

    ``quantize_activations`` quantizes each activation block's rows to fp8 in
    the kernel so both MXU operands are fp8.
    """
    assert x.ndim == 3
    assert x.dtype == jnp.bfloat16
    assert wo_a.ndim == 2 and wo_a_scale.ndim == 1

    num_tokens, num_heads, head_dim = x.shape
    reduction, out_features = wo_a.shape
    assert out_features == wo_a_scale.shape[0]

    assert reduction % head_dim == 0
    heads_per_group = reduction // head_dim

    # DSv4's heads_per_group is 8, exactly equal to SUBLANE
    # The kernel is based on this assumption, which makes it
    # simpler.
    assert heads_per_group == SUBLANE
    assert num_heads % heads_per_group == 0
    num_groups = num_heads // heads_per_group
    assert out_features % num_groups == 0
    lora_rank = out_features // num_groups

    assert cos_sin.ndim == 2 and cos_sin.shape == (num_tokens, 2 * LANE)
    assert head_dim % LANE == 0

    if tile_r is None:
        tile_r = lora_rank
    if tile_t is None:
        tile_t = _largest_divisor(num_tokens, cap=1024)
    if sub_t is None:
        # 128 is chosen based on large batch size (t) microbenchmarks
        sub_t = _largest_divisor(tile_t, cap=128)
    assert lora_rank % tile_r == 0
    assert num_tokens % tile_t == 0
    assert tile_t % sub_t == 0
    num_t_tiles = num_tokens // tile_t
    num_r_tiles = lora_rank // tile_r
    if groups_per_step is None:
        # Two groups per grid step halve the steps (and their fixed cost) at
        # decode-sized token tiles; a larger tile's doubled blocks overrun VMEM.
        # tpu7x, T=16/64/128: 28.6/32.0/37.2 -> 23.6/26.9/31.5 us.
        groups_per_step = 2 if tile_t <= 512 and num_r_tiles == 1 else 1
    gps = groups_per_step if num_groups % groups_per_step == 0 else 1
    assert gps == 1 or num_r_tiles == 1

    return pl.pallas_call(
        functools.partial(
            _kernel,
            tile_t=tile_t,
            num_sub_t=tile_t // sub_t,
            quantize_activations=quantize_activations,
            head_dim=head_dim,
            groups_per_step=gps,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, out_features), x.dtype),
        grid=(num_groups // gps, num_t_tiles, num_r_tiles),
        in_specs=[
            # x: `gps` groups' heads, [t, g, 0] over [T, G * H, head_dim].
            pl.BlockSpec(
                (tile_t, gps * heads_per_group, head_dim), lambda g, t, r: (t, g, 0)
            ),
            # wo_a: the whole reduction, column block g * R + r * tile_r.
            pl.BlockSpec(
                (reduction, gps * tile_r), lambda g, t, r: (0, g * num_r_tiles + r)
            ),
            # wo_a_scale
            pl.BlockSpec((1, gps * tile_r), lambda g, t, r: (0, g * num_r_tiles + r)),
            # cos_sin: one token block
            pl.BlockSpec((tile_t, 2 * LANE), lambda g, t, r: (t, 0)),
        ],
        out_specs=pl.BlockSpec(
            (tile_t, gps * tile_r), lambda g, t, r: (t, g * num_r_tiles + r)
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_budget.dsv4_limit_bytes(num_tokens),
            disable_bounds_checks=True,
        ),
        name=name,
    )(x, wo_a, wo_a_scale.reshape(1, out_features), cos_sin)


@functools.partial(
    jax.jit,
    static_argnames=(
        "head_dim",
        "inverse",
        "tile_t",
        "tile_r",
        "sub_t",
        "quantize_activations",
        "name",
    ),
)
def fused_reverse_rope_wo_a_projection(
    x: jax.Array,  # [T, G * H, head_dim] bf16
    positions: jax.Array,  # [T] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    wo_a: jax.Array,  # [D, G * R] fp8
    wo_a_scale: jax.Array,  # [G * R] float32
    *,
    head_dim: int,
    inverse: bool = True,
    tile_t: int | None = None,
    tile_r: int | None = None,
    sub_t: int | None = None,
    quantize_activations: bool = True,
    name: str = "wo_a_projection",
) -> jax.Array:
    """``The RoPE-fused ``wo_a_projection``."""
    assert x.ndim == 3
    assert x.shape[2] == head_dim

    cos_sin = gather_cos_sin(
        positions,
        cos_sin_cache,
        inverse=inverse,
        name=f"{name}_gather_cos_sin",
    )
    return wo_a_projection(
        x,
        wo_a,
        wo_a_scale,
        cos_sin,
        tile_t=tile_t,
        tile_r=tile_r,
        sub_t=sub_t,
        quantize_activations=quantize_activations,
        name=name,
    )


def _pair_kernel(
    first_ref: jax.Ref,  # VMEM (T, gps * H, head_dim) bf16, tokens [0, T)
    second_ref: jax.Ref,  # VMEM (T, gps * H, head_dim) bf16, tokens [T, 2T)
    w_ref: jax.Ref,  # VMEM (D, gps * tile_r) fp8
    scale_ref: jax.Ref,  # VMEM (1, gps * tile_r) f32
    cos_sin_ref: jax.Ref,  # VMEM (2T, 2 * LANE) f32
    out_ref: jax.Ref,  # VMEM (2T, gps * tile_r)
    *,
    sub_t: int,
    head_dim: int,
    groups_per_step: int,
) -> None:
    """`_kernel` over one 2T-token tile whose rows live in two refs.

    Each sub-tile takes the same rows `_kernel` would, assembled from whichever
    ref holds them, so every dot sees the same operands.
    """
    num_tokens = first_ref.shape[0]
    cos_sin = cos_sin_ref[...]
    out_dtype = first_ref.dtype
    heads = first_ref.shape[1] // groups_per_step
    cols = w_ref.shape[1] // groups_per_step
    for j in range(groups_per_step):
        rhs = w_ref[:, pl.ds(j * cols, cols)]
        scale = scale_ref[:, pl.ds(j * cols, cols)]
        for s in range(2 * num_tokens // sub_t):
            r0, r1 = s * sub_t, (s + 1) * sub_t
            pieces = []
            if r0 < num_tokens:
                pieces.append(
                    first_ref[r0 : min(r1, num_tokens), pl.ds(j * heads, heads)]
                )
            if r1 > num_tokens:
                pieces.append(
                    second_ref[
                        max(r0, num_tokens) - num_tokens : r1 - num_tokens,
                        pl.ds(j * heads, heads),
                    ]
                )
            x = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=0)
            x = _rope_heads(x, cos_sin[r0:r1], head_dim=head_dim)
            x = x.reshape(sub_t, -1)
            amax = jnp.max(jnp.abs(x), axis=1, keepdims=True)
            inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(
                jnp.bfloat16
            )
            lhs = (x * inv).astype(jnp.float8_e4m3fn)
            partial = jax.lax.dot_general(
                lhs,
                rhs,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            partial = partial * (1.0 / inv.astype(jnp.float32))
            out_ref[r0:r1, pl.ds(j * cols, cols)] = (partial * scale).astype(out_dtype)


# The pair kernel covers one token tile of the fused op: 2T <= 1024 tokens.
PAIR_MAX_TOKENS = 1024


@functools.partial(
    jax.jit,
    static_argnames=(
        "first_row0",
        "first_head0",
        "second_row0",
        "second_head0",
        "num_heads",
        "head_dim",
        "inverse",
        "name",
    ),
)
def fused_reverse_rope_wo_a_projection_pair(
    first: jax.Array,  # [R, heads, head_dim] bf16
    second: jax.Array,  # [R', heads', head_dim] bf16
    positions: jax.Array,  # [2T] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    wo_a: jax.Array,  # [D, G * R] fp8
    wo_a_scale: jax.Array,  # [G * R] float32
    *,
    first_row0: int,
    first_head0: int,
    second_row0: int,
    second_head0: int,
    num_heads: int,
    head_dim: int,
    inverse: bool = True,
    name: str = "wo_a_projection",
) -> jax.Array:
    """``fused_reverse_rope_wo_a_projection`` (activation-quantized) of the
    [2T, G*H, head_dim] input whose tokens [0, T) are
    ``first[first_row0 + t, first_head0 + h]`` and tokens [T, 2T) are
    ``second[second_row0 + t, second_head0 + h]``, both read in place.
    Bit-identical to the fused op on the assembled input, for 2T <=
    PAIR_MAX_TOKENS.
    """
    num_tokens = positions.shape[0] // 2
    reduction, out_features = wo_a.shape
    assert reduction % head_dim == 0 and reduction // head_dim == SUBLANE
    total = 2 * num_tokens
    assert total <= PAIR_MAX_TOKENS, "the pair kernel assumes a single token tile"
    assert num_heads % SUBLANE == 0
    num_groups = num_heads // SUBLANE
    assert out_features % num_groups == 0
    lora_rank = out_features // num_groups
    # The fused op's tiling at 2T tokens: one token tile, sub-tiles of
    # largest_divisor(2T, 128), and two groups per grid step up to 512 tokens.
    sub_t = _largest_divisor(total, cap=128)
    gps = 2 if total <= 512 and num_groups % 2 == 0 else 1
    width = gps * SUBLANE
    for row0, head0 in ((first_row0, first_head0), (second_row0, second_head0)):
        assert row0 % num_tokens == 0 and head0 % width == 0
    f_tile, f_head = first_row0 // num_tokens, first_head0 // width
    s_tile, s_head = second_row0 // num_tokens, second_head0 // width

    cos_sin = gather_cos_sin(
        positions, cos_sin_cache, inverse=inverse, name=f"{name}_gather_cos_sin"
    )
    return pl.pallas_call(
        functools.partial(
            _pair_kernel, sub_t=sub_t, head_dim=head_dim, groups_per_step=gps
        ),
        out_shape=jax.ShapeDtypeStruct((total, out_features), first.dtype),
        grid=(num_groups // gps,),
        in_specs=[
            pl.BlockSpec(
                (num_tokens, width, head_dim), lambda g: (f_tile, f_head + g, 0)
            ),
            pl.BlockSpec(
                (num_tokens, width, head_dim), lambda g: (s_tile, s_head + g, 0)
            ),
            pl.BlockSpec((reduction, gps * lora_rank), lambda g: (0, g)),
            pl.BlockSpec((1, gps * lora_rank), lambda g: (0, g)),
            pl.BlockSpec((total, 2 * LANE), lambda g: (0, 0)),
        ],
        out_specs=pl.BlockSpec((total, gps * lora_rank), lambda g: (0, g)),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_budget.dsv4_limit_bytes(total),
            disable_bounds_checks=True,
        ),
        name=name,
    )(first, second, wo_a, wo_a_scale.reshape(1, out_features), cos_sin)
