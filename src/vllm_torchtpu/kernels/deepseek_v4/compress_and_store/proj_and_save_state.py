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

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout


@dataclasses.dataclass(frozen=True)
class TileSizes:
    """Tile sizes for Pallas kernel."""

    tile_m: int  # output dimension
    tile_k: int  # hidden dimension
    tile_n: int  # sequence length


@dataclasses.dataclass(frozen=True)
class Dimensions:
    """Dimensions of the input and output tensors."""

    size_m: int  # output dimension
    size_k: int  # hidden dimension
    size_n: int  # sequence length

    # other values
    compress_ratio: int


@dataclasses.dataclass(frozen=True)
class Configs:
    """Configuration parameters for the projection and save state kernel."""

    tile_sizes: TileSizes
    dims: Dimensions
    token_page_size: int

    @property
    def num_m(self) -> int:
        return pl.cdiv(self.dims.size_m, self.tile_sizes.tile_m)

    @property
    def num_k(self) -> int:
        return pl.cdiv(self.dims.size_k, self.tile_sizes.tile_k)

    @property
    def num_n(self) -> int:
        return pl.cdiv(self.dims.size_n, self.tile_sizes.tile_n)


# Upper bound on tokens per grid step.
MAX_TILE_N = 128

# Strided accumulator rows reach 8 sublanes per vector access only when the
# token stride (in 128-lane chunks) has no factor of 8: stride 16 splits each
# access 4 ways, stride 8 two ways, odd strides not at all.
ACC_PAD_ROWS = 1


def acc_rows_per_token(slots_per_token: int, last_dim: int) -> int:
    """Accumulator rows per token: the state's slots plus padding when the
    unpadded chunk stride would split strided vector stores."""
    chunks_per_token = slots_per_token * last_dim // 128
    if chunks_per_token % 8 == 0:
        return slots_per_token + ACC_PAD_ROWS
    return slots_per_token


def store_state_chunks(ref, val, chunk_offset, chunks_per_token):
    """Stores ``val`` (tile_n, c * 128) as 128-lane chunks ``chunk_offset..``
    of each token's state in a token-major (tile_n * slots, 1, lanes) ref."""
    num_lanes = 128
    tile_n = val.shape[0]
    ref_flat = ref.reshape(-1, num_lanes)
    for i in range(val.shape[1] // num_lanes):
        val_slice = val[:, i * num_lanes : (i + 1) * num_lanes]
        ref_flat[pl.ds(chunk_offset + i, tile_n, chunks_per_token)] = val_slice


def load_state_chunks(ref, tile_n, chunk_offset, num_chunks, chunks_per_token):
    """Inverse of ``store_state_chunks``: (tile_n, num_chunks * 128) f32."""
    num_lanes = 128
    ref_flat = ref.reshape(-1, num_lanes)
    chunks = [
        ref_flat[pl.ds(chunk_offset + i, tile_n, chunks_per_token)]
        for i in range(num_chunks)
    ]
    return jnp.concat(chunks, axis=1)


def kernel(
    # scalar prefetch
    slot_mapping,
    positions,
    # input
    hidden_states,  # [num_tokens, hidden_size]
    wkv_wgate,  # [hidden_size, 2 * state_width]
    ape,  # [compress_ratio, state_width]
    cache,  # [num_pages, page_size, 4, 128] uint8 or [..., 128] int32 (aliased)
    # output
    _,  # unused output ref (aliased to cache)
    # scratch_memory
    w_vmem,  # [hidden_size, 2 * state_width], resident for the whole call
    w_sems,  # [num_k]
    acc_ref,  # (2, tile_n * acc_rows_per_token, 1, lanes) f32
    sem_ref,  # [2]
    write_rows_ref,  # [2] rows in flight from each acc buffer
    run_starts_ref,  # [tile_n + 1] first token of each run of one tile
    run_ends_ref,  # [tile_n + 1] one past the last token of each run
    *,
    cfgs: Configs,
):
    """Projects each token tile against the resident weight, adds APE, and
    writes every token's whole state (kv | score) with one DMA."""
    tile_n = cfgs.tile_sizes.tile_n
    tile_k = cfgs.tile_sizes.tile_k
    num_k = cfgs.num_k
    state_width = cfgs.dims.size_m // 2
    last_dim = acc_ref.shape[-1]
    slots_per_token = cfgs.dims.size_m // last_dim
    acc_pitch = acc_rows_per_token(slots_per_token, last_dim)
    chunk_pitch = acc_pitch * last_dim // 128
    acc_ref_u8 = acc_ref.bitcast(jnp.uint8)
    # A CSA int32 NoPE array hosts the state too; write it as uint8 slabs.
    cache = csa_cache_layout.as_u8_slabs(cache)
    padded = acc_pitch != slots_per_token
    if padded:
        num_pages, page_rows_, sub_rows, lanes = cache.shape
        assert page_rows_ % slots_per_token == 0
        # Token-major views, so one DMA moves a run's tokens and skips the pad.
        cache_tokens = cache.reshape(
            num_pages, page_rows_ // slots_per_token, slots_per_token, sub_rows, lanes
        )
        acc_tokens = acc_ref_u8.reshape(2, tile_n, acc_pitch, sub_rows, lanes)
    write_rows_ref[0] = 0
    write_rows_ref[1] = 0

    def _weight_copy(k):
        rows = pl.ds(k * tile_k, tile_k)
        return pltpu.make_async_copy(
            wkv_wgate.at[rows], w_vmem.at[rows], w_sems.at[k]
        )

    # Fetched once; step 0 waits for each K block just before its first use.
    for k in range(num_k):
        _weight_copy(k).start()

    def _wait_writes(buf_idx):
        rows = write_rows_ref[buf_idx]

        @pl.when(rows > 0)
        def _():
            buf = acc_ref_u8.at[buf_idx, pl.ds(0, rows)]
            pltpu.make_async_copy(buf, buf, sem_ref.at[buf_idx]).wait()

    def _dma_copy_out(n_idx, buf_idx):
        """Writes the tile's valid tokens from one acc buffer to the cache,
        one DMA per run of tokens whose states are consecutive in one page.

        A branch-free scan records where runs start and end; DMAs are issued
        only for runs, so a prefill tile costs a few DMAs instead of tile_n.
        """
        page_rows = cfgs.token_page_size
        base = n_idx * tile_n
        rows = 0
        num_starts = num_ends = jnp.int32(0)
        prev_slot = None
        for i in range(tile_n):
            slot = slot_mapping[base + i]
            valid = slot >= 0
            rows = rows + lax.select(valid, slots_per_token, 0)
            if prev_slot is None:
                starts = valid
            else:
                prev_valid = prev_slot >= 0
                extends = prev_valid & (slot == prev_slot + slots_per_token)
                if page_rows % slots_per_token == 0:
                    # Slots are page * page_rows + j * slots_per_token, so a
                    # run reaches the next page only when the page is full.
                    extends = extends & (slot % page_rows != 0)
                starts = valid & jnp.logical_not(extends)
                # Unconditional store: an index past the last end is scratch.
                run_ends_ref[num_ends] = i
                num_ends = num_ends + (prev_valid & jnp.logical_not(extends))
            run_starts_ref[num_starts] = i
            num_starts = num_starts + starts
            prev_slot = slot
        run_ends_ref[num_ends] = tile_n
        write_rows_ref[buf_idx] = rows

        @pl.loop(0, num_starts)
        def _(r):
            start = run_starts_ref[r]
            slot = slot_mapping[base + start]
            run_tokens = run_ends_ref[r] - start
            run_rows = run_tokens * slots_per_token
            if padded:
                dest = cache_tokens.at[
                    slot // page_rows,
                    pl.ds((slot % page_rows) // slots_per_token, run_tokens),
                ]
                src = acc_tokens.at[
                    buf_idx, pl.ds(start, run_tokens), pl.ds(0, slots_per_token)
                ]
            else:
                dest = cache.at[slot // page_rows, pl.ds(slot % page_rows, run_rows)]
                src = acc_ref_u8.at[
                    buf_idx, pl.ds(start * slots_per_token, run_rows)
                ]
            pltpu.make_async_copy(src, dest, sem_ref.at[buf_idx]).start()

    def inner_kernel(
        tiled_hidden_states_ref,  # (tile_n, hidden_size)
        tiled_positions_ref,  # (tile_n,)
    ):
        n_idx = pl.program_id(0)
        buf_idx = n_idx % 2
        acc = acc_ref.at[buf_idx]

        # The buffer's previous writes (tile n - 2) must land before reuse.
        _wait_writes(buf_idx)

        def _ape_selected():
            tile_positions = tiled_positions_ref[...]
            ape_rows = tile_positions % cfgs.dims.compress_ratio
            ape_val = ape[...].astype(jnp.float32)
            one_hot = (
                ape_rows[:, None] == jnp.arange(cfgs.dims.compress_ratio)[None, :]
            ).astype(jnp.float32)
            return jnp.matmul(
                one_hot, ape_val, precision=lax.Precision.HIGHEST
            )  #  (tile_n, state_width)

        # Golden's products and f32 sums, one (k, m) block at a time:
        # kv_k0 + kv_k1 ..., then + APE on the score half.
        for k in range(num_k):

            @pl.when(n_idx == 0)
            def _wait_weight():
                _weight_copy(k).wait()

            lhs = tiled_hidden_states_ref[:, pl.ds(k * tile_k, tile_k)]
            for m in range(2):
                rhs = w_vmem[
                    pl.ds(k * tile_k, tile_k), pl.ds(m * state_width, state_width)
                ]
                prod = lax.dot_general(
                    lhs,
                    rhs,
                    (((1,), (0,)), ((), (()))),
                    preferred_element_type=jnp.float32,
                )
                chunk_offset = m * (state_width // 128)
                num_chunks = state_width // 128
                if k == 0:
                    new_acc = prod
                else:
                    old_acc = load_state_chunks(
                        acc, tile_n, chunk_offset, num_chunks, chunk_pitch
                    )
                    new_acc = old_acc + prod
                if k == num_k - 1 and m == 1:
                    new_acc = new_acc + _ape_selected()
                store_state_chunks(acc, new_acc, chunk_offset, chunk_pitch)

        _dma_copy_out(n_idx, buf_idx)

    hidden_states_spec = pl.BlockSpec(
        (cfgs.tile_sizes.tile_n, cfgs.dims.size_k),
        lambda n: (n, 0),
        memory_space=pltpu.VMEM,
    )
    positions_spec = pl.BlockSpec(
        (cfgs.tile_sizes.tile_n,),
        lambda n: (n,),
        memory_space=pltpu.VMEM,
    )

    pipeline_fn = pltpu.emit_pipeline(
        inner_kernel,
        grid=(cfgs.num_n,),
        in_specs=(
            hidden_states_spec,
            positions_spec,
        ),
    )

    pipeline_fn(
        hidden_states,
        positions,
    )

    for b in range(2):
        _wait_writes(b)


def _select_tile_k(size_k: int, cap: int = 3584, align: int = 128) -> int:
    """Largest 128-aligned k tile that *divides* ``size_k``."""
    cap = max(align, (min(cap, size_k) // align) * align)
    for cand in range(cap, 0, -align):
        if size_k % cand == 0:
            return cand
    raise AssertionError(f"Cannot find a tile_k that divides size_k={size_k}")


@functools.partial(
    jax.jit,
    static_argnames=[
        "compress_ratio",
        "interpret",
    ],
)
def proj_and_save_state(
    hidden_states: jax.Array,  # [num_tokens, hidden_size]
    wkv_wgate: jax.Array,  # [hidden_size, 2 * state_width]
    ape: jax.Array,  # [compress_ratio, state_width]
    positions: jax.Array,  # [num_tokens]
    slot_mapping: jax.Array,  # [num_tokens]
    cache: jax.Array,  # [num_pages, page_size, 4, 128] uint8 / [..., 128] int32
    compress_ratio: int,
    interpret: bool = False,
) -> jax.Array:
    """Projects hidden states and saves them to the KV cache in HBM.

    Args:
      hidden_states: Input hidden states.
      wkv_wgate: Projection weights.
      ape: Absolute Position Embeddings.
      positions: Token positions.
      slot_mapping: Mapping from token index to cache slot.
      cache: KV cache tensor in HBM (updated in-place).
      compress_ratio: APE compression ratio.

    Returns:
      The updated KV cache tensor.
    """
    num_tokens, hidden_size = hidden_states.shape
    _, state_dim = wkv_wgate.shape

    token_page_size = cache.shape[1]

    tile_m = state_dim
    tile_k = _select_tile_k(hidden_size)

    # The token dimension must be an exact multiple of `tile_n`.
    sublane_tile = 8
    tile_n = min(MAX_TILE_N, pl.cdiv(num_tokens, sublane_tile) * sublane_tile)
    padded_num_tokens = pl.cdiv(num_tokens, tile_n) * tile_n
    if padded_num_tokens != num_tokens:
        pad = padded_num_tokens - num_tokens
        hidden_states = jnp.pad(hidden_states, ((0, pad), (0, 0)))
        positions = jnp.pad(positions, (0, pad))
        slot_mapping = jnp.pad(slot_mapping, (0, pad), constant_values=-1)
        num_tokens = padded_num_tokens

    dims = Dimensions(
        size_m=state_dim,
        size_k=hidden_size,
        size_n=num_tokens,
        compress_ratio=compress_ratio,
    )

    tile_sizes = TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)

    cfgs = Configs(
        tile_sizes=tile_sizes,
        dims=dims,
        token_page_size=token_page_size,
    )

    last_dim = cache.shape[-1]
    slots_per_token = state_dim // last_dim
    scratch_shapes = [
        pltpu.VMEM((hidden_size, state_dim), wkv_wgate.dtype),
        pltpu.SemaphoreType.DMA((cfgs.num_k,)),
        pltpu.VMEM(
            (2, tile_n * acc_rows_per_token(slots_per_token, last_dim), 1, last_dim),
            jnp.float32,
        ),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SMEM((2,), jnp.int32),
        pltpu.SMEM((tile_n + 1,), jnp.int32),
        pltpu.SMEM((tile_n + 1,), jnp.int32),
    ]

    out_shape = jax.ShapeDtypeStruct(cache.shape, cache.dtype)

    return pl.pallas_call(
        functools.partial(kernel, cfgs=cfgs),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # positions
                pl.BlockSpec(memory_space=pltpu.HBM),  # hidden_states
                pl.BlockSpec(memory_space=pltpu.HBM),  # wkv_wgate
                pl.BlockSpec(memory_space=pltpu.VMEM),  # ape (prefetch to VMEM)
                pl.BlockSpec(memory_space=pltpu.HBM),  # cache
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        input_output_aliases={5: 0},  # Alias output to cache (index 5)
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name="proj_and_save_state",
    )(
        slot_mapping,
        positions,
        # Left to XLA, a VMEM placement turns the kernel's DMAs into vector copies.
        pltpu.with_memory_space_constraint(hidden_states, pltpu.HBM),
        pltpu.with_memory_space_constraint(wkv_wgate, pltpu.HBM),
        ape,
        cache,
    )
