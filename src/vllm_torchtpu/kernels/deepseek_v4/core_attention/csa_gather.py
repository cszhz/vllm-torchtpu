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
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout

# Large primes that spread the cache rows of the -1 (padding) top-k entries,
# so their gathers do not all hit one row.
PAD_PAGE_PRIME = 104729
PAD_OFFSET_PRIME = 15485863
# The inner pipeline issues step `r + 1`'s gathers before running step `r`,
# so step `r` looks up the rows of step `r + 2`, under its own DMAs; only the
# first two steps' lookups run ahead of the gather.
LOOKUP_STEPS_AHEAD = 2
# 16-entry vectors per software-pipelined iteration of the lookup loop.
LOOKUP_UNROLL = 8
# Words of a vector subcore's VMEM `csa_gather_topk` may spend on its copies
# of the page table and the token -> sequence map.
LOOKUP_VMEM_WORDS = 64 * 1024


@dataclasses.dataclass(frozen=True)
class TopkLookup:
    """Static geometry of `csa_gather_topk`'s in-kernel row lookup."""

    token_start: int  # first token of the chunk in the top-k array
    num_tokens: int  # tokens in the chunk
    rows_per_tile: int  # top-k rows interleaved per 128-entry segment
    page_size: int
    pages_per_seq: int
    num_seqs: int
    total_pages: int


def _lookup_rows(
    topk_ref,
    idx_ref,
    seq_ref,
    table_ref,
    *,
    token,
    lookup,
    top_k,
    num_lanes,
    first=0,
    count=None,
):
    """Writes the cache rows of `token`'s top-k entries `[first, first +
    count)` (default: all) to `idx_ref`.

    Entry `x` >= 0 is a compressed token of the token's sequence `s` and
    reads row `page_table[s, x // page_size] * page_size + x % page_size`. A
    -1 entry reads a row spread by its flat index `token * top_k + k`. Both
    are the exact int32 expressions the TensorCore used to evaluate.
    """
    count = top_k if count is None else count
    page_shift = lookup.page_size.bit_length() - 1
    seq = plsc.load_gather(seq_ref, [jnp.full((num_lanes,), token, jnp.int32)])
    base = jnp.clip(seq, 0, lookup.num_seqs - 1) * lookup.pages_per_seq
    lane = lax.broadcasted_iota(jnp.int32, (num_lanes,), 0)
    segment_words = csa_cache_layout.ROW_WORDS

    @plsc.parallel_loop(0, count // num_lanes, unroll=LOOKUP_UNROLL)
    def _(v):
        k = first + v * num_lanes
        x = topk_ref[0, k // segment_words, 0, pl.ds(k % segment_words, num_lanes)]
        page = jnp.clip(
            lax.shift_right_arithmetic(x, page_shift), 0, lookup.pages_per_seq - 1
        )
        phys = plsc.load_gather(table_ref, [base + page])
        row = phys * lookup.page_size + (x & (lookup.page_size - 1))
        flat = token * top_k + k + lane
        pad_page = (flat * PAD_PAGE_PRIME) % lookup.total_pages
        pad_offset = (flat * PAD_OFFSET_PRIME) & (lookup.page_size - 1)
        pad_row = pad_page * lookup.page_size + pad_offset
        idx_ref[pl.ds(k, num_lanes)] = jnp.where(x == -1, pad_row, row)


def _topk_row_spec(lookup, top_k, num_cores, core_index):
    """Block `b` of subcore `core_index`: its token's top-k row, as
    `top_k // 128` segments of 128 entries. A subcore past the chunk's end
    fetches the last row of the chunk, which it then skips."""

    def topk_row(b):
        local = jnp.minimum(b * num_cores + core_index, lookup.num_tokens - 1)
        token = lookup.token_start + local
        return token // lookup.rows_per_tile, 0, token % lookup.rows_per_tile, 0

    segments = top_k // csa_cache_layout.ROW_WORDS
    return pl.BlockSpec((1, segments, 1, csa_cache_layout.ROW_WORDS), topk_row)


def main_kernel(
    nope_in_hbm_ref: Any,
    rope_in_hbm_ref: Any,
    indices_hbm_ref: Any,
    valid_indices_ref: Any,
    *refs: Any,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
    num_streams: int,
    top_k: int,
    lookup: TopkLookup | None = None,
):
    """Gathers the rows `indices_hbm_ref` names, or with `lookup`, the rows
    of the top-k entries it holds (see `csa_gather_topk`)."""
    if lookup is None:
        nope_out_hbm_ref, rope_out_hbm_ref, nope_sem = refs
    else:
        (
            seq_ids_hbm_ref,
            page_table_hbm_ref,
            nope_out_hbm_ref,
            rope_out_hbm_ref,
            nope_sem,
            idx_vmem_ref,
            seq_vmem_ref,
            table_vmem_ref,
            lookup_sem,
        ) = refs
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores
    if lookup is None:
        num_blocks = pl.cdiv(indices_hbm_ref.shape[0], block_size)
    else:
        # One token per subcore and block.
        assert row_chunk_size == top_k
        num_blocks = pl.cdiv(lookup.num_tokens, num_cores)

    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    nope_in_cols = nope_in_hbm_ref.shape[1]
    rope_in_cols = rope_in_hbm_ref.shape[1]
    # A `bf16[n, 64]` output is tiled to 128 lanes, so XLA pads it and it
    # costs twice its own bytes -- on this kernel's write and on the
    # consumer's read. The output is 128 lanes wide instead, pairing entry `i`
    # of each `top_k`-entry period with entry `i + top_k // 2`, which the
    # consumer splits apart with a lane slice and a row concatenate.
    rope_out_cols = rope_out_hbm_ref.shape[1] // 2

    def process_rope(gather_ref, out_ref, out_row_base=0):
        # bf16 output rows 2t and 2t + 1 are the low and high halves of int32
        # row t, so each output word pairs one rope value of an even entry
        # with the same value of the next entry. A cache rope word k holds
        # values k (low) and k + 32 (high).
        for t in range(num_simd_lanes // 2):
            even = gather_ref[pl.ds(2 * t, 1), :]
            odd = gather_ref[pl.ds(2 * t + 1, 1), :]
            row = pl.ds(out_row_base + t, 1)
            out_ref[row, pl.ds(0, rope_in_cols)] = jnp.bitwise_or(
                jnp.bitwise_and(even, 0xFFFF), jnp.left_shift(odd, 16)
            )
            out_ref[row, pl.ds(rope_in_cols, rope_in_cols)] = jnp.bitwise_or(
                lax.shift_right_logical(even, 16),
                jnp.bitwise_and(odd, jnp.int32(-65536)),  # 0xFFFF0000
            )

    def outer_pipeline(in_ref, valid_ref):
        valid_indices = valid_ref[pl.ds(0, row_subchunk_size)][0]
        b = pl.program_id(0)
        out_row_base = (b * num_cores + core_index) * num_row_subchunks
        idx_ref = in_ref if lookup is None else idx_vmem_ref
        if lookup is not None:

            @pl.when(b == 0)
            def _():
                for copy in lookup_copies:
                    copy.wait()

        # Subchunk handled by stream `s` at inner step `r`. `num_streams`
        # independent `pl.Indirect` gathers run concurrently per step, keeping
        # several gather DMAs in flight to raise effective read bandwidth. A
        # step's subchunks are consecutive, so they lie in one half of a
        # period.
        def subchunk(r, s):
            return r * num_streams + s

        def idx_window(r, s):
            return idx_ref[pl.ds(subchunk(r, s) * row_subchunk_size, row_subchunk_size)]

        # int32 rows produced per stream in the rope output (2 tokens per row).
        rope_rows_per_stream = row_subchunk_size // 2
        num_steps = num_row_subchunks // num_streams
        step_entries = num_streams * row_subchunk_size

        def lookup_step_rows(step, steps):
            """Rows of the entries inner steps `[step, step + steps)` gather."""
            _lookup_rows(
                in_ref,
                idx_vmem_ref,
                seq_vmem_ref,
                table_vmem_ref,
                token=lookup.token_start + b * num_cores + core_index,
                lookup=lookup,
                top_k=top_k,
                num_lanes=num_simd_lanes,
                first=step * step_entries,
                count=steps * step_entries,
            )

        def _body(*refs):
            r = pl.program_id(0)
            nope_g = refs[0 * num_streams : 1 * num_streams]
            rope_g = refs[1 * num_streams : 2 * num_streams]
            rope_o = refs[2 * num_streams]

            # nope needs no vector work at all. The nope output keeps the
            # cache's raw per-token layout. The gathered row *is* the output
            # row.
            #
            # DMA the gather buffer straight to HBM rather than routing it
            # through a pipeline output buffer: staging it there costs a
            # VMEM->VMEM copy.
            nope_copies = []
            for s in range(num_streams):
                out_row = (out_row_base + subchunk(r, s)) * row_subchunk_size
                copy = pltpu.make_async_copy(
                    nope_g[s],
                    nope_out_hbm_ref.at[pl.ds(out_row, row_subchunk_size)],
                    nope_sem.at[s],
                )
                copy.start()
                nope_copies.append(copy)

            if lookup is not None and num_steps > LOOKUP_STEPS_AHEAD:

                @pl.when(r + LOOKUP_STEPS_AHEAD < num_steps)
                def _():
                    lookup_step_rows(r + LOOKUP_STEPS_AHEAD, 1)

            for s in range(num_streams):
                process_rope(
                    gather_ref=rope_g[s],
                    out_ref=rope_o,
                    out_row_base=s * rope_rows_per_stream,
                )

            # Wait for all nope DMAs to complete.
            for copy in nope_copies:
                copy.wait()

        # Multiple parallel `pl.Indirect` gathers hide random-access read
        # latency. The writes are contiguous and need no extra parallelism:
        # rope writes one block per step, and nope issues one DMA per stream
        # only because each stream has its own gather buffer.

        # nope: gather int32 row == index (1 int32 row per entry).
        nope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), nope_in_cols),
                lambda r, s=s: (idx_window(r, s), 0),
            )
            for s in range(num_streams)
        )
        # rope: gather the entry's own 32-word quarter row == index.
        rope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), rope_in_cols),
                lambda r, s=s: (idx_window(r, s), 0),
            )
            for s in range(num_streams)
        )

        # One rope output block per step, covering all `num_streams`
        # subchunks: a step in the first half of a period fills int32 lanes
        # [0, rope_out_cols) of its rows, a step in the second half lanes
        # [rope_out_cols, 2 * rope_out_cols) of the same rows.
        steps_per_half = top_k // 2 // (num_streams * row_subchunk_size)

        def rope_out_block(r):
            half, step = divmod(out_row_base // num_streams + r, steps_per_half)
            return (half // 2) * steps_per_half + step, half % 2

        rope_out_spec = pl.BlockSpec(
            (num_streams * rope_rows_per_stream, rope_out_cols), rope_out_block
        )
        core_start_idx = out_row_base * row_subchunk_size
        run = core_start_idx < valid_indices
        if lookup is not None:
            run &= b * num_cores + core_index < lookup.num_tokens

        @pl.when(run)
        def _run_gather():
            if lookup is not None:
                lookup_step_rows(0, min(LOOKUP_STEPS_AHEAD, num_steps))
            pltpu.emit_pipeline(
                _body,
                grid=(num_steps,),
                in_specs=nope_in_specs + rope_in_specs,
                out_specs=(rope_out_spec,),
            )(
                *([nope_in_hbm_ref] * num_streams),
                *([rope_in_hbm_ref] * num_streams),
                rope_out_hbm_ref,
            )

    if lookup is None:
        in_spec = pl.BlockSpec(
            (row_chunk_size,), lambda b: (b * num_cores + core_index,)
        )
    else:
        in_spec = _topk_row_spec(lookup, top_k, num_cores, core_index)
        lookup_copies = (
            pltpu.make_async_copy(seq_ids_hbm_ref, seq_vmem_ref, lookup_sem.at[0]),
            pltpu.make_async_copy(page_table_hbm_ref, table_vmem_ref, lookup_sem.at[1]),
        )
        for copy in lookup_copies:
            copy.start()

    pltpu.emit_pipeline(
        outer_pipeline,
        grid=(num_blocks,),
        in_specs=(
            in_spec,
            pl.BlockSpec(
                (row_subchunk_size,),
                lambda b: (0,),
            ),
        ),
    )(indices_hbm_ref, valid_indices_ref)


@functools.partial(jax.jit, static_argnames=("top_k",))
def csa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    num_valid_indices: jax.Array | None = None,
    *,
    top_k: int = 1024,
) -> tuple[jax.Array, jax.Array]:
    """Fused SparseCore gather of the nope and rope caches.

    Args:
      nope_cache: (total_pages, page_size, 128) int32, one row per token: the
        (4, 128) uint8 nope slab (see `csa_cache_layout`).
      rope_cache: (total_pages, page_size // 4, 128) int32, 32 words per token
        (see `csa_cache_layout`).
      indices: (N,) int32. Token indices into the caches; N a multiple of
        top_k.
      num_valid_indices: Optional (1,) or scalar int32. Number of valid indices to
        gather, a multiple of top_k. A subcore skips every top_k period that
        starts at or past this count, leaving those output rows unwritten.
      top_k: the consumer's row block (the attention kernel's top-k), a
        multiple of 128. `rope_out` pairs entry i of a period with entry
        i + top_k // 2.

    Returns:
      Both outputs are int32 with 128 lanes, which XLA lays out byte-for-byte
      like the uint8 / bf16 arrays they encode, so the consumer views them with
      a free `ref.bitcast`.
      nope_out: (N, 128) int32. Row i is entry i's (4, 128) uint8 nope slab.
      rope_out: (N // 4, 128) int32, i.e. (N // 2, 128) bf16 once bitcast.
        For i < top_k // 2, bf16 row `period * (top_k // 2) + i` holds entry
        i of that period in lanes 0:64 and entry i + top_k // 2 in lanes
        64:128 -- 128 lanes so XLA does not pad the buffer to twice its size.
        The consumer restores a period's (top_k, 64) from its
        (top_k // 2, 128) block `blk` with
        `jnp.concatenate([blk[:, :64], blk[:, 64:]], axis=0)`.
    """
    assert indices.ndim == 1, "Indices must be 1D."
    assert nope_cache.dtype == rope_cache.dtype, "Caches must share a dtype."
    assert nope_cache.dtype == jnp.int32, "Caches must be int32."
    assert nope_cache.shape[2] == csa_cache_layout.ROW_WORDS
    assert rope_cache.shape[2] == csa_cache_layout.ROW_WORDS

    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    out_size = indices.size
    num_cores = sc_info.num_cores * sc_info.num_subcores
    assert out_size % top_k == 0
    block_size = top_k * num_cores
    out_pad_size = ((out_size + block_size - 1) // block_size) * block_size - out_size
    if out_pad_size:
        # spread the padding to avoid hotspots
        num_tokens = nope_cache.shape[0] * nope_cache.shape[1]
        pad = (jnp.arange(out_pad_size, dtype=indices.dtype) * 104729) % num_tokens
        indices = jnp.concatenate([indices, pad])
    nope_out, rope_out = _launch(
        nope_cache,
        rope_cache,
        (indices,),
        out_size if num_valid_indices is None else num_valid_indices,
        out_rows=out_size + out_pad_size,
        top_k=top_k,
    )
    return nope_out[:out_size], rope_out[: out_size // 4]


@functools.partial(
    jax.jit, static_argnames=("token_start", "num_tokens", "top_k", "pages_per_seq")
)
def csa_gather_topk(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    topk_indices: jax.Array,
    seq_ids: jax.Array,
    page_indices: jax.Array,
    num_valid_indices: jax.Array,
    *,
    token_start: int,
    num_tokens: int,
    top_k: int,
    pages_per_seq: int,
) -> tuple[jax.Array, jax.Array]:
    """`csa_gather` of tokens `[token_start, token_start + num_tokens)`, given
    their top-k entries rather than cache rows.

    The SparseCore maps each entry to its cache row itself (see
    `_lookup_rows`), so nothing runs on the TensorCore between the top-k and
    the gather.

    Args:
      topk_indices: (T, top_k) int32, per token the compressed tokens of its
        sequence to read, -1 for none.
      seq_ids: (>= T,) int32, each token's sequence.
      page_indices: (num_seqs * pages_per_seq,) int32 page table.
      num_valid_indices: entries of the chunk to gather, as in `csa_gather`.

    Returns:
      `csa_gather`'s outputs for the chunk's `num_tokens * top_k` entries.
    """
    num_all_tokens = topk_indices.shape[0]
    assert topk_indices.shape[1] == top_k and topk_indices.dtype == jnp.int32
    assert token_start >= 0 and token_start + num_tokens <= num_all_tokens
    assert top_k % csa_cache_layout.ROW_WORDS == 0
    _, page_size, _ = nope_cache.shape
    assert page_size & (page_size - 1) == 0, "page_size must be a power of two."
    assert page_indices.shape[0] % pages_per_seq == 0
    assert seq_ids.shape[0] + page_indices.shape[0] <= LOOKUP_VMEM_WORDS
    # The (T, top_k) array is tiled (8, 128): view it in that byte order, so
    # handing it to the SparseCore needs no relayout.
    segments = top_k // csa_cache_layout.ROW_WORDS
    rows_per_tile = 8 if num_all_tokens % 8 == 0 else 1
    topk_tiles = topk_indices.reshape(
        num_all_tokens // rows_per_tile,
        rows_per_tile,
        segments,
        csa_cache_layout.ROW_WORDS,
    ).transpose(0, 2, 1, 3)
    lookup = TopkLookup(
        token_start=token_start,
        num_tokens=num_tokens,
        rows_per_tile=rows_per_tile,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        num_seqs=page_indices.shape[0] // pages_per_seq,
        total_pages=nope_cache.shape[0],
    )
    return _launch(
        nope_cache,
        rope_cache,
        (topk_tiles,),
        num_valid_indices,
        out_rows=num_tokens * top_k,
        top_k=top_k,
        lookup=lookup,
        lookup_inputs=(seq_ids, page_indices),
    )


def _launch(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    inputs: tuple[jax.Array],
    num_valid_indices: jax.Array | int,
    *,
    out_rows: int,
    top_k: int,
    lookup: TopkLookup | None = None,
    lookup_inputs: tuple[jax.Array, ...] = (),
) -> tuple[jax.Array, jax.Array]:
    """Runs `main_kernel` over `inputs` and returns its two outputs."""
    # Free reshapes: a 128-lane int32 array's T(8,128) tiling is byte-identical
    # to row-major, and the kernel takes its operands untiled
    # (`use_tc_tiling_on_sc=False`), so token t's rope is words
    # [32 t, 32 t + 32).
    nope_cache = nope_cache.reshape(-1, csa_cache_layout.ROW_WORDS)
    rope_cache = rope_cache.reshape(-1, csa_cache_layout.ROPE_WORDS)
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    num_simd_lanes = sc_info.num_lanes
    row_subchunk_size = num_simd_lanes
    valid_indices = jnp.full((row_subchunk_size,), num_valid_indices, dtype=jnp.int32)

    # `num_streams` independent `pl.Indirect` gathers are issued per
    # pipeline step to keep multiple gather DMAs in flight.
    # See `outer_pipeline` for details.
    num_streams = 4
    num_row_subchunks = top_k // row_subchunk_size
    # One DMA semaphore per stream for the direct nope gather-buffer -> HBM
    # copies issued in `main_kernel`.
    scratch_types = [pltpu.SemaphoreType.DMA((num_streams,))]
    if lookup is not None:
        seq_ids, page_indices = lookup_inputs
        scratch_types += [
            pltpu.VMEM((top_k,), jnp.int32),  # the token's cache rows
            pltpu.VMEM(seq_ids.shape, jnp.int32),
            pltpu.VMEM(page_indices.shape, jnp.int32),
            pltpu.SemaphoreType.DMA((2,)),
        ]
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    nope_out, rope_out = cast(
        Any,
        pl.kernel(
            functools.partial(
                main_kernel,
                core_axis_name=vector_mesh.core_axis_name,
                subcore_axis_name=vector_mesh.subcore_axis_name,
                num_row_subchunks=num_row_subchunks,
                num_streams=num_streams,
                top_k=top_k,
                lookup=lookup,
            ),
            out_type=(
                jax.ShapeDtypeStruct((out_rows, csa_cache_layout.ROW_WORDS), jnp.int32),
                jax.ShapeDtypeStruct(
                    (out_rows // 4, csa_cache_layout.ROW_WORDS), jnp.int32
                ),
            ),
            scratch_types=tuple(scratch_types),
            compiler_params=pltpu.CompilerParams(
                use_tc_tiling_on_sc=False,
                needs_layout_passes=True,
                disable_bounds_checks=True,
            ),
            mesh=vector_mesh,
            name="sc_csa_gather" if lookup is None else "sc_csa_gather_topk",
        )(nope_cache, rope_cache, *inputs, valid_indices, *lookup_inputs),
    )  # pyrefly: ignore[not-iterable]
    return nope_out, rope_out
