# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 paged MQA scoring and torch/JAX operation registration."""

import functools
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import torch
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v41 import indexer
from vllm_torchtpu.kernels.deepseek_v41.index_selection import streaming_topk
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8_candidate_cache import (
    PreparedCandidates,
    compact_table_width,
    prepare_candidates,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8_paged_mqa_logits import (
    fp8_paged_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PAGE_RECORDS,
    PAGE_ROWS,
    PackedFP8Layout,
    read_fp8_records,
    write_fp8_records,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.publication import (
    representative_block_max,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    PACKED_FP8_DENSE_TOPK_THRESHOLD,
    PACKED_FP8_INDEXER_QUERY_ROWS,
    PACKED_FP8_MQA_CONFIG,
    PACKED_FP8_PREPARED_QUERY_ROWS,
    paged_mqa_key_tile,
)

from . import attention_kernels

if TYPE_CHECKING:
    from .indexer_cache import PackedFP8IndexCacheRows

_query_op = None
_write_op = None
_read_op = None
_select_op = None
_prepare_op = None
_prepared_select_op = None


def quantize_query(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q, quantize each head and fold its scale into FP32 weights."""
    assert _query_op is not None
    return _query_op(q, positions, cos_sin, weights)


def paged_indexer(
    q: torch.Tensor,
    cache: "PackedFP8IndexCacheRows",
    page_table: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    candidates: torch.Tensor,
    requests: torch.Tensor,
    *,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
    batch_max: torch.Tensor,
    score_width: int,
    stage_cache: dict[Any, Any] | None,
    source_prefix: str,
    candidate_source: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score the published K cache using FP8 Q and weights with Q scales folded in."""
    if (
        candidate_filter
        and stage_cache is not None
        and q.shape[0] > 0
        and block_size in (8, 16, 32, 64, 128)
        and 0 < candidates.shape[1] * block_size < score_width
    ):
        assert candidate_source is not None
        key = (
            "packed_fp8_candidate_keys",
            source_prefix,
            stage_cache.get(("candidate_index_version", source_prefix), 0),
            candidate_source,
            stage_cache.get(("candidate_publication", candidate_source), 0),
            # This cache lives within one batch. Hashing its symbolic row count
            # would force Dynamo to specialize the dynamic token dimension.
            candidates.shape[1],
            block_size,
            score_width,
            cache.layout,
        )
        prepared = stage_cache.get(key)
        if prepared is None:
            assert _prepare_op is not None
            prepared = _prepare_op(
                cache.banks,
                page_table.int(),
                candidates.int(),
                requests.int(),
                cache._layout_args(),
                block_size,
                score_width,
            )
            stage_cache[key] = prepared
        assert _prepared_select_op is not None
        selected = _prepared_select_op(
            q,
            weights,
            lengths.flatten().int(),
            prepared,
            topk,
            block_size,
            unordered,
            score_width,
        )
        return selected, candidates
    assert _select_op is not None
    return _select_op(
        q,
        cache.banks,
        page_table.int(),
        weights,
        lengths.flatten().int(),
        candidates.int(),
        requests.int(),
        batch_max.reshape(1).int(),
        cache._layout_args(),
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        score_width,
    )


def initialize() -> None:
    global _query_op, _write_op, _read_op, _select_op, _prepare_op, _prepared_select_op
    if _query_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_pallas_op_mesh

    def register(name, function, fake, arrays, statics, donate=()):
        op = pallas.jax_op(
            "pallas::dsv41_" + name,
            function,
            donate_argnums=donate,
            mesh=get_pallas_op_mesh(),
            input_partition_specs=cast(Any, (P(),) * arrays + (None,) * statics),
        )
        op.register_fake(fake)
        return op

    def fake_query(q, positions, cos_sin, weights):
        return torch.empty_like(q, dtype=torch.float8_e4m3fn), torch.empty_like(
            weights, dtype=torch.float32
        )

    def fake_write(banks, records, keys, after, layout):
        return tuple(torch.empty_like(bank) for bank in banks)

    def fake_read(banks, records, layout):
        return banks[0].new_empty((records.numel(), 128), dtype=torch.bfloat16)

    def fake_select(
        q,
        banks,
        page_table,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        layout,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        score_width,
    ):
        return (
            page_table.new_empty((q.shape[0], min(topk, score_width))),
            torch.empty_like(candidates),
        )

    def fake_prepare(banks, page_table, candidates, requests, layout, block, width):
        tokens, count = candidates.shape
        groups = page_table.shape[0]
        key_pages = (count * block + PAGE_RECORDS - 1) // PAGE_RECORDS
        return (
            banks[0].new_empty((groups * key_pages * PAGE_ROWS, 128)),
            torch.empty_like(candidates),
            requests.new_empty((groups + 1,)),
            torch.empty_like(requests),
            page_table.new_empty((groups, compact_table_width(count * block))),
            requests.new_empty((tokens,), dtype=torch.bool),
        )

    def fake_prepared(
        q,
        weights,
        lengths,
        prepared,
        topk,
        block,
        unordered,
        width,
    ):
        return q.new_empty((q.shape[0], min(topk, width)), dtype=torch.int32)

    _query_op = register("packed_fp8_quantize_query", _quantize_query, fake_query, 4, 0)
    _write_op = register("packed_fp8_write", _write, fake_write, 4, 1, (0,))
    _read_op = register("packed_fp8_read", _read, fake_read, 2, 1)
    _select_op = register("packed_fp8_select", select_indexer_jax, fake_select, 8, 7)
    _prepare_op = register(
        "packed_fp8_prepare_candidates", _prepare, fake_prepare, 4, 3
    )
    _prepared_select_op = register(
        "packed_fp8_select_prepared", _select_prepared, fake_prepared, 4, 4
    )


def _quantize_query(
    q: jax.Array,
    positions: jax.Array,
    cos_sin: jax.Array,
    weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    from vllm_torchtpu.kernels.deepseek_v4.rope import rope_quant

    values, scales = rope_quant(q, positions, cos_sin, power_of_two=True)
    return values, weights.astype(jnp.float32) * scales


def _write(
    banks: tuple[jax.Array, ...],
    records: jax.Array,
    keys: jax.Array,
    after: jax.Array,
    layout: tuple[int, ...],
) -> tuple[jax.Array, ...]:
    banks, _ = jax.lax.optimization_barrier((banks, after))
    return write_fp8_records(banks, records, keys, PackedFP8Layout(*layout))


def _read(
    banks: tuple[jax.Array, ...],
    records: jax.Array,
    layout: tuple[int, ...],
) -> jax.Array:
    return read_fp8_records(banks, records, PackedFP8Layout(*layout))


def _score_prepared(
    q, weights, lengths, prepared, topk, block, unordered, width, *, grouped=False
):
    keys, canonical, offsets, order, pages, valid_queries = prepared
    k = min(topk, width, canonical.shape[1] * block)
    lengths = jnp.where(valid_queries, jnp.clip(lengths, 0, width), 0)
    if not grouped:
        q, weights, lengths = q[order], weights[order], lengths[order]
    compact_lengths = jnp.sum(
        jnp.where(
            canonical >= 0,
            jnp.clip(lengths[:, None] - canonical * block, 0, block),
            0,
        ),
        axis=1,
    )
    rows = q.shape[0]
    capacity = canonical.shape[1] * block
    chunk = PACKED_FP8_PREPARED_QUERY_ROWS.get((rows, capacity, topk), rows)
    chunk = chunk if unordered and rows % chunk == 0 else rows

    def score(start, group):
        scorer = (
            fp8_paged_mqa_logits.__wrapped__ if chunk < rows else fp8_paged_mqa_logits
        )
        scores = scorer(
            q,
            keys,
            weights.astype(jnp.float32),
            compact_lengths,
            pages,
            query_start_loc=offsets,
            query_size=PACKED_FP8_MQA_CONFIG.query_tile,
            max_model_len=capacity,
            q_scale=jnp.ones_like(weights, dtype=jnp.float32),
            cache_layout=PackedFP8Layout(keys.shape[0] // PAGE_ROWS),
            query_start=start,
            query_count=chunk,
        )
        return (
            scores
            if group is None
            else xla_metadata.set_xla_metadata(scores, _scheduling_group_id=group)
        )

    if unordered:

        def select(tile, scores, group):
            return streaming_topk.select_candidate_tokens(
                scores,
                jax.lax.dynamic_slice_in_dim(canonical, tile * chunk, chunk),
                k=k,
                block_size=block,
                unordered=True,
                scheduling_group_id=group,
            )

        selected = indexer.pipeline_score_topk_bounded(
            rows // chunk,
            lambda tile: tile * chunk,
            score,
            select,
            lambda tile, src, dst: jax.lax.dynamic_update_slice_in_dim(
                dst, src, tile * chunk, axis=0
            ),
            jnp.full((rows, k), -1, jnp.int32),
        )
    else:
        scores = score(jnp.int32(0), None)
        selected = streaming_topk.select_candidate_tokens(
            scores, canonical, k=k, block_size=block, unordered=False
        )
    return jnp.pad(
        selected if grouped else jnp.empty_like(selected).at[order].set(selected),
        ((0, 0), (0, min(topk, width) - k)),
        constant_values=-1,
    )


def _select_request_blocks(scores, first_lengths, last_lengths, count, unordered):
    """Select from the first query, reserving blocks added by later queries."""
    blocks = scores.shape[1]
    columns = jnp.arange(blocks)[None, :]
    added = (columns >= first_lengths[:, None] - 1) & (columns < last_lengths[:, None])
    scores = jnp.where(added & (first_lengths[:, None] > 0), jnp.inf, scores)
    if (
        unordered
        and jax.default_backend() == "tpu"
        and count % plsc.get_sparse_core_info().num_lanes == 0
    ):
        lanes = plsc.get_sparse_core_info().num_lanes
        scores = jnp.pad(
            scores, ((0, 0), (0, -blocks % lanes)), constant_values=-jnp.inf
        )
        return attention_kernels._sparsecore_top_k(scores, count, last_lengths)
    values, indices = jax.lax.top_k(scores, count)
    return jnp.where(values > -jnp.inf, indices, -1).astype(jnp.int32)


def _select_indexer_jax(
    q: jax.Array,
    banks: tuple[jax.Array, ...],
    page_table: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    layout_args: tuple[int, ...],
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
    score_width: int,
    *,
    prefill_chunk_rows: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    layout = PackedFP8Layout(*layout_args)
    tokens, width = q.shape[0], score_width
    if tokens == 0:
        return jnp.empty((0, min(topk, width)), jnp.int32), candidates
    if q.shape[1:] != (32, 128) or width < 1:
        raise ValueError(
            "Packed FP8 model scorer requires H32/D128 and nonempty context"
        )
    if candidate_write and candidate_filter:
        raise ValueError("Candidate sources must not filter their own scores")
    if (
        candidate_filter
        and block_size in (8, 16, 32, 64, 128)
        and candidates.shape[1] * block_size < width
    ):

        def sparse():
            if not candidates.shape[1]:
                return jnp.full(
                    (q.shape[0], min(topk, width)), -1, jnp.int32
                ), candidates
            prepared = prepare_candidates(
                banks, page_table, candidates, requests, layout, block_size, width
            )
            return _score_prepared(
                q, weights, lengths, prepared, topk, block_size, unordered, width
            ), candidates

        return jax.lax.cond(
            jnp.minimum(batch_max.max(), width) <= topk,
            lambda: select_indexer_jax(
                q,
                banks,
                page_table,
                weights,
                lengths,
                candidates,
                requests,
                batch_max,
                layout_args,
                topk,
                block_size,
                False,
                False,
                unordered,
                width,
            ),
            sparse,
        )
    key_tile = paged_mqa_key_tile(
        width, layout.page_size, PACKED_FP8_MQA_CONFIG.key_tile
    )
    query_tile = PACKED_FP8_MQA_CONFIG.query_tile
    capacity = (width + key_tile - 1) // key_tile * key_tile
    pages = page_table[:, : (width + layout.page_size - 1) // layout.page_size]
    pages = jnp.pad(
        pages,
        ((0, 0), (0, capacity // layout.page_size - pages.shape[1])),
        constant_values=-1,
    )

    rows = prefill_chunk_rows or min(
        max(tokens, query_tile),
        max(query_tile, attention_kernels._INDEX_UNTILED_SCORE_ELEMENTS // capacity),
    )
    padded = (tokens + rows - 1) // rows * rows
    safe_requests = jnp.clip(requests, 0, page_table.shape[0] - 1)
    valid_requests = (requests >= 0) & (requests < page_table.shape[0])
    lengths = jnp.where(valid_requests, jnp.clip(lengths, 0, width), 0)
    inputs = (q, weights, lengths, candidates, safe_requests)
    inputs = tuple(
        jnp.pad(x, ((0, padded - tokens),) + ((0, 0),) * (x.ndim - 1)) for x in inputs
    )
    counts = jnp.bincount(inputs[-1], length=page_table.shape[0])
    query_offsets = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(counts)))
    stream_select = (
        unordered
        and not candidate_filter
        and topk == 512
        and width >= topk
        and jax.default_backend() == "tpu"
    )
    publish_shared = candidate_write and candidates.shape[1] * block_size < width
    select_topk = functools.partial(
        streaming_topk.streaming_topk,
        k=topk,
        exclude_neg_inf=True,
        reconstruct_threshold=PACKED_FP8_DENSE_TOPK_THRESHOLD,
        sorted_indices=False,
    )

    def block(start, rows, inputs, *, score_only=False, group=None, publication=None):
        shared_inputs = score_only and prefill_chunk_rows is None
        queries, row_weights, row_lengths, row_candidates, owners = (
            inputs
            if shared_inputs
            else tuple(jax.lax.dynamic_slice_in_dim(x, start, rows) for x in inputs)
        )
        offsets = (
            query_offsets if shared_inputs else jnp.clip(query_offsets - start, 0, rows)
        )
        pool = banks[0] if len(banks) == 1 else banks

        # Inline inside the outer model jit so scheduling metadata reaches
        # the Pallas call instead of being lost at the inner jit boundary.
        scorer = (
            fp8_paged_mqa_logits.__wrapped__ if score_only else fp8_paged_mqa_logits
        )
        scores = scorer(
            queries,
            pool,
            row_weights.astype(jnp.float32),
            row_lengths,
            pages,
            query_start_loc=offsets,
            max_model_len=capacity if stream_select else width,
            q_scale=jnp.ones_like(row_weights, dtype=jnp.float32),
            query_size=query_tile,
            cache_layout=layout,
            query_start=start if shared_inputs else 0,
            query_count=rows,
            reuse_keys=prefill_chunk_rows is not None,
        )
        if score_only:
            return (
                scores
                if group is None
                else xla_metadata.set_xla_metadata(scores, _scheduling_group_id=group)
            )
        if stream_select:
            selected = select_topk(scores, valid_lengths=row_lengths)
        else:
            selected, row_candidates = attention_kernels.select_indices_jax(
                scores,
                row_lengths,
                row_candidates,
                topk,
                block_size,
                candidate_write and publication is None,
                candidate_filter,
                unordered,
                batch_max,
            )
        if publication is not None:
            first, first_lengths, last_lengths = publication
            shared = _select_request_blocks(
                representative_block_max(
                    scores, jnp.clip(first - start, 0, rows - 1), block=block_size
                ),
                (first_lengths + block_size - 1) // block_size,
                (last_lengths + block_size - 1) // block_size,
                candidates.shape[1],
                unordered,
            )
            row_candidates = shared[owners]
        return selected, row_candidates

    def select_serial():
        order = jnp.argsort(inputs[-1], stable=True)
        sorted_inputs = tuple(x[order] for x in inputs)
        output = jnp.full((padded, min(topk, width)), -1, jnp.int32)
        published = jnp.full((padded, candidates.shape[1]), -1, jnp.int32)
        publication = None
        if publish_shared:
            first = jax.ops.segment_min(
                jnp.where(sorted_inputs[2] > 0, jnp.arange(padded), padded),
                sorted_inputs[-1],
                page_table.shape[0],
            )
            first_lengths = jnp.where(
                first < padded, sorted_inputs[2][jnp.minimum(first, padded - 1)], 0
            )
            last_lengths = jax.ops.segment_max(
                sorted_inputs[2], sorted_inputs[-1], page_table.shape[0]
            )
            publication = first, first_lengths, last_lengths

        # Unroll SparseCore calls to avoid XLA loop lowering failures;
        # barriers bound score-buffer lifetimes.
        for index in range(padded // rows):
            start, output, published = jax.lax.optimization_barrier(
                (jnp.int32(index * rows), output, published)
            )
            result = block(start, rows, sorted_inputs, publication=publication)
            output, published = (
                jax.lax.dynamic_update_slice_in_dim(dst, src, start, axis=0)
                for dst, src in zip((output, published), result)
            )
        inverse = jnp.argsort(order)
        if publication is not None:
            first = publication[0]
            shared = published[jnp.minimum(first, padded - 1)]
            shared = jnp.where(first[:, None] < padded, shared, -1)
            published = shared[inputs[-1]]
            published = published.at[:tokens].set(
                jnp.where(
                    valid_requests[:, None],
                    published[:tokens],
                    -1,
                )
            )
        else:
            published = published[inverse]
        return output[inverse][:tokens], published[:tokens]

    short = jnp.minimum(batch_max.max(), width) <= topk

    chunk_rows = prefill_chunk_rows or PACKED_FP8_INDEXER_QUERY_ROWS.get(
        (tokens, width, candidates.shape[1], topk, block_size, len(banks))
    )
    if not stream_select or chunk_rows is None or chunk_rows > rows or padded != tokens:
        return select_serial()

    def select_pipelined():
        def score(start, group):
            return block(start, chunk_rows, inputs, score_only=True, group=group)

        def select(tile, scores, group):
            row_lengths = jax.lax.dynamic_slice_in_dim(
                inputs[2], tile * chunk_rows, chunk_rows
            )
            selected = select_topk(
                scores,
                valid_lengths=row_lengths,
                scheduling_group_id=group,
            )
            if publish_shared:
                return selected, representative_block_max(
                    scores, jnp.arange(0, chunk_rows, query_tile), block=block_size
                )
            return selected

        def write(tile, selected, output):
            if publish_shared:
                return tuple(
                    jax.lax.dynamic_update_slice_in_dim(dst, src, tile * size, axis=0)
                    for dst, src, size in zip(
                        output, selected, (chunk_rows, chunk_rows // query_tile)
                    )
                )
            return jax.lax.dynamic_update_slice_in_dim(
                output, selected, tile * chunk_rows, axis=0
            )

        output = jnp.full((tokens, topk), -1, jnp.int32)
        if publish_shared:
            output = (
                output,
                jnp.empty(
                    (tokens // query_tile, (capacity + block_size - 1) // block_size),
                    jnp.float32,
                ),
            )
        selected = indexer.pipeline_score_topk_bounded(
            padded // chunk_rows,
            lambda tile: tile * chunk_rows,
            score,
            select,
            write,
            output,
        )
        if publish_shared:
            selected, representative_scores = selected
            owners = inputs[-1][::query_tile]
            first_lengths = inputs[2][::query_tile]
            last_lengths = inputs[2][jnp.clip(query_offsets[1:] - 1, 0, tokens - 1)]
            shared = _select_request_blocks(
                representative_scores,
                (first_lengths + block_size - 1) // block_size,
                (last_lengths[owners] + block_size - 1) // block_size,
                candidates.shape[1],
                unordered,
            )
            first = query_offsets[:-1] // query_tile
            shared = jnp.where(
                (counts > 0)[:, None],
                shared[jnp.minimum(first, owners.size - 1)],
                -1,
            )
            return selected, jnp.where(
                valid_requests[:, None],
                shared[safe_requests],
                -1,
            )
        return selected, candidates

    # Already grouped requests need neither sorting nor output reordering.
    # Decode preserves complete six-query groups. Prefill chunks instead match
    # the serial score-buffer boundaries, including groups cut by that boundary.
    # Both keep the same four-row/two-row head reduction and BF16 rounding.
    grouped = jnp.all(inputs[-1][1:] >= inputs[-1][:-1])
    complete = (
        jnp.bool_(True)
        if prefill_chunk_rows is not None
        else jnp.all(counts % query_tile == 0)
    )
    if publish_shared:
        # Ordered causal lengths let offsets identify both representatives and
        # final visibility; irregular or masked leading queries use exact routing.
        first_lengths = inputs[2][jnp.minimum(query_offsets[:-1], tokens - 1)]
        complete &= jnp.all((counts == 0) | (first_lengths > 0)) & jnp.all(
            (inputs[-1][1:] != inputs[-1][:-1]) | (inputs[2][1:] >= inputs[2][:-1])
        )
    return jax.lax.cond(grouped & complete & ~short, select_pipelined, select_serial)


def select_indexer_jax(
    q: jax.Array,
    banks: tuple[jax.Array, ...],
    page_table: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    layout_args: tuple[int, ...],
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
    score_width: int,
) -> tuple[jax.Array, jax.Array]:
    """Select exact FP8 scores with a bounded, overlapped prefill working set."""
    arguments = (
        q,
        banks,
        page_table,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        layout_args,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
    )
    layout = PackedFP8Layout(*layout_args)
    width = paged_mqa_key_tile(
        score_width, layout.page_size, PACKED_FP8_MQA_CONFIG.key_tile
    )
    capacity = (score_width + width - 1) // width * width
    chunk = max(
        PACKED_FP8_MQA_CONFIG.query_tile,
        attention_kernels._INDEX_UNTILED_SCORE_ELEMENTS // max(capacity, 1),
    )
    if (
        width >= score_width
        or chunk >= q.shape[0]
        or q.shape[0] % chunk
        or candidate_write
        or candidate_filter
        or not unordered
        or topk > width
        or jax.default_backend() != "tpu"
    ):
        return _select_indexer_jax(*arguments, score_width)

    # Preserve the existing score-buffer row boundaries, but use one retained
    # K tile when it covers every live key. Longer contexts keep the old path.
    live = (requests >= 0) & (requests < page_table.shape[0])
    longest = jnp.where(live, lengths, 0).max()
    grouped = jnp.all(requests[1:] >= requests[:-1])
    return jax.lax.cond(
        grouped & (longest <= width),
        lambda: _select_indexer_jax(*arguments, width, prefill_chunk_rows=chunk),
        lambda: _select_indexer_jax(*arguments, score_width),
    )


def _prepare(
    banks: tuple[jax.Array, ...],
    page_table: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    layout: tuple[int, ...],
    block: int,
    width: int,
) -> tuple[jax.Array, ...]:
    return tuple(
        prepare_candidates(
            banks,
            page_table,
            candidates,
            requests,
            PackedFP8Layout(*layout),
            block,
            width,
            grouped=True,
        )
    )


def _select_prepared(
    q: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    prepared: tuple[jax.Array, ...],
    topk: int,
    block: int,
    unordered: bool,
    width: int,
) -> jax.Array:
    candidates = PreparedCandidates(*prepared)
    lengths = jnp.where(candidates.valid_queries, lengths, 0)

    def short():
        ids = jnp.broadcast_to(
            jnp.arange(min(topk, width), dtype=jnp.int32),
            (q.shape[0], min(topk, width)),
        )
        return jnp.where(ids < lengths[:, None], ids, -1)

    return jax.lax.cond(
        jnp.max(lengths) <= topk,
        short,
        lambda: _score_prepared(
            q, weights, lengths, candidates, topk, block, unordered, width, grouped=True
        ),
    )
