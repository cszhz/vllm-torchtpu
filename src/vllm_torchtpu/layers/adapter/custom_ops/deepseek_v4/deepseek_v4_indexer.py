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
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from jax.sharding import PartitionSpec as P
from torch.distributed import _functional_collectives as funcol
from torch_tpu._internal import pallas
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.models.deepseek_v4.attention import DeepseekV4Indexer, DeepseekV4IndexerCache
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.rope import rope_quant
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    check_seq_along_lane_in_quad_pages,
    merge_kv_split_topk,
    streamindex_topk,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor import (
    VllmDeepseekCompressor,
    indexer_kv_layout_name,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_pallas_op_mesh,
)
from vllm_torchtpu.utils import align_to

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

logger = init_logger(__name__)

_indexer_op_cache = {}

# (decode, verify window, mixed) KV pages per block for each K cache record
# layout. Tuned on tpu7x at max-model-len 1M (256 compressed tokens a page):
# large KV blocks, and decode sequences scored one at a time rather than in
# batches of 4.
_KV_PAGES_PER_BLOCK = {
    "head_along_sublane": (16, 32, 8),
    "seq_along_lane": (32, 32, 16),
}

# Token buckets up to which the mixed pass scores 32-token q blocks instead of
# 128-token ones (over 16-page KV blocks either way). Its sequences there are
# chunks of at most a few dozen tokens next to the verify windows, and a
# 128-row q block is mostly padding. Measured on tpu7x (kernel time, T128, the
# 32-token block against the 128-token one): 30-90 token chunks at 40-150k
# context -15% to -65%, a 120-token chunk at 4.5k -36%, a 128-token chunk at
# 100k +11%; at T256/T512 the large block stays ahead. 16 KV pages against 8
# under the 32-token block: equal at 100k, -2% to -6% at 300-600k.
_SMALL_MIXED_BLOCK_MAX_TOKENS = 128

# Token buckets from which the call is cut into 4 pieces, so each piece's
# SparseCore top-k overlaps the next piece's scoring and only the last quarter
# is left exposed. Measured on tpu7x (one core, the 4-piece call against the
# 2-piece split, bitexact): T1024 -5%, T2048 -9..-10% at 16-300k context.
# 8 pieces at T2048 were 4x slower.
_FOUR_PIECE_MIN_TOKENS = 1024

BATCH_AXIS = None


# Module-level so `pallas.jax_op` can trace and register it as a torch op.
def _indexer_jax(
    q: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    uniform_q_len: int | None,
    kv_layout: str,
) -> jax.Array:
    """Top-k indices i32[T, k]."""
    return _indexer_impl(
        q,
        positions,
        cos_sin_cache,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        None,
        k=k,
        compression_ratio=compression_ratio,
        softmax_scale=softmax_scale,
        n_head=n_head,
        uniform_q_len=uniform_q_len,
        kv_layout=kv_layout,
    )


def _indexer_split_jax(
    q: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    split_rank: jax.Array,  # i32[1]
    *,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    uniform_q_len: int | None,
    kv_layout: str,
    kv_split: int,
) -> jax.Array:
    """Rank `split_rank`'s share of the top-k (`streamindex_topk`'s
    kv_split), i32[T, 2k] (indices, score bits), for `_merge_kv_split_jax`."""
    return _indexer_impl(
        q,
        positions,
        cos_sin_cache,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        split_rank[0],
        k=k,
        compression_ratio=compression_ratio,
        softmax_scale=softmax_scale,
        n_head=n_head,
        uniform_q_len=uniform_q_len,
        kv_layout=kv_layout,
        kv_split=kv_split,
    )


def _indexer_impl(
    q: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    split_rank: jax.Array | None,
    *,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    uniform_q_len: int | None,
    kv_layout: str,
    kv_split: int = 1,
) -> jax.Array:
    out_width = k if kv_split == 1 else 2 * k
    if cache_kv.shape[0] == 0:
        # Profiling-shape trace: the kernel is skipped. Unused inputs stay in
        # the compiled signature because torch_tpu jits with keep_unused=True.
        return jnp.zeros((q.shape[0], out_width), dtype=jnp.int32)
    if kv_layout == "seq_along_lane":
        check_seq_along_lane_in_quad_pages(cache_kv.shape, q.shape[-1])

    def _indexer_local(
        q,
        positions,
        cos_sin_cache,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ):
        # One kernel rotates the queries and quantizes each row on the way out.
        # Note: vLLM's implementation rounds the scale factors up to the next
        # power of 2, but the plain `abs_max / dtype_max` the kernel returns is
        # sufficient here.
        q_quant, q_scales = rope_quant(
            q, positions, cos_sin_cache, quant_dtype=jnp.float8_e4m3fn
        )

        kv_pages_per_block = _KV_PAGES_PER_BLOCK[kv_layout]
        mixed_q_block = 128
        if (
            kv_layout == "seq_along_lane"
            and q.shape[0] <= _SMALL_MIXED_BLOCK_MAX_TOKENS
        ):
            kv_pages_per_block = (*kv_pages_per_block[:2], 16)
            mixed_q_block = 32

        # Fold the query quantization scales into the weights.
        weights = indexer_weights * softmax_scale * (n_head**-0.5) * q_scales

        out = streamindex_topk(
            q=q_quant,
            indexer_weights=weights,
            cache_kv=cache_kv,
            seq_lens=seq_lens,
            page_indices=page_indices.flatten(),
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            k=k,
            compression_ratio=compression_ratio,
            num_kv_pages_per_block=kv_pages_per_block,
            num_queries_per_block=(1, 128, mixed_q_block),
            decode_req_batch_size=1,
            uniform_q_len=uniform_q_len,
            kv_layout=kv_layout,
            chunk_tokens=envs.TPU_STREAMIDX_CHUNK_TOKENS
            or (q.shape[0] // 4 if q.shape[0] >= _FOUR_PIECE_MIN_TOKENS else None),
            split_mixed_blocks=envs.TPU_STREAMIDX_SPLIT_MIXED
            and q.shape[0] >= envs.TPU_STREAMIDX_SPLIT_MIXED_MIN_TOKENS,
            strict_causal=envs.TPU_STREAMIDX_STRICT_CAUSAL,
            # A kv-split share's rows fit one subcore, so its top-k takes the
            # by-length pass. Compiled next to the merge's SparseCore top-k,
            # shares from the cooperative (shared-memory) pass came back wrong
            # on tpu7x while the same ops run one by one were right.
            sc_topk_one_launch=envs.TPU_DSV4_SC_TOPK_ONE_LAUNCH and kv_split == 1,
            sc_topk_coop_max_rows=(
                envs.TPU_DSV4_SC_TOPK_COOP_MAX_ROWS if kv_split == 1 else 0
            ),
            kv_split=kv_split,
            kv_split_rank=0 if split_rank is None else split_rank,
        )
        return out if kv_split == 1 else jnp.concatenate(out, axis=1)

    return _indexer_local(
        q,
        positions,
        cos_sin_cache,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )


def _merge_kv_split_jax(
    gathered: jax.Array, rank: jax.Array, *, k: int, kv_split: int
) -> jax.Array:
    """i32[kv_split * T, 2k] rank-major shares -> the unsplit top-k of token
    block `rank[0]`, i32[T / kv_split, k] (all of i32[T, k] when T is not a
    multiple of kv_split)."""
    shares = gathered.reshape(kv_split, -1, 2 * k)
    num_tokens = shares.shape[1]
    if num_tokens % kv_split == 0:
        rows = num_tokens // kv_split
        shares = jax.lax.dynamic_slice_in_dim(shares, rank[0] * rows, rows, axis=1)
    return merge_kv_split_topk(shares, k)


def _merge_kv_split_all_jax(gathered: jax.Array, *, k: int, kv_split: int) -> jax.Array:
    """i32[kv_split * T, 2k] rank-major shares -> the unsplit top-k of every
    token, i32[T, k] (no second collective)."""
    return merge_kv_split_topk(
        gathered.reshape(kv_split, -1, 2 * k), k, payload_merge=True
    )


def _merge_kv_split_a2a_jax(
    exchanged: jax.Array, *, k: int, kv_split: int
) -> jax.Array:
    """i32[T, 2k] after the all-to-all (row block r = rank r's share of this
    rank's token block) -> the unsplit top-k of this rank's block,
    i32[T / kv_split, k]."""
    return merge_kv_split_topk(exchanged.reshape(kv_split, -1, 2 * k), k)


def tp_rank_as_data(group, device: torch.device) -> torch.Tensor:
    """This rank's index in `group` as an i32[1] device value, so that every
    rank compiles the same program."""
    ids = torch.arange(group.world_size, dtype=torch.int32, device=device)
    got = funcol.all_to_all_single(ids, None, None, group.device_group)
    return funcol.wait_tensor(got)[:1]


def merge_kv_split_shares(
    share, merge_op, rank, kv_split: int, a2a_merge_op=None, group=None
) -> torch.Tensor:
    """Every TP rank's i32[T, 2k] share -> the unsplit i32[T, k] top-k.

    With T a multiple of the group size, each rank merges only its own block
    of T / TP tokens and the merged blocks are all-gathered: 1/TP of the
    merge work. With `a2a_merge_op`, each rank receives just the shares of its
    own block (an all-to-all over `group`) instead of every rank's whole share.
    """
    if a2a_merge_op is not None and share.shape[0] % kv_split == 0:
        exchanged = funcol.wait_tensor(
            funcol.all_to_all_single(share, None, None, group.device_group)
        )
        return tensor_model_parallel_all_gather(a2a_merge_op(exchanged), 0)
    merged = merge_op(tensor_model_parallel_all_gather(share, 0), rank)
    if share.shape[0] % kv_split:
        return merged
    return tensor_model_parallel_all_gather(merged, 0)


class VllmDeepseekV4IndexerCache(DeepseekV4IndexerCache):
    """Indexer K cache as a packed uint8 record on TPU."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = self.cache_config.block_size
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=align_to(128 + 1, 128),
            dtype=torch.uint8,
            tokens_per_state=min(self.compress_ratio, block_size),
            alignment=None,
        )


class VllmDeepseekV4Indexer(DeepseekV4Indexer):
    """TPU indexer: ranks compressed KV rows via the streamindex top-k kernel."""

    def __init__(self, *args, **kwargs) -> None:
        orig_compressor = dsv4_attention.DeepseekCompressor
        orig_cache = dsv4_attention.DeepseekV4IndexerCache
        dsv4_attention.DeepseekCompressor = VllmDeepseekCompressor
        dsv4_attention.DeepseekV4IndexerCache = VllmDeepseekV4IndexerCache
        try:
            super().__init__(*args, **kwargs)
        finally:
            dsv4_attention.DeepseekCompressor = orig_compressor
            dsv4_attention.DeepseekV4IndexerCache = orig_cache
        object.__setattr__(self, "indexer_op", self._build_indexer_op())
        # Every TP rank holds the whole indexer K cache and computes the same
        # top-k. From TPU_DSV4_INDEXER_KV_SPLIT_MIN_TOKENS tokens on, each rank
        # scores 1/TP of every sequence's positions and the ranks merge their
        # top-k candidates (same indices, same order).
        tp = get_tensor_model_parallel_world_size()
        split_min = envs.TPU_DSV4_INDEXER_KV_SPLIT_MIN_TOKENS
        decode_range = envs.TPU_DSV4_INDEXER_KV_SPLIT_DECODE_TOKENS
        if decode_range and len(decode_range) != 2:
            raise ValueError(
                "TPU_DSV4_INDEXER_KV_SPLIT_DECODE_TOKENS must be 'lo:hi', got "
                f"{decode_range}"
            )
        split_on = split_min > 0 or bool(decode_range)
        object.__setattr__(self, "kv_split", tp if (split_on and tp > 1) else 1)
        object.__setattr__(
            self, "kv_split_min_tokens", split_min if split_min > 0 else 1 << 30
        )
        object.__setattr__(
            self, "kv_split_decode_range", tuple(decode_range) or (1 << 30, 0)
        )
        if self.kv_split > 1:
            object.__setattr__(
                self, "indexer_split_ops", self._build_indexer_op(self.kv_split)
            )
        object.__setattr__(self.compressor, "k_cache", self.k_cache)

    def _build_indexer_op(self, kv_split: int = 1):
        """The indexer op; with `kv_split` > 1, the (share, merge) op pair."""
        mesh = get_pallas_op_mesh()
        # Speculative verify windows all hold K + 1 tokens; the runner puts
        # them in `[distribution[0], distribution[1])`.
        spec = get_current_vllm_config().speculative_config
        uniform_q_len = None if spec is None else spec.num_speculative_tokens + 1
        kv_layout = indexer_kv_layout_name()

        wrapped_fn = functools.partial(
            _indexer_jax if kv_split == 1 else _indexer_split_jax,
            k=self.topk_tokens,
            compression_ratio=self.compress_ratio,
            softmax_scale=self.softmax_scale,
            n_head=self.n_head,
            uniform_q_len=uniform_q_len,
            kv_layout=kv_layout,
            **({} if kv_split == 1 else {"kv_split": kv_split}),
        )

        _scale = (
            f"{self.softmax_scale:.6g}".replace(".", "p")
            .replace("-", "m")
            .replace("+", "")
        )
        op_name = (
            f"pallas::deepseek_v4_indexer_k{self.topk_tokens}"
            f"_c{self.compress_ratio}_h{self.n_head}_s{_scale}"
            f"{'' if uniform_q_len is None else f'_u{uniform_q_len}'}"
            f"_{kv_layout}"
            f"{'' if kv_split == 1 else f'_split{kv_split}'}"
        )
        global _indexer_op_cache
        if op_name in _indexer_op_cache:
            return _indexer_op_cache[op_name]

        attn_data_axis = None
        attn_head_axis = "model"
        batch_axis = BATCH_AXIS
        indexer_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=mesh,
            input_partition_specs=(
                P(attn_data_axis, attn_head_axis, None),  # q
                P(attn_data_axis),  # positions
                P(),  # cos_sin_cache
                P(attn_data_axis, attn_head_axis),  # indexer_weights
                P(),  # cache_kv
                P(batch_axis),  # seq_lens
                P(batch_axis),  # page_indices
                P(),  # cu_q_lens
                P(),  # distribution
                *((P(),) if kv_split > 1 else ()),  # split rank
            ),
        )
        out_width = self.topk_tokens * (1 if kv_split == 1 else 2)

        def _fake_indexer(q, *args, **kwargs):
            return torch.empty(
                (q.shape[0], out_width), dtype=torch.int32, device=q.device
            )

        indexer_jax_op.register_fake(_fake_indexer)
        if kv_split > 1:
            merge_op = pallas.jax_op(
                f"pallas::deepseek_v4_indexer_merge_k{self.topk_tokens}_s{kv_split}",
                functools.partial(
                    _merge_kv_split_jax, k=self.topk_tokens, kv_split=kv_split
                ),
                mesh=mesh,
                input_partition_specs=(P(), P()),
            )

            def _fake_merge(gathered, rank):
                num_tokens = gathered.shape[0] // kv_split
                if num_tokens % kv_split == 0:
                    num_tokens //= kv_split
                return torch.empty(
                    (num_tokens, self.topk_tokens),
                    dtype=torch.int32,
                    device=gathered.device,
                )

            merge_op.register_fake(_fake_merge)
            a2a_merge_op = pallas.jax_op(
                f"pallas::deepseek_v4_indexer_merge_a2a_k{self.topk_tokens}_s{kv_split}",
                functools.partial(
                    _merge_kv_split_a2a_jax, k=self.topk_tokens, kv_split=kv_split
                ),
                mesh=mesh,
                input_partition_specs=(P(),),
            )

            def _fake_merge_a2a(exchanged):
                return torch.empty(
                    (exchanged.shape[0] // kv_split, self.topk_tokens),
                    dtype=torch.int32,
                    device=exchanged.device,
                )

            a2a_merge_op.register_fake(_fake_merge_a2a)
            all_merge_op = pallas.jax_op(
                f"pallas::deepseek_v4_indexer_merge_all_k{self.topk_tokens}_s{kv_split}",
                functools.partial(
                    _merge_kv_split_all_jax, k=self.topk_tokens, kv_split=kv_split
                ),
                mesh=mesh,
                input_partition_specs=(P(),),
            )

            def _fake_merge_all(gathered):
                return torch.empty(
                    (gathered.shape[0] // kv_split, self.topk_tokens),
                    dtype=torch.int32,
                    device=gathered.device,
                )

            all_merge_op.register_fake(_fake_merge_all)
            indexer_jax_op = (indexer_jax_op, merge_op, a2a_merge_op, all_merge_op)

        _indexer_op_cache[op_name] = indexer_jax_op
        return indexer_jax_op

    def forward(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(query)
        q = q.view(-1, self.n_head, self.head_dim)

        self.compressor(hidden_states, positions, rotary_emb)

        # No KV cache is bound during the profiling forward; the values are
        # unused, so return correctly-shaped dummies. vLLM binds a numel==0
        # placeholder rather than None, so test emptiness.
        kv_cache = getattr(self.k_cache, "kv_cache", None)
        if kv_cache is None or kv_cache.numel() == 0:
            return torch.zeros(
                (q.shape[0], self.topk_tokens), dtype=torch.int32, device=q.device
            )

        # The TPU runner stores its own AttentionMetadata, not the vLLM base class.
        attn_ctx = cast(
            "dict[str, AttentionMetadata] | AttentionMetadata",
            get_forward_context().attn_metadata,
        )
        if isinstance(attn_ctx, dict):
            prefix_key = getattr(getattr(self, "k_cache", None), "prefix", None)
            if prefix_key not in attn_ctx:
                raise KeyError(
                    f"DeepSeek-V4 indexer prefix {prefix_key!r} has no "
                    f"attention metadata. Known: {sorted(attn_ctx)}"
                )
            attn_metadata = attn_ctx[prefix_key]
        else:
            attn_metadata = attn_ctx

        idx_seq_lens = attn_metadata.seq_lens
        idx_block_tables = attn_metadata.block_tables
        idx_q_start_loc = attn_metadata.query_start_loc
        idx_req_dist = attn_metadata.request_distribution

        args = (
            q,
            positions,
            rotary_emb.cos_sin_cache,
            indexer_weights,
            kv_cache,
            idx_seq_lens,
            idx_block_tables,
            idx_q_start_loc,
            idx_req_dist,
        )
        lo, hi = self.kv_split_decode_range
        decode_split = lo <= q.shape[0] <= hi
        if self.kv_split == 1 or (
            q.shape[0] < self.kv_split_min_tokens and not decode_split
        ):
            return self.indexer_op(*args)
        share_op, merge_op, a2a_merge_op, all_merge_op = self.indexer_split_ops
        rank = tp_rank_as_data(get_tp_group(), q.device)
        if decode_split:
            return all_merge_op(
                tensor_model_parallel_all_gather(share_op(*args, rank), 0)
            )
        return merge_kv_split_shares(
            share_op(*args, rank),
            merge_op,
            rank,
            self.kv_split,
            a2a_merge_op if envs.TPU_DSV4_INDEXER_KV_SPLIT_A2A else None,
            get_tp_group(),
        )
