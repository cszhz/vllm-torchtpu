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

import jax
import jax.numpy as jnp
import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import (
    AttentionBackend,  # isort: skip
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekV4SWACache,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_masked_dense
from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla import (
    mla_ragged_paged_attention,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa import (
    mla_sliding_window_ragged_paged_attention,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
    sparse_ragged_paged_attention,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention.verify_window.attn_swa_hca import (
    verify_swa_hca,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import align_to

logger = init_logger(__name__)

BATCH_AXIS = None

# Masked-dense CSA tiling, measured on tpu7x at T2048 with 16 heads per core.
MASKED_DENSE_KV_PAGES_PER_BLOCK = 4
MASKED_DENSE_QUERIES_PER_BLOCK = 64


def _run_swa(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    attention_sinks: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    non_causal_block: bool = False,
    verify_q_len: int | None = None,
    padded_lm_out: bool = False,
    verify_fast_path: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sliding-window pass: updates the SWA cache and returns its partials.

    When a compressed-KV pass follows, the output is left unnormalized so
    that pass can fold `swa_l` / `swa_m` into its own softmax accumulation;
    `padded_lm_out` hands those over lane-padded (see the kernel).
    """
    if sw_cache.shape[0] == 0:
        # Profiling-shape trace: no cache to attend over.
        return (
            jnp.zeros_like(q),
            sw_cache,
            jnp.zeros((q.shape[0], q.shape[1]), dtype=jnp.float32),
            jnp.full((q.shape[0], q.shape[1]), -1e9, dtype=jnp.float32),
        )
    return mla_sliding_window_ragged_paged_attention(
        q=q,
        new_kv=new_kv,
        cache_kv=sw_cache,
        kv_lens=swa_kv_lens,
        page_indices=swa_page_indices,
        cu_q_lens=swa_cu_q_lens,
        distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        # The following parameters are tuned based on microbenchmark results.
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 32, 32),
        q_compute_block_size=4,
        unnormalized_output=not swa_only,
        non_causal_block=non_causal_block,
        chunk_prefill_size=verify_q_len,
        padded_lm_out=padded_lm_out and not swa_only,
        verify_fast_path=verify_fast_path,
    )


# Module-level so `pallas.jax_op` can trace and register it as a torch op.
def _attention_hca(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,  # pass kv_lens_to_attend for HCA
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    two_caches_same_buffer: bool,
    non_causal_block: bool = False,
    verify_q_len: int | None = None,
    hca_mixed_q_block: int = 32,
    hca_pipelined: bool = False,
    swa_padded_lm: bool = False,
    verify_fast_path: bool = False,
    verify_fused_hca: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """SWA-only layers, and HCA layers over a single raw-bf16 latent array.

    `verify_q_len`: requests [distribution[0], distribution[1]) are speculative
    verify windows of exactly this many tokens; they run the kernels' static
    query-length pass instead of the mixed one.

    `hca_mixed_q_block`: query tokens per block of the HCA kernel's mixed
    (prefill) pass. Rows are independent and the KV blocking is unchanged, so
    any value gives bit-identical results; larger blocks re-stream the KV less.

    `hca_pipelined`: the HCA kernel copies the next kv block while computing on
    the current one, with static-size page copies (same arithmetic).
    `swa_padded_lm`: hand the SWA l/m to the next kernel lane-padded, without
    the zero-fill and slice/pad copies (same arithmetic on real rows).
    `verify_fast_path`: a batch of verify windows only skips the empty decode
    and mixed passes of both kernels (same arithmetic).
    `verify_fused_hca` (HCA layers, with `verify_fast_path`): such a batch runs
    SWA and HCA as one kernel and one online softmax (`verify_swa_hca`; the
    softmax blocking and the bf16 P of its PV matmul change the rounding).
    """
    if main_cache_kv.shape[0] == 0 or (swa_only and sw_cache.shape[0] == 0):
        # Profiling-shape trace: the kernel is skipped. Unused inputs stay in
        # the compiled signature because torch_tpu jits with keep_unused=True.
        return jnp.zeros(q.shape, dtype=q.dtype), sw_cache

    swa_page_indices = swa_page_indices.flatten()
    main_page_indices = main_page_indices.flatten()

    def _attention_hca_passes(sw_cache, verify_fast_path):
        return _hca_passes(
            q, new_kv, sw_cache, swa_kv_lens, swa_page_indices, swa_cu_q_lens,
            swa_distribution, main_cache_kv, main_kv_lens, extra, main_page_indices,
            main_cu_q_lens, main_distribution, attention_sinks,
            sm_scale=sm_scale, sliding_window=sliding_window,
            logical_page_size=logical_page_size, swa_only=swa_only,
            two_caches_same_buffer=two_caches_same_buffer,
            non_causal_block=non_causal_block, verify_q_len=verify_q_len,
            hca_mixed_q_block=hca_mixed_q_block, hca_pipelined=hca_pipelined,
            swa_padded_lm=swa_padded_lm, verify_fast_path=verify_fast_path,
        )  # fmt: skip

    fused = (
        verify_fused_hca
        and verify_fast_path
        and verify_q_len is not None
        and not swa_only
        and not non_causal_block
        and not two_caches_same_buffer
        and logical_page_size == sw_cache.shape[1] // 2
        and q.shape[0] >= verify_q_len
    )
    if fused:
        only_verify = (main_distribution[0] == 0) & (
            main_distribution[1] == main_distribution[2]
        )

        def fused_branch(sw_cache):
            return verify_swa_hca(
                q,
                new_kv,
                sw_cache,
                swa_kv_lens,
                swa_page_indices,
                main_cache_kv,
                main_page_indices,
                main_distribution[1],
                attention_sinks,
                q_len=verify_q_len,
                sm_scale=sm_scale,
                window=sliding_window,
            )

        def general_branch(sw_cache):
            return _attention_hca_passes(sw_cache, verify_fast_path=False)

        return jax.lax.cond(only_verify, fused_branch, general_branch, sw_cache)

    return _attention_hca_passes(sw_cache, verify_fast_path)


def _hca_passes(
    q, new_kv, sw_cache, swa_kv_lens, swa_page_indices, swa_cu_q_lens,
    swa_distribution, main_cache_kv, main_kv_lens, extra, main_page_indices,
    main_cu_q_lens, main_distribution, attention_sinks, *, sm_scale,
    sliding_window, logical_page_size, swa_only, two_caches_same_buffer,
    non_causal_block, verify_q_len, hca_mixed_q_block, hca_pipelined,
    swa_padded_lm, verify_fast_path,
):  # fmt: skip
    """`_attention_hca`'s SWA kernel then compressed-KV kernel."""
    swa_output, updated_sw_cache, swa_l, swa_m = _run_swa(
        q=q,
        new_kv=new_kv,
        sw_cache=sw_cache,
        swa_kv_lens=swa_kv_lens,
        swa_page_indices=swa_page_indices,
        swa_cu_q_lens=swa_cu_q_lens,
        swa_distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        swa_only=swa_only,
        non_causal_block=non_causal_block,
        verify_q_len=verify_q_len,
        padded_lm_out=swa_padded_lm,
        verify_fast_path=verify_fast_path,
    )

    if swa_only:
        return swa_output, updated_sw_cache

    # When both caches are one buffer, the compressed-KV read must see the
    # SWA kernel's write from this pass, not the pre-write snapshot.
    if two_caches_same_buffer:
        main_cache_kv = updated_sw_cache

    output = mla_ragged_paged_attention(
        q=q,
        cache_kv=main_cache_kv,
        kv_lens=main_kv_lens,
        kv_lens_to_attend=extra,
        page_indices=main_page_indices,
        cu_q_lens=main_cu_q_lens,
        distribution=main_distribution,
        attention_sinks=attention_sinks,
        swa_accumution=swa_output,
        swa_l=swa_l,
        swa_m=swa_m,
        sm_scale=sm_scale,
        # The following parameters are tuned based on microbenchmark results.
        num_kv_pages_per_block=(32, 64, 64),
        # Small buckets keep 32: their mixed pass holds few query rows.
        num_queries_per_block=(
            1,
            32,
            hca_mixed_q_block if q.shape[0] >= 512 else 32,
        ),
        chunk_prefill_size=verify_q_len,
        bkv_prefetch="after_wait" if hca_pipelined else "after_compute",
        fast_dma=hca_pipelined,
        verify_fast_path=verify_fast_path,
    )
    return output, updated_sw_cache


def _attention_csa(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,  # pass topk_indices for CSA
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    main_cache_rope: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    two_caches_same_buffer: bool,
    non_causal_block: bool = False,
    verify_q_len: int | None = None,
    csa_gather_no_pad: bool = False,
    masked_dense_max_kv: int = 0,
    masked_dense_min_tokens: int = 512,
    swa_padded_lm: bool = False,
    verify_fast_path: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """CSA: sparse gather over top-k rows, NoPE and RoPE in two arrays.

    `verify_q_len` as in `_attention_hca`; the sparse gather is per token and
    reads only distribution[2].

    `masked_dense_max_kv` > 0 (buckets of at least `masked_dense_min_tokens`):
    the trailing sequences whose compressed KV length is at most that take
    `csa_masked_dense`, the rest the sparse gather.
    """
    if main_cache_kv.shape[0] == 0 or main_cache_rope.shape[0] == 0:
        return jnp.zeros(q.shape, dtype=q.dtype), sw_cache

    swa_page_indices = swa_page_indices.flatten()
    main_page_indices = main_page_indices.flatten()

    swa_output, updated_sw_cache, swa_l, swa_m = _run_swa(
        q=q,
        new_kv=new_kv,
        sw_cache=sw_cache,
        swa_kv_lens=swa_kv_lens,
        swa_page_indices=swa_page_indices,
        swa_cu_q_lens=swa_cu_q_lens,
        swa_distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        swa_only=False,
        non_causal_block=non_causal_block,
        verify_q_len=verify_q_len,
        padded_lm_out=swa_padded_lm,
        verify_fast_path=verify_fast_path,
    )

    if two_caches_same_buffer:
        main_cache_kv = updated_sw_cache

    # First use of the top-k after the SWA pass: the indexer's SparseCore
    # top-k then completes behind the q projection and SWA, which do not need
    # it, instead of stalling the TensorCore before them.
    extra, swa_output = jax.lax.optimization_barrier((extra, swa_output))

    if masked_dense_max_kv > 0 and q.shape[0] >= masked_dense_min_tokens:
        output = csa_masked_dense.split_attention(
            q,
            main_cache_kv,
            main_cache_rope,
            extra,
            main_kv_lens,
            main_page_indices,
            main_cu_q_lens,
            main_distribution,
            attention_sinks,
            swa_output,
            swa_l,
            swa_m,
            sm_scale=sm_scale,
            max_kv_len=masked_dense_max_kv,
            num_kv_pages_per_block=MASKED_DENSE_KV_PAGES_PER_BLOCK,
            num_queries_per_block=MASKED_DENSE_QUERIES_PER_BLOCK,
            gather_no_pad=csa_gather_no_pad,
        )
        return output, updated_sw_cache

    output = sparse_ragged_paged_attention(
        q=q,
        cache_kv_nope=main_cache_kv,
        cache_kv_rope=main_cache_rope,
        topk_indices=extra,
        page_indices=main_page_indices,
        cu_q_lens=main_cu_q_lens,
        distribution=main_distribution,
        attention_sinks=attention_sinks,
        swa_accumution=swa_output,
        swa_l=swa_l,
        swa_m=swa_m,
        sm_scale=sm_scale,
        # Measured on tpu7x: 32-token gather chunks up to 128 tokens per step,
        # 64 above (each chunk's SparseCore gather overlaps the previous
        # chunk's attention); batches of 8 win throughout.
        gather_and_attention_chunk_size=32 if q.shape[0] <= 128 else 64,
        attention_kernel_batch_size=8,
        gather_no_pad=csa_gather_no_pad,
    )
    return output, updated_sw_cache


def get_packed_mla_head_size(hf_config) -> int:
    """Calculate TPU 128-byte aligned packed width for DSv4 FP8 cache entries."""
    kv_bytes = getattr(hf_config, "kv_lora_rank", 448)
    rope_bytes = getattr(hf_config, "qk_rope_head_dim", 64) * 2
    scale_bytes = kv_bytes // 64
    return align_to(kv_bytes + rope_bytes + scale_bytes, 128)


class VllmDeepseekSparseSWABackend(DeepseekSparseSWABackend):
    """TPU attention backend for DeepSeek-V4 SWA cache."""

    @classmethod
    def get_min_page_size(cls, vllm_config: VllmConfig) -> int:
        return 1


class VllmDeepseekV4SWACache(DeepseekV4SWACache):
    """Sliding-window KV cache, paged to fit inside a CSA main page."""

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        backend_cls: type[AttentionBackend] | None = None,
    ) -> None:
        super().__init__(
            head_dim,
            window_size,
            dtype,
            prefix,
            cache_config,
            backend_cls=backend_cls or VllmDeepseekSparseSWABackend,
        )
        # Initialize with default block size at construction time;
        # get_kv_cache_spec recomputes self.block_size from finalized runtime config.
        self.block_size = self._swa_block_size(cache_config.block_size, window_size)

    @staticmethod
    def _swa_block_size(compressed_kv_cache_bz: int, window_size: int | None) -> int:
        # We would like to overlay the SWA cache with CSA's main NOPE cache
        # on the same KV-Tensor, whose shape is [num_pages, page_size, 4, 128]
        # u8.
        # Thus set swa cache's block size accordingly.
        # In SWA, we store kv cache in bf16 to avoid expensive quantization
        # and dequantization. The extra storage overhead is small since kvs
        # outside the sliding window can be reclaimed as needed.
        # 2 because bf16 occupies 2 bytes per element.
        csa_compression_ratio = 4
        block_size = compressed_kv_cache_bz // csa_compression_ratio // 2
        if window_size is None:
            return block_size
        return min(block_size, window_size)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Update self.block_size so the returned spec and the Pallas kernel's
        # logical_page_size in attention.py agree on the final runtime block size.
        self.block_size = self._swa_block_size(
            vllm_config.cache_config.block_size, self.window_size
        )
        # `mla_swa` keeps the SWA entries as raw bf16, packed as uint8.
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=1024,
            dtype=torch.uint8,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=None,
        )
