# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Indexer 局部更新：保持现有 uint8[pages,page/4,4,width] 布局。

同一个 request 的新 token 连续排列；调度器保证写入的物理位置唯一。
一个 packed group 的所有新值在每次写该 group 时一起合并，重复访问幂等。
所有 cache 形状适配都发生在 Pallas ref 内，避免 XLA 重排整个池。
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _write(indices, values, cache, output, scratch, value_scratch, sem):
    step = pl.program_id(0)
    page, group = indices[0, step], indices[1, step]
    previous = jnp.maximum(step - 1, 0)
    first_in_group = ((step == 0) | (page != indices[0, previous]) |
                      (group != indices[1, previous]))

    @pl.when((page >= 0) & (page < cache.shape[0]) & first_in_group)
    def update():
        target = output.at[pl.ds(page, 1), pl.ds(group, 1), :, :]
        read = pltpu.make_async_copy(target, scratch, sem.at[0])
        read_values = pltpu.make_async_copy(values.at[pl.ds(step, 1), :, :],
                                            value_scratch, sem.at[1])
        read.start()
        read_values.start()
        read.wait()
        read_values.wait()
        rows = jax.lax.broadcasted_iota(jnp.int32, scratch.shape, 2)
        mask = jnp.zeros(scratch.shape, dtype=jnp.bool_)
        for row in range(scratch.shape[2]):
            mask |= (rows == row) & (indices[2 + row, step] != 0)
        scratch[...] = jnp.where(mask, value_scratch[...], scratch[...])
        write = pltpu.make_async_copy(scratch, target, sem.at[0])
        write.start()
        write.wait()


def update_indexer_cache(cache, values, page, slot):
    """更新非负物理 page/slot；正向越界 page 表示 padding。

    values 已由调用方填充到 cache 的字节宽度。
    支持 HEAD_ALONG_SUBLANE indexer 的四字节 packing。
    """
    if cache.ndim != 4 or cache.shape[2] != 4 or cache.dtype != jnp.uint8:
        raise ValueError(
            f"indexer cache 必须为 uint8[pages,slots,4,width]，实际 {cache.shape} {cache.dtype}"
        )
    if (page.ndim != 1 or values.dtype != jnp.uint8
            or values.shape != (page.shape[0], cache.shape[-1])
            or slot.shape != page.shape):
        raise ValueError("indexer values/page/slot 的 token 维度与字节宽度不匹配")
    tokens, width = values.shape
    if tokens == 0 or width % 128:
        raise ValueError("indexer 更新要求非空 token 桶以及 128 字节对齐的宽度")
    # TPU7x/JAX 0.10.2 的实测策略，不是硬件约束：1024-token scatter
    # 会重排全池，>=128 MiB 时局部更新有稳定收益。小桶不重排全池，
    # 原 scatter 更快；其余尚未测量的 geometry 保持原路径。
    if (tokens != 1024 or cache.shape[1:] != (256, 4, 256)
            or cache.shape[0] < 512):
        return cache.at[page, slot // 4, slot % 4].set(values)
    return _partial_update_indexer_cache(cache, values, page, slot)


def _partial_update_indexer_cache(cache, values, page, slot):
    tokens, width = values.shape
    packing = cache.shape[2]
    token = jnp.arange(tokens, dtype=jnp.int32)
    source = token[:, None] - (slot %
                               packing)[:, None] + jnp.arange(packing)[None, :]
    safe_source = jnp.clip(source, 0, tokens - 1)
    group = slot // packing
    same_group = ((source >= 0) & (source < tokens) &
                  (page[safe_source] == page[:, None]) &
                  (group[safe_source] == group[:, None]))
    grouped_values = values[safe_source]
    indices = jnp.concatenate((jnp.stack(
        (page, group)), same_group.T.astype(jnp.int32)))
    return pl.pallas_call(
        _write,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(tokens, ),
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),
                      pl.BlockSpec(memory_space=pltpu.HBM)),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(pltpu.VMEM((1, 1, packing, width), jnp.uint8),
                            pltpu.VMEM((1, packing, width), jnp.uint8),
                            pltpu.SemaphoreType.DMA((2, ))),
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={2: 0},
        name="indexer_cache_partial_update",
    )(indices, grouped_values, cache)
