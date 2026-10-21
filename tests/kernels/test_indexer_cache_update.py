# SPDX-License-Identifier: Apache-2.0
"""Indexer packed cache 更新必须逐字节保留未写入的邻居。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.indexer_cache_update import \
    _partial_update_indexer_cache


@pytest.mark.parametrize(
    "pages,slots",
    [
        ([], []),  # 全 padding 时不能修改任意 cache 字节。
        ([2], [3]),  # packed group 的最后一个 token，前缀必须保留。
        ([2, 2, 2, 2], [1, 2, 3, 4]),  # 重复访问同一 packed group。
        ([1, 1, 1, 3, 3, 3, 0, 0], [1021, 1022, 1023, 0, 1, 2, 6, 7]),
    ],
)
def test_preserves_untouched_bytes_across_packed_groups(pages, slots):
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas indexer writer requires TPU")
    rng = np.random.default_rng(53)
    before = rng.integers(0, 256, (4, 256, 4, 256), dtype=np.uint8)
    values = rng.integers(0, 256, (16, 256), dtype=np.uint8)
    expected = before.copy()
    for row, (page, slot) in enumerate(zip(pages, slots)):
        expected[page, slot // 4, slot % 4] = values[row]
    padding = 16 - len(pages)
    page_ids = jnp.array(pages + [2**30] * padding, dtype=jnp.int32)
    slot_ids = jnp.array(slots + [0] * padding, dtype=jnp.int32)
    # 小池直接覆盖 RMW kernel 的邻居保留契约；不受性能分派阈值影响。
    actual = jax.jit(_partial_update_indexer_cache)(jnp.asarray(before),
                                                    jnp.asarray(values),
                                                    page_ids, slot_ids)
    np.testing.assert_array_equal(np.asarray(actual), expected)
