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
"""SparseCore capability checks used to pick SparseCore vs TensorCore paths."""

import types

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.sparse_core import ragged_gather_v2 as ragged_gather_v2_mod
from vllm_torchtpu.kernels.sparse_core import support


def _sc_info(dma_granule_size_bytes):
    return types.SimpleNamespace(
        num_cores=1,
        num_subcores=4,
        num_lanes=16,
        dma_granule_size_bytes=dma_granule_size_bytes,
    )


@pytest.mark.parametrize(
    ("sc_info", "expected"),
    [
        (None, False),
        (_sc_info(32), True),
        (_sc_info(64), False),
        # Older TpuInfo objects without the field keep the SparseCore path.
        (types.SimpleNamespace(num_cores=2, num_subcores=16, num_lanes=16), True),
    ],
)
def test_is_supported_sparse_core(sc_info, expected):
    assert support.is_supported_sparse_core(sc_info) is expected


@pytest.mark.parametrize(("granule", "expected"), [(32, True), (64, False)])
def test_has_usable_sparse_core(monkeypatch, granule, expected):
    monkeypatch.setattr(
        support.pltpu,
        "get_tpu_info",
        lambda: types.SimpleNamespace(sparse_core=_sc_info(granule)),
    )
    assert support.has_usable_sparse_core() is expected


def test_has_usable_sparse_core_on_non_tpu_host(monkeypatch):
    def _raise():
        raise ValueError("Unsupported TPU device kind: cpu")

    monkeypatch.setattr(support.pltpu, "get_tpu_info", _raise)
    assert support.has_usable_sparse_core() is False


def test_ragged_gather_v2_falls_back_for_64_byte_dma_granule(monkeypatch):
    monkeypatch.setattr(
        ragged_gather_v2_mod.pltpu,
        "get_tpu_info",
        lambda: types.SimpleNamespace(sparse_core=_sc_info(64)),
    )
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    indices = jnp.array([3, 1, 7, 0], dtype=jnp.int32)

    out = ragged_gather_v2_mod.ragged_gather_v2(
        x, indices, jnp.array([0], jnp.int32), jnp.array([4], jnp.int32)
    )

    np.testing.assert_array_equal(np.asarray(out), np.asarray(x)[[3, 1, 7, 0]])
