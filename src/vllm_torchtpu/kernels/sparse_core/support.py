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
"""Whether the SparseCore kernels in this package can target the current TPU."""

from typing import Any

from jax.experimental.pallas import tpu as pltpu

# The SparseCore kernels in this package are written for 32-byte DMA granules
# (TPU v5p, v6e and TPU7x). TPU8i reports a SparseCore with 64-byte granules,
# so callers must take their TensorCore fallbacks there.
SUPPORTED_DMA_GRANULE_BYTES = 32


def is_supported_sparse_core(sc_info: Any) -> bool:
    """True if ``sc_info`` (``TpuInfo.sparse_core``) is usable by these kernels."""
    if sc_info is None:
        return False
    granule = getattr(sc_info, "dma_granule_size_bytes", SUPPORTED_DMA_GRANULE_BYTES)
    return granule <= SUPPORTED_DMA_GRANULE_BYTES


def has_usable_sparse_core() -> bool:
    """True if the current TPU has a SparseCore these kernels can target."""
    try:
        sc_info = pltpu.get_tpu_info().sparse_core
    except ValueError:
        # get_tpu_info raises for unsupported device kinds (e.g. CPU hosts).
        return False
    return is_supported_sparse_core(sc_info)
