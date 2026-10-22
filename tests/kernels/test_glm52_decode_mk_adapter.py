# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the glm52_decode_mk serving adapter's host-side helpers."""

import pytest

from vllm_torchtpu.kernels.glm52_decode_mk import config as C
from vllm_torchtpu.kernels.glm52_decode_mk.vllm_adapter import (
    Unsupported,
    index_pattern,
)


def _pat(fr, sk, L):
    cfg = C.MKConfig(
        R=16, L=L, first_dense=1, H=6144, NH=64, QL=2048, idx_freq=fr, idx_skip=sk
    )
    return [cfg.full_indexer(l) for l in range(L)]


def test_served_pattern_keeps_hf_values():
    served = _pat(4, 3, 78)
    assert index_pattern(served, 4, 3) == (4, 3)


def test_twin_fFs_maps_to_skip2():
    # the mirror twin's index_topk_pattern [F, F, S] (hf freq 4 / skip 3 would say F, F, F)
    fr, sk = index_pattern([True, True, False], 4, 3)
    assert _pat(fr, sk, 3) == [True, True, False]
    assert (fr, sk) == (4, 2)


def test_ten_layer_twin():
    pat = [True, True, True, False, False, False, True, False, False, False]
    assert index_pattern(pat, 4, 3) == (4, 3)


def test_unrepresentable_pattern_declines():
    with pytest.raises(Unsupported):
        index_pattern([False, True, False, True, True], 4, 3)
