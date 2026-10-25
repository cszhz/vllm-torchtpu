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
"""SWA + HCA megakernel attention vs the plain-JAX reference (interpret mode).

Tolerance tier: the kernel feeds bf16 probabilities to the PV matmul and
rounds the output to bf16; the reference is f32 throughout.
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_cases as C  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import (  # noqa: E402
    attn_ref,
    attn_swa_hca,
)

MAX_REL = 1e-2  # max |err| / max |ref|
MEAN_ABS = 1e-3


def _run(case, **kw):
    out = attn_swa_hca.swa_hca_attention(
        case.q, case.new_kv, case.swa_words, case.swa_table, case.hca_words,
        case.hca_table, case.seq_lens, case.sinks, sm_scale=case.sm_scale,
        interpret=True, **kw,
    )
    return np.asarray(out, np.float32)


def _ref(case, **kw):
    return np.asarray(attn_ref.swa_hca_reference(
        case.q, case.new_kv, case.swa_words, case.swa_table, case.hca_words,
        case.hca_table, case.seq_lens, case.sinks, sm_scale=case.sm_scale, **kw,
    ))


def _close(out, ref):
    err = np.abs(out - ref)
    return err.max() / np.abs(ref).max() <= MAX_REL and err.mean() <= MEAN_ABS


# Lengths cover: no compressed row yet (< 128), window shorter than 127,
# a compressed-row boundary inside the window (1030 -> rows 7, 8), several
# HCA blocks (blk_pages=2 -> 16 rows a block) and an exact page boundary.
@pytest.mark.parametrize(
    "heads,seq_lens,blk_pages",
    [
        (8, [60, 300, 1030, 2049], 2),
        (8, [128 * 16 + 3], 32),
        (16, [135, 4100], 4),
        (128, [1030, 250], 2),
    ],
)
def test_matches_reference(heads, seq_lens, blk_pages):
    case = C.swa_hca_case(np.random.default_rng(len(seq_lens) + heads), seq_lens, heads)
    out, ref = _run(case, blk_pages=blk_pages), _ref(case)
    assert _close(out, ref), (np.abs(out - ref).max(), np.abs(ref).max())


def test_bf16_pv_matches_reference():
    case = C.swa_hca_case(np.random.default_rng(7), [1030, 3000], 8)
    out, ref = _run(case, blk_pages=4, pv_dtype=jnp.bfloat16), _ref(case)
    assert _close(out, ref)


def test_swa_only_layer():
    case = C.swa_hca_case(np.random.default_rng(3), [70, 900], 8)
    out = _run(case, with_hca=False)
    ref = _ref(case, ratio=1 << 30)  # no compressed rows visible
    assert _close(out, ref)


def test_positive_controls():
    """The check must reject a window off by one and one corrupted HCA row."""
    case = C.swa_hca_case(np.random.default_rng(11), [1030, 2500], 8)
    out = _run(case, blk_pages=2)
    assert _close(out, _ref(case))
    assert not _close(out, _ref(case, window=127))
    # Overwrite compressed row 3 of request 0 (visible to every query row).
    page = int(case.hca_table[0, 0])
    words = np.asarray(case.hca_words).copy()
    words[page, 6:8] = np.asarray(
        jnp.full((2, 128), 0x41204120, jnp.int32))  # bf16 10.0 everywhere
    bad = C.dataclasses.replace(case, hca_words=jnp.asarray(words))
    assert not _close(out, _ref(bad))


def test_swa_write_stores_new_rows_bitexact():
    """write_swa: the q_len new rows land at their positions (incl. a page
    crossing at 255 -> 256); every other cache word is untouched."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_layout as L

    case = C.swa_hca_case(np.random.default_rng(3), [60, 300, 1030, 262, 129], 8)
    out, cache = attn_swa_hca.swa_hca_attention(
        case.q, case.new_kv, case.swa_words, case.swa_table, case.hca_words,
        case.hca_table, case.seq_lens, case.sinks, sm_scale=case.sm_scale,
        write_swa=True, interpret=True)
    assert _close(np.asarray(out, np.float32), _ref(case))
    exp = np.asarray(case.swa_words).copy().reshape(-1, C.SWA_PAGE, 2, 128)
    words = np.asarray(L.bf16_tokens_to_words(case.new_kv))
    tab = np.asarray(case.swa_table)
    for r, s in enumerate(np.asarray(case.seq_lens)):
        for t in range(C.Q_LEN):
            pos = s - C.Q_LEN + t
            exp[tab[r, pos // C.SWA_PAGE], pos % C.SWA_PAGE] = words[r, t]
    assert np.array_equal(np.asarray(cache).reshape(exp.shape), exp)
