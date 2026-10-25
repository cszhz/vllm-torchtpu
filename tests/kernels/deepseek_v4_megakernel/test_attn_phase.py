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
"""owner_attention_phase (interpret) vs the block references.

Attention: tolerance tier (max |err| / max |ref| <= 1e-2, mean <= 1e-3).
Caches: bit-exact (SWA rows, compressor / indexer stores)."""

import os
import sys

import numpy as np
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_phase_harness as H  # noqa: E402

KINDS = ("hca", "csa", "hca", "csa")


def _check(ls, kind, layer, inp, caches, out, n_active=None, heads=128, h0=0):
    """heads < 128: the kernel attended heads [h0, h0 + heads) (group TP)."""
    R = len(ls.seq_lens)
    n_active = R if n_active is None else n_active
    exp_c, exp_o, _ = H.reference(ls, kind, layer, inp)
    o = np.asarray(out, np.float32)[:n_active, :7 * heads].reshape(n_active, 7, heads, 512)
    e = np.asarray(exp_o)[:n_active, :, h0:h0 + heads]
    err = np.abs(o - e)
    assert err.max() / np.abs(e).max() <= 1e-2 and err.mean() <= 1e-3, err.max()
    for c, x in zip(caches, exp_c):
        assert np.array_equal(np.asarray(c), x)


@pytest.mark.parametrize("kind,layer,seq,max_ctx", [
    ("hca", 0, [1030, 3000], 4096),
    ("hca", 2, [255 + 7, 129], 4096),  # SWA page crossing, short window
    ("csa", 1, [1030, 3000], 4096),  # every visible row selected
    ("csa", 3, [9000, 300], 9216),  # real top-1024, two page blocks
])
def test_layer(kind, layer, seq, max_ctx):
    rng = np.random.default_rng(layer + len(seq))
    ls = H.build(rng, KINDS, seq, max_ctx)
    inp = H.make_inputs(rng, ls, kind)
    caches, out = H.run_phase(ls, kind, layer, inp)
    _check(ls, kind, layer, inp, caches, out)


@pytest.mark.parametrize("kind", ["hca", "csa"])
def test_idle_and_partial(kind):
    """num_req = 0 touches nothing; num_req = 1 of 2 slots leaves slot 1 alone."""
    rng = np.random.default_rng(5)
    ls = H.build(rng, KINDS, [1030, 2000], 4096)
    layer = 0 if kind == "hca" else 1
    inp = H.make_inputs(rng, ls, kind)
    caches, _ = H.run_phase(ls, kind, layer, inp, num_req=0)
    for c, a in zip(caches, ls.arrays):
        assert np.array_equal(np.asarray(c), np.asarray(a))
    caches, out = H.run_phase(ls, kind, layer, inp, num_req=1)
    one = H.dataclasses.replace(ls, seq_lens=ls.seq_lens[:1],
                                tables={k: v[:1] for k, v in ls.tables.items()},
                                swa_table={k: v[:1] for k, v in ls.swa_table.items()})
    inp1 = {k: v[:1] for k, v in inp.items()}
    _check(one, kind, layer, inp1, caches, out, n_active=1)


def test_positive_control():
    """A different attention sink in the reference must fail the check."""
    rng = np.random.default_rng(1)
    ls = H.build(rng, KINDS, [1030, 3000], 4096)
    inp = H.make_inputs(rng, ls, "hca")
    caches, out = H.run_phase(ls, "hca", 0, inp)
    bad = H.dataclasses.replace(ls, weights=dict(ls.weights,
                                                 attn_sink=ls.weights["attn_sink"] + 1.0))
    with pytest.raises(AssertionError):
        _check(bad, "hca", 0, inp, caches, out)


def test_csa_rows_mode():
    """The per-row staging path (CSA_MODE='rows') against the same references."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP

    rng = np.random.default_rng(9)
    ls = H.build(rng, KINDS, [9000, 300], 9216)
    inp = H.make_inputs(rng, ls, "csa")
    AP.CSA_MODE = "rows"
    try:
        caches, out = H.run_phase(ls, "csa", 3, inp)
    finally:
        AP.CSA_MODE = "union"
    _check(ls, "csa", 3, inp, caches, out)


@pytest.mark.parametrize("kind", ["hca", "csa"])
def test_bf16_q_storage(kind):
    """q and out held as bf16 (the step's q_in may be bf16): same references."""
    rng = np.random.default_rng(11)
    ls = H.build(rng, KINDS, [5000], 8192)
    layer = 0 if kind == "hca" else 1
    inp = H.make_inputs(rng, ls, kind)
    inp = dict(inp, q=inp["q"].astype(jnp.bfloat16))
    caches, out = H.run_phase(ls, kind, layer, inp)
    assert out.dtype == jnp.bfloat16
    _check(ls, kind, layer, inp, caches, out)


@pytest.mark.parametrize("kind,layer,h0", [("hca", 0, 16), ("csa", 3, 112), ("csa", 1, 0)])
def test_group_tp_heads(kind, layer, h0):
    """attn_heads=16 (group TP): the 16-head slice of the full-width reference."""
    rng = np.random.default_rng(13 + layer)
    ls = H.build(rng, KINDS, [9000, 300], 9216)
    inp = H.make_inputs(rng, ls, kind)
    caches, out = H.run_phase(ls, kind, layer, inp, attn_heads=16, h0=h0)
    _check(ls, kind, layer, inp, caches, out, heads=16, h0=h0)
    with pytest.raises(AssertionError):  # positive control: the neighbouring slice
        _check(ls, kind, layer, inp, caches, out, heads=16, h0=(h0 + 16) % 128)


def test_union_size_stats():
    """stats[layer, r] = the union size; with every visible row selected it is
    the request's compressed length."""
    rng = np.random.default_rng(17)
    ls = H.build(rng, KINDS, [1030, 3000], 4096)
    inp = H.make_inputs(rng, ls, "csa")
    fn, ops = H.build_call(ls, "csa", 1, inp, stats=True)
    res = fn(*ops)
    st = np.asarray(res[-1])
    assert st[1].tolist() == [1030 // 4, 3000 // 4], st[1]
