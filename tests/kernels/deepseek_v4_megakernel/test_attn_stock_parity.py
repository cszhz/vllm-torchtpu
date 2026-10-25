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
"""Megakernel attention vs the served stock kernels' arithmetic (CPU interpret).

The stock decode path is `_attention_hca` / `_attention_csa`
(layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py): `mla_swa`,
then `mla` (HCA) or `sparse_mla` (CSA) continuing its softmax state; SWA-only
layers stop after `mla_swa`. Their Pallas kernels do not run in interpret mode
(bitcast and dynamic-size DMA windows), so `attn_ref._attend_stock` transcribes
their arithmetic: ``s = (q @ k) * sm_scale``, ``p = exp(s - m)`` in f32 for the
row sum and P.V, the SWA accumulator handed to the compressed pass as bf16, and
``acc * reciprocal(l + exp(sink - m), approx=True)`` cast to bf16.

1. `VerifyWindow.update` / `finalize` on inputs whose scores and products are
   exact: the softmax state and output must equal the stock op sequence
   bit for bit (only the reduction order of the f32 sums could differ, and
   two visible keys per row leave none).
2. The kernels end to end: bf16 outputs may differ from the transcription only
   where f32 reduction-order noise flips a bf16 rounding (<= 1% of elements);
   the previous arithmetic (bf16 probabilities, no bf16 hand-off) differs on
   more than a quarter of them.
"""

import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_cases as C  # noqa: E402
import attn_phase_harness as PH  # noqa: E402
import attn_swa_harness as SH  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_core as core  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import (  # noqa: E402
    attn_csa,
    attn_ref,
    attn_swa_hca,
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP  # noqa: E402

MAX_MISMATCH = 0.01  # fraction of bf16 outputs allowed to differ
NEG = core.NEG

# ------------------------------------------------------------------ 1. bit level


def _window_kernel(cfg, n_blk, q_ref, kv_ref, vis_ref, sink_ref, out_ref, m_ref, l_ref,
                   acc_ref, *scr):
    scr = core.WindowScratch(*scr)
    win = core.VerifyWindow(cfg, seq_lens=None, swa_pt=None, q_ref=q_ref, new_kv_ref=None,
                            sink_ref=sink_ref, swa_hbm=None, out_ref=out_ref, scr=scr)
    for b in range(n_blk):
        kv = kv_ref[b]
        win.update(0, kv, kv.T if cfg.transposed else None, vis_ref[b] != 0, first=b == 0)
    m_ref[...], l_ref[...], acc_ref[...] = scr.m[...], scr.l[...], scr.acc[...]
    win.finalize(0)


def run_window(cfg, q, kv, vis, sinks):
    """q f32 [rows_pad, 512], kv bf16 [n_blk, n, 512], vis bool [n_blk, rows, n],
    sinks f32 [rows] -> (out, m, l, acc) per row, as [rows, ...]."""
    n_blk = kv.shape[0]
    R = cfg.rows_pad if cfg.transposed else cfg.rows8
    vis_k = jnp.pad(vis, ((0, 0), (0, R - vis.shape[1]), (0, 0))).astype(jnp.int32)
    stat = (1, cfg.rows_pad) if cfg.transposed else (cfg.rows8, 1)
    acc = (512, cfg.rows_pad) if cfg.transposed else (cfg.rows8, 512)
    if cfg.transposed:
        vis_k = vis_k.transpose(0, 2, 1)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    out, m, l, a = pl.pallas_call(  # noqa: E741
        functools.partial(_window_kernel, cfg, n_blk),
        out_shape=[jax.ShapeDtypeStruct((1, cfg.rows_pad, 512), jnp.float32),
                   jax.ShapeDtypeStruct(stat, jnp.float32),
                   jax.ShapeDtypeStruct(stat, jnp.float32),
                   jax.ShapeDtypeStruct(acc, jnp.float32)],
        in_specs=[vmem] * 4, out_specs=[vmem] * 4,
        scratch_shapes=core.window_scratch_shapes(cfg),
        interpret=pltpu.InterpretParams(),
    )(q[None], kv, vis_k, jnp.pad(sinks, (0, cfg.rows_pad - sinks.shape[0]))[None])
    rows = cfg.rows
    flat = lambda x: np.asarray(x).reshape(-1)[:rows]  # noqa: E731
    a = np.asarray(a).T if cfg.transposed else np.asarray(a)
    return np.asarray(out)[0, :rows], flat(m), flat(l), a[:rows]


@functools.partial(jax.jit, static_argnums=(4, 5))
def stock_window(q, kv, vis, sinks, sm_scale, round_window_acc):
    """The stock kernels' op sequence (mla_swa.py flash_attention, mla.py
    flash_attention, sparse_mla.py flash_attention_step1/2), block by block.
    Jitted as the interpreted kernel is: XLA:CPU then fuses the state update
    (``alpha * l + rowsum`` rounds once) on both sides alike."""
    m = l = acc = None  # noqa: E741
    for b in range(kv.shape[0]):
        k = kv[b].astype(jnp.float32)
        s = jnp.einsum("nd,md->nm", q, k, precision="highest") * sm_scale
        s = jnp.where(vis[b], s, NEG)
        s_max = jnp.max(s, axis=1)
        m_cur = s_max if m is None else jnp.maximum(m, s_max)
        p = jnp.exp(s - m_cur[:, None])
        pv = jnp.einsum("nm,md->nd", p, k, precision="highest")
        if m is None:
            l, acc = jnp.sum(p, axis=1), pv  # noqa: E741
            if round_window_acc:
                acc = acc.astype(jnp.bfloat16).astype(jnp.float32)
        else:
            alpha = jnp.exp(m - m_cur)
            l = alpha * l + jnp.sum(p, axis=1)  # noqa: E741
            acc = alpha[:, None] * acc + pv
        m = m_cur
    l_total = l + jnp.exp(sinks - m)
    out = (acc * attn_ref._approx_reciprocal(l_total)[:, None]).astype(jnp.bfloat16)
    return out.astype(jnp.float32), m, l, acc


def exact_case(rng, rows, n_blk, n, vis_per_row=2):
    """Integer queries, keys in {0, +-1/2, +-1, +-2}: every score and every p * v
    product is exact, so only exp, the scale and the softmax update can differ."""
    q = np.zeros((128 * -(-rows // 128), 512), np.float32)
    for i in range(rows):
        cols = rng.choice(512, 24, replace=False)
        q[i, cols] = rng.integers(-4, 5, 24)
    kv = rng.choice([0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0], size=(n_blk, n, 512)).astype(np.float32)
    vis = np.zeros((n_blk, rows, n), bool)
    for b in range(n_blk):
        for i in range(rows):
            vis[b, i, rng.choice(n, vis_per_row, replace=False)] = True
    sinks = rng.standard_normal(rows).astype(np.float32)
    return jnp.asarray(q), jnp.asarray(kv, jnp.bfloat16), jnp.asarray(vis), jnp.asarray(sinks)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("round_window_acc", [False, True])
def test_window_state_bitexact_vs_stock(transposed, round_window_acc):
    cfg = core.WindowConfig(heads=8, sm_scale=512 ** -0.5, q_len=16,
                            force_transposed=transposed, round_window_acc=round_window_acc)
    q, kv, vis, sinks = exact_case(np.random.default_rng(int(transposed)), cfg.rows, 3, 128)
    got = run_window(cfg, q, kv, vis, sinks)
    exp = stock_window(q[: cfg.rows], kv, vis, sinks, cfg.sm_scale, round_window_acc)
    for name, g, e in zip(("out", "m", "l", "acc"), got, map(np.asarray, exp)):
        assert np.array_equal(g, e), (name, np.abs(g - e).max())


def test_bitexact_check_detects_previous_softmax_form():
    """Negative control: the previous exp2((s - m) * scale * log2 e) on unscaled
    scores, with exact 1 / l, is not bit-identical to stock on the same inputs."""
    sm_scale = 512 ** -0.5
    q, kv, vis, sinks = exact_case(np.random.default_rng(0), 128, 1, 128)
    k = kv[0].astype(jnp.float32)
    s_raw = jnp.where(vis[0], jnp.einsum("nd,md->nm", q, k, precision="highest"), NEG)
    m_raw = jnp.max(s_raw, axis=1, keepdims=True)
    p_old = jnp.exp2((s_raw - m_raw) * (sm_scale * 1.4426950408889634))
    s = jnp.where(vis[0], jnp.einsum("nd,md->nm", q, k, precision="highest") * sm_scale, NEG)
    p_new = jnp.exp(s - jnp.max(s, axis=1, keepdims=True))
    live = np.asarray(vis[0]) & (np.asarray(p_new) < 1)
    assert np.mean(np.asarray(p_old)[live] != np.asarray(p_new)[live]) > 0.1
    l_new = jnp.sum(p_new, axis=1)
    recip_old = 1.0 / (l_new + jnp.exp(sinks - m_raw[:, 0] * sm_scale))
    recip_new = attn_ref._approx_reciprocal(l_new + jnp.exp(sinks - jnp.max(s, axis=1)))
    assert np.mean(np.asarray(recip_old) != np.asarray(recip_new)) > 0.1


# ------------------------------------------------------------------ 2. end to end


def mismatch(out, ref):
    """Fraction of bf16 outputs that differ (both sides bf16-rounded)."""
    o = np.asarray(jnp.asarray(out, jnp.bfloat16), np.float32)
    r = np.asarray(jnp.asarray(ref, jnp.bfloat16), np.float32)
    return float(np.mean(o != r))


def _config_override(module, **fields):
    """Builds every WindowConfig of `module` with `fields` forced (controls)."""
    orig = core.WindowConfig

    class Forced:
        def __enter__(self):
            module.core.WindowConfig = lambda **kw: orig(**{**kw, **fields})

        def __exit__(self, *exc):
            module.core.WindowConfig = orig

    return Forced()


PREVIOUS = [dict(pv_dtype=jnp.bfloat16), dict(round_window_acc=False)]


@pytest.mark.parametrize("heads,seq,with_hca", [
    (8, [60, 300, 1030, 2049], True),
    (16, [135, 4100], True),
    (128, [1030, 250], True),
    (8, [70, 900], False),  # SWA-only layer: no hand-off
])
def test_swa_hca_matches_stock(heads, seq, with_hca):
    case = C.swa_hca_case(np.random.default_rng(len(seq) + heads), seq, heads)

    def run():
        return attn_swa_hca.swa_hca_attention(
            case.q, case.new_kv, case.swa_words, case.swa_table, case.hca_words,
            case.hca_table, case.seq_lens, case.sinks, sm_scale=case.sm_scale,
            with_hca=with_hca, blk_pages=4, interpret=True)

    ref = attn_ref.swa_hca_reference(
        case.q, case.new_kv, case.swa_words, case.swa_table,
        case.hca_words if with_hca else None, case.hca_table, case.seq_lens, case.sinks,
        sm_scale=case.sm_scale, stock=True)
    assert mismatch(run(), ref) <= MAX_MISMATCH
    for fields in PREVIOUS[: 2 if with_hca else 1]:
        with _config_override(attn_swa_hca, **fields):
            assert mismatch(run(), ref) > 0.2, fields


@pytest.mark.parametrize("heads,seq,k", [(8, [300, 3000, 9000], 64), (128, [2100, 600], 128)])
def test_csa_matches_stock(heads, seq, k):
    case = C.csa_case(np.random.default_rng(heads), seq, heads=heads)
    w = case.win
    sc, rl = attn_csa.indexer_scores(
        case.q_idx, case.weights, case.idx_words, case.csa_table, w.seq_lens,
        max_len=max(seq), blk_pages=2, interpret=True)
    union, ulen, idx = attn_csa.topk_union_placeholder(sc, rl, k)

    def run():
        return attn_csa.csa_attention(
            w.q, w.new_kv, w.swa_words, w.swa_table, case.nope_words, case.rope_words,
            case.csa_table, union, ulen, w.seq_lens, w.sinks, sm_scale=w.sm_scale,
            interpret=True)

    ref = attn_ref.csa_reference(
        w.q, w.new_kv, w.swa_words, w.swa_table, case.nope_words, case.rope_words,
        case.csa_table, idx, w.seq_lens, w.sinks, sm_scale=w.sm_scale, stock=True)
    assert mismatch(run(), ref) <= MAX_MISMATCH
    for fields in PREVIOUS:
        with _config_override(attn_csa, **fields):
            assert mismatch(run(), ref) > 0.2, fields


KINDS = ("hca", "csa", "hca", "csa")


@pytest.mark.parametrize("kind,layer,seq,max_ctx", [
    ("hca", 0, [1030, 3000], 4096),
    ("csa", 3, [9000, 300], 9216),
])
def test_served_phase_matches_stock(kind, layer, seq, max_ctx):
    """owner_attention_phase, the step kernel's attention (`window_config`)."""
    rng = np.random.default_rng(layer + len(seq))
    ls = PH.build(rng, KINDS, seq, max_ctx)
    inp = PH.make_inputs(rng, ls, kind)
    _, ref, _ = PH.reference(ls, kind, layer, inp, stock=True)
    R = len(seq)

    def run():
        _, out = PH.run_phase(ls, kind, layer, inp)
        return np.asarray(out, np.float32)[:, :7 * 128].reshape(R, 7, 128, 512)

    assert mismatch(run(), ref) <= MAX_MISMATCH
    orig = AP.window_config
    AP.window_config = lambda d, kind="": orig(d, kind).__class__(
        **{**orig(d, kind).__dict__, "pv_dtype": jnp.bfloat16})
    try:
        assert mismatch(run(), ref) > 0.2
    finally:
        AP.window_config = orig


def test_served_draft_phase_matches_stock():
    """Kind "swa" (the DSpark draft): `mla_swa` alone, no hand-off."""
    rng = np.random.default_rng(1300)
    ss = SH.build(rng, 3, [1000, 300], [(995, 7), (299, 3)], 6, 4096)
    inp = SH.make_inputs(rng, ss)
    _, out = SH.run_phase(ss, 1, inp)
    _, ref = SH.reference(ss, 1, inp, stock=True)
    o = out[:, : 6 * 128].reshape(2, 6, 128, 512)
    assert mismatch(o, ref) <= MAX_MISMATCH
