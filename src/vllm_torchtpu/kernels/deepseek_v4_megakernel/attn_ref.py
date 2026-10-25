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
"""Plain-JAX reference of the served DeepSeek-V4 verify-window attention.

Reproduces, in f32 and over the full (unnarrowed) page tables, what the
serving path computes for R requests of ``q_len`` consecutive query rows:

  SWA   `mla_sliding_window_ragged_paged_attention`: query at position p sees
        keys ``p - window < k <= p``, the in-flight rows from ``new_kv``.
  HCA   `mla_ragged_paged_attention`: compressed rows ``j < (p + 1) // ratio``.
  CSA   `streamindex_topk` scores, `sparsecore_topk` selection (top-k by
        monotone key, ties to the lowest index, ascending, -1 padded) and
        `sparse_ragged_paged_attention` over the selected rows.
  All   one softmax over the union of the SWA and compressed keys plus the
        per-head attention sink, as the kernels' partial merge computes.

With ``stock=True`` the references instead follow the served kernels'
arithmetic step by step (`_attend_stock`), bf16-rounded output included.

Positions: ``seq_lens`` count the new rows, so the first query row of
request r sits at ``seq_lens[r] - q_len``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_layout as L


def _gather_bf16_tokens(cache_words, page_table_r, positions, page_size):
    """bf16-latent tokens at `positions` of one request -> f32 ``[n, 512]``."""
    phys = page_table_r[positions // page_size]
    slot = positions % page_size
    rows = cache_words.reshape(cache_words.shape[0], page_size, 2, L.LANES)
    w = rows[phys, slot]  # [n, 2, 128]
    return L.bf16_tokens_from_words(w[:, 0], w[:, 1]).astype(jnp.float32)


def swa_keys(new_kv_r, swa_words, swa_table_r, seq_len, *, q_len, window, page_size):
    """Keys the window can reach: positions ``[p0 - window + 1, p0 + q_len)``.

    Returns f32 keys ``[window - 1 + q_len, 512]`` and their positions (negative
    positions are padding, never visible).
    """
    p0 = seq_len - q_len
    pos = p0 - (window - 1) + np.arange(window - 1 + q_len)
    cached = _gather_bf16_tokens(
        swa_words, swa_table_r, np.maximum(pos[: window - 1], 0), page_size
    )
    keys = jnp.concatenate([cached, new_kv_r.astype(jnp.float32)], axis=0)
    return keys, pos


def _attend(q_t, keys, visible, sinks, sm_scale):
    """One query row: q_t ``[H, 512]``, keys ``[n, 512]``, visible ``[n]``."""
    s = jnp.einsum("hd,nd->hn", q_t, keys, precision="highest") * sm_scale
    s = jnp.where(visible[None, :], s, -jnp.inf)
    m = jnp.maximum(jnp.max(s, axis=1, keepdims=True), sinks[:, None])
    p = jnp.exp(s - m)
    denom = jnp.sum(p, axis=1, keepdims=True) + jnp.exp(sinks[:, None] - m)
    return jnp.einsum("hn,nd->hd", p, keys, precision="highest") / denom


# The kernels' `pl.reciprocal(approx=True)`, jitted (the primitive has no eager
# rule). Off TPU it lowers to 1 / bf16(x), as in interpret mode.
_approx_reciprocal = jax.jit(lambda x: pltpu.reciprocal(x, approx=True))


def _attend_stock(q_t, keys, visible, n_window, sinks, sm_scale, compressed):
    """`_attend` in the served kernels' arithmetic: bf16 ``[H, 512]`` as f32.

    Keys ``[:n_window]`` are the SWA pass (`mla_swa`), the rest the compressed
    pass (`mla` for HCA, `sparse_mla` for CSA) continuing its softmax state.
    Both: ``s = (q @ k) * sm_scale``, masked to the f32 minimum, ``p = exp(s -
    m)`` kept f32 for the row sum and P.V, and ``out = acc * reciprocal(l +
    exp(sink - m), approx=True)`` cast to bf16. With a compressed pass
    (`compressed`, even when it sees no rows) the SWA kernel hands its
    accumulator over as bf16.
    """
    neg = jnp.finfo(jnp.float32).min
    s = jnp.einsum("hd,nd->hn", q_t, keys, precision="highest") * sm_scale
    s = jnp.where(visible[None, :], s, neg)
    sw, kw = s[:, :n_window], keys[:n_window]
    m = jnp.max(sw, axis=1, keepdims=True)
    p = jnp.exp(sw - m)
    l = jnp.sum(p, axis=1, keepdims=True)  # noqa: E741
    acc = jnp.einsum("hn,nd->hd", p, kw, precision="highest")
    if compressed:
        acc = acc.astype(jnp.bfloat16).astype(jnp.float32)
        sc, kc = s[:, n_window:], keys[n_window:]
        if kc.shape[0]:
            m_new = jnp.maximum(m, jnp.max(sc, axis=1, keepdims=True))
            p = jnp.exp(sc - m_new)
            alpha = jnp.exp(m - m_new)
            l = alpha * l + jnp.sum(p, axis=1, keepdims=True)  # noqa: E741
            acc = alpha * acc + jnp.einsum("hn,nd->hd", p, kc, precision="highest")
            m = m_new
    l_total = l + jnp.exp(sinks[:, None] - m)
    out = acc * _approx_reciprocal(l_total)
    return out.astype(jnp.bfloat16).astype(jnp.float32)


def swa_hca_reference(
    q,  # bf16 [R, q_len, H, 512]
    new_kv,  # bf16 [R, q_len, 512]
    swa_words,  # int32 [N, 2 * swa_page, 128]
    swa_table,  # int32 [R, P_swa], logical page = position // swa_page
    hca_words,  # int32 [N_h, 2 * hca_page, 128] or None for SWA-only layers
    hca_table,  # int32 [R, P_hca], logical page = compressed row // hca_page
    seq_lens,  # int32 [R]
    sinks,  # f32 [H]
    *,
    sm_scale,
    window=128,
    swa_page=128,
    ratio=128,
    hca_page=8,
    stock=False,
):
    """f32 ``[R, q_len, H, 512]``."""
    q = np.asarray(q, np.float32)
    seq_lens = np.asarray(seq_lens)
    swa_table, hca_table = np.asarray(swa_table), np.asarray(hca_table)
    num_req, q_len = q.shape[:2]
    out = []
    for r in range(num_req):
        seq_len = int(seq_lens[r])
        p0 = seq_len - q_len
        keys, kpos = swa_keys(
            new_kv[r], swa_words, swa_table[r], seq_len,
            q_len=q_len, window=window, page_size=swa_page,
        )
        n_c = seq_len // ratio if hca_words is not None else 0
        if n_c:
            rows = np.arange(n_c)
            hkeys = _gather_bf16_tokens(hca_words, hca_table[r], rows, hca_page)
            keys = jnp.concatenate([keys, hkeys], axis=0)
        rows_t = []
        for t in range(q_len):
            qpos = p0 + t
            vis = (kpos >= 0) & (kpos <= qpos) & (kpos > qpos - window)
            if n_c:
                vis = np.concatenate([vis, np.arange(n_c) < (qpos + 1) // ratio])
            if stock:
                rows_t.append(_attend_stock(q[r, t], keys, jnp.asarray(vis), len(kpos),
                                            sinks, sm_scale, hca_words is not None))
            else:
                rows_t.append(_attend(q[r, t], keys, jnp.asarray(vis), sinks, sm_scale))
        out.append(jnp.stack(rows_t))
    return jnp.stack(out)


# ---------------------------------------------------------------- CSA


def sal_keys(idx_words, table_r, n_rows, *, page_size=256, head_dim=128):
    """Seq-along-lane indexer K of one request -> (f32 keys ``[n, 128]``,
    f32 scales ``[n]``)."""
    groups = head_dim // 4
    pages = idx_words[table_r[: -(-n_rows // page_size)]]  # [p, G, page]
    w = pages[:, :groups]  # [p, 32, page]
    dims = jnp.stack([L.e4m3_from_byte((w >> (8 * i)) & 0xFF) for i in range(4)], 2)
    keys = dims.reshape(pages.shape[0], head_dim, page_size).transpose(0, 2, 1)
    scale = L.e8m0_from_byte(pages[:, groups] & 0xFF)  # [p, page]
    return keys.reshape(-1, head_dim)[:n_rows], scale.reshape(-1)[:n_rows]


def indexer_scores_reference(
    q_idx,  # f32 values of the e4m3 queries [R, q_len, Hi, 128]
    weights,  # f32 [R, q_len, Hi] (query scales and softmax scale folded in)
    idx_words,  # int32 [N, G, page] seq-along-lane words
    idx_table,  # int32 [R, P]
    seq_lens,  # int32 [R]
    *,
    ratio=4,
    page_size=256,
    strict_causal=False,
):
    """Indexer logits as `streamindex_topk` scores a verify window.

    Returns f32 scores ``[R, q_len, n_max]`` (``n_max = max seq_len // ratio``,
    invisible entries ``-inf``) and the visible row lengths ``[R, q_len]``.
    The head sum runs head 0 first, one f32 multiply-add per head.
    """
    seq_lens = np.asarray(seq_lens)
    num_req, q_len, n_heads, _ = q_idx.shape
    n_max = int(seq_lens.max()) // ratio
    scores, lens = [], []
    for r in range(num_req):
        n = int(seq_lens[r]) // ratio
        keys, scale = sal_keys(idx_words, np.asarray(idx_table[r]), n, page_size=page_size)
        st = jnp.einsum("thd,nd->htn", q_idx[r], keys, precision="highest")
        acc = jnp.maximum(st[0], 0.0) * weights[r, :, 0:1]
        for h in range(1, n_heads):
            acc = acc + jnp.maximum(st[h], 0.0) * weights[r, :, h : h + 1]
        acc = acc * scale[None, :]
        pos = int(seq_lens[r]) - q_len + np.arange(q_len)
        vis = np.minimum(n, (pos + 1) // ratio if strict_causal else pos // ratio + 1)
        acc = jnp.where(np.arange(n)[None, :] < vis[:, None], acc, -jnp.inf)
        scores.append(jnp.pad(acc, ((0, 0), (0, n_max - n)), constant_values=-jnp.inf))
        lens.append(vis)
    return jnp.stack(scores), jnp.asarray(np.stack(lens), jnp.int32)


def topk_reference(scores, row_lens, k):
    """`sparsecore_topk` semantics: top-k by monotone key, ties to the lowest
    index, indices ascending, -1 padded.  ``scores [..., n]``, ``row_lens [...]``.
    """
    n = scores.shape[-1]
    col = jnp.arange(n, dtype=jnp.int32)
    keys = jnp.where(col < row_lens[..., None], L.monotone_key(scores), jnp.iinfo(jnp.int32).min)
    if n < k:
        keys = jnp.pad(keys, [(0, 0)] * (keys.ndim - 1) + [(0, k - n)],
                       constant_values=jnp.iinfo(jnp.int32).min)
    _, idx = jax.lax.top_k(keys, k)  # stable: equal keys keep index order
    take = jnp.arange(k) < jnp.minimum(row_lens, k)[..., None]
    idx = jnp.sort(jnp.where(take, idx, jnp.iinfo(jnp.int32).max), axis=-1)
    return jnp.where(idx == jnp.iinfo(jnp.int32).max, -1, idx).astype(jnp.int32)


def csa_rows(nope_words, rope_words, table_r, idx, *, page_size=256):
    """Compressed rows `idx` (>= 0) of one request -> f32 ``[n, 512]``."""
    phys = table_r[idx // page_size]
    slot = idx % page_size
    nope = L.nope_tokens_from_words(nope_words[phys, slot])
    rrow = rope_words[phys, slot // 4]  # [n, 128]
    seg = (slot % 4)[:, None] * L.ROPE_WORDS + np.arange(L.ROPE_WORDS)[None, :]
    rope = L.rope_tokens_from_words(jnp.take_along_axis(rrow, jnp.asarray(seg), axis=1))
    return jnp.concatenate([nope, rope], axis=-1).astype(jnp.float32)


def csa_reference(
    q,  # bf16 [R, q_len, H, 512]
    new_kv,  # bf16 [R, q_len, 512]
    swa_words,
    swa_table,
    nope_words,  # int32 [N, page, 128]
    rope_words,  # int32 [N, page // 4, 128]
    csa_table,  # int32 [R, P]
    topk_idx,  # int32 [R, q_len, k], ascending, -1 padded
    seq_lens,
    sinks,
    *,
    sm_scale,
    window=128,
    swa_page=128,
    page_size=256,
    stock=False,
):
    """f32 ``[R, q_len, H, 512]``: SWA + sparse attention over the top-k rows."""
    q = np.asarray(q, np.float32)
    seq_lens, topk_idx = np.asarray(seq_lens), np.asarray(topk_idx)
    swa_table, csa_table = np.asarray(swa_table), np.asarray(csa_table)
    num_req, q_len = q.shape[:2]
    out = []
    for r in range(num_req):
        seq_len = int(seq_lens[r])
        p0 = seq_len - q_len
        keys, kpos = swa_keys(
            new_kv[r], swa_words, swa_table[r], seq_len,
            q_len=q_len, window=window, page_size=swa_page,
        )
        rows_t = []
        for t in range(q_len):
            qpos = p0 + t
            sel = topk_idx[r, t]
            sel = sel[sel >= 0]
            ckeys = csa_rows(nope_words, rope_words, csa_table[r], sel, page_size=page_size)
            vis = (kpos >= 0) & (kpos <= qpos) & (kpos > qpos - window)
            vis = np.concatenate([vis, np.ones(len(sel), bool)])
            allk = jnp.concatenate([keys, ckeys], axis=0)
            if stock:
                rows_t.append(_attend_stock(q[r, t], allk, jnp.asarray(vis), len(kpos),
                                            sinks, sm_scale, True))
            else:
                rows_t.append(_attend(q[r, t], allk, jnp.asarray(vis), sinks, sm_scale))
        out.append(jnp.stack(rows_t))
    return jnp.stack(out)


def indexer_q_reference(idx_q, idx_w, cos_sin, positions, *, softmax_scale=128 ** -0.5,
                        n_head=64):
    """`rope_quant` + the weight fold of `_indexer_jax`, plain JAX.

    idx_q bf16 [R, T, 64 * 128] (pre-RoPE), idx_w f32 [R, T, 64] (the `weights_proj`
    accumulator: stock's bf16 linear rounds it to bf16), positions [R, T].
    Returns the e4m3 queries as f32 values [R, T, 64, 128] and the folded weights
    [R, T, 64].
    """
    R, T, _ = idx_q.shape
    x = idx_q.reshape(R, T, n_head, 128).astype(jnp.float32)
    cs = cos_sin[positions]  # [R, T, 64]
    half = cs.shape[-1] // 2
    ch = jnp.arange(128) - 64
    freq = jnp.maximum(ch, 0) // 2
    cos = jnp.where(ch >= 0, cs[..., freq], 1.0)[:, :, None, :]
    sin = jnp.where(ch >= 0, cs[..., half + freq], 0.0)[:, :, None, :]
    lane = jnp.arange(128)
    rot = x[..., lane ^ 1] * jnp.where(lane % 2 == 0, -1.0, 1.0)
    y = x * cos + rot * sin
    amax = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
    scale = amax / 448.0
    inv = jnp.where(amax == 0.0, 0.0, 1.0 / scale)
    q = (y * inv).astype(jnp.float8_e4m3fn).astype(jnp.float32)
    w = idx_w.astype(jnp.bfloat16) * softmax_scale * (n_head ** -0.5) * scale[..., 0]
    return q, w
