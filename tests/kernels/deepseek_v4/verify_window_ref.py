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
  All   one softmax over the union of the SWA and compressed keys plus the
        per-head attention sink, as the kernels' partial merge computes.

Positions: ``seq_lens`` count the new rows, so the first query row of
request r sits at ``seq_lens[r] - q_len``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.core_attention.verify_window import attn_layout as L


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
            rows_t.append(_attend(q[r, t], keys, jnp.asarray(vis), sinks, sm_scale))
        out.append(jnp.stack(rows_t))
    return jnp.stack(out)
