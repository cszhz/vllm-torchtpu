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
"""Qwen3.5 synthetic HuggingFace weight generator."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..ops import BF16, F32, FP8


def random_hf_weights(
    key: jax.Array,
    g: Any,
) -> dict[str, jax.Array]:
    """Generate unsharded HuggingFace-style Qwen3.5 weights (`[L, ...]`, full `E`)."""
    L, H, E, inter_dim, IS = g.L, g.H, g.E, g.I, g.IS
    NH, NKV, HD = g.NH, g.NKV, g.HD
    KH, VH, KD, VD = g.KH, g.VH, g.KD, g.VD
    natt, ngdn = g.NATT, g.NGDN

    keys = jax.random.split(key, 56)
    it = iter(keys)

    def rfp8(shape):
        return (jax.random.normal(next(it), shape, dtype=F32) * 0.25).astype(FP8)

    def rscale(shape):
        return jnp.abs(jax.random.normal(next(it), shape, dtype=F32)) * 0.02 + 0.01

    q_proj = rfp8((natt, H, NH * 2 * HD))
    q_proj_s = rscale((natt, NH * 2 * HD))
    k_proj = rfp8((natt, NKV, H, HD))
    k_proj_s = rscale((natt, NKV, HD))
    v_proj = rfp8((natt, NKV, H, HD))
    v_proj_s = rscale((natt, NKV, HD))
    sh_gate_proj = rfp8((L, H, IS))
    sh_gate_proj_s = rscale((L, IS))
    sh_up_proj = rfp8((L, H, IS))
    sh_up_proj_s = rscale((L, IS))
    e_gate_proj = rfp8((L, E, H, inter_dim))
    e_gate_proj_s = rscale((L, E, inter_dim))
    e_up_proj = rfp8((L, E, H, inter_dim))
    e_up_proj_s = rscale((L, E, inter_dim))

    hf: dict[str, jax.Array] = {
        "in_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "post_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "final_norm": jax.random.normal(next(it), (H,), dtype=F32) * 0.05,
        "q_proj": q_proj,
        "q_proj_s": q_proj_s,
        "k_proj": k_proj,
        "k_proj_s": k_proj_s,
        "v_proj": v_proj,
        "v_proj_s": v_proj_s,
        "att_out": rfp8((natt, NH * HD, H)),
        "att_out_s": rscale((natt, H)),
        "qn": jax.random.normal(next(it), (natt, NH, HD), dtype=F32) * 0.05,
        "kn": jax.random.normal(next(it), (natt, NKV, HD), dtype=F32) * 0.05,
        "router": jax.random.normal(next(it), (L, H, E), dtype=BF16) * 0.1,
        "sh_gate": jax.random.normal(next(it), (L, H), dtype=F32) * 0.1,
        "sh_gate_proj": sh_gate_proj,
        "sh_gate_proj_s": sh_gate_proj_s,
        "sh_up_proj": sh_up_proj,
        "sh_up_proj_s": sh_up_proj_s,
        "sh13": jnp.concatenate([sh_gate_proj, sh_up_proj], axis=-1),
        "sh13_s": jnp.concatenate([sh_gate_proj_s, sh_up_proj_s], axis=-1),
        "sh2": rfp8((L, IS, H)),
        "sh2_s": rscale((L, H)),
        "e_gate_proj": e_gate_proj,
        "e_gate_proj_s": e_gate_proj_s,
        "e_up_proj": e_up_proj,
        "e_up_proj_s": e_up_proj_s,
        "e13": jnp.concatenate([e_gate_proj, e_up_proj], axis=-1),
        "e13_s": jnp.concatenate([e_gate_proj_s, e_up_proj_s], axis=-1),
        "e2": rfp8((L, E, inter_dim, H)),
        "e2_s": rscale((L, E, H)),
    }
    if ngdn > 0:
        qkv_full = 2 * KH * KD + VH * VD
        hf.update(
            {
                "lin_q": rfp8((ngdn, H, KH * KD)),
                "lin_q_s": rscale((ngdn, KH * KD)),
                "lin_k": rfp8((ngdn, H, KH * KD)),
                "lin_k_s": rscale((ngdn, KH * KD)),
                "lin_v": rfp8((ngdn, H, VH * VD)),
                "lin_v_s": rscale((ngdn, VH * VD)),
                "lin_z": rfp8((ngdn, H, VH * VD)),
                "lin_z_s": rscale((ngdn, VH * VD)),
                "lin_b": rfp8((ngdn, H, VH)),
                "lin_b_s": rscale((ngdn, VH)),
                "lin_a": rfp8((ngdn, H, VH)),
                "lin_a_s": rscale((ngdn, VH)),
                "lin_out": rfp8((ngdn, VH * VD, H)),
                "lin_out_s": rscale((ngdn, H)),
                "conv": jax.random.normal(next(it), (ngdn, 4, qkv_full), dtype=F32)
                * 0.05,
                "alog": jax.random.normal(next(it), (ngdn, VH), dtype=F32) * 0.1 - 1.0,
                "dtb": jax.random.normal(next(it), (ngdn, VH), dtype=F32) * 0.1,
                "gnorm": jnp.ones((ngdn, VD), dtype=F32)
                + jax.random.normal(next(it), (ngdn, VD), dtype=F32) * 0.05,
            }
        )
    return hf


__all__ = [
    "random_hf_weights",
]
