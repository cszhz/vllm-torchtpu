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

"""Pallas FlashAttention inner-loop primitives for Batched Ragged Paged Attention.

Implements tiled QK^T matmul, online softmax rescaling, and PV accumulation
with optional native FP8 KV cache dequantization in VMEM.
"""

import math
import os

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, utils
from vllm_torchtpu.kernels.experimental.batched_rpa.schedule import SmemWrapper


def _use_native_fp8(cfgs: configs.RpaConfigs, which: str) -> bool:
    if cfgs.mode == configs.RpaCase.DECODE:
        return cfgs.serve.native_fp8_enabled(which)
    return cfgs.serve.native_fp8_enabled(which, prefill=True)


def _static_fp8(cfgs: configs.RpaConfigs, which: str) -> bool:
    """Whether a native fp8 operand takes the static scale."""
    if cfgs.mode == configs.RpaCase.DECODE:
        return cfgs.serve.native_fp8_static(which)
    return cfgs.serve.native_fp8_prefill_static(which)


# The softmax weights are at most 1. The static PV path casts p * P_SCALE
# to fp8, so the PV accumulator carries the factor, and the final division
# by l (which sums the unscaled weights) removes it. A power of two keeps the
# product exact in bf16 and below the e4m3 maximum.
P_SCALE = 256.0


def static_pv(cfgs: configs.RpaConfigs) -> bool:
    """Whether PV takes fp8 weights at the static scale P_SCALE.

    Per-token V scales multiply p before the cast and can push it past the
    fp8 range, so that case keeps the per-row scale.
    """
    return (
        _use_native_fp8(cfgs, "pv")
        and _static_fp8(cfgs, "pv")
        and not cfgs.serve.per_token_scale
    )


# Weight of the low fp8 half in a split (hi + lo / _SPLIT_LO) quantization.
_SPLIT_LO = 16.0


def _quantize_rows(
    x: jax.Array, dtype: jnp.dtype, split: bool = False
) -> tuple[jax.Array, jax.Array]:
    """Quantizes x to fp8 `dtype` with one dynamic scale per row (last axis).

    Returns the quantized array and the f32 row scales (keepdims), such that
    x ~= x_q * scale. Rows of zeros get scale 1 (and stay zeros).

    With `split`, also quantizes the rounding residual and stacks it under the
    high part along axis -2: x ~= (hi + lo / _SPLIT_LO) * scale, which keeps
    about 7 mantissa bits (bf16-like) at the cost of doubling the matmul's LHS
    rows, which are nearly free in decode (16 of the MXU's rows are used).
    """
    x = x.astype(jnp.float32)
    fp8_max = float(jnp.finfo(dtype).max)
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale = amax * (1.0 / fp8_max)
    scale = jnp.where(scale > 0, scale, 1.0)
    x_s = jnp.clip(x * (1.0 / scale), -fp8_max, fp8_max)
    x_q = x_s.astype(dtype)
    if split:
        resid = (x_s - x_q.astype(jnp.float32)) * _SPLIT_LO
        x_q = jnp.concatenate([x_q, resid.astype(dtype)], axis=-2)
    return x_q, scale


def _merge_split(y: jax.Array, rows: int) -> jax.Array:
    """Combines a matmul over split (hi; lo) LHS rows: [..., 2*rows, N]."""
    return y[..., :rows, :] + y[..., rows:, :] * (1.0 / _SPLIT_LO)


def decode_exp2_enabled() -> bool:
    """TPU_RPA_DECODE_EXP2=1 runs the DECODE softmax in base 2: log2(e) is
    folded into the score scale and the exponentials are `exp2`; read at
    trace time. Softmax is invariant to the base change; q (or the scores)
    takes one extra rounding of the folded scale."""
    return os.environ.get("TPU_RPA_DECODE_EXP2", "0") == "1"


def decode_exp2(cfgs: configs.RpaConfigs) -> bool:
    """Whether this kernel's softmax runs in base 2 (`decode_exp2_enabled`)."""
    return (
        decode_exp2_enabled()
        and cfgs.mode == configs.RpaCase.DECODE
        and cfgs.model.soft_cap is None
        and not cfgs.serve.return_lse
    )


def p_prescaled(cfgs: configs.RpaConfigs) -> bool:
    """Whether the softmax weights leave the exponential already times P_SCALE.

    With the base-2 decode softmax and the static fp8 PV, log2(P_SCALE) is
    subtracted from the row max before the exponential, so the weights, l
    and the accumulator all carry the factor and the separate multiply of
    the weights (and of l in `calculate_and_store_out`) is not needed.
    """
    return decode_exp2(cfgs) and static_pv(cfgs)


def decode_mxu_acc_mode() -> str:
    """TPU_RPA_DECODE_MXU_ACC runs DECODE dots as explicit MXU ops; read at
    trace time.

    The value is a comma-separated list: `pv` or `qkpv` names the dots that
    run as `matmul_push_rhs` / `matmul_acc_lhs` / `matmul_pop` (TPU generation
    7 and later), so a contraction over the KV block accumulates in the MXU
    result buffer and is popped once, instead of one pop and one vector add
    per 256-wide pass; `t` flips the transpose of the pushed RHS, `s1` uses a
    single staging register. Unset, every dot lowers as `lax.dot`.
    """
    return os.environ.get("TPU_RPA_DECODE_MXU_ACC", "")


def mxu_acc(cfgs: configs.RpaConfigs, which: str) -> bool:
    """Whether the `which` ("qk" or "pv") dot runs as explicit MXU ops."""
    mode = decode_mxu_acc_mode().split(",")[0]
    if which not in mode:
        return False
    if cfgs.mode != configs.RpaCase.DECODE:
        return False
    if cfgs.serve.kv_layout != configs.KVLayout.SEQ_ALONG_LANE:
        return False
    if cfgs.serve.native_fp8_split(which):
        return False
    if cfgs.bq_sz % cfgs.bq_c_sz != 0:
        return False
    info = pltpu.get_tpu_info()
    if info.generation < 7 or cfgs.aligned_kv_head_dim != info.mxu_column_size:
        return False
    return cfgs.bkv_sz % info.mxu_column_size == 0


def _mxu_acc_options() -> tuple[bool, int, bool]:
    """(flip the RHS transpose, staging registers, interleave the chains)."""
    opts = decode_mxu_acc_mode().split(",")[1:]
    return "t" in opts, (1 if "s1" in opts else 2), "i" in opts


def _mxu_accs_per_entry(which: str) -> int:
    """Accumulators per chain: the interleaved QK chain pops one chunk behind
    its pushes, so it alternates between two."""
    return 2 if (which == "qk" and _mxu_acc_options()[2]) else 1


def mxu_acc_shapes(cfgs: configs.RpaConfigs) -> dict[str, tuple]:
    """MXU accumulator scratch refs for the explicit dots, one per batch entry
    and KV head; `mxu_acc_ids` gives each entry's MXU."""
    shapes = {}
    rows = cfgs.bq_c_sz * cfgs.aligned_num_q_heads_per_kv_head
    cols = pltpu.get_tpu_info().mxu_column_size
    for which in ("qk", "pv"):
        if mxu_acc(cfgs, which):
            shapes[which] = tuple(
                pltpu.ACC(mxu)((rows, cols), jnp.float32)
                for mxu in mxu_acc_ids(cfgs)
                for _ in range(_mxu_accs_per_entry(which))
            )
    return shapes


def mxu_acc_ids(cfgs: configs.RpaConfigs) -> tuple[int, ...]:
    """MXU of each (batch entry, KV head) chain, round robin over the MXUs."""
    n = cfgs.batch_size * cfgs.model.num_kv_heads
    num_mxus = pltpu.get_tpu_info().num_mxus
    return tuple(i % num_mxus for i in range(n))


def _mxu_dot_t(
    lhs: jax.Array,  # [N, M, K]
    rhs_t: jax.Array,  # [N, C, K]: contracted on its last dim
    accs: tuple,  # N accumulator refs [M, 256]
    mxu_ids: tuple[int, ...],
    *,
    pop_each: bool,
) -> jax.Array:
    """Computes lhs @ rhs_t^T per batch entry with explicit MXU ops.

    One MXU chain per batch entry: every 256-wide K chunk pushes its RHS tile
    into a staging register and accumulates the LHS chunk into the entry's
    accumulator, which is popped once. With `pop_each`, each chunk's result
    is popped as it completes (the chunks are then output columns, not
    contraction passes, and `rhs_t` is [N, K, C] contracted on its middle
    dim).
    """
    info = pltpu.get_tpu_info()
    col = info.mxu_column_size
    flip_t, n_staging, interleave = _mxu_acc_options()
    n, rows, k_size = lhs.shape
    n_chunks = (rhs_t.shape[2] if pop_each else k_size) // col
    per_entry = len(accs) // n

    def push(i, c):
        reg = c % n_staging
        pltpu.matmul_push_rhs(
            rhs_t[i, :, c * col : (c + 1) * col],
            reg,
            mxu_ids[i],
            transpose=(flip_t if pop_each else not flip_t),
        )
        lhs_c = lhs[i] if pop_each else lhs[i, :, c * col : (c + 1) * col]
        pltpu.matmul_acc_lhs(accs[i * per_entry + c % per_entry], lhs_c, load_staged_rhs=reg)

    def pop(i, c):
        return pltpu.matmul_pop(accs[i * per_entry + c % per_entry])

    chunks = [[None] * n_chunks for _ in range(n)]
    if interleave:
        # Every chain's chunk c is pushed before any chunk c result is popped,
        # so the pops wait on work issued one chunk earlier.
        for c in range(n_chunks + 1):
            for i in range(n):
                if c < n_chunks:
                    push(i, c)
            for i in range(n):
                if pop_each and c >= 1:
                    chunks[i][c - 1] = pop(i, c - 1)
                elif not pop_each and c == n_chunks:
                    chunks[i][0] = pop(i, 0)
    else:
        for i in range(n):
            for c in range(n_chunks):
                push(i, c)
                if pop_each:
                    chunks[i][c] = pop(i, c)
            if not pop_each:
                chunks[i][0] = pop(i, 0)
    if pop_each:
        return jnp.stack([jnp.concatenate(ch, axis=1) for ch in chunks], axis=0)
    return jnp.stack([ch[0] for ch in chunks], axis=0)


def vpu_diet_enabled() -> bool:
    """A/B switch for the score-tile VPU savings (scale folding, mask skip).

    Read at trace time, like TPU_RPA_SKIP_MIXED. Both paths are numerically
    equivalent: the mask skip is exact, and the scale folding is exact when the
    folded scale is a power of two (sm_scale=1/16, k_scale=1 for head_dim 256).
    """
    return os.environ.get("TPU_RPA_VPU_DIET", "1") == "1"


def q_folded_scale(cfgs: configs.RpaConfigs) -> float | None:
    """Returns the static qk scale if it can be folded into q exactly.

    Scaling a bf16/f32 q by a power of two only changes its exponent, so
    dot(q * s, k) is bit-identical to dot(q, k) * s while touching a
    [TQ, head_dim] tile instead of the [TQ, bkv] f32 score tile.
    """
    serve = cfgs.serve
    if serve.scale_q is not None or serve.per_token_scale:
        return None
    if not jnp.issubdtype(serve.dtype_q, jnp.floating):
        return None
    if jnp.dtype(serve.dtype_q).itemsize < 2:
        return None
    scale = float(cfgs.model.sm_scale)
    if serve.scale_k is not None:
        scale *= float(serve.scale_k)
    if not (scale > 0.0 and math.isfinite(scale)):
        return None
    if math.frexp(scale)[0] != 0.5:
        return None
    return scale


def flash_attention_qk_softmax(
    step: jax.Array,
    q: jax.Array,  # [B, KV, TQ, H]
    k: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    m_prev: jax.Array,  # [KV, TQ, 128]
    l_prev: jax.Array,  # [KV, TQ, 128]
    is_last_k: SmemWrapper,  # [steps, B]
    *,
    custom_mask: jax.Array | None,
    cfgs: configs.RpaConfigs,
    bq_start: int,
    k_scale: jax.Array | float | None = None,  # [B, KV, S]
    dma_q: jax.Ref | None = None,
    mask_bias: jax.Array | None = None,  # [B, KV, TQ, S] f32
    mxu_accs: tuple | None = None,
):
    """Flash attention kernel.

    `custom_mask=None` means every score in the tile is visible (the caller
    proved the KV block lies entirely below the causal diagonal), so the
    mask select is skipped. `mask_bias` is an additive f32 mask (0 or
    `mask_value`) that joins the QK dot as its accumulator.
    """
    b, k_heads, tq, h_size = q.shape

    diet = vpu_diet_enabled()
    row_scale = None
    native_qk = _use_native_fp8(cfgs, "qk")
    static_qk = native_qk and _static_fp8(cfgs, "qk")
    split_qk = (
        native_qk and cfgs.mode == configs.RpaCase.DECODE
    ) and cfgs.serve.native_fp8_split("qk")
    use_exp2 = decode_exp2(cfgs)
    base_scale = math.log2(math.e) if use_exp2 else 1.0
    folded_scale = q_folded_scale(cfgs) if (diet and not native_qk) else None
    if folded_scale is not None and folded_scale * base_scale != 1.0:
        q = q * jnp.asarray(folded_scale * base_scale, dtype=q.dtype)

    if static_qk:
        fp8_max = float(jnp.finfo(k.dtype).max)
        q = jnp.clip(q, -fp8_max, fp8_max).astype(k.dtype)
    elif native_qk:
        q, row_scale = _quantize_rows(q, k.dtype, split=split_qk)
    elif cfgs.serve.scale_q is not None:
        q = q / cfgs.serve.scale_q
        if jnp.issubdtype(k.dtype, jnp.floating):
            dtype_info = jnp.finfo(k.dtype)
            minval = float(dtype_info.min)
            maxval = float(dtype_info.max)
            q = jnp.clip(q, min=minval, max=maxval)
        q = q.astype(k.dtype)

    q_rows = 2 * tq if split_qk else tq
    seq_along_lane = cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE
    if seq_along_lane and utils.bench_skip("matmul"):
        # Bench stand-in for QK: one K row per lane broadcast over the query
        # rows, so the score tile still depends on the fetched K.
        s = k.shape[-1]
        qk = jnp.broadcast_to(
            k.reshape(-1, h_size, s)[:, :1, :].astype(jnp.float32),
            (b * k_heads, q_rows, s),
        )
    elif seq_along_lane and mxu_acc(cfgs, "qk"):
        s = k.shape[-1]
        qk = _mxu_dot_t(
            q.reshape(-1, q_rows, h_size),
            k.reshape(-1, h_size, s),
            mxu_accs,
            mxu_acc_ids(cfgs),
            pop_each=True,
        )
    elif seq_along_lane:
        s = k.shape[-1]
        qk = lax.dot(
            q.reshape(-1, q_rows, h_size),
            k.reshape(-1, h_size, s),
            dimension_numbers=(([2], [1]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    else:
        s = k.shape[-2]
        qk = lax.dot(
            q.reshape(-1, q_rows, h_size),
            k.reshape(-1, s, h_size),
            dimension_numbers=(([2], [2]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    if mask_bias is not None and mask_bias.dtype == jnp.float32:
        # Written as a plain add so Mosaic folds it into the dot's accumulator.
        qk = qk + mask_bias.reshape(qk.shape)
    if split_qk:
        qk = _merge_split(qk, tq)
    qk = qk.reshape(b, k_heads, tq, s)

    if folded_scale is not None:
        pass  # Already applied to q.
    elif diet or native_qk:
        # Fold the static scalars together so the [TQ, S] f32 tile is scaled
        # once; a per-token k_scale is only [B, KV, 1, S].
        scale = float(cfgs.model.sm_scale) * base_scale
        if cfgs.serve.scale_q is not None and not native_qk:
            scale *= float(cfgs.serve.scale_q)
        if k_scale is None:
            pass
        elif isinstance(k_scale, (int, float)):
            scale *= float(k_scale)
        elif row_scale is None:
            k_scale = k_scale * scale
            scale = 1.0
        if row_scale is not None:
            qk *= row_scale.reshape(b, k_heads, tq, 1) * scale
        elif scale != 1.0:
            qk *= scale
        if k_scale is not None and not isinstance(k_scale, (int, float)):
            qk *= k_scale
    else:
        qk *= cfgs.model.sm_scale * base_scale
        if cfgs.serve.scale_q is not None:
            qk *= cfgs.serve.scale_q
        if k_scale is not None:
            qk *= k_scale

    # We convert to the output dtype after scaling, because especially for very
    # low precision, we want to have 32-bit scale factors and do scaling in fp32.
    qk = qk.astype(cfgs.serve.dtype_out)
    if mask_bias is not None and mask_bias.dtype != jnp.float32:
        qk = qk + mask_bias.astype(qk.dtype)

    if cfgs.model.soft_cap is not None:
        qk = cfgs.model.soft_cap * jnp.tanh(qk / cfgs.model.soft_cap)

    if custom_mask is not None:
        qk_masked = []
        for b_idx in range(cfgs.block.batch_size):
            mask_b = custom_mask[b_idx]
            qk_masked.append(jnp.where(mask_b, qk[b_idx], cfgs.model.mask_value))
        qk = jnp.stack(qk_masked, axis=0)

    # Bench stand-ins for the softmax: the row max becomes a lane slice and
    # the exponentials identities, keeping every tile shape and dependency.
    skip_softmax = utils.bench_skip("softmax")
    _exp = (lambda x: x) if skip_softmax else (jnp.exp2 if use_exp2 else jnp.exp)
    if skip_softmax:
        m_curr = qk[..., :1]
    else:
        m_curr = jnp.max(qk, axis=-1, keepdims=True)

    alpha_list = []
    m_next_list = []

    for b_idx in range(cfgs.block.batch_size):
        m_curr_b = m_curr[b_idx]
        m_next_b = jnp.maximum(m_prev, m_curr_b)
        alpha_b = _exp(m_prev - m_next_b)
        alpha_list.append(alpha_b)
        m_next_list.append(m_next_b)
        reset_m = is_last_k[step, b_idx] != 0
        if dma_q is not None:
            reset_m = jnp.logical_or(reset_m, bq_start >= dma_q[step, b_idx, 1])
        m_prev = jnp.where(reset_m, -jnp.inf, m_next_b)

    m_next = jnp.stack(m_next_list, axis=0)
    m_shift = m_next
    if p_prescaled(cfgs):
        m_shift = m_next - jnp.asarray(math.log2(P_SCALE), m_next.dtype)
    p = _exp(qk - utils.broadcast_minor(m_shift, qk.shape))
    p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=cfgs.serve.dtype_out)

    l_next_list = []
    for b_idx in range(cfgs.block.batch_size):
        l_next_b = alpha_list[b_idx] * l_prev + p_rowsum[b_idx]
        l_next_list.append(l_next_b)
        if cfgs.fuse_accum:
            # Reset with a select rather than relying on alpha == 0 (since
            # 0 * NaN == NaN). Note: these NaNs are not produced by this kernel
            # itself; they are observed mid-batch during decode when a corrupted
            # MTP draft writes NaN into one sequence's KV cache, which would
            # otherwise leak into every subsequent sequence in the batch. The
            # normal/default prefill path (`not cfgs.fuse_accum`) is unaffected
            # and skips this select to avoid extra vector ops across `bq_c`
            # query chunks.
            l_prev = jnp.where(is_last_k[step, b_idx], 0.0, l_next_b)
        else:
            l_prev = l_next_b

    l_next = jnp.stack(l_next_list, axis=0)

    return p, alpha_list, m_next, l_next, m_prev, l_prev


def flash_attention_qk(
    q: jax.Array,  # [B, KV, TQ, H]
    k: jax.Array,  # [B, KV, H, S]
    *,
    custom_mask: jax.Array | None,
    cfgs: configs.RpaConfigs,
    bq_start: int,
    k_scale: jax.Array | float | None = None,
    mask_bias: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Scores in the output dtype and their row max, for the skewed pipeline.

    The first half of `flash_attention_qk_softmax` (scale folding, the QK dot,
    the mask, the cast and the row max); SEQ_ALONG_LANE, no split or per-row
    quantized q, no soft cap. Returns `qk` [B, KV, TQ, S] and `m_curr`
    [B, KV, TQ, 1].
    """
    b, k_heads, tq, h_size = q.shape
    diet = vpu_diet_enabled()
    native_qk = _use_native_fp8(cfgs, "qk")
    static_qk = native_qk and _static_fp8(cfgs, "qk")
    assert not (native_qk and not static_qk), "skew: dynamic q quantization"
    use_exp2 = decode_exp2(cfgs)
    base_scale = math.log2(math.e) if use_exp2 else 1.0
    folded_scale = q_folded_scale(cfgs) if (diet and not native_qk) else None
    if folded_scale is not None and folded_scale * base_scale != 1.0:
        q = q * jnp.asarray(folded_scale * base_scale, dtype=q.dtype)
    if static_qk:
        fp8_max = float(jnp.finfo(k.dtype).max)
        q = jnp.clip(q, -fp8_max, fp8_max).astype(k.dtype)
    elif cfgs.serve.scale_q is not None:
        q = q / cfgs.serve.scale_q
        if jnp.issubdtype(k.dtype, jnp.floating):
            dtype_info = jnp.finfo(k.dtype)
            q = jnp.clip(q, min=float(dtype_info.min), max=float(dtype_info.max))
        q = q.astype(k.dtype)

    s = k.shape[-1]
    if utils.bench_skip("matmul"):
        qk = jnp.broadcast_to(
            k.reshape(-1, h_size, s)[:, :1, :].astype(jnp.float32),
            (b * k_heads, tq, s),
        )
    elif mxu_acc(cfgs, "qk"):
        raise NotImplementedError("skew with explicit MXU dots")
    else:
        qk = lax.dot(
            q.reshape(-1, tq, h_size),
            k.reshape(-1, h_size, s),
            dimension_numbers=(([2], [1]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    if mask_bias is not None and mask_bias.dtype == jnp.float32:
        qk = qk + mask_bias.reshape(qk.shape)
    qk = qk.reshape(b, k_heads, tq, s)

    if folded_scale is None:
        scale = float(cfgs.model.sm_scale) * base_scale
        if cfgs.serve.scale_q is not None and not native_qk:
            scale *= float(cfgs.serve.scale_q)
        if isinstance(k_scale, (int, float)):
            scale *= float(k_scale)
            k_scale = None
        if scale != 1.0:
            qk *= scale
        if k_scale is not None:
            qk *= k_scale
    qk = qk.astype(cfgs.serve.dtype_out)
    if mask_bias is not None and mask_bias.dtype != jnp.float32:
        qk = qk + mask_bias.astype(qk.dtype)
    assert cfgs.model.soft_cap is None
    if custom_mask is not None:
        qk = jnp.stack(
            [
                jnp.where(custom_mask[b_idx], qk[b_idx], cfgs.model.mask_value)
                for b_idx in range(cfgs.block.batch_size)
            ],
            axis=0,
        )
    if utils.bench_skip("softmax"):
        m_curr = qk[..., :1]
    else:
        m_curr = jnp.max(qk, axis=-1, keepdims=True)
    return qk, m_curr


def flash_attention_softmax(
    step: jax.Array,
    qk: jax.Array,  # [B, KV, TQ, S] in dtype_out
    m_curr: jax.Array,  # [B, KV, TQ, 128] or [B, KV, TQ, 1]
    m_prev: jax.Array,  # [KV, TQ, 128]
    l_prev: jax.Array,  # [KV, TQ, 128]
    is_last_k: SmemWrapper,
    *,
    cfgs: configs.RpaConfigs,
):
    """Online softmax of precomputed scores, for the skewed pipeline.

    The second half of `flash_attention_qk_softmax`: the running max, the
    rescale factors, the weights and the row sum, with the same carries.
    """
    use_exp2 = decode_exp2(cfgs)
    skip_softmax = utils.bench_skip("softmax")
    _exp = (lambda x: x) if skip_softmax else (jnp.exp2 if use_exp2 else jnp.exp)

    alpha_list = []
    m_next_list = []
    for b_idx in range(cfgs.block.batch_size):
        m_next_b = jnp.maximum(m_prev, m_curr[b_idx])
        alpha_list.append(_exp(m_prev - m_next_b))
        m_next_list.append(m_next_b)
        reset_m = is_last_k[step, b_idx] != 0
        m_prev = jnp.where(reset_m, -jnp.inf, m_next_b)

    m_next = jnp.stack(m_next_list, axis=0)
    m_shift = m_next
    if p_prescaled(cfgs):
        m_shift = m_next - jnp.asarray(math.log2(P_SCALE), m_next.dtype)
    p = _exp(qk - utils.broadcast_minor(m_shift, qk.shape))
    p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=cfgs.serve.dtype_out)

    l_next_list = []
    for b_idx in range(cfgs.block.batch_size):
        l_next_b = alpha_list[b_idx] * l_prev + p_rowsum[b_idx]
        l_next_list.append(l_next_b)
        # Same select reset as `flash_attention_qk_softmax` (fuse_accum).
        l_prev = jnp.where(is_last_k[step, b_idx], 0.0, l_next_b)
    l_next = jnp.stack(l_next_list, axis=0)
    return p, alpha_list, m_next, l_next, m_prev, l_prev


def flash_attention_pv(
    p: jax.Array,  # [B, KV, TQ, S]
    v: jax.Array,  # [B, KV, S, H] or [B, KV, H, S]
    alpha_list: list[jax.Array],  # B * [KV, TQ, 128]
    o_prev: jax.Array,  # [KV, TQ, H]
    step: jax.Array,
    is_last_k: SmemWrapper,  # [steps, B]
    cfgs: configs.RpaConfigs,
    v_scale: jax.Array | None = None,  # [B, KV, S]
    mxu_accs: tuple | None = None,
):
    """Flash attention kernel."""
    b, k_heads, tq, s = p.shape

    # Because we sum along the sequence dimension in this matmul, we must do the
    # per-token scaling before the matmul. Therefore, there is separate logic
    # here for per-token and per-tensor scaling, unlike the qk matmul.
    if cfgs.serve.per_token_scale:
        assert v_scale is not None
        v_sc = v_scale[:, :, jnp.newaxis, :]
        # We can scale p or v, and because p will be smaller especially for decode
        # workflows, we scale p to save some time.
        p = p.astype(v_sc.dtype) * v_sc

    p_row_scale = None
    native_pv = _use_native_fp8(cfgs, "pv")
    split_pv = (
        native_pv and cfgs.mode == configs.RpaCase.DECODE
    ) and cfgs.serve.native_fp8_split("pv")
    if p_prescaled(cfgs):
        p = p.astype(v.dtype)
    elif static_pv(cfgs):
        p = (p * jnp.asarray(P_SCALE, p.dtype)).astype(v.dtype)
    elif native_pv:
        p, p_row_scale = _quantize_rows(p, v.dtype, split=split_pv)
    p_rows = 2 * tq if split_pv else tq

    seq_along_lane = cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE
    if seq_along_lane and utils.bench_skip("matmul"):
        # Bench stand-in for PV: a slice of p plus one V element per lane, so
        # the output still depends on the softmax weights and the fetched V.
        h_size = v.shape[-2]
        pv = p.reshape(-1, p_rows, s)[..., :h_size].astype(jnp.float32) + v.reshape(
            -1, h_size, s
        )[:, :1, :1].astype(jnp.float32)
    elif seq_along_lane and mxu_acc(cfgs, "pv"):
        h_size = v.shape[-2]
        pv = _mxu_dot_t(
            p.reshape(-1, p_rows, s),
            v.reshape(-1, h_size, s),
            mxu_accs,
            mxu_acc_ids(cfgs),
            pop_each=False,
        )
    elif seq_along_lane:
        h_size = v.shape[-2]
        pv = lax.dot(
            p.reshape(-1, p_rows, s),
            v.reshape(-1, h_size, s),
            dimension_numbers=(([2], [2]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    else:
        h_size = v.shape[-1]
        pv = lax.dot(
            p.reshape(-1, p_rows, s),
            v.reshape(-1, s, h_size),
            dimension_numbers=(([2], [1]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
    if split_pv:
        pv = _merge_split(pv, tq)
    if p_row_scale is not None:
        pv = pv * p_row_scale.reshape(-1, tq, 1)
    pv = pv.astype(cfgs.serve.dtype_out).reshape(b, k_heads, tq, h_size)

    if not cfgs.serve.per_token_scale and cfgs.serve.scale_v is not None:
        pv *= cfgs.serve.scale_v

    o_next_list = []
    for b_idx in range(cfgs.block.batch_size):
        alpha_b = utils.broadcast_minor(alpha_list[b_idx], o_prev.shape)
        o_next_b = alpha_b * o_prev + pv[b_idx]
        o_next_list.append(o_next_b)
        if cfgs.fuse_accum:
            # Same select reset as l_prev in flash_attention_qk_softmax.
            o_prev = jnp.where(is_last_k[step, b_idx], 0.0, o_next_b)
        else:
            o_prev = o_next_b
    o_next = jnp.stack(o_next_list, axis=0)

    return o_next, o_prev
