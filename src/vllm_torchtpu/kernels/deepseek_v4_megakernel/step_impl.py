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
"""The adapter's operand contract (kernel_interface) on top of `step_kernel`.

`build_step(cfg, bucket)` returns the per-rank function the serving adapter
compiles: positional operands as `kernel_interface` lists them (the weights in
`weight_manifest` order), outputs `hidden [M, H]`, one `[M, H]` per aux layer,
then every cache. Around `step_kernel.build`'s program it does, in XLA:

- the per-role cache indices and block-table offsets of the owner attention
  phase (`step_phases.AttnMeta` / `AttnCaches`) from `cfg.layers`;
- after the call, the stock tail on this rank's own rows: the mHC head
  collapse + final RMSNorm (`layers.mhc_collapse_head`, vLLM RMSNorm) on the
  final residual streams, and the mean over the hc streams of each aux
  layer's settled streams (`DeepseekV4Model.forward`'s aux states).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_core
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_phases as SP
from vllm_torchtpu.kernels.deepseek_v4_megakernel import weight_map as WM

ROLES = SK.ROLES
ATTN_WEIGHTS = SP.AttnWeights._fields  # manifest names match the fields


@dataclasses.dataclass(frozen=True)
class CachePlan:
    """Static per-role addressing of the caches, from `StepConfig.layers`.

    For role `k`: `layers[k]` decoder layers carrying it, `arrays[k]` their
    index into the distinct cache list, `groups[k]` their block-table operand.
    `swa_groups` are the block-table groups of the SWA role, deduplicated."""

    layers: dict[str, tuple[int, ...]]
    arrays: dict[str, tuple[int, ...]]
    groups: dict[str, tuple[int, ...]]
    swa_groups: tuple[int, ...]

    @classmethod
    def of(cls, cfg: KI.StepConfig) -> CachePlan:
        layers, arrays, groups = {}, {}, {}
        for role in ROLES:
            refs = [(i, getattr(lc, role)) for i, lc in enumerate(cfg.layers)]
            refs = [(i, r) for i, r in refs if r is not None]
            layers[role] = tuple(i for i, _ in refs)
            arrays[role] = tuple(r.array for _, r in refs)
            groups[role] = tuple(r.group for _, r in refs)
        swa = tuple(dict.fromkeys(groups["swa"]))
        return cls(layers, arrays, groups, swa)


def attn_metadata(
    cfg: KI.StepConfig,
    plan: CachePlan,
    block_tables: Sequence[jax.Array],
    seq_lens: jax.Array,
    q: int,
) -> dict[str, Any]:
    """`bt_flat`: every group's [B, W_g] table, then one [B, 3] SWA window
    page block per SWA group (`attn_core.swa_window_pages`, zero for inactive
    requests), flattened and concatenated; per role (`SK.META_ROLES`, incl.
    "swa_pages") `bt_off` / `bt_w` int32 [L] with request r's table at
    `bt_flat[off + r * w : off + (r + 1) * w]` (0 / 0 where a layer lacks
    the role)."""
    L = len(cfg.layers)
    B = seq_lens.shape[0]
    parts, off = [], 0
    goff = []
    for t, w in zip(block_tables, cfg.group_widths):
        goff.append(off)
        parts.append(t.reshape(-1))
        off += B * w
    swa_off = {}
    for g in plan.swa_groups:
        p = attn_core.swa_window_pages(
            seq_lens,
            block_tables[g],
            q_len=q,
            window=cfg.model.sliding_window,
            swa_page=cfg.swa_page,
        )
        swa_off[g] = off
        parts.append(jnp.where((seq_lens > 0)[:, None], p, 0).reshape(-1))
        off += B * attn_core.SWA_PAGES
    bt_off, bt_w = {}, {}
    for role in SK.META_ROLES:
        o, w = [0] * L, [0] * L
        if role == "swa_pages":
            for li, g in zip(plan.layers["swa"], plan.groups["swa"]):
                o[li], w[li] = swa_off[g], attn_core.SWA_PAGES
        else:
            for li, g in zip(plan.layers[role], plan.groups[role]):
                o[li], w[li] = goff[g], cfg.group_widths[g]
        bt_off[role] = jnp.asarray(o, jnp.int32)
        bt_w[role] = jnp.asarray(w, jnp.int32)
    return dict(
        bt_flat=jnp.concatenate(parts).astype(jnp.int32), bt_off=bt_off, bt_w=bt_w
    )


def collapse_head(res, hc_fn, hc_scale, hc_base, norm_w, *, hc_mult, rms_eps, hc_eps):
    """`mhc_collapse_head` then the final RMSNorm (vLLM's native formula) on
    [M, hc*H] bf16 residual streams -> [M, H] bf16.

    The stock bf16 rounding points are `reduce_precision`, not dtype round
    trips: XLA may drop an f32->bf16->f32 pair, and whether it does differs
    between compile paths (torch_tpu's and jax.jit gave 1-ulp differences)."""
    m, hch = res.shape
    h = hch // hc_mult
    flat = res.astype(jnp.float32)
    normed = flat * jax.lax.rsqrt(jnp.mean(flat * flat, -1, keepdims=True) + rms_eps)
    # f32 as stock (F.linear on f32); HIGHEST so TPU does not drop to bf16 passes
    mix = jnp.dot(normed, hc_fn.T, precision=jax.lax.Precision.HIGHEST)
    mix = jax.nn.sigmoid(mix * hc_scale + hc_base) + hc_eps
    x = jnp.sum(mix[:, :, None] * flat.reshape(m, hc_mult, h), axis=1)
    x = _bf16(x)  # mhc_collapse_head returns bf16
    x = _bf16(x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + rms_eps))
    return (x * norm_w.astype(jnp.float32)).astype(jnp.bfloat16)


def _bf16(x):
    return jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)


def hc_mean(streams, hc_mult):
    """[M, hc*H] -> [M, H]: the aux state, mean over the hc streams."""
    m, hch = streams.shape
    x = streams.reshape(m, hc_mult, hch // hc_mult).astype(jnp.float32)
    return jnp.mean(x, axis=1).astype(jnp.bfloat16)


def kernel_config(
    cfg: KI.StepConfig, bucket: KI.Bucket, dense: Any = None, **kw
) -> SK.StepKernelConfig:
    """`dense` overrides the DSv4-Pro dense geometry (reduced-width tests)."""
    spec = cfg.model
    dense = dataclasses.replace(dense or WM.dense_config(spec), rows=bucket.rows)
    kw.setdefault("strict_causal", envs.TPU_STREAMIDX_STRICT_CAUSAL)
    kw.setdefault("vmem_share", envs.VLLM_TPU_DSV4_MK_VMEM_SHARE)
    return SK.StepKernelConfig(
        dense=dense,
        q=bucket.q,
        B=bucket.reqs,
        # aux_layers are 1-based "after layer" ids; the kernel takes 0-based
        aux_layers=tuple(a - 1 for a in spec.aux_layers),
        attn_tp=spec.attn_tp,
        moe_placement=spec.moe_placement,
        **kw,
    )


def build_step(
    cfg: KI.StepConfig,
    bucket: KI.Bucket,
    *,
    attn_impl: str = "real",
    moe_impl: str = "real",
    interpret: Any = False,
    dense: Any = None,
    cache_inplace: bool = False,
    max_ctx: int | None = None,
    logits: bool = False,
) -> Callable[..., tuple[jax.Array, ...]]:
    """`cache_inplace`: the kernel writes the caches it is handed and the fn
    returns no caches (`StepKernelConfig.cache_inplace`); else they come back
    updated, after the aux outputs. `max_ctx`: the longest request context
    the step serves (None: the kernel default). `logits`: the first output is
    the own rows' logits [M, V] bf16 instead of the hidden, from this rank's
    lm_head shard (weight "lm_head", vocab_head.shard_lm_head) on every row
    and an all-gather of the shards (vocab_head docstring for exactness)."""
    spec = cfg.model
    if spec.mode != "dp":
        raise ValueError(f"the L4e step kernel serves DP mode only, got {spec.mode!r}")
    ctx = {} if max_ctx is None else {"max_ctx": max_ctx}
    scfg = kernel_config(cfg, bucket, dense, attn_impl=attn_impl, moe_impl=moe_impl,
                         cache_inplace=cache_inplace, **ctx)  # fmt: skip
    step = SK.build(scfg, axis=spec.axis_name, interpret=interpret)
    names = [e.name for e in KI.weight_manifest(spec, logits=logits)]
    streams = scfg.dense.stream_names
    plan = CachePlan.of(cfg)
    G, W, L = len(cfg.group_widths), len(names), spec.num_layers
    n_aux = len(spec.aux_layers)

    def fn(meta, input_ids, positions, seq_lens, *rest):
        bts = rest[:G]
        w = dict(zip(names, rest[G : G + W]))
        caches = list(rest[G + W :])
        moe_w = dict(
            tid2eid=[w[f"tid2eid_{i}"] for i in range(spec.num_hash_layers)],
            bias=w["moe_bias"],
            **{
                k: [w[f"{k}_{li}"] for li in range(L)]
                for k in ("w13", "w13_scale", "w2", "w2_scale")
            },
            # static expert placement: this rank's placement_tables rows
            **{k: w["moe_" + k] for k in ("held", "copy") if "moe_" + k in w},
        )
        kw = {}
        if attn_impl != "stub":
            kw["attn"] = dict(
                arrays=caches,
                cache_index={k: list(v) for k, v in plan.arrays.items()},
                cache_layers={k: list(v) for k, v in plan.layers.items()},
                weights={k: w[k] for k in ATTN_WEIGHTS},
                **attn_metadata(cfg, plan, bts, seq_lens, bucket.q),
            )
        out = step(
            input_ids,
            positions,
            seq_lens,
            w["embed"],
            w["cos_sin"],
            w["dense_smalls"],
            {n: w["dense_" + n] for n in streams},
            moe_w if moe_impl == "real" else None,
            # the rank as an operand: lax.axis_index in the XLA part lowers to
            # partition-id, which the torch_tpu SPMD export rejects
            me=meta[KI.META_RANK],
            eblk=meta[KI.META_EBLK],
            with_all=logits,
            vocab_start=vocab_start(spec, meta),
            **kw,
        )
        res, aux = out[0], out[1]
        if attn_impl != "stub" and not cache_inplace:
            caches = list(out[5])

        def head(r):
            return collapse_head(
                r,
                w["hc_head_fn"],
                w["hc_head_scale"],
                w["hc_head_base"],
                w["final_norm"],
                hc_mult=4,
                rms_eps=spec.rms_eps,
                hc_eps=spec.hc_eps,
            )

        if logits:
            hidden = _own_logits(head(out[-1]), w["lm_head"], out[3], meta, spec)
        else:
            hidden = head(res)
        aux_h = [hc_mean(aux[a], 4) for a in range(n_aux)]
        return (hidden, *aux_h) if cache_inplace else (hidden, *aux_h, *caches)

    return fn


def vocab_start(spec: KI.ModelSpec, meta):
    """The first id of this rank's vocab-parallel embed shard, or None when
    the embed is whole (attn_tp 1)."""
    return meta[KI.META_VSTART] if spec.attn_tp > 1 else None


class StepKernel:
    """`build_step` side of `KernelImpl` (weights come from the adapter)."""

    def __init__(self, **kw):
        self.kw = kw

    def build_step(self, cfg: KI.StepConfig, bucket: KI.Bucket):
        return build_step(cfg, bucket, **self.kw)


# ------------------------------------------------------- DSpark draft step

DRAFT_GLOBALS = (
    "embed",  # the target's (aliased)
    "cos_sin",  # the draft layers' own rotary table, pack_cos_sin layout
    # this rank's output columns [r * H/R, (r+1) * H/R) of the fp8 main_proj as
    # bf16 (exact; an fp8 weight is relaid out by XLA on every small-M dot:
    # 147 MiB per step replicated) ...
    "main_proj",  # bf16 [n_aux * H, H / R]
    "main_proj_scale",  # ... and its f32 [H / R] per-channel scale
    "main_norm",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
)


def draft_weight_names(
    spec: KI.ModelSpec, dense: Any = None, logits: bool = False
) -> list[str]:
    """Operand names of the draft step's weights, in order: DRAFT_GLOBALS (+
    the draft's final norm and this rank's lm_head shard with `logits`), the
    dense ring of the draft layers, MoE per layer, then the SWA attention
    phase's weights (SP.AttnWeights fields)."""
    dense = dense or WM.dense_config(spec)
    names = list(DRAFT_GLOBALS) + (["norm", "lm_head"] if logits else [])
    names += ["dense_" + n for n in dense.stream_names] + ["dense_smalls"]
    names.append("moe_bias")
    for li in range(spec.num_layers):
        names += [f"{k}_{li}" for k in ("w13", "w13_scale", "w2", "w2_scale")]
    names += list(ATTN_WEIGHTS)
    return names


def build_draft_step(
    cfg: KI.StepConfig,
    bucket: KI.Bucket,
    ctx_q: int,
    *,
    attn_impl: str = "real",
    moe_impl: str = "real",
    interpret: Any = False,
    dense: Any = None,
    cache_inplace: bool = True,
    max_ctx: int | None = None,
    logits: bool = False,
) -> Callable[..., tuple[jax.Array, ...]]:
    """The DSpark draft forward as one kernel step (B9 Stage 1). `logits`:
    return the draft rows' logits (compute_draft_logits' lm_head(norm(.)))
    from this rank's lm_head shard instead of the hidden.

    `cfg.model` is the draft's ModelSpec (its SWA layers, q = the K draft
    rows per request, num_hash_layers 0, no aux layers); `ctx_q` is the
    target's verify rows per request. fn(meta, input_ids [M], positions [M],
    seq_lens [B], ctx_aux [Mc, n_aux*H] bf16, ctx_positions [Mc] int32,
    ctx_seq_lens [B], *block_tables, *weights (draft_weight_names), *caches)
    -> hidden [M, H] bf16: the draft rows' mhc_collapse_head output before the
    final norm (stock compute_draft_logits applies norm + lm_head).

    The context rows are this rank's verify rows (Mc = B * ctx_q, request r
    at rows r*ctx_q ..): main_x = main_norm(main_proj(ctx_aux)) as the stock
    combine_hidden_states, then gathered into the target step's global row
    layout, where every SWA layer projects their kv for its attention."""
    from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

    spec = cfg.model
    assert spec.mode == "dp" and spec.num_hash_layers == 0 and not spec.aux_layers
    ctx = {} if max_ctx is None else {"max_ctx": max_ctx}
    scfg = kernel_config(cfg, bucket, dense, attn_impl=attn_impl, moe_impl=moe_impl,
                         cache_inplace=cache_inplace, ctx_q=ctx_q,
                         num_hash_layers=0, **ctx)  # fmt: skip
    step = SK.build(scfg, axis=spec.axis_name, interpret=interpret)
    names = draft_weight_names(spec, scfg.dense, logits)
    streams = scfg.dense.stream_names
    plan = CachePlan.of(cfg)
    G, W, L = len(cfg.group_widths), len(names), spec.num_layers
    # the target step's row layout for the context rows (its q = ctx_q)
    ctx_layout = dataclasses.replace(scfg, q=ctx_q, ctx_q=0)

    def fn(meta, input_ids, positions, seq_lens, ctx_aux, ctx_positions,
           ctx_seq_lens, *rest):  # fmt: skip
        bts = rest[:G]
        w = dict(zip(names, rest[G : G + W]))
        caches = list(rest[G + W :])
        me = meta[KI.META_RANK]
        # xla_quantized_matmul's arithmetic on this rank's column slice (the
        # fp8 x fp8 products are exact in bf16), then the slices gathered
        x_q, x_s = quantize_tensor(ctx_aux, jnp.float8_e4m3fn)
        part = jax.lax.dot_general(
            x_q.astype(jnp.bfloat16), w["main_proj"], (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )  # fmt: skip
        part = (part * x_s * w["main_proj_scale"][None, :]).astype(ctx_aux.dtype)
        main_x = jax.lax.all_gather(part, spec.axis_name, axis=1, tiled=True)
        main_x = _rms_norm(main_x, w["main_norm"], spec.rms_eps)
        tokm, *_ = SK.step_metadata(
            ctx_layout, me, jnp.zeros_like(ctx_positions), ctx_positions,
            ctx_seq_lens, spec.axis_name,
        )  # fmt: skip
        T = scfg.T
        owner, lrow = tokm[2 * T : 3 * T], tokm[3 * T : 4 * T]
        allx = jax.lax.all_gather(main_x, spec.axis_name)  # [R, Mc, H]
        live = owner >= 0
        gx = allx[jnp.maximum(owner, 0), jnp.maximum(lrow, 0)]
        ctx = dict(
            main_x=jnp.where(live[:, None], gx, 0).astype(jnp.bfloat16),
            positions=jnp.where(live, tokm[T : 2 * T], -1),
        )
        moe_w = dict(
            tid2eid=[],
            bias=w["moe_bias"],
            **{
                k: [w[f"{k}_{li}"] for li in range(L)]
                for k in ("w13", "w13_scale", "w2", "w2_scale")
            },
        )
        kw = {}
        if attn_impl != "stub":
            kw["attn"] = dict(
                arrays=caches,
                cache_index={k: list(v) for k, v in plan.arrays.items()},
                cache_layers={k: list(v) for k, v in plan.layers.items()},
                weights={k: w[k] for k in ATTN_WEIGHTS},
                **attn_metadata(cfg, plan, bts, seq_lens, bucket.q),
            )
        out = step(
            input_ids, positions, seq_lens, w["embed"], w["cos_sin"],
            w["dense_smalls"], {n: w["dense_" + n] for n in streams},
            moe_w if moe_impl == "real" else None,
            me=me, eblk=meta[KI.META_EBLK], ctx=ctx, with_all=logits, **kw,
        )  # fmt: skip

        def head(r):
            return collapse_head_prenorm(
                r, w["hc_head_fn"], w["hc_head_scale"], w["hc_head_base"],
                hc_mult=4, rms_eps=spec.rms_eps, hc_eps=spec.hc_eps,
            )  # fmt: skip

        if not logits:
            return (head(out[0]),)
        x_all = _rms_norm(head(out[-1]), w["norm"], spec.rms_eps)
        return (_own_logits(x_all, w["lm_head"], out[3], meta, spec),)

    return fn


def _own_logits(x_all, lm_head_shard, ownt, meta, spec):
    """Rows x_all [T, H] (every global row, the same on every rank) -> this
    rank's own rows' logits [M, V] bf16: the shard's columns on every row,
    all-gathered, own rows picked (zero rows where none)."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import vocab_head as VH

    V = spec.vocab_size
    lg = VH.shard_logits(x_all, lm_head_shard, meta[KI.META_RANK], V)
    full = jax.lax.all_gather(lg, spec.axis_name, axis=1, tiled=True)[:, :V]
    own = ownt >= 0
    return jnp.where(own[:, None], full[jnp.maximum(ownt, 0)], 0).astype(jnp.bfloat16)


def _rms_norm(x, w, eps):
    """vLLM RMSNorm (f32 statistics, weight multiply in the weight dtype)."""
    xf = x.astype(jnp.float32)
    xf = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf.astype(w.dtype) * w).astype(x.dtype)


def collapse_head_prenorm(res, hc_fn, hc_scale, hc_base, *, hc_mult, rms_eps, hc_eps):
    """`mhc_collapse_head` alone (the draft's head: stock applies the final
    norm inside compute_draft_logits)."""
    m, hch = res.shape
    h = hch // hc_mult
    flat = res.astype(jnp.float32)
    normed = flat * jax.lax.rsqrt(jnp.mean(flat * flat, -1, keepdims=True) + rms_eps)
    mix = jnp.dot(normed, hc_fn.T, precision=jax.lax.Precision.HIGHEST)
    mix = jax.nn.sigmoid(mix * hc_scale + hc_base) + hc_eps
    x = jnp.sum(mix[:, :, None] * flat.reshape(m, hc_mult, h), axis=1)
    return x.astype(jnp.bfloat16)
