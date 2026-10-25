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
"""Served DeepSeek-V4 tensors -> the megakernel's per-rank weight operands.

Three steps, kept apart so each can be checked on its own:

1. `served_layer_spec` / `served_global_spec` name every tensor the stock
   path serves -- attribute path, shape and dtype AFTER the load-time
   transforms. The torch side (models/vllm/deepseek_v4/megakernel_weights.py)
   reads and validates the module tree against it; nothing is copied.
2. `NATURAL` rules (pure JAX) slice this rank's share and stack layers into
   the natural arrays `dense_weights.pack_stream` / `pack_smalls` consume.
   Slicing happens here, inside JAX, never as a torch view: torch_tpu
   re-materializes a view operand on every call.
3. `plan_operands` walks `kernel_interface.weight_manifest`: per operand,
   either one served tensor as is (zero-copy) or a JAX fn of named served
   tensors (a load-time copy), run one operand at a time so the transient
   peak is one stream, not the whole ring.

Rank r's shares (L4e, R ranks, DSv4-Pro R=16): wq_b heads [8r, 8r+8) =
columns [4096r, 4096r+4096); wo_a o-group r = columns [1024r, 1024r+1024);
wo_b rows of group r; shared expert intermediate [192r, 192r+192) (w13 gate
and up columns, w2 rows); per-channel scales sliced alike (wo_b / w2 scales
are per output column, unsliced). Everything else is replicated.

Under vLLM TP (TP8 x DP2, `ModelSpec.attn_tp` = 8) the served tensors are
already TP shard j of the mesh rank g * attn_tp + j: the same rules run on the
shard's geometry (`tp_local`) with the engine index as the rank
(`head_shard`), so rank (g, j) takes half g of shard j -- head block
j * G + g (step_kernel.head_block).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4_megakernel import (
    attn_phase,
    moe_experts,
    vocab_head,
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_weights as dw
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI

FP8, BF16, F32, I32 = "float8_e4m3fn", "bfloat16", "float32", "int32"
FP4X2 = "float4_e2m1fn_x2"  # torch's two-fp4-per-byte dtype


class Unsupported(ValueError):
    pass


def _need(cond: bool, why: str) -> None:
    if not cond:
        raise Unsupported(f"DeepSeek-V4 megakernel weights: {why}")


@dataclasses.dataclass(frozen=True)
class Geometry:
    """The served shapes, from the HF config."""

    hidden: int
    q_lora: int
    head_dim: int
    n_heads: int
    o_groups: int
    o_lora: int
    moe_inter: int
    n_experts: int
    idx_heads: int
    idx_head_dim: int
    hc_mult: int
    vocab: int
    ratios: tuple[int, ...]
    num_hash_layers: int
    topk: int = 6
    rope_dim: int = 64
    fp4_block: int = 512
    shared_tp: int = 1  # vLLM TP shards of the shared expert (`tp_local`)

    @property
    def shared_inter(self) -> int:  # the served shared expert's intermediate
        return self.moe_inter // self.shared_tp

    @classmethod
    def from_hf(cls, hf: Any, ratios: Sequence[int] | None = None) -> Geometry:
        n_layers = hf.num_hidden_layers
        return cls(
            hidden=hf.hidden_size,
            q_lora=hf.q_lora_rank,
            head_dim=hf.head_dim,
            n_heads=hf.num_attention_heads,
            o_groups=hf.o_groups,
            o_lora=hf.o_lora_rank,
            moe_inter=hf.moe_intermediate_size,
            n_experts=hf.n_routed_experts,
            idx_heads=hf.index_n_heads,
            idx_head_dim=hf.index_head_dim,
            hc_mult=hf.hc_mult,
            vocab=hf.vocab_size,
            ratios=tuple(ratios)
            if ratios is not None
            else tuple(max(1, r) for r in hf.compress_ratios[:n_layers]),
            num_hash_layers=hf.num_hash_layers,
            topk=hf.num_experts_per_tok,
            rope_dim=hf.qk_rope_head_dim,
        )

    @property
    def n_mix(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    def coff(self, ratio: int) -> int:
        return 2 if ratio == 4 else 1


def tp_local(g: Geometry, tp: int) -> Geometry:
    """The geometry of one vLLM TP shard: heads, o-groups and the shared
    expert's intermediate split `tp` ways (ColumnParallel wq_b / wo_a /
    gate_up, RowParallel wo_b / down; attn_sink per local head). Routed
    experts are expert-parallel, whole."""
    _need(
        g.n_heads % tp == 0 and g.o_groups % tp == 0 and g.shared_inter % tp == 0,
        f"TP {tp} does not divide {g.n_heads} heads / {g.o_groups} groups / "
        f"{g.shared_inter} shared intermediate",
    )
    return dataclasses.replace(
        g, n_heads=g.n_heads // tp, o_groups=g.o_groups // tp,
        shared_tp=g.shared_tp * tp,
    )  # fmt: skip


# ------------------------------------------------------------------ 1. served


@dataclasses.dataclass(frozen=True)
class Served:
    """One served tensor: attribute path from its owner (decoder layer, or
    the causal LM for globals), shape and dtype after the load-time
    transforms. `nk_ok`: an unquantized linear that may still be in vLLM's
    [n, k] (it never met a TPU linear method); it is transposed in JAX."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    nk_ok: bool = False


def served_layer_spec(g: Geometry, index: int, local_experts: int) -> dict[str, Served]:
    """What one `DeepseekV4DecoderLayer` serves, by megakernel key.

    fp8 linears: `VllmFp8LinearMethodTPU` requantizes to e4m3 with a per
    output channel f32 scale and stores [k, n]; `wo_a` is then [heads/group *
    head_dim, groups * o_lora] (`VllmDeepseekV4MLAAttention.process_weights_
    after_loading`). Compressors: `fused_wkv_wgate` transposed once to
    [hidden, 2 * coff * head_dim] by `DeepseekV4ForCausalLM.load_weights`; a
    load path that skips it (dummy weights) leaves vLLM's [n, k], hence
    `nk_ok`. The gate and the indexer `weights_proj` are unquantized linears
    that keep vLLM's [n, k] on TPU (measured on device). Routed experts: the
    W4A8 fused EP v2 layout -- fp4 pairs packed along the last axis, f32
    block-512 scale tables staged by `register_fused_scale_buffers`."""
    ratio = g.ratios[index]
    h, D, ql = g.hidden, g.head_dim, g.q_lora
    hpg = g.n_heads // g.o_groups
    out: dict[str, Served] = {}

    def fp8(key, path, k, n):
        out[key] = Served(f"{path}.weight", (k, n), FP8)
        out[key + "_scale"] = Served(f"{path}.weight_scale", (n,), F32)

    fp8("wqa", "attn.fused_wqa_wkv", h, ql + D)
    fp8("wq_b", "attn.wq_b", ql, g.n_heads * D)
    fp8("wo_a", "attn.wo_a", hpg * D, g.o_groups * g.o_lora)
    fp8("wo_b", "attn.wo_b", g.o_groups * g.o_lora, h)
    for key, path, n in (
        ("q_norm", "attn.q_norm.weight", ql),
        ("kv_norm", "attn.kv_norm.weight", D),
        ("attn_norm", "attn_norm.weight", h),
        ("ffn_norm", "ffn_norm.weight", h),
    ):
        out[key] = Served(path, (n,), BF16)
    out["attn_sink"] = Served("attn.attn_sink", (g.n_heads,), F32)
    for sub in ("attn", "ffn"):
        out[f"hc_{sub}_fn"] = Served(f"hc_{sub}_fn", (g.n_mix, g.hc_mult * h), F32)
        out[f"hc_{sub}_base"] = Served(f"hc_{sub}_base", (g.n_mix,), F32)
        out[f"hc_{sub}_scale"] = Served(f"hc_{sub}_scale", (3,), F32)
    if ratio > 1:
        co = g.coff(ratio)
        out["comp"] = Served(
            "attn.compressor.fused_wkv_wgate.weight", (h, 2 * co * D), BF16, True
        )
        out["comp_ape"] = Served("attn.compressor.ape", (ratio, co * D), F32)
        out["comp_norm"] = Served("attn.compressor.norm.weight", (D,), BF16)
    if ratio == 4:
        IH, ID = g.idx_heads, g.idx_head_dim
        fp8("idx_wq_b", "attn.indexer.wq_b", ql, IH * ID)
        out["wp"] = Served("attn.indexer.weights_proj.weight", (h, IH), BF16, True)
        out["idx_comp"] = Served(
            "attn.indexer.compressor.fused_wkv_wgate.weight", (h, 4 * ID), BF16, True
        )
        out["idx_comp_ape"] = Served("attn.indexer.compressor.ape", (4, 2 * ID), F32)
        out["idx_comp_norm"] = Served(
            "attn.indexer.compressor.norm.weight", (ID,), BF16
        )
    out["gate"] = Served("ffn.gate.weight", (h, g.n_experts), BF16, True)
    if index < g.num_hash_layers:
        out["tid2eid"] = Served("ffn.gate.tid2eid", (g.vocab, g.topk), I32)
    else:
        out["gate_bias"] = Served(
            "ffn.gate.e_score_correction_bias", (g.n_experts,), F32
        )
    fp8("sh_w13", "ffn.shared_experts.gate_up_proj", h, 2 * g.shared_inter)
    fp8("sh_w2", "ffn.shared_experts.down_proj", g.shared_inter, h)
    E, I, qb = local_experts, g.moe_inter, g.fp4_block  # noqa: E741
    re = "ffn.experts.routed_experts"
    out["w13"] = Served(f"{re}.w13_weight", (E, h, I), FP4X2)  # 2I fp4 values
    out["w2"] = Served(f"{re}.w2_weight", (E, I, h // 2), FP4X2)
    out["w13_scale"] = Served(f"{re}._tpu_fused_w13_scale", (E, h // qb, 2 * I), F32)
    out["w2_scale"] = Served(f"{re}._tpu_fused_w2_scale", (E, I // qb, h), F32)
    return out


def served_global_spec(g: Geometry, vocab_parallel: bool = False) -> dict[str, Served]:
    """Model-level tensors, paths from the causal LM. The rope tables are
    row-major on device after `relayout_rope_caches` (TPU_ROPE_CACHE_ROW_MAJOR,
    on in the served config); their length is the served one (-1).
    `vocab_parallel`: embed and lm_head are vLLM TP vocab shards (-1 rows)."""
    h = g.hidden
    vocab = -1 if vocab_parallel else g.vocab
    csa = g.ratios.index(4) if 4 in g.ratios else None
    hca = next((i for i, r in enumerate(g.ratios) if r > 4), None)
    out = {
        "embed": Served("model.embed_tokens.weight", (vocab, h), BF16),
        "final_norm": Served("model.norm.weight", (h,), BF16),
        "hc_head_fn": Served("model.hc_head_fn", (g.hc_mult, g.hc_mult * h), F32),
        "hc_head_base": Served("model.hc_head_base", (g.hc_mult,), F32),
        "hc_head_scale": Served("model.hc_head_scale", (1,), F32),
        "lm_head": Served("lm_head.weight", (vocab, h), BF16),
        "cos_sin": Served(
            "model.layers.0.attn.rotary_emb.cos_sin_cache", (-1, g.rope_dim), F32
        ),
    }
    if hca is not None:
        out["comp_cos_sin"] = Served(
            f"model.layers.{hca}.attn.rotary_emb.cos_sin_cache", (-1, g.rope_dim), F32
        )
    if csa is not None:
        out["idx_cos_sin"] = Served(
            f"model.layers.{csa}.attn.indexer_rotary_emb.cos_sin_cache",
            (-1, g.rope_dim),
            F32,
        )
    return out


def draft_geometry(g: Geometry, n_draft: int) -> Geometry:
    """The draft's own layer geometry: n_draft SWA layers (ratio 1), none
    hash-routed (they sit past the target's hash layers)."""
    return dataclasses.replace(g, ratios=(1,) * n_draft, num_hash_layers=0)


def served_draft_spec(
    g: Geometry,
    n_draft: int,
    n_aux: int,
    local_experts: int | Sequence[int],
    vocab_parallel: bool = False,
) -> tuple[dict[str, Served], list[dict[str, Served]]]:
    """What the DSpark draft serves: model-level tensors (paths from its
    `DSparkDeepseekV4Model`) and one `served_layer_spec` per draft layer.

    Draft layer i is the decoder layer at global index L + i (past every
    hash layer: noaux gate) and SWA-only, so it carries no compressor,
    indexer or compressed KV. The rotary table is the draft layers' own (it
    may differ from the target's, whose layers all compress). main_proj
    takes the concatenated aux states of the `n_aux` target layers."""
    h, gd = g.hidden, draft_geometry(g, n_draft)
    if isinstance(local_experts, int):  # per layer (a placement may differ)
        local_experts = [local_experts] * n_draft
    layers = [served_layer_spec(gd, i, e) for i, e in enumerate(local_experts)]
    glob = {
        "embed": Served(
            "embed_tokens.weight", (-1 if vocab_parallel else g.vocab, h), BF16
        ),
        "main_proj": Served("main_proj.weight", (n_aux * h, h), FP8),
        "main_proj_scale": Served("main_proj.weight_scale", (h,), F32),
        "main_norm": Served("main_norm.weight", (h,), BF16),
        "norm": Served("norm.weight", (h,), BF16),
        "hc_head_fn": Served("hc_head_fn", (g.hc_mult, g.hc_mult * h), F32),
        "hc_head_base": Served("hc_head_base", (g.hc_mult,), F32),
        "hc_head_scale": Served("hc_head_scale", (1,), F32),
        "cos_sin": Served(
            "layers.0.attn.rotary_emb.cos_sin_cache", (-1, g.rope_dim), F32
        ),
    }
    return glob, layers


# --------------------------------------------------------- 2. natural (JAX)


@dataclasses.dataclass(frozen=True)
class Shard:
    rank: int
    ranks: int
    # the mesh index, when `rank` is a TP shard's half (head_shard); the
    # "proj_cols" columns follow the mesh order
    mesh_rank: int | None = None

    @property
    def col_rank(self) -> int:
        return self.rank if self.mesh_rank is None else self.mesh_rank


def head_shard(rank: int, ranks: int, attn_tp: int = 1) -> Shard:
    """Mesh rank `rank`'s share of its served (TP-local, `tp_local`)
    tensors: half g = rank // attn_tp of them, over G = ranks // attn_tp."""
    _need(ranks % attn_tp == 0, f"attn_tp {attn_tp} does not divide {ranks} ranks")
    if attn_tp == 1:
        return Shard(rank, ranks)
    return Shard(rank // attn_tp, ranks // attn_tp, mesh_rank=rank)


def _cols(x, start, size):
    return jax.lax.slice_in_dim(x, start, start + size, axis=-1)


def _rows(x, start, size):
    return jax.lax.slice_in_dim(x, start, start + size, axis=-2)


@dataclasses.dataclass(frozen=True)
class Rule:
    """How one natural array (a `dense_weights.make_weights` key, per layer)
    comes from a served tensor: `fn(geometry, shard, served)`, on the layers
    whose ratio is in `ratios` (None: every layer)."""

    served: str
    fn: Callable[[Geometry, Shard, jax.Array], jax.Array]
    ratios: tuple[int, ...] | None = None


def _same(g, s, x):
    return x


def _q_cols(g, s, x):
    n = g.n_heads // s.ranks * g.head_dim
    return _cols(x, s.rank * n, n)


def _group_cols(g, s, x):
    n = g.o_groups // s.ranks * g.o_lora
    return _cols(x, s.rank * n, n)


def _group_rows(g, s, x):
    n = g.o_groups // s.ranks * g.o_lora
    return _rows(x, s.rank * n, n)


def _shared_gate_up(g, s, x):
    S, I = g.shared_inter // s.ranks, g.shared_inter  # noqa: E741
    return jnp.concatenate(
        [_cols(x, s.rank * S, S), _cols(x, I + s.rank * S, S)], axis=-1
    )


def _shared_rows(g, s, x):
    S = g.shared_inter // s.ranks
    return _rows(x, s.rank * S, S)


NATURAL: dict[str, Rule] = {
    "wqa": Rule("wqa", _same),
    "wqa_scale": Rule("wqa_scale", _same),
    "wq_b": Rule("wq_b", _q_cols),
    "wq_b_scale": Rule("wq_b_scale", _q_cols),
    "wo_a": Rule("wo_a", _group_cols),
    "wo_a_scale": Rule("wo_a_scale", _group_cols),
    "wo_b": Rule("wo_b", _group_rows),
    "wo_b_scale": Rule("wo_b_scale", _same),  # per output column: unsliced
    "w13": Rule("sh_w13", _shared_gate_up),
    "w13_scale": Rule("sh_w13_scale", _shared_gate_up),
    "w2": Rule("sh_w2", _shared_rows),
    "w2_scale": Rule("sh_w2_scale", _same),
    "gate": Rule("gate", _same),
    "idx_wq_b": Rule("idx_wq_b", _same, (4,)),
    "idx_wq_b_scale": Rule("idx_wq_b_scale", _same, (4,)),
    "wp": Rule("wp", _same, (4,)),
    "idx_comp": Rule("idx_comp", _same, (4,)),
    "comp_csa": Rule("comp", _same, (4,)),
    "comp_hca": Rule("comp", _same, (128,)),
    **{
        k: Rule(k, _same)
        for k in ("attn_norm", "ffn_norm", "q_norm", "kv_norm")
        + tuple(f"hc_{a}_{f}" for a in ("attn", "ffn") for f in ("fn", "base", "scale"))
    },
}


def check_shardable(g: Geometry, s: Shard) -> None:
    _need(0 <= s.rank < s.ranks, f"rank {s.rank} of {s.ranks}")
    _need(
        g.n_heads % s.ranks == 0 and g.o_groups % s.ranks == 0,
        f"{s.ranks} ranks do not divide {g.n_heads} heads / {g.o_groups} groups",
    )
    _need(
        g.shared_inter % s.ranks == 0,
        f"{s.ranks} ranks do not divide the shared expert's {g.shared_inter}",
    )


def pack_inputs(op: str) -> tuple[str, ...]:
    """Natural keys a packed manifest operand reads (`smalls` or a stream)."""
    if op == "smalls":
        return tuple(
            k for k in dc.DenseChainConfig().smalls_layout if not k.startswith("_")
        )
    return {
        "hc_attn": ("hc_attn_fn",),
        "hc_ffn": ("hc_ffn_fn",),
        "xproj_csa": ("wp", "idx_comp", "comp_csa"),
        "xproj_hca": ("comp_hca",),
        "wqa_ctx": ("wqa",),  # the kv columns of the SWA layers' wqa
    }.get(op, (op,))


# Manifest "stack" operands: served key, layers, and how a missing layer fills.
STACK: dict[str, tuple[str, tuple[int, ...] | None]] = {
    "attn_sink": ("attn_sink", None),
    "comp_ape": ("comp_ape", None),
    "comp_norm": ("comp_norm", None),
    "idx_comp_ape": ("idx_comp_ape", (4,)),
    "idx_comp_norm": ("idx_comp_norm", (4,)),
    "moe_bias": ("gate_bias", None),
}


@dataclasses.dataclass(frozen=True)
class LayerSource:
    """Which served tensor of which layer (None: a global) feeds a build."""

    layer: int | None
    key: str
    transpose: bool


def plan_inputs(
    keys: Sequence[str], served: Sequence[dict[str, bool]], g: Geometry, *, stack=False
) -> list[LayerSource]:
    """Served tensors a build reads: every (layer, served key) its natural
    (or, with `stack`, raw) keys come from, in layer order."""
    out = []
    for li, w in enumerate(served):
        seen = set()
        for k in keys:
            src, ratios = STACK[k] if stack else (NATURAL[k].served, NATURAL[k].ratios)
            if ratios is not None and g.ratios[li] not in ratios:
                continue
            if src not in w or src in seen:
                continue
            seen.add(src)
            out.append(LayerSource(li, src, w[src]))
    return out


def gather_natural(
    keys: Sequence[str],
    sources: Sequence[LayerSource],
    arrays: Sequence[jax.Array],
    g: Geometry,
    s: Shard,
) -> dict[str, jax.Array]:
    """The natural arrays `keys`, stacked over the layers that carry them."""
    by_layer: dict[int, dict[str, jax.Array]] = {}
    for src, a in zip(sources, arrays):
        by_layer.setdefault(src.layer, {})[src.key] = a.T if src.transpose else a
    out = {}
    for k in keys:
        rule = NATURAL[k]
        per = [
            rule.fn(g, s, w[rule.served])
            for li, w in sorted(by_layer.items())
            if rule.served in w and (rule.ratios is None or g.ratios[li] in rule.ratios)
        ]
        if per:  # absent: no layer of the chain carries it (CSA-only on a draft)
            out[k] = jnp.stack(per)
    return out


def packed_operand(
    op: str,
    sources: Sequence[LayerSource],
    arrays: Sequence[jax.Array],
    g: Geometry,
    s: Shard,
    cfg: dc.DenseChainConfig,
) -> jax.Array:
    nat = gather_natural(pack_inputs(op), sources, arrays, g, s)
    if op == "smalls":
        for name, (_, n) in cfg.smalls_layout.items():  # a chain without CSA layers
            if name != "_total" and name in pack_inputs(op):
                nat.setdefault(name, jnp.zeros((0, n), jnp.float32))
        return dw.pack_smalls(nat, cfg)
    return dw.pack_stream(op, nat, cfg)


def stacked_operand(
    name: str,
    sources: Sequence[LayerSource],
    arrays: Sequence[jax.Array],
    g: Geometry,
    width: int = -1,
) -> jax.Array:
    """A per-layer stack (f32). `comp_ape` pads HCA's [128, 512] and CSA's
    [4, 1024] to a common [max rows, max cols]; `moe_bias` is 0 on the hash
    layers; `attn_sink` of a TP shard (its heads) is padded to `width` with
    -inf (no sink), as vLLM pads it."""
    vals = [a.astype(jnp.float32) for a in arrays]
    if name == "attn_sink" and width > 0:
        _need(all(v.shape[0] <= width for v in vals), f"attn_sink over {width}")
        vals = [
            jnp.pad(v, (0, width - v.shape[0]), constant_values=-jnp.inf) for v in vals
        ]
    if name == "moe_bias":
        have = {src.layer: v for src, v in zip(sources, vals)}
        zero = jnp.zeros_like(vals[0])
        return jnp.stack([have.get(li, zero) for li in range(len(g.ratios))])
    if name == "comp_ape":
        rows = max(v.shape[0] for v in vals)
        cols = max(v.shape[1] for v in vals)
        vals = [
            jnp.pad(v, ((0, rows - v.shape[0]), (0, cols - v.shape[1]))) for v in vals
        ]
    return jnp.stack(vals)


# ------------------------------------------------------- 3. manifest operands


@dataclasses.dataclass(frozen=True)
class Build:
    """How to produce one manifest operand: `fn` None means the single source
    tensor itself (zero-copy); else a JAX fn of the sources' values, in order
    (a load-time copy). A source with layer None is a global."""

    entry: KI.WeightEntry
    sources: tuple[LayerSource, ...]
    fn: Callable[..., jax.Array] | None


_SERVED_LAYER_KEY = {"tid2eid": "tid2eid", "w13": "w13", "w2": "w2"}


def plan_operands(
    spec: KI.ModelSpec,
    served: Sequence[dict[str, bool]],
    g: Geometry,
    s: Shard,
    dense: dc.DenseChainConfig | None = None,
    manifest: Callable[[KI.ModelSpec, dc.DenseChainConfig], list] | None = None,
) -> list[Build]:
    """Every operand of `manifest(spec)` (default `KI.weight_manifest`; the
    draft's is `KI.draft_weight_manifest`) for rank `s`, in order.
    `served[layer]` maps each served key of that layer to whether it is
    stored [n, k] (see `Served.nk_ok`)."""
    check_shardable(g, s)
    _need(
        len(served) == spec.num_layers, f"{len(served)} layers, spec {spec.num_layers}"
    )
    cfg = dense or dense_config(spec)  # override: reduced-width tests
    entries = (manifest or KI.weight_manifest)(spec, cfg)
    return [_plan(e, served, g, s, cfg) for e in entries]


# The served ring: the MoE expert ring leases 4 of these banks. The ring
# default (10) does not fit VMEM beside the attention and MoE scratch
# (77.4 of 63.9 MiB at T = 32, tests/.../served_step_aot.py); bank count
# sizes the ring only, the packed tiles come from bank_bytes.
SERVED_N_BANKS = 4


def dense_config(spec: KI.ModelSpec) -> dc.DenseChainConfig:
    return dc.DenseChainConfig(
        hidden=spec.hidden_size,
        ratios=tuple(spec.compress_ratios),
        tp=spec.num_ranks,
        n_experts=spec.n_routed_experts,
        n_banks=SERVED_N_BANKS,
    )


def _plan(e, served, g, s, cfg) -> Build:
    if e.rule == "served":
        if e.layer is None:
            return Build(e, (LayerSource(None, e.name, False),), None)
        key = _SERVED_LAYER_KEY[e.name.rsplit("_", 1)[0]]
        return Build(e, (LayerSource(e.layer, key, False),), None)
    if e.rule == "pad":
        key = e.name.rsplit("_", 1)[0]  # w13_scale / w2_scale
        return Build(e, (LayerSource(e.layer, key, False),), moe_experts.pad_scale_rows)
    if e.rule == "rope":
        key = e.source.split(":")[0]
        return Build(e, (LayerSource(None, key, False),), attn_phase.pack_cos_sin)
    if e.rule == "pack":
        op = "smalls" if e.name == "dense_smalls" else e.name.removeprefix("dense_")
        if op != "smalls" and not cfg.stream_layers(op):
            # No layer carries the stream: the zero placeholder the kernel takes.
            return Build(e, (), lambda: dw.pack_stream(op, {}, cfg))
        src = tuple(plan_inputs(pack_inputs(op), served, g))
        return Build(e, src, lambda *a: packed_operand(op, src, a, g, s, cfg))
    if e.rule == "stack":
        src = tuple(plan_inputs((e.name,), served, g, stack=True))
        return Build(e, src, lambda *a: stacked_operand(e.name, src, a, g, e.shape[-1]))
    if e.rule == "proj_cols":  # rank s's output columns of a model-level linear
        key, n = e.source.split(":")[0], e.shape[-1]

        def proj_cols(x):
            x = _cols(x, s.col_rank * n, n)
            return x.astype(jnp.bfloat16) if e.dtype == BF16 else x

        return Build(e, (LayerSource(None, key, False),), proj_cols)
    if e.rule == "vocab_shard":
        shard = functools.partial(vocab_head.shard_lm_head, rank=s.rank, ranks=s.ranks)
        return Build(e, (LayerSource(None, "lm_head", False),), shard)
    if e.rule == "zeros":
        return Build(e, (), lambda: jnp.zeros(e.shape, jnp.dtype(e.dtype)))
    if e.rule == "placement":

        def given():
            raise Unsupported(f"{e.name}: passed by the adapter, not built")

        return Build(e, (), given)
    raise Unsupported(f"unknown manifest rule {e.rule!r} for {e.name}")


def entry_matches(e: KI.WeightEntry, shape: Sequence[int], dtype: str) -> bool:
    """A built operand against its manifest entry (-1 and "-" match any)."""
    return (
        len(e.shape) == len(shape)
        and all(a in (-1, b) for a, b in zip(e.shape, shape))
        and (e.dtype == "-" or e.dtype == dtype)
    )


# ------------------------------------------------------------- extra shares


def lm_head_rows(rank: int, ranks: int, vocab: int) -> tuple[int, int]:
    """(first row, rows) of this rank's vocab slice of the [V, H] lm_head.

    A row range of a row-major [V, H] bf16 array is one contiguous run of
    HBM (whole (16, 128) tiles when rows % 16 == 0), so the kernel reads it
    in place from the served tensor."""
    per = -(-vocab // ranks)
    start = min(rank * per, vocab)
    return start, min(per, vocab - start)
