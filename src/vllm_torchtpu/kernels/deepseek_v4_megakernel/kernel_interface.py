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
"""The contract between the DeepSeek-V4 megakernel and its serving adapter.

The adapter (`vllm_adapter.py`) owns everything that touches the runner:
step eligibility, buckets, cache binding and operand assembly. The kernel owns
the weights it reads and the per-rank program. They meet here:

- `ModelSpec` (known at model load) and `StepConfig` (plus the bound cache
  layout).
- `Bucket`: one compiled row count.
- `KernelImpl`: `collect_weights(model, spec)` and `build_step(cfg, bucket)`.

`build_step` returns a per-rank JAX function (the adapter wraps it in
`shard_map`) taking, positionally, with B = bucket.reqs (this rank's request
slots) and M = bucket.own_rows (B * q); bucket.rows is the global row count T:

    meta           int32 [META_LEN]   META_RANK: mesh index, META_Q: rows/request,
                                      META_EBLK: EP rank whose routed experts
                                      this rank serves (ep_rank_order()[mesh index]),
                                      META_VSTART: first vocab id of this rank's
                                      embed shard (attn_tp > 1)
    input_ids      int32 [M]          0 on inactive rows
    positions      int32 [M]          0 on inactive rows
    seq_lens       int32 [B]          tokens incl. this step's window, 0 = inactive
    block_tables   int32 [B, W_g]     one per KV-cache group, cfg.group_widths order
    *weights                          collect_weights(...) order
    *caches                           cfg.cache_shapes order (donated)

Row r belongs to request r // q at window offset r % q. It returns

    hidden         [M, hidden_size]   after the final mHC head collapse + norm
    *aux           A x [M, hidden_size], one per cfg.model.aux_layers entry
    *caches        updated caches, same order and shapes as passed

An inactive request (seq_lens 0) must read and write nothing: an idle DP rank
runs the program with every request inactive so the in-kernel collectives pair.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Protocol

META_LEN = 8
META_RANK = 0
META_Q = 1
META_EBLK = 2
META_VSTART = 3

CSA = "csa"
HCA = "hca"
SWA = "swa"  # sliding-window-only (the DSpark draft layers)


def request_block_rows(q: int) -> int:
    """Global rows one request occupies in the step: its q rows padded to a
    whole f32 sublane tile (step_kernel's QP). A bucket of T rows holds
    T // request_block_rows(q) requests over the whole DP group."""
    return -(-q // 8) * 8


@dataclasses.dataclass(frozen=True)
class Bucket:
    """One compiled step shape: `rows` = T global rows over the DP group (the
    sum of every rank's active rows fits), `q` rows per request, `reqs`
    request slots per rank (own rows M = reqs * q), and `max_ctx`, the
    longest request context it serves (None: the kernel's default)."""

    q: int
    rows: int
    reqs: int = 4
    max_ctx: int | None = None

    def __post_init__(self) -> None:
        if self.q < 1 or self.reqs < 1 or self.rows < self.q:
            raise ValueError(f"bucket needs rows >= q >= 1 and reqs >= 1, got {self}")

    @property
    def own_rows(self) -> int:
        return self.reqs * self.q

    @property
    def key(self) -> str:
        return f"mk_q{self.q}_t{self.rows}"


@dataclasses.dataclass(frozen=True)
class CacheRef:
    """Where one cache role of one layer lives: an index into
    `StepConfig.cache_shapes` (and the caches operand list) and an index into
    the block-table operands."""

    array: int
    group: int


@dataclasses.dataclass(frozen=True)
class LayerCaches:
    """The caches one decoder layer reads and writes.

    Several roles may name the same array (the runner overlays SWA caches and
    compressor states onto other layers' arrays); each array still appears once
    in the caches operand list.
    """

    kind: str  # CSA, HCA or SWA
    compress_ratio: int
    swa: CacheRef
    kv: CacheRef | None = None  # CSA: NoPE records; HCA: raw bf16 latents
    state: CacheRef | None = None  # compressor state (none on SWA)
    kv_rope: CacheRef | None = None  # CSA only
    indexer_k: CacheRef | None = None  # CSA only
    indexer_state: CacheRef | None = None  # CSA only


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """What the adapter knows at model load, before any cache exists."""

    hidden_size: int
    num_layers: int
    compress_ratios: tuple[int, ...]  # per decoder layer
    dtype: str  # hidden/aux output dtype name, e.g. "bfloat16"
    q: int  # rows per request: 1 + num_speculative_tokens
    mode: str  # "dp" (all-or-nothing over the DP group) or "tp"
    axis_name: str
    num_ranks: int
    # 1-based layer ids whose post-layer (mHC-collapsed) state the drafter
    # reads, as DeepseekV4Model.aux_hidden_state_layers.
    aux_layers: tuple[int, ...]
    sliding_window: int = 128
    rms_eps: float = 1e-6
    hc_eps: float = 1e-6
    vocab_size: int = 129280
    num_hash_layers: int = 3
    n_routed_experts: int = 384
    # vLLM's TP size inside one DP engine (TP8 x DP2: 8). Mesh index
    # g * attn_tp + j is TP rank j of engine g; its served tensors are TP
    # shard j, the embed a vocab shard (step_kernel.head_block).
    attn_tp: int = 1
    # static expert placement (TPU_EXPERT_PLACEMENT): "" none, "perm", or
    # "replicas" (step_kernel.StepKernelConfig.moe_placement), with
    # `expert_slots` local slots per layer
    moe_placement: str = ""
    expert_slots: int = 0

    def __post_init__(self) -> None:
        if self.attn_tp < 1 or self.num_ranks % self.attn_tp:
            raise ValueError(f"attn_tp {self.attn_tp} of {self.num_ranks} ranks")
        if self.moe_placement not in ("", "perm", "replicas") or (
            bool(self.moe_placement) != (self.expert_slots > 0)
        ):
            raise ValueError(
                f"moe_placement {self.moe_placement!r} with {self.expert_slots} slots"
            )

    @property
    def dp_groups(self) -> int:
        return self.num_ranks // self.attn_tp


@dataclasses.dataclass(frozen=True)
class StepConfig:
    """`ModelSpec` plus the bound cache layout."""

    model: ModelSpec
    layers: tuple[LayerCaches, ...]
    cache_shapes: tuple[tuple[tuple[int, ...], str], ...]
    group_widths: tuple[int, ...]  # block-table columns per group operand
    swa_page: int = 128  # tokens per SWA cache page


@dataclasses.dataclass(frozen=True)
class WeightEntry:
    """One weight operand, per rank, in `weight_manifest` order.

    rule:
      "served"  the served module tensor itself (zero-copy view), as named in
                `source` (module path relative to the decoder layer / model).
      "pack"    a load-time copy built by the named packer in
                `dense_weights.py` from natural per-rank arrays (`source`
                names its inputs); shape fixed by DenseChainConfig(tp=R).
      "stack"   a load-time copy: the per-layer served tensors named in
                `source` stacked on a new leading layer axis (f32 unless
                `dtype` says otherwise).
      "pad"     a load-time copy: moe_experts.pad_scale_rows(served).
      "vocab_shard" a load-time copy: this rank's lm_head rows (`lm_head_entry`).
      "proj_cols" a load-time copy: this rank's output columns of a global
                linear (shape[-1] of them), bf16 when `dtype` says so.
      "rope"    a load-time copy: attn_phase.pack_cos_sin of the served global
                rotary table named first in `source`.
      "placement" not from the module tree: the adapter passes it (the static
                expert placement tables).
    shape: per-rank shape; -1 = taken from the served tensor.
    layer: decoder layer for per-layer entries, None for global ones.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    rule: str
    source: str
    layer: int | None = None


def weight_manifest(
    spec: ModelSpec, dense: Any = None, logits: bool = False
) -> list[WeightEntry]:
    """The kernel's weight operands for one rank (DP16 + L4e, R = num_ranks).
    `logits`: the step also applies this rank's lm_head shard (appended last).

    Order: globals, dense ring (packed), attention per-layer stacks, MoE
    per layer. Rank r's slices of the sharded dense projections are taken by
    the packer inputs (dense_weights docstrings; B2_dense_ring.md)."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc

    H, L = spec.hidden_size, spec.num_layers
    # `dense` overrides the DSv4-Pro dense geometry (reduced-width tests)
    dense = dense or dc.DenseChainConfig(
        hidden=H,
        ratios=tuple(spec.compress_ratios),
        tp=spec.num_ranks,
        n_experts=spec.n_routed_experts,
    )
    csa = [i for i, r in enumerate(spec.compress_ratios) if r == 4]
    # attn_tp > 1: the served embed is this rank's vocab-parallel shard
    vocab_rows = spec.vocab_size if spec.attn_tp == 1 else -1
    out = [
        WeightEntry("embed", (vocab_rows, H), "bfloat16", "served",
                    "model.embed_tokens.weight"),
        WeightEntry("final_norm", (H,), "bfloat16", "served", "model.norm.weight"),
        WeightEntry("hc_head_fn", (4, 4 * H), "float32", "served",
                    "model.hc_head_fn"),
        WeightEntry("hc_head_base", (4,), "float32", "served", "model.hc_head_base"),
        WeightEntry("hc_head_scale", (1,), "float32", "served",
                    "model.hc_head_scale"),
        # Rotary tables packed once (attn_phase.pack_cos_sin): the served
        # [max_pos, 64] layout would be copied into the kernel's every step.
        WeightEntry("cos_sin", (-1, 128), "float32", "rope",
                    "cos_sin: layers.0.attn.rotary_emb.cos_sin_cache "
                    "(compress_rope_theta: every target layer has ratio > 1)"),
        WeightEntry("comp_cos_sin", (-1, 128), "float32", "rope",
                    "comp_cos_sin: the same tensor as cos_sin (the compressor "
                    "uses its layer's rotary_emb)"),
        WeightEntry("idx_cos_sin", (-1, 128), "float32", "rope",
                    "idx_cos_sin: layers.<first csa>.attn.indexer_rotary_emb"),
        WeightEntry("dense_smalls", (L, dense.smalls_rows, 128), "float32", "pack",
                    "dense_weights.pack_smalls"),
    ]  # fmt: skip
    for name, s in dense.streams.items():
        n_l = max(1, len(dense.stream_layers(name)))
        out.append(WeightEntry(
            "dense_" + name, (n_l, s.n_tiles, s.tile_rows, s.tile_cols), s.dtype,
            "pack", f"dense_weights.pack_stream('{name}')",
        ))  # fmt: skip
    out += [
        WeightEntry("attn_sink", (L, 128), "float32", "stack",
                    "layers.*.attn.attn_sink (a TP shard's heads first, -inf "
                    "padded)"),
        WeightEntry("comp_ape", (L, 128, 1024), "float32", "stack",
                    "layers.*.attn.compressor.ape: HCA [128, 512], CSA [4, 1024], "
                    "zero-padded"),
        WeightEntry("comp_norm", (L, 512), "float32", "stack",
                    "layers.*.attn.compressor.norm.weight"),
        WeightEntry("idx_comp_ape", (max(1, len(csa)), 4, 256), "float32", "stack",
                    "csa layers: attn.indexer.compressor.ape"),
        WeightEntry("idx_comp_norm", (max(1, len(csa)), 128), "float32", "stack",
                    "csa layers: attn.indexer.compressor.norm.weight"),
        WeightEntry("moe_bias", (L, spec.n_routed_experts), "float32", "stack",
                    "layers.*.ffn.gate.e_score_correction_bias (hash layers: 0)"),
    ]  # fmt: skip
    for li in range(spec.num_hash_layers):
        src = f"layers.{li}.ffn.gate.tid2eid"
        out.append(
            WeightEntry(
                f"tid2eid_{li}", (spec.vocab_size, 6), "int32", "served", src, li
            )
        )
    for li in range(L):
        fe = f"layers.{li}.ffn.experts.routed_experts"
        out += [
            WeightEntry(f"w13_{li}", (-1, -1, -1), "-", "served", fe + ".w13_weight",
                        li),
            WeightEntry(f"w13_scale_{li}", (-1, -1, -1), "float32", "pad",
                        fe + "._tpu_fused_w13_scale", li),
            WeightEntry(f"w2_{li}", (-1, -1, -1), "-", "served", fe + ".w2_weight", li),
            WeightEntry(f"w2_scale_{li}", (-1, -1, -1), "float32", "pad",
                        fe + "._tpu_fused_w2_scale", li),
        ]  # fmt: skip
    if spec.moe_placement:
        src = "the adapter's: this rank's EP row of moe_phase.placement_tables"
        slots = (L, spec.expert_slots)
        out.append(WeightEntry("moe_held", slots, "int32", "placement", src))
        if spec.moe_placement == "replicas":
            out.append(WeightEntry("moe_copy", slots, "int32", "placement", src))
    if logits:
        out.append(lm_head_entry(spec))
    return out


def lm_head_entry(spec: ModelSpec) -> WeightEntry:
    """This rank's vocab shard of the served lm_head [V, H] bf16
    (vocab_head.shard_lm_head: rows [r*Vs, (r+1)*Vs), zero past V)."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import vocab_head as VH

    vs = VH.vocab_shard(spec.vocab_size, spec.num_ranks)
    return WeightEntry("lm_head", (vs, spec.hidden_size), "bfloat16", "vocab_shard",
                       "lm_head.weight")  # fmt: skip


def draft_weight_manifest(
    spec: ModelSpec, dense: Any, logits: bool = False
) -> list[WeightEntry]:
    """The DSpark draft step's weight operands, per rank, in
    `step_impl.draft_weight_names` order. `spec` is the draft's (its SWA
    layers); globals are the draft model's (`embed` is the target's, aliased).
    Rule "zeros": a placeholder of `shape` (an AttnWeights field the SWA
    attention does not read)."""
    H, L = spec.hidden_size, spec.num_layers
    E = spec.n_routed_experts

    def g(name, shape, dtype, rule, source):
        return WeightEntry(name, shape, dtype, rule, source)

    vocab_rows = spec.vocab_size if spec.attn_tp == 1 else -1  # a TP vocab shard
    out = [
        g("embed", (vocab_rows, H), "bfloat16", "served", "embed_tokens.weight"),
        g("cos_sin", (-1, 128), "float32", "rope", "cos_sin: layers.0.attn.rotary_emb"),
        # this rank's output columns: bf16 (exact for fp8 values) + their scales
        g(
            "main_proj",
            (-1, H // spec.num_ranks),
            "bfloat16",
            "proj_cols",
            "main_proj: the fp8 weight's columns [r*H/R, (r+1)*H/R)",
        ),
        g(
            "main_proj_scale",
            (H // spec.num_ranks,),
            "float32",
            "proj_cols",
            "main_proj_scale",
        ),
        g("main_norm", (H,), "bfloat16", "served", "main_norm.weight"),
        g("hc_head_fn", (4, 4 * H), "float32", "served", "hc_head_fn"),
        g("hc_head_base", (4,), "float32", "served", "hc_head_base"),
        g("hc_head_scale", (1,), "float32", "served", "hc_head_scale"),
    ]
    if logits:  # compute_draft_logits: lm_head(norm(.)), the target's lm_head
        out += [g("norm", (H,), "bfloat16", "served", "norm.weight"),
                lm_head_entry(spec)]  # fmt: skip
    for name, s in dense.streams.items():
        n_l = max(1, len(dense.stream_layers(name)))
        shape = (n_l, s.n_tiles, s.tile_rows, s.tile_cols)
        out.append(g("dense_" + name, shape, s.dtype, "pack",
                     f"dense_weights.pack_stream('{name}')"))  # fmt: skip
    out.append(g("dense_smalls", (L, dense.smalls_rows, 128), "float32", "pack",
                 "dense_weights.pack_smalls"))  # fmt: skip
    out.append(g("moe_bias", (L, E), "float32", "stack",
                 "layers.*.ffn.gate.e_score_correction_bias"))  # fmt: skip
    for li in range(L):
        fe = f"layers.{li}.ffn.experts.routed_experts"
        out += [
            WeightEntry(f"w13_{li}", (-1, -1, -1), "-", "served", fe + ".w13_weight",
                        li),
            WeightEntry(f"w13_scale_{li}", (-1, -1, -1), "float32", "pad",
                        fe + "._tpu_fused_w13_scale", li),
            WeightEntry(f"w2_{li}", (-1, -1, -1), "-", "served", fe + ".w2_weight", li),
            WeightEntry(f"w2_scale_{li}", (-1, -1, -1), "float32", "pad",
                        fe + "._tpu_fused_w2_scale", li),
        ]  # fmt: skip
    # SP.AttnWeights, in field order: the SWA phase reads attn_sink and cos_sin
    out += [
        g("attn_sink", (L, 128), "float32", "stack", "layers.*.attn.attn_sink"),
        g("comp_ape", (1, 8, 128), "float32", "zeros", "unread by swa"),
        g("comp_norm", (1, 512), "float32", "zeros", "unread by swa"),
        g("idx_comp_ape", (1, 4, 256), "float32", "zeros", "unread by swa"),
        g("idx_comp_norm", (1, 128), "float32", "zeros", "unread by swa"),
        g("cos_sin", (-1, 128), "float32", "rope", "cos_sin: layers.0.attn.rotary_emb"),
        g("comp_cos_sin", (8, 128), "float32", "zeros", "unread by swa"),
        g("idx_cos_sin", (8, 128), "float32", "zeros", "unread by swa"),
    ]
    if spec.moe_placement:
        src = "the adapter's: this rank's EP row of the draft layers' placement"
        slots = (L, spec.expert_slots)
        out.append(g("moe_held", slots, "int32", "placement", src))
        if spec.moe_placement == "replicas":
            out.append(g("moe_copy", slots, "int32", "placement", src))
    return out


class KernelImpl(Protocol):
    # True: build_step's fn writes the caches it is handed and returns none.
    cache_inplace: bool

    # Runs at model load, before the KV cache is sized: a weight copy made
    # later would come out of memory the cache already claimed.
    def collect_weights(self, model: Any, spec: ModelSpec) -> list[Any]: ...

    def build_step(
        self, cfg: StepConfig, bucket: Bucket
    ) -> Callable[..., Sequence[Any]]: ...


class PlaceholderKernel:
    """Stands in until the kernel lands; enabling the megakernel with it fails
    at model load rather than serving the stock path under the flag."""

    cache_inplace = False

    def collect_weights(self, model: Any, spec: ModelSpec) -> list[Any]:
        raise NotImplementedError("DeepSeek-V4 megakernel: kernel not built yet")

    def build_step(
        self, cfg: StepConfig, bucket: Bucket
    ) -> Callable[..., Sequence[Any]]:
        raise NotImplementedError("DeepSeek-V4 megakernel: kernel not built yet")


def default_kernel() -> KernelImpl:
    return PlaceholderKernel()
