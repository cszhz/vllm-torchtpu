from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

import jax
import torch
from vllm.distributed import get_pcp_group
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder as BaseAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    DEFAULT_SEQUENCE_LAYOUT_PROTOCOL,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
        "mamba_state_indices",
        "mamba_slot_read_offsets",
        "mamba_ckpt_indices",
        "mamba_request_distribution",
    ],
    meta_fields=[
        "sequence_layout_kind",
        "sequence_layout_protocol",
        "sequence_layout_version",
    ],
    drop_fields=["query_start_loc_cpu", "seq_lens_cpu", "prompt_lens_cpu"],
)
@dataclass
class AttentionMetadata:
    # (padded_total_num_scheduled_tokens,)
    input_positions: jax.Array
    # (max_num_seqs * max_num_blocks_per_req,)
    block_tables: jax.Array = None
    # (max_num_seqs,)
    seq_lens: jax.Array = None
    # (max_num_seqs + 1,)
    query_start_loc: jax.Array = None
    # (3,)
    request_distribution: jax.Array = None
    # (max_num_seqs,) int32 - physical slot id in the mamba kv-cache for the
    # request currently in each persistent-batch position. Compact-mamba uses
    # an explicit slot from its independent pool; unified align mode derives
    # the slot from the current entry in the device block table. None for
    # attention groups, non-mamba models, and Mamba modes whose native fallback
    # addresses block_tables[:, 0].
    mamba_state_indices: jax.Array | None = None
    # (mamba_num_blocks,) int32 — per-*slot* read offset for speculative
    # decoding with mamba layers. `mamba_slot_read_offsets[base_slot]` is
    # `num_accepted - 1` from the request's most recent verify step: the GDN
    # kernel reads the request's initial state from `base_slot + offset`
    # (the checkpoint of the last accepted token) and writes fresh
    # checkpoints starting at `base_slot`. Indexed by physical slot (not
    # batch position) so the value survives requests being rescheduled or
    # condensed. Updated on device after each sampling step; None unless the
    # model has mamba layers *and* speculative decoding is enabled.
    mamba_slot_read_offsets: jax.Array | None = None
    # `(num_reqs, num_spec + 1)` source block per checkpoint. Unified pool
    # with speculative decoding only; None elsewhere, where the sole
    # checkpoint is the request's own state block.
    mamba_ckpt_indices: jax.Array | None = None
    # (3,) int32 — GDN-specific request distribution, same format as
    # `request_distribution` but with the first segment covering all
    # *windowed* sequences (plain decodes and speculative verify windows of
    # up to num_spec + 1 tokens) instead of only 1-token decodes. The
    # persistent batch is ordered [decode][verify][prefill/mixed] so both
    # segmentations hold at once: ragged paged attention keeps its 1-token
    # decode front segment while the GDN kernel runs its windowed mode over
    # the first two groups. None unless the model has mamba layers and spec
    # decoding is enabled.
    mamba_request_distribution: jax.Array | None = None
    sequence_layout_kind: str = SequenceLayoutKind.ALL.value
    sequence_layout_protocol: str = DEFAULT_SEQUENCE_LAYOUT_PROTOCOL
    sequence_layout_version: int = 1

    # Immutable host scheduling snapshots for model-side input planning. They
    # never participate in device graphs and require no device-to-host reads.
    query_start_loc_cpu: tuple[int, ...] | None = field(default=None, init=False)
    seq_lens_cpu: tuple[int, ...] | None = field(default=None, init=False)
    prompt_lens_cpu: tuple[int, ...] | None = field(default=None, init=False)


@dataclass
class AttentionMetadataBuilderContext:
    """Per-call inputs the TPU metadata builder reads from the runner.

    Stashed on TPUModelRunner before invoking _build_attention_metadata so
    builder.build() can pick them up. Only fields TPU populates differently from
    the parent runner (seq_lens / query_start_loc) live in TPU's own _cpu
    staging tensors, not parent's CpuGpuBuffers) or that have no
    `CommonAttentionMetadata` equivalent (num_reqs, use_max_model_len,
    start_index, request_distribution, position_ids_override) live here.
    """

    num_reqs: int
    start_index: int
    use_max_model_len: bool
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    request_distribution: torch.Tensor
    position_ids_override: torch.Tensor | None = None
    query_start_loc_cpu: tuple[int, ...] | None = None
    seq_lens_cpu: tuple[int, ...] | None = None
    prompt_lens_cpu: tuple[int, ...] | None = None
    # Compact-mamba per-request recurrent-slot ids (device int32, length =
    # target_num_reqs). Only the mamba group's builder reads it; None when the
    # model has no mamba layers. See AttentionMetadata.mamba_state_indices.
    mamba_state_indices: torch.Tensor | None = None
    # Spec decode with mamba layers only; see the AttentionMetadata fields of
    # the same names. Only the mamba group's builder reads them.
    mamba_slot_read_offsets: torch.Tensor | None = None
    # State checkpoints per request on the unified pool (num_spec + 1, each
    # its own block). 1 means a single state per request, so the builder
    # emits no `mamba_ckpt_indices`.
    mamba_ckpt_window: int = 1
    mamba_request_distribution: torch.Tensor | None = None
    # Unified pool + spec decode only: the runner seeds this with an empty
    # list and each mamba group's builder appends the per-request state
    # block ids it derived from its block table, so the post-sampling
    # read-offset scatter can address every group's state block (groups
    # allocate distinct blocks, unlike the compact pool's shared slots).
    unified_mamba_state_indices: list[torch.Tensor] | None = None
    sequence_layout_descriptor: SequenceLayoutDescriptor = (
        DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR
    )
    # Per-chunk cache of the shared mamba row-offset plan, keyed by
    # (target_block_size, target_num_blocks) -> (state_index, ckpt_index,
    # ckpt_in_row). Valid only because every context is constructed fresh
    # per chunk: the plan derives from this chunk's seq_lens/ckpt_window,
    # which the key deliberately omits. Do not reuse a context across
    # chunks.
    mamba_row_plans: dict[
        tuple[int, int], tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ] = field(default_factory=dict)
    # Per-chunk batched block-table upload: kv_cache_group_id -> flat device
    # view into one staged H2D transfer (`stage_block_table_uploads`); None
    # means each build() uploads its own table (standalone builds, tests).
    staged_block_tables: dict | None = None
    # Per-chunk batched unified-pool mamba indices: kv_cache_group_id ->
    # (state_indices, ckpt_indices | None) from the single
    # `_staged_walk_products_compiled` program; None -> per-group fallback.
    staged_mamba_products: dict | None = None


_STAGE_BT_ROW_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
# Shared column divisors corresponding to context buckets (8k, 16k, 32k, 64k,
# 128k, 256k, 512k at max_model_len=524288): cols_g = cdiv(tnb, col_div).
_STAGE_BT_COL_DIVS: tuple[int, ...] = (64, 32, 16, 8, 4, 2, 1)


def _select_stage_row_bucket(num_reqs: int, num_reqs_padded: int) -> int:
    for rb in _STAGE_BT_ROW_BUCKETS:
        if rb >= num_reqs and rb < num_reqs_padded:
            return rb
    return num_reqs_padded


def _select_stage_col_div(
    entries: list[tuple[int, Any, int]],
    ctx: AttentionMetadataBuilderContext,
) -> int:
    if ctx.position_ids_override is not None or ctx.num_reqs <= 0:
        return 1
    needs: list[tuple[int, int]] = []
    for _, block_table_obj, tnb in entries:
        nbpr = getattr(block_table_obj, "num_blocks_per_row", None)
        if nbpr is None:
            return 1
        row_slice = nbpr[ctx.start_index : ctx.start_index + ctx.num_reqs]
        need_g = min(
            tnb, max(1, int(row_slice.max()) if len(row_slice) > 0 else 1)
        )
        needs.append((need_g, tnb))
    for col_div in _STAGE_BT_COL_DIVS:
        if all(
            min(tnb, max(1, cdiv(tnb, col_div))) >= need_g
            for need_g, tnb in needs
        ):
            return col_div
    return 1


def _staged_walk_products(
    staged_dev: torch.Tensor,
    seq_lens: torch.Tensor,
    table_geom: tuple,
    mamba_geom: tuple,
    ckpt_window: int,
) -> tuple:
    """Splits and zero-pads the staged flat block-table buffer into per-group
    tables `[num_reqs_padded * target_num_blocks]` and derives unified-pool
    mamba state and checkpoint indices."""
    outs = []
    tables_2d = []
    for geom in table_geom:
        if len(geom) == 3:
            off, n_pad, t_pad = geom
            r_up, c_up = n_pad, t_pad
        else:
            off, r_up, c_up, n_pad, t_pad = geom
        sub = staged_dev[off : off + r_up * c_up].view(r_up, c_up)
        if r_up == n_pad and c_up == t_pad:
            tbl_2d = sub
            outs.append(sub.reshape(n_pad * t_pad).clone())
        else:
            tbl_2d = (
                torch.cat(
                    (
                        sub,
                        torch.zeros(
                            (r_up, t_pad - c_up),
                            dtype=sub.dtype,
                            device=sub.device,
                        ),
                    ),
                    dim=1,
                )
                if t_pad > c_up
                else sub
            )
            if n_pad > r_up:
                tbl_2d = torch.cat(
                    (
                        tbl_2d,
                        torch.zeros(
                            (n_pad - r_up, t_pad),
                            dtype=tbl_2d.dtype,
                            device=tbl_2d.device,
                        ),
                    ),
                    dim=0,
                )
            outs.append(tbl_2d.reshape(n_pad * t_pad))
        tables_2d.append(tbl_2d)
    states = []
    ckpts = []
    if mamba_geom:
        active = seq_lens > 0
        for ti, t, bs in mamba_geom:
            tbl = tables_2d[ti]
            state_offsets = torch.clamp((seq_lens - 1) // bs, min=0, max=t - 1).to(
                torch.int64
            )
            gathered = torch.gather(tbl, 1, state_offsets.unsqueeze(1)).squeeze(1)
            states.append(torch.where(active, gathered, torch.zeros_like(gathered)))
            if ckpt_window > 1:
                ckpt_offsets = state_offsets.unsqueeze(1) + torch.arange(
                    ckpt_window, device=seq_lens.device, dtype=torch.int64
                ).unsqueeze(0)
                in_row = (ckpt_offsets < t) & active.unsqueeze(1)
                safe = torch.where(in_row, ckpt_offsets, torch.zeros_like(ckpt_offsets))
                ckpts.append(torch.gather(tbl, 1, safe) * in_row)
    return tuple(outs) + tuple(states) + tuple(ckpts)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _staged_walk_products_compiled(
    staged_dev: torch.Tensor,
    seq_lens: torch.Tensor,
    table_geom: tuple,
    mamba_geom: tuple,
    ckpt_window: int,
) -> tuple:
    """One device program for everything the metadata walk consumes:
    per-group block-table splits and 2D zero-padding (graph outputs are distinct
    buffers, so no per-group eager clone) plus the unified-pool mamba
    state/checkpoint gathers. `table_geom` is
    `((offset, row_bucket, cols_g, num_reqs_padded, target_num_blocks), ...)` in
    walk order; `mamba_geom` is `((table_geom_index, num_blocks, block_size),
    ...)`; both are static per (row_bucket, col_div) bucket. Returns `tables... +
    states... + (ckpts... if ckpt_window > 1)`. Issued for every staged walk
    (dummy runs included) so warmup and lockstep ranks dispatch the same
    program sequence.
    """
    return _staged_walk_products(
        staged_dev, seq_lens, table_geom, mamba_geom, ckpt_window
    )


class AttentionMetadataBuilder(BaseAttentionMetadataBuilder):
    """Single shared metadata builder for every TPU attention group."""

    def __init__(
        self, kv_cache_spec, layer_names, vllm_config, device, runner, kv_cache_group_id
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.runner = runner
        self.kv_cache_group_id = kv_cache_group_id
        self.is_mamba_group = isinstance(kv_cache_spec, MambaSpec)
        self.target_block_size = self.kv_cache_spec.block_size
        if self.is_mamba_group:
            try:
                cp_world_size = get_pcp_group().world_size
            except Exception:
                cp_world_size = 1
            self.target_block_size *= cp_world_size
        # Only mamba/GDN layers consume physical state slot ids; attention
        # groups leave AttentionMetadata.mamba_state_indices None.

        block_table_obj = runner.input_batch.block_table[self.kv_cache_group_id]
        self.block_tables_cpu = torch.zeros(
            (runner.max_num_reqs, block_table_obj.max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu",
        )

    def _mamba_row_plan(
        self, ctx: AttentionMetadataBuilderContext, target_num_blocks: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Row offsets into a mamba block table, cached and shared across groups.

        Mamba groups share identical sequence lengths, block sizes, and checkpoint
        geometry. Deriving row offset indices once per (block_size, target_num_blocks)
        allows all mamba groups to share the plan, leaving only per-group gathers.
        """
        key = (self.target_block_size, target_num_blocks)
        plan = ctx.mamba_row_plans.get(key)
        if plan is not None:
            return plan

        is_active = (ctx.seq_lens > 0).unsqueeze(1)
        state_offsets = torch.clamp(
            (ctx.seq_lens - 1) // self.target_block_size,
            min=0,
            max=target_num_blocks - 1,
        ).to(torch.int64)
        state_index = state_offsets.unsqueeze(1)

        ckpt_index: torch.Tensor | None = None
        in_row: torch.Tensor | None = None
        if ctx.mamba_ckpt_window > 1:
            # Spec decoding: the manager allocates `window - 1` checkpoint
            # blocks right after the positional state block, so checkpoint t
            # is the row entry `state_offsets + t`.
            ckpt_offsets = state_index + torch.arange(
                ctx.mamba_ckpt_window,
                device=state_offsets.device,
                dtype=state_offsets.dtype,
            ).unsqueeze(0)
            # A row too short for the group would alias checkpoints onto one
            # block; clamp to the null block and let the caller zero those
            # columns via `in_row`.
            in_row = (ckpt_offsets < target_num_blocks) & is_active
            ckpt_index = torch.where(
                in_row, ckpt_offsets, torch.zeros_like(ckpt_offsets)
            )

        plan = (state_index, ckpt_index, in_row)
        ctx.mamba_row_plans[key] = plan
        return plan

    def _block_table_geometry(
        self, ctx: AttentionMetadataBuilderContext
    ) -> tuple[Any, int]:
        """The (block_table_obj, target_num_blocks) this group's build() uses.

        Shared with `stage_block_table_uploads`, which must pack each group's
        rows with exactly the width the group's build() will slice back out.
        """
        runner = self.runner
        block_table_obj = runner.input_batch.block_table[self.kv_cache_group_id]
        if ctx.use_max_model_len:
            target_num_blocks = block_table_obj.max_num_blocks_per_req
        else:
            assert runner.most_model_len is not None
            target_num_blocks = cdiv(runner.most_model_len, self.target_block_size)
            if self.is_mamba_group and ctx.mamba_ckpt_window > 1:
                # Speculative decoding: the manager appends
                # `num_speculative_blocks` (= window - 1) checkpoint blocks
                # after the request's positional state block, so a mamba row
                # is that much wider than the positional part alone. This
                # mirrors `MambaSpec.max_num_blocks_per_req`, which the
                # use_max_model_len branch above already gets for free.
                # Without it the row stops short of the checkpoint group and
                # `in_row` below silently redirects checkpoints to the null
                # block, costing rollback accuracy with no error.
                target_num_blocks = min(
                    target_num_blocks + ctx.mamba_ckpt_window - 1,
                    block_table_obj.max_num_blocks_per_req,
                )
        return block_table_obj, target_num_blocks

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        runner = self.runner
        ctx = runner._attn_metadata_builder_ctx
        target_num_reqs = common_attn_metadata.num_reqs
        block_table_obj, target_num_blocks = self._block_table_geometry(ctx)

        # `position_ids` is only used for the dummy run in dummy runs, where we
        # want to use fixed position IDs instead of copying from the CPU tensor
        # that gets updated every step.
        staged = (
            ctx.staged_block_tables.get(self.kv_cache_group_id)
            if ctx.staged_block_tables is not None
            else None
        )
        if staged is not None:
            # The walk pre-uploaded every group's table in one batched
            # transfer; see `stage_block_table_uploads`.
            block_tables_dev = staged
            input_positions = (
                ctx.position_ids_override
                if ctx.position_ids_override is not None
                else runner.position_ids
            )
        elif ctx.position_ids_override is not None:
            block_tables_dev = torch.zeros(
                (target_num_reqs * target_num_blocks,), dtype=torch.int32
            ).to(runner.device)
            input_positions = ctx.position_ids_override
        else:
            block_tables = self.block_tables_cpu[:target_num_reqs, :target_num_blocks]
            block_tables.zero_()
            source_block_tables = block_table_obj.get_cpu_tensor()
            block_tables[: ctx.num_reqs, :target_num_blocks] = source_block_tables[
                ctx.start_index : ctx.start_index + ctx.num_reqs, :target_num_blocks
            ]
            # Flatten on CPU before H2D to avoid device-side
            # as_strided/reshape materialization on every decode step.
            block_tables_dev = block_tables.reshape(-1).to(
                runner.device, non_blocking=True
            )
            input_positions = runner.position_ids

        mamba_ckpt_indices = None
        if not self.is_mamba_group:
            mamba_state_indices = None
        elif ctx.mamba_state_indices is not None:
            # Default compact-mamba path: the runner provides a per-request
            # physical slot id from the compact slot pool.
            mamba_state_indices = ctx.mamba_state_indices
        elif runner._unified_kv_layout:
            # The pool keys mamba state by vLLM block ids in every cache
            # mode (it has no compact slot pool). Derive the current physical
            # state
            # slot on device from the zero-padded `block_tables_dev` (padded
            # tail rows resolve to the null block, never a stale id — the
            # GDN op scans the full length every step), avoiding a
            # D2H -> CPU gather -> H2D dependency.
            staged_products = (
                ctx.staged_mamba_products.get(self.kv_cache_group_id)
                if ctx.staged_mamba_products is not None
                else None
            )
            if staged_products is not None:
                # Derived in the single `_staged_walk_products_compiled`
                # program alongside the staged tables.
                mamba_state_indices, mamba_ckpt_indices = staged_products
            else:
                block_tables_2d = block_tables_dev.reshape(
                    target_num_reqs, target_num_blocks
                )
                # Everything but the gathers is shared with the other mamba
                # groups; see `_mamba_row_plan`.
                state_index, ckpt_index, ckpt_in_row = self._mamba_row_plan(
                    ctx, target_num_blocks
                )
                gathered_state_indices = torch.gather(
                    block_tables_2d,
                    dim=1,
                    index=state_index,
                ).squeeze(1)
                mamba_state_indices = torch.where(
                    ctx.seq_lens > 0,
                    gathered_state_indices,
                    torch.zeros_like(gathered_state_indices),
                )
                if ckpt_index is not None:
                    mamba_ckpt_indices = (
                        torch.gather(
                            block_tables_2d,
                            dim=1,
                            index=ckpt_index,
                        )
                        * ckpt_in_row
                    )
            if ctx.unified_mamba_state_indices is not None:
                # Spec decode: expose this group's state blocks for the
                # post-sampling read-offset scatter.
                ctx.unified_mamba_state_indices.append(mamba_state_indices)
        else:
            mamba_state_indices = None

        if self.is_mamba_group:
            mamba_slot_read_offsets = ctx.mamba_slot_read_offsets
            mamba_request_distribution = ctx.mamba_request_distribution
        else:
            mamba_slot_read_offsets = None
            mamba_request_distribution = None

        metadata = AttentionMetadata(
            input_positions=input_positions,
            block_tables=block_tables_dev,
            seq_lens=ctx.seq_lens,
            query_start_loc=ctx.query_start_loc,
            request_distribution=ctx.request_distribution,
            mamba_state_indices=mamba_state_indices,
            mamba_slot_read_offsets=mamba_slot_read_offsets,
            mamba_ckpt_indices=mamba_ckpt_indices,
            mamba_request_distribution=mamba_request_distribution,
            sequence_layout_kind=ctx.sequence_layout_descriptor.kind.value,
            sequence_layout_protocol=(ctx.sequence_layout_descriptor.protocol),
            sequence_layout_version=ctx.sequence_layout_descriptor.version,
        )
        metadata.query_start_loc_cpu = ctx.query_start_loc_cpu
        metadata.seq_lens_cpu = ctx.seq_lens_cpu
        metadata.prompt_lens_cpu = ctx.prompt_lens_cpu
        return metadata


def stage_block_table_uploads(
    runner: Any,
    ctx: AttentionMetadataBuilderContext,
    num_reqs_padded: int,
    gids: set[int] | None = None,
) -> None:
    """One batched H2D transfer for every block table a walk will build.

    Packs all participating groups' rows into one flat CPU scratch, transfers
    once, and hands each group's build() a flat view of its region via
    `ctx.staged_block_tables` (instead of one small per-group H2D per build).
    `gids` limits staging to a subset of groups; None stages every group.
    Dummy runs stage the same layout zero-filled so warmup dispatches the
    program sequence serving will. Views die with the per-chunk context.
    """
    if not runner.attn_groups:
        # Pre-initialization dummy runs build metadata before the KV cache
        # groups exist; each build() synthesizes its own table there.
        return

    # Walk in the same order the builds will, deduping shared group ids.
    entries: list[tuple[int, Any, int]] = []
    entry_builders: list[Any] = []
    seen: set[int] = set()
    for group_list in runner.attn_groups:
        for group in group_list:
            builder = group.metadata_builders[0]
            gid = builder.kv_cache_group_id
            if gid in seen or (gids is not None and gid not in gids):
                continue
            seen.add(gid)
            block_table_obj, target_num_blocks = builder._block_table_geometry(ctx)
            entries.append((gid, block_table_obj, target_num_blocks))
            entry_builders.append(builder)
    if not entries:
        return

    unified_mamba = ctx.mamba_state_indices is None and runner._unified_kv_layout
    mamba_geom: list[tuple[int, int, int]] = []
    mamba_entry_indices: list[int] = []
    if unified_mamba:
        for i, builder in enumerate(entry_builders):
            if builder.is_mamba_group:
                mamba_geom.append((i, entries[i][2], builder.target_block_size))
                mamba_entry_indices.append(i)

    if ctx.start_index > 0:
        runner._block_tables_dirty = True
        runner._block_tables_append_only = False
    bt_cache_key = (
        ctx.start_index,
        ctx.num_reqs,
        num_reqs_padded,
        ctx.use_max_model_len,
        tuple((gid, tnb) for gid, _, tnb in entries),
    )
    can_cache_bt = (
        ctx.position_ids_override is None
        and (not mamba_geom or ctx.mamba_ckpt_window <= 1)
        and gids is None
        and ctx.start_index == 0
    )
    if (
        can_cache_bt
        and not getattr(runner, "_block_tables_dirty", True)
        and getattr(runner, "_cached_bt_key", None) == bt_cache_key
        and getattr(runner, "_cached_staged_block_tables", None) is not None
    ):
        ctx.staged_block_tables = runner._cached_staged_block_tables
        ctx.staged_mamba_products = getattr(
            runner, "_cached_staged_mamba_products", None
        )
        return

    if ctx.position_ids_override is None and ctx.num_reqs > 0:
        row_bucket = _select_stage_row_bucket(ctx.num_reqs, num_reqs_padded)
        col_div = _select_stage_col_div(entries, ctx)
    else:
        row_bucket = num_reqs_padded
        col_div = 1

    upload_cols = [
        min(tnb, max(1, cdiv(tnb, col_div))) for _, _, tnb in entries
    ]
    total = sum(row_bucket * cols_g for cols_g in upload_cols)
    full_total = sum(num_reqs_padded * tnb for _, _, tnb in entries)
    scratch = runner._block_table_stage_cpu
    if scratch is None or scratch.numel() < full_total:
        scratch = torch.zeros(
            (full_total,), dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
        )
        runner._block_table_stage_cpu = scratch
        runner._block_tables_append_only = False
    flat = scratch[:total]

    stage_layout_key = (
        row_bucket,
        tuple(upload_cols),
        ctx.num_reqs,
        tuple(runner.input_batch.req_ids[: ctx.num_reqs])
        if getattr(runner, "input_batch", None) is not None
        else (),
        tuple(id(bt) for _, bt, _ in entries),
    )
    prev_nbpr_map = getattr(runner, "_bt_prev_nbpr", None)
    use_incremental = (
        can_cache_bt
        and getattr(runner, "_block_tables_append_only", False)
        and getattr(runner, "_cached_bt_stage_layout", None) == stage_layout_key
        and prev_nbpr_map is not None
        and all(
            getattr(bt, "num_blocks_per_row", None) is not None
            and gid in prev_nbpr_map
            for gid, bt, _ in entries
        )
    )
    if use_incremental:
        flat_np = flat.numpy()
        offset = 0
        for (gid, block_table_obj, _), cols_g in zip(entries, upload_cols):
            prev_nbpr = prev_nbpr_map[gid]
            cur_nbpr = block_table_obj.num_blocks_per_row[: ctx.num_reqs]
            changed = (cur_nbpr != prev_nbpr).nonzero()[0]
            if changed.size > 0:
                rows_np = flat_np[offset : offset + row_bucket * cols_g].reshape(
                    row_bucket, cols_g
                )
                src_np = block_table_obj.get_numpy_array()
                for r in changed:
                    c0 = int(prev_nbpr[r])
                    c1 = min(cols_g, int(cur_nbpr[r]))
                    if c0 < c1:
                        rows_np[r, c0:c1] = src_np[r, c0:c1]
                prev_nbpr[:] = cur_nbpr
            offset += row_bucket * cols_g
    else:
        if not can_cache_bt:
            runner._block_tables_dirty = True
            runner._block_tables_append_only = False
            runner._cached_bt_stage_layout = None
        flat.zero_()
        if ctx.position_ids_override is None:
            offset = 0
            for (_, block_table_obj, _), cols_g in zip(entries, upload_cols):
                rows = flat[offset : offset + row_bucket * cols_g].view(
                    row_bucket, cols_g
                )
                source = block_table_obj.get_cpu_tensor()
                rows[: ctx.num_reqs] = source[
                    ctx.start_index : ctx.start_index + ctx.num_reqs, :cols_g
                ]
                offset += row_bucket * cols_g

    staged_dev = flat.to(runner.device, non_blocking=True)

    # One compiled program splits and zero-pads the staged tensor into static
    # per-group `[num_reqs_padded * target_num_blocks]` tables and derives
    # every unified-pool mamba group's state/checkpoint indices.
    table_geom: list[tuple[int, int, int, int, int]] = []
    offset = 0
    for (_, _, target_num_blocks), cols_g in zip(entries, upload_cols):
        table_geom.append(
            (offset, row_bucket, cols_g, num_reqs_padded, target_num_blocks)
        )
        offset += row_bucket * cols_g

    if (
        len(entries) == 1
        and not mamba_geom
        and row_bucket == num_reqs_padded
        and upload_cols[0] == entries[0][2]
    ):
        gid, _, target_num_blocks = entries[0]
        length = num_reqs_padded * target_num_blocks
        ctx.staged_block_tables = {gid: staged_dev[:length]}
        if can_cache_bt:
            runner._cached_bt_key = bt_cache_key
            runner._cached_bt_stage_layout = stage_layout_key
            runner._cached_staged_block_tables = ctx.staged_block_tables
            runner._cached_staged_mamba_products = None
            runner._bt_prev_nbpr = {
                g: bt.num_blocks_per_row[: ctx.num_reqs].copy()
                for g, bt, _ in entries
                if getattr(bt, "num_blocks_per_row", None) is not None
            }
            runner._block_tables_dirty = False
            runner._block_tables_append_only = True
        return

    if staged_dev.device.type == "tpu":
        products = _staged_walk_products_compiled(
            staged_dev,
            ctx.seq_lens,
            tuple(table_geom),
            tuple(mamba_geom),
            ctx.mamba_ckpt_window,
        )
    else:
        products = _staged_walk_products(
            staged_dev,
            ctx.seq_lens,
            tuple(table_geom),
            tuple(mamba_geom),
            ctx.mamba_ckpt_window,
        )

    n_tables = len(entries)
    n_mamba = len(mamba_geom)
    ctx.staged_block_tables = {
        gid: products[i] for i, (gid, _, _) in enumerate(entries)
    }
    if n_mamba:
        states = products[n_tables : n_tables + n_mamba]
        has_ckpt = ctx.mamba_ckpt_window > 1
        ckpts = products[n_tables + n_mamba :] if has_ckpt else (None,) * n_mamba
        ctx.staged_mamba_products = {
            entries[ei][0]: (states[j], ckpts[j] if has_ckpt else None)
            for j, ei in enumerate(mamba_entry_indices)
        }
    if can_cache_bt:
        runner._cached_bt_key = bt_cache_key
        runner._cached_bt_stage_layout = stage_layout_key
        runner._cached_staged_block_tables = ctx.staged_block_tables
        runner._cached_staged_mamba_products = ctx.staged_mamba_products
        runner._bt_prev_nbpr = {
            g: bt.num_blocks_per_row[: ctx.num_reqs].copy()
            for g, bt, _ in entries
            if getattr(bt, "num_blocks_per_row", None) is not None
        }
        runner._block_tables_dirty = False
        runner._block_tables_append_only = True


@torch.no_grad()
def precompile_staged_block_table_uploads(runner: Any) -> None:
    """Warm `_staged_walk_products_compiled` across all 2D `(row_bucket, col_div)`
    variants under `no_grad` so no serving-time recompile occurs inside
    `execute_model`."""
    if not runner.attn_groups or runner.device.type != "tpu":
        return
    from vllm_torchtpu.utils import synchronize_tensors

    modes = [True]
    if runner.most_model_len is not None:
        modes.append(False)

    for use_max_model_len in modes:
        num_reqs_padded = (
            runner.num_reqs_max_model_len
            if use_max_model_len
            else runner.num_reqs_most_model_len
        )
        assert num_reqs_padded is not None
        dummy_seq_lens = torch.zeros(
            (num_reqs_padded,), dtype=torch.int32, device=runner.device
        )
        dummy_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs_padded,
            start_index=0,
            use_max_model_len=use_max_model_len,
            seq_lens=dummy_seq_lens,
            query_start_loc=torch.zeros(
                (num_reqs_padded + 1,), dtype=torch.int32, device=runner.device
            ),
            request_distribution=torch.zeros(
                (3,), dtype=torch.int32, device=runner.device
            ),
            mamba_ckpt_window=runner._mamba_ckpt_window,
        )
        entries: list[tuple[int, Any, int]] = []
        entry_builders: list[Any] = []
        seen: set[int] = set()
        for group_list in runner.attn_groups:
            for group in group_list:
                builder = group.metadata_builders[0]
                gid = builder.kv_cache_group_id
                if gid in seen:
                    continue
                seen.add(gid)
                block_table_obj, tnb = builder._block_table_geometry(dummy_ctx)
                entries.append((gid, block_table_obj, tnb))
                entry_builders.append(builder)
        if not entries:
            continue

        unified_mamba = (
            not runner._has_mamba_state and runner._unified_kv_layout
        )
        mamba_geom: list[tuple[int, int, int]] = []
        if unified_mamba:
            for i, builder in enumerate(entry_builders):
                if builder.is_mamba_group:
                    mamba_geom.append((i, entries[i][2], builder.target_block_size))

        row_buckets = sorted(
            {
                *(rb for rb in _STAGE_BT_ROW_BUCKETS if rb < num_reqs_padded),
                num_reqs_padded,
            }
        )
        seen_shapes: set[tuple[int, tuple[int, ...]]] = set()
        for rb in row_buckets:
            for col_div in _STAGE_BT_COL_DIVS:
                upload_cols = tuple(
                    min(tnb, max(1, cdiv(tnb, col_div)))
                    for _, _, tnb in entries
                )
                shape_key = (rb, upload_cols)
                if shape_key in seen_shapes:
                    continue
                seen_shapes.add(shape_key)
                if (
                    len(entries) == 1
                    and not mamba_geom
                    and rb == num_reqs_padded
                    and upload_cols[0] == entries[0][2]
                ):
                    continue
                table_geom: list[tuple[int, int, int, int, int]] = []
                offset = 0
                for (_, _, tnb), cols_g in zip(entries, upload_cols):
                    table_geom.append(
                        (offset, rb, cols_g, num_reqs_padded, tnb)
                    )
                    offset += rb * cols_g
                dummy_staged = torch.zeros(
                    (offset,), dtype=torch.int32, device=runner.device
                )
                products = _staged_walk_products_compiled(
                    dummy_staged,
                    dummy_seq_lens,
                    tuple(table_geom),
                    tuple(mamba_geom),
                    runner._mamba_ckpt_window,
                )
                synchronize_tensors(list(products))
