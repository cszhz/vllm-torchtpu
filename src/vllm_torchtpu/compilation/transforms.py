# SPDX-License-Identifier: Apache-2.0
"""Graph-level compilation transforms for TorchTPU (e.g. ``transforms.microbatch``).

Provides a user-facing context manager ``transforms.microbatch(min_len=..., groups=...)``
that brackets a model region with zero-overhead FX marker ops during Dynamo tracing:

.. code-block:: python

    from vllm_torchtpu import transforms

    with transforms.microbatch(min_len=1024, groups=2):
        x_full = tensor_model_parallel_all_gather(x_shard, dim=0)
        gate_up, _ = self.gate_up_proj(x_full)

During per-bucket compilation in :class:`TpuCompilerAdaptor`,
:func:`apply_microbatch_pass` inspects the concrete token bucket ``size``:

- When ``size < min_len`` (or ``groups <= 1`` or ``size`` is not divisible by
  ``groups * min_chunk_multiple``), the marker ops are stripped in-place and the
  region executes unsplit with zero overhead.
- When ``size >= min_len``, the bracketed region is automatically partitioned
  into a 2-stage pipeline (``Collective -> Compute`` or ``Compute -> Collective``),
  live-in token tensors are sliced along ``split_dim`` into ``groups`` chunks,
  shared weights/parameters are reused without duplication, and a 2-barrier
  diamond pipeline schedule overlaps ``Stage2(g)`` with ``Stage1(g + 1)`` using
  ``jax.lax.optimization_barrier`` before concatenating live-out tensors along
  ``split_dim``.
"""

from __future__ import annotations

import contextlib
import operator
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Custom Marker Ops & JAX Optimization Barrier Op
# ---------------------------------------------------------------------------


@torch.library.custom_op("tpu_transforms::microbatch_enter", mutates_args=())
def _microbatch_enter_op(
    min_len: int,
    groups: int,
    split_dim: int,
    barrier: bool,
    min_chunk_multiple: int,
) -> torch.Tensor:
    del min_len, groups, split_dim, barrier, min_chunk_multiple
    return torch.empty(0, dtype=torch.int32, device="cpu")


@_microbatch_enter_op.register_fake
def _(
    min_len: int,
    groups: int,
    split_dim: int,
    barrier: bool,
    min_chunk_multiple: int,
) -> torch.Tensor:
    del min_len, groups, split_dim, barrier, min_chunk_multiple
    return torch.empty(0, dtype=torch.int32, device="cpu")


@torch.library.custom_op(
    "tpu_transforms::microbatch_stage_boundary", mutates_args=()
)
def _microbatch_stage_boundary_op(token: torch.Tensor) -> torch.Tensor:
    return token.clone()


@_microbatch_stage_boundary_op.register_fake
def _(token: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(token)


@torch.library.custom_op("tpu_transforms::microbatch_exit", mutates_args=())
def _microbatch_exit_op(token: torch.Tensor) -> torch.Tensor:
    return token.clone()


@_microbatch_exit_op.register_fake
def _(token: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(token)


# Mark marker ops as side-effecting so Dynamo and FX dead-code elimination
# preserve them until `apply_microbatch_pass` runs in `TpuCompilerAdaptor`.
torch.fx.node.has_side_effect(torch.ops.tpu_transforms.microbatch_enter)
torch.fx.node.has_side_effect(torch.ops.tpu_transforms.microbatch_enter.default)
torch.fx.node.has_side_effect(torch.ops.tpu_transforms.microbatch_stage_boundary)
torch.fx.node.has_side_effect(
    torch.ops.tpu_transforms.microbatch_stage_boundary.default
)
torch.fx.node.has_side_effect(torch.ops.tpu_transforms.microbatch_exit)
torch.fx.node.has_side_effect(torch.ops.tpu_transforms.microbatch_exit.default)

_ENTER_TARGETS = {
    _microbatch_enter_op,
    torch.ops.tpu_transforms.microbatch_enter,
    torch.ops.tpu_transforms.microbatch_enter.default,
}
_STAGE_BOUNDARY_TARGETS = {
    _microbatch_stage_boundary_op,
    torch.ops.tpu_transforms.microbatch_stage_boundary,
    torch.ops.tpu_transforms.microbatch_stage_boundary.default,
}
_EXIT_TARGETS = {
    _microbatch_exit_op,
    torch.ops.tpu_transforms.microbatch_exit,
    torch.ops.tpu_transforms.microbatch_exit.default,
}


@torch.library.custom_op(
    "tpu_transforms::microbatch_barrier_2_fallback", mutates_args=()
)
def _microbatch_barrier_2_fallback_op(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return a.clone(), b.clone()


@_microbatch_barrier_2_fallback_op.register_fake
def _(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(a), torch.empty_like(b)


def _build_opt_barrier_op() -> Any:
    """Register ``pallas::microbatch_barrier_2`` backed by ``jax.lax.optimization_barrier``."""
    try:
        import jax
        from torch_tpu._internal import pallas

        def _barrier_2_jax(a, b):
            return jax.lax.optimization_barrier((a, b))

        _barrier_2_jax.__annotations__ = {
            "a": jax.Array,
            "b": jax.Array,
            "return": tuple[jax.Array, jax.Array],
        }

        op = pallas.jax_op("pallas::microbatch_barrier_2", _barrier_2_jax)

        def _fake_barrier_2(
            a: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.empty_like(a), torch.empty_like(b)

        op.register_fake(_fake_barrier_2)
        return torch.ops.pallas.microbatch_barrier_2.default
    except Exception:
        return torch.ops.tpu_transforms.microbatch_barrier_2_fallback.default


_MICROBATCH_BARRIER_2_OP = _build_opt_barrier_op()


def _select_barrier_op_for_node(node: fx.Node) -> Any:
    """Pick TPU Pallas barrier op for TPU tensors and CPU fallback for CPU tensors."""
    val = _node_tensor_val(node)
    if val is not None and val.device.type != "tpu":
        return torch.ops.tpu_transforms.microbatch_barrier_2_fallback.default
    return _MICROBATCH_BARRIER_2_OP


_ACTIVE_TOKENS: list[torch.Tensor] = []


def is_microbatch_active() -> bool:
    """Return True if execution is currently inside a ``with transforms.microbatch(...)`` block."""
    return bool(_ACTIVE_TOKENS)


# ---------------------------------------------------------------------------
# User-Facing Context Manager & Helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def microbatch(
    min_len: int = 1024,
    groups: int = 2,
    *,
    split_dim: int = 0,
    barrier: bool = True,
    min_chunk_multiple: int = 8,
) -> Iterator[None]:
    """Context manager marking a region for automatic token microbatching.

    Parameters
    ----------
    min_len:
        Minimum token bucket size at which microbatching activates (inclusive).
        Buckets with ``size < min_len`` run the unsplit region as-is.
    groups:
        Number of microbatch chunks ``G`` to split the token dimension into when
        ``size >= min_len``. If ``groups <= 1``, this context manager is a no-op.
    split_dim:
        Tensor dimension corresponding to tokens (default ``0``).
    barrier:
        Whether to insert ``jax.lax.optimization_barrier`` pairs in a 2-stage
        diamond pipeline so ``Stage2(g)`` overlaps with ``Stage1(g + 1)``.
    min_chunk_multiple:
        Require ``(size // groups) % min_chunk_multiple == 0`` (default ``8``
        for TPU sublane tiling). Set to ``1`` to allow any divisible size.
    """
    if int(groups) <= 1:
        yield
        return

    token = torch.ops.tpu_transforms.microbatch_enter(
        int(min_len),
        int(groups),
        int(split_dim),
        bool(barrier),
        int(min_chunk_multiple),
    )
    _ACTIVE_TOKENS.append(token)
    try:
        yield
    finally:
        _ACTIVE_TOKENS.pop()
        torch.ops.tpu_transforms.microbatch_exit(token)


def stage_boundary() -> None:
    """Explicitly mark the boundary between Stage 1 and Stage 2 inside ``with transforms.microbatch(...)``."""
    if _ACTIVE_TOKENS:
        torch.ops.tpu_transforms.microbatch_stage_boundary(_ACTIVE_TOKENS[-1])


# ---------------------------------------------------------------------------
# FX Region Discovery & Classification
# ---------------------------------------------------------------------------


@dataclass
class _MicrobatchRegion:
    enter_node: fx.Node
    exit_node: fx.Node
    boundary_node: fx.Node | None
    body_nodes: list[fx.Node]
    min_len: int
    groups: int
    split_dim: int
    barrier: bool
    min_chunk_multiple: int


def _find_microbatch_regions(graph: fx.Graph) -> list[_MicrobatchRegion]:
    """Locate matched ``(microbatch_enter, microbatch_exit)`` regions in topological order."""
    regions: list[_MicrobatchRegion] = []
    stack: list[tuple[fx.Node, fx.Node | None, list[fx.Node]]] = []

    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in _ENTER_TARGETS:
            stack.append((node, None, []))
        elif node.op == "call_function" and node.target in _STAGE_BOUNDARY_TARGETS:
            if stack:
                enter_n, _, body = stack[-1]
                stack[-1] = (enter_n, node, body)
                body.append(node)
        elif node.op == "call_function" and node.target in _EXIT_TARGETS:
            if stack:
                enter_node, boundary_node, body_nodes = stack.pop()
                args = enter_node.args
                min_len = int(args[0]) if len(args) > 0 else 1024
                groups = int(args[1]) if len(args) > 1 else 2
                split_dim = int(args[2]) if len(args) > 2 else 0
                barrier = bool(args[3]) if len(args) > 3 else True
                min_chunk_multiple = int(args[4]) if len(args) > 4 else 8
                regions.append(
                    _MicrobatchRegion(
                        enter_node=enter_node,
                        exit_node=node,
                        boundary_node=boundary_node,
                        body_nodes=body_nodes,
                        min_len=min_len,
                        groups=groups,
                        split_dim=split_dim,
                        barrier=barrier,
                        min_chunk_multiple=min_chunk_multiple,
                    )
                )
        else:
            if stack:
                stack[-1][2].append(node)

    return regions


def _target_name(node: fx.Node) -> str:
    if node.op not in ("call_function", "call_method"):
        return ""
    return getattr(node.target, "__name__", str(node.target)).lower()


def _is_collective_start(node: fx.Node) -> bool:
    """True if ``node`` initiates a distributed collective communication op."""
    name = _target_name(node)
    if not name or "wait_tensor" in name:
        return False
    return any(
        kw in name
        for kw in (
            "all_gather",
            "reduce_scatter",
            "all_reduce",
            "all_to_all",
            "dcp_lse_ag",
            "dcp_reduce_scatter",
            "cp_lse_ag_out_rs",
        )
    )


def _is_wait_tensor(node: fx.Node) -> bool:
    return "wait_tensor" in _target_name(node)


def _is_heavy_compute_op(node: fx.Node) -> bool:
    """True if ``node`` is a matrix multiplication or Pallas/TPU compute kernel."""
    name = _target_name(node)
    if not name or _is_collective_start(node) or _is_wait_tensor(node):
        return False
    return any(
        kw in name
        for kw in (
            "mm",
            "bmm",
            "matmul",
            "linear",
            "addmm",
            "dot",
            "conv",
            "vspace_weighted",
            "lse_rescale",
            "fp8_linear",
            "tpu_matmul",
            "gmm",
            "ragged_dot",
        )
    )


def _partition_region_stages(
    body_nodes: list[fx.Node],
    boundary_node: fx.Node | None,
) -> tuple[list[fx.Node], list[fx.Node]]:
    """Split ``body_nodes`` into ``(stage1_nodes, stage2_nodes)`` for diamond pipelining.

    Supports both:
    - ``Compute -> Collective`` (and ``[LSE_AG +] Compute -> Collective``):
      ``Stage 1`` runs all nodes prior to the final collective start node;
      ``Stage 2`` runs the final collective, its ``wait_tensor``, and any
      trailing view/cast ops.
    - ``Collective -> Compute`` (e.g. ``all_gather -> gate_up_proj``):
      ``Stage 1`` runs the initial collective chain (through ``wait_tensor`` and
      any reshape/view before the first heavy compute op); ``Stage 2`` runs the
      heavy compute op and everything after it.
    """
    if not body_nodes:
        return [], []

    if boundary_node is not None and boundary_node in body_nodes:
        idx = body_nodes.index(boundary_node)
        stage1 = body_nodes[:idx]
        stage2 = body_nodes[idx + 1 :]
        if stage1 and stage2:
            return stage1, stage2
        body_nodes = stage1 + stage2

    coll_indices = [i for i, n in enumerate(body_nodes) if _is_collective_start(n)]
    compute_indices = [i for i, n in enumerate(body_nodes) if _is_heavy_compute_op(n)]

    if coll_indices and compute_indices:
        last_coll_idx = coll_indices[-1]
        first_coll_idx = coll_indices[0]
        # Case 1: Compute occurs before the last collective (Compute -> Collective,
        # or LSE_AG -> Compute -> ReduceScatter).
        if any(c_idx < last_coll_idx for c_idx in compute_indices):
            return body_nodes[:last_coll_idx], body_nodes[last_coll_idx:]

        # Case 2: Collective occurs before Compute (Collective -> Compute,
        # e.g. all_gather -> gate_up_proj).
        first_compute_after_coll = next(
            (c_idx for c_idx in compute_indices if c_idx > first_coll_idx), None
        )
        if first_compute_after_coll is not None:
            return (
                body_nodes[:first_compute_after_coll],
                body_nodes[first_compute_after_coll:],
            )

    # Fallback: if there is at least one collective and arithmetic ops (e.g.
    # logsumexp / exp / mul / where -> reduce_scatter), split right before the
    # last collective or right after the first collective's wait_tensor.
    if coll_indices:
        last_coll_idx = coll_indices[-1]
        if last_coll_idx > 0:
            return body_nodes[:last_coll_idx], body_nodes[last_coll_idx:]
        wait_idx = next(
            (i for i, n in enumerate(body_nodes) if _is_wait_tensor(n)), 0
        )
        split_at = min(wait_idx + 1, len(body_nodes))
        return body_nodes[:split_at], body_nodes[split_at:]

    return body_nodes, []


def _node_tensor_val(node: fx.Node) -> torch.Tensor | None:
    val = node.meta.get("example_value", node.meta.get("val"))
    if isinstance(val, torch.Tensor):
        return val
    return None


_PASSTHROUGH_VIEW_CAST_OPS = (
    "to",
    "contiguous",
    "transpose",
    "t",
    "permute",
    "view",
    "reshape",
    "_unsafe_view",
    "unsqueeze",
    "squeeze",
    "expand",
)


def _is_weight_only_input(node: fx.Node, region_set: set[fx.Node]) -> bool:
    """True when ``node`` is a model parameter/weight/scale/cache rather than a token activation."""
    if node.op == "get_attr":
        return True
    lower_name = node.name.lower()
    if lower_name.startswith("l_self_") or any(
        kw in lower_name
        for kw in (
            "weight",
            "scale",
            "bias",
            "cache",
            "table",
            "w_uv",
            "w_uk",
            "rank",
            "padding_mask",
        )
    ):
        return True
    val = _node_tensor_val(node)
    if isinstance(val, torch.nn.Parameter):
        return True

    # If every user in `region_set` only reads element 0 (`select(node, 0, 0)`,
    # e.g. `small_all_reduce_op.retain`), treat `node` as a metadata buffer.
    in_region_users = [u for u in node.users if u in region_set]
    if in_region_users and all(
        "select" in _target_name(u) and len(u.args) >= 3 and u.args[1] == 0 and u.args[2] == 0
        for u in in_region_users
    ):
        return True

    # Trace through view/cast ops inside `region_set` to see how `node` is consumed.
    frontier = [
        (node, u) for u in node.users if u in region_set
    ]
    if not frontier:
        return False
    saw_compute_rhs = False
    while frontier:
        src, user = frontier.pop()
        name = _target_name(user)
        if "vspace_weighted" in name:
            if len(user.args) >= 3 and src in user.args[2:]:
                saw_compute_rhs = True
                continue
            return False
        if "addmm" in name:
            if len(user.args) >= 3 and src is not user.args[1]:
                saw_compute_rhs = True
                continue
            return False
        if any(
            kw in name
            for kw in (
                "mm",
                "bmm",
                "linear",
                "matmul",
                "fp8_linear",
                "tpu_matmul",
            )
        ):
            if user.args and user.args[0] is not src:
                saw_compute_rhs = True
                continue
            return False
        if name in _PASSTHROUGH_VIEW_CAST_OPS:
            for next_u in user.users:
                if next_u in region_set:
                    frontier.append((user, next_u))
            continue
        return False
    return saw_compute_rhs


def _sym_or_int_key(v: Any) -> tuple[str, Any]:
    """Return a guard-free hashable key for a ``SymInt`` or ``int`` dimension."""
    if isinstance(v, torch.SymInt):
        node = getattr(v, "node", None)
        expr = getattr(node, "expr", None)
        return ("sym", str(expr if expr is not None else v))
    return ("int", int(v))


def _resolve_region_sliced_live_ins(
    live_in_nodes: list[fx.Node],
    region_set: set[fx.Node],
    split_dim: int,
    size: int | None,
) -> tuple[list[fx.Node], set[tuple[str, Any]]]:
    """Return ``(sliced_live_ins, token_sym_keys)`` for the region."""
    sym_candidates: list[tuple[fx.Node, Any]] = []
    int_candidates: list[fx.Node] = []
    for inp in live_in_nodes:
        if _is_weight_only_input(inp, region_set):
            continue
        val = _node_tensor_val(inp)
        if val is None or val.ndim <= split_dim:
            continue
        dim_val = val.shape[split_dim]
        if isinstance(dim_val, torch.SymInt):
            sym_candidates.append((inp, dim_val))
        elif size is not None and int(dim_val) == int(size):
            int_candidates.append(inp)

    if sym_candidates:
        # In dynamically traced FX graphs, token activations carry SymInt at `split_dim`
        # while static weights/buffers carry concrete ints.
        sliced = [inp for inp, _ in sym_candidates]
        sym_keys = {_sym_or_int_key(dim_val) for _, dim_val in sym_candidates}
        return sliced, sym_keys

    return int_candidates, set()


def _find_matching_token_symint_placeholders(
    graph: fx.Graph, token_sym_keys: set[tuple[str, Any]]
) -> set[fx.Node]:
    """Identify scalar SymInt placeholder nodes matching ``token_sym_keys``."""
    sym_nodes: set[fx.Node] = set()
    if not token_sym_keys:
        return sym_nodes
    for node in graph.nodes:
        if node.op != "placeholder":
            break
        val = node.meta.get("example_value", node.meta.get("val"))
        if isinstance(val, torch.SymInt) and _sym_or_int_key(val) in token_sym_keys:
            sym_nodes.add(node)
    return sym_nodes


def _select_barrier1_inputs(
    stage1_sliced_ins: list[fx.Node], stage1_set: set[fx.Node]
) -> list[fx.Node]:
    """Pick the minimal subset of ``stage1_sliced_ins`` needed to gate all Stage 1 ops."""
    if len(stage1_sliced_ins) <= 1:
        return stage1_sliced_ins

    def _non_view_stage1_descendants(root: fx.Node) -> set[fx.Node]:
        visited: set[fx.Node] = set()
        stack = [root]
        while stack:
            cur = stack.pop()
            for u in cur.users:
                if u in stage1_set and u not in visited:
                    visited.add(u)
                    stack.append(u)
        return {
            n
            for n in visited
            if _target_name(n) not in _PASSTHROUGH_VIEW_CAST_OPS
        }

    inp_with_desc = [
        (inp, _non_view_stage1_descendants(inp)) for inp in stage1_sliced_ins
    ]
    inp_with_desc.sort(key=lambda item: -len(item[1]))

    selected: list[fx.Node] = []
    covered: set[fx.Node] = set()
    for inp, desc in inp_with_desc:
        if not selected or not desc.issubset(covered):
            selected.append(inp)
            covered |= desc
    return selected


def _adjust_view_reshape_args(
    orig_node: fx.Node,
    mapped_args: tuple[Any, ...],
    groups: int,
    size: int,
) -> tuple[Any, ...]:
    """Update hardcoded concrete token dimensions in view/reshape ops when needed."""
    if orig_node.op not in ("call_function", "call_method"):
        return mapped_args
    name = _target_name(orig_node)
    if not any(kw in name for kw in ("view", "reshape", "_unsafe_view")):
        return mapped_args
    if len(mapped_args) < 2:
        return mapped_args

    # Handle both `view(x, [d0, d1, ...])` and `view(x, d0, d1, ...)`.
    shape_arg = mapped_args[1]
    if isinstance(shape_arg, (list, tuple)) and shape_arg:
        first_dim = shape_arg[0]
        if isinstance(first_dim, int) and first_dim > 0 and first_dim % size == 0:
            new_shape = list(shape_arg)
            new_shape[0] = first_dim // groups
            return (mapped_args[0], type(shape_arg)(new_shape), *mapped_args[2:])
    elif isinstance(shape_arg, int) and shape_arg > 0 and shape_arg % size == 0:
        return (mapped_args[0], shape_arg // groups, *mapped_args[2:])
    return mapped_args


def _emit_barrier_2(
    graph: fx.Graph,
    a: fx.Node,
    b: fx.Node,
    *,
    use_first_output: bool = True,
) -> tuple[fx.Node, fx.Node]:
    """Emit ``a_bar, b_bar = pallas::microbatch_barrier_2(a, b)`` in ``graph``."""
    barrier_op = _select_barrier_op_for_node(a)
    barrier_call = graph.call_function(barrier_op, (a, b))
    if use_first_output:
        a_bar = graph.call_function(operator.getitem, (barrier_call, 0))
        if "val" in a.meta:
            a_bar.meta["val"] = a.meta["val"]
        if "example_value" in a.meta:
            a_bar.meta["example_value"] = a.meta["example_value"]
    else:
        a_bar = a
    b_bar = graph.call_function(operator.getitem, (barrier_call, 1))
    if "val" in b.meta:
        b_bar.meta["val"] = b.meta["val"]
    if "example_value" in b.meta:
        b_bar.meta["example_value"] = b.meta["example_value"]
    return a_bar, b_bar


def _erase_marker_nodes(region: _MicrobatchRegion, graph: fx.Graph) -> None:
    if region.exit_node in graph.nodes:
        graph.erase_node(region.exit_node)
    if region.boundary_node is not None and region.boundary_node in graph.nodes:
        graph.erase_node(region.boundary_node)
    if region.enter_node in graph.nodes:
        graph.erase_node(region.enter_node)


def _rewrite_single_region(
    graph: fx.Graph,
    region: _MicrobatchRegion,
    size: int | None,
) -> bool:
    """Rewrite one ``_MicrobatchRegion`` in ``graph``. Returns True if microbatched."""
    groups = region.groups
    min_len = region.min_len
    split_dim = region.split_dim
    min_mult = max(region.min_chunk_multiple, 1)

    if (
        groups <= 1
        or size is None
        or size < min_len
        or size % (groups * min_mult) != 0
        or not region.body_nodes
    ):
        _erase_marker_nodes(region, graph)
        return False

    body_nodes = [
        n for n in region.body_nodes if n is not region.boundary_node
    ]
    region_set = set(body_nodes)

    # Identify live-in and live-out nodes of the region.
    live_in_nodes: list[fx.Node] = []
    seen_in: set[fx.Node] = set()
    for n in body_nodes:
        for inp in n.all_input_nodes:
            if inp not in region_set and inp not in seen_in:
                seen_in.add(inp)
                live_in_nodes.append(inp)

    live_out_nodes: list[fx.Node] = []
    for n in body_nodes:
        if any(
            user not in region_set
            and user is not region.exit_node
            and user is not region.boundary_node
            for user in n.users
        ):
            live_out_nodes.append(n)

    sliced_live_ins, token_dim_expr = _resolve_region_sliced_live_ins(
        live_in_nodes, region_set, split_dim, size
    )
    if not sliced_live_ins or not live_out_nodes:
        _erase_marker_nodes(region, graph)
        return False

    token_sym_nodes = _find_matching_token_symint_placeholders(
        graph, token_dim_expr
    )

    stage1_nodes, stage2_nodes = _partition_region_stages(
        region.body_nodes, region.boundary_node
    )
    stage1_set = set(stage1_nodes)
    stage2_set = set(stage2_nodes)

    # Identify intermediate tensors produced in Stage 1 that are consumed by Stage 2
    # or are live-out of the region.
    stage1_to_stage2_nodes: list[fx.Node] = [
        n
        for n in stage1_nodes
        if any(u in stage2_set for u in n.users) or n in live_out_nodes
    ]

    # Which sliced live-in tensors are directly consumed by Stage 1?
    stage1_sliced_ins = [
        inp
        for inp in sliced_live_ins
        if any(u in stage1_set for u in inp.users)
    ]

    step = size // groups

    with graph.inserting_before(region.enter_node):
        # Map any dynamic SymInt token placeholders to `sym // groups` inside the region.
        sym_div_map: dict[fx.Node, fx.Node] = {}
        for sym_n in token_sym_nodes:
            if any(u in region_set for u in sym_n.users):
                sym_div_map[sym_n] = graph.call_function(
                    operator.floordiv, (sym_n, groups)
                )

        # Slice each token-indexed live-in tensor into `groups` chunks along `split_dim`.
        sliced_chunks: dict[fx.Node, list[fx.Node]] = {}
        for inp in sliced_live_ins:
            chunks_for_inp: list[fx.Node] = []
            for g in range(groups):
                slc = graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    (inp, split_dim, g * step, (g + 1) * step, 1),
                )
                if "val" in inp.meta:
                    slc.meta["val"] = inp.meta["val"]
                if "example_value" in inp.meta:
                    slc.meta["example_value"] = inp.meta["example_value"]
                chunks_for_inp.append(slc)
            sliced_chunks[inp] = chunks_for_inp

        env_per_group: list[dict[fx.Node, fx.Node]] = []
        for g in range(groups):
            env_g: dict[fx.Node, fx.Node] = dict(sym_div_map)
            for inp, chunks_for_inp in sliced_chunks.items():
                env_g[inp] = chunks_for_inp[g]
            env_per_group.append(env_g)

        def _clone_stage(stage_nodes: list[fx.Node], g: int) -> None:
            env_g = env_per_group[g]
            for orig_n in stage_nodes:
                new_n = graph.node_copy(
                    orig_n, lambda x, _env=env_g: _env.get(x, x)
                )
                new_n.args = _adjust_view_reshape_args(
                    orig_n, new_n.args, groups, size
                )
                env_g[orig_n] = new_n

        use_diamond = (
            region.barrier
            and bool(stage1_nodes)
            and bool(stage2_nodes)
            and bool(stage1_to_stage2_nodes)
            and bool(stage1_sliced_ins)
        )

        if not use_diamond:
            for g in range(groups):
                _clone_stage(body_nodes, g)
        else:
            # Prologue: Stage 1 for microbatch 0.
            _clone_stage(stage1_nodes, 0)

            # Diamond pipeline across groups g = 0 .. groups - 2:
            #   Barrier 1: gate Stage1(g+1) inputs on Stage1(g) completion
            #              without tying Stage2(g) to Stage1(g+1) inputs
            #   Parallel diamond: Stage2(g) || Stage1(g+1)
            #   Barrier 2: synchronize Stage2(g) output with Stage1(g+1) output
            barrier1_ins = _select_barrier1_inputs(
                stage1_sliced_ins, stage1_set
            )
            for g in range(groups - 1):
                mid_g_lead = env_per_group[g][stage1_to_stage2_nodes[-1]]
                for inp in barrier1_ins:
                    in_next = env_per_group[g + 1][inp]
                    _, in_next_bar = _emit_barrier_2(
                        graph, mid_g_lead, in_next, use_first_output=False
                    )
                    env_per_group[g + 1][inp] = in_next_bar

                # Emit Stage 2(g) and Stage 1(g + 1) with zero data dependency
                # between them so XLA's LatencyHidingScheduler overlaps them.
                _clone_stage(stage2_nodes, g)
                _clone_stage(stage1_nodes, g + 1)

                out_g_lead = env_per_group[g][live_out_nodes[-1]]
                mid_next_lead = env_per_group[g + 1][stage1_to_stage2_nodes[-1]]
                out_g_bar, mid_next_bar = _emit_barrier_2(
                    graph, out_g_lead, mid_next_lead, use_first_output=True
                )
                env_per_group[g][live_out_nodes[-1]] = out_g_bar
                env_per_group[g + 1][stage1_to_stage2_nodes[-1]] = mid_next_bar

            # Epilogue: Stage 2 for microbatch (groups - 1).
            _clone_stage(stage2_nodes, groups - 1)

        # Concatenate each live-out tensor across all groups along `split_dim`.
        for out_n in live_out_nodes:
            pieces = [env_per_group[g][out_n] for g in range(groups)]
            cat_node = graph.call_function(
                torch.ops.aten.cat.default, (pieces, split_dim)
            )
            if "val" in out_n.meta:
                cat_node.meta["val"] = out_n.meta["val"]
            if "example_value" in out_n.meta:
                cat_node.meta["example_value"] = out_n.meta["example_value"]
            out_n.replace_all_uses_with(cat_node)

    # Erase original region nodes in reverse topological order, then erase markers.
    for orig_n in reversed(body_nodes):
        if not orig_n.users:
            graph.erase_node(orig_n)
    _erase_marker_nodes(region, graph)
    return True


def apply_microbatch_pass(
    graph_module: fx.GraphModule,
    example_inputs: list[Any] | tuple[Any, ...],
    size: int | None,
) -> int:
    """Rewrite all ``transforms.microbatch`` regions in ``graph_module`` for bucket ``size``.

    Returns the number of regions that were rewritten into microbatch pipelines.
    """
    del example_inputs  # Shape metadata is carried on FX nodes and `size`.
    graph = graph_module.graph
    regions = _find_microbatch_regions(graph)
    if not regions:
        return 0

    rewritten_count = 0
    for region in regions:
        if _rewrite_single_region(graph, region, size):
            rewritten_count += 1

    graph.lint()
    graph_module.recompile()
    if rewritten_count > 0:
        logger.debug(
            "transforms.microbatch rewrote %d region(s) for token bucket size=%s",
            rewritten_count,
            size,
        )
    return rewritten_count
