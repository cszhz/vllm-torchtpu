# SPDX-License-Identifier: Apache-2.0
"""Tests for ``transforms.microbatch`` context manager and FX region rewriter pass."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.fx as fx
from torch import nn

from vllm_torchtpu import envs, transforms
from vllm_torchtpu.layers.adapter import cp_mla_attention


# ---------------------------------------------------------------------------
# Mock Collective & Compute Custom Ops for Deterministic FX & Eager Testing
# ---------------------------------------------------------------------------


@torch.library.custom_op("tpu_transforms_test::all_gather_dim0", mutates_args=())
def _mock_all_gather_dim0(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0


@_mock_all_gather_dim0.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("tpu_transforms_test::all_reduce", mutates_args=())
def _mock_all_reduce(x: torch.Tensor) -> torch.Tensor:
    return x * 1.5


@_mock_all_reduce.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "tpu_transforms_test::reduce_scatter_heads", mutates_args=()
)
def _mock_reduce_scatter_heads(
    x: torch.Tensor, world_size: int
) -> torch.Tensor:
    # x is heads-major [H, T, V] -> returns [H // world_size, T, V]
    h = x.shape[0] // world_size
    pieces = [x[r * h : (r + 1) * h] for r in range(world_size)]
    return sum(pieces)


@_mock_reduce_scatter_heads.register_fake
def _(x: torch.Tensor, world_size: int) -> torch.Tensor:
    h, t, v = x.shape
    return x.new_empty((h // world_size, t, v))


@torch.library.custom_op("tpu_transforms_test::wait_tensor", mutates_args=())
def _mock_wait_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_mock_wait_tensor.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def tensor_model_parallel_all_gather(
    x_shard: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    del dim
    out = torch.ops.tpu_transforms_test.all_gather_dim0(x_shard)
    return torch.ops.tpu_transforms_test.wait_tensor(out)


def tensor_model_parallel_all_reduce(x: torch.Tensor) -> torch.Tensor:
    out = torch.ops.tpu_transforms_test.all_reduce(x)
    return torch.ops.tpu_transforms_test.wait_tensor(out)


class _MockGateUpProj(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return torch.nn.functional.linear(x, self.weight), None


class _MockRowParallelOProj(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        partial = torch.nn.functional.linear(x, self.weight)
        out = tensor_model_parallel_all_reduce(partial)
        return out, None


class _MockDcpGroup:
    def __init__(self, world_size: int = 2) -> None:
        self.world_size = world_size

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        del dim
        # Repeat x `[1, T, H]` -> `[world_size, T, H]` via a custom collective op.
        gathered = torch.ops.tpu_transforms_test.all_gather_dim0(
            x.expand(self.world_size, -1, -1).contiguous()
        )
        return torch.ops.tpu_transforms_test.wait_tensor(gathered)

    def reduce_scatter(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if dim == 0:
            out = torch.ops.tpu_transforms_test.reduce_scatter_heads(
                x, self.world_size
            )
            return torch.ops.tpu_transforms_test.wait_tensor(out)
        # dim == 1: [T, H, V] -> transpose to [H, T, V], reduce_scatter, transpose back
        out = torch.ops.tpu_transforms_test.reduce_scatter_heads(
            x.transpose(0, 1).contiguous(), self.world_size
        )
        return torch.ops.tpu_transforms_test.wait_tensor(out).transpose(0, 1)


def _capture_dynamic_fx_graph(
    fn: Any, *example_args: torch.Tensor
) -> tuple[fx.GraphModule, list[Any]]:
    """Capture a single dynamic-token FX graph via ``torch.compile``."""
    torch._dynamo.reset()
    captured: list[tuple[fx.GraphModule, list[Any]]] = []

    def _capture_backend(gm: fx.GraphModule, example_inputs: list[Any]):
        captured.append((copy.deepcopy(gm), list(example_inputs)))
        return gm.forward

    for arg in example_args:
        if isinstance(arg, torch.Tensor) and arg.ndim > 0:
            torch._dynamo.mark_dynamic(arg, 0)

    compiled = torch.compile(fn, backend=_capture_backend)
    compiled(*example_args)
    assert len(captured) == 1, f"Expected 1 captured FX graph, got {len(captured)}"
    return captured[0]


def _count_targets(gm: fx.GraphModule, substring: str) -> int:
    count = 0
    for n in gm.graph.nodes:
        if n.op in ("call_function", "call_method"):
            name = getattr(n.target, "__name__", str(n.target)).lower()
            if substring in name:
                count += 1
    return count


# ---------------------------------------------------------------------------
# 1. Collective -> Compute: AllGather -> gate_up_proj
# ---------------------------------------------------------------------------


def test_collective_to_compute_all_gather_gate_up_proj() -> None:
    torch.manual_seed(0)

    class AllGatherMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = _MockGateUpProj(16, 32)

        def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
            with transforms.microbatch(min_len=1024, groups=2):
                x_full = tensor_model_parallel_all_gather(x_shard, dim=0)
                gate_up, _ = self.gate_up_proj(x_full)
            return gate_up + 1.0

    model = AllGatherMLP().eval()
    gm_orig, example_inputs = _capture_dynamic_fx_graph(
        model, torch.randn(64, 16)
    )

    # Bucket 512 (< min_len=1024): markers stripped, unsplit graph.
    gm_512 = copy.deepcopy(gm_orig)
    rewritten_512 = transforms.apply_microbatch_pass(
        gm_512, example_inputs, size=512
    )
    assert rewritten_512 == 0
    assert _count_targets(gm_512, "microbatch_enter") == 0
    assert _count_targets(gm_512, "microbatch_exit") == 0
    assert _count_targets(gm_512, "all_gather_dim0") == 1
    assert _count_targets(gm_512, "linear") == 1
    assert _count_targets(gm_512, "microbatch_barrier_2") == 0

    x_512 = torch.randn(512, 16)
    ref_512 = (tensor_model_parallel_all_gather(x_512, dim=0) @ model.gate_up_proj.weight.T) + 1.0
    out_512 = model(x_512)
    torch.testing.assert_close(out_512, ref_512)

    # Bucket 1024 (>= min_len=1024): rewritten into 2-stage diamond pipeline!
    gm_1024 = copy.deepcopy(gm_orig)
    rewritten_1024 = transforms.apply_microbatch_pass(
        gm_1024, example_inputs, size=1024
    )
    assert rewritten_1024 == 1
    assert _count_targets(gm_1024, "microbatch_enter") == 0
    assert _count_targets(gm_1024, "microbatch_exit") == 0
    assert _count_targets(gm_1024, "slice") == 2
    assert _count_targets(gm_1024, "all_gather_dim0") == 2
    assert _count_targets(gm_1024, "linear") == 2
    assert _count_targets(gm_1024, "microbatch_barrier_2") == 2
    assert _count_targets(gm_1024, "cat") == 1

    # Verify topological order of the diamond:
    # Stage1(0) [all_gather_0] -> Barrier1 -> Stage2(0) [linear_0] -> Stage1(1) [all_gather_1] -> Barrier2 -> Stage2(1) [linear_1]
    op_sequence = [
        getattr(n.target, "__name__", str(n.target)).lower()
        for n in gm_1024.graph.nodes
        if n.op == "call_function"
        and any(
            k in getattr(n.target, "__name__", str(n.target)).lower()
            for k in ("all_gather_dim0", "linear", "microbatch_barrier_2", "cat")
        )
    ]
    assert len(op_sequence) == 7
    assert "all_gather_dim0" in op_sequence[0]
    assert "microbatch_barrier_2" in op_sequence[1]
    assert "linear" in op_sequence[2]
    assert "all_gather_dim0" in op_sequence[3]
    assert "microbatch_barrier_2" in op_sequence[4]
    assert "linear" in op_sequence[5]
    assert "cat" in op_sequence[6]

    # Verify numerical equivalence on concrete 1024-token input!
    x_1024 = torch.randn(1024, 16)
    ref_1024 = (tensor_model_parallel_all_gather(x_1024, dim=0) @ model.gate_up_proj.weight.T) + 1.0
    # Execute the rewritten FX GraphModule directly with concrete args
    weight_param = model.gate_up_proj.weight
    # Find placeholder order in gm_1024
    phs = [n for n in gm_1024.graph.nodes if n.op == "placeholder"]
    call_args = []
    for ph in phs:
        val = ph.meta.get("example_value")
        if isinstance(val, torch.SymInt):
            call_args.append(1024)
        elif "weight" in ph.name:
            call_args.append(weight_param)
        else:
            call_args.append(x_1024)
    out_1024 = gm_1024(*call_args)
    if isinstance(out_1024, tuple):
        out_1024 = out_1024[0]
    torch.testing.assert_close(out_1024, ref_1024)


# ---------------------------------------------------------------------------
# 2. Compute -> Collective: GLM-5.2 o_proj (GEMM -> AllReduce)
# ---------------------------------------------------------------------------


def test_compute_to_collective_o_proj_all_reduce() -> None:
    torch.manual_seed(1)

    class OProjBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = _MockRowParallelOProj(16, 32)

        def forward(self, attn_out: torch.Tensor) -> torch.Tensor:
            with transforms.microbatch(min_len=1024, groups=2):
                out, _ = self.o_proj(attn_out)
            return out

    model = OProjBlock().eval()
    gm_orig, example_inputs = _capture_dynamic_fx_graph(
        model, torch.randn(64, 16)
    )

    gm_1024 = copy.deepcopy(gm_orig)
    rewritten = transforms.apply_microbatch_pass(
        gm_1024, example_inputs, size=1024
    )
    assert rewritten == 1

    # Verify topological order of Compute -> Collective diamond:
    # Stage1(0) [linear_0] -> Barrier1 -> Stage2(0) [all_reduce_0] -> Stage1(1) [linear_1] -> Barrier2 -> Stage2(1) [all_reduce_1] -> cat
    op_sequence = [
        getattr(n.target, "__name__", str(n.target)).lower()
        for n in gm_1024.graph.nodes
        if n.op == "call_function"
        and any(
            k in getattr(n.target, "__name__", str(n.target)).lower()
            for k in ("all_reduce", "linear", "microbatch_barrier_2", "cat")
        )
    ]
    assert len(op_sequence) == 7
    assert "linear" in op_sequence[0]
    assert "microbatch_barrier_2" in op_sequence[1]
    assert "all_reduce" in op_sequence[2]
    assert "linear" in op_sequence[3]
    assert "microbatch_barrier_2" in op_sequence[4]
    assert "all_reduce" in op_sequence[5]
    assert "cat" in op_sequence[6]

    x_1024 = torch.randn(1024, 16)
    ref_1024 = tensor_model_parallel_all_reduce(x_1024 @ model.o_proj.weight.T)
    phs = [n for n in gm_1024.graph.nodes if n.op == "placeholder"]
    call_args = [
        1024
        if isinstance(ph.meta.get("example_value"), torch.SymInt)
        else (model.o_proj.weight if "weight" in ph.name else x_1024)
        for ph in phs
    ]
    out_1024 = gm_1024(*call_args)
    if isinstance(out_1024, tuple):
        out_1024 = out_1024[0]
    torch.testing.assert_close(out_1024, ref_1024)


# ---------------------------------------------------------------------------
# 3. GLM-5.2 DCP V-Space Merge via transforms.microbatch (groups=2 and groups=8)
# ---------------------------------------------------------------------------


def test_glm52_cp_mla_attention_vspace_merge_with_transforms(monkeypatch) -> None:
    torch.manual_seed(2)
    dcp_group = _MockDcpGroup(world_size=2)
    monkeypatch.setattr(cp_mla_attention, "get_dcp_group", lambda: dcp_group)
    monkeypatch.setattr(envs, "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS", 1024)
    monkeypatch.setattr(envs, "TPU_DCP_MERGE_CHUNKED_LSE_AG", False)
    monkeypatch.setattr(envs, "TPU_DCP_MLA_MERGE_PIECES", 0)

    num_heads = 8
    kv_lora_rank = 16
    v_head_dim = 12
    w_uv = torch.randn(num_heads, kv_lora_rank, v_head_dim)
    w_uv_scale = torch.rand(num_heads, 1, v_head_dim)

    def _fn_groups_4(
        partial: torch.Tensor, lse: torch.Tensor
    ) -> torch.Tensor:
        return cp_mla_attention._merge_chunked(
            partial,
            lse,
            dcp_group,
            chunks=4,
            rs_dtype=torch.bfloat16,
            vspace=(w_uv, w_uv_scale),
        )

    gm_orig, example_inputs = _capture_dynamic_fx_graph(
        _fn_groups_4,
        torch.randn(128, num_heads, kv_lora_rank, dtype=torch.bfloat16),
        torch.randn(128, num_heads, dtype=torch.float32),
    )

    gm_2048 = copy.deepcopy(gm_orig)
    rewritten = transforms.apply_microbatch_pass(
        gm_2048, example_inputs, size=2048
    )
    assert rewritten == 1
    # 2 sliced live-ins (`partial` and `lse_f32`) x 4 groups = 8 slice ops.
    # Weights `w_uv` and `w_uv_scale` must NOT be sliced!
    assert _count_targets(gm_2048, "slice") == 8
    assert _count_targets(gm_2048, "bmm") == 4
    assert _count_targets(gm_2048, "reduce_scatter_heads") == 4
    # For groups=4 and 2 sliced inputs feeding Stage 1:
    # Each of the 3 diamond transitions emits 2 Barrier-1 calls (for partial and lse)
    # + 1 Barrier-2 call (synchronizing Stage2(g) with Stage1(g+1)) = 9 barrier calls.
    assert _count_targets(gm_2048, "microbatch_barrier_2") == 9
    assert _count_targets(gm_2048, "cat.default") == 1

    # When TPU_DCP_MLA_MERGE_PIECES=2, barrier=False so XLA's SliceConcatForwarding
    # can fuse directly with the Pallas MLA pieces.
    monkeypatch.setattr(envs, "TPU_DCP_MLA_MERGE_PIECES", 2)
    gm_pieces, example_inputs_pieces = _capture_dynamic_fx_graph(
        _fn_groups_4,
        torch.randn(128, num_heads, kv_lora_rank, dtype=torch.bfloat16),
        torch.randn(128, num_heads, dtype=torch.float32),
    )
    rewritten_pieces = transforms.apply_microbatch_pass(
        gm_pieces, example_inputs_pieces, size=2048
    )
    assert rewritten_pieces == 1
    assert _count_targets(gm_pieces, "microbatch_barrier_2") == 0
    assert _count_targets(gm_pieces, "reduce_scatter_heads") == 4

    # Compare numerical output against the eager reference execution.
    partial_2048 = torch.randn(
        2048, num_heads, kv_lora_rank, dtype=torch.bfloat16
    )
    lse_2048 = torch.randn(2048, num_heads, dtype=torch.float32)

    ref_2048 = cp_mla_attention._merge_chunked(
        partial_2048,
        lse_2048,
        dcp_group,
        chunks=4,
        rs_dtype=torch.bfloat16,
        vspace=(w_uv, w_uv_scale),
    )

    phs = [n for n in gm_2048.graph.nodes if n.op == "placeholder"]
    call_args = []
    for ph in phs:
        val = ph.meta.get("example_value")
        if isinstance(val, torch.SymInt):
            call_args.append(2048)
        elif "partial" in ph.name:
            call_args.append(partial_2048)
        elif "lse" in ph.name:
            call_args.append(lse_2048)
        elif "scale" in ph.name:
            call_args.append(w_uv_scale)
        else:
            call_args.append(w_uv)

    out_2048 = gm_2048(*call_args)
    if isinstance(out_2048, tuple):
        out_2048 = out_2048[0]
    torch.testing.assert_close(out_2048, ref_2048)
