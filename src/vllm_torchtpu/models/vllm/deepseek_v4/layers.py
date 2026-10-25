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
"""DeepSeek-V4 feed-forward and Manifold-Harmonized Coupling helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig


# Pure PyTorch head projection for MHC manifold collapse;
# upstream vLLM only provides Triton/TileLang GPU kernels for hc_head.
def mhc_collapse_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Collapse multi-stream residual manifold into a single vector before LM head."""
    residual_flat = hidden_states.flatten(-2).float()
    residual_norm = residual_flat * torch.rsqrt(
        residual_flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps
    )
    pre_mix = torch.nn.functional.linear(residual_norm, hc_fn)
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    return torch.sum(pre_mix.unsqueeze(-1) * hidden_states.float(), dim=-2).to(
        hidden_states.dtype
    )


# The shared-MLP JAX ops of TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS, one per
# configuration rather than per layer (one exported program each).
_SHARED_OPS: dict[tuple, object] = {}


class DeepseekV4MLP(nn.Module):
    """DeepSeek-V4 feed-forward layer for dense/shared expert projections."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def arm_tp_slice(self, tp_rank: int, tp_size: int) -> bool:
        """Keep this rank's TP shard of a replicated (sequence-parallel) MLP's
        weights, for the buckets TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS runs on the
        replicated layout: there the shared expert is the TP-partial one the
        fused EP op reduces, as without sequence parallelism. Called once after
        weight processing, outside any trace. False (nothing kept) when the
        weight layout is not the canonical FP8 (k, n) one this slices."""
        from vllm_torchtpu.layers.adapter.linear_common import (
            FP8_PALLAS_ATTR,
            KEEP_VLLM_LAYOUT_ATTR,
        )
        from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

        gu, dn = self.gate_up_proj, self.down_proj
        if not hasattr(self.act_fn, "swiglu_limit"):
            return False
        if any(getattr(m, KEEP_VLLM_LAYOUT_ATTR, False) for m in (gu, dn)):
            return False
        w, ws = gu.weight, getattr(gu, "weight_scale", None)
        d, ds = dn.weight, getattr(dn, "weight_scale", None)
        if ws is None or ds is None or w.dim() != 2 or d.dim() != 2:
            return False
        n = w.shape[1] // 2
        if n % tp_size or d.shape[0] != n:
            return False
        s = n // tp_size

        def cols(t: torch.Tensor) -> torch.Tensor:
            dim = t.dim() - 1
            return torch.cat(
                [t.narrow(dim, tp_rank * s, s), t.narrow(dim, n + tp_rank * s, s)],
                dim,
            )

        if ws.dim() not in (1, 4) or ws.shape[-1] != 2 * n:
            return False
        if ds.dim() == 4:
            blocks = ds.shape[1]
            if n % blocks or s % (n // blocks):
                return False
            blk = n // blocks
            ds_tp = ds.narrow(1, tp_rank * s // blk, s // blk)
        elif ds.dim() == 1:
            ds_tp = ds  # per output channel: the K split leaves it whole
        else:
            return False
        tensors = [
            fresh_buffer(cols(w)),
            fresh_buffer(cols(ws)),
            fresh_buffer(d.narrow(0, tp_rank * s, s)),
            fresh_buffer(ds_tp),
        ]
        synchronize_tensors(tensors)
        pins = (getattr(gu, FP8_PALLAS_ATTR, None), getattr(dn, FP8_PALLAS_ATTR, None))
        object.__setattr__(self, "_tp_slice", (tensors, pins))
        self._build_shared_ops(pins)
        return True

    def _shared_op(self, replicated: bool):
        """The MLP as one JAX op over BOTH weight sets -- the replicated
        weights and this rank's TP slice -- using one of them. Each side of the
        TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS shape branch is its own trace, and a
        bucket compiled by one trace is served through another, which needs the
        same graph inputs in the same order (compilation/shape_variants); both
        sides therefore read both sets. Built outside the trace (arm_tp_slice)."""
        return self.__dict__["_tp_slice_ops"][replicated]

    def _build_shared_ops(self, pins) -> None:
        import jax
        from torch_tpu._internal import pallas

        from vllm_torchtpu.kernels.megablox.gmm_v2 import silu_and_mul_with_clamp
        from vllm_torchtpu.layers.adapter.linear_common import (
            _fp8_linear_pallas,
            _quantized_matmul_jax,
        )

        limit = float(self.act_fn.swiglu_limit)
        pin_g, pin_d = pins

        def mm(a, w, sc, pin):
            if pin is None:
                return _quantized_matmul_jax(a, w, sc)
            return _fp8_linear_pallas(a, w, sc, hbm_pin=bool(pin))

        def make(replicated: bool):
            def fn(
                x: jax.Array,
                wf: jax.Array,
                sf: jax.Array,
                df: jax.Array,
                dsf: jax.Array,
                wt: jax.Array,
                st: jax.Array,
                dt: jax.Array,
                dst: jax.Array,
            ) -> jax.Array:
                w, sc, d, ds = (wt, st, dt, dst) if replicated else (wf, sf, df, dsf)
                h = mm(x, w, sc, pin_g)
                n = h.shape[1] // 2
                h = silu_and_mul_with_clamp(h[:, :n], h[:, n:], limit).astype(x.dtype)
                return mm(h, d, ds, pin_d)

            op = pallas.jax_op(
                f"pallas::dsv4_shared_mlp_{'tp' if replicated else 'full'}"
                f"_g{pin_g}_d{pin_d}_l{str(limit).replace('.', 'p')}",
                fn,
            )

            def fake(x, wf, sf, df, dsf, wt, st, dt, dst):
                return torch.empty(
                    x.shape[0], df.shape[1], dtype=x.dtype, device=x.device
                )

            op.register_fake(fake)
            return op

        ops = {}
        for replicated in (False, True):
            key = (replicated, pin_g, pin_d, limit)
            if key not in _SHARED_OPS:
                _SHARED_OPS[key] = make(replicated)
            ops[replicated] = _SHARED_OPS[key]
        object.__setattr__(self, "_tp_slice_ops", ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if "_tp_slice" in self.__dict__:
            from vllm_torchtpu.layers.adapter.fused_moe_ep import SP_GATE

            slices, _ = self.__dict__["_tp_slice"]
            gu, dn = self.gate_up_proj, self.down_proj
            return self._shared_op(SP_GATE["replicated"])(
                x, gu.weight, gu.weight_scale, dn.weight, dn.weight_scale, *slices
            )
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
