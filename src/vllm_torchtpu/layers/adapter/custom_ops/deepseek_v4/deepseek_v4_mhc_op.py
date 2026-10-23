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
"""DeepSeek-V4 mHC Pallas kernels.

Drop-in replacements for vLLM's ``mhc_pre_torch`` / ``mhc_post_torch``, plus
the fused seam op that runs one layer's post together with the next layer's
pre as a single kernel.
"""

import functools
from typing import Any, NamedTuple

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.deepseek_v4.mhc import (
    fused_post_pre_kernel,
    post_head_kernel,
    post_kernel,
    pre_kernel,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_pallas_op_mesh,
)

# Every mHC tensor is token-major, so its leading axis is the attention-DP
# token axis. As in the other DeepSeek-V4 ops, the mesh carries no such axis
# today, so it resolves to None.
ATTN_DATA_AXIS = None

_mhc_op_cache: dict[str, Any] = {}


def _name_float(value: float) -> str:
    """A float rendered so it is safe inside a torch op name."""
    return f"{value:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def _gate_suffix(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> str:
    """The gate constants are baked into the traced op, so they name it."""
    return (
        f"_r{_name_float(rms_eps)}_p{_name_float(hc_pre_eps)}"
        f"_s{_name_float(hc_sinkhorn_eps)}"
        f"_a{_name_float(hc_post_mult_value)}_i{sinkhorn_repeat}"
    )


# Module-level so `pallas.jax_op` can trace and register them as torch ops.
def _mhc_pre_jax(
    residual: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mix GEMM + stream collapse in Pallas, gates/Sinkhorn in XLA."""
    return pre_kernel.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )


def _mhc_pre_single_stream_jax(
    x: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``pre`` of hc_mult copies of one stream, reading the stream once."""
    return pre_kernel.mhc_pre_single_stream(
        x,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )


def _mhc_post_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
) -> jax.Array:
    """Recombine the sublayer output into the mHC residual streams."""
    return post_kernel.mhc_post(x, residual, post_layer_mix, comb_res_mix)


def _mhc_post_head_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    norm_weight: jax.Array,
    *,
    rms_eps: float,
    hc_eps: float,
) -> jax.Array:
    """The last layer's post, the hc head collapse and the model's final
    RMSNorm (weight ``norm_weight``, epsilon ``rms_eps``) as one kernel."""
    return post_head_kernel.mhc_post_head(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        norm_weight=norm_weight,
        norm_eps=rms_eps,
    )


def _mhc_post_head_shared_jax(
    x: jax.Array,
    shared: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    norm_weight: jax.Array,
    *,
    rms_eps: float,
    hc_eps: float,
    routed_scale: float,
) -> jax.Array:
    """``_mhc_post_head_jax`` on an MoE output given as (routed, shared)."""
    return post_head_kernel.mhc_post_head(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        shared=shared,
        routed_scale=routed_scale,
        norm_weight=norm_weight,
        norm_eps=rms_eps,
    )


def _mhc_fused_post_pre_shared_jax(
    x: jax.Array,
    shared: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    routed_scale: float,
) -> tuple[jax.Array, ...]:
    """``_mhc_fused_post_pre_jax`` on an MoE output given as (routed, shared)."""
    return fused_post_pre_kernel.mhc_fused_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        shared=shared,
        routed_scale=routed_scale,
        emit_sumsq=True,
    )


def _mhc_fused_post_pre_normed_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    norm_weight: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The seam op with the next sublayer's RMSNorm applied to its input."""
    return fused_post_pre_kernel.mhc_fused_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=rms_eps,
    )


def _mhc_fused_post_pre_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, ...]:
    """One layer's post and the next layer's pre as a single kernel; the
    layer input's squared row sum comes last, for the attention RMSNorm."""
    return fused_post_pre_kernel.mhc_fused_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        emit_sumsq=True,
    )


def _fake_pre_outputs(
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(post_mix, comb_mix, layer_input) placeholders for Dynamo tracing."""
    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    return (
        torch.empty(
            (*outer_shape, hc_mult, 1), dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            (*outer_shape, hc_mult, hc_mult),
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            (*outer_shape, hidden_size), dtype=residual.dtype, device=residual.device
        ),
    )


def _fake_mhc_pre(residual, fn, hc_scale, hc_base, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return _fake_pre_outputs(residual)


def _as_streams(x: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """A (..., hc_mult, hidden) stand-in for a single (..., hidden) stream."""
    hc_mult = fn.shape[-1] // x.shape[-1]
    return x.unsqueeze(-2).expand(*x.shape[:-1], hc_mult, x.shape[-1])


def _fake_mhc_pre_single_stream(x, fn, hc_scale, hc_base, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return _fake_pre_outputs(_as_streams(x, fn))


def _fake_mhc_post(x, residual, post_layer_mix, comb_res_mix, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return torch.empty_like(residual)


def _fake_mhc_post_head(x, residual, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return torch.empty(
        (*residual.shape[:-2], residual.shape[-1]),
        dtype=residual.dtype,
        device=residual.device,
    )


def _fake_mhc_fused_post_pre(
    x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, *args, **kwargs
):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    if residual.dim() == x.dim():
        residual = _as_streams(residual, fn)
    return (torch.empty_like(residual), *_fake_pre_outputs(residual))


def _fake_mhc_fused_post_pre_sumsq(x, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    outs = _fake_mhc_fused_post_pre(x, *args, **kwargs)
    return (*outs, torch.empty_like(outs[-1][..., :1], dtype=torch.float32))


def _fake_mhc_fused_post_pre_shared(x, shared, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return _fake_mhc_fused_post_pre_sumsq(x, *args, **kwargs)


def _fake_mhc_post_head_shared(x, shared, residual, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return _fake_mhc_post_head(x, residual)


def _mhc_pre_op(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    op_name = "pallas::deepseek_v4_mhc_pre_v1" + _gate_suffix(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            functools.partial(
                _mhc_pre_jax,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(),  # fn
                P(),  # hc_scale
                P(),  # hc_base
            ),
        )
        op.register_fake(_fake_mhc_pre)
        _mhc_op_cache[op_name] = op
    return op


def _mhc_pre_single_stream_op(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    op_name = "pallas::deepseek_v4_mhc_pre_single_stream_v1" + _gate_suffix(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            functools.partial(
                _mhc_pre_single_stream_jax,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                P(),  # fn
                P(),  # hc_scale
                P(),  # hc_base
            ),
        )
        op.register_fake(_fake_mhc_pre_single_stream)
        _mhc_op_cache[op_name] = op
    return op


def get_mhc_post_op():
    """The ``post`` op alone, for callers that settle a deferred post.

    ``post`` takes no gate constants, so this needs none. Same
    outside-the-trace requirement as ``get_mhc_ops``.
    """
    op_name = "pallas::deepseek_v4_mhc_post_v1"
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            _mhc_post_jax,
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(ATTN_DATA_AXIS, None, None),  # post_layer_mix
                P(ATTN_DATA_AXIS, None, None),  # comb_res_mix
            ),
        )
        op.register_fake(_fake_mhc_post)
        _mhc_op_cache[op_name] = op
    return op


def get_mhc_post_head_op(rms_eps: float, hc_eps: float):
    """``post`` fused with the model's hc head collapse and final RMSNorm,
    whose weight is the op's last input.

    Same outside-the-trace requirement as ``get_mhc_ops``.
    """
    op_name = (
        "pallas::deepseek_v4_mhc_post_head_v2"
        f"_r{_name_float(rms_eps)}_e{_name_float(hc_eps)}"
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            functools.partial(_mhc_post_head_jax, rms_eps=rms_eps, hc_eps=hc_eps),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(ATTN_DATA_AXIS, None, None),  # post_layer_mix
                P(ATTN_DATA_AXIS, None, None),  # comb_res_mix
                P(),  # hc_fn
                P(),  # hc_scale
                P(),  # hc_base
                P(),  # norm_weight
            ),
        )
        op.register_fake(_fake_mhc_post_head)
        _mhc_op_cache[op_name] = op
    return op


def get_mhc_post_head_shared_op(rms_eps: float, hc_eps: float, routed_scale: float):
    """``get_mhc_post_head_op`` taking the MoE output as (routed, shared).

    Same outside-the-trace requirement as ``get_mhc_ops``.
    """
    op_name = (
        "pallas::deepseek_v4_mhc_post_head_shared_v2"
        f"_r{_name_float(rms_eps)}_e{_name_float(hc_eps)}"
        f"_rs{_name_float(routed_scale)}"
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            functools.partial(
                _mhc_post_head_shared_jax,
                rms_eps=rms_eps,
                hc_eps=hc_eps,
                routed_scale=routed_scale,
            ),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                P(ATTN_DATA_AXIS, None),  # shared
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(ATTN_DATA_AXIS, None, None),  # post_layer_mix
                P(ATTN_DATA_AXIS, None, None),  # comb_res_mix
                P(),  # hc_fn
                P(),  # hc_scale
                P(),  # hc_base
                P(),  # norm_weight
            ),
        )
        op.register_fake(_fake_mhc_post_head_shared)
        _mhc_op_cache[op_name] = op
    return op


def _mhc_fused_post_pre_shared_op(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    routed_scale: float,
):
    op_name = (
        "pallas::deepseek_v4_mhc_fused_post_pre_shared_v2"
        + _gate_suffix(
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
        )
        + f"_rs{_name_float(routed_scale)}"
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            functools.partial(
                _mhc_fused_post_pre_shared_jax,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                routed_scale=routed_scale,
            ),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                P(ATTN_DATA_AXIS, None),  # shared
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(ATTN_DATA_AXIS, None, None),  # post_layer_mix
                P(ATTN_DATA_AXIS, None, None),  # comb_res_mix
                P(),  # fn
                P(),  # hc_scale
                P(),  # hc_base
            ),
        )
        op.register_fake(_fake_mhc_fused_post_pre_shared)
        _mhc_op_cache[op_name] = op
    return op


def _mhc_fused_post_pre_op(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    single_stream_residual: bool = False,
    normed: bool = False,
):
    variant = ("_res1" if single_stream_residual else "") + (
        "_normed_v1" if normed else "_v2"
    )
    op_name = f"pallas::deepseek_v4_mhc_fused_post_pre{variant}" + _gate_suffix(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
    )
    residual_spec = (
        P(ATTN_DATA_AXIS, None)
        if single_stream_residual
        else P(ATTN_DATA_AXIS, None, None)
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            functools.partial(
                _mhc_fused_post_pre_normed_jax if normed else _mhc_fused_post_pre_jax,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                residual_spec,  # residual
                P(ATTN_DATA_AXIS, None, None),  # post_layer_mix
                P(ATTN_DATA_AXIS, None, None),  # comb_res_mix
                P(),  # fn
                P(),  # hc_scale
                P(),  # hc_base
                *([P()] if normed else []),  # norm_weight
            ),
        )
        op.register_fake(
            _fake_mhc_fused_post_pre if normed else _fake_mhc_fused_post_pre_sumsq
        )
        _mhc_op_cache[op_name] = op
    return op


class MHCOps(NamedTuple):
    """The three registered mHC ops for one set of gate constants.

    ``pre`` and ``post`` match vLLM's ``mhc_pre_torch`` / ``mhc_post_torch``
    contracts; ``fused`` is the seam op, equivalent to ``post`` followed by
    ``pre`` on its output but sharing one pass over the residual streams.

    ``pre(residual, fn, hc_scale, hc_base)``
        residual (..., hc_mult, hidden_size) bf16 in; post_mix
        (..., hc_mult, 1) f32, comb_mix (..., hc_mult, hc_mult) f32,
        layer_input (..., hidden_size) bf16 out.
    ``post(x, residual, post_layer_mix, comb_res_mix)``
        (..., hc_mult, hidden_size) bf16 out.
    ``fused(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base)``
        residual_cur -- the ``post`` output -- then ``pre``'s three outputs
        computed from it, then layer_input's squared row sum (..., 1) f32
        for the RMSNorm that follows (the attention's).
    ``fused_with_shared(routed, shared, residual, ...)``
        ``fused`` on an MoE output given as its routed sum (before the
        routed scale) and its f32 shared-expert output.
    ``fused_normed(..., norm_weight)``
        ``fused``'s first four outputs, the layer input returned through an
        RMSNorm with ``norm_weight`` and epsilon ``rms_eps``.
    ``pre_single_stream(x, ...)`` / ``fused_normed_from_single_stream(x, r, ...)``
        ``pre`` / ``fused_normed`` for a residual whose hc_mult streams are
        copies of one (..., hidden_size) stream, which is passed instead.
    """

    pre: Any
    fused: Any
    post: Any
    fused_with_shared: Any
    fused_normed: Any
    pre_single_stream: Any
    fused_normed_from_single_stream: Any


def get_mhc_ops(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    routed_scale: float = 1.0,
) -> MHCOps:
    """Build (or fetch) every mHC op for these gate constants.

    Must be called outside the compiled region: the builders construct jax
    ``PartitionSpec``s, which Dynamo cannot trace, so a cold cache inside a
    compiled forward is a hard error rather than a graph break. Callers hold
    the returned ops and invoke them directly, keeping the op lookup out of
    the traced graph entirely.
    """
    return MHCOps(
        pre=_mhc_pre_op(
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
        ),
        fused=_mhc_fused_post_pre_op(
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
        ),
        post=get_mhc_post_op(),
        fused_with_shared=_mhc_fused_post_pre_shared_op(
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            routed_scale,
        ),
        fused_normed=_mhc_fused_post_pre_op(
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            normed=True,
        ),
        pre_single_stream=_mhc_pre_single_stream_op(
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
        ),
        fused_normed_from_single_stream=_mhc_fused_post_pre_op(
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            single_stream_residual=True,
            normed=True,
        ),
    )
