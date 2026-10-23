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
"""Runs a model's homogeneous decoder layers as one `torch.scan` (stablehlo.while).

Enabled with `VLLM_TPU_SCAN_LAYERS=1` (requires `VLLM_TPU_BLOCK_MAJOR_KV=1`).

The per-layer weights are stacked once after loading into `[num_layers, ...]`
buffers (the per-layer parameters become views of the stacks, so no extra
memory is used). The block-major KV bundle is threaded through the scan as a
carry and each iteration attends with the bundled RPA kernel at its layer
index, so XLA compiles the decoder layer body once instead of once per layer.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def enabled() -> bool:
    return os.environ.get("VLLM_TPU_SCAN_LAYERS", "0") == "1"


# Model-specific scan bodies, keyed by decoder layer class name.
_SCAN_ATTR = "_tpu_scan_stack"


def _stack_layout(dtype: torch.dtype, ndim: int):
    """Device layout to store a weight stack in, or None for the default.

    Inside the scan XLA reads bf16 weight slices with (16, 128) tiling. A stack
    stored in the default (8, 128) tiling is re-laid out in full on every call
    (a SparseCore data-format op over the whole model), so store it in the
    tiling the loop wants. VLLM_TPU_SCAN_STACK_TILE=0 disables this.
    """
    rows = int(os.environ.get("VLLM_TPU_SCAN_STACK_TILE", "16"))
    if rows <= 0 or dtype != torch.bfloat16 or ndim < 3:
        return None
    from torch_tpu._internal.device_utils.annotations import TpuLayout

    return TpuLayout(
        minor_to_major=list(range(ndim - 1, -1, -1)), tiles=[[rows, 128], [2, 1]]
    )


# (dtype, shape) -> (minor_to_major, tiles) for stacks stored in a non-default
# layout. The compiler sees fake example inputs, so it cannot read the real
# buffers' layouts; pin them as entry layouts instead.
_PINNED_STACK_LAYOUTS: dict = {}
_pin_installed = False


def _install_stack_layout_pin() -> None:
    global _pin_installed
    if _pin_installed:
        return
    from torch_tpu._internal.compile import compiler as tt_compiler

    original_call = tt_compiler.StaticCompiler.__call__

    def _call(self, graph_module, example_inputs, *args, **kwargs):
        layouts = kwargs.get("argument_layouts")
        if _PINNED_STACK_LAYOUTS and len(args) < 3:
            tensors = [v for v in example_inputs if isinstance(v, torch.Tensor)]
            new = list(layouts) if layouts is not None else [[] for _ in tensors]
            pinned = 0
            for i, val in enumerate(tensors):
                pin = _PINNED_STACK_LAYOUTS.get((val.dtype, tuple(val.shape)))
                if pin is not None and not new[i]:
                    new[i] = pin
                    pinned += 1
            if pinned:
                kwargs["argument_layouts"] = new
        return original_call(self, graph_module, example_inputs, *args, **kwargs)

    tt_compiler.StaticCompiler.__call__ = _call
    _pin_installed = True


def _to_device(host: torch.Tensor, device, layout) -> torch.Tensor:
    if layout is None:
        return host.to(device)
    from torch_tpu._internal.device_utils.annotations import LayoutContext

    with LayoutContext(layout):
        out = host.to(device)
    _PINNED_STACK_LAYOUTS[(out.dtype, tuple(out.shape))] = (
        list(layout.minor_to_major),
        [list(t) for t in layout.tiles],
    )
    _install_stack_layout_pin()
    return out


_HOST_STAGING_BYTES = int(os.environ.get("VLLM_TPU_SCAN_HOST_STAGING_BYTES", str(1 << 30)))


def _write_layer(stacked: torch.Tensor, i: int, value: torch.Tensor) -> None:
    """Writes `value` into `stacked[i]` in place.

    `stacked[i].copy_()` writes through a view, which torch_tpu lowers to a
    full-buffer as_strided-inverse copy (2x the stack in HBM). `index_copy_` on
    the base tensor is an in-place update whose input buffer is donated.
    """
    index = torch.tensor([i], dtype=torch.int64, device=stacked.device)
    stacked.index_copy_(0, index, value.unsqueeze(0))


class _ScanStack:
    """Stacked weights and the layer body for one homogeneous run of layers."""

    def __init__(self, layers: list[torch.nn.Module], body: Callable[..., Any]):
        self.layers = layers
        self.template = layers[0]
        self.body = body
        self.names = [n for n, _ in self.template.named_parameters()]
        self.stacked: dict[str, torch.Tensor] = {}
        for name in self.names:
            per_layer = [dict(layer.named_parameters())[name] for layer in layers]
            first = per_layer[0].data
            dtype, device = first.dtype, first.device
            shape = (len(per_layer), *first.shape)
            nbytes = first.numel() * first.element_size() * len(per_layer)
            del first
            layout = _stack_layout(dtype, len(shape))
            if nbytes >= _HOST_STAGING_BYTES or layout is not None:
                # Large stacks are assembled in host memory: each layer is
                # copied out and its device buffer freed before the stack is
                # uploaded, so device memory never holds both copies.
                host = torch.empty(shape, dtype=dtype, device="cpu")
                for i, p in enumerate(per_layer):
                    host[i].copy_(p.data.to("cpu"))
                    p.data = torch.empty(0, dtype=dtype, device=device)
                stacked = _to_device(host, device, layout)
                del host
            else:
                stacked = torch.empty(shape, dtype=dtype, device=device)
                # Free each layer's original right after copying it, and only
                # take views at the end: a live view would block the in-place
                # update.
                for i, p in enumerate(per_layer):
                    _write_layer(stacked, i, p.data)
                    p.data = torch.empty(0, dtype=dtype, device=device)
            for i, p in enumerate(per_layer):
                p.data = stacked[i]
            self.stacked[name] = stacked


def _attn_of(layer: torch.nn.Module) -> torch.nn.Module:
    return layer.self_attn.attn


def prepare(model: torch.nn.Module) -> int:
    """Builds the stacked weights for every supported decoder stack in `model`."""
    count = 0
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if not isinstance(layers, torch.nn.ModuleList) or not hasattr(
            module, "start_layer"
        ):
            continue
        run = list(layers[module.start_layer : module.end_layer])
        if len(run) < 2:
            continue
        cls_names = {type(layer).__name__ for layer in run}
        if len(cls_names) != 1 or next(iter(cls_names)) not in _GENERIC_BODIES:
            logger.warning("scan_layers: unsupported decoder layers %s", cls_names)
            continue
        shapes = {
            tuple((n, tuple(p.shape), p.dtype) for n, p in layer.named_parameters())
            for layer in run
        }
        if len(shapes) != 1:
            logger.warning("scan_layers: heterogeneous layer parameters; skipping")
            continue
        cls_name = next(iter(cls_names))
        body, prefixes = _GENERIC_BODIES[cls_name]
        stack = _ScanStack(run, body)
        stack.module_ids = {}
        stack.shared = {}
        for prefix in prefixes:
            module_id = _register_module(run[0].get_submodule(prefix))
            stack.module_ids[prefix] = (module_id, _MODULES[module_id][1])
            for path, kind in _MODULES[module_id][2]:
                _stack_slot(stack, run, prefix, path, kind)
        if os.environ.get("VLLM_TPU_SCAN_MOE_FULL_STACK", "1") == "1":
            _mark_full_stacks(stack, run)
        stack.body = functools.partial(_generic_body_adapter, body, stack)
        # Materialize the stacks now: the compiler reads argument layouts from
        # materialized device buffers (see _stack_layout).
        torch.accelerator.synchronize()
        # Stored outside the module tree so `named_parameters` is unchanged.
        module.__dict__[_SCAN_ATTR] = stack
        count += 1
        logger.info(
            "scan_layers: stacked %d %s layers (%d weights)",
            len(run),
            next(iter(cls_names)),
            len(stack.names),
        )
    return count


def _stack_slot(stack, run, prefix, path, kind):
    """Stacks a per-layer non-parameter tensor, or records it as shared."""
    owners = [_get_slot(layer.get_submodule(prefix), path) for layer in run]
    values = [_read_slot(o, a, kind) for o, a in owners]
    full = f"{prefix}.{path}"
    first = values[0]
    if any(v.shape != first.shape or v.dtype != first.dtype for v in values):
        raise NotImplementedError(f"scan_layers: heterogeneous tensor {full}")
    # Identical across layers (e.g. this rank's expert offsets): keep one copy
    # as a loop invariant instead of slicing it per iteration.
    if all(v is first or torch.equal(v, first) for v in values):
        stack.shared[full] = first
        return
    stacked = torch.empty((len(values), *first.shape), dtype=first.dtype, device=first.device)
    for i, v in enumerate(values):
        _write_layer(stacked, i, v)
    del values
    for i, (owner, attr) in enumerate(owners):
        _set_slot(owner, attr, kind, stacked[i])
    stack.stacked[full] = stacked
    stack.names.append(full)
    logger.info("scan_layers: stacked non-parameter tensor %s %s", full, tuple(first.shape))


def _mark_full_stacks(stack, run) -> None:
    """Pass MoE expert stacks whole (plus a layer index) instead of slicing."""
    stack.full = set()
    for prefix, (module_id, names) in stack.module_ids.items():
        module = run[0].get_submodule(prefix)
        owners = set()
        for n in names:
            owner_path, _, attr = n.rpartition(".")
            if attr in _FULL_STACK_SUFFIXES:
                stack.full.add(f"{prefix}.{n}")
                owners.add(owner_path)
        if owners:
            _INDEX_OWNERS[module_id] = sorted(owners)
    if stack.full:
        stack.names = [n for n in stack.names if n not in stack.full]
        logger.info("scan_layers: full-stack MoE params %s", sorted(stack.full))


def _generic_body_adapter(body, stack, layer, p, positions, h, res, kv, idx, attn_metadata):
    return body(layer, p, positions, h, res, kv, idx, attn_metadata, stack)


def get_stack(module: torch.nn.Module) -> _ScanStack | None:
    return module.__dict__.get(_SCAN_ATTR)


def run_stack(
    stack: _ScanStack,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs the stacked layers as a scan over (weights, layer index)."""
    from torch._higher_order_ops.scan import scan
    from vllm.model_executor.layers.attention.attention import get_attention_context

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    bundle = get_vllm_model_wrapper_context().kv_cache_bundle
    template = stack.template
    attn = _attn_of(template)
    if bundle is None:
        # kv=0 profile run (its graph is discarded): scan without attention so
        # the per-layer weight views are never materialized one by one.
        attn_metadata = None
        kv_in = torch.zeros(1, device=hidden_states.device)
        layer_idx = torch.arange(len(stack.layers), dtype=torch.int32, device=hidden_states.device)
    else:
        attn_metadata = get_attention_context(attn.layer_name)[0]
        kv_in = bundle
        layer_idx = torch.stack(
            [_attn_of(layer).impl._bundle_layer_idx_tensor for layer in stack.layers]
        )
    weights = [stack.stacked[n] for n in stack.names]
    if residual is None:
        residual = torch.zeros_like(hidden_states)
    # Optional partial unroll: each scan step runs `unroll` consecutive layers,
    # trading compile time for fewer loop iterations (and cross-layer fusion).
    unroll = int(os.environ.get("VLLM_TPU_SCAN_UNROLL", "1"))
    if unroll < 1 or len(stack.layers) % unroll:
        unroll = 1
    if unroll > 1:
        groups = len(stack.layers) // unroll
        weights = [w.view(groups, unroll, *w.shape[1:]) for w in weights]
        layer_idx = layer_idx.view(groups, unroll, *layer_idx.shape[1:])

    def combine(carry, xs):
        h, res, kv = carry
        *ws, idx = xs
        if unroll == 1:
            params = dict(zip(stack.names, ws))
            h, res, kv = stack.body(
                template, params, positions, h, res, kv, idx, attn_metadata
            )
        else:
            for j in range(unroll):
                params = {n: w[j] for n, w in zip(stack.names, ws)}
                h, res, kv = stack.body(
                    template, params, positions, h, res, kv, idx[j], attn_metadata
                )
        return (h, res, kv), idx.clone()

    (hidden_states, residual, new_bundle), _ = scan(
        combine, (hidden_states, residual, kv_in), (*weights, layer_idx)
    )
    if bundle is not None:
        bundle.copy_(new_bundle)
    return hidden_states, residual


# --------------------------------------------------------------------------- #
# Generic explicit-weight module calls.
#
# `call_module` runs a template submodule with its parameters temporarily
# replaced by the given tensors. Dynamo/AOT see an opaque op whose weights are
# explicit operands (so the scan slices flow in as xs); the swap itself only
# happens when the op executes -- under fake tensors for shape propagation and
# under torch_tpu's FX interpreter when the scan body is lowered -- so quantized
# linears, MoE runners and other modules that look weights up at run time work
# unchanged.
# --------------------------------------------------------------------------- #
_MODULES: list[tuple[torch.nn.Module, list[str], list[tuple[str, str]]]] = []
# Submodules (per registered module) that read `_tpu_scan_layer_index`.
_INDEX_OWNERS: dict[int, list[str]] = {}
# Expert parameters passed as whole layer stacks plus a layer index, so the MoE
# kernel reads the layer in place instead of a materialized per-layer slice.
_FULL_STACK_SUFFIXES = (
    "w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv",
    "w13_bias", "w2_bias",
)


def _tensor_slots(module: torch.nn.Module) -> list[tuple[str, str]]:
    """Non-parameter device tensors a module reads at run time: (path, kind)."""
    slots = []
    for mod_name, mod in module.named_modules():
        items = [(a, v, "buffer") for a, v in mod._buffers.items()]
        items += [(a, v, "attr") for a, v in vars(mod).items()]
        for attr, val, kind in items:
            if (
                isinstance(val, torch.Tensor)
                and not isinstance(val, torch.nn.Parameter)
                and val.device.type != "cpu"
            ):
                slots.append((f"{mod_name}.{attr}" if mod_name else attr, kind))
    return slots


def _register_module(module: torch.nn.Module) -> int:
    names = [n for n, _ in module.named_parameters()]
    _MODULES.append((module, names, _tensor_slots(module)))
    return len(_MODULES) - 1


def _get_slot(module, path):
    owner_name, _, attr = path.rpartition(".")
    owner = module.get_submodule(owner_name) if owner_name else module
    return owner, attr


def _set_slot(owner, attr, kind, value):
    if kind == "param":
        owner._parameters[attr] = value
    elif kind == "buffer":
        owner._buffers[attr] = value
    else:
        owner.__dict__[attr] = value


def _read_slot(owner, attr, kind):
    if kind == "param":
        return owner._parameters[attr]
    if kind == "buffer":
        return owner._buffers[attr]
    return owner.__dict__[attr]


def _swap_call(
    module_id: int,
    args: list[torch.Tensor],
    params: list[torch.Tensor],
    extras: list[torch.Tensor],
    layer_index: list[torch.Tensor] = (),
):
    module, names, slots = _MODULES[module_id]
    plan = [(n, "param") for n in names] + list(slots)
    if layer_index:
        plan += [(f"{p}._tpu_scan_layer_index" if p else "_tpu_scan_layer_index", "attr")
                 for p in _INDEX_OWNERS.get(module_id, [])]
        extras = list(extras) + [layer_index[0]] * len(_INDEX_OWNERS.get(module_id, []))
    saved = []
    try:
        for (path, kind), value in zip(plan, list(params) + list(extras)):
            owner, attr = _get_slot(module, path)
            prev = owner.__dict__.get(attr) if kind == "attr" else _read_slot(owner, attr, kind)
            saved.append((owner, attr, kind, prev))
            _set_slot(owner, attr, kind, value)
        out = module(*args)
    finally:
        for owner, attr, kind, value in reversed(saved):
            if kind == "attr" and value is None:
                owner.__dict__.pop(attr, None)
            else:
                _set_slot(owner, attr, kind, value)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    result = []
    for t in outs:
        if not isinstance(t, torch.Tensor):
            continue
        if any(t is a for a in list(args) + list(params) + list(extras)):
            t = t.clone()
        result.append(t)
    return result


@torch.library.custom_op("tpu_scan::call_module", mutates_args=())
def call_module(
    module_id: int,
    args: list[torch.Tensor],
    params: list[torch.Tensor],
    extras: list[torch.Tensor],
    layer_index: list[torch.Tensor],
) -> list[torch.Tensor]:
    return _swap_call(module_id, args, params, extras, layer_index)


@call_module.register_fake
def _call_module_fake(module_id, args, params, extras, layer_index):
    # Modules may still hold real device tensors outside the swapped slots;
    # let the active fake mode accept them while shapes are propagated.
    from torch._guards import detect_fake_mode

    mode = detect_fake_mode(list(args) + list(params) + list(extras))
    if mode is None:
        return _swap_call(module_id, args, params, extras, layer_index)
    prev = mode.allow_non_fake_inputs
    mode.allow_non_fake_inputs = True
    try:
        return _swap_call(module_id, args, params, extras, layer_index)
    finally:
        mode.allow_non_fake_inputs = prev


def _call(stack, prefix: str, p: dict, *args: torch.Tensor, idx=None) -> list[torch.Tensor]:
    module_id, names = stack.module_ids[prefix]
    slots = _MODULES[module_id][2]
    full = getattr(stack, "full", set())
    # Full-stack params are loop invariants (closure); the rest are xs slices.
    params = [
        stack.stacked[f"{prefix}.{n}"] if f"{prefix}.{n}" in full else p[f"{prefix}.{n}"]
        for n in names
    ]
    extras = [
        p[f"{prefix}.{path}"] if f"{prefix}.{path}" in p else stack.shared[f"{prefix}.{path}"]
        for path, _ in slots
    ]
    use_index = idx is not None and module_id in _INDEX_OWNERS
    return call_module(module_id, list(args), params, extras, [idx] if use_index else [])


def _qwen_like_body(layer, p, positions, h, res, kv, idx, attn_metadata, stack):
    """Qwen3/Qwen3-MoE decoder layer with explicit weights and functional KV.

    Norms, projections and the MLP/MoE run through `call_module`, so any
    quantization method or MoE implementation works unchanged.
    """
    sa = layer.self_attn
    h, res = _call(stack, "input_layernorm", p, h, res)
    qkv = _call(stack, "self_attn.qkv_proj", p, h)[0]
    q, k, v = qkv.split([sa.q_size, sa.kv_size, sa.kv_size], dim=-1)
    q = _call(
        stack, "self_attn.q_norm", p,
        q.view(*q.shape[:-1], q.shape[-1] // sa.head_dim, sa.head_dim),
    )[0].view(q.shape)
    k = _call(
        stack, "self_attn.k_norm", p,
        k.view(*k.shape[:-1], k.shape[-1] // sa.head_dim, sa.head_dim),
    )[0].view(k.shape)
    q, k = sa.rotary_emb(positions, q, k)
    attn = sa.attn
    if attn_metadata is None:
        # kv=0 profile run: no cache; keep the per-layer activation footprint.
        out = torch.zeros_like(q)
        kv = kv.clone()  # a carry returned unchanged would alias the input
    else:
        out, kv = attn.impl.forward_bundled_functional(
            attn, q, k, v, kv, idx, attn_metadata
        )
    out = out.reshape(out.shape[0], -1)
    h = _call(stack, "self_attn.o_proj", p, out)[0]
    h, res = _call(stack, "post_attention_layernorm", p, h, res)
    h = _call(stack, "mlp", p, h, idx=idx)[0]
    return h, res, kv


_QWEN_LIKE_MODULES = (
    "input_layernorm",
    "self_attn.qkv_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "self_attn.o_proj",
    "post_attention_layernorm",
    "mlp",
)
_GENERIC_BODIES: dict[str, tuple[Callable[..., Any], tuple[str, ...]]] = {
    "Qwen3MoeDecoderLayer": (_qwen_like_body, _QWEN_LIKE_MODULES),
}


def _make_forward(original):
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        stack = get_stack(self)
        if (
            stack is None
            or kwargs
            or getattr(self, "aux_hidden_state_layers", None)
        ):
            return original(
                self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        from vllm.distributed import get_pp_group

        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
            return original(
                self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        hidden_states, residual = run_stack(stack, positions, hidden_states, None)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    forward._tpu_scan_patch = True
    return forward


def install() -> None:
    """Routes supported models' layer loops through the scan when prepared."""
    import importlib

    for mod_name, cls_name in (
        ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeModel"),
    ):
        cls = getattr(importlib.import_module(mod_name), cls_name)
        if getattr(cls.forward, "_tpu_scan_patch", False):
            continue
        cls.forward = _make_forward(cls.forward)
        logger.info("scan_layers: installed %s.forward scan path", cls_name)
