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
"""Bridge between `megakernel` (`make_flat_pooled_step`) and PyTorch/vLLM.

Provides:
- `build_pooled_step_op`: wraps `cache.make_flat_pooled_step` into a
  `sharded_jax_callable` + `@torch.library.custom_op` (taking `list[Tensor]`
  for `weights` and `pools` to stay under the PyTorch dispatcher's 64-argument
  schema limit), with automatic CPU `interpret=True` execution when called with
  CPU `torch.Tensor` operands in unit tests.
- `execute_pooled_step_inplace`: invokes the custom op and copies the donated
  pool outputs back into `pools` via `dst.copy_(src)`, preserving `dst.data_ptr()`
  (Raiden pointer identity) and folding into an XLA `input_output_alias` /
  `tf.aliasing_output` under `torch.compile(backend="tpu")`.
- `export_pooled_step`: offline inspection/serialization helper that exports the
  StableHLO module and aliasing manifest (`PooledStepExport`) via `jax.export`
  (serving traces directly through `build_pooled_step_op` / `JaxCallable`).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import cache
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.pool.config import cache_exec
from vllm_torchtpu.megakernel.runtime import build_global_avals
from vllm_torchtpu.megakernel.spec import ModelSpec


def _leaf_output_specs(output_partition_specs: Any) -> list[P]:
    """Flatten `output_partition_specs` to one `PartitionSpec` per output leaf."""
    return jax.tree.leaves(output_partition_specs, is_leaf=lambda s: isinstance(s, P))


def sharded_jax_callable(
    name: str,
    fn: Callable[..., object],
    *,
    mesh: jax.sharding.Mesh,
    input_partition_specs: Sequence[P],
    output_partition_specs: Any,
    donate_argnums: Sequence[int] | None = None,
) -> Any:
    """Wrap an SPMD JAX function over `mesh` into a `torch_tpu` `JaxCallable`.

    Self-contained so `exported_bridge` does not depend on any `vllm` or
    `vllm_torchtpu.distributed` modules when extracted to a standalone repo.
    """
    from torch_tpu._internal.pallas import pallas as pallas_impl

    if "::" not in name or len(name.split("::")) != 2:
        raise ValueError(f"Op name {name} must be 'namespace::op', with one '::'.")

    init_sig = inspect.signature(pallas_impl.JaxCallable.__init__)
    if "output_partition_specs" not in init_sig.parameters:
        raise RuntimeError(
            "torch_tpu.JaxCallable.__init__ does not support 'output_partition_specs'; "
            "upgrade torch_tpu to use sharded_jax_callable."
        )

    signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    input_partition_specs = tuple(input_partition_specs)
    out_shardings = jax.tree.map(
        lambda s: NamedSharding(mesh, s),
        output_partition_specs,
        is_leaf=lambda s: isinstance(s, P),
    )

    jit_fn = jax.jit(
        fn,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        out_shardings=out_shardings,
        keep_unused=True,
    )
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(fn)}",
        [],
        {
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums,
            "output_partition_specs": str(output_partition_specs),
        },
    )
    return pallas_impl.JaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=tuple(_leaf_output_specs(output_partition_specs)),
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )


@dataclass(frozen=True)
class PooledStepExport:
    """Offline serialized StableHLO export and aliasing manifest for a pooled step."""

    op_name: str
    exported: Any
    mlir_module_serialized: bytes
    in_specs: tuple[P, ...]
    out_specs: tuple[P, ...]
    donate_argnums: tuple[int, ...]
    input_output_aliases: dict[int, int]


_OP_COUNTER = 0


def _torch_cpu_to_jax(
    t: torch.Tensor, sharding: NamedSharding | None = None
) -> jax.Array:
    """Convert a CPU `torch.Tensor` to a `jax.Array` without precision loss."""
    t_c = t.detach().cpu().contiguous()
    if t_c.dtype == torch.float8_e4m3fn:
        arr = jnp.asarray(t_c.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    elif t_c.dtype == torch.bfloat16:
        arr = jnp.asarray(t_c.view(torch.uint16).numpy()).view(jnp.bfloat16)
    else:
        arr = jnp.asarray(t_c.numpy())
    if sharding is not None:
        arr = jax.device_put(arr, sharding)
    return arr


def _jax_to_torch_cpu(a: jax.Array, like: torch.Tensor) -> torch.Tensor:
    """Convert a `jax.Array` to a CPU `torch.Tensor` matching `like.dtype`."""
    np_arr = np.asarray(a)
    if like.dtype == torch.float8_e4m3fn:
        out = torch.from_numpy(np_arr.view(np.uint8).copy()).view(torch.float8_e4m3fn)
    elif like.dtype == torch.bfloat16:
        out = torch.from_numpy(np_arr.view(np.uint16).copy()).view(torch.bfloat16)
    else:
        out = torch.from_numpy(np_arr.copy()).to(like.dtype)
    return out


def make_typed_step_wrapper(
    fn_name: str,
    step_fn: Callable[..., Any],
    num_inputs: int,
    num_outputs: int,
) -> Callable[..., Any]:
    """Synthesize a fixed-arity `jax.Array`-annotated wrapper for `JaxCallable`.

    `torch_tpu`'s `JaxCallable` inspects the function's explicit positional
    parameters and rejects `*args` signatures.
    """
    args = [f"x{i}" for i in range(num_inputs)]
    ret = "tuple[" + ", ".join(["jax.Array"] * num_outputs) + "]"
    src = (
        f"def {fn_name}({', '.join(a + ': jax.Array' for a in args)}) -> {ret}:\n"
        f"    return _step_fn({', '.join(args)})\n"
    )
    ns: dict[str, Any] = {"jax": jax, "_step_fn": step_fn}
    exec(compile(src, f"<{fn_name}>", "exec"), ns)  # noqa: S102
    return ns[fn_name]


def export_pooled_step(
    mesh: jax.sharding.Mesh,
    spec: ModelSpec,
    ex: ExecSpec,
    w_flat: Sequence[torch.Tensor],
    raw_pools: Sequence[torch.Tensor],
    *,
    max_pages_per_seq: int,
    ckpt_cols: int | None = None,
    op_name: str = "megakernel_runtime::exported_pooled_step",
    platforms: Sequence[str] = ("tpu",),
) -> PooledStepExport:
    """Export `make_flat_pooled_step` to a serialized StableHLO module.

    Note: At serving time, `build_pooled_step_op` traces through
    `sharded_jax_callable` (`JaxCallable`). This helper is provided for offline
    StableHLO inspection and serialization tests.
    """
    step_fn, in_specs, out_specs, donate_argnums = cache.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=True
    )
    in_sh = tuple(NamedSharding(mesh, s) for s in in_specs)
    out_sh = tuple(NamedSharding(mesh, s) for s in out_specs)
    avals = build_global_avals(
        mesh,
        spec,
        ex,
        w_flat,
        raw_pools,
        max_pages_per_seq=max_pages_per_seq,
        ckpt_cols=ckpt_cols,
    )
    j_fn = jax.jit(
        step_fn,
        in_shardings=in_sh,
        out_shardings=out_sh,
        donate_argnums=donate_argnums,
        keep_unused=True,
    )
    try:
        with jax._src.config.export_ignore_forward_compatibility(True):
            exported: Any = jax.export.export(
                j_fn,
                platforms=list(platforms),
                disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("ALL")],
            )(*avals)
        mlir_bytes = exported.mlir_module_serialized
    except NotImplementedError:
        if not ex.interpret:
            raise
        lowered = j_fn.lower(*avals)
        exported = lowered
        mlir_bytes = lowered.as_text("stablehlo").encode("utf-8")
    n_w = len(w_flat)
    n_p = len(raw_pools)
    aliases = {2 + n_w + i: 1 + i for i in range(n_p)}
    return PooledStepExport(
        op_name=op_name,
        exported=exported,
        mlir_module_serialized=mlir_bytes,
        in_specs=in_specs,
        out_specs=out_specs,
        donate_argnums=donate_argnums,
        input_output_aliases=aliases,
    )


def build_pooled_step_op(
    mesh: jax.sharding.Mesh,
    spec: ModelSpec,
    ex: ExecSpec,
    *,
    op_name: str | None = None,
    donate_pools: bool = True,
) -> tuple[Any, tuple[P, ...], tuple[P, ...], tuple[int, ...]]:
    """Build a PyTorch custom op wrapping `cache.make_flat_pooled_step`.

    Returns `(op, in_specs, out_specs, donate_argnums)` where `op` takes
    `(pos, hidden, page_table, ckpt_indices, read_offsets, weights, pools)`.
    """
    global _OP_COUNTER
    step_fn, in_specs, out_specs, donate_argnums = cache.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=donate_pools
    )
    _OP_COUNTER += 1
    if op_name is None:
        n_pages = cache_exec(ex).num_pages
        op_name = f"megakernel_runtime::step_t{ex.T}_b{ex.B}_p{n_pages}_{_OP_COUNTER}"
    elif "::" not in op_name:
        op_name = f"megakernel_runtime::{op_name}"

    fn_name = op_name.replace("::", "_")
    typed_fn = make_typed_step_wrapper(
        fn_name, step_fn, num_inputs=len(in_specs), num_outputs=len(out_specs)
    )
    try:
        flat_callable: Any = sharded_jax_callable(
            op_name,
            typed_fn,
            mesh=mesh,
            input_partition_specs=in_specs,
            output_partition_specs=out_specs,
            donate_argnums=donate_argnums,
        )
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("torch_tpu"):
            flat_callable = None
        else:
            raise

    in_sh = tuple(NamedSharding(mesh, s) for s in in_specs)
    out_sh = tuple(NamedSharding(mesh, s) for s in out_specs)
    cpu_jit_fn = jax.jit(
        step_fn,
        in_shardings=in_sh,
        out_shardings=out_sh,
        donate_argnums=donate_argnums,
        keep_unused=True,
    )

    @torch.library.custom_op(op_name, mutates_args=())
    def op(
        pos: torch.Tensor,
        hidden: torch.Tensor,
        page_table: torch.Tensor,
        ckpt_indices: torch.Tensor,
        read_offsets: torch.Tensor,
        weights: list[torch.Tensor],
        pools: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        all_tensors = [
            pos,
            hidden,
            *weights,
            *pools,
            page_table,
            ckpt_indices,
            read_offsets,
        ]
        if pos.device.type == "cpu":
            # CPU interpret execution path for unit tests without a TPU host.
            jax_inputs = [
                _torch_cpu_to_jax(t, sh)
                for t, sh in zip(all_tensors, in_sh, strict=True)
            ]
            jax_outs = cpu_jit_fn(*jax_inputs)
            like_tensors = [hidden, *pools]
            return [
                _jax_to_torch_cpu(o, lk)
                for o, lk in zip(jax_outs, like_tensors, strict=True)
            ]
        if flat_callable is None:
            raise RuntimeError(
                "torch_tpu is required to execute build_pooled_step_op on TPU"
            )
        return list(flat_callable(*all_tensors))

    @op.register_fake
    def _(
        pos: torch.Tensor,
        hidden: torch.Tensor,
        page_table: torch.Tensor,
        ckpt_indices: torch.Tensor,
        read_offsets: torch.Tensor,
        weights: list[torch.Tensor],
        pools: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [torch.empty_like(hidden)] + [torch.empty_like(p) for p in pools]

    return op, in_specs, out_specs, donate_argnums


def execute_pooled_step_inplace(
    op: Callable[..., list[torch.Tensor]],
    pos: torch.Tensor,
    hidden: torch.Tensor,
    page_table: torch.Tensor,
    ckpt_indices: torch.Tensor,
    read_offsets: torch.Tensor,
    weights: Sequence[torch.Tensor],
    pools: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Execute `op`, copy donated outputs into `pools` in-place, and return `hid_out`.

    Preserves `pool.data_ptr()` and `pool.storage().data_ptr()` on every pool
    tensor so `raiden`-pinned buffers remain valid without re-registration.
    """
    outs = op(
        pos,
        hidden,
        page_table,
        ckpt_indices,
        read_offsets,
        list(weights),
        list(pools),
    )
    n_p = len(pools)
    for dst, src in zip(pools, outs[1 : 1 + n_p]):
        dst.copy_(src)
    return outs[0]
