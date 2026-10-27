# SPDX-License-Identifier: Apache-2.0
"""Logs the tensor arguments a compiled executable receives as unmaterialized
views (VLLM_TPU_GLUE_DEBUG=1).

torch_tpu materializes every view argument of a compiled executable with its
own tt_jit_execute_as_strided program on each call. The log names the view
(when its base is a registered parameter, buffer or runner attribute), its
geometry and the executable's argument count, for the first calls of every
executable.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

import torch


def _log(msg: str, *args: Any) -> None:
    # Worker stderr: the vllm_torchtpu.compilation logger has no handler of
    # its own in the engine workers.
    print("[glue_debug] " + (msg % args if args else msg), file=sys.stderr, flush=True)

_MAX_CALLS = 3
_FULL_DUMP_MIN_ARGS = 200
_names: dict[int, str] = {}
_calls: dict[int, int] = {}
_arg_ids: dict[int, list[int]] = {}
_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("VLLM_TPU_GLUE_DEBUG", "0").strip().lower() not in (
        "0",
        "",
        "false",
        "no",
        "off",
    )


def register_module(prefix: str, module: torch.nn.Module) -> None:
    for name, p in module.named_parameters(remove_duplicate=False):
        _names.setdefault(id(p), f"{prefix}.{name}")
    for name, b in module.named_buffers(remove_duplicate=False):
        _names.setdefault(id(b), f"{prefix}.{name}")
    # Plain tensor attributes (not parameters or buffers) of every submodule.
    for mod_name, mod in module.named_modules():
        for attr, value in list(vars(mod).items()):
            if attr.startswith("_parameters") or attr.startswith("_buffers"):
                continue
            if isinstance(value, (torch.Tensor, list, tuple, dict)):
                _register_value(f"{prefix}.{mod_name}.{attr}", value, 1)


def register_object(prefix: str, obj: Any, depth: int = 2) -> None:
    """Registers the tensors reachable from `obj.__dict__` (and from the
    lists, tuples and dicts in it) under `prefix.<attr>` names."""
    attrs = getattr(obj, "__dict__", None)
    if not isinstance(attrs, dict):
        return
    for attr, value in list(attrs.items()):
        _register_value(f"{prefix}.{attr}", value, depth)


def _register_value(name: str, value: Any, depth: int) -> None:
    if isinstance(value, torch.Tensor):
        _names.setdefault(id(value), name)
    elif isinstance(value, torch.nn.Module):
        register_module(name, value)
    elif depth <= 0:
        return
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value[:512]):
            _register_value(f"{name}[{i}]", v, depth - 1)
    elif isinstance(value, dict):
        for k, v in list(value.items())[:512]:
            _register_value(f"{name}[{k!r}]", v, depth - 1)


def _storage_numel(t: torch.Tensor) -> int | None:
    try:
        return t.untyped_storage().nbytes() // t.element_size()
    except Exception:  # noqa: BLE001
        return None


def _is_view(t: torch.Tensor) -> bool:
    if t._base is not None or t.storage_offset() != 0 or not t.is_contiguous():
        return True
    n = _storage_numel(t)
    return n is not None and n != t.numel()


def _describe(i: int, t: torch.Tensor) -> str:
    base = t._base
    name = _names.get(id(t))
    if name is None and base is not None:
        base_name = _names.get(id(base))
        name = f"view of {base_name}" if base_name else "view of ?"
    if name is None:
        name = "?"
    return (
        f"arg[{i}] {name} shape={tuple(t.shape)} {t.dtype} "
        f"stride={tuple(t.stride())} offset={t.storage_offset()} "
        f"storage_numel={_storage_numel(t)} "
        f"base={tuple(base.shape) if base is not None else None}"
    )


def install() -> None:
    from torch_tpu._internal.compile import torch_tpu_compiled_executable as m

    cls = m.TorchTpuCompiledExecutable
    if getattr(cls, "_glue_debug_installed", False):
        return
    orig_call = cls.__call__

    def __call__(self, *args, **kwargs):
        with _lock:
            n = _calls.get(id(self), 0)
            _calls[id(self)] = n + 1
        if n < _MAX_CALLS:
            flat = (
                args[0]
                if len(args) == 1 and isinstance(args[0], (list, tuple))
                else args
            )
            tensors = [(i, a) for i, a in enumerate(flat) if isinstance(a, torch.Tensor)]
            views = [(i, a) for i, a in tensors if _is_view(a)]
            _log(
                "executable %x call %d: %d args, %d tensors, %d views",
                id(self),
                n,
                len(flat),
                len(tensors),
                len(views),
            )
            for i, a in views[:120]:
                _log("  %s", _describe(i, a))
            if len(flat) >= _FULL_DUMP_MIN_ARGS:
                ids = [id(a) for _, a in tensors]
                prev = _arg_ids.get(id(self))
                _arg_ids[id(self)] = ids
                if n == 1:
                    for i, a in tensors:
                        _log("  all %s", _describe(i, a))
                elif n == 2 and prev is not None and len(prev) == len(ids):
                    changed = [
                        (i, a) for (i, a), p_id in zip(tensors, prev) if id(a) != p_id
                    ]
                    _log("  %d of %d tensor args are new objects since the previous call",
                         len(changed), len(tensors))
                    for i, a in changed[:200]:
                        _log("  new %s", _describe(i, a))
        return orig_call(self, *args, **kwargs)

    cls.__call__ = __call__
    cls._glue_debug_installed = True
    _log("compiled executable argument logging installed")
