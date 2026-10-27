# SPDX-License-Identifier: Apache-2.0
"""Trace-time facts about the request mix that replace device-side guards.

The attention and GDN wrappers guard their per-sequence kernels with a
``lax.cond`` on the request distribution, so a step without prefill
sequences skips them. A prefill-only runner (an MTP producer under context
or pipeline parallelism, see ``TPUModelRunner._prefill_only_mtp``) never
runs a step without them. With ``TPU_PREFILL_ONLY_STATIC_GUARDS`` the
runner declares that here once at startup and the wrappers call the
kernels directly. Worker-process state, read when a program is traced.
"""

_prefill_only = False


def set_prefill_only(value: bool = True) -> None:
    """Declares that every forward of this process has prefill sequences."""
    global _prefill_only
    _prefill_only = bool(value)


def prefill_only() -> bool:
    return _prefill_only
