# SPDX-License-Identifier: Apache-2.0
"""Opt-in HBM placement for Pallas call operands (TPU_PIN_OPERANDS_HBM).

A pallas_call in_spec's memory space only types the ref inside the kernel. XLA's
memory-space assignment reads the operand's aval, so an operand the kernel reads
with its own DMAs (or through a pipeline) may still be copied into VMEM by XLA
right before the call, and that copy is not overlapped with anything. Constraining
the operand to HBM (`pltpu.with_memory_space_constraint`, which lowers to the
custom call's input memory space) removes the copy; the kernel's DMAs then read
HBM. The constraint binds only pallas_call inputs and must be traced under jit.
"""

from __future__ import annotations

import jax
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs

SITES = (
    "skinny",  # fp8_skinny: activation and per-channel scale
    "proj",  # compressor proj_and_save_state: weight, activation, positions, APE
    "combine",  # fused EP MoE decode combine: arrivals and their scales
    "select",  # fused EP MoE routed top-k select: score bias and slot labels
    "pages",  # sparse MLA gather_page_ids: per-token page-table windows
    "plan",  # fused EP MoE decode plan: writes the fused kernel's tables to HBM
)


def enabled(site: str) -> bool:
    if site not in SITES:
        raise ValueError(f"unknown operand-pin site {site!r}; known: {SITES}")
    # ":" also separates, so the list fits in a comma-separated env list.
    listed = {
        s.strip() for s in envs.TPU_PIN_OPERANDS_HBM.replace(":", ",").split(",")
    } - {""}
    unknown = listed - set(SITES) - {"all"}
    if unknown:
        raise ValueError(f"TPU_PIN_OPERANDS_HBM names unknown sites {sorted(unknown)}")
    return "all" in listed or site in listed


def hbm(x: jax.Array) -> jax.Array:
    return pltpu.with_memory_space_constraint(x, pltpu.HBM)
