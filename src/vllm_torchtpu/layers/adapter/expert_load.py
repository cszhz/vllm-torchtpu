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
"""TPU_MOE_EXPERT_LOAD_LOG_EVERY=N: the fused EP kernel's own per-slot routed
row counts, accumulated on the device and read back every N steps.

The fused EP layer all-gathers every rank's per-slot pair counts on every
step (`rows_by_dest`, [ep, physical slots]) to plan its dispatch, so every
rank already knows the whole step's load per physical slot with no further
collective. With this on, each armed layer carries a [physical slots] int32
accumulator that the op takes as a donated operand and returns updated; the
runner reads the accumulators back every N steps and logs, per layer,

    moe_lb = rows on the slowest rank / rows on the mean rank

over the window (DeepInfer's imbalance metric; under DP lockstep the slowest
rank is the layer's cost), the layer's hottest slot, and the window's worst
layers. TPU_MOE_EXPERT_LOAD_DIR writes the cumulative counts as
expert_load_dp<rank>_<seq>.npz as well. Off (0) the program is unchanged: the
accumulator is one more operand and one [ep, P] reduce per layer, and the
readback is 43 x P int32 per window, so a tapped start costs little, but it
is still not a timing arm.
"""

import os
import time

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LOAD_ATTR = "_tpu_expert_load"

# (layer name, layer, ep) for every registered layer of this process, in
# registration order. The layers own the buffers; this only lists them.
_REGISTERED: list[tuple[str, torch.nn.Module, int]] = []
_STATE: dict = {"steps": 0, "seq": 0, "previous": None}


def enabled() -> bool:
    """The accumulator exists when the log asks for it or the dynamic
    re-layout needs it (expert_relayout.py reads it through snapshot())."""
    return envs.TPU_MOE_EXPERT_LOAD_LOG_EVERY > 0 or bool(envs.TPU_MOE_EPLB_DYNAMIC)


def register(layer, *, physical: int, ep: int) -> None:
    """Give an armed layer its [physical] accumulator. Call after weight load."""
    if not enabled():
        return
    layer.register_buffer(
        LOAD_ATTR,
        torch.zeros((physical,), dtype=torch.int32, device=layer.w13_weight.device),
        persistent=False,
    )
    _REGISTERED.append((layer.layer_name, layer, ep))


def operand(layer):
    """The layer's accumulator, or None when the tap is off."""
    return getattr(layer, LOAD_ATTR, None)


def commit(layer, updated: torch.Tensor) -> None:
    """Write the op's returned accumulator back: the operand was donated."""
    getattr(layer, LOAD_ATTR).copy_(updated)


def registered_layers() -> int:
    """How many layers carry an accumulator in this process."""
    return len(_REGISTERED)


def _read_all() -> np.ndarray:
    """Every registered layer's accumulator as one [layers, widest] int64 host
    array (rows zero-padded to the widest: the target's layers and the
    drafter's have different slot counts). One device -> host transfer rather
    than one per layer: this runs in the step hook."""
    buffers = [getattr(layer, LOAD_ATTR) for _, layer, _ in _REGISTERED]
    width = max(b.numel() for b in buffers)
    if all(b.numel() == width for b in buffers):
        stacked = torch.stack(buffers)
    else:
        stacked = torch.stack(
            [torch.nn.functional.pad(b, (0, width - b.numel())) for b in buffers]
        )
    return stacked.cpu().numpy().astype(np.int64)


def snapshot() -> dict[tuple[str, int], np.ndarray]:
    """Every registered layer's cumulative counts, keyed by its layout
    (group, layer index), as int64 host arrays. The counts are replicated
    sums, identical on every rank after the same step, which is what lets
    the dynamic re-layout solve without a collective (expert_relayout.py)."""
    from vllm_torchtpu import expert_placement

    if not _REGISTERED:
        return {}
    host = _read_all()
    out = {}
    for i, (name, layer, _) in enumerate(_REGISTERED):
        key = expert_placement.layout_key(layer)
        if key is None:
            continue
        out[key] = host[i, : getattr(layer, LOAD_ATTR).numel()].copy()
    return out


def summarize(window: np.ndarray, ep: int) -> dict:
    """Per-layer balance of a [physical] row-count window over `ep` ranks."""
    per_rank = window.reshape(ep, -1).sum(axis=1)
    mean = per_rank.mean()
    return {
        "moe_lb": float(per_rank.max() / mean) if mean > 0 else 1.0,
        "slowest_rank": int(per_rank.argmax()),
        "hottest_slot": int(window.argmax()),
        "hottest_rows": int(window.max()),
        "rows": int(window.sum()),
    }


def maybe_log(dp_rank: int) -> None:
    """Called once per runner step on the host; reads back every N steps."""
    if not _REGISTERED:
        return
    _STATE["steps"] += 1
    every = envs.TPU_MOE_EXPERT_LOAD_LOG_EVERY
    if every <= 0 or _STATE["steps"] % every:
        return
    # One readback for every layer, rows padded to the widest slot count and
    # summarized at their own width.
    widths = [getattr(layer, LOAD_ATTR).numel() for _, layer, _ in _REGISTERED]
    host = _read_all()
    previous = _STATE["previous"]
    window = host if previous is None else host - previous
    _STATE["previous"] = host
    stats = [
        summarize(window[i, : widths[i]], ep)
        for i, (_, _, ep) in enumerate(_REGISTERED)
    ]
    lbs = np.array([s["moe_lb"] for s in stats])
    worst = np.argsort(-lbs)[:3]
    if dp_rank == 0:
        logger.info(
            "expert load | step %d | window %d steps | moe_lb mean %.3f max %.3f "
            "| worst layers %s | per layer %s",
            _STATE["steps"],
            every,
            lbs.mean(),
            lbs.max(),
            ", ".join(
                f"{_REGISTERED[i][0]}={lbs[i]:.2f} (slot {stats[i]['hottest_slot']}: "
                f"{stats[i]['hottest_rows']} rows)"
                for i in worst
            ),
            " ".join(f"{lb:.2f}" for lb in lbs),
        )
    directory = envs.TPU_MOE_EXPERT_LOAD_DIR
    if directory:
        directory = os.path.expandvars(directory)
        os.makedirs(directory, exist_ok=True)
        np.savez(
            os.path.join(directory, f"expert_load_dp{dp_rank}_{_STATE['seq']:04d}.npz"),
            counts=host,
            window=window,
            layer_names=np.array([name for name, _, _ in _REGISTERED]),
            ep=np.array([ep for _, _, ep in _REGISTERED]),
            physical=np.array(widths),
            step=_STATE["steps"],
            wall=time.time(),
        )
    _STATE["seq"] += 1
