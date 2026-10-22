# SPDX-License-Identifier: Apache-2.0
"""Forward checked activations and update exclusively owned snapshot buffers."""

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.hidden_nan_flags import record_hidden_nan

_record = pallas.jax_op(
    "pallas::hidden_nan_flags", record_hidden_nan, donate_argnums=(0, 1, 2, 8)
)


def _fake_record(
    state,
    saved_hidden,
    saved_tokens,
    armed,
    hidden,
    input_ids,
    input_rows,
    layer_slot,
    round_state,
):
    return tuple(
        torch.empty_like(tensor)
        for tensor in (state, saved_hidden, saved_tokens, round_state, hidden)
    )


_record.register_fake(_fake_record)


def record_layer_nan(
    state: torch.Tensor,
    saved_hidden: torch.Tensor,
    saved_tokens: torch.Tensor,
    armed: torch.Tensor,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    input_rows: torch.Tensor,
    layer_slot: int,
    round_state: torch.Tensor,
) -> torch.Tensor:
    slot = torch.scalar_tensor(layer_slot, dtype=torch.int32, device=hidden.device)
    result = _record(
        state,
        saved_hidden,
        saved_tokens,
        armed,
        hidden,
        input_ids,
        input_rows,
        slot,
        round_state,
    )
    for destination, source in zip(
        (state, saved_hidden, saved_tokens, round_state), result
    ):
        destination.copy_(source)
    return result[-1]
