# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Match full-vocabulary greedy selection, including ties and padding."""

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample.vocab_parallel import (
    local_argmax_pair,
    select_argmax_pair,
)


@pytest.mark.parametrize("tp_size", [1, 4, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("case", ["random", "ties", "nan", "negative_inf"])
def test_sharded_argmax_matches_full_vocab(tp_size, dtype, case):
    torch.manual_seed(19)
    vocab_size = 257
    width = (vocab_size + tp_size - 1) // tp_size
    logits = torch.randn(9, width * tp_size).to(dtype)
    if case == "ties":
        logits[:, [0, 127, 256]] = 100
    elif case == "nan":
        logits[:, [17, 224]] = float("nan")
    elif case == "negative_inf":
        logits.fill_(-float("inf"))
    # Padded entries must never win, even if larger than all valid logits.
    logits[:, vocab_size:] = float("inf")
    pairs = torch.stack(
        [
            local_argmax_pair(
                logits[:, rank * width : (rank + 1) * width],
                torch.tensor(rank * width),
                torch.tensor(min((rank + 1) * width, vocab_size)),
            )
            for rank in range(tp_size)
        ],
        dim=1,
    )
    torch.testing.assert_close(
        select_argmax_pair(pairs), logits[:, :vocab_size].argmax(dim=-1)
    )
