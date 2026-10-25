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
"""Predicted rank-dependent attention time of a scheduled chunk, in microseconds.

Under DP lockstep the ranks differ only in their attention work (indexer, HCA,
CSA, SWA, compressor); MoE and the dense layers cost every rank the same. On
DeepSeek-V4-Pro (tpu7x, EP16, c=160 AgentX) that time per rank and step is
``US_PER_TOKEN`` per scheduled token plus a per-(token x context) term, larger
for decode (speculative verify) tokens than for prefill tokens: a fit over 284
core-steps of xprof against the batch each rank ran gave R^2 = 0.988, residual
0.39 ms. A chunk of ``n`` tokens starting at context ``c`` is charged
``n * (c + n / 2)`` token-context.
"""

import numpy as np

US_PER_TOKEN = 63.0
US_PER_M_DECODE_TOKEN_CONTEXT = 535.0
US_PER_M_PREFILL_TOKEN_CONTEXT = 259.0


def chunk_cost_us(num_tokens: int, context: int, decode: bool) -> float:
    per_m = US_PER_M_DECODE_TOKEN_CONTEXT if decode else US_PER_M_PREFILL_TOKEN_CONTEXT
    return num_tokens * (US_PER_TOKEN + per_m * 1e-6 * (context + num_tokens / 2))


def dp_balance_budget_us(demand_us, decode_us, quantile: float) -> int | None:
    """Next step's per-rank budget: the larger of the highest decode-only time
    (decodes are never clipped) and the ``quantile`` of the busy ranks' demands.
    None when no rank was busy."""
    demand = np.asarray(demand_us)
    busy = demand[demand > 0]
    if busy.size == 0:
        return None
    return int(max(np.max(decode_us), np.quantile(busy, quantile, method="higher")))
