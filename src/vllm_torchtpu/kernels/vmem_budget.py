# SPDX-License-Identifier: Apache-2.0
"""Scoped-VMEM request of the DeepSeek-V4 Pallas kernels (TPU_DSV4_VMEM_LIMIT_MIB).

A Mosaic call reserves its `vmem_limit_bytes` of VMEM for as long as it runs, and
XLA's memory-space assignment can only keep tensors in VMEM, or prefetch the next
call's operands into it, out of what the running call leaves free. The DSv4
kernels ask for 100 MiB, more than the whole VMEM, so every operand they touch
lives in HBM and every staging copy XLA makes for the next call waits until the
current call ends. Their decode working sets are a few MiB; a smaller request
leaves XLA the rest. A request below a kernel's real need fails compilation
(scoped VMEM OOM), it does not change what the kernel computes.
"""

from vllm_torchtpu import envs

LEGACY_LIMIT_BYTES = 100 * 1024 * 1024


# Kernels whose blocks grow with the token count take the smaller request only
# up to this many tokens (decode buckets); prefill chunks keep the legacy one.
DECODE_MAX_TOKENS = 32


def dsv4_limit_bytes(num_tokens: int | None = None) -> int:
    """The request for a kernel whose blocks are bounded whatever the token count
    (num_tokens=None), or for one at `num_tokens` tokens."""
    mib = envs.TPU_DSV4_VMEM_LIMIT_MIB
    if not mib or (num_tokens is not None and num_tokens > DECODE_MAX_TOKENS):
        return LEGACY_LIMIT_BYTES
    return mib * 1024 * 1024
