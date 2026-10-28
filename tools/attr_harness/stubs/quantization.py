"""Stub of vllm_torchtpu.layers.core.quantization: the real module imports torch.
Only dtype_max is needed (copied verbatim from the epic)."""

import jax.numpy as jnp


def dtype_max(dtype: jnp.dtype) -> float:
    info = jnp.iinfo(dtype) if jnp.issubdtype(dtype, jnp.integer) else jnp.finfo(dtype)
    return float(info.max)
