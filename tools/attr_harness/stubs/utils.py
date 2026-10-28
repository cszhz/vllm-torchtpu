"""Stub of vllm_torchtpu.utils: the real module imports torch and vllm.
Only largest_divisor is needed (copied verbatim from the epic)."""


def largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of `x` that is <= `cap`."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1
