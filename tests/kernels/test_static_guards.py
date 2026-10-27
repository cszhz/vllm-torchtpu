# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm_torchtpu.kernels import static_guards


@pytest.fixture(autouse=True)
def _reset():
    yield
    static_guards.set_prefill_only(False)


def test_prefill_only_is_off_until_declared():
    assert static_guards.prefill_only() is False
    static_guards.set_prefill_only()
    assert static_guards.prefill_only() is True
    static_guards.set_prefill_only(False)
    assert static_guards.prefill_only() is False
