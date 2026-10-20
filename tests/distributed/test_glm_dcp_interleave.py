# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest


def _raiden_glm_env(monkeypatch, admission):
    monkeypatch.setenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER", "1")
    monkeypatch.setenv("TPU_RAIDEN_GLM_ADMISSION", "1" if admission else "0")


def _interleave_config(connector, *, interleave=4, dcp=16, dcp_interleave=1):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector=connector),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=interleave,
            dcp_kv_cache_interleave_size=dcp_interleave,
        ),
    )


def _kv_cache_config(block_size=256):
    spec = SimpleNamespace(block_size=block_size)
    return SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])


def _unpatched_interleave_adjust(monkeypatch):
    """Start from vLLM's method; monkeypatch restores both after the test."""
    from vllm.config import VllmConfig

    monkeypatch.setattr(
        VllmConfig,
        "adjust_dcp_kv_cache_interleave_size",
        VllmConfig.adjust_dcp_kv_cache_interleave_size,
    )
    monkeypatch.setattr(
        VllmConfig, "_tpu_raiden_keep_interleave_patch", False, raising=False
    )
    return VllmConfig


@pytest.mark.parametrize(
    ("admission", "connector", "expected"),
    [
        (True, "TPURaidenConnector", 4),
        # Other connectors keep vLLM's block-size alignment.
        (True, "NixlConnector", 256),
        # Without Raiden GLM admission nothing is patched.
        (False, "TPURaidenConnector", 256),
    ],
)
def test_raiden_glm_keeps_configured_interleave(
    monkeypatch, admission, connector, expected
):
    import vllm_torchtpu

    config_cls = _unpatched_interleave_adjust(monkeypatch)
    _raiden_glm_env(monkeypatch, admission)
    vllm_torchtpu._patch_raiden_identity_keeps_cp_interleave()
    config = _interleave_config(connector)
    config_cls.adjust_dcp_kv_cache_interleave_size(config, _kv_cache_config())
    assert config.parallel_config.cp_kv_cache_interleave_size == expected


def test_interleave_patch_keeps_deprecated_dcp_override(monkeypatch):
    import vllm_torchtpu

    config_cls = _unpatched_interleave_adjust(monkeypatch)
    _raiden_glm_env(monkeypatch, admission=True)
    vllm_torchtpu._patch_raiden_identity_keeps_cp_interleave()
    vllm_torchtpu._patch_raiden_identity_keeps_cp_interleave()  # idempotent
    config = _interleave_config("TPURaidenConnector", dcp_interleave=64)
    config_cls.adjust_dcp_kv_cache_interleave_size(config, _kv_cache_config())
    assert config.parallel_config.cp_kv_cache_interleave_size == 64
