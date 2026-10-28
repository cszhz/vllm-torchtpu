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
"""libtpu identification when the pip package may be absent."""

from importlib import metadata as importlib_metadata

import pytest

from vllm_torchtpu import tpu_info
from vllm_torchtpu.runner import tpu_runner


def _versions(libtpu):
    def version(name):
        if name == "libtpu" and libtpu is None:
            raise importlib_metadata.PackageNotFoundError(name)
        return libtpu if name == "libtpu" else "1"

    return version


def test_get_libtpu_version_uses_package_metadata(monkeypatch):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions("0.0.50"))

    assert tpu_info.get_libtpu_version() == "0.0.50"
    assert tpu_info.get_libtpu_cache_key() == "0.0.50"


def test_get_libtpu_version_without_package(monkeypatch):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions(None))

    assert tpu_info.get_libtpu_version() is None


def test_libtpu_cache_key_fingerprints_tpu_library_path(monkeypatch, tmp_path):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions(None))
    first = tmp_path / "a" / "libtpu.so"
    second = tmp_path / "b" / "libtpu.so"
    for path, payload in ((first, b"build-a" * 4096), (second, b"build-b" * 4096)):
        path.parent.mkdir()
        path.write_bytes(payload)

    monkeypatch.setenv("TPU_LIBRARY_PATH", str(first))
    key_first = tpu_info.get_libtpu_cache_key()
    assert key_first.startswith("unpackaged-")
    assert tpu_info.get_libtpu_cache_key() == key_first

    monkeypatch.setenv("TPU_LIBRARY_PATH", str(second))
    assert tpu_info.get_libtpu_cache_key() != key_first


def test_libtpu_cache_key_without_package_or_library(monkeypatch, tmp_path):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions(None))
    monkeypatch.setenv("TPU_LIBRARY_PATH", str(tmp_path / "missing.so"))

    assert tpu_info.get_libtpu_cache_key() == "unpackaged-unknown"


def test_validate_libtpu_version_skips_without_package(monkeypatch):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions(None))

    tpu_runner._validate_libtpu_version()


def test_validate_libtpu_version_still_rejects_old_package(monkeypatch):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions("0.0.34"))

    with pytest.raises(RuntimeError, match="libtpu < 0.0.35"):
        tpu_runner._validate_libtpu_version()
