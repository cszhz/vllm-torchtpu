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
"""The object-storage config pull is serialized and done once per server start."""

import os
import time
from unittest.mock import patch

import pytest

from vllm_torchtpu import object_storage_pull_lock as pull_lock

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def storage_model(tmp_path, monkeypatch):
    from vllm.transformers_utils import runai_utils as ru

    pulls = []

    def fake_pull(self, model_path="", allow_pattern=None, ignore_pattern=None):
        pulls.append((model_path, allow_pattern, ignore_pattern))
        (tmp_path / "config.json").write_text("{}")

    monkeypatch.setattr(ru.ObjectStorageModel, "pull_files", fake_pull)
    monkeypatch.setattr(
        ru.ObjectStorageModel, "_tpu_pull_lock_patch", False, raising=False
    )
    pull_lock.patch_object_storage_pull_lock()
    obj = ru.ObjectStorageModel.__new__(ru.ObjectStorageModel)
    obj.dir = str(tmp_path)
    return obj, pulls


def test_the_same_request_is_pulled_once_per_start(storage_model):
    obj, pulls = storage_model
    obj.pull_files("gs://bucket/model", allow_pattern=["*.json"])
    obj.pull_files("gs://bucket/model", allow_pattern=["*.json"])
    assert len(pulls) == 1
    assert any(f.startswith(".pulled-") for f in os.listdir(obj.dir))
    assert os.path.exists(obj.dir.rstrip(os.sep) + ".lock")


def test_a_different_request_or_a_stale_marker_pulls_again(storage_model):
    obj, pulls = storage_model
    obj.pull_files("gs://bucket/model", allow_pattern=["*.json"])
    obj.pull_files("gs://bucket/model", allow_pattern=["*.json", "*.py"])
    assert len(pulls) == 2
    marker = next(f for f in os.listdir(obj.dir) if f.startswith(".pulled-"))
    old = time.time() - 2 * pull_lock._PULL_MARKER_TTL_S
    os.utime(os.path.join(obj.dir, marker), (old, old))
    with patch.object(pull_lock, "_PULL_MARKER_TTL_S", 3600.0):
        obj.pull_files("gs://bucket/model", allow_pattern=["*.json"])
        obj.pull_files("gs://bucket/model", allow_pattern=["*.json", "*.py"])
    assert len(pulls) == 3


def test_the_patch_is_registered_for_the_import_stage():
    from vllm_torchtpu import patch_registry

    stages = {
        p.target: p.stages
        for p in patch_registry.PATCHES
        if p.target.endswith("patch_object_storage_pull_lock")
    }
    assert stages and "import" in next(iter(stages.values()))


def test_the_patch_waits_for_the_module_when_it_is_not_loaded(tmp_path, monkeypatch):
    """The plugin's import stage must not import vllm, so with the module
    absent the patch leaves a one-shot hook that wraps the class the moment
    the module first imports, and removes itself."""
    import sys

    name = "fake_runai_utils_for_the_pull_lock"
    (tmp_path / f"{name}.py").write_text(
        "class ObjectStorageModel:\n"
        "    def pull_files(self, model_path='', allow_pattern=None,\n"
        "                   ignore_pattern=None):\n"
        "        pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(pull_lock, "_PULL_LOCK_MODULE", name)
    monkeypatch.delitem(sys.modules, name, raising=False)
    pull_lock.patch_object_storage_pull_lock()
    hooks = [f for f in sys.meta_path if isinstance(f, pull_lock._AfterImport)]
    assert [h.name for h in hooks] == [name]
    # Idempotent: a second application does not stack a second hook.
    pull_lock.patch_object_storage_pull_lock()
    assert sum(isinstance(f, pull_lock._AfterImport) for f in sys.meta_path) == 1
    module = __import__(name)
    assert module.ObjectStorageModel._tpu_pull_lock_patch is True
    assert module.ObjectStorageModel.pull_files.__name__ == "pull_files"
    assert not any(isinstance(f, pull_lock._AfterImport) for f in sys.meta_path)
    monkeypatch.delitem(sys.modules, name, raising=False)
