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
"""One object-storage model pull per server start, across its processes.

Applied at the plugin's import stage (patch_registry.py), which must stay
free of torch, torch_tpu and vllm imports: the plugin import sets the
environment JAX and torch_tpu read before either loads, and importing torch
autoloads torch_tpu. So this module is standard library only, and it reaches
vLLM's ``ObjectStorageModel`` either at once, when its module is already
loaded (it is, inside a vLLM process), or through a one-shot post-import hook.
"""

import fcntl
import hashlib
import importlib.abc
import importlib.util
import json
import os
import sys
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# One object-storage pull per (cache directory, request) per hour, serialized
# across the processes of one server start.
_PULL_MARKER_TTL_S = 3600.0


_PULL_LOCK_MODULE = "vllm.transformers_utils.runai_utils"


class _AfterImport(importlib.abc.MetaPathFinder):
    """Run ``callback(module)`` right after module ``name`` first imports.

    One shot: the hook removes itself from ``sys.meta_path`` the first time
    the import system asks it about ``name`` (so its own lookup of the real
    spec does not come back to it) and wraps the spec's loader so the
    callback runs once the module body has executed.
    """

    def __init__(self, name: str, callback) -> None:
        self.name = name
        self.callback = callback

    def find_spec(self, fullname, path, target=None):
        if fullname != self.name:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is not None and spec.loader is not None:
            spec.loader = _LoaderThen(spec.loader, self.callback)
        return spec


class _LoaderThen(importlib.abc.Loader):
    """The module's own loader, then ``callback(module)``."""

    def __init__(self, loader, callback) -> None:
        self._loader = loader
        self._callback = callback

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        self._callback(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


def patch_object_storage_pull_lock() -> None:
    """Serialize ``ObjectStorageModel.pull_files`` across a server's processes.

    ``ModelConfig.__post_init__`` pulls a ``gs://`` model's ``*.json`` files
    into one shared cache directory (``~/.cache/vllm/assets/model_streamer/
    <hash>``), and every API-server process does it again on its own config:
    at DP32 that is 33 concurrent downloads of the same ``config.json`` into
    one path with no lock, and a reader that lands on another process's
    half-written file dies with "config.json is not a valid JSON file"
    (vLLM's ``repo_utils`` error) and takes the server down. Seen in 3 of 4
    DP32 starts and ≈2 of 5 DP8 starts.

    Wrapped: the pull takes an exclusive ``flock`` on ``<dir>.lock``, and a
    process that finds a fresh marker for the same request (same path and
    patterns, pulled within the last hour) skips the download. The first
    process downloads; the others wait on the lock and reuse its files. A
    marker older than an hour re-pulls, so a server started later still sees
    a changed checkpoint.

    Applied at the plugin's import stage, which must not import vllm itself
    (the plugin import sets the environment JAX and torch_tpu read before
    either loads, and tests/platforms/test_patch_registry.py holds it to
    that), so the class is wrapped at once when its module is already loaded
    -- it is, inside a vLLM process -- and otherwise the moment it first
    imports, by a one-shot hook.
    """
    module = sys.modules.get(_PULL_LOCK_MODULE)
    if module is not None:
        _wrap_object_storage_pull(module)
    elif not any(
        isinstance(f, _AfterImport) and f.name == _PULL_LOCK_MODULE
        for f in sys.meta_path
    ):
        sys.meta_path.insert(
            0, _AfterImport(_PULL_LOCK_MODULE, _wrap_object_storage_pull)
        )


def _wrap_object_storage_pull(module) -> None:
    cls = module.ObjectStorageModel
    if getattr(cls, "_tpu_pull_lock_patch", False):
        return
    original = cls.pull_files

    def pull_files(self, model_path="", allow_pattern=None, ignore_pattern=None):
        request = json.dumps(
            [model_path, allow_pattern, ignore_pattern], sort_keys=True
        )
        marker = os.path.join(
            self.dir, f".pulled-{hashlib.sha256(request.encode()).hexdigest()[:16]}"
        )
        lock_path = self.dir.rstrip(os.sep) + ".lock"
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if (
                    os.path.exists(marker)
                    and time.time() - os.path.getmtime(marker) < _PULL_MARKER_TTL_S
                ):
                    return
                original(self, model_path, allow_pattern, ignore_pattern)
                with open(marker, "w") as f:
                    f.write(request)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    cls.pull_files = pull_files
    cls._tpu_pull_lock_patch = True
    logger.info("Applied TPU patch: serialized object-storage model pulls.")
