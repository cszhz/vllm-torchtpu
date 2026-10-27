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
"""Keep each spawned TPU worker group's logs as CI artifacts.

A worker group occasionally fails to bring the slice up ("START_SESSION
failed ... Max # of tries(=3) exhausted"). One suspect is how the group
before it tore its session down, but that group passed, so its output was
discarded, and libtpu's own logs - which record every session start and stop
with its id and timestamp - stayed in the pod's /tmp and went with it.

When the step sets ARTIFACTS_DIR, which run.sh uploads from the pod, every
launch gets a directory of its own under `tpu_workers/`, in launch order:

  launch.json   test, wall-clock start and finish, return code, and the
                processes holding a TPU device node at either end
  torchrun.log  the workers' combined output
  tpu_logs/     libtpu's per-process logs (TPU_LOG_DIR)
  rank*.json    the per-rank results the workers wrote

Without ARTIFACTS_DIR, nothing is written.
"""

import datetime
import itertools
import json
import os
import re
import shutil
from pathlib import Path

from tests.tpu_release import tpu_device_owners

_SEQ = itertools.count(1)


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds")


def _current_test() -> str:
    # "tests/x/test_y.py::test_z (call)"
    return os.environ.get("PYTEST_CURRENT_TEST", "unknown").rsplit(" ", 1)[0]


class WorkerGroupLog:
    """The log directory of one worker-group launch.

    Build it just before the launch, pass the worker env through `env`, and
    call `finish` once the group has exited or been killed, including on the
    timeout path.
    """

    def __init__(self) -> None:
        self.test = _current_test()
        self.started_at = _now()
        self.dir: Path | None = None
        self.owners_at_start: dict[str, int] = {}
        root = os.environ.get("ARTIFACTS_DIR")
        if not root:
            return
        name = re.sub(r"[^\w.-]+", "_", self.test.rsplit("/", 1)[-1])
        self.dir = (
            Path(root).resolve()
            / "tpu_workers"
            / f"{next(_SEQ):02d}-{os.getpid()}-{name}"
        )
        (self.dir / "tpu_logs").mkdir(parents=True)
        self.owners_at_start = tpu_device_owners()
        print(f"TPU worker logs: {self.dir}")

    def env(self, env: dict[str, str]) -> dict[str, str]:
        if self.dir is None:
            return env
        return {**env, "TPU_LOG_DIR": str(self.dir / "tpu_logs")}

    def finish(
        self,
        returncode: int | None,
        output: str,
        result_dir: Path | None = None,
    ) -> None:
        if self.dir is None:
            return
        (self.dir / "torchrun.log").write_text(output, encoding="utf-8")
        if result_dir is not None:
            for path in sorted(result_dir.glob("rank*.json")):
                shutil.copy(path, self.dir / path.name)
        # libtpu points `tpu_driver.INFO` and friends at the newest file of
        # each level; uploading them would store every log twice.
        for path in (self.dir / "tpu_logs").iterdir():
            if path.is_symlink():
                path.unlink()
        record = {
            "test": self.test,
            "started_at": self.started_at,
            "finished_at": _now(),
            "returncode": returncode,
            "device_owners_at_start": self.owners_at_start,
            "device_owners_at_finish": tpu_device_owners(),
        }
        (self.dir / "launch.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
