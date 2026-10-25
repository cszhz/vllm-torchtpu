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
"""Whole-step kernel on 4 simulated ranks under the interpreter's cross-core
race detector (step_interpret_check --detect-races), CPU only.

The two configurations reach both all-to-all receive buffers with two
requests per rank (--per-rank 2, both request slots) and with 2-rank
attention groups. Each runs in its own process: the simulated rank count is
fixed when JAX starts.
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.mark.parametrize(
    "args",
    [
        ["--T", "32", "--per-rank", "2"],
        ["--attn-tp", "2", "--T", "16"],
    ],
    ids=["dp_two_slots", "attn_tp2"],
)
def test_step_kernel_has_no_cross_core_races(args):
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_allow_excess_precision=false")  # fmt: skip
    mod = "tests.kernels.deepseek_v4_megakernel.step_interpret_check"
    p = subprocess.run(
        [sys.executable, "-m", mod, "--R", "4", "--detect-races", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800,
    )  # fmt: skip
    assert p.returncode == 0, p.stderr[-4000:]
    lines = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
    assert len(lines) == 1, p.stdout[-4000:]
    out = json.loads(lines[0][len("RESULT ") :])
    reports = p.stdout.count("RACE DETECTED")
    assert not out["races"] and reports == 0, p.stdout[-6000:]
    # the kernel still matches the plain-JAX composition
    assert out["res_rel_korder"] < out["band_res"], out
    assert out["inactive_rows_zero"], out
