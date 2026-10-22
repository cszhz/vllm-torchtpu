#!/usr/bin/env bash
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

set -euo pipefail
ulimit -c 0
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SKIP_JAX_PRECOMPILE=1

: "${ARTIFACT_LOCAL_DIR:?Falcon artifact directory is required}"
: "${MOE_RUN_NAME:?Set a unique run name}"
run_local="/workspace/moe-results/$MOE_RUN_NAME"
run_artifact="$ARTIFACT_LOCAL_DIR"
test ! -e "$run_local"
# One benchmark per Falcon experiment: analyzers read the artifact root.
for entry in manifest.json profiling rank-{0..7}; do
    test ! -e "$run_artifact/$entry"
done
mkdir -p "$run_local" "$run_artifact"

publish() {
    code=$?
    trap - EXIT
    printf '%s\n' "$code" > "$run_local/exit-status.txt"
    if [ "$code" -eq 0 ]; then
        if python3 - "$run_local" <<'PY'
import sys
from pathlib import Path
from scripts.dsv41_bench.bench_prefill_moe import export_falcon_artifact

run = Path(sys.argv[1])
export_falcon_artifact(run, run / "artifact")
PY
        then
            if ! python3 - "$run_local/artifact" "$run_artifact" <<'PY'
import sys
from pathlib import Path
from scripts.dsv41_bench.bench_prefill_moe import publish_falcon_artifact

publish_falcon_artifact(Path(sys.argv[1]), Path(sys.argv[2]))
PY
            then
                echo "ARTIFACT_PUBLISH_FAILED source=$run_local" >&2
                code=1
            fi
        else
            echo "ARTIFACT_CONTRACT_FAILED source=$run_local" >&2
            code=1
        fi
    fi
    if [ "$code" -ne 0 ]; then
        printf '%s\n' "$code" > "$run_local/exit-status.txt"
        failure_dir="$run_artifact/failed/$MOE_RUN_NAME"
        if ! mkdir -p "$failure_dir" || ! cp -R "$run_local/." "$failure_dir/"; then
            echo "ARTIFACT_FAILURE_LOG_PUBLISH_FAILED source=$run_local" >&2
            exit 1
        fi
    fi
    echo "RUN_FINISHED exit=$code artifact=$run_artifact"
    exit "$code"
}
trap publish EXIT
exec > >(tee "$run_local/launcher.log") 2>&1

echo "RUNTIME_SETUP_START"
export VLLM_TARGET_DEVICE=empty VLLM_PLUGINS="" TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TMPDIR="${TMPDIR:-/tmp/tpu_logs/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/tpu_logs/pip-cache}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$run_local/setup"
python3 - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 12):
    raise RuntimeError("The mounted TorchTPU wheel requires Python 3.12")
if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
    raise RuntimeError("The mounted TorchTPU wheel requires Linux x86_64")
PY
export VIRTUAL_ENV="/tmp/tpu_logs/moe-venvs/$MOE_RUN_NAME"
test ! -e "$VIRTUAL_ENV"
python3 -m venv "$VIRTUAL_ENV"
export PATH="$VIRTUAL_ENV/bin:$PATH"
python3 -m pip install --upgrade pip setuptools wheel packaging tomli-w

torch_tpu_wheel=/wheels/torch-tpu/torch_tpu-0.2.0rc1-cp312-cp312-manylinux_2_31_x86_64.whl
python3 - "$run_local/setup" "$VIRTUAL_ENV/plugin" "$torch_tpu_wheel" <<'PY'
import hashlib
from pathlib import Path
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import tomli_w

from scripts.dsv41_bench.bench_prefill_moe import RUNTIME_PINS, VLLM_COMMIT

output, plugin, wheel = map(Path, sys.argv[1:])
expected = '915bf99d453e33b46933ad9baa37af24b930dcafc5e015a75f9f2b7943087d38'
actual = hashlib.sha256(wheel.read_bytes()).hexdigest()
if actual != expected:
    raise RuntimeError(f'TorchTPU wheel hash mismatch: {actual}')
(output / 'torch-tpu-wheel.sha256').write_text(f'{actual}  {wheel}\n')
vllm_source = Path('/workspace/vllm')
commit = subprocess.check_output(
    ['git', '-C', str(vllm_source), 'rev-parse', 'HEAD'], text=True,
).strip()
if commit != VLLM_COMMIT:
    raise RuntimeError(f'vLLM source: expected {VLLM_COMMIT}, found {commit}')
subprocess.run(['git', '-C', str(vllm_source), 'diff', '--exit-code', 'HEAD'], check=True)

repo = Path.cwd()
config = tomllib.loads((repo / 'pyproject.toml').read_text())
dependencies = []
cpu_requirements = []
for text in config['project']['dependencies']:
    name = canonicalize_name(Requirement(text).name)
    if name == 'vllm':
        text = 'vllm'
    elif name in RUNTIME_PINS:
        text = f'{name}=={RUNTIME_PINS[name]}'
    dependencies.append(text)
    if name in ('torch', 'torchvision'):
        cpu_requirements.append(text)
dependencies.extend(f'{name}=={version}' for name, version in RUNTIME_PINS.items())
config['project']['dependencies'] = list(dict.fromkeys(dependencies))
# Editable sources stay in the checkout; only this venv's dependency metadata
# uses the benchmark pins instead of the serving environment's default pins.
config['project'].pop('dynamic')
config['project']['version'] = '0.0.0'
config['tool'].pop('setuptools_scm')
plugin.mkdir()
for name in ('src', 'README.md'):
    (plugin / name).symlink_to(repo / name)
runtime_project = tomli_w.dumps(config)
(plugin / 'pyproject.toml').write_text(runtime_project)
(output / 'runtime-pyproject.toml').write_text(runtime_project)
(output / 'constraints.txt').write_text(''.join(
    f'{name}=={version}\n' for name, version in RUNTIME_PINS.items()
))
(output / 'cpu-requirements.txt').write_text('\n'.join(cpu_requirements) + '\n')
build_requirements = set(config['build-system']['requires'])
build_requirements.update(
    tomllib.loads((vllm_source / 'pyproject.toml').read_text())['build-system']['requires']
)
build_requirements.update(('cmake', 'ninja', 'pybind11'))
(output / 'build-requirements.txt').write_text('\n'.join(sorted(build_requirements)) + '\n')
(output / 'vllm-version.txt').write_text(f'0.1.dev1+g{VLLM_COMMIT[:9]}.empty\n')
PY
python3 -m pip install --constraint "$run_local/setup/constraints.txt" \
    --index-url https://download.pytorch.org/whl/cpu \
    --requirement "$run_local/setup/cpu-requirements.txt"
python3 -m pip install --constraint "$run_local/setup/constraints.txt" \
    --requirement "$run_local/setup/build-requirements.txt"
VLLM_VERSION_OVERRIDE="$(cat "$run_local/setup/vllm-version.txt")" \
    python3 -m pip install --no-build-isolation \
    --constraint "$run_local/setup/constraints.txt" \
    --editable /workspace/vllm --editable "$VIRTUAL_ENV/plugin" "$torch_tpu_wheel" \
    'runai-model-streamer[gcs]==0.17.0' 'runai-model-streamer-gcs==0.17.0' \
    'frozendict==2.4.7' 'immutabledict==4.3.1' 'tensorboard==2.21.0'
python3 -m pip check | tee "$run_local/setup/pip-check.txt"
python3 -m pip freeze > "$run_local/setup/pip-freeze.txt"
unset TORCH_DEVICE_BACKEND_AUTOLOAD
export VLLM_TARGET_DEVICE=tpu VLLM_PLUGINS=torchtpu
echo "RUNTIME_SETUP_COMPLETE"

python3 - "$run_local" <<'PY'
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
from scripts.dsv41_bench.bench_prefill_moe import check_runtime

out = Path(sys.argv[1])
(out / 'python-packages.txt').write_text(''.join(
    f'{name}=={version}\n' for name, version in sorted(
        (dist.metadata['Name'], dist.version) for dist in md.distributions()
    )
))
manifest = json.loads(Path('.dsv41_bench/source-files.json').read_text())
for name, expected in manifest['files'].items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Source hash mismatch: {name}')
versions = check_runtime()
(out / 'runtime-versions.json').write_text(json.dumps(versions, indent=2))
(out / 'source-files.json').write_text(json.dumps(manifest, indent=2))
print('SOURCE_AND_RUNTIME_OK', flush=True)
PY

run_capture() {
    phase=$1
    warmup=$2
    steps=$3
    shift 3
    mkdir -p "$run_local/$phase"
    echo "PHASE_START $phase global_tokens=16384 local_tokens=2048 tp=1 dp=8 ep=8 token_layout=sharded_no_replicas"
    local status=0
    timeout --signal=TERM --kill-after=30s 1200s \
        python3 -u -m scripts.dsv41_bench.bench_prefill_moe \
        --global-tokens 16384 --tp-size 1 --dp-size 8 --warmup "$warmup" \
        --profile-steps "$steps" --seed 0 \
        --result-dir "$run_local/$phase/results" \
        --profile "$run_local/$phase/xprof" \
        --golden-dir "$run_local/golden" "$@" \
        > "$run_local/$phase/bench.log" 2>&1 || status=$?
    if [ "$status" -ne 0 ]; then
        echo "PHASE_FAILED $phase exit=$status"
        tail -n 200 "$run_local/$phase/bench.log"
        if [ -f "$run_local/$phase/results/workers.log" ]; then
            tail -n 300 "$run_local/$phase/results/workers.log"
        fi
        return "$status"
    fi
    test -s "$run_local/$phase/results/device-metrics.json"
    echo "PHASE_COMPLETE $phase"
}

run_capture smoke 1 1 --update-golden
run_capture baseline 5 5 --verify --dump-llo "$run_local/baseline/compiler"
echo "BASELINE_DEVICE_CAPTURE_OK"
