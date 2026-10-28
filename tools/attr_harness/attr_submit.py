"""Build/submit the light B-only git-source Falcon profiling run.

usage: attr_submit.py NAME OUT_DIR [--submit]
"""

import json
import subprocess
import sys
from pathlib import Path

FORMAL = "4d35f9e61d81b32cc9668c786d41403b2f714204"
ROOT = Path(__file__).resolve().parents[2]
name, out_dir = sys.argv[1], Path(sys.argv[2])
submit = "--submit" in sys.argv
DIAG = subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
).strip()
script = (
    Path(__file__)
    .with_name("run_git_sources.sh")
    .read_text()
    .replace("__DIAG_SHA__", DIAG)
)
assert len(script) < 11000
# Script is already in the git source; this command only selects it.
cmd = "cd /workspace/diag && exec bash tools/attr_harness/run_git_sources.sh"
task = {
    "name": name,
    "exp_type": "PROFILING",
    "artifact_type": "GCS",
    "cluster_name": "gke-tpu-train-us-central1-2-prod",
    "role_to_task_spec": {
        "worker": {
            "shared_memory_size": "64Gi",
            "image": (
                "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1"
            ),
            "device_type": "v7x",
            "device_count": 8,
            "device_topo": "2x2x1",
            "replica": 1,
            "command": cmd,
            "sources": [
                {
                    "name": "formal",
                    "type": "git",
                    "repo": "https://github.com/vllm-project/vllm-torchtpu.git",
                    "ref": "dev/dsv41-attn-preproc-opt-20261027",
                    "commit": FORMAL,
                    "path": "/workspace/formal",
                    "auth": {"cred": "vllm-torchtpu-carmendong-prod2-20261015"},
                },
                {
                    "name": "diag",
                    "type": "git",
                    "repo": "https://github.com/vllm-project/vllm-torchtpu.git",
                    "ref": "dev/dsv41-attn-preproc-opt-20261027-diag",
                    "commit": DIAG,
                    "path": "/workspace/diag",
                    "auth": {"cred": "vllm-torchtpu-carmendong-prod2-20261015"},
                },
            ],
        }
    },
    "profile": True,
    "tags": ["profile", "operator-optimization"],
}
out_dir.mkdir(parents=True, exist_ok=True)
path = out_dir / f"task_{name}.json"
path.write_text(json.dumps(task, indent=2))
print(json.dumps({"task": str(path), "diag_sha": DIAG, "command_bytes": len(cmd)}))
if submit:
    r = subprocess.run(
        ["falcon", "workflow", "exp", "submit", "-f", str(path), "--output", "json"],
        text=True,
        capture_output=True,
    )
    print(r.stdout)
    print(r.stderr, file=sys.stderr)
    raise SystemExit(r.returncode)
