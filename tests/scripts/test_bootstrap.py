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
"""Unit tests for .buildkite/scripts/bootstrap.sh and pipeline_config.sh."""

import json
import os
import pathlib
import stat
import subprocess
import tempfile
import unittest

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BOOTSTRAP = _REPO_ROOT / ".buildkite" / "scripts" / "bootstrap.sh"
_TESTS_LANE = _REPO_ROOT / ".buildkite" / "pipeline_tests.yml"

# Logs the environment the real agent would interpolate into a pipeline at
# upload, next to the call itself.
_FAKE_AGENT = """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${TEST_AGENT_LOG}/calls.log"
if [[ "$1" == "pipeline" && "$2" == "upload" ]]; then
  log="${TEST_AGENT_LOG}/calls.log"
  echo "upload-env TPU_STEPS_SKIP=${TPU_STEPS_SKIP:-}" >> "$log"
  echo "upload-env OMNI_IMAGE_SKIP=${OMNI_IMAGE_SKIP:-}" >> "$log"
  shopt -s nullglob
  if [[ "${3:-}" == "--dry-run" ]]; then
    dfiles=("${TEST_AGENT_LOG}"/dry_run_*.yml)
    cat > "${TEST_AGENT_LOG}/dry_run_${#dfiles[@]}.yml"
    echo '{"steps": [{"label": "stub", "agents": {"queue": "kube"}}]}'
  else
    ufiles=("${TEST_AGENT_LOG}"/upload_*.yml)
    cat > "${TEST_AGENT_LOG}/upload_${#ufiles[@]}.yml"
  fi
fi
"""

_FAKE_GIT = """#!/usr/bin/env bash
if [[ "${1:-}" == "fetch" ]]; then
  exit 0
fi
if [[ "${1:-}" == "diff" || "${1:-}" == "diff-tree" ]]; then
  echo "src/vllm_torchtpu/runner/tpu_runner.py"
  exit 0
fi
exec /usr/bin/git "$@"
"""

# The agent's GitHub App credential helper, which bootstrap.sh asks for a token
# to read a merged PR's labels.
_FAKE_GITHUB_APP_HELPER = """#!/usr/bin/env bash
echo "password=test-token"
"""

# No network in tests: a label lookup that reaches curl fails, as it would in
# an outage, unless a test installs a curl that answers.
_FAKE_CURL_FAILS = """#!/usr/bin/env bash
echo "$*" >> "${TEST_AGENT_LOG}/curl.log"
exit 22
"""


def _fake_curl_with_labels(*labels: str) -> str:
    """A curl that answers GitHub's issue labels request with these labels."""
    body = json.dumps([{"name": label} for label in labels])
    return f"""#!/usr/bin/env bash
echo "$*" >> "${{TEST_AGENT_LOG}}/curl.log"
printf '%s' '{body}'
"""


def run_bootstrap(
    curl: str = _FAKE_CURL_FAILS,
    **env: str,
) -> tuple[list[str], list[str], list[str], dict[str, str | list[str]]]:
    """Run bootstrap.sh against a stub buildkite-agent."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, script in (
            ("buildkite-agent", _FAKE_AGENT),
            ("git", _FAKE_GIT),
            ("curl", curl),
            ("git-credential-github-app", _FAKE_GITHUB_APP_HELPER),
        ):
            path = bin_dir / name
            path.write_text(script)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)

        # Provide a clean base environment so host Buildkite vars never leak in.
        run_env = {
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(tmp_path),
            "TEST_AGENT_LOG": str(tmp_path),
            "GITHUB_APP_CREDENTIAL_HELPER": str(bin_dir / "git-credential-github-app"),
            "BUILDKITE_COMMIT": "HEAD",
            "BUILDKITE_BRANCH": "feature-branch",
            "BUILDKITE_PULL_REQUEST": "false",
            "BUILDKITE_PULL_REQUEST_LABELS": "",
            "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci",
            **env,
        }
        proc = subprocess.run(
            ["bash", str(_BOOTSTRAP)],
            cwd=_REPO_ROOT,
            env=run_env,
            capture_output=True,
            text=True,
            check=True,
        )
        calls_file = tmp_path / "calls.log"
        calls = calls_file.read_text().splitlines() if calls_file.exists() else []
        curl_file = tmp_path / "curl.log"
        curl_calls = curl_file.read_text().splitlines() if curl_file.exists() else []
        uploads = [p.read_text() for p in sorted(tmp_path.glob("upload_*.yml"))]
        dry_runs = [p.read_text() for p in sorted(tmp_path.glob("dry_run_*.yml"))]
        return (
            calls,
            uploads,
            dry_runs,
            {"stdout": proc.stdout, "stderr": proc.stderr, "curl": curl_calls},
        )


class TestBootstrap(unittest.TestCase):
    """Verify lane selection, priority, and NIGHTLY substitution in bootstrap.sh."""

    def test_plain_pr_skips_perf_and_keeps_nightly_unset(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready",
        )
        # 3 lane files (verify, integration, tests) + pipeline_build.yml
        self.assertEqual(len(uploads), 4)
        self.assertNotIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        self.assertIn("meta-data set JOB_PRIORITY 6", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY pre-merge", calls)
        for content in uploads:
            self.assertTrue(content.startswith("priority: 6\n"))
        for content in uploads[:-1]:
            self.assertIn('build.env("NIGHTLY")', content)
        self.assertFalse(any("annotate" in c for c in calls))
        # Nothing skips the TPU steps or the -omni image build.
        self.assertIn("upload-env TPU_STEPS_SKIP=", calls)
        self.assertIn("upload-env OMNI_IMAGE_SKIP=", calls)
        self.assertNotIn("upload-env TPU_STEPS_SKIP=skip-tpu-ci", calls)

    def test_skip_tpu_ci_label_keeps_the_cpu_steps(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,skip-tpu-ci",
        )
        # The tests lane + pipeline_build.yml: no verify or integration lane.
        self.assertEqual(len(uploads), 2)
        self.assertIn("Uploading .buildkite/pipeline_tests.yml", out["stdout"])
        self.assertNotIn("pipeline_integration.yml", out["stdout"])
        self.assertNotIn("pipeline_verify.yml", out["stdout"])
        self.assertTrue(
            any(
                "--context skip-tpu-ci" in c and "skip-tpu-ci` label" in c
                for c in calls
            )
        )
        # The agent interpolates the skip reason into the lane's TPU steps, and
        # into the -omni image build nothing left uses.
        self.assertIn("upload-env TPU_STEPS_SKIP=skip-tpu-ci", calls)
        self.assertIn(
            "upload-env OMNI_IMAGE_SKIP=Only the omni TPU tests use the -omni image; "
            "this build skips them.",
            calls,
        )
        self.assertIn('skip: "${TPU_STEPS_SKIP:-}"', uploads[0])
        self.assertEqual(out["curl"], [])

    def test_skip_tpu_in_the_environment_works_without_the_label(self) -> None:
        calls, uploads, _, _ = run_bootstrap(SKIP_TPU="1")
        self.assertEqual(len(uploads), 2)
        self.assertTrue(
            any("--context skip-tpu-ci" in c and "SKIP_TPU=1" in c for c in calls)
        )
        self.assertIn("upload-env TPU_STEPS_SKIP=skip-tpu-ci", calls)

    def test_skip_tpu_ci_label_wins_over_run_full_ci(self) -> None:
        calls, uploads, _, _ = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci,skip-tpu-ci",
        )
        # The tests lane, with NIGHTLY substituted, + pipeline_build.yml.
        self.assertEqual(len(uploads), 2)
        self.assertNotIn('build.env("NIGHTLY")', uploads[0])
        self.assertIn("upload-env TPU_STEPS_SKIP=skip-tpu-ci", calls)

    def test_post_merge_build_of_a_skip_tpu_ci_pr_skips_the_tpu_steps(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            curl=_fake_curl_with_labels("ready", "skip-tpu-ci"),
            BUILDKITE_BRANCH="main",
            BUILDKITE_SOURCE="webhook",
            BUILDKITE_MESSAGE="[Frontend] A change (#2449)\n\nSigned-off-by: A <a@b.c>",
        )
        self.assertEqual(len(uploads), 2)
        self.assertTrue(any("issues/2449/labels" in c for c in out["curl"]))
        self.assertTrue(
            any("--context skip-tpu-ci" in c and "#2449 has" in c for c in calls)
        )
        self.assertIn("upload-env TPU_STEPS_SKIP=skip-tpu-ci", calls)
        # Still a post-merge build.
        self.assertIn("meta-data set WORKLOAD_PRIORITY post-merge", calls)

    def test_post_merge_build_of_an_unlabeled_pr_runs_everything(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            curl=_fake_curl_with_labels("ready"),
            BUILDKITE_BRANCH="main",
            BUILDKITE_SOURCE="webhook",
            BUILDKITE_MESSAGE="[Core] A change (#2450)",
        )
        self.assertEqual(len(uploads), 4)
        self.assertTrue(any("issues/2450/labels" in c for c in out["curl"]))
        self.assertFalse(any("--context skip-tpu-ci" in c for c in calls))
        self.assertIn("upload-env TPU_STEPS_SKIP=", calls)
        self.assertNotIn("upload-env TPU_STEPS_SKIP=skip-tpu-ci", calls)

    def test_post_merge_build_runs_everything_when_the_label_lookup_fails(
        self,
    ) -> None:
        # The default curl fails every request.
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_BRANCH="main",
            BUILDKITE_SOURCE="webhook",
            BUILDKITE_MESSAGE="[Core] A change (#2450)",
        )
        self.assertEqual(len(uploads), 4)
        self.assertTrue(any("issues/2450/labels" in c for c in out["curl"]))
        self.assertFalse(any("--context skip-tpu-ci" in c for c in calls))

    def test_scheduled_build_on_main_does_not_look_up_labels(self) -> None:
        _, uploads, _, out = run_bootstrap(
            curl=_fake_curl_with_labels("skip-tpu-ci"),
            BUILDKITE_BRANCH="main",
            BUILDKITE_SOURCE="schedule",
            BUILDKITE_MESSAGE="[Core] A change (#2450)",
        )
        self.assertEqual(len(uploads), 4)
        self.assertEqual(out["curl"], [])

    def test_integration_pipeline_ignores_the_label(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            curl=_fake_curl_with_labels("skip-tpu-ci"),
            BUILDKITE_PIPELINE_SLUG="vllm-torchtpu-integration",
            BUILDKITE_BRANCH="main",
            BUILDKITE_SOURCE="webhook",
            BUILDKITE_MESSAGE="[Core] A change (#2450)",
        )
        # Every lane + promote + pipeline_build.yml, nothing skipped.
        self.assertEqual(len(uploads), 6)
        self.assertEqual(out["curl"], [])
        self.assertNotIn("upload-env TPU_STEPS_SKIP=skip-tpu-ci", calls)

    def test_every_tpu_step_of_the_tests_lane_reads_the_skip(self) -> None:
        """Only the TPU steps skip: a CPU step that read TPU_STEPS_SKIP would
        defeat the point of the skip-tpu-ci label."""
        lane = yaml.safe_load(_TESTS_LANE.read_text())

        def steps(items):
            for step in items:
                if isinstance(step.get("steps"), list):
                    yield from steps(step["steps"])
                else:
                    yield step

        seen_tpu = seen_cpu = 0
        for step in steps(lane["steps"]):
            on_kube = (step.get("agents") or {}).get("queue") == "kube"
            if on_kube and (step.get("env") or {}).get("SHAPE") != "cpu":
                seen_tpu += 1
                self.assertEqual(step.get("skip"), "${TPU_STEPS_SKIP:-}", step["key"])
            else:
                seen_cpu += 1
                self.assertNotIn("skip", step, step["key"])
        self.assertGreater(seen_tpu, 0)
        self.assertGreater(seen_cpu, 0)

    def test_run_perf_eval_adds_perf_lane_without_nightly(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-perf-eval",
        )
        # 4 lane files (verify, perf, integration, tests) + pipeline_build.yml
        self.assertEqual(len(uploads), 5)
        self.assertIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        self.assertTrue(any("--context perf-eval" in c for c in calls))
        self.assertFalse(any("--context full-ci" in c for c in calls))
        for content in uploads[:-1]:
            self.assertIn('build.env("NIGHTLY")', content)

    def test_run_full_ci_enables_all_lanes_and_substitutes_nightly(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci",
        )
        self.assertEqual(len(uploads), 5)
        self.assertIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        # Priority remains pre-merge (6), not nightly (1).
        self.assertIn("meta-data set JOB_PRIORITY 6", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY pre-merge", calls)
        self.assertTrue(any("--context full-ci" in c for c in calls))
        self.assertFalse(any("--context perf-eval" in c for c in calls))
        for content in uploads:
            self.assertTrue(content.startswith("priority: 6\n"))
            self.assertNotIn('build.env("NIGHTLY")', content)
            parsed = yaml.safe_load(content)
            self.assertIsInstance(parsed, dict)
            self.assertIn("steps", parsed)

    def test_run_full_ci_with_oncall_fix_keeps_oncall_priority(self) -> None:
        calls, uploads, _, _ = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci,oncall-fix",
        )
        self.assertEqual(len(uploads), 5)
        self.assertIn("meta-data set JOB_PRIORITY 12", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY oncall-fix", calls)
        self.assertTrue(any("--context full-ci" in c for c in calls))
        for content in uploads:
            self.assertTrue(content.startswith("priority: 12\n"))
            self.assertNotIn('build.env("NIGHTLY")', content)

    def test_run_full_ci_with_author_caps_tpu_concurrency_with_nightly_substituted(
        self,
    ) -> None:
        _, uploads, dry_runs, _ = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci",
            BUILDKITE_BUILD_AUTHOR_EMAIL="Alice@example.com",
        )
        self.assertEqual(len(uploads), 5)
        self.assertEqual(len(dry_runs), 5)
        for raw_yaml in dry_runs:
            self.assertNotIn('build.env("NIGHTLY")', raw_yaml)
        for content in uploads:
            doc = json.loads(content)
            self.assertEqual(
                doc["steps"][0]["concurrency_group"],
                "tpu-pr/alice@example.com",
            )

    def test_scheduled_nightly_keeps_nightly_priority(self) -> None:
        calls, uploads, _, _ = run_bootstrap(
            BUILDKITE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            NIGHTLY="1",
        )
        self.assertEqual(len(uploads), 5)
        self.assertIn("meta-data set JOB_PRIORITY 1", calls)
        self.assertFalse(any("WORKLOAD_PRIORITY" in c for c in calls))
        self.assertFalse(any("--context full-ci" in c for c in calls))
        for content in uploads:
            self.assertTrue(content.startswith("priority: 1\n"))
            self.assertNotIn('build.env("NIGHTLY")', content)


if __name__ == "__main__":
    unittest.main()
