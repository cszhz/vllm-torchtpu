#!/usr/bin/env bash
set -euo pipefail

role="${1:?role is required}"
shift
original=("$@")
inner_wrapper="${PD_CAPACITY_PREEMPTION_INNER_WRAPPER:-}"

if [[ "${role}" == "decode" ]]; then
  load_failure_policy=recompute
  if [[ "${PREEMPTION_EXPECT:?PREEMPTION_EXPECT is required}" == "abort" ]]; then
    load_failure_policy=fail
  fi
  kv_config_index=""
  for index in "${!original[@]}"; do
    arg="${original[index]}"
    if [[ "${arg}" == "--kv-transfer-config" ]]; then
      if [[ -n "${kv_config_index}" || $((index + 1)) -ge ${#original[@]} ]]; then
        echo "Decode command has an ambiguous --kv-transfer-config" >&2
        exit 2
      fi
      kv_config_index=$((index + 1))
    fi
  done
  if [[ -z "${kv_config_index}" ]]; then
    echo "Decode command has no --kv-transfer-config to configure" >&2
    exit 2
  fi
  original[kv_config_index]="$(
    python - "${original[kv_config_index]}" "${load_failure_policy}" <<'PY'
import json
import sys

config = json.loads(sys.argv[1])
policy = sys.argv[2]
existing = config.get("kv_load_failure_policy")
if existing not in (None, policy):
    raise SystemExit(
        f"Decode kv_load_failure_policy conflicts with required {policy} policy: "
        f"{existing!r}"
    )
config["kv_load_failure_policy"] = policy
print(json.dumps(config, separators=(",", ":"), sort_keys=True))
PY
  )"
  if [[ -n "${RUN_DIR:-}" ]]; then
    printf 'role=decode kv_load_failure_policy=%s\n' "${load_failure_policy}" \
      >>"${RUN_DIR}/role_wrapper.log"
  fi
fi

if [[ -n "${inner_wrapper}" ]]; then
  exec "${inner_wrapper}" "${role}" "${original[@]}"
fi
exec "${original[@]}"
