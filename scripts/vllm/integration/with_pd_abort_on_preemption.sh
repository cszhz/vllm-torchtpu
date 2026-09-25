#!/usr/bin/env bash
set -euo pipefail

role="${1:?role is required}"
shift
original=("$@")
inner_wrapper="${PD_CAPACITY_PREEMPTION_INNER_WRAPPER:-}"

if [[ "${role}" == "decode" ]]; then
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
    if [[ "${arg}" == "--additional-config" ]]; then
      echo "Decode command already has --additional-config; refusing to replace it" >&2
      exit 2
    fi
  done
  if [[ -z "${kv_config_index}" ]]; then
    echo "Decode command has no --kv-transfer-config to configure" >&2
    exit 2
  fi
  original[kv_config_index]="$(
    python - "${original[kv_config_index]}" <<'PY'
import json
import sys

config = json.loads(sys.argv[1])
existing = config.get("kv_load_failure_policy")
if existing not in (None, "fail"):
    raise SystemExit(
        "Decode kv_load_failure_policy conflicts with required fail policy: "
        f"{existing!r}"
    )
config["kv_load_failure_policy"] = "fail"
print(json.dumps(config, separators=(",", ":"), sort_keys=True))
PY
  )"
  if [[ "${PREEMPTION_EXPECT:?PREEMPTION_EXPECT is required}" == "abort" ]]; then
    original+=(--additional-config '{"abort_on_preemption":true}')
  fi
  if [[ -n "${RUN_DIR:-}" ]]; then
    printf '%s\n' 'role=decode kv_load_failure_policy=fail' \
      >>"${RUN_DIR}/role_wrapper.log"
    if [[ "${PREEMPTION_EXPECT}" == "abort" ]]; then
      printf '%s\n' \
        'role=decode additional_config={"abort_on_preemption":true}' \
        >>"${RUN_DIR}/role_wrapper.log"
    fi
  fi
fi

if [[ -n "${inner_wrapper}" ]]; then
  exec "${inner_wrapper}" "${role}" "${original[@]}"
fi
exec "${original[@]}"
