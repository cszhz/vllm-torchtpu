#!/bin/bash
set -eu
ulimit -c 0
: "${ARTIFACT_LOCAL_DIR:?ARTIFACT_LOCAL_DIR is required}"
export TMPDIR="${TMPDIR:-/tmp/tpu_logs/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/tpu_logs/pip-cache}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" /tmp/res/formal /tmp/res/diag-light /tmp/stubs "$ARTIFACT_LOCAL_DIR"
finish() {
  find /tmp/res -name '*schedule-analysis*' -delete
  tar -czf "$ARTIFACT_LOCAL_DIR/results.tar.gz" -C /tmp res
}
trap finish EXIT
# Falcon git sources are checked out before this script starts. Verify the exact
# immutable commits and use the committed harness; no source is embedded here.
test "$(git -C /workspace/formal rev-parse HEAD)" = 4d35f9e61d81b32cc9668c786d41403b2f714204
test "$(git -C /workspace/diag rev-parse HEAD)" = __DIAG_SHA__
git -C /workspace/formal status --short > /tmp/res/formal_git_status.txt
git -C /workspace/diag status --short > /tmp/res/diag_git_status.txt
for v in formal diag; do
  mkdir -p "/tmp/shim_$v/vllm_torchtpu/layers/core"
  : > "/tmp/shim_$v/vllm_torchtpu/__init__.py"
  : > "/tmp/shim_$v/vllm_torchtpu/layers/__init__.py"
  : > "/tmp/shim_$v/vllm_torchtpu/layers/core/__init__.py"
  ln -s "/workspace/$v/src/vllm_torchtpu/kernels" "/tmp/shim_$v/vllm_torchtpu/kernels"
  cp /workspace/diag/tools/attr_harness/stubs/logger.py /workspace/diag/tools/attr_harness/stubs/utils.py "/tmp/shim_$v/vllm_torchtpu/"
  cp /workspace/diag/tools/attr_harness/stubs/quantization.py "/tmp/shim_$v/vllm_torchtpu/layers/core/"
done
python3 -m pip install -q jax==0.11.2 jaxlib==0.11.2 libtpu==0.0.50 numpy==2.5.4 ml_dtypes
python3 -c 'import jax; print("jax", jax.__version__, jax.devices())' | tee /tmp/res/env.log
export JAX_ENABLE_COMPILATION_CACHE=false
export ATTR_CASES=B_2048,B_288
for mode in formal diag-light; do
  out=/tmp/res/$mode
  mkdir -p "$out/llo_dump" "$out/llo_keep"
  v=$mode
  [ "$mode" = diag-light ] && v=diag
  export PYTHONPATH="/tmp/shim_$v"
  export AOT_OUT="$out" GAP_LLO_DUMP="$out/llo_dump" GAP_LLO_KEEP="$out/llo_keep"
  export LIBTPU_INIT_ARGS="--xla_tpu_dvfs_p_state=7 --xla_jf_dump_to=$out/llo_dump --xla_jf_dump_llo_text=true --xla_xprof_register_llo_debug_info=true --xla_enable_custom_call_region_trace=true --xla_xprof_enable_custom_call_tracing=false"
  python3 -c 'import vllm_torchtpu.kernels.attention_preprocessing.qb_rope as m, os; print("module", os.path.realpath(m.__file__))' > "$out/import.log"
  (cd /workspace/diag/tools/attr_harness && timeout 3600 python3 -u attr_run.py "$mode") > "$out/run.log" 2>&1
  rm -rf "$out/llo_dump"
done
