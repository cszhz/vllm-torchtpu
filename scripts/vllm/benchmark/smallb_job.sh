#!/bin/bash
# Job body for p3_d_leg_bench.py (small-batch decode study) in a simple-run-command pod.
# Usage: bash scripts/vllm/benchmark/smallb_job.sh <tag> <p3_d_leg_bench args...>
set -o pipefail
TAG=$1; shift
W=/workspace/vllm-torchtpu
for d in /root/torchtpu-vllm/src /root/cloud-devkit/vllm-torchtpu/src; do
  [ -d "$d" ] && cp -rf $W/src/vllm_torchtpu "$d/" || true
done
export PYTHONPATH="$W/src:$W${PYTHONPATH:+:$PYTHONPATH}"
# Run on local disk: the harness appends one JSON line per engine step to <out>/steps.jsonl
# (open/append/close per record); on the gcsfuse-backed CDK_OUTPUT_DIR every close re-uploads the
# object, which stalled the engine core to ~140 ms per step in j-5a00436f. Copy out at the end.
FINAL=${CDK_OUTPUT_DIR:-/tmp/out}/$TAG
OUT=/tmp/sbout/$TAG
mkdir -p $OUT $FINAL
cd $W
echo "SMALLB_SRC $(cat $W/SMALLB_COMMIT 2>/dev/null) date=$(date -u +%FT%TZ)"
python3 -c 'import vllm, vllm_torchtpu; print("SMALLB_IMPORT", vllm.__version__, vllm_torchtpu.__file__)'
python3 -u scripts/vllm/benchmark/p3_d_leg_bench.py --tag $TAG --out-dir $OUT \
  --model-dir $W/qwen35-397b-config --cache-root /tmp/p3cache "$@" 2>&1 | tee $OUT/bench.log
rc=$?
echo "SMALLB_BENCH_EXIT=$rc"
sleep 60  # let workers finish writing traces
if [ -d $OUT/xprof ]; then
  find $OUT/xprof -type f | head -80; du -sh $OUT/xprof
fi
if [ -d $OUT/xprof ] && [ "${XPLANE_IN_POD:-0}" = "1" ]; then
  find $OUT/xprof -name '*.xplane.pb' | head -40
  for f in $(find $OUT/xprof -name '*.xplane.pb' | sort | head -${XPLANE_MAX_FILES:-64}); do
    b=$(echo "$f" | sed "s#$OUT/xprof/##; s#/#_#g")
    python3 -u scripts/vllm/benchmark/xplane_ops.py "$f" --steps ${PROFILE_STEPS:-1} --top 120 \
      --json $OUT/ops_$b.json > $OUT/ops_$b.txt 2>&1 || echo "xplane_ops failed on $f"
  done
fi
cp -r $OUT/. $FINAL/ && echo "SMALLB_COPIED to $FINAL"
echo "SMALLB_DONE rc=$rc"
exit $rc
