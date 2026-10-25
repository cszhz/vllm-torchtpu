#!/bin/bash
# Both modes in separate processes, then compare; xprof dirs go to the job outputs.
set -x
O=${CDK_OUTPUT_DIR:-/cdk-outputs}; mkdir -p $O
TPU_PIN_OPERANDS_HBM= python tests/glue/pin_parity.py run /tmp/off.npz $O/xprof_off || exit 1
TPU_PIN_OPERANDS_HBM=${PIN_SITES:-all} python tests/glue/pin_parity.py run /tmp/on.npz $O/xprof_on || exit 2
python tests/glue/pin_parity.py compare /tmp/off.npz /tmp/on.npz
