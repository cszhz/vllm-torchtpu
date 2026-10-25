#!/bin/bash
set -x
O=${CDK_OUTPUT_DIR:-/cdk-outputs}; mkdir -p $O
python tests/glue/vmem_pressure.py $O/xprof_vmem || exit 1
TPU_PIN_OPERANDS_HBM=proj python tests/glue/vmem_pressure.py $O/xprof_vmem_pin || exit 2
echo PARITY n/a
