#!/bin/bash
set -x
O=${CDK_OUTPUT_DIR:-/cdk-outputs}; mkdir -p $O
for f in true false; do
  LIBTPU_INIT_ARGS="--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=$f" python tests/glue/vmem_pressure.py $O/xprof_adj_$f || echo FAIL adj=$f
done
LIBTPU_INIT_ARGS="--xla_tpu_msa_reduce_scoped_vmem_limit=true" python tests/glue/vmem_pressure.py $O/xprof_reduce || echo FAIL reduce
echo PARITY n/a
