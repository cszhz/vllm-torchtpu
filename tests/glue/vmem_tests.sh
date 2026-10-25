#!/bin/bash
# Kernel unit tests at the default DSv4 scoped-VMEM request (twice: run-to-run determinism) and at
# ${CAP:-32} MiB: every run must pass, and every recorded kernel output must be byte-identical.
set -x
O=${CDK_OUTPUT_DIR:-/cdk-outputs}; mkdir -p $O
T=${TESTS:-"tests/kernels/deepseek_v4/mhc_test.py tests/kernels/deepseek_v4/rope_test.py tests/kernels/deepseek_v4/o_projection_test.py tests/kernels/deepseek_v4/mla_test.py tests/kernels/deepseek_v4/mla_swa_test.py tests/kernels/deepseek_v4/sparse_mla_test.py tests/kernels/test_dsv4_streamindex_topk.py tests/kernels/test_masked_dense_mla.py"}
export PYTHONPATH=$PWD/tests/glue:$PYTHONPATH
run() {  # run <tag> <cap>
  TPU_DSV4_VMEM_LIMIT_MIB=$2 RECORD_ACTUALS=/tmp/act_$1.npz timeout 2400 python -m pytest -p record_actuals -q --maxfail=60 -p no:cacheprovider $T 2>&1 | tail -70 | tee $O/pytest_$1.txt
}
run base 0; run base2 0; run cap ${CAP:-32}
python tests/glue/compare_actuals.py /tmp/act_base.npz /tmp/act_base2.npz
python tests/glue/compare_actuals.py /tmp/act_base.npz /tmp/act_cap.npz
echo PARITY done
