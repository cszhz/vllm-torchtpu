#!/bin/bash
# combo3.sh: the SP piece checks under the combo flags (pins, skinny table v2), then the torch ops.
export TPU_SKIP_MDS_QUERY=true TPU_PIN_OPERANDS_HBM=skinny:combine:select TPU_FP8_SKINNY_TABLE=v2
bash tests/forge_prefill/run_py.sh tests/forge_prefill/check_seq_parallel_parts.py moe mhc rs
echo "=== torch ops (8 procs)"
unset LIBTPU_INIT_ARGS
python3 tests/forge_prefill/torch_sp_ops_multichip.py
