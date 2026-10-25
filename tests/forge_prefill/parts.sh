#!/bin/bash
# parts.sh: check_seq_parallel_parts.py under the TP8 cell's libtpu flags, then the
# reduce-scatter check again with the SparseCore reduce-scatter offload enabled.
export TPU_SKIP_MDS_QUERY=true
bash tests/forge_prefill/run_py.sh tests/forge_prefill/check_seq_parallel_parts.py
echo "=== SC reduce-scatter offload"
B=$(grep -o '"--xla[^"]*"' tests/forge_prefill/run_py.sh | tr -d '"')
LIBTPU_INIT_ARGS="$B --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true" python3 tests/forge_prefill/check_seq_parallel_parts.py rs
echo "=== torch ops (8 procs)"
unset LIBTPU_INIT_ARGS
python3 tests/forge_prefill/torch_sp_ops_multichip.py
