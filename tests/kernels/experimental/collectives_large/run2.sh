#!/bin/bash
D=tests/kernels/experimental/collectives_large
timeout 600 python $D/bench_ici.py 2>&1 | grep -E "^a2a|^pair|Error" | head -40
B="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=131072 --xla_all_reduce_latency_bound_threshold_in_bytes=524288 --xla_tpu_small_all_reduce_binomial_threshold_bytes=524288"
echo "=== shipped flags"
LIBTPU_INIT_ARGS="$B --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false" timeout 1200 python $D/bench_large.py 2>&1 | grep -E "^TP|Error|error|Traceback" | head -20
