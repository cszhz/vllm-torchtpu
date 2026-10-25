#!/bin/bash
D=tests/kernels/experimental/collectives_large
B="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=131072 --xla_all_reduce_latency_bound_threshold_in_bytes=524288 --xla_tpu_small_all_reduce_binomial_threshold_bytes=524288"
S="--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true"
echo "=== base"; LIBTPU_INIT_ARGS="$B --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false" python $D/bench_overlap2.py 2>&1 | grep -E "^TP|^async|Error|rror:"
echo "=== SCAR"; LIBTPU_INIT_ARGS="$B $S" python $D/bench_overlap2.py 2>&1 | grep -E "^TP|^async|Error|rror:"
