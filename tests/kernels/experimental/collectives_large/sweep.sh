#!/bin/bash
# ICI calibration, then the XLA collective bench under libtpu flag variants.
D=tests/kernels/experimental/collectives_large
timeout 600 python $D/bench_ici.py 2>&1 | grep -E "^device|^pair|Error" 
B="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=131072 --xla_all_reduce_latency_bound_threshold_in_bytes=524288 --xla_tpu_small_all_reduce_binomial_threshold_bytes=524288"
for V in "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false" \
  "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true" \
  "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_decompose_all_reduce_bidirectional_communication=true" \
  "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_enable_all_reduce_native_decomposition=true" \
  "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true" \
  "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_enable_all_reduce_scatter_fusion=false" \
  "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_jf_use_multi_colors_in_all_reduce_if_supported=true"; do
  echo "=== VARIANT $V"
  LIBTPU_INIT_ARGS="$B $V" timeout 900 python $D/bench_xla_coll.py 2>&1 | grep -E "^TP|Error|error" | head -12
done
