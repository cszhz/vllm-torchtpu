#!/bin/bash
# Decode-side XLA collectives: c4 cell libtpu flags vs + the SC hierarchical AR flags.
D=tests/kernels/experimental/collectives_large
B="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --deepsea_slice_builder_heartbeat_interval=30m --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=131072 --xla_all_reduce_latency_bound_threshold_in_bytes=524288 --xla_tpu_small_all_reduce_binomial_threshold_bytes=524288"
S="--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true"
for rep in 1 2; do
echo "=== base"; LIBTPU_INIT_ARGS="$B" python $D/bench_decode_flags.py 2>&1 | grep -E "us$|Error"
echo "=== base+SCAR"; LIBTPU_INIT_ARGS="$B $S" python $D/bench_decode_flags.py 2>&1 | grep -E "us$|Error"
done
