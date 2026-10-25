#!/bin/bash
# The exchange bench under the c4 libtpu flags with SparseCore all-gather offload forced on, then off.
B="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_enable_async_all_gather=true --xla_all_reduce_latency_bound_threshold_in_bytes=524288 --xla_tpu_small_all_reduce_binomial_threshold_bytes=524288"
echo "== SC offload on"
LIBTPU_INIT_ARGS="$B --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0" python tests/kernels/experimental/small_collectives/bench_exchange.py 1 2 4
echo "== SC offload off"
LIBTPU_INIT_ARGS="$B --xla_tpu_enable_sparse_core_collective_offload_all_gather=false" python tests/kernels/experimental/small_collectives/bench_exchange.py 1 2 4
