# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""8-device 3-slot KV ring race, dynamic scratch fit, and float64 accuracy tests."""

import functools
import gc
import math
import os
import subprocess
import sys
import types
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives import util
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as brpa_wrapper
from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    kernel as pcp_kernel,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    wrapper as pcp_wrapper,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
)

pytestmark = pytest.mark.multichip

P = jax.sharding.PartitionSpec
PCP_AXIS = pcp_wrapper.PCP_AXIS_NAME
TP_AXIS = pcp_wrapper.TP_AXIS_NAME
WORLD_SIZE = 8
RACE_NUM_GROUPS = 4
RACE_PAGE_SIZE = 128
RACE_WIDTH = 128
RACE_SLOW_RANK = 3
RACE_SKEW_ITERS = 24_000

# IEEE-754 precision constants used to derive per-instance float64 oracle bounds:
# - bfloat16 has 7 fraction bits: 1 ULP on [1, 2) is EPS_BF16 = 2^-7, and
#   round-to-nearest-even half-ULP unit roundoff is U_BF16 = 2^-8.
# - float32 has 23 fraction bits: half-ULP unit roundoff is U_FP32 = 2^-24.
EPS_BF16 = 2.0**-7
U_BF16 = 2.0**-8
U_FP32 = 2.0**-24
NON_TRIVIAL_K_SCALE = 0.37
NON_TRIVIAL_V_SCALE = 0.43


class _ParallelismEnvelopeCase(NamedTuple):
    regime_label: str
    pcp_size: int
    tp_size: int
    total_q_heads: int
    total_kv_heads: int
    head_dim: int
    q_block_size: int
    page_size: int
    kv_dtype: jnp.dtype
    kv_layout: KVLayout
    max_model_len: int = 262_144
    max_num_seqs: int = 64
    global_bucket_tokens: int = 32_768


class _SupportedTpuConfigCase(NamedTuple):
    label: str
    mesh_shape: tuple[int, ...]
    total_q_heads: int
    total_kv_heads: int
    head_dim: int
    q_block_size: int
    page_size: int
    interleave_size: int
    kv_dtype: jnp.dtype
    kv_layout: KVLayout
    history_lens: tuple[int, ...] | None = None
    query_lens: tuple[int, ...] | None = None
    sigma: float = 0.25
    block_major: bool = False
    check_negative_control: bool = False


# Full single-host and multi-host serving envelope (including 32-core PCP16xTP2,
# PCP8xTP4, and PCP4xTP8 topologies at 256K context / 64 sequences / 32K bucket).
ALL_PARALLELISM_ENVELOPE_CASES: tuple[_ParallelismEnvelopeCase, ...] = (
    # TP=1 regimes: PCP8xTP1, PP4xPCP8xTP1, DP8xPCP8xTP1 (all model heads local)
    _ParallelismEnvelopeCase(
        "pcp8_tp1_32q_2kv_d256_p256_fp8_nhd",
        8,
        1,
        32,
        2,
        256,
        512,
        256,
        jnp.float8_e4m3fn,
        KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _ParallelismEnvelopeCase(
        "pcp8_tp1_32q_2kv_d256_p256_fp8_hnd",
        8,
        1,
        32,
        2,
        256,
        512,
        256,
        jnp.float8_e4m3fn,
        KVLayout.SEQ_ALONG_LANE,
    ),
    _ParallelismEnvelopeCase(
        "pcp8_tp1_32q_2kv_d256_p512_bf16_nhd",
        8,
        1,
        32,
        2,
        256,
        512,
        512,
        jnp.bfloat16,
        KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _ParallelismEnvelopeCase(
        "pcp8_tp1_max_kv_32q_8kv_d256_p512_bf16_nhd",
        8,
        1,
        32,
        8,
        256,
        512,
        512,
        jnp.bfloat16,
        KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _ParallelismEnvelopeCase(
        "pcp8_tp1_64q_4kv_d128_p256_fp8_nhd",
        8,
        1,
        64,
        4,
        128,
        512,
        256,
        jnp.float8_e4m3fn,
        KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _ParallelismEnvelopeCase(
        "pcp8_tp1_64q_8kv_d128_p256_fp8_hnd",
        8,
        1,
        64,
        8,
        128,
        512,
        256,
        jnp.float8_e4m3fn,
        KVLayout.SEQ_ALONG_LANE,
    ),
    _ParallelismEnvelopeCase(
        "pcp8_tp1_max_q_64q_8kv_d128_p256_bf16_nhd",
        8,
        1,
        64,
        8,
        128,
        512,
        256,
        jnp.bfloat16,
        KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _ParallelismEnvelopeCase(
        "pp4_pcp8_tp1_64q_4kv_d256_bq256_p128_fp8_hnd",
        8,
        1,
        64,
        4,
        256,
        256,
        128,
        jnp.float8_e4m3fn,
        KVLayout.SEQ_ALONG_LANE,
    ),
    # TP=2 regimes: PCP4xTP2, DP2xPCP4xTP2, and multi-host PCP16xTP2
    _ParallelismEnvelopeCase(
        "pcp4_tp2_32q_2kv_d256_p256_fp8_nhd",
        4,
        2,
        32,
        2,
        256,
        512,
        256,
        jnp.float8_e4m3fn,
        KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _ParallelismEnvelopeCase(
        "pcp4_tp2_24q_6kv_odd_local_kv_d128_p256_fp8_nhd",
        4,
        2,
        24,
        6,
        128,
        512,
        256,
        jnp.float8_e4m3fn,
        KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _ParallelismEnvelopeCase(
        "pcp4_tp2_64q_4kv_d128_p256_fp8_nhd",
        4,
        2,
        64,
        4,
        128,
        512,
        256,
        jnp.float8_e4m3fn,
        KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _ParallelismEnvelopeCase(
        "pcp16_tp2_32q_2kv_d256_p256_fp8_nhd",
        16,
        2,
        32,
        2,
        256,
        512,
        256,
        jnp.float8_e4m3fn,
        KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _ParallelismEnvelopeCase(
        "pcp16_tp2_64q_4kv_d256_p256_fp8_hnd",
        16,
        2,
        64,
        4,
        256,
        512,
        256,
        jnp.float8_e4m3fn,
        KVLayout.SEQ_ALONG_LANE,
    ),
    # TP=4 and TP=8 regimes: PCP2xTP4, multi-host PCP8xTP4, and multi-host PCP4xTP8
    _ParallelismEnvelopeCase(
        "pcp2_tp4_128q_8kv_d128_p256_bf16_nhd",
        2,
        4,
        128,
        8,
        128,
        512,
        256,
        jnp.bfloat16,
        KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _ParallelismEnvelopeCase(
        "pcp8_tp4_128q_8kv_d128_p256_bf16_nhd",
        8,
        4,
        128,
        8,
        128,
        512,
        256,
        jnp.bfloat16,
        KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _ParallelismEnvelopeCase(
        "pcp4_tp8_64q_8kv_d128_p256_bf16_nhd",
        4,
        8,
        64,
        8,
        128,
        512,
        256,
        jnp.bfloat16,
        KVLayout.HEAD_ALONG_SUBLANE,
    ),
)


# Combined 8-device TPU compile + scratch capacity fit + float64 oracle matrix across
# 1D ("pcp",) and 2D ("pcp", "tp") meshes: (8,), (8, 1), (4, 2), and (2, 4).
SUPPORTED_TPU_CONFIG_CASES: tuple[_SupportedTpuConfigCase, ...] = (
    # 1D ("pcp",) PCP=8 ragged multi-request cases
    # (BF16, FP8 large-logit, HND block-major)
    _SupportedTpuConfigCase(
        label="pcp8_1d_4q_2kv_d128_p128_bf16_nhd_ragged",
        mesh_shape=(8,),
        total_q_heads=4,
        total_kv_heads=2,
        head_dim=128,
        q_block_size=128,
        page_size=128,
        interleave_size=128,
        kv_dtype=jnp.bfloat16,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
        history_lens=(4096, 2048, 0),
        query_lens=(1152, 700, 1024),
    ),
    _SupportedTpuConfigCase(
        label="pcp8_1d_4q_2kv_d128_p128_fp8_nhd_large_logits",
        mesh_shape=(8,),
        total_q_heads=4,
        total_kv_heads=2,
        head_dim=128,
        q_block_size=128,
        page_size=128,
        interleave_size=128,
        kv_dtype=jnp.float8_e4m3fn,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
        history_lens=(4096, 2048, 0),
        query_lens=(1152, 700, 1024),
        sigma=2.0,
        check_negative_control=True,
    ),
    _SupportedTpuConfigCase(
        label="pcp8_1d_4q_2kv_d128_p256_fp8_hnd_interleave128_block_major",
        mesh_shape=(8,),
        total_q_heads=4,
        total_kv_heads=2,
        head_dim=128,
        q_block_size=256,
        page_size=256,
        interleave_size=128,
        kv_dtype=jnp.float8_e4m3fn,
        kv_layout=KVLayout.SEQ_ALONG_LANE,
        history_lens=(4096, 2048, 0),
        query_lens=(1152, 700, 1024),
        block_major=True,
    ),
    # 2D (8, 1) ("pcp", "tp"): PCP8xTP1, PP4xPCP8xTP1, DP8xPCP8xTP1
    # up to 59.01 MiB scratch
    _SupportedTpuConfigCase(
        label="pcp8_tp1_32q_2kv_d256_p256_fp8_nhd",
        mesh_shape=(8, 1),
        total_q_heads=32,
        total_kv_heads=2,
        head_dim=256,
        q_block_size=512,
        page_size=256,
        interleave_size=256,
        kv_dtype=jnp.float8_e4m3fn,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _SupportedTpuConfigCase(
        label="pcp8_tp1_max_kv_32q_8kv_d256_p512_bf16_nhd",
        mesh_shape=(8, 1),
        total_q_heads=32,
        total_kv_heads=8,
        head_dim=256,
        q_block_size=512,
        page_size=512,
        interleave_size=512,
        kv_dtype=jnp.bfloat16,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _SupportedTpuConfigCase(
        label="pcp8_tp1_max_q_64q_8kv_d128_p256_bf16_nhd",
        mesh_shape=(8, 1),
        total_q_heads=64,
        total_kv_heads=8,
        head_dim=128,
        q_block_size=512,
        page_size=256,
        interleave_size=256,
        kv_dtype=jnp.bfloat16,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _SupportedTpuConfigCase(
        label="pp4_pcp8_tp1_64q_4kv_d256_bq256_p128_fp8_hnd",
        mesh_shape=(8, 1),
        total_q_heads=64,
        total_kv_heads=4,
        head_dim=256,
        q_block_size=256,
        page_size=128,
        interleave_size=128,
        kv_dtype=jnp.float8_e4m3fn,
        kv_layout=KVLayout.SEQ_ALONG_LANE,
    ),
    # 2D (4, 2) ("pcp", "tp"): PCP4xTP2, odd local KV heads,
    # and PCP16xTP2 per-shard shape
    _SupportedTpuConfigCase(
        label="pcp4_tp2_24q_6kv_odd_local_kv_d128_p256_fp8_nhd",
        mesh_shape=(4, 2),
        total_q_heads=24,
        total_kv_heads=6,
        head_dim=128,
        q_block_size=512,
        page_size=256,
        interleave_size=256,
        kv_dtype=jnp.float8_e4m3fn,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _SupportedTpuConfigCase(
        label="pcp4_tp2_64q_4kv_d256_p256_fp8_hnd",
        mesh_shape=(4, 2),
        total_q_heads=64,
        total_kv_heads=4,
        head_dim=256,
        q_block_size=512,
        page_size=256,
        interleave_size=256,
        kv_dtype=jnp.float8_e4m3fn,
        kv_layout=KVLayout.SEQ_ALONG_LANE,
    ),
    # 2D (2, 4) ("pcp", "tp"): PCP2xTP4 and PCP8xTP4 per-shard shape across 4 TP shards
    _SupportedTpuConfigCase(
        label="pcp2_tp4_128q_8kv_d128_p256_bf16_nhd",
        mesh_shape=(2, 4),
        total_q_heads=128,
        total_kv_heads=8,
        head_dim=128,
        q_block_size=512,
        page_size=256,
        interleave_size=256,
        kv_dtype=jnp.bfloat16,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    ),
)


@pytest.fixture
def release_jax_backend():
    yield
    jax.clear_caches()
    gc.collect()


def _get_dynamic_tpu_info():
    """Query live TPU ChipInfo on TPU hardware or TPU v7x ChipInfo on CPU."""
    devices = jax.local_devices()
    if devices and devices[0].platform == "tpu":
        return pltpu.get_tpu_info()
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _require_tpu_devices(min_count: int = WORLD_SIZE):
    devices = jax.local_devices()
    if len(devices) < min_count or devices[0].platform != "tpu":
        pytest.skip(f"3-slot PCP ring test requires {min_count} TPU devices.")
    return devices[:min_count]


def _sum_vmem_scratch_bytes(scratch_shapes: tuple[object, ...]) -> int:
    total = 0
    for spec in scratch_shapes:
        if getattr(spec, "memory_space", None) == pltpu.MemorySpace.VMEM:
            shape = getattr(spec, "shape", ())
            dtype = getattr(spec, "dtype", None)
            assert dtype is not None
            total += int(math.prod(shape)) * int(jnp.dtype(dtype).itemsize)
    return total


def _assert_recorded_3slot_vmem_calls(
    recorded_calls: list[tuple[str, tuple[object, ...], int]],
    *,
    vmem_capacity_bytes: int,
    label: str,
    expected_min_calls: int,
) -> None:
    streaming_calls = [
        (name, scratch, limit)
        for name, scratch, limit in recorded_calls
        if "page_groups_multi_head" in name
    ]
    assert len(streaming_calls) >= expected_min_calls, (
        f"{label}: expected at least {expected_min_calls} streaming pallas calls, "
        f"recorded {[name for name, _, _ in recorded_calls]}"
    )
    for name, scratch_shapes, vmem_limit in streaming_calls:
        assert vmem_limit == vmem_capacity_bytes, (
            f"{label} ({name}): compiler_params.vmem_limit_bytes "
            f"{vmem_limit} != tpu_info.vmem_capacity_bytes {vmem_capacity_bytes}"
        )
        kv_vmem_specs = [
            spec
            for spec in scratch_shapes
            if hasattr(spec, "shape") and len(spec.shape) == 6
        ]
        assert len(kv_vmem_specs) == 1, name
        assert kv_vmem_specs[0].shape[0] == 3, (
            f"{label} ({name}) expected 3 KV ring slots, got {kv_vmem_specs[0].shape}"
        )
        vmem_scratch_bytes = _sum_vmem_scratch_bytes(scratch_shapes)
        assert 0 < vmem_scratch_bytes <= vmem_capacity_bytes, (
            f"{label} ({name}) 3-slot VMEM scratch {vmem_scratch_bytes} B "
            f"exceeds tpu_info.vmem_capacity_bytes {vmem_capacity_bytes} B"
        )


def _skewed_3slot_ring_kernel(
    local_pages_ref,
    o_ref,
    delay_sink_ref,
    local_dma_sem,
    remote_send_sems,
    remote_recv_sems,
    remote_sync_sems,
    group_slot_free_sems,
    kv_vmem_ref,
    delay_vmem_ref,
    *,
    pcp_size: int,
    num_groups: int,
    skew_iters: int,
    delay_round: int,
    intra_sync: bool,
    group_sync: bool,
):
    """Exercise 3-slot intra-group and cross-group semaphores under rank skew.

    - Intra-group hazard (`delay_round=0, intra_sync=False, num_groups=1`):
      fast ranks 0 -> 1 -> 2 reach round 2 while rank 3 is still reading slot 0
      in round 0, overwriting rank 3's slot 0 (`(2 + 1) % 3 == 0`).
    - Cross-group hazard (`delay_round=7, group_sync=False, num_groups=4`):
      on PCP=8 the final round (`round_idx=7`) reads slot `7 % 3 == 1`, while
      the next group's round 0 remote copy writes into `next_slot == 1`.
    """
    my_id = lax.axis_index(PCP_AXIS)
    next_rank = lax.rem(my_id + 1, pcp_size)
    prev_rank = lax.rem(my_id + pcp_size - 1, pcp_size)

    delay_vmem_ref[...] = jnp.ones_like(delay_vmem_ref[...])
    util.local_barrier(prev_rank, next_rank)

    def _group_loop(group_idx, group_carry):
        del group_carry
        load_op = pltpu.make_async_copy(
            src_ref=local_pages_ref.at[group_idx],
            dst_ref=kv_vmem_ref.at[0],
            sem=local_dma_sem,
        )
        load_op.start()
        load_op.wait()

        if group_sync:

            @pl.when(group_idx > 0)
            def _wait_prev_group_slot_free():
                pl.semaphore_wait(group_slot_free_sems.at[0], 1)

        def _round_loop(round_idx, round_carry):
            del round_carry
            curr_slot = lax.rem(round_idx, 3)
            next_slot = lax.rem(round_idx + 1, 3)
            safe_round_idx = jnp.minimum(round_idx, pcp_size - 2)
            safe_slot_free_round_idx = jnp.clip(round_idx - 2, 0, pcp_size - 2)

            if intra_sync:

                @pl.when(jnp.logical_and(round_idx >= 2, round_idx < pcp_size - 1))
                def _wait_remote_sync():
                    pl.semaphore_wait(
                        remote_sync_sems.at[0, safe_slot_free_round_idx], 1
                    )

            remote_op = pltpu.make_async_remote_copy(
                src_ref=kv_vmem_ref.at[curr_slot],
                dst_ref=kv_vmem_ref.at[next_slot],
                send_sem=remote_send_sems.at[0, safe_round_idx],
                recv_sem=remote_recv_sems.at[0, safe_round_idx],
                device_id=(next_rank,),
                device_id_type=pl.DeviceIdType.MESH,
            )

            @pl.when(round_idx < pcp_size - 1)
            def _start_remote_copy():
                remote_op.start()

            @pl.when(
                jnp.logical_and(
                    my_id == RACE_SLOW_RANK,
                    jnp.logical_or(delay_round < 0, round_idx == delay_round),
                )
            )
            def _delay_slow_rank():
                init_acc = delay_vmem_ref[...]

                def _spin_body(_i, acc):
                    return acc * jnp.float32(1.0000001) + jnp.float32(1e-6)

                delay_vmem_ref[...] = lax.fori_loop(0, skew_iters, _spin_body, init_acc)

            store_op = pltpu.make_async_copy(
                src_ref=kv_vmem_ref.at[curr_slot],
                dst_ref=o_ref.at[group_idx, round_idx],
                sem=local_dma_sem,
            )
            store_op.start()
            store_op.wait()

            @pl.when(round_idx < pcp_size - 1)
            def _finish_remote_copy():
                remote_op.wait_send()
                if intra_sync:

                    @pl.when(round_idx < pcp_size - 3)
                    def _signal_prev_device():
                        pl.semaphore_signal(
                            remote_sync_sems.at[0, safe_round_idx],
                            1,
                            device_id=(prev_rank,),
                            device_id_type=pl.DeviceIdType.MESH,
                        )

                remote_op.wait_recv()

            return jnp.int32(0)

        lax.fori_loop(0, pcp_size, _round_loop, jnp.int32(0), unroll=False)

        if group_sync:

            @pl.when(group_idx < num_groups - 1)
            def _signal_next_group_slot_free():
                pl.semaphore_signal(
                    group_slot_free_sems.at[0],
                    1,
                    device_id=(prev_rank,),
                    device_id_type=pl.DeviceIdType.MESH,
                )

        return jnp.int32(0)

    lax.fori_loop(0, num_groups, _group_loop, jnp.int32(0), unroll=False)
    sink_store = pltpu.make_async_copy(
        src_ref=delay_vmem_ref,
        dst_ref=delay_sink_ref,
        sem=local_dma_sem,
    )
    sink_store.start()
    sink_store.wait()


def _run_skewed_3slot_ring(
    devices,
    local_pages: jax.Array,
    *,
    num_groups: int,
    delay_round: int,
    intra_sync: bool,
    group_sync: bool,
    collective_id: int,
):
    pcp_size = len(devices)
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    mesh = jax.sharding.Mesh(np.asarray(devices), (PCP_AXIS,))

    def _call(pages):
        out, _sink = pl.pallas_call(
            functools.partial(
                _skewed_3slot_ring_kernel,
                pcp_size=pcp_size,
                num_groups=num_groups,
                skew_iters=RACE_SKEW_ITERS,
                delay_round=delay_round,
                intra_sync=intra_sync,
                group_sync=group_sync,
            ),
            out_shape=(
                jax.ShapeDtypeStruct(
                    (num_groups, pcp_size, RACE_PAGE_SIZE, RACE_WIDTH),
                    pages.dtype,
                ),
                jax.ShapeDtypeStruct((8, 128), jnp.float32),
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
                out_specs=(
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                ),
                scratch_shapes=(
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA((1, pcp_size - 1)),
                    pltpu.SemaphoreType.DMA((1, pcp_size - 1)),
                    pltpu.SemaphoreType.REGULAR((1, pcp_size - 1)),
                    pltpu.SemaphoreType.REGULAR((1,)),
                    pltpu.VMEM((3, RACE_PAGE_SIZE, RACE_WIDTH), pages.dtype),
                    pltpu.VMEM((8, 128), jnp.float32),
                ),
                grid=(1,),
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            name=(
                f"pcp8_3slot_race_g{num_groups}_d{delay_round}"
                f"_i{int(intra_sync)}_gs{int(group_sync)}"
            ),
        )(pages)
        return out

    fn = jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=P(PCP_AXIS, None, None, None),
            out_specs=P(PCP_AXIS, None, None, None, None),
            check_vma=False,
        )
    )
    out = fn(local_pages)
    out.block_until_ready()
    return np.asarray(jax.device_get(out))


# ---------------------------------------------------------------------------
# Pillar 1: 8-Device 3-Slot Ring Race Check (Intra-Group + Cross-Group)
# ---------------------------------------------------------------------------
def test_3slot_ring_rank_skewed_race_with_negative_control(release_jax_backend):
    """Prove both intra-group and cross-group 3-slot barriers prevent races."""
    devices = _require_tpu_devices(WORLD_SIZE)
    ranks = np.arange(WORLD_SIZE, dtype=np.int32)[:, None, None, None]
    groups = np.arange(RACE_NUM_GROUPS, dtype=np.int32)[None, :, None, None]
    offsets = np.arange(RACE_PAGE_SIZE * RACE_WIDTH, dtype=np.int32).reshape(
        1, 1, RACE_PAGE_SIZE, RACE_WIDTH
    )
    local_np = ranks * 10_000_000 + groups * 1_000_000 + offsets
    local_pages = jnp.asarray(local_np)

    expected_np = np.empty(
        (WORLD_SIZE, RACE_NUM_GROUPS, WORLD_SIZE, RACE_PAGE_SIZE, RACE_WIDTH),
        dtype=np.int32,
    )
    for rank in range(WORLD_SIZE):
        for group_idx in range(RACE_NUM_GROUPS):
            for round_idx in range(WORLD_SIZE):
                src_rank = (rank - round_idx) % WORLD_SIZE
                expected_np[rank, group_idx, round_idx] = local_np[src_rank, group_idx]

    no_intra_sync_np = _run_skewed_3slot_ring(
        devices,
        local_pages[:, :1],
        num_groups=1,
        delay_round=0,
        intra_sync=False,
        group_sync=False,
        collective_id=29,
    )
    assert not np.array_equal(no_intra_sync_np, expected_np[:, :1]), (
        "Intra-group negative control failed: omitting round_idx-2 "
        "remote_sync_sems must expose slot-0 overwrite under rank skew."
    )

    no_group_sync_np = _run_skewed_3slot_ring(
        devices,
        local_pages,
        num_groups=RACE_NUM_GROUPS,
        delay_round=WORLD_SIZE - 1,
        intra_sync=True,
        group_sync=False,
        collective_id=30,
    )
    assert not np.array_equal(no_group_sync_np, expected_np), (
        "Cross-group negative control failed: omitting group_slot_free_sems "
        "must expose PCP=8 round-7 slot-1 overwrite across page groups."
    )

    synced_np = _run_skewed_3slot_ring(
        devices,
        local_pages,
        num_groups=RACE_NUM_GROUPS,
        delay_round=-1,
        intra_sync=True,
        group_sync=True,
        collective_id=31,
    )
    np.testing.assert_array_equal(synced_np, expected_np)


# ---------------------------------------------------------------------------
# Pillars 2 + 3 (Combined & Parameterized): Dynamic tpu_info Scratch Capacity Fit
# + First-Principles Host float64 Oracle Accuracy Across Supported Configs
# ---------------------------------------------------------------------------
def _is_power_of_two(val: float) -> bool:
    if val <= 0.0:
        return False
    mantissa, _exp = math.frexp(val)
    return mantissa == 0.5


def _quantize_stored_kv_for_fp64_oracle(
    raw_arr: np.ndarray,
    *,
    cache_dtype: jnp.dtype,
    scale: float | None,
) -> np.ndarray:
    """Reproduce exact stored KV values in float64 without extra bfloat16 rounding.

    Caller inputs to `sharded_pcp_ragged_paged_attention` arrive as
    bfloat16 (`raw_arr / scale` when FP8 KV quantization is active), and
    `_prepare_packed_kv_for_cache` casts those bfloat16 values to
    `cache_dtype`. Dequantizing directly to `np.float64` audits the
    kernel's internal `(kv.astype(f32) * scale).astype(bf16)` cast as well
    as online-softmax accumulation and output rounding.
    """
    prescaled = raw_arr if scale is None else (raw_arr / np.float32(scale))
    bf16_in = jnp.asarray(prescaled, dtype=jnp.bfloat16)
    if cache_dtype == jnp.float8_e4m3fn:
        assert scale is not None
        stored_fp8 = bf16_in.astype(jnp.float8_e4m3fn)
        return np.asarray(
            jax.device_get(stored_fp8.astype(jnp.float32)), dtype=np.float64
        ) * float(scale)
    return np.asarray(jax.device_get(bf16_in.astype(jnp.float32)), dtype=np.float64)


def _fp64_causal_gqa_reference(
    q_chunk: np.ndarray,
    k_full: np.ndarray,
    v_full: np.ndarray,
    *,
    q_start_pos: int,
    sm_scale: float,
    kv_dtype: jnp.dtype,
    k_scale: float | None,
    v_scale: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    q64 = _quantize_stored_kv_for_fp64_oracle(
        q_chunk, cache_dtype=jnp.bfloat16, scale=None
    )
    k64 = _quantize_stored_kv_for_fp64_oracle(
        k_full, cache_dtype=kv_dtype, scale=k_scale
    )
    v64 = _quantize_stored_kv_for_fp64_oracle(
        v_full, cache_dtype=kv_dtype, scale=v_scale
    )
    q_per_kv = q64.shape[1] // k64.shape[1]
    k_rep = np.repeat(k64, q_per_kv, axis=1)
    v_rep = np.repeat(v64, q_per_kv, axis=1)

    scores = np.einsum("thd,shd->ths", q64, k_rep) * float(sm_scale)
    q_pos = q_start_pos + np.arange(q64.shape[0])
    k_pos = np.arange(k64.shape[0])
    causal_mask = q_pos[:, None] >= k_pos[None, :]
    scores = np.where(causal_mask[:, None, :], scores, -np.inf)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.einsum("ths,shd->thd", probs, v_rep), v64


def _derive_fp64_tolerances(
    expected_fp64: np.ndarray,
    v_fp64: np.ndarray,
    *,
    num_kv_tiles: int,
    k_scale: float | None,
    v_scale: float | None,
) -> tuple[float, float]:
    """Derive per-instance (tol_rmse, tol_max_abs) from IEEE constants and norms.

    Precision stages in `pcp_streaming_rpa/kernel.py`:
      1. `QK^T`, `exp`, and online rescaling `alpha * acc + pv` run in float32
         (`U_FP32 = 2^-24` per KV tile).
      2. Softmax weights cast `p.astype(bfloat16)` (`U_BF16 = 2^-8`).
      3. Normalized output casts `(acc / l).astype(bfloat16)` (`U_BF16 = 2^-8`).
      4. When FP8 `k_scale`/`v_scale` are not exact powers of 2, dequantizing
         `(kv.astype(f32) * scale).astype(bfloat16)` adds 1 bfloat16 rounding on V
         and 1 first-order logit rounding term on K.
    """
    non_pow2_fp8 = (k_scale is not None and not _is_power_of_two(k_scale)) or (
        v_scale is not None and not _is_power_of_two(v_scale)
    )
    ref_rms = float(np.sqrt(np.mean(expected_fp64**2)))
    v_rms = float(np.sqrt(np.mean(v_fp64**2)))
    ref_abs_max = float(np.max(np.abs(expected_fp64)))
    v_abs_max = float(np.max(np.abs(v_fp64)))

    n_value_bf16_casts = 3 if non_pow2_fp8 else 2
    k_logit_factor = 1.0 if non_pow2_fp8 else 0.0
    bf16_rmse_factor = math.sqrt(n_value_bf16_casts) + k_logit_factor

    tol_rmse = (
        bf16_rmse_factor * U_BF16 * ref_rms + max(1, num_kv_tiles) * U_FP32 * v_rms
    )
    pointwise_bf16_ulps = 2.0 + (1.0 if non_pow2_fp8 else 0.0)
    tol_max_abs = (
        pointwise_bf16_ulps * EPS_BF16 * ref_abs_max
        + EPS_BF16 * v_abs_max
        + max(1, num_kv_tiles) * (2.0 * U_FP32) * v_abs_max
    )
    return tol_rmse, tol_max_abs


def _assert_fp64_parity(
    observed: np.ndarray,
    expected_fp64: np.ndarray,
    v_fp64: np.ndarray,
    *,
    num_kv_tiles: int,
    k_scale: float | None,
    v_scale: float | None,
    label: str,
) -> tuple[float, float]:
    obs64 = np.asarray(observed, dtype=np.float64)
    exp64 = np.asarray(expected_fp64, dtype=np.float64)
    assert np.all(np.isfinite(obs64)), f"{label}: non-finite output detected"
    diff = obs64 - exp64
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    tol_rmse, tol_max_abs = _derive_fp64_tolerances(
        exp64,
        v_fp64,
        num_kv_tiles=num_kv_tiles,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    assert rmse <= tol_rmse, (
        f"{label}: RMSE {rmse:.6e} exceeded derived float64 bound {tol_rmse:.6e}"
    )
    assert max_abs <= tol_max_abs, (
        f"{label}: max_abs {max_abs:.6e} exceeded derived float64 bound "
        f"{tol_max_abs:.6e}"
    )
    return rmse, tol_rmse


def _pack_step_tensors(
    q_reqs: list[np.ndarray],
    k_reqs: list[np.ndarray],
    v_reqs: list[np.ndarray],
    *,
    pcp_size: int,
    interleave_size: int,
    global_padded_tokens: int,
    k_scale: float | None,
    v_scale: float | None,
):
    q_lens = [int(q.shape[0]) for q in q_reqs]
    owner_starts = np.zeros(len(q_lens), dtype=np.int64)
    if len(q_lens) > 1:
        owner_starts[1:] = np.cumsum(q_lens[:-1])
    token_order, inverse_order = build_pcp_rank_major_token_order(
        q_lens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        padded_num_tokens=global_padded_tokens,
        token_owner_start_offsets_per_req=owner_starts,
    )
    valid = token_order >= 0

    def _pack(reqs: list[np.ndarray], scale: float | None):
        flat = np.concatenate(reqs, axis=0)
        if scale is not None:
            flat = flat / np.float32(scale)
        out = np.zeros((global_padded_tokens, *flat.shape[1:]), dtype=np.float32)
        out[valid] = flat[token_order[valid]]
        return jnp.asarray(out, dtype=jnp.bfloat16)

    return (
        _pack(q_reqs, None),
        _pack(k_reqs, k_scale),
        _pack(v_reqs, v_scale),
        q_lens,
        inverse_order,
    )


def _build_pcp_mesh(
    devices, mesh_shape: tuple[int, ...]
) -> tuple[jax.sharding.Mesh, int, int]:
    if len(mesh_shape) == 1:
        (pcp_size,) = mesh_shape
        tp_size = 1
        mesh = jax.sharding.Mesh(np.asarray(devices[:pcp_size]), (PCP_AXIS,))
        return mesh, pcp_size, tp_size
    if len(mesh_shape) == 2:
        pcp_size, tp_size = mesh_shape
        mesh_devices = np.asarray(devices[: pcp_size * tp_size]).reshape(
            pcp_size, tp_size
        )
        mesh = jax.sharding.Mesh(mesh_devices, (PCP_AXIS, TP_AXIS))
        return mesh, pcp_size, tp_size
    raise ValueError(f"Unsupported mesh_shape: {mesh_shape}")


def _invoke_streaming_attention(
    *,
    mesh: jax.sharding.Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    sm_scale: float,
    k_scale: float | None,
    v_scale: float | None,
    interleave_size: int,
    q_block_size: int,
    kv_layout: KVLayout,
    block_major: bool,
) -> tuple[jax.Array, jax.Array]:
    if block_major:
        devs = jax.local_devices()
        if not devs or devs[0].platform != "tpu":
            for mod_name in (
                "torch",
                "torch_tpu",
                "torch_tpu._internal",
                "torch_tpu._internal.pallas",
                "vllm_torchtpu.envs",
                "vllm_torchtpu.distributed",
                "vllm_torchtpu.distributed.pcp",
            ):
                sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
            _pallas_mod = types.ModuleType("pallas")
            _pallas_mod.JaxCallable = object
            sys.modules["torch_tpu._internal.pallas"].pallas = _pallas_mod
            sys.modules["vllm_torchtpu.distributed.pcp"].get_or_create_pcp_mesh = (
                lambda *a, **k: None
            )
        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
            vllm_adapter as pcp_vllm_adapter,
        )

        kernel_fn = pcp_vllm_adapter.make_pcp_streaming_rpa_kernel(
            q_scale=None,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh=mesh,
            sliding_window=None,
            sm_scale=sm_scale,
            soft_cap=None,
            skip_kv_update=False,
            cp_kv_cache_interleave_size=interleave_size,
            q_block_size=q_block_size,
            q_compute_size=q_block_size,
            kv_layout=kv_layout,
            block_major=True,
        )
        new_cache, out = kernel_fn(
            kv_cache,
            q,
            k,
            v,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
        return out, new_cache
    return pcp_wrapper.sharded_pcp_ragged_paged_attention(
        mesh=mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sink=None,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_kv_cache_interleave_size=interleave_size,
        q_block_size=q_block_size,
        q_compute_size=q_block_size,
        kv_layout=kv_layout,
    )


def _run_supported_tpu_config_case(
    devices,
    case: _SupportedTpuConfigCase,
    *,
    monkeypatch,
    seed: int,
) -> None:
    vmem_capacity_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    assert vmem_capacity_bytes > 0

    recorded_calls: list[tuple[str, tuple[object, ...], int]] = []
    real_pallas_call = pcp_kernel.pl.pallas_call

    def _spy_pallas_call(
        kernel_fn, *, out_shape, grid_spec, compiler_params, name=None, **kw
    ):
        recorded_calls.append(
            (
                str(name or ""),
                tuple(grid_spec.scratch_shapes),
                int(compiler_params.vmem_limit_bytes),
            )
        )
        return real_pallas_call(
            kernel_fn,
            out_shape=out_shape,
            grid_spec=grid_spec,
            compiler_params=compiler_params,
            name=name,
            **kw,
        )

    with monkeypatch.context() as mp:
        mp.setattr(pcp_kernel.pl, "pallas_call", _spy_pallas_call)
        _run_supported_tpu_config_case_inner(
            devices,
            case,
            recorded_calls=recorded_calls,
            vmem_capacity_bytes=vmem_capacity_bytes,
            seed=seed,
        )


def _run_supported_tpu_config_case_inner(
    devices,
    case: _SupportedTpuConfigCase,
    *,
    recorded_calls: list[tuple[str, tuple[object, ...], int]],
    vmem_capacity_bytes: int,
    seed: int,
) -> None:
    mesh, pcp_size, tp_size = _build_pcp_mesh(devices, case.mesh_shape)
    assert case.total_q_heads % tp_size == 0
    assert case.total_kv_heads % tp_size == 0
    local_kv_heads = case.total_kv_heads // tp_size

    hist_groups = 2 if 2 * case.page_size <= case.q_block_size else 1
    history_lens = (
        list(case.history_lens)
        if case.history_lens is not None
        else [hist_groups * pcp_size * case.page_size]
    )
    query_lens = (
        list(case.query_lens)
        if case.query_lens is not None
        else [pcp_size * case.q_block_size]
    )

    rng = np.random.default_rng(seed)
    sm_scale = 1.0 / math.sqrt(case.head_dim)
    k_scale = NON_TRIVIAL_K_SCALE if case.kv_dtype == jnp.float8_e4m3fn else None
    v_scale = NON_TRIVIAL_V_SCALE if case.kv_dtype == jnp.float8_e4m3fn else None

    num_reqs = len(query_lens)
    max_num_seqs = max(4, num_reqs)
    virtual_page = pcp_size * case.page_size
    pages_per_req = [
        max(1, math.ceil((h_len + q_len) / virtual_page))
        for h_len, q_len in zip(history_lens, query_lens)
    ]
    max_pages_per_req = max(pages_per_req)
    local_num_blocks = max(4, sum(pages_per_req) + 1)
    if case.block_major and local_num_blocks % 2 != 0:
        local_num_blocks += 1

    block_tables = np.zeros((max_num_seqs, max_pages_per_req), dtype=np.int32)
    next_page = 0
    for req_id, count in enumerate(pages_per_req):
        block_tables[req_id, :count] = np.arange(
            next_page, next_page + count, dtype=np.int32
        )
        next_page += count
    page_indices = jnp.asarray(block_tables.reshape(-1), dtype=jnp.int32)

    kv_packing = pcp_wrapper.get_dtype_packing(case.kv_dtype)
    if case.kv_layout == KVLayout.SEQ_ALONG_LANE:
        flat_cache_shape = (
            pcp_size * local_num_blocks,
            2 * local_kv_heads,
            tp_size * (case.head_dim // kv_packing),
            kv_packing,
            case.page_size,
        )
    else:
        local_packed_groups = math.ceil((2 * local_kv_heads) / kv_packing)
        flat_cache_shape = (
            pcp_size * local_num_blocks,
            case.page_size,
            tp_size * local_packed_groups,
            kv_packing,
            case.head_dim,
        )
    kv_cache_shape = (
        (flat_cache_shape[0] // 2, 2, *flat_cache_shape[1:])
        if case.block_major
        else flat_cache_shape
    )
    kv_cache = jnp.zeros(kv_cache_shape, dtype=case.kv_dtype)

    full_q = [
        (
            rng.standard_normal((h_len + q_len, case.total_q_heads, case.head_dim))
            * case.sigma
        ).astype(np.float32)
        for h_len, q_len in zip(history_lens, query_lens)
    ]
    full_k = [
        (
            rng.standard_normal((h_len + q_len, case.total_kv_heads, case.head_dim))
            * case.sigma
        ).astype(np.float32)
        for h_len, q_len in zip(history_lens, query_lens)
    ]
    full_v = [
        (
            rng.standard_normal((h_len + q_len, case.total_kv_heads, case.head_dim))
            * case.sigma
        ).astype(np.float32)
        for h_len, q_len in zip(history_lens, query_lens)
    ]

    # Step 0: populate multi-page cached history over the 3-slot ring.
    hist_req_ids = [idx for idx, h_len in enumerate(history_lens) if h_len > 0]
    if hist_req_ids:
        assert hist_req_ids == list(range(len(hist_req_ids)))
        hist_only = [history_lens[idx] for idx in hist_req_ids]
        local_hist_padded = max(
            case.q_block_size,
            case.page_size,
            math.ceil(sum(hist_only) / (pcp_size * case.q_block_size))
            * case.q_block_size,
        )
        global_hist_padded = pcp_size * local_hist_padded
        q_s0, k_s0, v_s0, s0_lens, s0_inv = _pack_step_tensors(
            [full_q[idx][: history_lens[idx]] for idx in hist_req_ids],
            [full_k[idx][: history_lens[idx]] for idx in hist_req_ids],
            [full_v[idx][: history_lens[idx]] for idx in hist_req_ids],
            pcp_size=pcp_size,
            interleave_size=case.interleave_size,
            global_padded_tokens=global_hist_padded,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        kv_lens_s0 = np.zeros(max_num_seqs, dtype=np.int32)
        kv_lens_s0[: len(s0_lens)] = s0_lens
        cu_q_s0 = np.zeros(max_num_seqs + 1, dtype=np.int32)
        cu_q_s0[1 : len(s0_lens) + 1] = np.cumsum(s0_lens)
        cu_q_s0[len(s0_lens) + 1 :] = cu_q_s0[len(s0_lens)]

        out_s0, kv_cache = _invoke_streaming_attention(
            mesh=mesh,
            q=q_s0,
            k=k_s0,
            v=v_s0,
            kv_cache=kv_cache,
            kv_lens=jnp.asarray(kv_lens_s0),
            page_indices=page_indices,
            cu_q_lens=jnp.asarray(cu_q_s0),
            distribution=jnp.asarray([0, 0, len(s0_lens)], dtype=jnp.int32),
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            interleave_size=case.interleave_size,
            q_block_size=case.q_block_size,
            kv_layout=case.kv_layout,
            block_major=case.block_major,
        )
        out_s0.block_until_ready()
        kv_cache.block_until_ready()
        obs_s0 = np.asarray(jax.device_get(out_s0.astype(jnp.float32)))[s0_inv]
        s0_refs = [
            _fp64_causal_gqa_reference(
                full_q[idx][: history_lens[idx]],
                full_k[idx][: history_lens[idx]],
                full_v[idx][: history_lens[idx]],
                q_start_pos=0,
                sm_scale=sm_scale,
                kv_dtype=case.kv_dtype,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            for idx in hist_req_ids
        ]
        exp_s0 = np.concatenate([pair[0] for pair in s0_refs], axis=0)
        v64_s0 = np.concatenate([pair[1] for pair in s0_refs], axis=0)
        s0_tiles = max(math.ceil(h_len / case.page_size) for h_len in hist_only)
        _assert_fp64_parity(
            obs_s0,
            exp_s0,
            v64_s0,
            num_kv_tiles=s0_tiles,
            k_scale=k_scale,
            v_scale=v_scale,
            label=f"{case.label}::step0_history_prefill",
        )

    # Step 1: stream current and cached history pages over the 3-slot ring.
    local_q_padded = max(
        case.q_block_size,
        case.page_size,
        math.ceil(sum(query_lens) / (pcp_size * case.q_block_size)) * case.q_block_size,
    )
    global_q_padded = pcp_size * local_q_padded
    q_s1, k_s1, v_s1, s1_lens, s1_inv = _pack_step_tensors(
        [full_q[idx][history_lens[idx] :] for idx in range(num_reqs)],
        [full_k[idx][history_lens[idx] :] for idx in range(num_reqs)],
        [full_v[idx][history_lens[idx] :] for idx in range(num_reqs)],
        pcp_size=pcp_size,
        interleave_size=case.interleave_size,
        global_padded_tokens=global_q_padded,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    kv_lens_s1 = np.zeros(max_num_seqs, dtype=np.int32)
    kv_lens_s1[:num_reqs] = [h + q for h, q in zip(history_lens, query_lens)]
    cu_q_s1 = np.zeros(max_num_seqs + 1, dtype=np.int32)
    cu_q_s1[1 : num_reqs + 1] = np.cumsum(s1_lens)
    cu_q_s1[num_reqs + 1 :] = cu_q_s1[num_reqs]

    out_s1, updated_cache = _invoke_streaming_attention(
        mesh=mesh,
        q=q_s1,
        k=k_s1,
        v=v_s1,
        kv_cache=kv_cache,
        kv_lens=jnp.asarray(kv_lens_s1),
        page_indices=page_indices,
        cu_q_lens=jnp.asarray(cu_q_s1),
        distribution=jnp.asarray([0, 0, num_reqs], dtype=jnp.int32),
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        interleave_size=case.interleave_size,
        q_block_size=case.q_block_size,
        kv_layout=case.kv_layout,
        block_major=case.block_major,
    )
    out_s1.block_until_ready()
    updated_cache.block_until_ready()

    obs_s1 = np.asarray(jax.device_get(out_s1.astype(jnp.float32)))[s1_inv]
    s1_refs = [
        _fp64_causal_gqa_reference(
            full_q[idx][history_lens[idx] :],
            full_k[idx],
            full_v[idx],
            q_start_pos=history_lens[idx],
            sm_scale=sm_scale,
            kv_dtype=case.kv_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        for idx in range(num_reqs)
    ]
    exp_s1 = np.concatenate([pair[0] for pair in s1_refs], axis=0)
    v64_s1 = np.concatenate([pair[1] for pair in s1_refs], axis=0)
    s1_tiles = max(
        math.ceil((h_len + q_len) / case.page_size)
        for h_len, q_len in zip(history_lens, query_lens)
    )
    _rmse, tol_rmse = _assert_fp64_parity(
        obs_s1,
        exp_s1,
        v64_s1,
        num_kv_tiles=s1_tiles,
        k_scale=k_scale,
        v_scale=v_scale,
        label=f"{case.label}::step1_mixed_ring",
    )

    if case.check_negative_control:
        # Decidability check: dropping cached history pages must violate tol_rmse.
        dropped_hist_ref, _ = _fp64_causal_gqa_reference(
            full_q[0][history_lens[0] :],
            full_k[0][history_lens[0] :],
            full_v[0][history_lens[0] :],
            q_start_pos=0,
            sm_scale=sm_scale,
            kv_dtype=case.kv_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        dropped_rmse = float(
            np.sqrt(
                np.mean(
                    (obs_s1[: query_lens[0]].astype(np.float64) - dropped_hist_ref) ** 2
                )
            )
        )
        assert dropped_rmse > 5.0 * tol_rmse, (
            f"{case.label}: derived tolerance {tol_rmse:.6e} is too loose; "
            f"history-dropped RMSE {dropped_rmse:.6e} did not exceed 5x bound."
        )

    _assert_recorded_3slot_vmem_calls(
        recorded_calls,
        vmem_capacity_bytes=vmem_capacity_bytes,
        label=case.label,
        expected_min_calls=4 if hist_req_ids else 2,
    )
    assert updated_cache.shape == kv_cache_shape
    cache_np = np.asarray(jax.device_get(updated_cache.astype(jnp.float32)))
    assert np.all(np.isfinite(cache_np)), f"{case.label}: non-finite KV cache"
    assert float(np.abs(cache_np).max()) > 0.0, f"{case.label}: KV cache remained zero"


@pytest.mark.parametrize(
    "case",
    SUPPORTED_TPU_CONFIG_CASES,
    ids=[c.label for c in SUPPORTED_TPU_CONFIG_CASES],
)
def test_supported_configs_3slot_vmem_fit_and_fp64_accuracy(
    case: _SupportedTpuConfigCase,
    monkeypatch,
    release_jax_backend,
):
    """Verify dynamic tpu_info 3-slot scratch fit, compile, and float64 accuracy."""
    devices = _require_tpu_devices(WORLD_SIZE)
    seed = 2263 + SUPPORTED_TPU_CONFIG_CASES.index(case)
    _run_supported_tpu_config_case(
        devices,
        case,
        monkeypatch=monkeypatch,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Multi-Host (>8-Core) & 256K-Context / 64-Seq / 32K-Bucket Envelope Check
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "case",
    ALL_PARALLELISM_ENVELOPE_CASES,
    ids=[c.regime_label for c in ALL_PARALLELISM_ENVELOPE_CASES],
)
def test_all_parallelism_regimes_3slot_vmem_envelope(
    case: _ParallelismEnvelopeCase,
    monkeypatch,
):
    """Verify 3-slot KV ring scratch fits dynamic tpu_info across all regimes."""
    tpu_info = _get_dynamic_tpu_info()
    vmem_capacity_bytes = int(tpu_info.vmem_capacity_bytes)
    assert vmem_capacity_bytes > 0

    devices = jax.local_devices()
    with monkeypatch.context() as mp:
        if not devices or devices[0].platform != "tpu":
            mp.setattr(pcp_kernel.pltpu, "get_tpu_info", lambda: tpu_info)
            mp.setattr(brpa_wrapper.pltpu, "get_tpu_info", lambda: tpu_info)

        mp.setattr(
            pcp_kernel,
            "write_captured_kv_to_local_cache",
            lambda kv_cache_local, *_args, **_kwargs: kv_cache_local,
        )

        recorded_calls: list[tuple[str, tuple[object, ...], int]] = []

        def _fake_pallas_call(
            _kernel_fn, *, out_shape, grid_spec, compiler_params, name, **_kw
        ):
            recorded_calls.append(
                (
                    str(name),
                    tuple(grid_spec.scratch_shapes),
                    int(compiler_params.vmem_limit_bytes),
                )
            )

            def _runner(*_args):
                return jax.tree.map(
                    lambda spec: jnp.zeros(spec.shape, dtype=spec.dtype),
                    out_shape,
                )

            return _runner

        mp.setattr(pcp_kernel.pl, "pallas_call", _fake_pallas_call)

        assert case.total_q_heads % case.tp_size == 0
        assert case.total_kv_heads % case.tp_size == 0
        local_q_heads = case.total_q_heads // case.tp_size
        local_kv_heads = case.total_kv_heads // case.tp_size
        assert local_q_heads % local_kv_heads == 0
        q_per_kv = local_q_heads // local_kv_heads
        kv_packing = pcp_wrapper.get_dtype_packing(case.kv_dtype)
        local_tokens = max(
            case.q_block_size,
            case.page_size,
            case.global_bucket_tokens // case.pcp_size,
        )
        pages_per_seq = max(
            2,
            math.ceil(case.max_model_len / (case.pcp_size * case.page_size)),
        )

        q_local = jax.ShapeDtypeStruct(
            (local_tokens, local_kv_heads, q_per_kv, case.head_dim),
            jnp.bfloat16,
        )
        if case.kv_layout == KVLayout.SEQ_ALONG_LANE:
            current_kv_spec = jax.ShapeDtypeStruct(
                (
                    2 * local_kv_heads,
                    case.head_dim // kv_packing,
                    kv_packing,
                    local_tokens,
                ),
                case.kv_dtype,
            )
            kv_cache_spec = jax.ShapeDtypeStruct(
                (
                    4,
                    2 * local_kv_heads,
                    case.head_dim // kv_packing,
                    kv_packing,
                    case.page_size,
                ),
                case.kv_dtype,
            )
        else:
            packed_groups = math.ceil((2 * local_kv_heads) / kv_packing)
            current_kv_spec = jax.ShapeDtypeStruct(
                (local_tokens, packed_groups, kv_packing, case.head_dim),
                case.kv_dtype,
            )
            kv_cache_spec = jax.ShapeDtypeStruct(
                (4, case.page_size, packed_groups, kv_packing, case.head_dim),
                case.kv_dtype,
            )

        kv_lens = np.zeros(case.max_num_seqs, dtype=np.int32)
        kv_lens[0] = case.pcp_size * case.page_size + case.pcp_size * local_tokens
        cu_q_lens = np.zeros(case.max_num_seqs + 1, dtype=np.int32)
        cu_q_lens[1:] = case.pcp_size * local_tokens
        page_indices = jnp.zeros((case.max_num_seqs * pages_per_seq,), dtype=jnp.int32)
        k_scale = NON_TRIVIAL_K_SCALE if case.kv_dtype == jnp.float8_e4m3fn else None
        v_scale = NON_TRIVIAL_V_SCALE if case.kv_dtype == jnp.float8_e4m3fn else None

        def _trace_entry(q_in, cur_kv_in, cache_in):
            p_kernel = pcp_kernel
            fn = p_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata
            return fn(
                q_in,
                cur_kv_in,
                cache_in,
                jnp.asarray(kv_lens),
                page_indices,
                jnp.asarray(cu_q_lens),
                jnp.asarray([0, 0, 1], dtype=jnp.int32),
                jnp.zeros((3, 8), dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                pcp_size=case.pcp_size,
                interleave_size=case.page_size,
                q_block_size=case.q_block_size,
                sm_scale=1.0 / math.sqrt(case.head_dim),
                k_scale=k_scale,
                v_scale=v_scale,
                kv_layout=case.kv_layout,
            )

        jax.eval_shape(_trace_entry, q_local, current_kv_spec, kv_cache_spec)

    _assert_recorded_3slot_vmem_calls(
        recorded_calls,
        vmem_capacity_bytes=vmem_capacity_bytes,
        label=case.regime_label,
        expected_min_calls=2,
    )


def test_cpu_fp64_oracle_and_step_packing_branches():
    """Exercise pure-host oracle, tolerance, packing, and 8-device branches on CPU."""
    assert _is_power_of_two(0.5)
    assert _is_power_of_two(1.0)
    assert _is_power_of_two(2.0)
    assert not _is_power_of_two(NON_TRIVIAL_K_SCALE)
    assert not _is_power_of_two(NON_TRIVIAL_V_SCALE)
    assert not _is_power_of_two(0.0)
    assert not _is_power_of_two(-1.0)

    rng = np.random.default_rng(2263)
    q_f32 = (rng.standard_normal((16, 4, 16)) * 0.5).astype(np.float32)
    k_f32 = (rng.standard_normal((32, 2, 16)) * 0.5).astype(np.float32)
    v_f32 = (rng.standard_normal((32, 2, 16)) * 0.5).astype(np.float32)
    sm_scale = 1.0 / math.sqrt(16)

    for kv_dtype, k_sc, v_sc in (
        (jnp.bfloat16, None, None),
        (jnp.float8_e4m3fn, NON_TRIVIAL_K_SCALE, NON_TRIVIAL_V_SCALE),
    ):
        ref_fp64, v_fp64 = _fp64_causal_gqa_reference(
            q_f32,
            k_f32,
            v_f32,
            q_start_pos=16,
            sm_scale=sm_scale,
            kv_dtype=kv_dtype,
            k_scale=k_sc,
            v_scale=v_sc,
        )
        assert ref_fp64.shape == (16, 4, 16)
        assert v_fp64.shape == (32, 2, 16)
        obs_bf16 = np.asarray(
            jnp.asarray(ref_fp64, dtype=jnp.bfloat16).astype(jnp.float32)
        )
        _assert_fp64_parity(
            obs_bf16,
            ref_fp64,
            v_fp64,
            num_kv_tiles=4,
            k_scale=k_sc,
            v_scale=v_sc,
            label=f"cpu_oracle_{kv_dtype}",
        )
        corrupted = obs_bf16.copy()
        corrupted[0] += 0.25
        with pytest.raises(AssertionError):
            _assert_fp64_parity(
                corrupted,
                ref_fp64,
                v_fp64,
                num_kv_tiles=4,
                k_scale=k_sc,
                v_scale=v_sc,
                label=f"cpu_oracle_corrupted_{kv_dtype}",
            )

    q_reqs = [q_f32[:8], q_f32[8:]]
    k_reqs = [k_f32[:8], k_f32[8:16]]
    v_reqs = [v_f32[:8], v_f32[8:16]]
    q_p, k_p, v_p, q_lens, inv = _pack_step_tensors(
        q_reqs,
        k_reqs,
        v_reqs,
        pcp_size=4,
        interleave_size=4,
        global_padded_tokens=32,
        k_scale=NON_TRIVIAL_K_SCALE,
        v_scale=NON_TRIVIAL_V_SCALE,
    )
    assert q_lens == [8, 8]
    assert q_p.shape == (32, 4, 16)
    assert k_p.shape == (32, 2, 16)
    assert v_p.shape == (32, 2, 16)
    np.testing.assert_array_equal(
        np.asarray(q_p.astype(jnp.float32))[inv],
        np.asarray(jnp.asarray(q_f32, dtype=jnp.bfloat16).astype(jnp.float32)),
    )

    child_env = dict(os.environ)
    existing_xla = child_env.get("XLA_FLAGS", "")
    child_env["XLA_FLAGS"] = (
        f"{existing_xla} --xla_force_host_platform_device_count=8".strip()
    )
    child_env["JAX_PLATFORMS"] = "cpu"
    child_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    child_env["VLLM_PLUGINS"] = ""
    probe_code = (
        "import importlib.util, jax, jax.numpy as jnp\n"
        f"spec = importlib.util.spec_from_file_location('m', {__file__!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "devs = jax.local_devices(); assert len(devs) == 8\n"
        "ti = m._get_dynamic_tpu_info()\n"
        "orig_ti1 = m.pcp_kernel.pltpu.get_tpu_info\n"
        "orig_ti2 = m.brpa_wrapper.pltpu.get_tpu_info\n"
        "orig_wc = m.pcp_kernel.write_captured_kv_to_local_cache\n"
        "orig_pc = m.pcp_kernel.pl.pallas_call\n"
        "try:\n"
        "    m.pcp_kernel.pltpu.get_tpu_info = lambda: ti\n"
        "    m.brpa_wrapper.pltpu.get_tpu_info = lambda: ti\n"
        "    m.pcp_kernel.write_captured_kv_to_local_cache = lambda c, *a, **k: c\n"
        "    def _fake_pc(_f, *, out_shape, **_k):\n"
        "        return lambda *_a: jax.tree.map(\n"
        "            lambda s: jnp.zeros(s.shape, dtype=s.dtype), out_shape\n"
        "        )\n"
        "    m.pcp_kernel.pl.pallas_call = _fake_pc\n"
        "    for idx in (2, 7):\n"
        "        c = m.SUPPORTED_TPU_CONFIG_CASES[idx]\n"
        "        mesh, pcp_sz, tp_sz = m._build_pcp_mesh(devs, c.mesh_shape)\n"
        "        lq = c.q_block_size; gq = pcp_sz * lq\n"
        "        lkv = c.total_kv_heads // tp_sz\n"
        "        pack = m.pcp_wrapper.get_dtype_packing(c.kv_dtype)\n"
        "        if c.kv_layout == m.KVLayout.SEQ_ALONG_LANE:\n"
        "            h_pk = tp_sz * (c.head_dim // pack)\n"
        "            cs = (pcp_sz * 4, 2 * lkv, h_pk, pack, c.page_size)\n"
        "        else:\n"
        "            lg = (2 * lkv + pack - 1) // pack\n"
        "            cs = (pcp_sz * 4, c.page_size, tp_sz * lg, pack, c.head_dim)\n"
        "        if c.block_major:\n"
        "            cs = (cs[0] // 2, 2, *cs[1:])\n"
        "        jax.eval_shape(\n"
        "            lambda q_t, k_t, v_t, kc_t: m._invoke_streaming_attention(\n"
        "                mesh=mesh,\n"
        "                q=q_t,\n"
        "                k=k_t,\n"
        "                v=v_t,\n"
        "                kv_cache=kc_t,\n"
        "                kv_lens=jnp.asarray([gq, 0, 0, 0], dtype=jnp.int32),\n"
        "                page_indices=jnp.zeros((16,), dtype=jnp.int32),\n"
        "                cu_q_lens=jnp.asarray([0, gq, gq, gq, gq], dtype=jnp.int32),\n"
        "                distribution=jnp.asarray([0, 0, 1], dtype=jnp.int32),\n"
        "                sm_scale=0.0625,\n"
        "                k_scale=0.37 if c.kv_dtype == jnp.float8_e4m3fn else None,\n"
        "                v_scale=0.43 if c.kv_dtype == jnp.float8_e4m3fn else None,\n"
        "                interleave_size=c.interleave_size,\n"
        "                q_block_size=c.q_block_size,\n"
        "                kv_layout=c.kv_layout,\n"
        "                block_major=c.block_major,\n"
        "            ),\n"
        "            jax.ShapeDtypeStruct(\n"
        "                (gq, c.total_q_heads, c.head_dim), jnp.bfloat16\n"
        "            ),\n"
        "            jax.ShapeDtypeStruct(\n"
        "                (gq, c.total_kv_heads, c.head_dim), jnp.bfloat16\n"
        "            ),\n"
        "            jax.ShapeDtypeStruct(\n"
        "                (gq, c.total_kv_heads, c.head_dim), jnp.bfloat16\n"
        "            ),\n"
        "            jax.ShapeDtypeStruct(cs, c.kv_dtype),\n"
        "        )\n"
        "finally:\n"
        "    m.pcp_kernel.pltpu.get_tpu_info = orig_ti1\n"
        "    m.brpa_wrapper.pltpu.get_tpu_info = orig_ti2\n"
        "    m.pcp_kernel.write_captured_kv_to_local_cache = orig_wc\n"
        "    m.pcp_kernel.pl.pallas_call = orig_pc\n"
    )
    subprocess.run(
        [sys.executable or "python3", "-c", probe_code],
        env=child_env,
        check=True,
        timeout=60,
    )
