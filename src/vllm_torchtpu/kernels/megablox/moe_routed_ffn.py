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
"""Small-batch MoE expert layer from the router's top-k straight to [T, H].

With replicated tokens (TP + EP) every rank holds the whole batch and a decode
step hits only a few of its local experts. This kernel takes the global expert
ids and routing weights as they leave the router: the vector and matrix units
place each local route in an expert-major route list while the first experts'
weights stream in, the scalar unit stores the list and picks the active
experts, every listed expert's gate/up,
activation and down run with gmm_v2's tile arithmetic over the rows of the
tokens routed to it, and each token's routing-weighted expert output is
accumulated in f32. No EP mask, sort, permute or combine runs around it.

Each route's expert output is rounded to the out dtype before weighting, as
the grouped-GEMM path rounds `gmm2_res`, and a token's experts are summed in
ascending expert id, the order of that path's expert-sorted combine rows.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    FusedWeightsRef,
    GmmConfigs,
    WeightsRef,
    align_to,
    apply_act_fn,
    tile_matmul,
)
from vllm_torchtpu.kernels.megablox.gmm_v2_ffn import make_ffn_configs

# Most weight slots: the expert being computed plus experts in flight. Fewer
# are used when the token-sized buffers leave no room for this many.
MAX_NBUF = 3
# Most routed rows one expert matmul takes; an expert with more runs in chunks.
ROW_CHUNK = 128
# Indices into an expert's (w1, w1 scale, w2, w2 scale) copies.
_UP, _DOWN = (0, 1), (2, 3)
# A route is stored as token << _K_BITS | top-k slot.
_K_BITS = 5
MAX_TOPK = 1 << _K_BITS


def _default_vmem_limit() -> int:
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)


def _padded_rows(num_tokens: int, dtype: jnp.dtype) -> int:
    return align_to(num_tokens, pltpu.get_tpu_info().get_sublane_tiling(dtype))


def _chunk_rows(num_tokens: int, dtype: jnp.dtype) -> int:
    """Rows per expert matmul: a power of two, so chunk counts are shifts."""
    rows = _padded_rows(num_tokens, dtype)
    return min(ROW_CHUNK, 1 << (rows - 1).bit_length())


def _vmem_estimate(up: GmmConfigs, down: GmmConfigs, nbuf: int) -> int:
    tokens = up.dims.size_m
    rows = _chunk_rows(tokens, up.lhs_cfgs.dtype)
    hidden, inter = down.out_size_n, up.out_size_n
    weight_bits = jax.dtypes.itemsize_bits(up.rhs_cfgs.quant_dtype or up.rhs_cfgs.dtype)
    weights = (up.dims.size_k * 2 * inter + inter * hidden) * weight_bits // 8
    # f32 [blocks, 1, n] scale blocks pad the singleton sublane to 8.
    scale_lanes = up.num_quant_blocks_per_tile_k * 2 * inter
    scale_lanes += down.num_quant_blocks_per_tile_k * hidden
    scales = 8 * 4 * scale_lanes
    total = nbuf * (weights + scales)
    # Tokens (bf16, reused for the output), f32 tokens and the f32 accumulator.
    total += tokens * hidden * (2 + 4 + 4)
    total += rows * hidden * 4  # one chunk's gathered rows, then its outputs
    total += rows * 2 * inter * (4 + 4) + rows * hidden * 4  # tile values
    total += 2 * align_to(tokens, 8) * 128 * 4  # routing ids and weights
    return int(total)


def make_routed_ffn_configs(
    hidden_states: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    *,
    fuse_act: str,
    vmem_limit_bytes: int | None = None,
) -> tuple[GmmConfigs, GmmConfigs, int] | None:
    """gmm_v2 configs for both legs over all padded token rows and the number
    of expert weight slots, or None.

    None when the expert weights do not fit one tile per leg or two experts'
    weights and the token buffers do not fit in VMEM together.
    """
    if vmem_limit_bytes is None:
        vmem_limit_bytes = _default_vmem_limit()
    num_tokens, size_k = hidden_states.shape
    rows = _padded_rows(num_tokens, hidden_states.dtype)
    lhs = jax.ShapeDtypeStruct((rows, size_k), hidden_states.dtype)
    group_sizes = jax.ShapeDtypeStruct((w1.shape[0],), jnp.int32)
    cfgs = make_ffn_configs(
        lhs,
        w1,
        w1_scale,
        None,
        w2,
        w2_scale,
        None,
        group_sizes,
        fuse_act=fuse_act,
        maybe_quantize_lhs=True,
        rhs_quant_dtype=None,
        out_dtype=hidden_states.dtype,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    if cfgs is None:
        return None
    for nbuf in range(MAX_NBUF, 1, -1):
        if _vmem_estimate(*cfgs, nbuf) <= vmem_limit_bytes:
            return (*cfgs, nbuf)
    return None


def _row_buckets(rows: int, base: int) -> list[int]:
    """Row counts the per-expert matmul is compiled for: doubling from `base`."""
    buckets = [base]
    while buckets[-1] < rows:
        buckets.append(min(2 * buckets[-1], rows))
    return buckets


def _route_slots(
    ids: jax.Array,  # int32[topk, T]: global expert ids
    weights: jax.Array,  # f32[topk, T]
    first_expert: jax.Array,  # int32[]: global id of local expert 0
    *,
    num_experts: int,
    skip_zero_weights: bool,
) -> tuple[list[jax.Array], jax.Array]:
    """Each route's index in the expert-major route list and the routes per expert.

    Local routes of expert e fill e * T + [0, count) in token-major order, as a
    serial scan over the routes would place them; every other route goes to the
    spare entry E * T. Returns topk int32[T] index rows and int32[E] counts.
    """
    topk, num_tokens = ids.shape
    local = ids - first_expert
    valid = jnp.logical_and(local >= 0, local < num_experts)
    if skip_zero_weights:
        valid = jnp.logical_and(valid, weights != 0)
    local = jnp.where(valid, local, num_experts)
    expert = lax.broadcasted_iota(jnp.int32, (num_experts, num_tokens), 0)
    hits = [(local[k : k + 1] == expert).astype(jnp.float32) for k in range(topk)]
    routes = functools.reduce(jnp.add, hits)  # [E, T]: routes of token t on e
    # Routes of earlier tokens, by a 0/1 matmul: exact in the f32 accumulator.
    earlier = lax.broadcasted_iota(jnp.int32, (num_tokens, num_tokens), 0) < (
        lax.broadcasted_iota(jnp.int32, (num_tokens, num_tokens), 1)
    )
    rank = jnp.dot(
        routes.astype(jnp.bfloat16),
        earlier.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )
    first_index = rank + (expert * num_tokens).astype(jnp.float32)
    rows = []
    for k in range(topk):
        index = jnp.sum(hits[k] * first_index, axis=0, keepdims=True)
        # Top-k ids are distinct, but a repeated id still gets its own entry.
        for j in range(k):
            index += (local[j : j + 1] == local[k : k + 1]).astype(jnp.float32)
        index = jnp.where(valid[k : k + 1], index, num_experts * num_tokens)
        rows.append(index.astype(jnp.int32).reshape(num_tokens))
    return rows, jnp.sum(routes, axis=1).astype(jnp.int32)


def _routed_ffn_kernel(
    # Scalar prefetch
    start_sm: jax.Array,  # int32[1]: global id of local expert 0
    weights_sm: jax.Array,  # f32[T * topk] routing weights
    # In
    ids_hbm: jax.Array,  # int32[T, topk]: global expert ids
    rw_hbm: jax.Array,  # [T, topk]: routing weights
    x_hbm: jax.Array,  # [T, H]
    w1_hbm: jax.Array,  # [E, H, 2 * I]: gate then up
    s1_hbm: jax.Array,  # f32[E, blocks, 1, 2 * I]
    w2_hbm: jax.Array,  # [E, I, H]
    s2_hbm: jax.Array,  # f32[E, blocks, 1, H]
    # Out
    out_hbm: jax.Array,  # [T, H]
    # Scratch
    w1_vm: jax.Array,  # [nbuf, H, 2 * I]
    s1_vm: jax.Array,  # [nbuf, blocks, 1, 2 * I]
    w2_vm: jax.Array,  # [nbuf, I, H]
    s2_vm: jax.Array,  # [nbuf, blocks, 1, H]
    sems: jax.Array,  # DMA[4, nbuf]
    x_vm: jax.Array,  # [T, H]: the tokens, then the output
    x_sem: jax.Array,  # DMA
    x32_vm: jax.Array,  # f32[T, H]
    rows_vm: jax.Array,  # f32[rows, H]: one chunk's gathered rows, then outputs
    acc_vm: jax.Array,  # f32[T, H]
    ids_vm: jax.Array,  # int32[T, topk]
    rw_vm: jax.Array,  # [T, topk]
    index_vm: list[jax.Array],  # topk x int32[T]: staging for index_sm
    count_vm: jax.Array,  # int32[E]: staging for count_sm
    meta_sems: jax.Array,  # DMA[topk + 3]
    index_sm: list[jax.Array],  # topk x int32[T]: each route's index in route_sm
    count_sm: jax.Array,  # int32[E]: routes per expert
    visit_sm: jax.Array,  # int32[E]
    route_sm: jax.Array,  # int32[E * T + 1]: each expert's routes, token-major
    *,
    up: GmmConfigs,
    down: GmmConfigs,
    topk: int,
    nbuf: int,
    skip_zero_weights: bool,
):
    num_tokens = x_vm.shape[0]
    num_experts = visit_sm.shape[0]
    num_rows = rows_vm.shape[0]
    inter = up.out_size_n
    pairs = ((w1_hbm, w1_vm), (s1_hbm, s1_vm), (w2_hbm, w2_vm), (s2_hbm, s2_vm))

    def copy(i, expert, slot):
        src, dst = pairs[i]
        return pltpu.make_async_copy(src.at[expert], dst.at[slot], sems.at[i, slot])

    def start(expert, slot, operands):
        for i in operands:
            copy(i, expert, slot).start()

    def wait(slot, operands):
        for i in operands:
            copy(i, 0, slot).wait()

    # The routing inputs gate everything else, so they are requested first.
    ids_copy = pltpu.make_async_copy(ids_hbm, ids_vm, meta_sems.at[0])
    rw_copy = pltpu.make_async_copy(rw_hbm, rw_vm, meta_sems.at[1])
    ids_copy.start()
    rw_copy.start()

    # The lowest local experts are the first visits whenever they are active,
    # which at decode batch sizes is almost always, so their weights are
    # requested before the routing and land while it runs.
    for slot in range(min(nbuf, num_experts)):
        start(slot, slot, _UP + _DOWN)

    # The token rows are first read after the routing, so their copy runs under it.
    x_copy = pltpu.make_async_copy(x_hbm, x_vm, x_sem)
    x_copy.start()

    # Route indices and counts come from the vector and matrix units; the
    # scalar unit then only stores each route at its index, with no
    # read-modify-write of a counter between routes.
    ids_copy.wait()
    rw_copy.wait()
    index, counts = _route_slots(
        jnp.transpose(ids_vm[...]),
        jnp.transpose(rw_vm[...].astype(jnp.float32)),
        start_sm[0],
        num_experts=num_experts,
        skip_zero_weights=skip_zero_weights,
    )
    # One 1-D array per top-k slot keeps every scalar read at base + token.
    meta_copies = [
        pltpu.make_async_copy(src, dst, meta_sems.at[2 + i])
        for i, (src, dst) in enumerate(
            zip([*index_vm, count_vm], [*index_sm, count_sm], strict=True)
        )
    ]
    for k in range(topk):
        index_vm[k][...] = index[k]
    count_vm[...] = counts
    for c in meta_copies:
        c.start()
    x_copy.wait()
    x32_vm[...] = x_vm[...].astype(jnp.float32)
    acc_vm[...] = jnp.zeros_like(acc_vm)
    for c in meta_copies:
        c.wait()

    def store_token(t, carry):
        for k in range(topk):
            route_sm[index_sm[k][t]] = (t << _K_BITS) | k
        return carry

    lax.fori_loop(0, num_tokens, store_token, 0)

    # Visit order is ascending expert id: each token then accumulates its
    # experts in the order of the grouped path's expert-sorted rows.
    def list_expert(e, n_visit):
        visit_sm[n_visit] = e
        return n_visit + (count_sm[e] > 0).astype(jnp.int32)

    n_visit = lax.fori_loop(0, num_experts, list_expert, jnp.int32(0))

    # Re-request a slot whose speculative expert is not its visit.
    for slot in range(min(nbuf, num_experts)):
        hit = jnp.logical_and(slot < n_visit, visit_sm[slot] == slot)

        @pl.when(jnp.logical_not(hit))
        def _(slot=slot):
            wait(slot, _UP + _DOWN)

            @pl.when(slot < n_visit)
            def _():
                start(visit_sm[slot], slot, _UP + _DOWN)

    buckets = _row_buckets(num_rows, _padded_rows(1, x_vm.dtype))
    chunk_shift = num_rows.bit_length() - 1

    def visit(j, carry):
        expert = visit_sm[j]
        slot = lax.rem(j, nbuf)
        count = count_sm[expert]
        base = expert * num_tokens
        num_chunks = (count + num_rows - 1) >> chunk_shift

        w13 = WeightsRef(weight=w1_vm.at[slot], scale=s1_vm.at[slot], bias=None)
        w13 = FusedWeightsRef(
            gate=jax.tree.map(lambda r: r.at[..., :inter], w13),
            up=jax.tree.map(lambda r: r.at[..., inter:], w13),
        )
        w2 = WeightsRef(weight=w2_vm.at[slot], scale=s2_vm.at[slot], bias=None)

        def chunk(c, carry):
            first = base + (c << chunk_shift)
            n = jnp.minimum(count - (c << chunk_shift), num_rows)

            # Rows go through f32 because a single bf16 row is not addressable
            # at a dynamic offset; bf16 -> f32 -> bf16 is exact.
            @pl.loop(0, n)
            def _(i):
                token = route_sm[first + i] >> _K_BITS
                rows_vm[pl.ds(i, 1), :] = x32_vm[pl.ds(token, 1), :]

            @pl.when(c == 0)
            def _():
                wait(slot, _UP)

            # Rows past `n` are stale. Every op is row-local and the combine
            # reads only the first `n` rows, so they never reach the output.
            def run_rows(bucket_m: int):
                matmul = functools.partial(
                    tile_matmul, bucket_m=bucket_m, is_last_k_step=True
                )
                lhs = rows_vm[:bucket_m, :].astype(x_vm.dtype)
                hidden = apply_act_fn(matmul(lhs, w13, cfgs=up), up.fuse_act)

                # The last expert's gate/up runs while its down weights land.
                @pl.when(c == 0)
                def _():
                    wait(slot, _DOWN)

                acc = matmul(hidden.astype(up.out_dtype), w2, cfgs=down)
                # The per-route output is rounded where the grouped path stores it.
                rows_vm[:bucket_m, :] = acc.astype(down.out_dtype).astype(jnp.float32)

            bucket_idx = sum((n > b).astype(jnp.int32) for b in buckets[:-1])
            lax.switch(bucket_idx, [functools.partial(run_rows, b) for b in buckets])

            # The slot's weights are read for the last time; refill it with
            # the expert `nbuf` visits ahead while this chunk is combined.
            @pl.when(jnp.logical_and(c == num_chunks - 1, j + nbuf < n_visit))
            def _():
                start(visit_sm[j + nbuf], slot, _UP + _DOWN)

            @pl.loop(0, n)
            def _(i):
                route = route_sm[first + i]
                token, k = route >> _K_BITS, route & ((1 << _K_BITS) - 1)
                acc_vm[pl.ds(token, 1), :] += (
                    weights_sm[token * topk + k] * rows_vm[pl.ds(i, 1), :]
                )

            return carry

        lax.fori_loop(0, num_chunks, chunk, 0)
        return carry

    lax.fori_loop(0, n_visit, visit, 0)
    x_vm[...] = acc_vm[...].astype(x_vm.dtype)
    out_copy = pltpu.make_async_copy(x_vm, out_hbm, x_sem)
    out_copy.start()
    out_copy.wait()


def _f32_scale(scale: jax.Array) -> jax.Array:
    if scale.ndim == 3:
        scale = jnp.expand_dims(scale, 2)
    return scale.astype(jnp.float32)


@jax.jit(static_argnames=["fuse_act", "skip_zero_weights", "vmem_limit_bytes"])
def moe_routed_ffn(
    hidden_states: jax.Array,  # [T, H]
    topk_ids: jax.Array,  # int32[T, topk]: global expert ids
    topk_weights: jax.Array,  # [T, topk]
    w1: jax.Array,  # [E, H, 2 * I]: gate then up along the last dim
    w1_scale: jax.Array,  # [E, blocks, (1,) 2 * I]
    w2: jax.Array,  # [E, I, H]
    w2_scale: jax.Array,  # [E, blocks, (1,) H]
    experts_start: jax.Array | None = None,  # int32[]: global id of local expert 0
    *,
    fuse_act: str,
    skip_zero_weights: bool = False,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
    """sum_k w[t, k] * ffn_{ids[t, k]}(x[t]) over the routes local to this rank.

    A route is local if its id is in [experts_start, experts_start + E) and,
    with `skip_zero_weights`, its weight is nonzero. Only for shapes
    `make_routed_ffn_configs` accepts.
    """
    if vmem_limit_bytes is None:
        vmem_limit_bytes = _default_vmem_limit()
    w1_scale, w2_scale = _f32_scale(w1_scale), _f32_scale(w2_scale)
    cfgs = make_routed_ffn_configs(
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        fuse_act=fuse_act,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    if cfgs is None or topk_ids.shape[1] > MAX_TOPK:
        raise ValueError(
            f"moe_routed_ffn cannot run {hidden_states.shape=} {w1.shape=} "
            f"{w2.shape=} {topk_ids.shape=}; check make_routed_ffn_configs and "
            "MAX_TOPK before calling it."
        )
    up, down, nbuf = cfgs
    num_tokens, hidden = hidden_states.shape
    topk = topk_ids.shape[1]
    num_experts = w1.shape[0]
    rows = _chunk_rows(num_tokens, hidden_states.dtype)
    topk_ids = topk_ids.astype(jnp.int32)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scratch_shapes = [
        pltpu.VMEM((nbuf, *w1.shape[1:]), w1.dtype),
        pltpu.VMEM((nbuf, *w1_scale.shape[1:]), jnp.float32),
        pltpu.VMEM((nbuf, *w2.shape[1:]), w2.dtype),
        pltpu.VMEM((nbuf, *w2_scale.shape[1:]), jnp.float32),
        pltpu.SemaphoreType.DMA((4, nbuf)),
        pltpu.VMEM((num_tokens, hidden), hidden_states.dtype),
        pltpu.SemaphoreType.DMA(()),
        pltpu.VMEM((num_tokens, hidden), jnp.float32),
        pltpu.VMEM((rows, hidden), jnp.float32),
        pltpu.VMEM((num_tokens, hidden), jnp.float32),
        pltpu.VMEM((num_tokens, topk), jnp.int32),
        pltpu.VMEM((num_tokens, topk), topk_weights.dtype),
        [pltpu.VMEM((num_tokens,), jnp.int32)] * topk,
        pltpu.VMEM((num_experts,), jnp.int32),
        pltpu.SemaphoreType.DMA((topk + 3,)),
        [pltpu.SMEM((num_tokens,), jnp.int32)] * topk,
        pltpu.SMEM((num_experts,), jnp.int32),
        pltpu.SMEM((num_experts,), jnp.int32),
        pltpu.SMEM((num_experts * num_tokens + 1,), jnp.int32),
    ]
    weight_bytes = sum(
        a.size * jax.dtypes.itemsize_bits(a.dtype) // 8
        for a in (w1, w1_scale, w2, w2_scale)
    )
    name = (
        f"gmm_v2_routed_ffn-g_{num_experts}-m_{num_tokens}-k_{hidden}"
        f"-i_{up.out_size_n}-n_{down.out_size_n}-act_{fuse_act}-b_{nbuf}"
    )
    return pl.pallas_call(
        functools.partial(
            _routed_ffn_kernel,
            up=up,
            down=down,
            topk=topk,
            nbuf=nbuf,
            skip_zero_weights=skip_zero_weights,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, hidden), hidden_states.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[hbm] * 7,
            out_specs=hbm,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes, disable_bounds_checks=True
        ),
        name=name,
        cost_estimate=pl.CostEstimate(
            # Upper bound: every local expert active.
            flops=2 * num_tokens * hidden * 3 * up.out_size_n * num_experts,
            bytes_accessed=weight_bytes + 2 * hidden_states.size * 2,
            transcendentals=0,
        ),
    )(
        jnp.zeros((1,), jnp.int32)
        if experts_start is None
        else jnp.reshape(experts_start, (1,)).astype(jnp.int32),
        topk_weights.astype(jnp.float32).reshape(-1),
        topk_ids,
        topk_weights,
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
    )
