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
"""Unconstrained single-pool mixed and xP/yD disaggregated TPU mesh solver."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from .chips import TOPOLOGIES, ChipSpec, SliceTopology
from .collectives import submesh_for_group
from .geometry import ModelGeometry
from .modules import Dtypes, Parallel, StepShape
from .step import spec_cycle_floor, step_floor


@dataclass(frozen=True)
class TrafficScenario:
    """User-selected workload distribution and optional p90 SLO bounds."""

    name: str
    isl: int
    osl: int
    concurrencies: tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128, 256)
    prefix_hit_rate: float = 0.0
    accept_len: float = 1.0
    prefill_chunk: int = 2048
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None


SCENARIO_PRESETS: dict[str, TrafficScenario] = {
    "chat-8k1k": TrafficScenario(
        name="chat-8k1k",
        isl=8192,
        osl=1024,
        concurrencies=(1, 4, 8, 16, 32, 64, 128, 256),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        prefill_chunk=2048,
    ),
    "reasoning-1k8k": TrafficScenario(
        name="reasoning-1k8k",
        isl=1024,
        osl=8192,
        concurrencies=(1, 4, 8, 16, 32, 64, 128, 256),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        prefill_chunk=2048,
    ),
    "coding-agent": TrafficScenario(
        name="coding-agent",
        isl=16384,
        osl=1024,
        concurrencies=(1, 4, 8, 16, 32, 64, 128),
        prefix_hit_rate=0.92,
        accept_len=3.4,
        prefill_chunk=2048,
    ),
    "custom": TrafficScenario(
        name="custom",
        isl=4096,
        osl=1024,
        concurrencies=(1, 4, 16, 64),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        prefill_chunk=2048,
    ),
}

SLO_TIERS: dict[str, tuple[float | None, float | None]] = {
    "none": (None, None),
    "interactive": (5000.0, 33.33),
    "throughput": (5000.0, 66.67),
}


@dataclass(frozen=True)
class PoolMeshPlan:
    """Topology, parallelism, and HBM footprint for one serving pool."""

    role: Literal["mixed", "prefill", "decode"]
    instances: int
    topology: str
    chips_per_instance: int
    tp: int
    dp: int
    ep: int
    pp: int
    pcp: int
    kv_split: int
    mesh_axes: tuple[int, int, int, int]
    hbm_used_gb: float
    hbm_limit_gb: float


@dataclass(frozen=True)
class ParetoPoint:
    """One feasible operating point on the p90 interactivity vs throughput frontier."""

    mode: Literal["mixed", "disagg"]
    concurrency: int
    total_chips: int
    prefill_pool: PoolMeshPlan | None
    decode_pool: PoolMeshPlan
    p90_ttft_ms: float
    p90_tpot_ms: float
    p90_interactivity_tok_s: float
    output_throughput_per_chip: float
    token_throughput_per_chip: float
    p_mixed: float
    binding: str
    slo_met: bool


@dataclass(frozen=True)
class ServingPlan:
    """Ranked serving plan with recommended point, Pareto frontier, and warnings."""

    model_layers: int
    chip: str
    scenario: TrafficScenario
    mode: str
    recommended: ParetoPoint | None
    frontier: tuple[ParetoPoint, ...]
    warnings: tuple[str, ...]


def hbm_footprint_per_core_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
    seqs_per_dp: int,
    context: int,
) -> float:
    """Static model weights plus active KV/recurrent state bytes on one TensorCore."""
    tp = max(1, par.tp)
    pp = max(1, par.pp)
    ep = par.ep if (par.moe == "ep" and par.ep > 1) else (tp if par.moe == "ep" else 1)

    n_gqa = 0
    n_swa = 0
    n_mla = 0
    n_gdn = 0
    n_kda = 0
    for tok in geom.layer_types:
        if tok in ("mla", "mla_attention", "dsa", "dsa_attention", "sparse_mla"):
            n_mla += 1
        elif tok in ("sliding_attention", "swa", "sliding_window_attention"):
            n_swa += 1
        elif tok in ("kda", "kimi_linear", "kda_attention"):
            n_kda += 1
        elif tok in ("linear_attention", "gdn", "gated_deltanet", "mamba"):
            n_gdn += 1
        elif geom.kv_lora_rank > 0:
            n_mla += 1
        else:
            n_gqa += 1

    q_mult = 2 if geom.attn_output_gate else 1
    kv_div = min(tp, max(1, geom.num_kv_heads))
    w_attn = 0.0
    if n_gqa + n_swa > 0:
        w_q_o = (
            float(n_gqa + n_swa)
            * float(
                geom.hidden * geom.num_heads * geom.head_dim * q_mult
                + geom.num_heads * geom.head_dim * geom.hidden
            )
            * dt.w_dense
            / (tp * pp)
        )
        w_kv = (
            float(n_gqa + n_swa)
            * float(2 * geom.hidden * geom.num_kv_heads * geom.head_dim)
            * dt.w_dense
            / (kv_div * pp)
        )
        w_attn += w_q_o + w_kv
    if n_mla > 0:
        q_lora = geom.q_lora_rank or max(512, geom.hidden // 4)
        kv_lora = geom.kv_lora_rank or 512
        nope_d = geom.qk_nope_head_dim or 128
        rope_d = geom.qk_rope_head_dim or 64
        v_d = geom.v_head_dim or nope_d
        w_mla = (
            float(n_mla)
            * (
                float(geom.hidden * q_lora + geom.hidden * (kv_lora + rope_d)) / pp
                + float(
                    q_lora * geom.num_heads * (nope_d + rope_d)
                    + kv_lora * geom.num_heads * (nope_d + v_d)
                    + geom.num_heads * v_d * geom.hidden
                )
                / (tp * pp)
            )
            * dt.w_dense
        )
        w_attn += w_mla

    w_rec = 0.0
    if n_gdn > 0:
        kh = max(1, geom.linear_k_heads // tp)
        vh = max(1, geom.linear_v_heads // tp)
        qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
        z_r = vh * geom.linear_v_dim
        w_rec += (
            float(n_gdn)
            * (
                float(geom.hidden * (qkv_r + z_r) + z_r * geom.hidden) * dt.w_dense
                + float(geom.hidden * 2 * vh + qkv_r * geom.conv_kernel) * dt.w_aux
            )
            / pp
        )
    if n_kda > 0:
        kh = max(1, (geom.linear_k_heads or geom.num_heads) // tp)
        vh = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
        kd = geom.linear_k_dim or geom.head_dim
        vd = geom.linear_v_dim or geom.head_dim
        w_rec += (
            float(n_kda)
            * float(geom.hidden * (2 * kh * kd + 2 * vh * vd) + vh * vd * geom.hidden)
            * dt.w_dense
            / pp
        )

    w_ffn = 0.0
    if geom.num_experts > 0:
        n_dense_ffn = min(geom.num_layers, geom.first_k_dense_replace)
        n_moe_ffn = geom.num_layers - n_dense_ffn
        if n_dense_ffn > 0 and geom.dense_inter > 0:
            w_ffn += (
                float(n_dense_ffn * 3 * geom.hidden * geom.dense_inter)
                * dt.w_dense
                / (tp * pp)
            )
        routed_div = ep * pp if par.moe == "ep" else tp * pp
        w_ffn += (
            float(n_moe_ffn * geom.num_experts * 3 * geom.hidden * geom.moe_inter)
            * dt.w_expert
            / routed_div
        )
        if geom.shared_inter > 0:
            isr_eff = (
                geom.shared_inter
                if par.residual == "sp"
                else max(128, geom.shared_inter // tp)
            )
            w_ffn += float(n_moe_ffn * 3 * geom.hidden * isr_eff) * dt.w_dense / pp
        w_ffn += float(n_moe_ffn * geom.hidden * geom.num_experts) * dt.router / pp
    else:
        w_ffn += (
            float(geom.num_layers * 3 * geom.hidden * geom.dense_inter)
            * dt.w_dense
            / (tp * pp)
        )

    w_head = float(2 * geom.vocab * geom.hidden) * dt.lm_head / (tp * pp)
    w_mtp = (
        float(geom.mtp_layers * 2 * geom.hidden * geom.hidden) * dt.w_aux / tp
        if geom.mtp_layers > 0
        else 0.0
    )

    kvh = max(1, geom.num_kv_heads // tp) if geom.num_kv_heads > 0 else 0
    kv_gqa = (
        float(seqs_per_dp * n_gqa * 2 * kvh * geom.head_dim * context)
        * dt.kv
        / (par.kv_split * par.pcp * pp)
    )
    sw_ctx = min(context, geom.sliding_window) if geom.sliding_window > 0 else context
    kv_swa = (
        float(seqs_per_dp * n_swa * 2 * kvh * geom.head_dim * sw_ctx)
        * dt.kv
        / (par.kv_split * par.pcp * pp)
    )
    latent_w = (geom.kv_lora_rank or 512) + (geom.qk_rope_head_dim or 64)
    kv_mla = (
        float(seqs_per_dp * n_mla * latent_w * context)
        * dt.kv
        / (par.kv_split * par.pcp * pp)
    )

    st_bytes = 0.0
    if n_gdn > 0:
        kh = max(1, geom.linear_k_heads // tp)
        vh = max(1, geom.linear_v_heads // tp)
        qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
        st_bytes += (
            float(seqs_per_dp * n_gdn)
            * (
                float(vh * geom.linear_k_dim * geom.linear_v_dim) * dt.state
                + float(qkv_r * max(0, geom.conv_kernel - 1)) * dt.conv
            )
            / pp
        )
    if n_kda > 0:
        vh = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
        kd = geom.linear_k_dim or geom.head_dim
        vd = geom.linear_v_dim or geom.head_dim
        st_bytes += float(seqs_per_dp * n_kda * vh * kd * vd) * dt.state / pp

    return w_attn + w_rec + w_ffn + w_head + w_mtp + kv_gqa + kv_swa + kv_mla + st_bytes


def _divisors(n: int) -> tuple[int, ...]:
    return tuple(d for d in range(1, n + 1) if n % d == 0)


def _candidate_decode_parallels(
    geom: ModelGeometry, topo: SliceTopology
) -> list[Parallel]:
    cores = topo.num_cores
    out: list[Parallel] = []
    for tp in _divisors(cores):
        if geom.num_heads > 0 and tp > geom.num_heads:
            continue
        dp = cores // tp
        kv_group = (
            max(1, tp // geom.num_kv_heads)
            if geom.num_kv_heads > 0 and geom.kv_lora_rank == 0
            else tp
        )
        kv_splits = (1, kv_group) if kv_group > 1 else (1,)
        moe_mode: Literal["tp_slice", "ep"] = (
            "ep" if geom.num_experts > 0 and cores > 1 else "tp_slice"
        )
        ep_val = cores if moe_mode == "ep" else 1
        if moe_mode == "ep" and geom.num_experts < ep_val:
            continue
        disp: Literal["allreduce", "ag_ep"] = (
            "ag_ep" if (moe_mode == "ep" and (dp > 1 or cores >= 16)) else "allreduce"
        )
        mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)
        for kvs in kv_splits:
            out.append(
                Parallel(
                    tp=tp,
                    kv_split=kvs,
                    moe=moe_mode,
                    moe_dispatch=disp,
                    ranks_per_chip=topo.cores_per_chip,
                    dp=dp,
                    ep=ep_val,
                    pp=1,
                    pcp=1,
                    mesh_axes=mesh_ax,
                )
            )
    return out


def _candidate_prefill_parallels(
    geom: ModelGeometry, topo: SliceTopology
) -> list[Parallel]:
    cores = topo.num_cores
    out: list[Parallel] = []
    for pp in (1, 2, 4):
        if cores % pp != 0 or geom.num_layers % pp != 0:
            continue
        rem = cores // pp
        for pcp in (1, 2, 4, 8):
            if rem % pcp != 0:
                continue
            tp = rem // pcp
            if tp < 1 or (geom.num_heads > 0 and tp > geom.num_heads):
                continue
            moe_mode: Literal["tp_slice", "ep"] = (
                "ep"
                if geom.num_experts > 0 and (tp * pcp) > 1 and geom.num_experts >= tp
                else "tp_slice"
            )
            ep_val = tp if moe_mode == "ep" else 1
            mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)
            out.append(
                Parallel(
                    tp=tp,
                    kv_split=1,
                    moe=moe_mode,
                    moe_dispatch="ag_ep"
                    if (moe_mode == "ep" and tp >= 16)
                    else "allreduce",
                    ranks_per_chip=topo.cores_per_chip,
                    dp=1,
                    ep=ep_val,
                    pp=pp,
                    pcp=pcp,
                    mesh_axes=mesh_ax,
                )
            )
    return out


def _pareto_filter(points: list[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    if not points:
        return ()
    ordered = sorted(
        points,
        key=lambda p: (
            -p.p90_interactivity_tok_s,
            -p.token_throughput_per_chip,
            p.total_chips,
        ),
    )
    kept: list[ParetoPoint] = []
    best_tput = -1.0
    for pt in ordered:
        if pt.token_throughput_per_chip > best_tput * (1.0 + 1e-9):
            kept.append(pt)
            best_tput = pt.token_throughput_per_chip
    return tuple(kept)


def solve_serving_plan(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    *,
    mode: Literal["auto", "mixed", "disagg"] = "auto",
    max_chips: int | None = None,
) -> ServingPlan:
    """Search single-pool mixed and unconstrained xP/yD disagg plans."""
    hbm_limit_b = chip.usable_hbm_frac * float(chip.hbm_bytes)
    hbm_limit_gb = hbm_limit_b / 1e9
    eff_al = (
        max(1.0, scenario.accept_len)
        if geom.mtp_layers > 0 and scenario.accept_len > 1.0
        else 1.0
    )
    spec_k = max(geom.mtp_layers, math.ceil(eff_al) - 1) if eff_al > 1.0 else 0
    ctx_full = scenario.isl + scenario.osl
    ctx_mid = scenario.isl + scenario.osl // 2
    eff_isl = max(1, math.ceil((1.0 - scenario.prefix_hit_rate) * scenario.isl))
    n_chunks = max(1, math.ceil(eff_isl / max(1, scenario.prefill_chunk)))
    chunk_tok = min(eff_isl, scenario.prefill_chunk)

    warnings: list[str] = []
    min_c1_tpot_ms = math.inf
    min_req_ttft_ms = math.inf

    topos = [
        t for t in TOPOLOGIES.values() if max_chips is None or t.num_chips <= max_chips
    ]

    # Precompute feasible prefill instances per topology (fastest rate per chip).
    prefill_options: list[tuple[SliceTopology, Parallel, float, float, float]] = []
    for topo in topos:
        best_for_topo: tuple[SliceTopology, Parallel, float, float, float] | None = None
        for par in _candidate_prefill_parallels(geom, topo):
            hbm_b = hbm_footprint_per_core_bytes(geom, dtypes, par, 1, scenario.isl)
            if hbm_b > hbm_limit_b:
                continue
            sf_p = step_floor(
                geom,
                chip,
                par,
                StepShape(
                    batch=0,
                    rows_per_seq=0,
                    context=0,
                    state_writes=0,
                    prefill_tokens=scenario.isl,
                    prefix_hit_rate=scenario.prefix_hit_rate,
                ),
                dtypes,
                topology=topo,
            )
            ttft_ms = (
                sf_p.serial_floor_s
                * (1.0 + float(par.pp - 1) / float(max(1, n_chunks)))
                * 1e3
            )
            min_req_ttft_ms = min(min_req_ttft_ms, ttft_ms)
            rate_p = 1.0 / max(1e-9, sf_p.serial_floor_s)
            if best_for_topo is None or rate_p > best_for_topo[3]:
                best_for_topo = (topo, par, ttft_ms, rate_p, hbm_b / 1e9)
        if best_for_topo is not None:
            prefill_options.append(best_for_topo)

    candidates: list[ParetoPoint] = []

    # 1. Single-pool mixed candidates
    for topo in topos:
        for par in _candidate_decode_parallels(geom, topo):
            hbm_c1 = hbm_footprint_per_core_bytes(geom, dtypes, par, 1, ctx_full)
            if hbm_c1 <= hbm_limit_b:
                sf_c1 = (
                    spec_cycle_floor(
                        geom, chip, par, 1, ctx_mid, spec_k, dtypes, topology=topo
                    )
                    if spec_k > 0
                    else step_floor(
                        geom,
                        chip,
                        par,
                        StepShape(batch=1, context=ctx_mid),
                        dtypes,
                        topology=topo,
                    )
                )
                min_c1_tpot_ms = min(
                    min_c1_tpot_ms, sf_c1.serial_floor_s * 1e3 / eff_al
                )

            if mode not in ("auto", "mixed"):
                continue

            sf_ttft = step_floor(
                geom,
                chip,
                par,
                StepShape(
                    batch=0,
                    rows_per_seq=0,
                    context=0,
                    state_writes=0,
                    prefill_tokens=scenario.isl,
                    prefix_hit_rate=scenario.prefix_hit_rate,
                ),
                dtypes,
                topology=topo,
            )
            min_req_ttft_ms = min(min_req_ttft_ms, sf_ttft.serial_floor_s * 1e3)

            for c in scenario.concurrencies:
                b_local = math.ceil(c / max(1, par.dp))
                hbm_b = hbm_footprint_per_core_bytes(
                    geom, dtypes, par, b_local, ctx_full
                )
                if hbm_b > hbm_limit_b:
                    continue
                sf_dec = (
                    spec_cycle_floor(
                        geom, chip, par, b_local, ctx_mid, spec_k, dtypes, topology=topo
                    )
                    if spec_k > 0
                    else step_floor(
                        geom,
                        chip,
                        par,
                        StepShape(batch=b_local, context=ctx_mid),
                        dtypes,
                        topology=topo,
                    )
                )
                sf_mix = step_floor(
                    geom,
                    chip,
                    par,
                    StepShape(
                        batch=b_local,
                        rows_per_seq=spec_k + 1,
                        context=ctx_mid,
                        state_writes=spec_k + 1,
                        prefill_tokens=chunk_tok,
                        prefix_hit_rate=0.0,
                    ),
                    dtypes,
                    topology=topo,
                )
                dec_steps_per_req = max(1.0, scenario.osl / eff_al)
                total_req_work_s = dec_steps_per_req * sf_dec.serial_floor_s + float(
                    c * n_chunks
                ) * sf_mix.serial_floor_s / float(max(1, par.dp))
                rate_req_s = float(c) / max(1e-9, total_req_work_s)
                p_mix = min(
                    1.0,
                    rate_req_s
                    * float(n_chunks)
                    * sf_mix.serial_floor_s
                    / float(max(1, par.dp)),
                )
                p90_step_s = (
                    sf_mix.serial_floor_s if p_mix >= 0.10 else sf_dec.serial_floor_s
                )
                p90_tpot_ms = p90_step_s * 1e3 / eff_al
                p90_ttft_ms = sf_ttft.serial_floor_s * 1e3 + (
                    sf_mix.serial_floor_s * 1e3 if c > 1 else 0.0
                )
                interactivity = 1000.0 / max(1e-6, p90_tpot_ms)
                out_tput = rate_req_s * float(scenario.osl) / float(topo.num_chips)
                tot_tput = (
                    rate_req_s
                    * float(scenario.isl + scenario.osl)
                    / float(topo.num_chips)
                )
                slo_ok = (
                    scenario.ttft_slo_ms is None or p90_ttft_ms <= scenario.ttft_slo_ms
                ) and (
                    scenario.tpot_slo_ms is None or p90_tpot_ms <= scenario.tpot_slo_ms
                )
                pool = PoolMeshPlan(
                    role="mixed",
                    instances=1,
                    topology=topo.name,
                    chips_per_instance=topo.num_chips,
                    tp=par.tp,
                    dp=par.dp,
                    ep=par.ep,
                    pp=par.pp,
                    pcp=par.pcp,
                    kv_split=par.kv_split,
                    mesh_axes=(
                        par.mesh_axes
                        if par.mesh_axes is not None
                        else submesh_for_group(par.tp, topo, topo.cores_per_chip)
                    ),
                    hbm_used_gb=hbm_b / 1e9,
                    hbm_limit_gb=hbm_limit_gb,
                )
                candidates.append(
                    ParetoPoint(
                        mode="mixed",
                        concurrency=c,
                        total_chips=topo.num_chips,
                        prefill_pool=None,
                        decode_pool=pool,
                        p90_ttft_ms=p90_ttft_ms,
                        p90_tpot_ms=p90_tpot_ms,
                        p90_interactivity_tok_s=interactivity,
                        output_throughput_per_chip=out_tput,
                        token_throughput_per_chip=tot_tput,
                        p_mixed=p_mix,
                        binding=sf_mix.binding if p_mix >= 0.10 else sf_dec.binding,
                        slo_met=slo_ok,
                    )
                )

    # 2. Unconstrained balanced and imbalanced xP/yD disaggregated candidates
    if mode in ("auto", "disagg") and prefill_options:
        for topo_d in topos:
            for par_d in _candidate_decode_parallels(geom, topo_d):
                for c in scenario.concurrencies:
                    b_local = math.ceil(c / max(1, par_d.dp))
                    hbm_d = hbm_footprint_per_core_bytes(
                        geom, dtypes, par_d, b_local, ctx_full
                    )
                    if hbm_d > hbm_limit_b:
                        continue
                    sf_d = (
                        spec_cycle_floor(
                            geom,
                            chip,
                            par_d,
                            b_local,
                            ctx_mid,
                            spec_k,
                            dtypes,
                            topology=topo_d,
                        )
                        if spec_k > 0
                        else step_floor(
                            geom,
                            chip,
                            par_d,
                            StepShape(batch=b_local, context=ctx_mid),
                            dtypes,
                            topology=topo_d,
                        )
                    )
                    p90_tpot_ms = sf_d.serial_floor_s * 1e3 / eff_al
                    interactivity = 1000.0 / max(1e-6, p90_tpot_ms)
                    dec_steps_per_req = max(1.0, scenario.osl / eff_al)
                    rate_d_one = float(c) / (
                        dec_steps_per_req * max(1e-9, sf_d.serial_floor_s)
                    )

                    for topo_p, par_p, ttft_ms, rate_p_one, hbm_p_gb in prefill_options:
                        if max_chips is None:
                            ratio_xp = math.ceil(rate_d_one / max(1e-9, rate_p_one)) + 1
                            ratio_yd = math.ceil(rate_p_one / max(1e-9, rate_d_one)) + 1
                            max_xp = min(16, max(6, ratio_xp))
                            max_yd = min(16, max(6, ratio_yd))
                        else:
                            max_xp = min(16, max(1, max_chips // topo_p.num_chips))
                            max_yd = min(16, max(1, max_chips // topo_d.num_chips))
                        for x_p in range(1, max_xp + 1):
                            for y_d in range(1, max_yd + 1):
                                total_chips = (
                                    x_p * topo_p.num_chips + y_d * topo_d.num_chips
                                )
                                if max_chips is not None and total_chips > max_chips:
                                    continue
                                coupled_rate = min(x_p * rate_p_one, y_d * rate_d_one)
                                out_tput = (
                                    coupled_rate
                                    * float(scenario.osl)
                                    / float(total_chips)
                                )
                                tot_tput = (
                                    coupled_rate
                                    * float(scenario.isl + scenario.osl)
                                    / float(total_chips)
                                )
                                slo_ok = (
                                    scenario.ttft_slo_ms is None
                                    or ttft_ms <= scenario.ttft_slo_ms
                                ) and (
                                    scenario.tpot_slo_ms is None
                                    or p90_tpot_ms <= scenario.tpot_slo_ms
                                )
                                p_pool = PoolMeshPlan(
                                    role="prefill",
                                    instances=x_p,
                                    topology=topo_p.name,
                                    chips_per_instance=topo_p.num_chips,
                                    tp=par_p.tp,
                                    dp=par_p.dp,
                                    ep=par_p.ep,
                                    pp=par_p.pp,
                                    pcp=par_p.pcp,
                                    kv_split=par_p.kv_split,
                                    mesh_axes=(
                                        par_p.mesh_axes
                                        if par_p.mesh_axes is not None
                                        else submesh_for_group(
                                            par_p.tp, topo_p, topo_p.cores_per_chip
                                        )
                                    ),
                                    hbm_used_gb=hbm_p_gb,
                                    hbm_limit_gb=hbm_limit_gb,
                                )
                                d_pool = PoolMeshPlan(
                                    role="decode",
                                    instances=y_d,
                                    topology=topo_d.name,
                                    chips_per_instance=topo_d.num_chips,
                                    tp=par_d.tp,
                                    dp=par_d.dp,
                                    ep=par_d.ep,
                                    pp=par_d.pp,
                                    pcp=par_d.pcp,
                                    kv_split=par_d.kv_split,
                                    mesh_axes=(
                                        par_d.mesh_axes
                                        if par_d.mesh_axes is not None
                                        else submesh_for_group(
                                            par_d.tp, topo_d, topo_d.cores_per_chip
                                        )
                                    ),
                                    hbm_used_gb=hbm_d / 1e9,
                                    hbm_limit_gb=hbm_limit_gb,
                                )
                                candidates.append(
                                    ParetoPoint(
                                        mode="disagg",
                                        concurrency=c * y_d,
                                        total_chips=total_chips,
                                        prefill_pool=p_pool,
                                        decode_pool=d_pool,
                                        p90_ttft_ms=ttft_ms,
                                        p90_tpot_ms=p90_tpot_ms,
                                        p90_interactivity_tok_s=interactivity,
                                        output_throughput_per_chip=out_tput,
                                        token_throughput_per_chip=tot_tput,
                                        p_mixed=0.0,
                                        binding=sf_d.binding,
                                        slo_met=slo_ok,
                                    )
                                )

    if (
        scenario.tpot_slo_ms is not None
        and math.isfinite(min_c1_tpot_ms)
        and scenario.tpot_slo_ms < min_c1_tpot_ms
    ):
        warnings.append(
            f"INFEASIBLE_SLO: tpot_slo_ms={scenario.tpot_slo_ms:.2f} < "
            f"single-request floor {min_c1_tpot_ms:.2f} ms"
        )
    if (
        scenario.ttft_slo_ms is not None
        and math.isfinite(min_req_ttft_ms)
        and scenario.ttft_slo_ms < min_req_ttft_ms
    ):
        warnings.append(
            f"INFEASIBLE_SLO: ttft_slo_ms={scenario.ttft_slo_ms:.2f} < "
            f"single-request prefill floor {min_req_ttft_ms:.2f} ms"
        )

    slo_candidates = [p for p in candidates if p.slo_met]
    pool_for_frontier = slo_candidates if slo_candidates else candidates
    if candidates and not slo_candidates and not warnings:
        warnings.append("SLO_UNMET: no feasible topology met both TTFT and TPOT bounds")

    # Keep full frontier plus any distinct pool geometry points if needed.
    frontier = _pareto_filter(pool_for_frontier)
    recommended = (
        max(frontier, key=lambda p: p.token_throughput_per_chip) if frontier else None
    )
    return ServingPlan(
        model_layers=geom.num_layers,
        chip=chip.name,
        scenario=scenario,
        mode=mode,
        recommended=recommended,
        frontier=frontier,
        warnings=tuple(warnings),
    )


def build_scenario(
    preset: str = "chat-8k1k",
    *,
    isl: int | None = None,
    osl: int | None = None,
    concurrencies: tuple[int, ...] | None = None,
    prefix_hit_rate: float | None = None,
    accept_len: float | None = None,
    slo_tier: str = "none",
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
) -> TrafficScenario:
    """Resolve a TrafficScenario from a preset name and optional flag overrides."""
    base = SCENARIO_PRESETS[preset]
    tier_ttft, tier_tpot = SLO_TIERS[slo_tier]
    return replace(
        base,
        isl=isl if isl is not None else base.isl,
        osl=osl if osl is not None else base.osl,
        concurrencies=concurrencies
        if concurrencies is not None
        else base.concurrencies,
        prefix_hit_rate=(
            prefix_hit_rate if prefix_hit_rate is not None else base.prefix_hit_rate
        ),
        accept_len=accept_len if accept_len is not None else base.accept_len,
        ttft_slo_ms=ttft_slo_ms if ttft_slo_ms is not None else tier_ttft,
        tpot_slo_ms=tpot_slo_ms if tpot_slo_ms is not None else tier_tpot,
    )
