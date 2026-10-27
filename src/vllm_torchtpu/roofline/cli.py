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
"""CLI entrypoint for solve, buckets, emit, ceiling, plan, and explain."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace

from .buckets import recommend_bucket_regimes
from .chips import CHIPS, TOPOLOGIES
from .emit import emit_cdk_ceiling_rows, emit_engine_args
from .geometry import from_hf_config
from .hf_loader import dtypes_from_hf_config, load_hf_config
from .modules import Dtypes, Parallel, StepShape
from .serving import plan
from .solver import SCENARIO_PRESETS, SLO_TIERS, build_scenario, solve_serving_plan
from .step import spec_cycle_floor, step_floor

_WEIGHT_BYTES: dict[str, float] = {"bf16": 2.0, "fp8": 1.0}
_EXPERT_BYTES: dict[str, float] = {
    "bf16": 2.0,
    "fp8": 1.0,
    "nvfp4": 0.5625,
    "mxfp4": 0.53125,
}
_KV_BYTES: dict[str, float] = {"bf16": 2.0, "fp8": 1.0}
_STATE_BYTES: dict[str, float] = {"fp32": 4.0, "bf16": 2.0}


def _explicit_flags(argv: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for tok in argv:
        if tok.startswith("--"):
            out.add(tok.split("=", 1)[0])
    return out


def _add_model_source(p: argparse.ArgumentParser) -> None:
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--config", help="Path to HF config.json")
    grp.add_argument("--model", help="HF model ID, directory, or config.json")


def _add_common_args(p: argparse.ArgumentParser, *, tp_required: bool = True) -> None:
    _add_model_source(p)
    p.add_argument("--chip", default="tpu7x", choices=sorted(CHIPS))
    p.add_argument("--tp", type=int, required=tp_required, default=None)
    p.add_argument("--context", type=int, default=0)
    p.add_argument("--kv-split", type=int, default=1)
    p.add_argument("--kv-merge", default="ag_rs", choices=("ag_rs", "exchange"))
    p.add_argument("--moe", default="tp_slice", choices=("tp_slice", "ep"))
    p.add_argument(
        "--moe-dispatch",
        default="allreduce",
        choices=("allreduce", "a2a_push", "ag_ep", "fused_rs"),
    )
    p.add_argument("--residual", default="replicated", choices=("replicated", "sp"))
    p.add_argument(
        "--ar-algo",
        default="one_shot",
        choices=("one_shot", "two_shot", "ring"),
    )
    p.add_argument("--weight-dtype", default="fp8", choices=("bf16", "fp8"))
    p.add_argument(
        "--expert-dtype",
        default="fp8",
        choices=("bf16", "fp8", "nvfp4", "mxfp4"),
    )
    p.add_argument("--aux-dtype", default="bf16", choices=("bf16", "fp8"))
    p.add_argument("--kv-dtype", default="fp8", choices=("bf16", "fp8"))
    p.add_argument("--state-dtype", default="fp32", choices=("fp32", "bf16"))
    p.add_argument("--act", default="a8", choices=("a8", "a16"))
    p.add_argument("--ici-bw-gbps", type=float, default=None)
    p.add_argument("--collective-latency-us", type=float, default=None)
    p.add_argument("--json", action="store_true")


def _add_scenario_args(p: argparse.ArgumentParser) -> None:
    _add_model_source(p)
    p.add_argument("--chip", default="tpu7x", choices=sorted(CHIPS))
    p.add_argument("--scenario", default="chat-8k1k", choices=sorted(SCENARIO_PRESETS))
    p.add_argument("--mode", default="auto", choices=("auto", "mixed", "disagg"))
    p.add_argument("--max-chips", type=int, default=None)
    p.add_argument("--slo-tier", default="none", choices=sorted(SLO_TIERS))
    p.add_argument("--ttft-slo-ms", type=float, default=None)
    p.add_argument("--tpot-slo-ms", type=float, default=None)
    p.add_argument("--isl", type=int, default=None)
    p.add_argument("--osl", type=int, default=None)
    p.add_argument("--concurrency", default=None)
    p.add_argument("--prefix-hit-rate", type=float, default=None)
    p.add_argument("--accept-len", type=float, default=None)
    p.add_argument("--json", action="store_true")


def _topology_for_cores(cores: int) -> str | None:
    for name, topo in TOPOLOGIES.items():
        if topo.num_cores == cores:
            return name
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the roofline CLI subcommand."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    explicit = _explicit_flags(raw_argv)
    parser = argparse.ArgumentParser(prog="python -m vllm_torchtpu.roofline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_solve = sub.add_parser("solve")
    _add_scenario_args(p_solve)

    p_buck = sub.add_parser("buckets")
    _add_model_source(p_buck)
    p_buck.add_argument("--chip", default="tpu7x", choices=sorted(CHIPS))
    p_buck.add_argument("--topology", default=None, choices=sorted(TOPOLOGIES))
    p_buck.add_argument("--tp", type=int, default=8)
    p_buck.add_argument("--ep", type=int, default=None)
    p_buck.add_argument("--moe", default="ep", choices=("tp_slice", "ep"))
    p_buck.add_argument(
        "--scenario", default="chat-8k1k", choices=sorted(SCENARIO_PRESETS)
    )
    p_buck.add_argument("--context", type=int, default=8192)
    p_buck.add_argument("--spec-window", type=int, default=0)
    p_buck.add_argument("--acceptance-length", type=float, default=None)
    p_buck.add_argument("--max-num-batched-tokens", type=int, default=None)
    p_buck.add_argument("--buckets", default=None)
    p_buck.add_argument("--bucket-override", default=None)
    p_buck.add_argument("--engine", default="vllm", choices=("vllm", "sglang"))
    p_buck.add_argument("--json", action="store_true")

    p_emit = sub.add_parser("emit")
    _add_scenario_args(p_emit)
    p_emit.add_argument("--engine", default="vllm", choices=("vllm", "sglang"))
    p_emit.add_argument("--bucket-override", default=None)
    p_emit.add_argument("--target-envs", default=None)
    p_emit.add_argument("--strict", action="store_true")
    p_emit.add_argument("--topology", default=None, choices=sorted(TOPOLOGIES))
    p_emit.add_argument("--tp", type=int, default=None)
    p_emit.add_argument("--ep", type=int, default=None)
    p_emit.add_argument("--moe", default=None, choices=("tp_slice", "ep"))
    p_emit.add_argument("--spec-window", type=int, default=0)
    p_emit.add_argument("--acceptance-length", type=float, default=None)
    p_emit.add_argument("--max-num-batched-tokens", type=int, default=None)

    p_ceil = sub.add_parser("ceiling")
    _add_scenario_args(p_ceil)

    p_plan = sub.add_parser("plan")
    _add_common_args(p_plan)
    p_plan.add_argument("--concurrency", default="1,4,16,64")
    p_plan.add_argument("--spec-window", type=int, default=0)
    p_plan.add_argument("--acceptance-length", type=float, default=1.0)

    p_exp = sub.add_parser("explain")
    _add_common_args(p_exp)
    p_exp.add_argument("--batch", type=int, default=1)
    p_exp.add_argument("--rows-per-seq", type=int, default=1)
    p_exp.add_argument("--spec-window", type=int, default=0)

    args = parser.parse_args(raw_argv)
    src = args.model if args.model is not None else args.config
    cfg = load_hf_config(src)
    geom = from_hf_config(cfg)
    chip = CHIPS[args.chip]

    if args.cmd in ("solve", "emit", "ceiling"):
        concs = (
            tuple(int(x) for x in args.concurrency.split(",") if x.strip())
            if args.concurrency
            else None
        )
        eff_accept = (
            args.acceptance_length
            if (args.cmd == "emit" and args.acceptance_length is not None)
            else args.accept_len
        )
        scen = build_scenario(
            args.scenario,
            isl=args.isl,
            osl=args.osl,
            concurrencies=concs,
            prefix_hit_rate=args.prefix_hit_rate,
            accept_len=eff_accept,
            slo_tier=args.slo_tier,
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
        )
        dt_auto = dtypes_from_hf_config(cfg)
        max_chips = args.max_chips
        if max_chips is None and args.cmd == "emit" and args.topology is not None:
            max_chips = TOPOLOGIES[args.topology].num_cores // 2
        splan = solve_serving_plan(
            geom, dt_auto, chip, scen, mode=args.mode, max_chips=max_chips
        )
        if args.cmd == "solve":
            if args.json:
                sys.stdout.write(json.dumps(asdict(splan)) + "\n")
                return 0
            for w in splan.warnings:
                sys.stdout.write(f"WARNING: {w}\n")
            sys.stdout.write(
                f"{'mode':<7}  {'c':>5}  {'chips':>5}  {'pools':<24}  "
                f"{'p90_int_tok_s':>13}  {'tok_tput_chip':>13}  {'p90_ttft_ms':>11}  "
                f"{'p90_tpot_ms':>11}\n"
            )
            for pt in splan.frontier:
                if pt.prefill_pool is not None:
                    pools = (
                        f"{pt.prefill_pool.instances}x{pt.prefill_pool.topology}P+"
                        f"{pt.decode_pool.instances}x{pt.decode_pool.topology}D"
                    )
                else:
                    pools = f"1x{pt.decode_pool.topology}(TP{pt.decode_pool.tp})"
                sys.stdout.write(
                    f"{pt.mode:<7}  {pt.concurrency:>5}  {pt.total_chips:>5}  "
                    f"{pools:<24}  {pt.p90_interactivity_tok_s:>13.2f}  "
                    f"{pt.token_throughput_per_chip:>13.1f}  "
                    f"{pt.p90_ttft_ms:>11.1f}  {pt.p90_tpot_ms:>11.2f}\n"
                )
            return 0

        if args.cmd == "ceiling":
            rows_ceil = emit_cdk_ceiling_rows(splan)
            if args.json:
                sys.stdout.write(json.dumps(rows_ceil) + "\n")
                return 0
            for rc_row in rows_ceil:
                sys.stdout.write(json.dumps(rc_row) + "\n")
            return 0

        rec = splan.recommended
        if args.tp is not None or args.ep is not None or args.topology is not None:
            tp_e = (
                args.tp if args.tp is not None else (rec.decode_pool.tp if rec else 8)
            )
            moe_e = (
                args.moe
                if args.moe is not None
                else ("ep" if geom.num_experts > 0 else "tp_slice")
            )
            ep_e = args.ep if args.ep is not None else (tp_e if moe_e == "ep" else 1)
            topo_name_e = (
                args.topology
                or (rec.decode_pool.topology if rec else None)
                or _topology_for_cores(max(tp_e, ep_e))
            )
            topo_emit = TOPOLOGIES.get(topo_name_e) if topo_name_e is not None else None
            dp_e = (
                max(1, ep_e // max(1, tp_e)) if (moe_e == "ep" and ep_e > tp_e) else 1
            )
            par_emit = Parallel(
                tp=tp_e,
                dp=dp_e,
                ep=ep_e,
                moe=moe_e,
                moe_dispatch=(
                    "ag_ep"
                    if (moe_e == "ep" and (dp_e > 1 or ep_e >= 16))
                    else "allreduce"
                ),
            )
        elif rec is not None:
            d_pool = rec.decode_pool
            topo_emit = TOPOLOGIES.get(d_pool.topology)
            par_emit = Parallel(
                tp=d_pool.tp,
                dp=d_pool.dp,
                ep=d_pool.ep,
                pp=d_pool.pp,
                pcp=d_pool.pcp,
                kv_split=d_pool.kv_split,
                moe="ep" if d_pool.ep > 1 else "tp_slice",
                moe_dispatch=(
                    "ag_ep"
                    if (d_pool.ep > 1 and (d_pool.dp > 1 or d_pool.ep >= 16))
                    else "allreduce"
                ),
                mesh_axes=d_pool.mesh_axes,
            )
        else:
            topo_emit = TOPOLOGIES["2x2x1"]
            par_emit = Parallel(
                tp=8,
                moe="ep" if geom.num_experts > 0 else "tp_slice",
                ep=8 if geom.num_experts > 0 else 1,
            )
        bplan_emit = recommend_bucket_regimes(
            geom,
            dt_auto,
            chip,
            topo_emit,
            par_emit,
            context=scen.isl,
            overrides=args.bucket_override,
            engine=args.engine,
            scenario=scen,
            spec_window=args.spec_window,
            acceptance_length=args.acceptance_length,
            max_num_batched_tokens=args.max_num_batched_tokens,
        )
        emitted = emit_engine_args(
            splan,
            bplan_emit,
            engine=args.engine,
            dtypes=dt_auto,
            target_envs=args.target_envs,
        )
        if args.strict and emitted["refused_flags"]:
            sys.stderr.write(
                f"Refused flags absent from target envs: {emitted['refused_flags']}\n"
            )
            return 1
        if args.json:
            sys.stdout.write(json.dumps(emitted) + "\n")
            return 0
        env_prefix = " ".join(f"{k}={v}" for k, v in emitted["env"].items())
        cli_joined = " ".join(emitted["cli_args"])
        sys.stdout.write(f"{env_prefix} {args.engine} {cli_joined}\n")
        return 0

    if args.cmd == "buckets":
        dt_b = dtypes_from_hf_config(cfg)
        ep_b = args.ep if args.ep is not None else (args.tp if args.moe == "ep" else 1)
        topo_name = args.topology or _topology_for_cores(max(args.tp, ep_b))
        topo_b = TOPOLOGIES.get(topo_name) if topo_name is not None else None
        dp_b = (
            max(1, ep_b // max(1, args.tp))
            if (args.moe == "ep" and ep_b > args.tp)
            else 1
        )
        par_b = Parallel(
            tp=args.tp,
            dp=dp_b,
            moe=args.moe,
            ep=ep_b,
            moe_dispatch="ag_ep"
            if (args.moe == "ep" and (dp_b > 1 or ep_b >= 16))
            else "allreduce",
        )
        b_list = (
            [int(x) for x in args.buckets.split(",") if x.strip()]
            if args.buckets
            else None
        )
        scen_b = (
            build_scenario(args.scenario, accept_len=args.acceptance_length)
            if "--scenario" in explicit
            else None
        )
        bplan = recommend_bucket_regimes(
            geom,
            dt_b,
            chip,
            topo_b,
            par_b,
            buckets=b_list,
            context=args.context,
            overrides=args.bucket_override,
            engine=args.engine,
            scenario=scen_b,
            spec_window=args.spec_window,
            acceptance_length=args.acceptance_length,
            max_num_batched_tokens=args.max_num_batched_tokens,
        )
        if args.json:
            sys.stdout.write(json.dumps(asdict(bplan)) + "\n")
            return 0
        for band in bplan.bands:
            sys.stdout.write(
                f"{band.min_tokens}..{band.max_tokens}: residual={band.residual} "
                f"dispatch={band.moe_dispatch} mk={int(band.megakernel)} "
                f"({band.source})\n"
            )
        return 0

    if args.ici_bw_gbps is not None:
        chip = replace(chip, ici_bw=args.ici_bw_gbps * 1e9)
    lat_s = (args.collective_latency_us or 0.0) * 1e-6

    if args.tp is None or args.tp <= 0:
        raise ValueError(f"--tp must be >= 1, got {args.tp}")
    par = Parallel(
        tp=args.tp,
        kv_split=args.kv_split,
        kv_merge=args.kv_merge,
        moe=args.moe,
        moe_dispatch=args.moe_dispatch,
        residual=args.residual,
        ar_algo=args.ar_algo,
    )
    dt = Dtypes(
        w_dense=_WEIGHT_BYTES[args.weight_dtype],
        w_expert=_EXPERT_BYTES[args.expert_dtype],
        w_aux=_WEIGHT_BYTES[args.aux_dtype],
        kv=_KV_BYTES[args.kv_dtype],
        state=_STATE_BYTES[args.state_dtype],
        act=args.act,
    )

    if args.cmd == "plan":
        concs_list = [int(x) for x in args.concurrency.split(",") if x.strip()]
        rows = plan(
            geom=geom,
            chip=chip,
            par=par,
            dt=dt,
            context=args.context,
            concurrencies=concs_list,
            spec_window=args.spec_window,
            acceptance_length=args.acceptance_length,
            collective_latency_us=args.collective_latency_us,
        )
        if args.json:
            sys.stdout.write(json.dumps([asdict(r) for r in rows]) + "\n")
            return 0
        hdr = (
            f"{'c':>4}  {'rows':>6}  {'binding':<7}  "
            f"{'hard_floor_ms':>13}  {'serial_floor_ms':>15}  "
            f"{'tpot_floor_ms':>13}  {'interactivity_ceil_tok_s':>24}  "
            f"{'output_tput_ceil_per_chip':>25}  {'flag':<7}\n"
        )
        sys.stdout.write(hdr)
        for r in rows:
            line = (
                f"{r.c:>4}  {r.rows:>6}  {r.binding:<7}  "
                f"{r.hard_floor_ms:>13.3f}  {r.serial_floor_ms:>15.3f}  "
                f"{r.tpot_floor_ms:>13.3f}  {r.interactivity_ceil_tok_s:>24.1f}  "
                f"{r.output_tput_ceil_per_chip:>25.1f}  {r.flag:<7}\n"
            )
            sys.stdout.write(line)
        return 0

    if args.spec_window > 0:
        sf = spec_cycle_floor(
            geom,
            chip,
            par,
            args.batch,
            args.context,
            args.spec_window,
            dt,
            collective_latency_s=lat_s,
        )
    else:
        sf = step_floor(
            geom=geom,
            chip=chip,
            par=par,
            shape=StepShape(
                batch=args.batch,
                rows_per_seq=args.rows_per_seq,
                context=args.context,
            ),
            dt=dt,
            collective_latency_s=lat_s,
        )
    if args.json:
        sys.stdout.write(
            json.dumps(
                [
                    {
                        **asdict(m),
                        "ici_bytes": m.ici_bytes,
                        "n_collectives": m.n_collectives,
                        "chain_hops": m.chain_hops,
                    }
                    for m in sf.modules
                ]
            )
            + "\n"
        )
        return 0
    sys.stdout.write(
        f"{'module':<10}  {'hbm_mb':>10}  {'fp8_gflops':>10}  "
        f"{'bf16_gflops':>11}  {'collectives':>11}  {'wire_kb':>10}  {'hops':>6}\n"
    )
    for m in sf.modules:
        wire_kb = sum(c.wire_bytes for c in m.collectives) / 1e3
        sys.stdout.write(
            f"{m.name:<10}  {m.hbm_bytes / 1e6:>10.2f}  "
            f"{m.fp8_flops / 1e9:>10.2f}  {m.bf16_flops / 1e9:>11.2f}  "
            f"{m.n_collectives:>11}  {wire_kb:>10.2f}  {m.chain_hops:>6}\n"
        )
    return 0
