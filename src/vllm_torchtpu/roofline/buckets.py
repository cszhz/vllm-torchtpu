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
"""Stage-2 compile-bucket regime scorer, band compactor, and zero-reshard guard."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from .chips import ChipSpec, SliceTopology
from .geometry import ModelGeometry
from .modules import Dtypes, Parallel, StepShape
from .registry import KNOBS
from .solver import SCENARIO_PRESETS, TrafficScenario
from .step import step_floor

ResidualMode = Literal["replicated", "sp"]
DispatchMode = Literal["allreduce", "a2a_push", "ag_ep", "fused_rs"]
RegimeSource = Literal["explicit_override", "roofline_auto"]

DEFAULT_VLLM_BUCKETS: tuple[int, ...] = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
)
DEFAULT_SGLANG_BUCKETS: tuple[int, ...] = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
)

_SHARDING_KEYS: tuple[str, ...] = ("tp", "dp", "ep", "pp", "pcp", "kv_split", "moe")
_DTYPE_KEYS: tuple[str, ...] = (
    "w_dense",
    "w_expert",
    "w_aux",
    "kv",
    "state",
    "lm_head",
    "act",
)


@dataclass(frozen=True)
class BucketRegimeRow:
    """Winning zero-reshard regime and floor breakdown for one bucket."""

    bucket_tokens: int
    residual: ResidualMode
    moe_dispatch: DispatchMode
    megakernel: bool
    serial_floor_ms: float
    hard_floor_ms: float
    hbm_ms: float
    mm_ms: float
    chain_ms: float
    scratchpad_bytes: int
    binding: str
    source: RegimeSource


@dataclass(frozen=True)
class BucketThresholdBand:
    """Contiguous monotone compile-bucket band sharing one execution regime."""

    min_tokens: int
    max_tokens: int
    residual: ResidualMode
    moe_dispatch: DispatchMode
    megakernel: bool
    source: RegimeSource


@dataclass(frozen=True)
class BucketRegimePlan:
    """Per-bucket winners, compacted 2..3 threshold bands, and env threshold cutoffs."""

    rows: tuple[BucketRegimeRow, ...]
    bands: tuple[BucketThresholdBand, ...]
    megakernel_max_tokens: int
    sp_min_tokens: int | None
    fused_ep_min_tokens: int | None
    precedence: str = "explicit_override > roofline_auto"


# Double-buffered 3-projection (gate, up, down) weight staging tile bounds.
_MK_DOUBLE_BUFFER = 2
_MK_FFN_PROJECTIONS = 3
_MK_MIN_INTER_TILE = 128
_MK_MAX_INTER_TILE = 1024
# Persistent bf16 activation + top-k expert intermediate staging slots.
_MK_ACT_STAGE_SLOTS = 8
_MK_TOPK_STAGE_SLOTS = 2
_MK_BF16_ACT_BYTES = 2


def _sp_supported(geom: ModelGeometry, parallel: Parallel) -> bool:
    return (
        parallel.tp > 1
        and parallel.dp <= 1
        and parallel.pp <= 1
        and (geom.num_experts == 0 or parallel.moe == "ep")
    )


def megakernel_scratchpad_bytes(
    geom: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    bucket_tokens: int,
) -> int:
    """Estimate per-layer weight tile and activation bytes in scratchpad."""
    raw_inter = geom.moe_inter or geom.dense_inter or geom.hidden
    sharded_inter = (
        raw_inter
        if (geom.num_experts > 0 and parallel.moe == "ep")
        else max(1, raw_inter // max(1, parallel.tp))
    )
    inter_tile = min(max(_MK_MIN_INTER_TILE, sharded_inter), _MK_MAX_INTER_TILE)
    base_tile_b = int(
        _MK_DOUBLE_BUFFER
        * _MK_FFN_PROJECTIONS
        * geom.hidden
        * inter_tile
        * dtypes.w_dense
    )
    act_slots = _MK_ACT_STAGE_SLOTS + _MK_TOPK_STAGE_SLOTS * max(1, geom.top_k)
    tok_tile_b = int(
        max(1, bucket_tokens) * geom.hidden * act_slots * _MK_BF16_ACT_BYTES
    )
    return base_tile_b + tok_tile_b


def _bucket_step_shape(
    scenario: TrafficScenario,
    spec_window: int,
    bucket_tokens: int,
    *,
    acceptance_length: float | None = None,
) -> StepShape:
    """Build a mean-field mixed ``StepShape`` for ``bucket_tokens``."""
    in_len = int(scenario.isl)
    out_len = int(scenario.osl)
    prefix_hit = float(scenario.prefix_hit_rate)
    al = (
        float(acceptance_length)
        if (acceptance_length is not None and acceptance_length > 0.0)
        else max(float(scenario.accept_len), 1e-6)
    )
    k = max(0, int(spec_window))
    rows_per_seq = k + 1
    p_rows = float(in_len) * (1.0 - prefix_hit)
    v_rows = (float(out_len) / al) * float(rows_per_seq)
    denom = p_rows + v_rows
    phi = (p_rows / denom) if denom > 0.0 else 0.0

    t_tok = max(1, int(bucket_tokens))
    prefill_est = min(max(0, int(round(phi * t_tok))), t_tok)
    batch = (t_tok - prefill_est) // rows_per_seq
    # Any remainder smaller than one (k + 1) verify group counts as prefill so
    # batch * (k + 1) + prefill_tokens == t_tok holds identically for every T >= 1.
    prefill_tokens = t_tok - batch * rows_per_seq
    context = int(round(in_len + out_len / 2.0))
    state_writes = rows_per_seq if batch > 0 else 0
    return StepShape(
        batch=batch,
        rows_per_seq=rows_per_seq,
        context=context,
        state_writes=state_writes,
        prefill_tokens=prefill_tokens,
        prefix_hit_rate=prefix_hit,
    )


bucket_step_shape = _bucket_step_shape


def parse_bucket_overrides(raw: str) -> list[dict[str, Any]]:
    """Parse ``min..max:residual,dispatch,mk=0|1`` clauses separated by semicolons."""
    out: list[dict[str, Any]] = []
    for clause in raw.split(";"):
        item = clause.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid bucket override clause {item!r}: missing ':'")
        span_part, spec_part = item.split(":", 1)
        if ".." in span_part:
            lo_s, hi_s = span_part.split("..", 1)
            lo, hi = int(lo_s.strip()), int(hi_s.strip())
        else:
            lo = hi = int(span_part.strip())
        if lo < 1 or hi < lo:
            raise ValueError(f"Invalid token range {span_part!r} in override")
        parts = [p.strip() for p in spec_part.split(",") if p.strip()]
        if len(parts) < 2:
            raise ValueError(
                f"Override {item!r} must specify residual,moe_dispatch[,mk=0|1]"
            )
        residual = parts[0]
        dispatch = parts[1]
        mk = False
        extra: dict[str, Any] = {}
        for extra_tok in parts[2:]:
            if "=" not in extra_tok:
                raise ValueError(f"Invalid key=value token {extra_tok!r} in override")
            k, v = [x.strip() for x in extra_tok.split("=", 1)]
            if k in ("mk", "megakernel"):
                mk = v.lower() in ("1", "true", "yes")
            elif k in ("tp", "dp", "ep", "pp", "pcp", "kv_split"):
                extra[k] = int(v)
            elif k in ("w_dense", "w_expert", "w_aux", "kv", "state", "lm_head"):
                extra[k] = float(v)
            else:
                extra[k] = v
        out.append(
            {
                "min_tokens": lo,
                "max_tokens": hi,
                "residual": residual,
                "moe_dispatch": dispatch,
                "megakernel": mk,
                **extra,
            }
        )
    return out


def validate_zero_reshard_regimes(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    parallel: Parallel,
    specs: str | Sequence[Mapping[str, Any]],
) -> tuple[BucketThresholdBand, ...]:
    """Validate that bucket overrides keep sharding and dtypes invariant."""
    parsed: Sequence[Mapping[str, Any]] = (
        parse_bucket_overrides(specs) if isinstance(specs, str) else specs
    )
    base_shard = {
        "tp": parallel.tp,
        "dp": parallel.dp,
        "ep": parallel.ep,
        "pp": parallel.pp,
        "pcp": parallel.pcp,
        "kv_split": parallel.kv_split,
        "moe": parallel.moe,
    }
    base_dt = {
        "w_dense": dtypes.w_dense,
        "w_expert": dtypes.w_expert,
        "w_aux": dtypes.w_aux,
        "kv": dtypes.kv,
        "state": dtypes.state,
        "lm_head": dtypes.lm_head,
        "act": dtypes.act,
    }
    ep_ways = parallel.ep if parallel.ep > 1 else parallel.tp
    bands: list[BucketThresholdBand] = []
    for entry in parsed:
        for sk in _SHARDING_KEYS:
            if sk in entry and entry[sk] != base_shard[sk]:
                raise ValueError(
                    f"Cross-bucket weight reshard forbidden: {sk}={entry[sk]!r} "
                    f"differs from startup {base_shard[sk]!r}"
                )
        for dk in _DTYPE_KEYS:
            if dk in entry and entry[dk] != base_dt[dk]:
                raise ValueError(
                    f"Cross-bucket dtype mutation forbidden: {dk}={entry[dk]!r} "
                    f"differs from startup {base_dt[dk]!r}"
                )
        lo = int(entry["min_tokens"])
        hi = int(entry["max_tokens"])
        res = str(entry["residual"])
        disp = str(entry["moe_dispatch"])
        mk = bool(entry.get("megakernel", False))
        if res not in ("replicated", "sp"):
            raise ValueError(f"Unsupported residual mode {res!r}")
        if disp not in ("allreduce", "a2a_push", "ag_ep", "fused_rs"):
            raise ValueError(f"Unsupported moe_dispatch mode {disp!r}")
        if res == "sp" and parallel.tp <= 1:
            raise ValueError("Sequence parallel residual='sp' requires tp > 1")
        if res == "sp" and not _sp_supported(geom, parallel):
            req = (
                "tp > 1, dp == 1, pp == 1, and moe='ep'"
                if geom.num_experts > 0
                else "tp > 1, dp == 1, and pp == 1"
            )
            raise ValueError(f"Sequence parallel residual requires {req}")
        if disp == "fused_rs" and (
            geom.num_experts > 256
            or parallel.dp > 1
            or parallel.pp > 1
            or ep_ways != parallel.tp
        ):
            raise ValueError(
                "Expert dispatch 'fused_rs' requires "
                "dp == 1, pp == 1, ep == tp, and <= 256 experts"
            )
        if disp in ("a2a_push", "ag_ep", "fused_rs") and (
            parallel.moe != "ep" or ep_ways <= 1
        ):
            raise ValueError(f"Expert dispatch {disp!r} requires moe='ep' and ep > 1")
        if mk:
            if geom.num_experts > 0 and parallel.moe == "ep" and ep_ways > parallel.tp:
                raise ValueError(
                    "megakernel=True does not support "
                    "cross-DP expert parallelism (ep > tp)"
                )
            if res != "replicated" or disp != "allreduce":
                raise ValueError(
                    "megakernel=True requires residual='replicated' "
                    "and moe_dispatch='allreduce'"
                )
            scratch = megakernel_scratchpad_bytes(geom, dtypes, parallel, hi)
            if scratch > chip.vmem_bytes:
                raise ValueError(
                    f"megakernel=True at {hi} tokens needs {scratch} B > "
                    f"on-chip scratchpad {chip.vmem_bytes} B"
                )
        bands.append(
            BucketThresholdBand(
                min_tokens=lo,
                max_tokens=hi,
                residual="sp" if res == "sp" else "replicated",
                moe_dispatch=(
                    "a2a_push"
                    if disp == "a2a_push"
                    else (
                        "ag_ep"
                        if disp == "ag_ep"
                        else ("fused_rs" if disp == "fused_rs" else "allreduce")
                    )
                ),
                megakernel=mk,
                source="explicit_override",
            )
        )
    return tuple(bands)


def _registry_candidates(
    geom: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    bucket_tokens: int,
    engine: Literal["vllm", "sglang"],
) -> list[tuple[ResidualMode, DispatchMode, bool]]:
    """Derive admissible bucket candidates from ``KNOBS`` with non-null cost hooks."""
    ep_ways = parallel.ep if parallel.ep > 1 else parallel.tp
    if engine == "sglang":
        if geom.num_experts > 0 and parallel.moe == "ep" and ep_ways > 1:
            if (
                geom.num_experts <= 256
                and parallel.dp <= 1
                and parallel.pp <= 1
                and ep_ways == parallel.tp
            ):
                return [("replicated", "fused_rs", False)]
            return [
                (
                    "replicated",
                    "ag_ep" if ep_ways > parallel.tp else "allreduce",
                    False,
                )
            ]
        return [("replicated", "allreduce", False)]

    base_dispatch: DispatchMode = (
        "ag_ep"
        if (geom.num_experts > 0 and parallel.moe == "ep" and ep_ways > parallel.tp)
        else "allreduce"
    )
    base_par = replace(parallel, residual="replicated", moe_dispatch=base_dispatch)
    active_knobs = [
        k
        for k in KNOBS
        if k.cost is not None and k.supports(geom, base_par, bucket_tokens, dtypes)
    ]

    residuals: list[ResidualMode] = ["replicated"]
    dispatches: list[DispatchMode] = [base_dispatch]
    for knob in active_knobs:
        assert knob.cost is not None
        mutated = knob.cost(base_par)
        if type(mutated) is Parallel:
            if mutated.residual not in residuals:
                residuals.append(mutated.residual)
            if mutated.moe_dispatch not in dispatches:
                dispatches.append(mutated.moe_dispatch)

    return [(r, d, False) for r in residuals for d in dispatches]


def _score_bucket_candidate(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    topology: SliceTopology | None,
    parallel: Parallel,
    bucket_tokens: int,
    scenario: TrafficScenario,
    spec_window: int,
    acceptance_length: float | None,
    residual: ResidualMode,
    moe_dispatch: DispatchMode,
    megakernel: bool,
    source: RegimeSource,
) -> BucketRegimeRow:
    shape = _bucket_step_shape(
        scenario,
        spec_window,
        bucket_tokens,
        acceptance_length=acceptance_length,
    )
    cand_par = replace(parallel, residual=residual, moe_dispatch=moe_dispatch)
    hop_lat = 0.0 if megakernel else chip.contraction_floor_s
    sf = step_floor(
        geom,
        chip,
        cand_par,
        shape,
        dtypes,
        collective_latency_s=hop_lat,
        topology=topology,
    )
    scratch = megakernel_scratchpad_bytes(geom, dtypes, parallel, bucket_tokens)
    return BucketRegimeRow(
        bucket_tokens=bucket_tokens,
        residual=residual,
        moe_dispatch=moe_dispatch,
        megakernel=megakernel,
        serial_floor_ms=sf.serial_floor_s * 1e3,
        hard_floor_ms=sf.hard_floor_s * 1e3,
        hbm_ms=sf.hbm_s * 1e3,
        mm_ms=sf.mm_s * 1e3,
        chain_ms=sf.chain_s * 1e3,
        scratchpad_bytes=scratch,
        binding=sf.binding,
        source=source,
    )


def _regime_order(
    key: tuple[ResidualMode, DispatchMode, bool],
) -> tuple[int, int, int]:
    res, disp, mk = key
    return (
        0 if mk else 1,
        1 if res == "sp" else 0,
        1 if disp in ("a2a_push", "fused_rs") else 0,
    )


def _monotone_le(
    left: tuple[ResidualMode, DispatchMode, bool],
    right: tuple[ResidualMode, DispatchMode, bool],
) -> bool:
    lo = _regime_order(left)
    ro = _regime_order(right)
    return lo[0] <= ro[0] and lo[1] <= ro[1] and lo[2] <= ro[2]


def _compact_bands(
    chosen_rows: list[BucketRegimeRow],
    score_lookup: Mapping[
        tuple[int, ResidualMode, DispatchMode, bool], BucketRegimeRow
    ],
    *,
    max_bands: int = 3,
) -> tuple[list[BucketRegimeRow], tuple[BucketThresholdBand, ...]]:
    """Compact per-bucket winners into at most `max_bands` monotone threshold bands."""
    if not chosen_rows:
        return [], ()

    has_override = any(r.source == "explicit_override" for r in chosen_rows)
    if not has_override:
        n = len(chosen_rows)
        per_bucket_keys: list[list[tuple[ResidualMode, DispatchMode, bool]]] = []
        for r in chosen_rows:
            ks = sorted(
                {
                    (res, disp, mk)
                    for (tok, res, disp, mk) in score_lookup
                    if tok == r.bucket_tokens
                },
                key=_regime_order,
            )
            if not ks:
                ks = [(r.residual, r.moe_dispatch, r.megakernel)]
            per_bucket_keys.append(ks)

        dp: dict[
            tuple[tuple[ResidualMode, DispatchMode, bool], int],
            tuple[float, list[BucketRegimeRow]],
        ] = {}
        for k0 in per_bucket_keys[0]:
            row0 = score_lookup.get((chosen_rows[0].bucket_tokens, *k0), chosen_rows[0])
            dp[(k0, 1)] = (row0.serial_floor_ms, [row0])

        for i in range(1, n):
            tok = chosen_rows[i].bucket_tokens
            nxt: dict[
                tuple[tuple[ResidualMode, DispatchMode, bool], int],
                tuple[float, list[BucketRegimeRow]],
            ] = {}
            for (prev_k, b_used), (cost_so_far, path) in dp.items():
                for cur_k in per_bucket_keys[i]:
                    if not _monotone_le(prev_k, cur_k):
                        continue
                    nb = b_used if cur_k == prev_k else b_used + 1
                    if nb > max_bands:
                        continue
                    row_i = score_lookup.get((tok, *cur_k), chosen_rows[i])
                    cand_cost = cost_so_far + row_i.serial_floor_ms
                    st = (cur_k, nb)
                    if st not in nxt or cand_cost < nxt[st][0]:
                        nxt[st] = (cand_cost, [*path, row_i])
            if nxt:
                dp = nxt

        if dp:
            _, chosen_rows = min(dp.values(), key=lambda item: item[0])

    runs: list[list[BucketRegimeRow]] = []
    for r in chosen_rows:
        if (
            runs
            and runs[-1][0].residual == r.residual
            and runs[-1][0].moe_dispatch == r.moe_dispatch
            and runs[-1][0].megakernel == r.megakernel
            and runs[-1][0].source == r.source
        ):
            runs[-1].append(r)
        else:
            runs.append([r])

    while len(runs) > max_bands:
        best_cost = float("inf")
        best_idx = -1
        best_dir = 0
        for idx in range(len(runs) - 1):
            left, right = runs[idx], runs[idx + 1]
            if (
                left[0].source == "explicit_override"
                or right[0].source == "explicit_override"
            ):
                continue
            l_key = (left[0].residual, left[0].moe_dispatch, left[0].megakernel)
            if all((b.bucket_tokens, *l_key) in score_lookup for b in right):
                cost_l = sum(
                    score_lookup[(b.bucket_tokens, *l_key)].serial_floor_ms
                    - b.serial_floor_ms
                    for b in right
                )
                if cost_l < best_cost:
                    best_cost, best_idx, best_dir = cost_l, idx, 0
            r_key = (right[0].residual, right[0].moe_dispatch, right[0].megakernel)
            if all((b.bucket_tokens, *r_key) in score_lookup for b in left):
                cost_r = sum(
                    score_lookup[(b.bucket_tokens, *r_key)].serial_floor_ms
                    - b.serial_floor_ms
                    for b in left
                )
                if cost_r < best_cost:
                    best_cost, best_idx, best_dir = cost_r, idx, 1
        if best_idx < 0:
            break
        left, right = runs[best_idx], runs[best_idx + 1]
        if best_dir == 0:
            l_key = (left[0].residual, left[0].moe_dispatch, left[0].megakernel)
            merged = left + [score_lookup[(b.bucket_tokens, *l_key)] for b in right]
        else:
            r_key = (right[0].residual, right[0].moe_dispatch, right[0].megakernel)
            merged = [score_lookup[(b.bucket_tokens, *r_key)] for b in left] + right
        runs[best_idx : best_idx + 2] = [merged]

    flat = [r for grp in runs for r in grp]
    bands = tuple(
        BucketThresholdBand(
            min_tokens=grp[0].bucket_tokens,
            max_tokens=grp[-1].bucket_tokens,
            residual=grp[0].residual,
            moe_dispatch=grp[0].moe_dispatch,
            megakernel=grp[0].megakernel,
            source=grp[0].source,
        )
        for grp in runs
    )
    return flat, bands


def _resolve_scenario(
    scenario: TrafficScenario | str | None,
    context: int,
) -> TrafficScenario:
    if type(scenario) is TrafficScenario:
        return scenario
    if type(scenario) is str:
        if scenario not in SCENARIO_PRESETS:
            raise ValueError(f"Unknown scenario preset {scenario!r}")
        return SCENARIO_PRESETS[scenario]
    # Legacy callers pass context=8192 without scenario; default to a decode-led
    # chat profile at the requested context so small buckets stay decode-shaped.
    return TrafficScenario(
        name="default",
        isl=max(1, int(context)),
        osl=max(1024, int(context)),
        prefix_hit_rate=0.9,
        accept_len=1.0,
    )


def recommend_bucket_regimes(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    topology: SliceTopology | None = None,
    parallel: Parallel | None = None,
    buckets: Sequence[int] | None = None,
    context: int = 8192,
    overrides: str | Sequence[Mapping[str, Any]] | None = None,
    engine: Literal["vllm", "sglang"] = "vllm",
    *,
    scenario: TrafficScenario | str | None = None,
    spec_window: int = 0,
    acceptance_length: float | None = None,
    max_num_batched_tokens: int | None = None,
) -> BucketRegimePlan:
    """Select per-bucket regimes with explicit override and auto floor priority."""
    if parallel is None:
        parallel = Parallel(tp=8)
    scen = _resolve_scenario(scenario, context)
    default_b = DEFAULT_SGLANG_BUCKETS if engine == "sglang" else DEFAULT_VLLM_BUCKETS
    raw_buckets = buckets if buckets is not None else default_b
    bucket_list = sorted(
        int(x)
        for x in raw_buckets
        if max_num_batched_tokens is None or int(x) <= max_num_batched_tokens
    )
    override_bands = (
        validate_zero_reshard_regimes(geom, dtypes, chip, parallel, overrides)
        if overrides is not None
        else ()
    )

    score_lookup: dict[
        tuple[int, ResidualMode, DispatchMode, bool], BucketRegimeRow
    ] = {}
    chosen_rows: list[BucketRegimeRow] = []

    for n_tok in bucket_list:
        matched_ov: BucketThresholdBand | None = None
        for ob in override_bands:
            if ob.min_tokens <= n_tok <= ob.max_tokens:
                matched_ov = ob
                break

        cands = _registry_candidates(geom, dtypes, parallel, n_tok, engine)
        for res, disp, mk in cands:
            score_lookup[(n_tok, res, disp, mk)] = _score_bucket_candidate(
                geom,
                dtypes,
                chip,
                topology,
                parallel,
                n_tok,
                scen,
                spec_window,
                acceptance_length,
                res,
                disp,
                mk,
                "roofline_auto",
            )

        if matched_ov is not None:
            ov_row = _score_bucket_candidate(
                geom,
                dtypes,
                chip,
                topology,
                parallel,
                n_tok,
                scen,
                spec_window,
                acceptance_length,
                matched_ov.residual,
                matched_ov.moe_dispatch,
                matched_ov.megakernel,
                "explicit_override",
            )
            score_lookup[
                (
                    n_tok,
                    matched_ov.residual,
                    matched_ov.moe_dispatch,
                    matched_ov.megakernel,
                )
            ] = ov_row
            chosen_rows.append(ov_row)
        else:
            best = min(
                (score_lookup[(n_tok, res, disp, mk)] for res, disp, mk in cands),
                key=lambda r: r.serial_floor_ms,
            )
            chosen_rows.append(best)

    compacted_rows, bands = _compact_bands(chosen_rows, score_lookup, max_bands=3)
    mk_rows = [r.bucket_tokens for r in compacted_rows if r.megakernel]
    sp_rows = [r.bucket_tokens for r in compacted_rows if r.residual == "sp"]
    ep_rows = [
        r.bucket_tokens
        for r in compacted_rows
        if r.moe_dispatch in ("a2a_push", "fused_rs")
    ]
    return BucketRegimePlan(
        rows=tuple(compacted_rows),
        bands=bands,
        megakernel_max_tokens=max(mk_rows) if mk_rows else 0,
        sp_min_tokens=min(sp_rows) if sp_rows else None,
        fused_ep_min_tokens=min(ep_rows) if ep_rows else None,
    )
