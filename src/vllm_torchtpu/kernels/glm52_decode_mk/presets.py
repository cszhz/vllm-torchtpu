"""Stage-B lever presets for glm52_decode_mk (WHITEBOARD.v3 §7 composition).

A lever is one stage-B lane's measured best variant, expressed as Tiles fields. ``tiles_kwargs(T, levers)`` merges
the named levers for the row bucket T (16 or 32), so a ladder rung is the previous rung plus exactly one lever name,
and ``Tiles()`` with no lever is the stage-A program (e852e35b7). Every lever is exact: its output is bit-identical
to stage A (A0 TPU oracle per lane, and in the compose correctness jobs).

  L4   collectives: lse tile + A2A in 4-row chunks started from inside the attention loop (C3, c060d5d0a)
  L1   fresh-layer threshold: count unroll 8, 4-bit digits only, SWAR count, deferred send waits (U8WSD, 3234e5fc9)
  L1e  the L1 exact early exit. Kept apart from L1 because it turns an exchange race into a cross-rank hang
       (stageB/L1-threshold/RESULTS.md §5.8); the clean protocol has no such race
  L2   IndexShare layers: chip-pair q all-gather + early gather issue in AR#2 + mid pre-issue sibling forward (R3,
       abb6053ac); gearly / qmid differ per bucket
  L3   served operand form: one lax.cond per level of the expert-operand cascade (srv_a, 2f65c7fe7); no effect on
       the stacked form (stack_e=True has no cascade)
  LV   live-row collectives (stageB/lv-probe, lv-prod): ARs / pair q AG / A2A move only the live rows. Not in ORDER
       ("B"); name it explicitly ("B,LV" or "L4,L1,L2,L3,L1e,LV"); needs L4 and L2. Its floor is the next-smaller
       row bucket (`rows`: the adapter's GLM52_DECODE_MK_ROWS; default 16, 32)
"""

LV_EIGHT = "split"  # placeholder until lv-prod measures split vs round
LV_LOOP = 0

ORDER = (
    "L4",
    "L1",
    "L2",
    "L3",
    "L1e",
)  # forward-ladder order, ranked by measured lane gain (compose README)


def lever(name, T, rows=None):
    if name == "L4":
        return dict(a2a_lse="tile", a2a_rw=4)
    if name == "L1":
        return dict(thr_unroll=8, thr_switch=1 << 30, thr_swar=True, thr_dsend=True)
    if name == "L1e":
        return dict(thr_early=True)
    if name == "L2":
        return dict(
            qag="pair", gearly=(4, 8) if T <= 16 else (6, 12), qmid=0 if T <= 16 else 28
        )
    if name == "L3":
        return dict(ecas="cond")
    if name == "LV":
        below = [r for r in (rows if rows else (16, 32)) if r < T]
        return dict(lv=True, lv_floor=max(below) if below else 0, lv_eight=LV_EIGHT, lv_loop=LV_LOOP)
    raise ValueError(f"unknown lever {name!r}")


def parse(spec):
    """'L4,L1,L2' | 'A' | '' | 'B' (= every lever in ORDER) -> tuple of lever names."""
    spec = (spec or "").strip()
    if spec in ("", "A", "none"):
        return ()
    if spec in ("B", "all"):
        return ORDER
    if spec in ("B,LV", "all,LV"):
        return ORDER + ("LV",)
    names = tuple(x.strip() for x in spec.split(",") if x.strip())
    for n in names:
        lever(n, 16)
    return names


def tiles_kwargs(T, levers, rows=None):
    out = {}
    for n in levers:
        kw = lever(n, T, rows)
        clash = set(kw) & set(out)
        assert not clash, ("two levers set the same Tiles field", n, clash)
        out.update(kw)
    return out
