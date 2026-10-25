"""seq_parallel's torch ops on one 8-core host, eager and torch.compile'd.

Checks that take -> all_gather round-trips a replicated tensor, and that
reduce_scatter matches this core's rows of the f32 sum of the partials. Launch:
    python3 tests/forge_prefill/torch_sp_ops_multichip.py
"""

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

WORLD = 8


def _launch():
    from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
        prepare_tpu_environment,
    )

    prepare_tpu_environment(world_size=WORLD)
    env = os.environ.copy()
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    out = Path(tempfile.mkdtemp())
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD}",
        str(Path(__file__).resolve()),
        "--worker",
        str(out),
    ]
    p = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=900)
    print(p.stdout[-6000:], p.stderr[-6000:])
    for f in sorted(out.glob("rank_*.json")):
        print(f.name, f.read_text())
    sys.exit(p.returncode)


def _measure(rank):
    import torch
    from torch_tpu._internal import compile as tpu_compile
    from torch_tpu._internal.pallas import get_pallas_mesh

    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME
    from vllm_torchtpu.models.vllm.deepseek_v4 import seq_parallel as sp

    mesh = get_pallas_mesh(axis_names=(EP_AXIS_NAME,))
    groups = (tuple(range(WORLD)),)
    ag, rs, take = sp._build_ops(mesh, groups)
    slot = torch.tensor([rank], dtype=torch.int32, device="tpu")
    rows, hidden = 2048, 7168
    torch.manual_seed(1)
    full = torch.randn(rows, 4, hidden, dtype=torch.bfloat16).to("tpu")
    partials = []
    for r in range(WORLD):
        torch.manual_seed(100 + r)
        partials.append(torch.randn(rows, hidden, dtype=torch.bfloat16))
    partial = partials[rank].to("tpu")
    n = rows // WORLD
    want_rs = sum(p.float() for p in partials)[rank * n : (rank + 1) * n]

    def fn(full, partial, slot):
        local = take(full, slot)
        back = ag(local.reshape(local.shape[0], -1), full).reshape(full.shape)
        return local, back, rs(partial)

    res = {"rank": rank}
    for name, f in (
        ("eager", fn),
        (
            "compiled",
            torch.compile(
                fn, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
            ),
        ),
    ):
        local, back, r = (t.cpu() for t in f(full, partial, slot))
        fc = full.cpu()
        res[name] = {
            "take_exact": bool(torch.equal(local, fc[rank * n : (rank + 1) * n])),
            "roundtrip_exact": bool(torch.equal(back, fc)),
            "rs_shape": list(r.shape),
            "rs_vs_f32sum_rel_l2": float((r.float() - want_rs).norm() / want_rs.norm()),
            "rs_vs_f32sum_max_abs": float((r.float() - want_rs).abs().max()),
        }
    res["symbolic"] = _symbolic(ag, rs, take, full, partial, slot)
    return res


def _symbolic(ag, rs, take, full, partial, slot):
    """vLLM's route: one Dynamo trace with the token dim symbolic, then the
    TPU backend compiles that graph for concrete inputs. Case "derived"
    allocates by cores * (tokens // cores), the shape a gather sized off its
    input carries; case "like" by the gather sized off positions."""
    import torch
    from torch_tpu._internal import compile as tpu_compile

    rows = full.shape[0]
    positions = torch.arange(rows, dtype=torch.int64).to("tpu")
    out = {}
    for case in ("like", "derived"):

        def fn(full, partial, positions, slot, case=case):
            local = take(full, slot)
            g = ag(local[:, 0].contiguous(), positions)
            n = g.shape[0] if case == "like" else local.shape[0] * WORLD
            z = torch.zeros((n, 1024), dtype=torch.int32, device=full.device)
            y = (rs(partial) + local[:, 1]) * 2.5
            return g, z, y

        captured = {}

        def capture(gm, example_inputs):
            captured["gm"], captured["inputs"] = gm, example_inputs
            return gm.forward

        try:
            torch._dynamo.reset()
            torch.compile(fn, dynamic=True, backend=capture)(
                full, partial, positions, slot
            )
            tensors = iter((full, partial, positions, slot))
            concrete = [
                rows if isinstance(e, torch.SymInt) else next(tensors)
                for e in captured["inputs"]
            ]
            exe = tpu_compile.TpuBackend()(captured["gm"], concrete)
            g, z, y = (t.cpu() for t in exe(*concrete))
            out[case] = {
                "ok": True,
                "gather_exact": bool(torch.equal(g, full.cpu()[:, 0])),
                "shapes": [list(g.shape), list(z.shape), list(y.shape)],
            }
        except Exception as exc:  # noqa: BLE001 - recorded per case
            out[case] = {"ok": False, "error": repr(exc)[:400]}
    return out


def _worker(out):
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    try:
        dist.init_process_group(backend="tpu_dist")
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed.parallel_state import init_distributed_environment

        init_distributed_environment(
            world_size=WORLD,
            rank=rank,
            local_rank=rank,
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        result = _measure(rank)
    except BaseException as exc:  # noqa: BLE001 - reported to the launcher
        traceback.print_exc()
        result = {"rank": rank, "error": repr(exc)}
    (Path(out) / f"rank_{rank}.json").write_text(json.dumps(result))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        _worker(sys.argv[2])
    else:
        _launch()
