"""Sweep geometry fields of the Qwen3.8 tail segment in CPU interpret mode to find
which one makes kernel rows disagree with the reference (needs a large-RAM host)."""
import dataclasses, sys, time, jax, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_inputs import pack_segment, random_segment_inputs
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
from vllm_torchtpu.megakernel.models.qwen38_reference import segment_reference

def run(name, R=8, B=8, T=4, L=1, key=8, **over):
    g = dataclasses.replace(qwen38.FULL, **over)
    tiles = Tiles(ebank=2 << 20, nbe=12, nbd=10, vmem_mib=8192)
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=tiles)
    spec = qwen38.segment_spec(g, R, L=L)
    inp = random_segment_inputs(jax.random.PRNGKey(key), g, ex, L, gain=0.25, router_gain=30.0)
    ref, _ = segment_reference(g, ex, inp.x, inp.hc, inp.w_l, inp.st_l, inp.attn, inp.o_proj, L=L)
    w, st, attn = pack_segment(spec, g, ex, inp)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos, hid, w, st = shard_step_inputs(mesh, ex, inp.pos, inp.x, w, st)
    a = jax.device_put(kernel.fold(attn), NamedSharding(mesh, P("tp")))
    step, _ = kernel.make_step(mesh, spec, ex)
    out = step(pos, hid, w, st, a)[1][RESIDUAL_OUT]
    res = np.asarray(out.reshape(R, ex.M, -1)[0, :B], np.float32)
    refn = np.asarray(ref, np.float32)
    d = np.abs(res - refn).max(1) / np.abs(refn).mean()
    print(f"{name}: rows>10%={int((d > 0.1).sum())}/{B}", np.round(d, 3), flush=True)

cases = [
    ("full", {}),
    ("H1280", dict(H=1280)),
    ("H512", dict(H=512)),
    ("E128", dict(E=128)),
    ("I128", dict(I=128)),
    ("NHC1", dict(NHC=1)),
    ("LR64", dict(LOWRANK=64)),
    ("H512_E128_I128", dict(H=512, E=128, I=128)),
    ("H256_E128_I128_LR64", dict(H=256, E=128, I=128, LOWRANK=64)),
]
for name, over in cases:
    try:
        run(name, **over)
    except Exception as e:
        print(name, "ERR", type(e).__name__, str(e)[:200], flush=True)
