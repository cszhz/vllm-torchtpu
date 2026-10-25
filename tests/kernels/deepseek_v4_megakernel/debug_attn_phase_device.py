"""Device debug: CSA phase top-k / scores vs the reference."""
import os, sys
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_phase_harness as H
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_ref, attn_phase as AP
ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
served = (sys.argv[2] == "1") if len(sys.argv) > 2 else True
rng = np.random.default_rng(0)
ls = H.build(rng, ("hca", "csa"), [ctx - 3], ctx)
inp = H.make_inputs(rng, ls, "csa")
fn, ops = H.build_call(ls, "csa", 1, inp, interpret=False, served_dtypes=served, debug_topk=True)
res = jax.jit(fn)(*ops)
n_c = len(ls.arrays)
out, tk, sc, nscan = res[n_c], res[n_c + 1], res[n_c + 2], res[n_c + 3]
exp_c, exp_o, topk = H.reference(ls, "csa", 1, inp)
tk, topk = np.asarray(tk)[0, :7], np.asarray(topk)[0]
print("topk rows identical", [bool(np.array_equal(tk[t], topk[t])) for t in range(7)],
      "overlap", [len(set(tk[t]) & set(topk[t])) for t in range(7)])
seq = jnp.asarray(ls.seq_lens); pos = seq[:, None] - 7 + jnp.arange(7)[None, :]
qi, wi = attn_ref.indexer_q_reference(inp["idx_q"][:, :7], inp["idx_w"][:, :7, :64], ls.weights["idx_cos_sin"], pos)
ai = ls.role_arr["indexer_k"][0]
ref, rl = attn_ref.indexer_scores_reference(qi, wi, jnp.asarray(exp_c[ai]), jnp.asarray(ls.tables[("indexer_k", 1)]), seq)
g = np.asarray(sc)[0].reshape(-1, 8, 128).transpose(1, 0, 2).reshape(8, -1).view(np.float32)[:7]
n = ls.seq_lens[0] // 4
b = np.asarray(ref)[0, :, :n]; a = g[:, :n]
vis = np.isfinite(b)
print("scores max rel", np.abs(a[vis] - b[vis]).max() / np.abs(b[vis]).max(), "bitexact frac", np.mean(a[vis] == b[vis]))
o = np.asarray(out, np.float32)[0, :896].reshape(7, 128, 512); e = np.asarray(exp_o)[0]
print("per-row err", (np.abs(o - e).max(axis=(1, 2)) / np.abs(e).max()).round(4))
# reference attention with the kernel's own top-k
out2 = attn_ref.csa_reference(inp["q"][:, :896].reshape(1, 7, 128, 512).astype(jnp.bfloat16), inp["kv"][:, :7],
    jnp.asarray(ls.arrays[ls.role_arr["swa"][1]]), jnp.asarray(ls.swa_table[1]), jnp.asarray(exp_c[ls.role_arr["kv"][1]]),
    jnp.asarray(exp_c[ls.role_arr["kv_rope"][0]]), jnp.asarray(ls.tables[("kv", 1)]), jnp.asarray(tk[None]), seq,
    ls.weights["attn_sink"][1], sm_scale=AP.SM_SCALE)
e2 = np.asarray(out2)[0]
print("vs ref with kernel top-k: per-row err", (np.abs(o - e2).max(axis=(1, 2)) / np.abs(e2).max()).round(4))
# Stock path on the same post-write indexer cache: rope_quant + fold + streamindex_topk (SC top-k).
from vllm_torchtpu.kernels.deepseek_v4.rope import rope_quant
from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_attn_device import stock_indexer
qx = inp["idx_q"][0, :7].reshape(7, 64, 128)
q8, qs = rope_quant(qx, pos[0].astype(jnp.int32), ls.weights["idx_cos_sin"], quant_dtype=jnp.float8_e4m3fn)
w = H.stock_indexer_weights(inp["idx_w"][0, :7, :64], qs.reshape(7, 64))
u8 = csa_cache_layout.words_to_slabs(jnp.asarray(exp_c[ai]))
stk = np.asarray(stock_indexer(q8, w, u8, seq, jnp.asarray(ls.tables[("indexer_k", 1)])))
print("kernel top-k == stock streamindex_topk:", [bool(np.array_equal(tk[t], stk[t])) for t in range(7)],
      "overlap", [len(set(tk[t]) & set(stk[t])) for t in range(7)])
