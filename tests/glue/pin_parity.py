# SPDX-License-Identifier: Apache-2.0
"""Device parity + profile for TPU_PIN_OPERANDS_HBM (kernels/operand_pin.py).

    python tests/glue/pin_parity.py run OUT.npz [XPROF_DIR]   # one mode, env decides
    python tests/glue/pin_parity.py compare A.npz B.npz       # bitwise equality

Each site runs as a chain of calls with distinct per-layer weights, the
activation produced by a fusion just before each call, as in the decode step.
"""
import sys

import numpy as np

L = 6


def _chains():
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
        proj_and_save_state as proj,
    )
    from vllm_torchtpu.kernels.fused_moe.v2 import plan_kernel
    from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

    key = iter(jax.random.split(jax.random.PRNGKey(0), 1000))
    m = 16

    def nxt(x, y):
        s = jnp.mean(y.astype(jnp.float32), axis=1, keepdims=True)
        return (x.astype(jnp.float32) * (1.0 + 1e-3 * s)).astype(x.dtype)

    chains = {}
    for k, n in fp8_skinny.TUNED_HBM:
        ws = [jax.random.normal(next(key), (k, n)).astype(jnp.float8_e4m3fn) for _ in range(L)]
        ss = [jax.random.uniform(next(key), (n,), minval=0.5, maxval=1.5) for _ in range(L)]
        x0 = jax.random.normal(next(key), (m, k)).astype(jnp.bfloat16)

        def f(x, ws, ss):
            ys = []
            for w, s in zip(ws, ss):
                y = fp8_skinny.fp8_skinny_matmul(x, w, s, hbm_pin=True)
                ys.append(y)
                x = nxt(x, y)
            return ys

        chains[f"skinny_{k}x{n}"] = (jax.jit(f), (x0, ws, ss))

    # (state width, compress ratio, cache shape (pages, rows, ...), dtype); DSv4 TP8 shapes.
    pages = 64
    for name, sdim, ratio, cshape, cdt in [
        ("proj_csa", 2048, 4, (pages, 256, 128), jnp.int32),
        ("proj_hca", 1024, 128, (pages, 256, 128), jnp.int32),
        ("proj_idx", 512, 4, (pages, 64, 4, 256), jnp.uint8),
    ]:
        sw = sdim // 2
        ws = [(0.02 * jax.random.normal(next(key), (7168, sdim))).astype(jnp.bfloat16) for _ in range(L)]
        apes = [jax.random.normal(next(key), (ratio, sw)) for _ in range(L)]
        caches = [jnp.zeros(cshape, cdt) for _ in range(L)]
        last = cshape[-1]
        per_tok = 2 * (sw // last)  # slots one token's state spans
        rows = cshape[1]
        n_real = 7
        tok = np.arange(m)
        slot = np.where(tok < n_real, (tok * 5 % pages) * rows + (tok % (rows // per_tok)) * per_tok, -1)
        slot = jnp.asarray(slot, jnp.int32)
        pos = jnp.asarray(1000 + 37 * tok, jnp.int32)
        h0 = jax.random.normal(next(key), (m, 7168)).astype(jnp.bfloat16)

        def g(h, ws, apes, caches, ratio=ratio, slot=slot, pos=pos):
            outs = []
            for w, a, c in zip(ws, apes, caches):
                c = proj.proj_and_save_state(h, w, a, pos, slot, c, compress_ratio=ratio)
                outs.append(c)
                h = nxt(h, c.reshape(c.shape[0], -1)[:m, :128].astype(jnp.float32))
            return outs

        chains[name] = (jax.jit(g, donate_argnums=(3,)), (h0, ws, apes, caches))

    t_local, topk, R, S = 2, 6, 396, 773
    arrs = [jax.random.normal(next(key), (R, 56, 128)).astype(jnp.float8_e4m3fn) for _ in range(L)]
    scs = [jax.random.uniform(next(key), (S, 128)) for _ in range(L)]
    pos0 = jax.random.randint(next(key), (t_local, topk), 0, R)
    mir0 = jax.random.randint(next(key), (t_local, topk), 0, S * 128)
    wts = jax.random.uniform(next(key), (t_local, topk))

    def c(pos, mir, wts, arrs, scs):
        outs = []
        for a, s in zip(arrs, scs):
            o = plan_kernel.build_decode_combine(a, s, pos, mir, wts, jnp.bfloat16)
            outs.append(o)
            pos = (pos + (o[0, 0, 0] > 1e9).astype(jnp.int32) + 1) % R
        return outs

    chains["combine"] = (jax.jit(c), (pos0, mir0, wts, arrs, scs))

    from vllm_torchtpu.kernels.fused_moe.v2.router_ops import pallas_select_routed

    E = 384
    biases = [jax.random.normal(next(key), (E,)) for _ in range(L)]
    labels = [jax.random.permutation(next(key), E).astype(jnp.int32) for _ in range(L)]
    gate0 = jax.random.normal(next(key), (t_local, E))

    def sel(gate, biases, labels):
        outs = []
        for b, lab in zip(biases, labels):
            w, i, r = pallas_select_routed(
                jax.nn.sigmoid(gate), gate, topk=topk, block_rows=t_local,
                score_bias=b, labels=lab,
            )
            outs += [w, i, r.astype(jnp.int32)]
            gate = gate + 1e-3 * w.sum()
        return outs

    chains["select"] = (jax.jit(sel), (gate0, biases, labels))

    from vllm_torchtpu.kernels.mla.sparse.kernel import gather_page_ids

    seqs, pps = 4, 1000
    tables = [jax.random.randint(next(key), (seqs * pps,), 0, 1 << 20) for _ in range(L)]
    spi0 = jax.random.randint(next(key), (m, 1024), 0, pps)
    seg = jnp.asarray(np.arange(m) % seqs, jnp.int32)

    def pg(spi, tables, seg=seg):
        outs = []
        for t in tables:
            o = gather_page_ids(t, spi, seg, seqs)
            outs.append(o)
            spi = (spi + (o[:, :1] & 1)) % pps
        return outs

    chains["pages"] = (jax.jit(pg), (spi0, tables))

    ep, E, stride = 16, 384, 768
    P = t_local * topk
    pairs0 = jax.random.randint(next(key), (ep, P), 0, E + 1)
    bits0 = jax.lax.bitcast_convert_type(
        jax.random.uniform(next(key), (ep * t_local,), minval=0.01, maxval=2.0), jnp.int32
    )

    def plan(pairs, bits):
        outs = []
        for r in range(L):
            p = plan_kernel.build_decode_plan(
                pairs, bits, jnp.int32(r % ep), e_total=E, ep=ep, t_local=t_local,
                topk=topk, block=4, tile_m=128, shard_stride=stride, row_stride=3,
            )
            leaves = jax.tree_util.tree_leaves(p)
            outs += [jax.lax.bitcast_convert_type(x, jnp.int32) if x.dtype == jnp.float32 else x for x in leaves]
            pairs = (pairs + (leaves[0][0, 0] & 1) + 7) % (E + 1)
        return outs

    chains["plan"] = (jax.jit(plan), (pairs0, bits0))
    return chains


def run(out, xprof=None):
    import jax

    chains = _chains()
    res = {}
    def call(name):
        f, args = chains[name]
        o = jax.block_until_ready(f(*args))
        if name.startswith("proj"):
            chains[name] = (f, (*args[:3], o))
        return o

    for name in chains:
        o = call(name)
        leaves = jax.tree_util.tree_leaves(o)
        res[name] = np.concatenate([np.ascontiguousarray(np.asarray(x)).reshape(-1).view(np.uint8) for x in leaves])
        print(name, "ok", res[name].size, flush=True)
    if xprof:
        for name in chains:
            for _ in range(3):
                call(name)
        with jax.profiler.trace(xprof):
            for name in chains:
                for _ in range(5):
                    call(name)
    np.savez(out, **res)


def compare(a, b):
    A, B = np.load(a), np.load(b)
    bad = 0
    for k in sorted(A.files):
        same = A[k].shape == B[k].shape and np.array_equal(A[k], B[k])
        diff = 0 if same else int(np.sum(A[k] != B[k])) if A[k].shape == B[k].shape else -1
        print(f"{k:24s} {'BITEXACT' if same else f'DIFF bytes={diff}'} ({A[k].size} bytes)")
        bad += not same
    print("PARITY", "PASS" if not bad else f"FAIL ({bad})")
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
