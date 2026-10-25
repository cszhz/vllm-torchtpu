# SPDX-License-Identifier: Apache-2.0
"""Which roundings does a torch_tpu-compiled `shared + routed * scale` (bf16) apply?

Prints, per scale, how many elements match each numpy model:
  two:    bf16(bf16(routed * scale) + shared)   (product rounded, then sum)
  one:    bf16(f32(routed) * scale + shared)    (one rounding, f32 inside)
"""

import numpy as np
import torch
import torch_tpu  # noqa: F401
from torch_tpu._internal import compile as tpu_compile


def bf16(a):
    return (
        torch.from_numpy(np.asarray(a, np.float32)).to(torch.bfloat16).float().numpy()
    )


def main():
    rng = np.random.default_rng(0)
    rows, hidden = 8, 7168
    shared = bf16(
        rng.standard_normal((rows, hidden)) * np.exp(rng.uniform(-6, 2, (rows, hidden)))
    )
    routed = bf16(
        rng.standard_normal((rows, hidden)) * np.exp(rng.uniform(-6, 2, (rows, hidden)))
    )
    s_t = torch.from_numpy(shared).to(torch.bfloat16).to("tpu")
    backend = tpu_compile.TpuBackend()
    for scale in (2.5, 1.5, 2.0, 2.446, 1.0):

        def fn(s, r):
            r = r.clone()
            r *= scale
            return s + r

        f = torch.compile(fn, fullgraph=True, dynamic=False, backend=backend)
        r_t = torch.from_numpy(routed).to(torch.bfloat16).to("tpu")
        got = f(s_t, r_t).float().cpu().numpy()
        eager = fn(s_t, r_t).float().cpu().numpy()
        sc_b = float(bf16(scale))
        models = {
            "two_f32scale": bf16(bf16(routed * np.float32(scale)) + shared),
            "two_bf16scale": bf16(bf16(routed * np.float32(sc_b)) + shared),
            "one_f32scale": bf16(routed * np.float32(scale) + shared),
            "one_bf16scale": bf16(routed * np.float32(sc_b) + shared),
        }
        res = {k: int((v == got).sum()) for k, v in models.items()}
        res_e = {k: int((v == eager).sum()) for k, v in models.items()}
        print(
            f"scale {scale}: compiled {res} | eager {res_e} | of {got.size}", flush=True
        )


if __name__ == "__main__":
    main()
