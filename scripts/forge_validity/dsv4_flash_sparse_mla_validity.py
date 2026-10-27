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

r"""Validity adapter for DeepSeek-V4 CSA `sparse_ragged_paged_attention` at
DeepSeek-V4-Flash-0731 DP8/EP8 serving shapes.

Run it through perfkit, never directly:

    perfkit validity run \
        --adapter scripts/forge_validity/dsv4_flash_sparse_mla_validity.py \
        --arm <checkout>

`perfkit/validity_arm.py` defines the hooks and the two checks (`aot_all_shapes`,
`pipeline_trip_count`). Nothing here uses a TPU: compiles are ahead of time for
the tpu7x topology on the CPU, and the run-time check is Pallas interpret mode on
the CPU.

What it checks
--------------
`SHAPES` is every token count the server sends a CSA layer under
`deepseek-v4-flash-dp8-ep`: the runner pads each engine's batch to one of 7 token
counts (16 to 1024, `--max-num-batched-tokens=1024`). Each shape is called with
the chunking `_attention_csa` picks for it (128-token gather chunks in 16-token
kernel batches, 32 in 8 at 128 tokens or fewer), the model's 64 heads of 512, the
indexer's top-512, and the cache the server allocates (14579 pages of 256
compressed tokens, 256 sequences of up to 9 pages).

`aot_all_shapes` compiles the whole op -- page-id gather, SparseCore KV gather and
attention kernel -- for tpu7x at each shape.

The attention kernel double-buffers its own DMAs and sizes its grid at run time,
from the valid tokens in each chunk: anything from 0 to `chunk / batch` steps.
Warmup's dummy batches and the decode tail reach the small end, which the
layer-run score never does. `trip_counts` runs the op in Pallas interpret mode
over runtime inputs that reach every grid size of each chunk (one prefill
sequence of 0..T tokens, and 1..256 decode sequences of one token), with each
DMA replaced by one that moves nothing and counts its start and its wait per
kernel launch and semaphore. An input fails when a launch starts a copy it never
waits for (Mosaic's exit check halts the core on it), waits for one it never
started (the core hangs), or raises. Interpret mode cannot run the SparseCore
gather, so the gather is replaced by `jnp` indexing that writes the same layout
(`csa_gather.csa_gather`'s docstring); it is compiled, not run. DMA sizes are
not compared, so a wait whose byte count differs from its start passes here.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import os
import sys
import types
from pathlib import Path

KERNEL_DIR = "src/vllm_torchtpu/kernels/deepseek_v4/core_attention"
TOPOLOGY = "tpu7x:2x2x1"

HEADS = 64
HEAD_DIM = 512
TOPK = 512
PAGE_SIZE = 256
TOTAL_PAGES = 14579
MAX_SEQS = 256
PAGES_PER_SEQ = 9  # 9216 tokens / 4 per compressed token / 256 per page
ROW_WORDS = 128
ROPE_WORDS = 32
TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)
#: The interpret runs use a small cache; its size sizes no grid.
INTERPRET_PAGES = 16


def _chunking(tokens):
    """`(gather_and_attention_chunk_size, attention_kernel_batch_size)` as
    `_attention_csa` picks them."""
    return (32, 8) if tokens <= 128 else (128, 16)


SHAPES = {
    f"csa_t{t}": {"tokens": t, "chunk": _chunking(t)[0], "batch": _chunking(t)[1]}
    for t in TOKEN_PADDINGS
}


def _load_kernel(root: Path):
    """Import sparse_mla from the arm root without the package's own `__init__`,
    which imports torch and the serving stack. `env_override` is loaded for the
    server's LIBTPU_INIT_ARGS and import-time Pallas patches."""
    pkg = root / "src" / "vllm_torchtpu"

    def package(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
        return mod

    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    top = package("vllm_torchtpu", pkg)
    for name in ("envs", "logger", "patch_registry"):
        setattr(top, name, module(f"vllm_torchtpu.{name}", pkg / f"{name}.py"))
    module("vllm_torchtpu.env_override", pkg / "env_override.py")
    package("vllm_torchtpu.kernels", pkg / "kernels")
    dsv4 = package("vllm_torchtpu.kernels.deepseek_v4", pkg / "kernels" / "deepseek_v4")
    dsv4.csa_cache_layout = module(
        "vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout",
        pkg / "kernels" / "deepseek_v4" / "csa_cache_layout.py",
    )
    core = package(
        "vllm_torchtpu.kernels.deepseek_v4.core_attention", root / KERNEL_DIR
    )
    core.csa_gather = module(
        "vllm_torchtpu.kernels.deepseek_v4.core_attention.csa_gather",
        root / KERNEL_DIR / "csa_gather.py",
    )
    return module(
        "vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla",
        root / KERNEL_DIR / "sparse_mla.py",
    )


def setup(root):
    # CPU only: jax must never open the host's TPU, and the compile-only
    # topology does not need the libtpu lock.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-4")
    sys.dont_write_bytecode = True
    kernel = _load_kernel(Path(root).resolve())

    import jax
    import jax.numpy as jnp
    import numpy as np

    return types.SimpleNamespace(kernel=kernel, jax=jax, jnp=jnp, np=np, device=None)


def origin(handle) -> str:
    return handle.kernel.__file__


def _entry(handle, params):
    """The op called as `_attention_csa` calls it, at one shape."""
    fn = handle.kernel.sparse_ragged_paged_attention
    return functools.partial(
        fn,
        sm_scale=HEAD_DIM**-0.5,
        gather_and_attention_chunk_size=params["chunk"],
        attention_kernel_batch_size=params["batch"],
    )


def _arg_specs(handle, tokens, pages=TOTAL_PAGES):
    jnp = handle.jnp
    return [
        ((tokens, HEADS, HEAD_DIM), jnp.bfloat16),  # q
        ((pages, PAGE_SIZE, ROW_WORDS), jnp.int32),  # cache_kv_nope
        ((pages, PAGE_SIZE // 4, ROW_WORDS), jnp.int32),  # cache_kv_rope
        ((tokens, TOPK), jnp.int32),  # topk_indices
        ((MAX_SEQS * PAGES_PER_SEQ,), jnp.int32),  # page_indices
        ((MAX_SEQS + 1,), jnp.int32),  # cu_q_lens
        ((3,), jnp.int32),  # distribution
        ((HEADS,), jnp.float32),  # attention_sinks
        ((tokens, HEADS, HEAD_DIM), jnp.bfloat16),  # swa_accumution
        ((tokens, HEADS), jnp.float32),  # swa_l
        ((tokens, HEADS), jnp.float32),
    ]  # swa_m


def _tpu7x_mesh():
    """Trace as for one tpu7x core, so the kernel's chip queries answer for it."""
    from jax._src import mesh as mesh_lib

    return mesh_lib.use_abstract_mesh(
        mesh_lib.AbstractMesh(
            (),
            (),
            abstract_device=mesh_lib.AbstractDevice(
                device_kind="TPU7x", num_cores=1, platform="tpu"
            ),
        )
    )


def compile(handle, name, params):
    jax = handle.jax
    if handle.device is None:
        from jax.experimental import topologies

        handle.device = topologies.get_topology_desc(TOPOLOGY, platform="tpu").devices[
            0
        ]
    sharding = jax.sharding.SingleDeviceSharding(handle.device)
    args = [
        jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
        for shape, dtype in _arg_specs(handle, params["tokens"])
    ]
    with _tpu7x_mesh():
        lowered = (
            jax.jit(_entry(handle, params))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    mem = lowered.compile().memory_analysis()
    text = lowered.as_text()
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "kernels": text.count("tpu_custom_call"),
        "attention": "SparseMLA" in text,
    }


def limit(name, result):
    if not result["attention"]:
        return "the compiled program launches no SparseMLA kernel: nothing was checked"
    return None


class _Override:
    """A module with some attributes replaced."""

    def __init__(self, real, **replaced):
        self._real = real
        self.__dict__.update(replaced)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _reference_gather(
    jnp, nope_cache, rope_cache, indices, num_valid_indices=None, *, top_k
):
    """`csa_gather.csa_gather` as `jnp` indexing: the same two outputs, rows past
    `num_valid_indices` left zero (the kernel leaves them unwritten)."""
    nope = nope_cache.reshape(-1, ROW_WORDS)[indices]
    rope = rope_cache.reshape(-1, ROPE_WORDS)[indices]  # [N, 32] words
    if num_valid_indices is not None:
        keep = (jnp.arange(indices.shape[0]) < num_valid_indices)[:, None]
        nope = jnp.where(keep, nope, 0)
        rope = jnp.where(keep, rope, 0)
    # Entry i of each top_k period pairs with entry i + top_k // 2, 64 bf16 each,
    # which is 32 words each: [N // 2, 64] words = [N // 4, 128].
    rope = rope.reshape(-1, 2, top_k // 2, ROPE_WORDS).transpose(0, 2, 1, 3)
    return nope, rope.reshape(-1, ROW_WORDS)


class _Ledger:
    """Every DMA start and wait the kernels issue, per kernel launch and semaphore.

    Interpret mode cannot order callbacks inside a kernel, so the ledger counts
    rather than replays: per launch and semaphore, more waits than starts would
    hang the core, and more starts than waits is a copy no step waited for, which
    Mosaic's exit check halts on. A launch is numbered when it is traced."""

    def __init__(self):
        self.tracing = 0
        self.counts = {}

    def event(self, kind, key, *index):
        key = (*key, *(int(i) for i in index))
        starts, waits = self.counts.get(key, (0, 0))
        self.counts[key] = (starts + (kind == "start"), waits + (kind == "wait"))

    def faults(self):
        out = []
        for (launch, sem, *index), (starts, waits) in sorted(self.counts.items()):
            if starts != waits:
                out.append(
                    f"launch {launch} semaphore {sem}{index}: {starts} copies started, "
                    f"{waits} waited"
                )
        return out


class _CountedCopy:
    """A DMA that moves nothing and logs its start and wait on its semaphore.

    The semaphore is named by the scratch array it belongs to and its run-time
    index into it."""

    def __init__(self, jax, jnp, ledger, names, sem):
        self._jax, self._ledger = jax, ledger
        self._key = (
            ledger.tracing,
            names.setdefault(id(getattr(sem, "ref", sem)), len(names)),
        )
        self._index = [
            jnp.asarray(getattr(i, "start", i), jnp.int32)
            for t in getattr(sem, "transforms", ())
            for i in getattr(t, "indices", ())
        ]

    def _log(self, kind):
        self._jax.debug.callback(
            functools.partial(self._ledger.event, kind, self._key), *self._index
        )

    def start(self, *args, **kwargs):
        self._log("start")

    def wait(self, *args, **kwargs):
        self._log("wait")


@contextlib.contextmanager
def _interpret(handle):
    """Run the op's Pallas kernels in interpret mode with every DMA replaced by a
    `_CountedCopy`, and yield the `_Ledger` they log to.

    The copies move nothing because interpret mode cannot read the bitcast HBM
    refs the attention kernel copies from (jax 0.11.2), so the arithmetic runs on
    zeros and only the DMA schedule is checked."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    jax, jnp, k = handle.jax, handle.jnp, handle.kernel
    ledger, names = _Ledger(), {}
    params = pltpu.InterpretParams(
        uninitialized_memory="zero", out_of_bounds_reads="raise"
    )

    def pallas_call(*args, **kwargs):
        launch = pl.pallas_call(*args, interpret=params, **kwargs)

        def numbered(*operands):
            ledger.tracing += 1
            return launch(*operands)

        return numbered

    def make_async_copy(src, dst, sem, *args, **kwargs):
        return _CountedCopy(jax, jnp, ledger, names, sem)

    def async_copy(src, dst, sem, *args, **kwargs):
        copy = _CountedCopy(jax, jnp, ledger, names, sem)
        copy.start()
        return copy

    real = (k.pl, k.pltpu, k.csa_gather)
    k.pl = _Override(pl, pallas_call=pallas_call)
    k.pltpu = _Override(pltpu, make_async_copy=make_async_copy, async_copy=async_copy)
    k.csa_gather = _Override(
        real[2], csa_gather=functools.partial(_reference_gather, jnp)
    )
    try:
        yield ledger
    finally:
        k.pl, k.pltpu, k.csa_gather = real


def _runtime_inputs(tokens, chunk, batch):
    """`(description, per-sequence query lengths, distribution kind, padded topk)`.

    The prefill lengths reach every grid size of the first chunk and the edges of
    the later ones; the decode counts cover a single sequence, one kernel batch
    and the whole batch."""
    edges = {
        0,
        1,
        batch - 1,
        batch,
        batch + 1,
        chunk - 1,
        chunk,
        chunk + 1,
        tokens - 1,
        tokens,
    }
    edges |= set(range(0, min(chunk, tokens) + 1, batch))
    lengths = sorted(n for n in edges if 0 <= n <= tokens)
    for n in lengths:
        yield (
            f"one prefill sequence of {n} tokens",
            [n] if n else [],
            "prefill",
            n < TOPK,
        )
    for n in sorted({1, batch, min(tokens, MAX_SEQS)}):
        yield f"{n} decode sequences of one token", [1] * n, "decode", False


def _inputs_for(handle, tokens, lengths, kind, padded):
    """Arguments for one runtime input, with a small cache."""
    jnp, np = handle.jnp, handle.np
    rng = np.random.default_rng(len(lengths) * 7919 + tokens)
    pages = INTERPRET_PAGES
    q = jnp.asarray(rng.standard_normal((tokens, HEADS, HEAD_DIM)), jnp.bfloat16)
    # The copies move nothing (`_interpret`), so the cache contents are never read.
    nope = jnp.zeros((pages, PAGE_SIZE, ROW_WORDS), jnp.int32)
    rope = jnp.zeros((pages, PAGE_SIZE // 4, ROW_WORDS), jnp.int32)
    entries = PAGES_PER_SEQ * PAGE_SIZE
    topk = rng.integers(0, min(entries, pages * PAGE_SIZE), (tokens, TOPK)).astype(
        np.int32
    )
    if padded:
        topk[:, TOPK // 3 :] = -1
    page_indices = (np.arange(MAX_SEQS * PAGES_PER_SEQ) % pages).astype(np.int32)
    cu = np.zeros(MAX_SEQS + 1, np.int32)
    cu[1 : len(lengths) + 1] = np.cumsum(lengths)
    cu[len(lengths) + 1 :] = cu[len(lengths)]
    n = len(lengths)
    dist = np.array([n, n, n] if kind == "decode" else [0, n, n], np.int32)
    sinks = jnp.zeros((HEADS,), jnp.float32)
    swa = jnp.asarray(rng.standard_normal((tokens, HEADS, HEAD_DIM)), jnp.bfloat16)
    swa_l = jnp.ones((tokens, HEADS), jnp.float32)
    swa_m = jnp.zeros((tokens, HEADS), jnp.float32)
    return (
        q,
        nope,
        rope,
        jnp.asarray(topk),
        jnp.asarray(page_indices),
        jnp.asarray(cu),
        jnp.asarray(dist),
        sinks,
        swa,
        swa_l,
        swa_m,
    )


def _grid_steps(tokens, chunk, batch, valid):
    """Grid steps the attention kernel takes over all chunks for `valid` tokens."""
    steps = 0
    for start in range(0, tokens, chunk):
        end = min(start + chunk, tokens)
        b = max(d for d in range(1, batch + 1) if (end - start) % d == 0)
        steps += -(-max(0, min(valid, end) - start) // b)
    return steps


def trip_counts(handle, name, params):
    jax = handle.jax
    tokens, chunk, batch = params["tokens"], params["chunk"], params["batch"]
    records = []
    with _interpret(handle) as ledger:
        run = jax.jit(_entry(handle, params))
        for description, lengths, kind, padded in _runtime_inputs(tokens, chunk, batch):
            ledger.counts.clear()
            jax.block_until_ready(
                run(*_inputs_for(handle, tokens, lengths, kind, padded))
            )
            jax.effects_barrier()
            if sum(lengths) and not ledger.counts:
                raise RuntimeError(
                    f"{description}: the kernels started no DMA this adapter "
                    f"can see, so nothing was checked"
                )
            if ledger.faults():
                raise RuntimeError(f"{description}: {'; '.join(ledger.faults()[:2])}")
            records.append(
                {
                    "pipe": "attention kernel double buffers",
                    "buffer_count": 2,
                    "trip_count": _grid_steps(tokens, chunk, batch, sum(lengths)),
                    "lookahead": False,
                    "input": description,
                }
            )
    return records
