# Experimental PP MTP prefill

`TPU_MTP_PREFILL_ONLY=1` is a diagnostic escape hatch for validating a whole
Qwen3.5 MoE MTP forward on the final pipeline stage. It does not enable
speculative decoding. Remove this diagnostic mode once the PD producer path
owns its lifecycle and acceptance tests.

The initial target is two hosts with PP2 + PCP8, one stage per host. The
last stage loads the complete MTP head and its embedding. Its PCP group runs
one MTP forward for each scheduled prefill chunk. Intermediate chunks use
the next known prompt token; the final chunk uses the target's first sampled
token. Earlier
stages load no MTP weights. The target hidden states and shifted prompt tokens
populate the MTP attention cache; there is no draft argmax, acceptance test,
verification forward or KV transfer.

## Supported envelope

- Qwen3.5 MoE, one MTP layer, K=1; target TP1, DP1, DCP1.
- Synchronous scheduling and text prompts; no multimodal input or KV connector.
- MTP supports chunked prefill with prefix caching either on or off.
  Hybrid prefix caching requires `mamba_cache_mode=align`, the unified pool
  and chunked prefill enabled, as required by upstream vLLM.
- Requests must use `max_tokens=1`. That token seeds the final MTP input row.
- Fused EP MoE remains the recommended tested serving kernel. This new
  experimental PP MTP path has the limited hardware validation below.

## Two-host diagnostic

First reserve both hosts; stop other device users only when authorized.
Use the same checkout, environment and model path on both. Keep the existing
TPU runtime and multi-host networking settings, including the host-specific
`VLLM_HOST_IP` and `TPU_WORKER_ID`. Set these before starting Python:

```bash
export TPU_MTP_PREFILL_ONLY=1
export MOE_FUSED_EP_KERNEL_IMPL=v2
export MOE_FUSED_EP_KERNEL_MIN_TOKENS=0
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_MULTIHOST_BACKEND=mp
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
```

When testing a separate checkout with the serving virtual environment, run
from the diagnostic checkout root and also export `PYTHONPATH="$PWD/src"`.
This selects the diagnostic code instead of the serving editable install.

The serialization opt-in is needed for this offline diagnostic's trusted
worker RPC functions. Do not expose this process as a public API server.

On host 0 and host 1, run the following concurrently, changing `--node-rank`:

```bash
python examples/pp_mtp_prefill.py \
  --model /path/to/Qwen3.5-35B-A3B-FP8 \
  --nnodes 2 --node-rank 0 --pp 2 --pcp 8 \
  --master-addr HOST_0_INTERNAL_IP --master-port 29510 \
  --output-dir /path/to/snapshots/pp2
```

The script clears its own allocated cache before non-prefix-cached requests, saves
CPU snapshots on the workers' hosts, and asserts earlier stages have no
drafter. It records each final-stage PCP worker's shifted input IDs, positions,
target hidden states, MTP hidden states and MTP KV views. It does not touch
another service's caches.

For a reference on the same two-host slice, run `--nnodes 2 --pp 1 --pcp 16`
after both PP2 processes exit. Keep the checkpoint, prompt, quantization,
interleave and bucket identical. This reference changes PCP/EP group size,
so compare numerical error rather than requiring bitwise equality.

Some pinned vLLM builds omit PCP from the early `EngineArgs` multi-host
world-size check. For those builds, the reference requires fixing both
`world_size` and `world_size_within_dp` in that check to include
`prefill_context_parallel_size`. Keep this upstream fix in an isolated
test installation; it does not change the PP2 MTP implementation.

Pass `--prompt-lengths 11 129 257 511` to exercise interleave boundaries
with repeated prompt token IDs. Each length is a separate fresh request
and writes to a `tokens-N` subdirectory. To isolate MTP from backbone
differences, pass `--replay-reference /path/to/pp2/snapshots` to the
reference run. It first saves the normal result, then clears caches and
replays MTP with the saved target hidden states and boundary token,
writing a separate `tokens-N-replay` directory. The driver must have all
reference rank files available locally.

The driver also writes `first-token-logprobs.json` for each request, with
its prompt token IDs, selected target token and top candidate logprobs.
Use `--logprobs N` to select the candidate count (default 20).

Gather the final-stage `rank-*.pt` files from both hosts into one directory
per run. `examples/compare_mtp_prefill.py REFERENCE CANDIDATE --atol A --rtol R`
reconstructs request-major order across PCP ranks, requires exact token and
position alignment, checks finite outputs, and compares target/MTP hidden
states with explicitly chosen tolerances. Report the measured error as well
as the tolerances. Do not relax tolerances to hide a position/layout bug.

MTP KV views are also saved for inspection. KV comparisons must align the
allocated logical block IDs, heads and valid token slots rather than blindly
comparing unrelated physical pool rows. The current comparator explicitly
does not claim KV numerical equivalence.

A successful startup alone is not a numerical correctness result. Full PD
and MTP decode are separate follow-up work.

## Chunked prefill and prefix-cache diagnostics

Add `--chunked-prefill --chunk-size 128` to split the 511-token prompt while
keeping compile bucket 512. Add `--prefix-caching --prefix-match-unit 128`
with chunked prefill enabled, and use `--repeat 2` to exercise cold and warm requests in the
same engine. `--mutate-token-at 256` adds a prompt differing at the MTP
lookahead boundary, to check that a matching prefix alone does not allow
reuse of KV depending on a different next token. The driver records
`num_cached_tokens` and collects all executed chunks per request.

The comparator aligns absolute positions, including MRoPE positions. For a
warm request it compares only the executed suffix against the cold reference.
It does not manufacture hidden states for skipped cached rows. Do not clear
worker caches between warm requests: that would invalidate the scheduler's
cached-block metadata.

The implementation reuses vLLM's cache groups, block hashes, allocation,
eviction and copy-on-write. It marks Qwen MTP attention groups for the upstream
EAGLE last-block drop, leaving target Mamba groups unmarked. For fine-grained
hits, it also preserves the target Mamba checkpoint and attention hash at the
MTP replay boundary (one hash unit before the last reusable prompt hash
boundary). This avoids losing all cache hits when upstream only retains the
last prompt-tail checkpoint. The adapters are gated by `TPU_MTP_PREFILL_ONLY`;
ordinary PP and decode scheduling keep their existing behavior.

Partial-prefill samples are discarded: the head's shifted input uses the known
prompt token at the chunk end instead. A request must still use `max_tokens=1`,
and the runner rejects any scheduled span extending beyond the prompt.

## Initial hardware validation

The Qwen3.5-35B-A3B-FP8 two-host PP2 + PCP8 run completed target and final-stage
MTP forwards for 11, 129, 257 and 511 tokens, using fused MoE v2, FP8 KV and
compile bucket 512. Only the final stage loaded MTP. A repeated 11-token run
produced bitwise-identical target and MTP hidden states.

With identical target hidden states, shifted tokens and positions, the
PP1 + PCP16 reference reproduced all four PP2 MTP outputs bitwise. This verifies
MTP layout consistency within this implementation, not independent model
accuracy against a separate golden implementation.

The normal end-to-end PP2/PCP8 versus PP1/PCP16 comparison did **not** pass
`atol=0.02, rtol=0.02`: target hidden differences already exist before MTP.
Their maximum absolute differences were 1.1875, 4.140625, 12.8125 and 12.8125
for the four lengths. First output tokens matched, but that is insufficient
to declare end-to-end numerical correctness. The backbone discrepancy
remains unresolved. These runs used max model length 512; 8K, chunked
prefill, KV-transfer correctness and decode remain unvalidated.

A subsequent first-token logprob comparison on the same four inputs returned
identical selected-token logprobs in both layouts: -0.125, 0, 0 and 0.
Top-20 candidate overlap was 20/20, 18/20, 18/20 and 18/20; the maximum
absolute logprob differences among shared candidates were 0.3125, 0.6875,
1.125 and 0.3125. Other candidate probabilities and ranks therefore differ.
The returned zeros should not be interpreted as exact probability one:
normalization and subtraction precede the FP32 cast in `gather_logprobs`.
These diagnostic results do not establish full-distribution equivalence.

## Chunk/cache validation results

Two-host PP2 + PCP8, Qwen3.5-35B-A3B-FP8, fused MoE v2, FP8 KV,
compile bucket 512 and prefix match unit 128:

- A 511-token prompt with a 128-token chunk limit reused 256 tokens on a
  repeat request. Target and MTP hidden states for the executed suffix were
  bitwise identical to the cold request.
- With no extra chunk-size limit, repeats also reused 256 tokens and matched
  both hidden tensors bitwise. Changing token 256 caused a cache miss;
  changing token 384 still safely reused the first 256 tokens.
- An 8191-token prompt with max model length 8192 and a 512-token batch budget
  reused 7936 tokens. Its 255-token suffix matched the cold target and MTP
  hidden states bitwise.
- 95 CPU regression tests passed, including real upstream hybrid cache lookup,
  changed-lookahead rejection, copy-on-write lifecycle and cache reset.

Chunked and whole-prompt execution are **not numerically equivalent** under
these tests. For the 511-token prompt, ordinary chunked-vs-whole target hidden
relative L2 error was 0.33387. With target hidden states held fixed using
`--target-reference /path/to/whole-prompt/snapshots`, the first 128 MTP rows
matched bitwise; later chunks differed, with overall MTP relative L2 0.02335
and maximum absolute difference 3.53125. The strict `atol=rtol=0.02` comparison
failed. Inputs and positions matched exactly. The cause of this numerical
difference is not established; this is not an independent accuracy acceptance
result. The successful warm/cold comparisons verify cache restoration within
a given execution pattern, not equivalence across chunking patterns.

These are single-request diagnostic runs. Concurrent-request throughput,
MTP KV transfer, asynchronous PP and MTP decode remain outside this validation.
