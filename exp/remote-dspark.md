# Remote Draft graduation project

The opt-in path keeps Target verification on D and Draft execution on P after
one Target/Draft KV handoff. D publishes its actual request order, generation,
round and token frontier before P schedules the corresponding work. Terminal
messages release held P ownership. Preemption and failed external KV loads fail
explicitly instead of replaying Target prefill.

## Serving scope

Select the `kimi-k3-pruned75-dspark-tp8-ep-remote-planned` benchmark profile on
both roles, setting `K3_DSPARK_PD_ROLE=prefill` or `decode`. This selects the
capacity-aware schedulers explicitly; the native scheduler remains the default.
The supported topology is synchronous TP-only 1P1D, with prefix caching off.
The planned profile defaults to five Draft candidates for the Block5 checkpoint;
set `DSPARK_NUM_SPECULATIVE_TOKENS` identically on both roles to override it.
The profile uses a 300-second transport timeout and 360-second worker RPC bound.
Set `PROXY_FRESH_CONNECTIONS=1` on the proxy to disable HTTP connection reuse.

Admission reserves future growth through the generation cap and speculative
lookahead for all live owners, including P requests held outside the running
batch. Physically deferred connector frees remain charged to the block pool.
Non-align recurrent groups reserve positional pages plus speculative scratch;
they do not need the two extra align-mode pages. CPU tests exercise the upstream
allocator through full growth, speculative rollback and delayed release.

## Retained measurements

These historical runs used base commit
`43fc3573a2200e4ac453b21c34215ce1b06a0b8b` plus isolated source overlays.
They establish evidence for those snapshots, not a fresh measurement of this
commit. No additional model experiment was run while preparing the commit.

### Warm speed probe, v27

Random input length 8, output length 128, seed 42, concurrency 2; two warmup
requests excluded and eight measured requests per configuration, all successful.

| Metric | Target-only TP8 | Remote 2 x TP8 |
| --- | ---: | ---: |
| Output tokens/s | 58.270 | 68.792 |
| Mean TPOT (ms) | 33.738 | 26.539 |
| Mean TTFT (ms) | 100.789 | 279.228 |
| Mean ITL (ms) | 33.738 | 51.164 |

Throughput was 1.1806x and TPOT improved, while TTFT and ITL increased.
Remote used twice the hardware. This small probe does not establish resource
efficiency or a general speedup. Speculative chunks also make ITL differ from
TPOT. Baseline experiment: `exp-rq4kksdmoh`; remote P: `exp-e3s64hteam`,
D: `exp-diyytmcy48`. Overlay SHA256:
`788b218d926df8e51bba9388e084e5bff30683662a59ce72ecb8e1995a54ceeb`.

### GSM8K accuracy, v32

Kimi-K3-pruned75 with Inferact/Kimi-K3-DSpark-Block5, TP8 per role, five Draft
candidates plus one Target verification position. EvalScope 1.8.1 used 64
questions, seed 42, four shots, client concurrency 32, temperature 0 and output
cap 1024. The report, 64 unique predictions and 64 completed request lifecycle
records were validated: **30/64 (46.88%)**. Twenty outputs reached the cap and
44 stopped normally. Startup took about 33 minutes; evaluation took 7m08s.

P: `exp-1c7k1w0y5l`, artifact `art-5o5zwffccc`; D: `exp-59gy4cvman`, artifact
`art-p9gaa4jufm`. Both succeeded. Official analysis `an-6coaingq6a` retained
the report, predictions, validated summary, tokenization receipt and lifecycle
summary. The source overlay SHA256 was
`0db31392dae697dfdc65d7d76249a2d1fc44f6c2a04db217b88ebb469d6e05e7`.

Observed proposal concurrency peaked at 11 despite client concurrency 32.
P had 221 cache blocks and D had 317, each 829440 bytes. The validated non-align
bound was 2 MLA pages plus three recurrent groups of 1 positional + 5 scratch
pages, or 20 pages per request, consistent with 11 admissions on P.
In the same 221-block CPU fixture, removing unnecessary recurrent reservations
changes K1 admission from 15 to 27; K5 admits 11. The old v31 runtime pool
receipt was not retained, so its historical active count cannot be attributed
precisely from runtime counters.

### Native P/D comparison, v33

The native run retained local DSpark K5 on D and used the default scheduler.
It configured the same checkpoint, seed, 64-question selection and generation
settings. The completed log printed **26/64 (40.62%)**, but the post-evaluation
paired-input check failed because it hashed EvalScope's random message UUIDs.
Both experiments therefore ended failed: P `exp-uzh6poxm5z`,
D `exp-p9taxqqyyj`. Startup took about 53 minutes; evaluation took 5m55s.

The checker correction excludes message IDs and performance metadata while
retaining prompt content, roles, question indices and metadata. Its positive
and negative local checks passed, but the actual native predictions could not
be recovered through the official failed-artifact surface. **The native score
is provisional: paired accuracy equivalence or an advantage is not validated.**
The earlier 1319-question run also timed out and is not a validated full score.

## Reproduction settings

Use EvalScope 1.8.1 with the official GSM8K local train/test files. Supply the
same files and rendered four-shot inputs to both modes, retain their hashes,
and compare actual prediction indices and non-assistant prompt content after
evaluation. Random message IDs must not participate in prompt identity.

```bash
evalscope eval --model /models/Kimi-K3-pruned75 --model-id kimi-k3-remote-dspark \
  --api-url http://127.0.0.1:8050/v1/chat/completions --api-key EMPTY \
  --model-args max_retries=0 --eval-type openai_api \
  --datasets gsm8k --dataset-hub local \
  --dataset-args '{"gsm8k":{"local_path":"/path/to/gsm8k-data","subset_list":["default"],"few_shot_num":4}}' \
  --limit 64 --eval-batch-size 32 --seed 42 --repeats 1 \
  --generation-config '{"max_tokens":1024,"temperature":0.0,"timeout":1800,"retries":1,"retry_interval":0}' \
  --work-dir /path/to/fresh-results --no-timestamp
```

Accept a result only after checking report count 64, unique prediction indices
0 through 63, no model errors, bounded output usage, completed request lifecycle
records, and matching paired inputs. The command alone does not perform those
checks. Raw datasets, model weights and temporary runtime archives are not
versioned here. Further model experiments were stopped at the user's request.
