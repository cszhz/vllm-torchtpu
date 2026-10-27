# DP4xTP2 1P1D c152: kernel and Stage-3 flags (cell 5)

Commit 7b9bbd5e2 reproduces the `klb-tree-v4-aff` tree that j-c85b4fe5 mounted
(dp4tp2-klb-aff @ 723adab93, which is not on origin). The commit after it adds what
j-ea603d0b ran on top of that tree. The new kernel and connector paths are selected
by environment flags that default off. `git diff 7b9bbd5e2 acef37df9` shows the
change.

## Results

Both jobs ran c152 with the same image and otherwise the same variables.

| | j-c85b4fe5 | j-ea603d0b |
| --- | --- | --- |
| x, p90 interactivity (tok/s) | 96.34 | 119.73 |
| y, total tok/s per chip | 70976 | 73874 |
| TTFT p50 (s) | 2.80 | 2.26 |
| TPOT p50 (ms) | 9.86 | 7.84 |

Three runs with the j-c85b4fe5 flags measured x 95.35 to 96.34 and y 68.2k to
71.0k. The x gain is well outside that spread and matches the decode step at 16
requests in xprof, 34.3 ms without the decode flags and 28.4 ms with them. The y
gain is one run near the edge of the spread and needs a repeat.

## Running it

1. Check out cloud-devkit-recipes `benliangw/agentx-dp4tp2-opt5-flags` in the
   checkout cdk reads. It is one commit on master that adds the per-leg
   passthroughs below to `qwen3.5-397b-pd-agentx.jsonnet`; unset, the template
   renders the same as master.
2. Check out this branch and copy `expert_placement/` from `klb-tree-v4-aff` into
   it. `DECODE_TPU_EXPERT_PLACEMENT` points there.
3. Start from the j-c85b4fe5 command (`cdk job desc j-c85b4fe5-5824-4306-8e93`).
   Point `--map-dir` at this checkout, set a new `BENCH_NAME` and
   `COMPILE_CACHE_TAG`, change `PREFILL_TPU_RPA_NATIVE_FP8_PREFILL=0` to `pvs`, and
   append:

```text
DECODE_TPU_RPA_DECODE_CAUSAL_MASK=1 DECODE_TPU_RPA_DECODE_EXP2=1
DECODE_TPU_RPA_NATIVE_FP8=pvs DECODE_TPU_RPA_DECODE_SKEW=1
DECODE_TPU_FUSED_SE_GATE_FRESH_BUFFER=1 DECODE_TPU_SAMPLER_TOPK_CANDIDATES=40
DECODE_TPU_SAMPLER_TOPK_PALLAS=1 DECODE_TPU_SAMPLER_TOPK_MAX_K=20
DECODE_TPU_SMALL_ALL_REDUCE_TP=1 DECODE_TPU_SMALL_ALL_REDUCE_TP_FORM=oneshot
DECODE_TPU_GDN_KT_XPOSE=1
DECODE_TPU_MOE_LOWTOK_FUSED=1 DECODE_TPU_MOE_LOWTOK_FUSED_DISPATCH_FORM=hier
DECODE_TPU_MOE_LOWTOK_FUSED_COMBINE_FORM=hier
DECODE_TPU_MOE_LOWTOK_FUSED_FP8_DISPATCH=1 DECODE_TPU_MOE_LOWTOK_GATHER=1
DECODE_TPU_RAIDEN_STAGE3_LATE_POLL=1
PREFILL_TPU_GDN_PREFILL_SHARED_KKT=1 PREFILL_TPU_GDN_PREFILL_LAZY_CONV_STATE=1
PREFILL_TPU_GDN_PREFILL_DOT_PRECISION=bf16 PREFILL_TPU_GDN_PREFILL_HEAD_GROUP=32
PREFILL_TPU_MOE_NONEP_PERMUTE=two_sorts
PREFILL_TPU_RAIDEN_STAGE3_ASYNC_REGISTER=1 PREFILL_TPU_PP_EXTRA_STEPS_IN_FLIGHT=1
```

The master template changed after 2026-10-23, so j-c85b4fe5 is not a like-for-like
baseline for a run on these branches. Run the same command without the appended
variables (and with `PREFILL_TPU_RPA_NATIVE_FP8_PREFILL=0`) next to it.

## Notes

- j-ea603d0b also ran a prefill xprof (ranks 0 and 7, three 15 s captures) and a
  64-question GSM8K known-answer probe from a separate recipe; neither is in the
  command above. The probe scored 0.828. A c8 run with the decode flags and no
  prefill flags scored 0.812, and healthy runs ranged from 0.766 to 0.906. No run
  with every flag off took the probe.
- Do not set `TPU_GDN_PREFILL_TINV_MODE=doubling`. It produces NaN on served
  activations, and the decode output degenerates.
- With `HEAD_GROUP=32`, `DOT_PRECISION=bf16`, `SHARED_KKT` and `LAZY_CONV_STATE`,
  the cell 5 prefill trace shows the GDN prefill kernel 7 to 9% slower at the 6528
  and 8192 buckets than a run without them. Dropping `HEAD_GROUP` and
  `DOT_PRECISION` while keeping `LAZY_CONV_STATE` is the next thing to try; it has
  not run.
- `TPU_SAMPLER_TOPK_MAX_K=20` is exact only while every request has top_k <= 20;
  above that the runner falls back to the full path.
- `SYNTHETIC_ACCEPTANCE_LENGTH=3.39` fixes the MTP acceptance in both jobs, so a
  change in real draft acceptance from the fp8 and bf16 flags would not show in x
  or y.
- The branch is committed as the job ran it. The repo's pre-commit hooks were not
  applied (ruff check, env reads outside `envs.py`, pyrefly).
