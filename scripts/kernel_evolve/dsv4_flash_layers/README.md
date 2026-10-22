# DeepSeek-V4-Flash layer-run campaign: what the scorer reads

The campaign is scored by forge's `scorers/serve_e2e.py` on
`deepseek-v4-flash-dp8-ep@layers`: the model cut to its leading 5 of 43 layers, which
hold every kind of layer it has, served on one tpu7x 2x2x1 node through Cloud DevKit
and scored on the device's op time per step.

That scorer reads its serve config, its target and the cell's rows from a perfkit
corpus. They are here, so a run on any host reads them from its clone of this tree:

    PERFKIT_CORPUS=<clone>/scripts/kernel_evolve/dsv4_flash_layers/corpus \
    PERFKIT_PYTHON=<a python with jax and numpy> \
      perfkit scorer run --bench <frozen serve_e2e.py> \
        --arm golden=<golden checkout>:control --arm cand=<candidate checkout> \
        --config deepseek-v4-flash-dp8-ep@layers --candidate cand --job-id <id>

`PERFKIT_PYTHON` is the python the scorer reads the arms' profiles with: jax's
`ProfileData` reads the captures, so it needs jax (CPU is enough) and numpy.

- `corpus/scorers/serve_e2e.toml`: the serve config, its cell, `remote = "cdk"`, and
  the gs:// checkpoint the cut is planned from.
- `corpus/targets/vllm-torchtpu-dsv4-flash.toml`: the target. Its `repo` is `.`, the
  arm's own tree (each arm runs with its tree as the working directory). It names the
  serving image by digest and the checkpoint the pods stream (`MODEL_URI`, gs://).
- `corpus/ledger/cell-blocks.toml`: the cell's block rules, rows and census from
  golden's layer runs; `kv-blocks.json`: the full model's KV cache allocation, which
  every layer arm pins its cut to; `cell-map.json`, `experiments.jsonl`: the cell.
- `shim/`: lets `run_benchmarks.sh --dry-run`, which perfkit reads the serve command
  from, run on a host without vLLM installed.
- `validity.py`: the campaign's validity check. The arm's benchmark config is the one
  it is served with, so this fails a tree whose config launches another cell.
