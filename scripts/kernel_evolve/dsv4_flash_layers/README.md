# DeepSeek-V4-Flash layer-run campaign: what the scorer reads

The campaign is scored by forge's `scorers/serve_e2e.py` on
`deepseek-v4-flash-dp8-ep@layers`: the model cut to its leading layers that hold every
kind of layer it has, served on one tpu7x 2x2x1 node through Cloud DevKit, and scored on
the device's op time per step.

That scorer reads its serve config, its target and the cell's rows from a perfkit corpus.
They are here, so a run on any host finds them in its clone of this tree:

    PERFKIT_CORPUS=<clone>/scripts/kernel_evolve/dsv4_flash_layers/corpus \
      perfkit scorer run --bench <frozen serve_e2e.py> --config deepseek-v4-flash-dp8-ep@layers ...

- `corpus/scorers/serve_e2e.toml`: the serve config, its cell and `remote = "cdk"`.
- `corpus/targets/vllm-torchtpu-dsv4-flash.toml`: the target, with the serving image by
  digest and the checkpoint the pods stream (`MODEL_URI`, a gs:// copy).
- `corpus/ledger/cell-blocks.toml`: the cell's rows and census from golden's layer runs.
- `shim/`: lets `run_benchmarks.sh --dry-run`, which perfkit reads the serve command
  from, run on a host without vLLM installed.
