# DeepSeek-V4.1-Flash 1P1D on TPU v7x (vllm-torchtpu, Stage-3 Raiden).
# Everything except the model weights, which are streamed from GCS
# (gs://zzl-bucket/models/DeepSeek-V4.1-Flash) at startup.
#
# Ubuntu 24.04: the venv was built on its /usr/bin/python3.12 (3.12.3), and the
# custom tpu-sync extension needs glibc 2.38.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 libpython3.12 ca-certificates git curl procps iproute2 hostname \
        libibverbs1 libnuma1 libgomp1 libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# Inference venv: torch 2.13 / torch-tpu 0.2.0 / jax 0.11.2 / libtpu 0.0.50 /
# vLLM v0.31.1rc0, runtime patches applied (apply_venv_patches.py), tpu-sync
# extension replaced by the per-layer-major-dim build. Paths rewritten to /opt.
COPY venv-epic /opt/venv-epic
# Benchmark client (aiperf with --scenario).
COPY venv-aiperf /opt/venv-aiperf
# vllm-torchtpu, cszhz/dsv41-1p1d (shallow clone; the venv's editable install
# points at /opt/vllm-torchtpu/src).
COPY code /opt/vllm-torchtpu
# Tokenizer files from the model directory (the proxy tokenizes locally).
COPY dsv41-tokenizer /opt/dsv41-tokenizer
COPY dsv41-pd-entrypoint.sh /usr/local/bin/dsv41-pd

RUN chmod +x /usr/local/bin/dsv41-pd \
    && git config --system --add safe.directory /opt/vllm-torchtpu \
    && /opt/venv-epic/bin/python -c "import importlib.util as u; assert all(u.find_spec(m) for m in ('vllm', 'vllm_torchtpu', 'torch_tpu', 'tpu_sync', 'jax')); print('packages found')"

ENV PATH=/opt/venv-epic/bin:$PATH
WORKDIR /opt/vllm-torchtpu
ENTRYPOINT ["/usr/local/bin/dsv41-pd"]
