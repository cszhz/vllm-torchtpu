#!/usr/bin/env bash
# Build the dsv41-pd image (everything except the model weights) and export it
# as a gzip tarball with a sha256 file.
#
#   bash build_image.sh                       # branch tip of cszhz/dsv41-1p1d
#   COMMIT=4b0fb058d bash build_image.sh      # a specific commit
#
# The image reuses the host venvs as-is (they were built on Ubuntu 24.04's
# /usr/bin/python3.12), so build on an Ubuntu 24.04 host that has them:
#   VENV_EPIC     inference venv (vLLM + torch-tpu + jax + libtpu, patched by
#                 apply_venv_patches.py, custom tpu-sync extension), editable
#                 install of vllm-torchtpu from CODE_SRC/src
#   VENV_AIPERF   aiperf client venv
#   TOKENIZER_DIR tokenizer files of DeepSeek-V4.1-Flash
# Hard-coded /zzlfs paths inside the venvs (shebangs, editable .pth, RECORD)
# are rewritten to /opt/... in the build context copy.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VENV_EPIC=${VENV_EPIC:-/zzlfs/temu/.venv-epic}
VENV_AIPERF=${VENV_AIPERF:-/zzlfs/temu/.venv-aiperf}
TOKENIZER_DIR=${TOKENIZER_DIR:-/zzlfs/temu/tools/dsv41-tokenizer}
CODE_SRC=${CODE_SRC:-/zzlfs/temu/src/vllm-torchtpu-cszhz}   # path the editable install points at
REPO=${REPO:-https://github.com/cszhz/vllm-torchtpu.git}
BRANCH=${BRANCH:-cszhz/dsv41-1p1d}
COMMIT=${COMMIT:-}
CTX=${CTX:-/var/tmp/dsv41-docker-ctx}
OUT_DIR=${OUT_DIR:-/zzlfs/temu/docker}
DOCKER=${DOCKER:-sudo docker}

echo "== build context $CTX"
rm -rf "$CTX" && mkdir -p "$CTX" && cd "$CTX"
rsync -a "$VENV_EPIC/" venv-epic/
rsync -a "$VENV_AIPERF/" venv-aiperf/
rsync -a "$TOKENIZER_DIR/" dsv41-tokenizer/

echo "== rewrite venv paths to /opt"
rewrite() {  # dir old new
  grep -rIl --exclude='*.pyc' -F "$2" "$1" | xargs -r sed -i "s#$2#$3#g"
}
rewrite venv-epic "$VENV_EPIC" /opt/venv-epic
rewrite venv-aiperf "$VENV_AIPERF" /opt/venv-aiperf
rewrite venv-epic "$CODE_SRC" /opt/vllm-torchtpu
left=$(grep -rIl --exclude='*.pyc' '/zzlfs' venv-epic venv-aiperf | wc -l)
echo "remaining /zzlfs references: $left"
[[ "$left" == 0 ]] || grep -rIl --exclude='*.pyc' '/zzlfs' venv-epic venv-aiperf | head -5

echo "== code"
if [[ -n "$COMMIT" ]]; then
  git clone -q --filter=blob:none --no-checkout --branch "$BRANCH" "$REPO" code
  git -C code checkout -q "$COMMIT"
else
  git clone -q --depth 1 --branch "$BRANCH" "$REPO" code
fi
SHA=$(git -C code rev-parse --short=9 HEAD)
echo "code at $SHA: $(git -C code log --oneline -1)"

cp "$HERE/Dockerfile" "$HERE/dsv41-pd-entrypoint.sh" .
bash -n dsv41-pd-entrypoint.sh
TAG=dsv41-pd:$SHA
echo "== docker build $TAG"
$DOCKER build -t "$TAG" -t dsv41-pd:latest .

echo "== smoke checks (no TPU needed)"
$DOCKER run --rm --entrypoint bash "$TAG" -c '
  python -c "import torch, jax, vllm, vllm_torchtpu; print(\"torch\", torch.__version__, \"jax\", jax.__version__, \"vllm\", vllm.__version__)"
  git -C /opt/vllm-torchtpu log --oneline -1 && test -z "$(git -C /opt/vllm-torchtpu status --porcelain)"
  /opt/venv-aiperf/bin/aiperf --version
  test -f /opt/dsv41-tokenizer/tokenizer.json'

echo "== export"
mkdir -p "$OUT_DIR" && cd "$OUT_DIR"
F=dsv41-pd-$SHA.tar.gz
command -v pigz >/dev/null && Z="pigz -p 32" || Z="gzip -1"
$DOCKER save "$TAG" | $Z > "$F"
sha256sum "$F" > "$F.sha256"
ls -la "$OUT_DIR/$F"; cat "$F.sha256"
echo "upload: gcloud storage cp $OUT_DIR/$F $OUT_DIR/$F.sha256 gs://<bucket>/dsv41_1p1d/"
