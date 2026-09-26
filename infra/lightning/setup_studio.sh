#!/usr/bin/env bash
# Runs INSIDE a Lightning Studio (1 x NVIDIA L4 on GCP). Two Studios: ff-prod (vLLM + loadgen = the
# users) and ff-shadow (vLLM for experiments). The node controller listens on :9000, which the Studio
# exposes as a public HTTPS URL; every request needs the bearer token. vLLM binds to 127.0.0.1 only.
#   bash infra/lightning/setup_studio.sh prod|shadow   (idempotent; needs ~/.ff/node.env from studio.py)
set -euo pipefail  # no xtrace: this script handles secrets
ROLE=$1
FF=/teamspace/studios/this_studio/ff
APP=$FF/app
cd $APP
set -a; source ~/.ff/node.env; set +a

# 1. Python env with vLLM + our node deps (uv is preinstalled on Studios and is fast).
if [ ! -x $FF/venv/bin/vllm ]; then
  uv venv --python 3.12 $FF/venv
  VIRTUAL_ENV=$FF/venv uv pip install "vllm${VLLM_VERSION:+==$VLLM_VERSION}" -r requirements-node.txt
fi
$FF/venv/bin/vllm --version || true

# 2. Weights on persistent storage (so restarts only reload from disk).
export HF_HOME=$FF/hf
$FF/venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"

# 3. Deterministic dataset (same fingerprint as everywhere else).
$FF/venv/bin/python -m workload.dataset build --out $FF/dataset.json

bash infra/lightning/start.sh "$ROLE"
echo "SETUP COMPLETE ($ROLE)"
