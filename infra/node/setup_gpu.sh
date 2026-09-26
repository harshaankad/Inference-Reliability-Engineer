#!/usr/bin/env bash
# GPU node setup (AWS Deep Learning Base AMI, Ubuntu 22.04, A10G). Run as root from /opt/firefighter.
#   bash infra/node/setup_gpu.sh prod|shadow <region>
# Installs the controller (+ loadgen on prod), pre-pulls vLLM and the model weights, and deploys the
# initial "legal but wrong" production config.
set -euxo pipefail
ROLE=$1
REGION=$2
APP=/opt/firefighter
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:latest}
cd $APP

param() { aws ssm get-parameter --region "$REGION" --name "/firefighter/$1" --with-decryption \
  --query Parameter.Value --output text 2>/dev/null || true; }
CONTROLLER_TOKEN=$(param controller_token)
HF_TOKEN=$(param hf_token)
[ -n "$CONTROLLER_TOKEN" ] || { echo "missing /firefighter/controller_token (run infra/aws/secrets.sh)"; exit 1; }

nvidia-smi
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-venv python3-pip sqlite3 jq
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-node.txt
.venv/bin/python -m workload.dataset build --out data/dataset.json

mkdir -p /opt/hf state /etc/firefighter
docker pull "$VLLM_IMAGE"
# Pre-download weights so every later deploy is a ~1-3 min restart, not a 15 GB download.
docker run --rm -v /opt/hf:/root/.cache/huggingface -e HF_TOKEN="$HF_TOKEN" --entrypoint python3 "$VLLM_IMAGE" \
  -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"

umask 077
cat > /etc/firefighter/node.env <<EOF
CONTROLLER_TOKEN=$CONTROLLER_TOKEN
HF_TOKEN=$HF_TOKEN
SLOTS=$ROLE:0:8100
ENGINE_DRIVER=docker
MODEL=$MODEL
SERVED_MODEL_NAME=qwen2.5-7b
VLLM_IMAGE=$VLLM_IMAGE
HF_CACHE=/opt/hf
STATE_DIR=$APP/state
DATASET_PATH=$APP/data/dataset.json
HEALTH_TIMEOUT_S=900
EOF
umask 022

cat > /etc/systemd/system/ff-controller.service <<EOF
[Unit]
Description=Inference Firefighter node controller ($ROLE)
After=docker.service network-online.target
Requires=docker.service

[Service]
EnvironmentFile=/etc/firefighter/node.env
WorkingDirectory=$APP
ExecStart=$APP/.venv/bin/uvicorn controller.app:app --host 0.0.0.0 --port 9000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/ff-loadgen.service <<EOF
[Unit]
Description=Inference Firefighter production traffic generator
After=ff-controller.service

[Service]
EnvironmentFile=/etc/firefighter/node.env
WorkingDirectory=$APP
ExecStart=$APP/.venv/bin/python -m workload.loadgen --base-url http://127.0.0.1:8100 --dataset $APP/data/dataset.json --scenario-file $APP/state/scenario.json --db $APP/state/requests.db
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now ff-controller
sleep 5
curl -sf localhost:9000/health

# Deploy the initial config and wait until vLLM is healthy.
CFG=$(.venv/bin/python -c "import yaml,json;print(json.dumps(yaml.safe_load(open('infra/configs/prod_initial.yaml'))))")
MSG="Initial prod config for qwen2.5-7b (reviewed)"
[ "$ROLE" = shadow ] && MSG="shadow baseline"
DEP=$(curl -sf -XPOST "localhost:9000/slots/$ROLE/deploy" -H "Authorization: Bearer $CONTROLLER_TOKEN" \
  -H 'content-type: application/json' -d "{\"config\":$CFG,\"author\":\"platform-team\",\"message\":\"$MSG\"}" | jq -r .id)
for _ in $(seq 1 200); do
  STATUS=$(curl -sf "localhost:9000/deploys/$DEP" -H "Authorization: Bearer $CONTROLLER_TOKEN" | jq -r .status)
  echo "deploy $DEP: $STATUS"
  case "$STATUS" in healthy) break ;; failed|error|rollback_failed) docker logs --tail 80 "vllm-$ROLE" || true; exit 1 ;; esac
  sleep 6
done
docker logs "vllm-$ROLE" 2>&1 | grep -iE "KV cache|blocks|maximum concurrency" | tail -5 || true

if [ "$ROLE" = prod ]; then
  cp workload/scenarios/healthy.json state/scenario.json
  systemctl enable --now ff-loadgen
fi
echo "SETUP COMPLETE ($ROLE)"
