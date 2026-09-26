#!/usr/bin/env bash
# (Re)start the node controller (+ prod traffic on ff-prod) inside a Studio, and bring its vLLM slot
# back to the last active config (or the initial config on first run). Safe after a Studio restart.
#   bash infra/lightning/start.sh prod|shadow
set -euo pipefail
ROLE=$1
FF=/teamspace/studios/this_studio/ff
APP=$FF/app
cd $APP
set -a; source ~/.ff/node.env; set +a
export HF_HOME=$FF/hf ENGINE_DRIVER=process VLLM_BIN=$FF/venv/bin/vllm STATE_DIR=$FF/state \
       DATASET_PATH=$FF/dataset.json SLOTS="$ROLE:0:8100" HEALTH_TIMEOUT_S=900 ROLE
mkdir -p $FF/state $FF/logs
PY=$FF/venv/bin/python

pkill -f "uvicorn controller.app:app" || true
pkill -f "workload.loadgen" || true
sleep 2
nohup $FF/venv/bin/uvicorn controller.app:app --host 0.0.0.0 --port 9000 > $FF/logs/controller.log 2>&1 &
for _ in $(seq 1 30); do curl -sf localhost:9000/health >/dev/null && break; sleep 1; done
curl -sf localhost:9000/health; echo

# Bring each slot to its last active config (first run: the initial "legal but wrong" config).
$PY - <<'EOF'
import json, os, time, httpx, yaml
H = {"Authorization": f"Bearer {os.environ['CONTROLLER_TOKEN']}"}
c = httpx.Client(base_url="http://127.0.0.1:9000", headers=H, timeout=30)
initial = yaml.safe_load(open("infra/configs/prod_initial.yaml"))
deps = {}
for slot in (os.environ["ROLE"],):
    info = c.get(f"/slots/{slot}").json()
    if info["healthy"]:
        print(slot, "already healthy"); continue
    act = info.get("active_version")
    cfg = act["config"] if act else initial
    msg = "restart after Studio resume" if act else (
        "Initial prod config for qwen2.5-7b (reviewed)" if slot == "prod" else "shadow baseline")
    deps[slot] = c.post(f"/slots/{slot}/deploy", json={"config": cfg, "author": "platform-team", "message": msg}).json()["id"]
deadline = time.time() + 1200
while deps and time.time() < deadline:
    time.sleep(8)
    for slot, dep in list(deps.items()):
        st = c.get(f"/deploys/{dep}").json()
        print(slot, st["status"], flush=True)
        if st["status"] not in ("queued", "stopping", "starting"):
            if st["status"] != "healthy":
                print(st.get("failure_log_tail") or st.get("error"))
            del deps[slot]
EOF

# Users only start once prod is serving (prod Studio only).
if [ "$ROLE" = prod ]; then
[ -f $FF/state/scenario.json ] || cp workload/scenarios/healthy.json $FF/state/scenario.json
nohup $PY -m workload.loadgen --base-url http://127.0.0.1:8100 --dataset $FF/dataset.json \
  --scenario-file $FF/state/scenario.json --db $FF/state/requests.db > $FF/logs/loadgen.log 2>&1 &
fi
grep -hiE "KV cache|maximum concurrency" $FF/state/*/vllm.log | tail -4 || true
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "STARTED"
