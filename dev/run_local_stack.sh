#!/usr/bin/env bash
# LOCAL PLUMBING TEST STACK (laptop, no GPU): two controllers on the FAKE engine, loadgen,
# MCP server. Proves wiring only; the numbers are not real. Never use for the demo.
#   ./dev/run_local_stack.sh start|stop
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=${LOG_DIR:-/tmp/ff}
mkdir -p "$LOG"

stop() {
  pkill -f "uvicorn controller.app:app" || true
  pkill -f "dev.fake_vllm" || true
  pkill -f "workload.loadgen" || true
  pkill -f "mcp_server.server" || true
}

start() {
  stop; sleep 1
  rm -rf state
  [ -f data/dataset.json ] || $PY -m workload.dataset build --out data/dataset.json
  export ENGINE_DRIVER=fake CONTROLLER_TOKEN=devtoken
  SLOTS=prod:0:8100 STATE_DIR=state/prod-node nohup .venv/bin/uvicorn controller.app:app --port 9000 >"$LOG/ctl-prod.log" 2>&1 &
  SLOTS=shadow:0:8101 STATE_DIR=state/shadow-node nohup .venv/bin/uvicorn controller.app:app --port 9001 >"$LOG/ctl-shadow.log" 2>&1 &
  sleep 3
  CFG=$($PY -c "import yaml,json;print(json.dumps(yaml.safe_load(open('infra/configs/prod_initial.yaml'))))")
  for pair in 9000:prod 9001:shadow; do
    curl -sf -XPOST "localhost:${pair%%:*}/slots/${pair##*:}/deploy" -H "Authorization: Bearer devtoken" \
      -H 'content-type: application/json' \
      -d "{\"config\":$CFG,\"author\":\"platform-team\",\"message\":\"Initial prod config for qwen2.5-7b (reviewed)\"}" >/dev/null
  done
  sleep 5
  cp workload/scenarios/healthy.json state/prod-node/scenario.json
  nohup $PY -m workload.loadgen --base-url http://127.0.0.1:8100 --dataset data/dataset.json \
    --scenario-file state/prod-node/scenario.json --db state/prod-node/requests.db >"$LOG/loadgen.log" 2>&1 &
  MCP_AUTH_TOKEN=mcptoken PROD_CONTROLLER_URL=http://127.0.0.1:9000 SHADOW_CONTROLLER_URL=http://127.0.0.1:9001 \
    MCP_STATE_DIR=state/mcp MCP_HOST=127.0.0.1 MCP_PORT=8765 \
    nohup $PY -m mcp_server.server >"$LOG/mcp.log" 2>&1 &
  sleep 3
  echo "up: controllers :9000/:9001, MCP http://127.0.0.1:8765/mcp (token mcptoken), logs in $LOG"
  echo "shift traffic: CONTROLLER_TOKEN=devtoken PROD_CONTROLLER_URL=http://127.0.0.1:9000 $PY -m chaos.chaos traffic long_context_shift"
}

"${1:-start}"
