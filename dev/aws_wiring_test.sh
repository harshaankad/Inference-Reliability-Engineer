#!/usr/bin/env bash
# WIRING TEST ONLY, on the AWS control node, before GPU nodes exist. Runs the FAKE engine behind two
# local controllers and points the MCP server at them, so TrueForge -> MCP -> sandbox -> approval can
# be exercised end to end. Numbers are NOT real; never use this for calibration, evidence or the demo.
#   sudo bash dev/aws_wiring_test.sh start|stop|status
# `stop` removes everything and points the MCP server back at the real config.
set -euo pipefail
APP=/opt/firefighter
FAKE=$APP/state/fake
cd $APP
set -a; source /etc/firefighter/control.env; set +a

start() {
  [ -f data/dataset.json ] || .venv/bin/python -m workload.dataset build --out data/dataset.json
  mkdir -p "$FAKE"
  for spec in prod:8100:9100 shadow:8101:9101; do
    IFS=: read -r slot eport cport <<<"$spec"
    systemctl stop "ff-fake-$slot" 2>/dev/null || true
    systemd-run --unit "ff-fake-$slot" --working-directory=$APP \
      -E ENGINE_DRIVER=fake -E CONTROLLER_TOKEN="$CONTROLLER_TOKEN" -E SLOTS="$slot:0:$eport" \
      -E STATE_DIR="$FAKE/$slot-node" -E DATASET_PATH=$APP/data/dataset.json \
      $APP/.venv/bin/uvicorn controller.app:app --host 127.0.0.1 --port "$cport"
  done
  sleep 4
  export PROD_CONTROLLER_URL=http://127.0.0.1:9100 SHADOW_CONTROLLER_URL=http://127.0.0.1:9101
  .venv/bin/python -m chaos.chaos deploy-initial
  cp workload/scenarios/healthy.json "$FAKE/prod-node/scenario.json"
  systemctl stop ff-fake-loadgen 2>/dev/null || true
  systemd-run --unit ff-fake-loadgen --working-directory=$APP \
    $APP/.venv/bin/python -m workload.loadgen --base-url http://127.0.0.1:8100 --dataset $APP/data/dataset.json \
    --scenario-file "$FAKE/prod-node/scenario.json" --db "$FAKE/prod-node/requests.db"
  # Point the MCP server at the fakes via a drop-in (a later EnvironmentFile overrides control.env).
  mkdir -p /etc/systemd/system/ff-mcp.service.d
  cat > /etc/firefighter/fake.env <<EOF
PROD_CONTROLLER_URL=http://127.0.0.1:9100
SHADOW_CONTROLLER_URL=http://127.0.0.1:9101
MCP_STATE_DIR=$FAKE/mcp
PROMETHEUS_URL=
EOF
  printf '[Service]\nEnvironmentFile=/etc/firefighter/fake.env\n' > /etc/systemd/system/ff-mcp.service.d/fake.conf
  systemctl daemon-reload && systemctl restart ff-mcp
  echo "WIRING TEST STACK UP (fake engine). Stop with: sudo bash dev/aws_wiring_test.sh stop"
}

stop() {
  systemctl stop ff-fake-loadgen ff-fake-prod ff-fake-shadow 2>/dev/null || true
  pkill -f dev.fake_vllm || true
  rm -f /etc/systemd/system/ff-mcp.service.d/fake.conf /etc/firefighter/fake.env
  systemctl daemon-reload && systemctl restart ff-mcp
  rm -rf "$FAKE"
  echo "wiring test stack removed; MCP server back on the real config"
}

status() {
  systemctl is-active ff-fake-prod ff-fake-shadow ff-fake-loadgen ff-mcp || true
  ls /etc/systemd/system/ff-mcp.service.d/ 2>/dev/null || echo "no MCP drop-in (real config)"
}

"${1:-status}"
