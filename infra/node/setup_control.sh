#!/usr/bin/env bash
# Control node setup (Ubuntu 22.04, t3.large). Run as root from /opt/firefighter.
#   bash infra/node/setup_control.sh <region> <prod> <shadow>
#   <prod>/<shadow>: an AWS private IP (-> http://IP:9000) or a full controller URL
#   (e.g. a Lightning Studio port URL https://9000-<id>.cloudspaces.litng.ai)
# Installs: MCP server (systemd ff-mcp), Prometheus (unless /firefighter/prometheus_url points at
# yours), TrueForge (npx, systemd ff-trueforge, bound to localhost, reached via SSM tunnel).
set -euxo pipefail
REGION=$1
to_url() { case "$1" in *://*) echo "${1%/}" ;; "") echo "" ;; *) echo "http://$1:9000" ;; esac; }
PROD_URL=$(to_url "${2:-}")     # may be empty while GPU nodes don't exist yet; re-run setup once they do
SHADOW_URL=$(to_url "${3:-}")
scheme_of() { echo "${1%%://*}"; }
target_of() { local h=${1#*://}; echo "${h%%/*}"; }
APP=/opt/firefighter
cd $APP
# A real (re)deploy always removes the fake-engine wiring-test stack if it is running.
[ -f dev/aws_wiring_test.sh ] && [ -f /etc/firefighter/control.env ] && bash dev/aws_wiring_test.sh stop || true

param() { aws ssm get-parameter --region "$REGION" --name "/firefighter/$1" --with-decryption \
  --query Parameter.Value --output text 2>/dev/null || true; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-venv python3-pip jq curl unzip ca-certificates
if ! command -v aws >/dev/null; then
  curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip
  unzip -q -o /tmp/awscli.zip -d /tmp && /tmp/aws/install
fi
if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sh; fi
if ! node --version 2>/dev/null | grep -qE '^v(2[2-9]|[3-9][0-9])'; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs
fi

CONTROLLER_TOKEN=$(param controller_token)
MCP_TOKEN=$(param mcp_token)
[ -n "$CONTROLLER_TOKEN" ] && [ -n "$MCP_TOKEN" ] || { echo "missing tokens (run infra/aws/secrets.sh)"; exit 1; }
PROM_URL=$(param prometheus_url)
IMDS=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
CONTROL_IP=$(curl -s -H "X-aws-ec2-metadata-token: $IMDS" http://169.254.169.254/latest/meta-data/local-ipv4)

python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-control.txt
mkdir -p /etc/firefighter state/mcp /etc/prometheus

# ---- Prometheus (skip if you already run one: set /firefighter/prometheus_url and add the
# ---- jobs from infra/prometheus/prometheus.yml.tmpl to it)
if [ -z "$PROM_URL" ] && [ -n "$PROD_URL" ]; then
  umask 077; printf '%s' "$CONTROLLER_TOKEN" > /etc/prometheus/controller_token; umask 022
  chmod 644 /etc/prometheus/controller_token  # prom container runs as nobody; host is single-purpose
  sed -e "s#__PROD_SCHEME__#$(scheme_of "$PROD_URL")#g; s#__PROD_TARGET__#$(target_of "$PROD_URL")#g" \
      -e "s#__SHADOW_SCHEME__#$(scheme_of "$SHADOW_URL")#g; s#__SHADOW_TARGET__#$(target_of "$SHADOW_URL")#g" \
      infra/prometheus/prometheus.yml.tmpl > /etc/prometheus/prometheus.yml
  docker rm -f prometheus 2>/dev/null || true
  docker run -d --name prometheus --restart=always --network host \
    -v /etc/prometheus:/etc/prometheus -v prometheus-data:/prometheus prom/prometheus:latest \
    --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.retention.time=3d --web.listen-address=127.0.0.1:9090
  PROM_URL=http://127.0.0.1:9090
elif [ -z "$PROM_URL" ]; then
  echo "GPU node URLs unknown: skipping Prometheus for now (re-run this script once GPU nodes exist)"
fi

GITHUB_TOKEN=$(param github_token)
GITHUB_REPO=$(param github_repo)
umask 077
cat > /etc/firefighter/control.env <<EOF
CONTROLLER_TOKEN=$CONTROLLER_TOKEN
MCP_AUTH_TOKEN=$MCP_TOKEN
PROD_CONTROLLER_URL=$PROD_URL
SHADOW_CONTROLLER_URL=$SHADOW_URL
PROD_SLOT=prod
SHADOW_SLOT=shadow
MCP_STATE_DIR=$APP/state/mcp
MCP_HOST=0.0.0.0
MCP_PORT=8765
PROMETHEUS_URL=$PROM_URL
GITHUB_TOKEN=$GITHUB_TOKEN
GITHUB_REPO=$GITHUB_REPO
CONTROL_IP=$CONTROL_IP
MCP_PUBLIC_URL=http://$CONTROL_IP:8765/mcp
TRUEFORGE_BASE_URL=http://127.0.0.1:8790
EOF
umask 022

cat > /etc/systemd/system/ff-mcp.service <<EOF
[Unit]
Description=Inference Firefighter MCP server (inference-ops)
After=network-online.target

[Service]
EnvironmentFile=/etc/firefighter/control.env
WorkingDirectory=$APP
ExecStart=$APP/.venv/bin/python -m mcp_server.server
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# TrueForge local mode. It blocks outbound calls to private IPs by default, so our MCP server's
# private IP must be allow-listed explicitly (JSON array; single quotes keep the inner quotes,
# which systemd would strip from an Environment= line).
cat > /etc/firefighter/trueforge.env <<EOF
SQLITE_PATH=$APP/state/trueforge.sqlite
PUBLIC_BASE_URL=http://localhost:8790
HOST=127.0.0.1
OUTBOUND_URL_ALLOWED_HOSTS='["$CONTROL_IP"]'
# An incident run (several shadow deploys + load tests) takes 15-30 min and single tool calls can
# take several minutes; the defaults (600 s per turn, 4 min per MCP call) cut runs short.
SERVER_EXECUTION_TIMEOUT_SECONDS=3600
MCP_REQUEST_TIMEOUT_MS=1200000
HOME=/root
EOF

cat > /etc/systemd/system/ff-trueforge.service <<EOF
[Unit]
Description=TrueForge agent harness
After=network-online.target ff-mcp.service

[Service]
EnvironmentFile=/etc/firefighter/trueforge.env
WorkingDirectory=$APP
ExecStart=/usr/bin/npx -y @truefoundry/trueforge@latest --port 8790
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ff-mcp ff-trueforge
systemctl restart ff-mcp          # pick up a re-rendered control.env on re-runs
systemctl restart ff-trueforge    # picks up trueforge.env changes; data persists in SQLite
sleep 20
curl -s -o /dev/null -w "mcp without token -> %{http_code} (expect 401)\n" -XPOST "http://$CONTROL_IP:8765/mcp" || true
[ -n "$PROD_URL" ] && curl -sf "$PROD_URL/health" && echo " prod controller reachable" || echo "prod controller not up yet (GPU setup may still be running)"
[ -n "$SHADOW_URL" ] && curl -sf "$SHADOW_URL/health" && echo " shadow controller reachable" || echo "shadow controller not up yet"
for _ in $(seq 1 30); do curl -sf 127.0.0.1:8790/api/v1/agents >/dev/null && break; sleep 5; done
echo "SETUP COMPLETE (control). TrueForge on localhost:8790 (tunnel: ./infra/aws/tunnel.sh)"
