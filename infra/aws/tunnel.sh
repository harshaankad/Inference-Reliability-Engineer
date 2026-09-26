#!/usr/bin/env bash
# Open the TrueForge UI (and optionally Prometheus) on your laptop through SSM. Nothing is exposed
# publicly. Needs the AWS CLI Session Manager plugin.
#   ./infra/aws/tunnel.sh            -> http://localhost:8790  (TrueForge)
#   ./infra/aws/tunnel.sh 9090       -> http://localhost:9090  (Prometheus on the control node)
set -euo pipefail
cd "$(dirname "$0")"
source instances.env
PORT=${1:-8790}
exec aws ssm start-session --region "$REGION" --target "$CONTROL_ID" --document-name AWS-StartPortForwardingSession \
  --parameters "{\"portNumber\":[\"$PORT\"],\"localPortNumber\":[\"$PORT\"]}"
