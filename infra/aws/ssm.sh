#!/usr/bin/env bash
# Run a shell command on a node via SSM and print its output.
#   ./infra/aws/ssm.sh control 'systemctl status ff-mcp --no-pager'
#   ./infra/aws/ssm.sh control 'cd /opt/firefighter && ./infra/node/chaos.sh traffic long_context_shift'
set -euo pipefail
cd "$(dirname "$0")"
source instances.env
export AWS_DEFAULT_REGION=$REGION AWS_PAGER=""
case "$1" in prod) ID=$PROD_ID ;; shadow) ID=$SHADOW_ID ;; control) ID=$CONTROL_ID ;; *) echo "role: prod|shadow|control"; exit 1 ;; esac
CMD=$(python3 -c 'import json,sys; print(json.dumps({"commands": [sys.argv[1]], "executionTimeout": ["3600"]}))' "$2")
CID=$(aws ssm send-command --instance-ids "$ID" --document-name AWS-RunShellScript --parameters "$CMD" \
  --query Command.CommandId --output text)
for _ in $(seq 1 720); do
  STATUS=$(aws ssm get-command-invocation --command-id "$CID" --instance-id "$ID" --query Status --output text 2>/dev/null || echo Pending)
  case "$STATUS" in Pending|InProgress|Delayed) sleep 5 ;; *) break ;; esac
done
aws ssm get-command-invocation --command-id "$CID" --instance-id "$ID" --query StandardOutputContent --output text
ERR=$(aws ssm get-command-invocation --command-id "$CID" --instance-id "$ID" --query StandardErrorContent --output text)
[ -z "$ERR" ] || echo "stderr: $ERR" >&2
echo "[$STATUS]"
