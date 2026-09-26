#!/usr/bin/env bash
# Store secrets in SSM Parameter Store (SecureString) under /firefighter/*. Instances read them
# with their IAM role at setup time. Nothing secret goes into the repo, scripts or SSM command
# history. Internal tokens are generated here if you don't pass them.
#
#   HF_TOKEN=hf_...  [GITHUB_TOKEN=... GITHUB_REPO=owner/repo]  [PROMETHEUS_URL=http://...]  ./infra/aws/secrets.sh
#   (OpenAI and Daytona keys are NOT stored here: paste them into TrueForge Settings.)
set -euo pipefail
cd "$(dirname "$0")"
source instances.env
export AWS_DEFAULT_REGION=$REGION AWS_PAGER=""

put() {  # name value [type]
  [ -n "$2" ] || return 0
  aws ssm put-parameter --name "/firefighter/$1" --value "$2" --type "${3:-SecureString}" --overwrite >/dev/null
  echo "set /firefighter/$1"
}
exists() { aws ssm get-parameter --name "/firefighter/$1" >/dev/null 2>&1; }

exists controller_token || put controller_token "$(openssl rand -hex 24)"
exists mcp_token || put mcp_token "$(openssl rand -hex 24)"
put hf_token "${HF_TOKEN:-}"
put github_token "${GITHUB_TOKEN:-}"
put github_repo "${GITHUB_REPO:-}" String
put prometheus_url "${PROMETHEUS_URL:-}" String
echo "done. Read a value (e.g. to paste the MCP token somewhere): aws ssm get-parameter --name /firefighter/mcp_token --with-decryption --query Parameter.Value --output text"
