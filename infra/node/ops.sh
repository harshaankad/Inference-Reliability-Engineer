#!/usr/bin/env bash
# Convenience wrapper on the CONTROL node (loads /etc/firefighter/control.env):
#   ./infra/node/ops.sh chaos traffic long_context_shift
#   ./infra/node/ops.sh chaos reset --clear-evidence /opt/firefighter/state/mcp
#   ./infra/node/ops.sh chaos status
#   ./infra/node/ops.sh register-agent openai/<model-id>     # after OpenAI + Daytona are set in the UI
#   ./infra/node/ops.sh smoke                                  # MCP smoke test (shadow only)
#   ./infra/node/ops.sh tool get_slo_status '{"window_minutes": 5}'   # call any MCP tool
set -euo pipefail
cd /opt/firefighter
set -a; source /etc/firefighter/control.env; set +a
cmd=$1; shift
case "$cmd" in
  chaos) exec .venv/bin/python -m chaos.chaos "$@" ;;
  register-agent) AGENT_MODEL=$1 exec .venv/bin/python -m agent.create_agent "${@:2}" ;;
  smoke) MCP_URL=$MCP_PUBLIC_URL exec .venv/bin/python -m tests.smoke_mcp "$@" ;;
  tool) MCP_URL=$MCP_PUBLIC_URL exec .venv/bin/python -m tests.mcp_call "$@" ;;
  *) echo "usage: ops.sh chaos|register-agent|smoke|tool ..."; exit 1 ;;
esac
