#!/usr/bin/env bash
# Ship this repo to all three instances and run their setup scripts via SSM (no SSH, no open ports).
# Code goes through a private S3 bucket + a 1-hour presigned URL (code only; secrets come from SSM).
#
#   ./infra/aws/deploy_code.sh              # all nodes
#   ./infra/aws/deploy_code.sh control      # just one role (prod|shadow|control)
# GPU setup pulls the vLLM image (~10 GB) and Qwen2.5-7B weights (~15 GB): expect 10-20 minutes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source infra/aws/instances.env
export AWS_DEFAULT_REGION=$REGION AWS_PAGER=""
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=ff-code-$ACCOUNT-$REGION
ONLY=${1:-all}

if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi
TGZ=$(mktemp -t ff-code).tgz
tar czf "$TGZ" --exclude .venv --exclude state --exclude data --exclude .git --exclude __pycache__ \
  --exclude 'infra/aws/instances.env' --exclude '*.sqlite' --exclude '.env' .
aws s3 cp "$TGZ" "s3://$BUCKET/code.tgz" --quiet
URL=$(aws s3 presign "s3://$BUCKET/code.tgz" --expires-in 3600)

run() {  # instance-id setup-command
  aws ssm send-command --instance-ids "$1" --document-name AWS-RunShellScript --timeout-seconds 600 \
    --comment "firefighter setup" \
    --parameters "{\"executionTimeout\":[\"5400\"],\"commands\":[\"set -e\",\"mkdir -p /opt/firefighter && cd /opt/firefighter\",\"curl -sfL '$URL' | tar xz\",\"$2 > /var/log/firefighter-setup.log 2>&1\"]}" \
    --query Command.CommandId --output text
}

PROD_IP=${PROD_IP:-} SHADOW_IP=${SHADOW_IP:-}
# GPU nodes on Lightning AI instead of AWS: use the Studio controller URLs for the control node.
if [ -f infra/lightning/studio.env ]; then
  # shellcheck disable=SC1091
  source infra/lightning/studio.env
  PROD_IP=${PROD_CONTROLLER_URL:-$PROD_IP} SHADOW_IP=${SHADOW_CONTROLLER_URL:-$SHADOW_IP}
  echo "GPU nodes: Lightning ($PROD_IP, $SHADOW_IP)"
fi
if [[ -z "$PROD_ID" && $ONLY != control ]]; then echo "GPU nodes not launched: deploying control only"; ONLY=control; fi
if [[ $ONLY == all || $ONLY == prod ]]; then echo "prod: $(run "$PROD_ID" "bash infra/node/setup_gpu.sh prod $REGION")"; fi
if [[ $ONLY == all || $ONLY == shadow ]]; then echo "shadow: $(run "$SHADOW_ID" "bash infra/node/setup_gpu.sh shadow $REGION")"; fi
if [[ $ONLY == all || $ONLY == control ]]; then
  echo "control: $(run "$CONTROL_ID" "bash infra/node/setup_control.sh $REGION $PROD_IP $SHADOW_IP")"
fi
echo "follow a node:  ./infra/aws/ssm.sh <prod|shadow|control> 'tail -n 30 /var/log/firefighter-setup.log'"
