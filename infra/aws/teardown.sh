#!/usr/bin/env bash
# Terminate the fleet (GPU credits are finite). --all also removes the SG, IAM role, code bucket
# and /firefighter/* parameters.
set -euo pipefail
cd "$(dirname "$0")"
REGION=${AWS_REGION:-$(grep ^REGION= instances.env 2>/dev/null | cut -d= -f2)}
REGION=${REGION:-ap-south-1}
export AWS_DEFAULT_REGION=$REGION AWS_PAGER=""
TAG=inference-firefighter
IDS=$(aws ec2 describe-instances --filters Name=tag:Project,Values=$TAG Name=instance-state-name,Values=pending,running,stopped \
  --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$IDS" ]; then
  echo "terminating: $IDS"
  aws ec2 terminate-instances --instance-ids $IDS >/dev/null
  aws ec2 wait instance-terminated --instance-ids $IDS
fi
if [ "${1:-}" = "--all" ]; then
  ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
  aws s3 rb "s3://ff-code-$ACCOUNT-$REGION" --force 2>/dev/null || true
  SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$TAG-sg --query 'SecurityGroups[0].GroupId' --output text)
  [ "$SG" = "None" ] || aws ec2 delete-security-group --group-id "$SG"
  aws iam remove-role-from-instance-profile --instance-profile-name $TAG-node --role-name $TAG-node 2>/dev/null || true
  aws iam delete-instance-profile --instance-profile-name $TAG-node 2>/dev/null || true
  aws iam delete-role-policy --role-name $TAG-node --policy-name $TAG-params 2>/dev/null || true
  aws iam detach-role-policy --role-name $TAG-node --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true
  aws iam delete-role --role-name $TAG-node 2>/dev/null || true
  for p in $(aws ssm get-parameters-by-path --path /firefighter --query 'Parameters[].Name' --output text); do
    aws ssm delete-parameter --name "$p"
  done
fi
echo "teardown done"
