#!/usr/bin/env bash
# Launch the fleet in the organizer AWS account: 2 x g6.xlarge (NVIDIA L4, prod + shadow) and
# 1 x t3.large control node (TrueForge, MCP server, Prometheus), in ap-south-1 (Mumbai).
# Idempotent: re-running reuses the security group, IAM role and any running instances.
#
#   AWS_REGION=ap-south-1 ./infra/aws/launch.sh
# Writes infra/aws/instances.env (instance ids + private IPs) for the other scripts.
set -euo pipefail
cd "$(dirname "$0")"
REGION=${AWS_REGION:-ap-south-1}
GPU_TYPE=${GPU_INSTANCE_TYPE:-g6.xlarge}
CONTROL_TYPE=${CONTROL_INSTANCE_TYPE:-t3.large}
TAG=inference-firefighter
ROLE=$TAG-node
export AWS_DEFAULT_REGION=$REGION AWS_PAGER=""

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "account $ACCOUNT region $REGION"

VPC=${VPC_ID:-$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)}
[ "$VPC" != "None" ] || { echo "no default VPC: set VPC_ID and SUBNET_ID"; exit 1; }
AZS=$(aws ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters Name=instance-type,Values="$GPU_TYPE" --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' ',')
[ -n "$AZS" ] || { echo "$GPU_TYPE is not offered in $REGION"; exit 1; }
SUBNET=${SUBNET_ID:-$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC" Name=default-for-az,Values=true \
  Name=availability-zone,Values="$AZS" --query 'Subnets[0].SubnetId' --output text)}
echo "vpc $VPC subnet $SUBNET ($GPU_TYPE AZs: $AZS)"

# Security group: nodes talk to each other on any port; NOTHING is open to the internet.
# Humans reach the fleet only through SSM Session Manager.
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$TAG-sg Name=vpc-id,Values="$VPC" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --group-name $TAG-sg --vpc-id "$VPC" \
    --description "inference firefighter: intra-group only" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG}]"
fi
echo "security group $SG"

# IAM role: SSM access + read our /firefighter/* secrets. No static AWS keys on any instance.
if ! aws iam get-role --role-name $ROLE >/dev/null 2>&1; then
  aws iam create-role --role-name $ROLE --assume-role-policy-document file://iam/trust-ec2.json >/dev/null
  aws iam attach-role-policy --role-name $ROLE --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  sed -e "s/__REGION__/$REGION/; s/__ACCOUNT__/$ACCOUNT/" iam/params-policy.json.tmpl > /tmp/ff-params-policy.json
  aws iam put-role-policy --role-name $ROLE --policy-name $TAG-params --policy-document file:///tmp/ff-params-policy.json
  aws iam create-instance-profile --instance-profile-name $ROLE >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name $ROLE --role-name $ROLE
  echo "created IAM role $ROLE; waiting for propagation"; sleep 15
fi

GPU_AMI=${GPU_AMI:-$(aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id --query Parameter.Value --output text)}
CONTROL_AMI=${CONTROL_AMI:-$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id --query Parameter.Value --output text)}
echo "gpu ami $GPU_AMI (DLAMI base, NVIDIA driver + Docker), control ami $CONTROL_AMI"

launch() {  # name type ami disk_gb
  local existing
  existing=$(aws ec2 describe-instances --filters Name=tag:Project,Values=$TAG Name=tag:Role,Values="$1" \
    Name=instance-state-name,Values=pending,running --query 'Reservations[0].Instances[0].InstanceId' --output text)
  if [ "$existing" != "None" ]; then echo "$existing"; return; fi
  local root
  root=$(aws ec2 describe-images --image-ids "$3" --query 'Images[0].RootDeviceName' --output text)
  aws ec2 run-instances --image-id "$3" --instance-type "$2" --subnet-id "$SUBNET" --security-group-ids "$SG" \
    --iam-instance-profile Name=$ROLE --associate-public-ip-address \
    --metadata-options HttpTokens=required,HttpEndpoint=enabled \
    --block-device-mappings "DeviceName=$root,Ebs={VolumeSize=$4,VolumeType=gp3,DeleteOnTermination=true}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG-$1},{Key=Project,Value=$TAG},{Key=Role,Value=$1}]" \
    --query 'Instances[0].InstanceId' --output text
}

# Control first: it needs no GPU quota. SKIP_GPU=1 launches only the control node (e.g. while a
# GPU quota increase is pending); re-run without it later to add the GPU nodes.
CONTROL_ID=$(launch control "$CONTROL_TYPE" "$CONTROL_AMI" 40)
PROD_ID="" SHADOW_ID=""
if [ "${SKIP_GPU:-0}" != "1" ]; then
  PROD_ID=$(launch prod "$GPU_TYPE" "$GPU_AMI" 200)
  SHADOW_ID=$(launch shadow "$GPU_TYPE" "$GPU_AMI" 200)
fi
echo "instances: control $CONTROL_ID prod ${PROD_ID:-skipped} shadow ${SHADOW_ID:-skipped}; waiting for running"
# shellcheck disable=SC2086
aws ec2 wait instance-running --instance-ids $CONTROL_ID $PROD_ID $SHADOW_ID

ip() { [ -n "$1" ] || return 0; aws ec2 describe-instances --instance-ids "$1" --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text; }
cat > instances.env <<EOF
REGION=$REGION
PROD_ID=$PROD_ID
SHADOW_ID=$SHADOW_ID
CONTROL_ID=$CONTROL_ID
PROD_IP=$(ip "$PROD_ID")
SHADOW_IP=$(ip "$SHADOW_ID")
CONTROL_IP=$(ip "$CONTROL_ID")
EOF
cat instances.env
echo "next: ./infra/aws/secrets.sh, then ./infra/aws/deploy_code.sh (SSM agent needs ~2 min after boot)"
