#!/usr/bin/env bash
set -euo pipefail

PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-us-west-2}"
ACCOUNT="${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
REPOSITORY="${ECR_REPOSITORY:-navverse-benchmark}"
TAG="${1:-$(date -u +%Y%m%d-%H%M%S)}"
IMAGE_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY}:${TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

command -v aws >/dev/null || { echo "aws CLI is required" >&2; exit 2; }
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }

aws ecr describe-repositories --profile "${PROFILE}" --region "${REGION}" \
  --repository-names "${REPOSITORY}" >/dev/null 2>&1 || \
aws ecr create-repository --profile "${PROFILE}" --region "${REGION}" \
  --repository-name "${REPOSITORY}" --image-scanning-configuration scanOnPush=true >/dev/null

aws ecr get-login-password --profile "${PROFILE}" --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

"${REPO_ROOT}/docker/build_docker.sh" "${IMAGE_URI}"
docker push "${IMAGE_URI}"
DIGEST="$(aws ecr describe-images --profile "${PROFILE}" --region "${REGION}" \
  --repository-name "${REPOSITORY}" --image-ids imageTag="${TAG}" \
  --query 'imageDetails[0].imageDigest' --output text)"
echo "ECR_IMAGE_URI=${IMAGE_URI}"
echo "ECR_IMAGE_DIGEST=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY}@${DIGEST}"
