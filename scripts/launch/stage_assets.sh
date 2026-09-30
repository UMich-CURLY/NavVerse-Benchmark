#!/usr/bin/env bash
set -euo pipefail

PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-us-west-2}"
BUCKET="${SAGEMAKER_BUCKET:?Set SAGEMAKER_BUCKET}"
PREFIX="${SAGEMAKER_ASSET_PREFIX:-navverse-assets}"

usage() {
  echo "Usage: $0 [--scenes PATH] [--checkpoint PATH] [--base-model PATH] [--uninavid PATH]"
}

command -v aws >/dev/null || { echo "aws CLI is required" >&2; exit 2; }
[[ $# -gt 0 ]] || { usage; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenes|--checkpoint|--base-model|--uninavid)
      key="${1#--}"
      key="${key//-/_}"
      path="${2:?missing path for $1}"
      [[ -e "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
      uri="s3://${BUCKET}/${PREFIX}/${key}/"
      if [[ -d "${path}" ]]; then
        aws s3 sync "${path}" "${uri}" --profile "${PROFILE}" --region "${REGION}" --only-show-errors
      else
        aws s3 cp "${path}" "${uri}" --profile "${PROFILE}" --region "${REGION}" --only-show-errors
      fi
      echo "inputs.${key}_s3_uri=${uri}"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
