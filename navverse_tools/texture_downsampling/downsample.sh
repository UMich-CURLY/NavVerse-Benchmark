#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/navverse_data/grscenes_commercial}"
DEST_ROOT="${DEST_ROOT:-${REPO_ROOT}/navverse_data/grscenes_commercial_downsample_texture_ae_$(date +%Y%m%d_%H%M%S)}"
USD_GLOB="${USD_GLOB:-scenes/scene*/start_result_navigation.usd}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/texture_downsample_$(basename "${DEST_ROOT}")_$(date +%Y%m%d_%H%M%S)}"
COMPRESSONATOR="${COMPRESSONATOR:-}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_grscene_ae_downsample.py"
  --source_root "${SOURCE_ROOT}"
  --dest_root "${DEST_ROOT}"
  --repo_root "${REPO_ROOT}"
  --log_dir "${LOG_DIR}"
  --usd_glob "${USD_GLOB}"
)
if [[ -n "${COMPRESSONATOR}" ]]; then
  cmd+=(--compressonator "${COMPRESSONATOR}")
fi
cmd+=("$@")

mkdir -p "${LOG_DIR}"

echo "[INFO] SOURCE_ROOT=${SOURCE_ROOT}"
echo "[INFO] DEST_ROOT=${DEST_ROOT}"
echo "[INFO] USD_GLOB=${USD_GLOB}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
printf "[CMD]"
printf " %q" "${cmd[@]}"
printf "\n"

"${cmd[@]}"
