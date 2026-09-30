#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-internvla_n1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM_DIR="$REPO_ROOT/baselines/submodules/InternNav"
ENV_YML="$REPO_ROOT/baselines/internvla_n1/environment.yml"

if ! command -v conda &>/dev/null; then
    for candidate in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "$candidate" ]; then
            source "$candidate"
            break
        fi
    done
fi
eval "$(conda shell.bash hook)"

if ! conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env create -f "$ENV_YML" -n "$ENV_NAME"
fi
conda activate "$ENV_NAME"

mkdir -p "$REPO_ROOT/baselines/submodules"
UPSTREAM_URL="https://github.com/InternRobotics/InternNav.git"
UPSTREAM_REVISION="7a5c62400ac45b313d9b709c740b64191556a242"
if [ -e "$UPSTREAM_DIR" ]; then
    [ -d "$UPSTREAM_DIR/.git" ] || { echo "Expected a Git checkout: $UPSTREAM_DIR" >&2; exit 1; }
    [ "$(git -C "$UPSTREAM_DIR" remote get-url origin)" = "$UPSTREAM_URL" ] || { echo "Unexpected upstream origin" >&2; exit 1; }
    [ -z "$(git -C "$UPSTREAM_DIR" status --porcelain)" ] || { echo "Preserving modified upstream; setup stopped" >&2; exit 1; }
    [ "$(git -C "$UPSTREAM_DIR" rev-parse HEAD)" = "$UPSTREAM_REVISION" ] || { echo "Preserving different upstream revision; setup stopped" >&2; exit 1; }
else
    git clone --no-checkout "$UPSTREAM_URL" "$UPSTREAM_DIR"
    git -C "$UPSTREAM_DIR" checkout --detach "$UPSTREAM_REVISION"
fi
git -C "$UPSTREAM_DIR" submodule update --init --recursive

python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==${INTERNVLA_N1_TORCH_VERSION:-2.8.0}" \
    "torchvision==${INTERNVLA_N1_TORCHVISION_VERSION:-0.23.0}"
python -m pip install -r "$UPSTREAM_DIR/requirements/core_requirements.txt"
FILTERED_REQUIREMENTS="$(mktemp --suffix=.txt)"
trap 'rm -f "$FILTERED_REQUIREMENTS"' EXIT
grep -Ev '^(flash_attn|triton)==' "$UPSTREAM_DIR/requirements/internvla_n1.txt" > "$FILTERED_REQUIREMENTS"
python -m pip install --no-build-isolation -r "$FILTERED_REQUIREMENTS"
python -m pip install "numpy-quaternion>=2023.0.4"
python -m pip install "diffusers==${INTERNVLA_N1_DIFFUSERS_VERSION:-0.32.2}"
MAX_JOBS="${MAX_JOBS:-4}" python -m pip install --no-build-isolation \
    "flash-attn==${NAVVERSE_FLASH_ATTN_VERSION:-2.8.3.post1}"
python -m pip install --no-deps --no-build-isolation -e "$UPSTREAM_DIR"

DEPTH_CHECKPOINT="$REPO_ROOT/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
if [ ! -f "$DEPTH_CHECKPOINT" ]; then
    mkdir -p "$(dirname "$DEPTH_CHECKPOINT")"
    python - "$DEPTH_CHECKPOINT" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

target = Path(sys.argv[1])
downloaded = Path(
    hf_hub_download(
        repo_id="depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
        filename=target.name,
        local_dir=target.parent,
    )
)
if downloaded.resolve() != target.resolve():
    raise RuntimeError(f"Depth Anything checkpoint was downloaded to {downloaded}, expected {target}")
PY
fi

echo "InternVLA-N1 setup complete. Activate with: conda activate $ENV_NAME"
