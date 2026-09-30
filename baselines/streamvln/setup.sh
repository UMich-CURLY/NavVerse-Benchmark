#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-streamvln}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM_DIR="$REPO_ROOT/baselines/submodules/StreamVLN"
ENV_YML="$REPO_ROOT/baselines/streamvln/environment.yml"

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
UPSTREAM_URL="https://github.com/InternRobotics/StreamVLN.git"
UPSTREAM_REVISION="e48f6ff7e9201d93aae003e8f64b04c00cec13bc"
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
    "torch==${STREAMVLN_TORCH_VERSION:-2.8.0}" \
    "torchvision==${STREAMVLN_TORCHVISION_VERSION:-0.23.0}"
FILTERED_REQUIREMENTS="$(mktemp --suffix=.txt)"
trap 'rm -f "$FILTERED_REQUIREMENTS"' EXIT
grep -Ev '^(av|wavedrom|torch|torchvision|torch-scatter|triton)==|^nvidia-' "$UPSTREAM_DIR/requirements.txt" > "$FILTERED_REQUIREMENTS"
python -m pip install --no-build-isolation -r "$FILTERED_REQUIREMENTS"
python -m pip install "numpy-quaternion>=2023.0.4"
MAX_JOBS="${MAX_JOBS:-4}" python -m pip install --no-build-isolation \
    "flash-attn==${NAVVERSE_FLASH_ATTN_VERSION:-2.8.3.post1}"
python -m pip install "python-multipart>=0.0.9,<0.1"

echo "StreamVLN setup complete. Activate with: conda activate $ENV_NAME"
