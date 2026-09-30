#!/bin/bash
# Internal implementation for `navverse-run setup`.
# Direct use remains supported for bootstrapping the launcher.

set -e

ENV_NAME="${1:-navverse}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="3.11"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
ISAACLAB_SPEC="isaaclab[isaacsim,all]==2.3.2.post1"
NVIDIA_INDEX_URL="https://pypi.nvidia.com"

# --- 0. Conda Initialization (Anti-double-sourcing) ---
# If conda is not a command/function, we need to source conda.sh
if ! command -v conda &> /dev/null; then
    CONDA_SH=""
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
    fi

    if [ -n "$CONDA_SH" ]; then
        source "$CONDA_SH"
        eval "$(conda shell.bash hook)" 
    else
        echo "[ERROR] Could not find conda.sh." >&2
        exit 1
    fi
fi
if ! type conda | grep -q "function" &> /dev/null; then
    eval "$(conda shell.bash hook)"
fi

echo "[1/7] Ensuring conda env '$ENV_NAME' exists (python=$PYTHON_VERSION)"
if ! conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

echo "[2/7] Installing torch/cu128"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url "$TORCH_INDEX_URL"

echo "[3/7] Installing Isaac Lab / Isaac Sim python stack"
python -m pip install "$ISAACLAB_SPEC" --extra-index-url "$NVIDIA_INDEX_URL"

echo "[4/7] Installing NavVerse package via setup.py"
cd "$REPO_ROOT"
python -m pip install -e .

echo "[5/7] Installing PyRecastDetour binary"
bash "$REPO_ROOT/navverse_tools/install_pyrecastdetour.sh"

echo "[6/7] Installing dataset and USD tooling"
conda install -n "$ENV_NAME" -c conda-forge -y \
    huggingface_hub \
    pillow
# hf_transfer, hf-xet, and usd-core are installed via pip into the same env. usd-core
# provides the pxr bindings used by pure USD composition tools.
python -m pip install hf_transfer hf-xet usd-core
python -m pip install -r "$REPO_ROOT/scripts/requirements_isaac_compat.txt"
python -m pip check

echo "[7/7] Verifying imports"
echo "Yes" | python -c "from isaaclab.app import AppLauncher"
python - <<'PY'
import torch
import yaml
import cv2
import ray
import open3d
import PyRecastDetour
from pxr import Usd
from isaaclab.app import AppLauncher
import argparse
args = argparse.Namespace()
args.kit_args = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=fatal"
app_launcher = AppLauncher(args)
import navverse.episode
import navverse.navmesh
print('torch', torch.__version__)
print('cv2', cv2.__version__)
print('ray', ray.__version__)
print('open3d', open3d.__version__)
print('PyRecastDetour OK')
print('isaaclab AppLauncher OK')
print('navverse imports OK')
PY

echo "[setup_navverse] Done. Activate with: conda activate $ENV_NAME"
