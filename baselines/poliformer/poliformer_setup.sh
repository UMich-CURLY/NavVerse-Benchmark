#!/bin/bash
# Set up the PoliFormer baseline: conda env, submodules (PoliFormer + Detic),
# checkpoints, objaverse_houses datasets (val + train), objaverse assets, and
# nltk wordnet corpora.
#
# Usage:
#   bash baselines/poliformer/poliformer_setup.sh [env_name]
#
# Env vars:
#   POLIFORMER_DATA_DIR  - where to download objaverse data (default: $HOME/scratch/poliformer)

set -eo pipefail

ENV_NAME="${1:-poliformer}"
PY_VERSION="3.10"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/baselines"
SUBMODULES_DIR="$BASELINES_DIR/submodules"
POLIFORMER_DIR="$SUBMODULES_DIR/poliformer"
DETIC_DIR="$SUBMODULES_DIR/Detic"
DATA_DIR="${POLIFORMER_DATA_DIR:-$HOME/scratch/poliformer}"

# --- conda bootstrap ---
if ! command -v conda &>/dev/null; then
    for CANDIDATE in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "$CANDIDATE" ]; then source "$CANDIDATE"; break; fi
    done
fi
eval "$(conda shell.bash hook)"

echo "[poliformer 1/7] Creating conda env '$ENV_NAME' (python=$PY_VERSION)"
if ! conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi
conda activate "$ENV_NAME"
# Pin setuptools<70 so legacy packages (openai/CLIP) that still import
# pkg_resources can build. setuptools 80+ removed it.
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<70"

echo "[poliformer 2/7] Cloning PoliFormer submodule"
mkdir -p "$SUBMODULES_DIR"
if [ -e "$POLIFORMER_DIR" ]; then
    [ -d "$POLIFORMER_DIR/.git" ] || { echo "Expected a Git checkout: $POLIFORMER_DIR" >&2; exit 1; }
    [ "$(git -C "$POLIFORMER_DIR" remote get-url origin)" = "https://github.com/tccoin/PoliFormer.git" ] || { echo "Unexpected upstream origin" >&2; exit 1; }
    [ "$(git -C "$POLIFORMER_DIR" rev-parse HEAD)" = "fca1e911e1b4514d61a9df4d751011fbf8571901" ] || { echo "Preserving different upstream revision; setup stopped" >&2; exit 1; }
else
    git clone --no-checkout "https://github.com/tccoin/PoliFormer.git" "$POLIFORMER_DIR"
    git -C "$POLIFORMER_DIR" checkout --detach "fca1e911e1b4514d61a9df4d751011fbf8571901"
fi
git -C "$POLIFORMER_DIR" submodule update --init --recursive

python - "$POLIFORMER_DIR/training/online/online_eval.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
original = "    args = parser.parse_args()\n"
patched = "    args, _ = parser.parse_known_args()\n"
if text.count(original) == 1 and patched not in text:
    path.write_text(text.replace(original, patched, 1))
elif text.count(patched) != 1 or original in text:
    raise SystemExit(f"Unexpected PoliFormer argument parser: {path}")
PY

echo "[poliformer 3/7] Installing pinned requirements"
# Install swig first so subsequent native builds (box2d-py) can find the
# `swig` CLI on PATH.
python -m pip install swig==4.4.0
# openai/CLIP's setup.py imports pkg_resources which newer build-isolation
# setuptools (>=70) doesn't provide. Install it first with --no-build-isolation
# while setuptools+wheel are still in the env.
python -m pip install --no-build-isolation \
    git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
# Clean up the requirements file:
#   - drop the clip git line (installed above)
#   - drop the trailing duplicate/loose entries that conflict with the pinned
#     section (the ai2thor line at the bottom is the special build that we
#     install separately below, and the rest are redundant)
#   - drop the conflicting ai2thor==5.0.0 line so the special build wins
#   - drop box2d-py so we can install it separately with --no-build-isolation
#     (the in-process swig shim from the `swig` pypi package cannot be reached
#     from a fresh build-isolation env)
TMP_REQ="$(mktemp --suffix=.txt)"
awk '/^clip @/ {next}
     /^ai2thor==5\.0\.0/ {next}
     /^objathor[[:space:]]*$/ {next}
     /^ai2thor==0\+/ {next}
     /^swig[[:space:]]*$/ {next}
     /^allenact[[:space:]]*$/ {next}
     /^prior[[:space:]]*$/ {next}
     /^lightning[[:space:]]*$/ {next}
     /^wandb[[:space:]]*$/ {next}
     /^prettytable[[:space:]]*$/ {next}
     /^fire[[:space:]]*$/ {next}
     /^transformers\[torch\][[:space:]]*$/ {next}
     /^shapely[[:space:]]*$/ {next}
     /^h5py[[:space:]]*$/ {next}
     /^box2d-py==/ {next}
     {print}' "$BASELINES_DIR/poliformer/requirements_poliformer.txt" > "$TMP_REQ"
python -m pip install -r "$TMP_REQ" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://ai2thor-pypi.allenai.org \
    --extra-index-url https://miropsota.github.io/torch_packages_builder
rm -f "$TMP_REQ"
# Install box2d-py without build isolation so it uses the env's swig CLI.
python -m pip install --no-build-isolation box2d-py==2.3.8
# Install the AI2-THOR commit-pinned build separately
python -m pip install --extra-index-url https://ai2thor-pypi.allenai.org \
    "ai2thor==0+966bd7758586e05d18f6181f459c0e90ba318bec"

echo "[poliformer 4/7] Cloning Detic submodule"
if [ -e "$DETIC_DIR" ]; then
    [ -d "$DETIC_DIR/.git" ] || { echo "Expected a Git checkout: $DETIC_DIR" >&2; exit 1; }
    [ "$(git -C "$DETIC_DIR" remote get-url origin)" = "https://github.com/facebookresearch/Detic.git" ] || { echo "Unexpected upstream origin" >&2; exit 1; }
    [ "$(git -C "$DETIC_DIR" rev-parse HEAD)" = "436cda2a2347df60a7c66daca0e8c59f93dc5e79" ] || { echo "Preserving different upstream revision; setup stopped" >&2; exit 1; }
else
    git clone --no-checkout "https://github.com/facebookresearch/Detic.git" "$DETIC_DIR"
    git -C "$DETIC_DIR" checkout --detach "436cda2a2347df60a7c66daca0e8c59f93dc5e79"
fi
git -C "$DETIC_DIR" submodule update --init --recursive
python -m pip install -r "$DETIC_DIR/requirements.txt"

echo "[poliformer 5/7] Downloading Detic weights"
mkdir -p "$DETIC_DIR/models"
DETIC_CKPT="$DETIC_DIR/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
if [ ! -f "$DETIC_CKPT" ]; then
    wget -O "$DETIC_CKPT" \
        https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
fi

echo "[poliformer 6/7] Downloading PoliFormer ckpt + objaverse data into $DATA_DIR"
mkdir -p "$DATA_DIR"
# PoliFormer trained ckpt (script lives inside the cloned submodule)
python "$POLIFORMER_DIR/scripts/download_trained_ckpt.py" \
    --save_dir "$POLIFORMER_DIR/ckpt"
# Objaverse houses (val + train) and assets — large (~100 GB total)
cd "$REPO_ROOT"
python -m baselines.submodules.poliformer.scripts.download_objaverse_houses \
    --save_dir "$DATA_DIR/objaverse_houses" --subset val
python -m baselines.submodules.poliformer.scripts.download_objaverse_houses \
    --save_dir "$DATA_DIR/objaverse_houses" --subset train
python -m objathor.dataset.download_annotations --version 2023_07_28 --path "$DATA_DIR/objaverse_assets"
python -m objathor.dataset.download_assets       --version 2023_07_28 --path "$DATA_DIR/objaverse_assets"

echo "[poliformer 7/7] Downloading nltk corpora"
python -c "import nltk; nltk.download('wordnet'); nltk.download('wordnet2022')"

echo "[poliformer] Done. Activate with: conda activate $ENV_NAME"
echo "[poliformer] Data dir: $DATA_DIR"
