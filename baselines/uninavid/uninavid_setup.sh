#!/bin/bash
# Set up the Uni-NaVid baseline without sys.path shims.
#
# Usage:
#   bash baselines/uninavid/uninavid_setup.sh [env_name]

set -euo pipefail

ENV_NAME="${1:-uninavid}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/baselines"
UPSTREAM_DIR="$BASELINES_DIR/submodules/Uni-NaVid"
MODEL_ZOO="$UPSTREAM_DIR/model_zoo"
LEGACY_MODEL_ZOO="$BASELINES_DIR/utils/uninavid/model_zoo"
ENV_YML="$BASELINES_DIR/uninavid/environment_uninavid.yml"
MODEL_NAME="uninavid-7b-full-224-video-fps-1-grid-2"

if ! command -v conda &>/dev/null; then
    for CANDIDATE in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "$CANDIDATE" ]; then
            source "$CANDIDATE"
            break
        fi
    done
fi
eval "$(conda shell.bash hook)"

echo "[uninavid 1/6] Ensuring conda env '$ENV_NAME'"
if ! conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    TMP_YML="$(mktemp --suffix=.yml)"
    grep -v '^prefix:' "$ENV_YML" > "$TMP_YML"
    PIP_USE_PEP517=1 conda env create -f "$TMP_YML" -n "$ENV_NAME"
    rm -f "$TMP_YML"
fi
conda activate "$ENV_NAME"

echo "[uninavid 2/6] Ensuring upstream Uni-NaVid checkout"
mkdir -p "$BASELINES_DIR/submodules"
UPSTREAM_URL="https://github.com/CharlesChang012/Uni-NaVid.git"
UPSTREAM_REVISION="64444eba4d4503d5b08f080673e87229f7fe7f25"
if [ -e "$UPSTREAM_DIR" ]; then
    [ -d "$UPSTREAM_DIR/.git" ] || { echo "Expected a Git checkout: $UPSTREAM_DIR" >&2; exit 1; }
    [ "$(git -C "$UPSTREAM_DIR" remote get-url origin)" = "$UPSTREAM_URL" ] || { echo "Unexpected upstream origin" >&2; exit 1; }
    [ "$(git -C "$UPSTREAM_DIR" rev-parse HEAD)" = "$UPSTREAM_REVISION" ] || { echo "Preserving different upstream revision; setup stopped" >&2; exit 1; }
else
    git clone --no-checkout "$UPSTREAM_URL" "$UPSTREAM_DIR"
    git -C "$UPSTREAM_DIR" checkout --detach "$UPSTREAM_REVISION"
fi

echo "[uninavid 3/6] Installing editable packages"
python -m pip install --upgrade \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.0" \
    "torchvision==0.22.0" \
    "torchaudio==2.7.0"
python -m pip uninstall -y bitsandbytes || true
python -m pip install --no-deps \
    "sentencepiece>=0.1.99" \
    "accelerate>=0.21.0" \
    "huggingface_hub>=0.16.0" \
    "einops>=0.6.1" \
    "einops-exts>=0.0.4" \
    "imageio-ffmpeg>=0.6.0" \
    "timm==0.6.13" \
    "shortuuid>=1.0.11"
python -m pip install --no-build-isolation -e "$REPO_ROOT"
python -m pip install --no-deps --no-build-isolation -e "$UPSTREAM_DIR"

echo "[uninavid 4/6] Applying Uni-NaVid runtime compatibility patches"
python - <<PY
from pathlib import Path

llava_path = Path("$UPSTREAM_DIR/uninavid/model/language_model/llava_llama_vid.py")
if llava_path.exists():
    text = llava_path.read_text()
    robust_register = '''try:
    AutoConfig.register("llava", LlavaConfig, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register("llava", LlavaConfig)
    except ValueError:
        pass

try:
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaAttForCausalLM, exist_ok=True)
except TypeError:
    try:
        AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaAttForCausalLM)
    except ValueError:
        pass'''
    for snippet in [
        'AutoConfig.register("llava", LlavaConfig)\\nAutoModelForCausalLM.register(LlavaConfig, LlavaLlamaAttForCausalLM)',
        'AutoConfig.register("llava", LlavaConfig, exist_ok=True)\\nAutoModelForCausalLM.register(LlavaConfig, LlavaLlamaAttForCausalLM, exist_ok=True)',
    ]:
        text = text.replace(snippet, robust_register)
    llava_path.write_text(text)

arch_path = Path("$UPSTREAM_DIR/uninavid/model/uninavid_arch.py")
if arch_path.exists():
    text = arch_path.read_text()
    robust_attention_mask = '''                batch_size = input_ids.shape[0]
                mask_dtype = attention_mask.dtype if attention_mask is not None else torch.bool
                mask_device = attention_mask.device if attention_mask is not None else input_ids.device
                past_sequence_length = 0
                if hasattr(past_key_values, "get_seq_length"):
                    try:
                        past_sequence_length = int(past_key_values.get_seq_length() or 0)
                    except Exception:
                        past_sequence_length = 0
                if past_sequence_length == 0:
                    for layer_past in reversed(past_key_values):
                        if layer_past is None:
                            continue
                        past_states = layer_past if isinstance(layer_past, (tuple, list)) else (layer_past,)
                        for past_state in reversed(past_states):
                            if past_state is not None and hasattr(past_state, "shape"):
                                past_sequence_length = past_state.shape[-2]
                                break
                        if past_sequence_length:
                            break
                attention_mask = torch.ones((batch_size, past_sequence_length + 1),
                                            dtype=mask_dtype, device=mask_device)'''
    text = text.replace(
        '''                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)''',
        robust_attention_mask,
    )
    text = text.replace(
        '''                batch_size = input_ids.shape[0]
                mask_dtype = attention_mask.dtype if attention_mask is not None else torch.bool
                mask_device = attention_mask.device if attention_mask is not None else input_ids.device
                attention_mask = torch.ones((batch_size, past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=mask_dtype, device=mask_device)''',
        robust_attention_mask,
    )
    arch_path.write_text(text)
PY

echo "[uninavid 5/6] Ensuring model_zoo under $MODEL_ZOO"
mkdir -p "$MODEL_ZOO"
if [ ! -f "$MODEL_ZOO/eva_vit_g.pth" ] && [ -f "$LEGACY_MODEL_ZOO/eva_vit_g.pth" ]; then
    mv "$LEGACY_MODEL_ZOO/eva_vit_g.pth" "$MODEL_ZOO/eva_vit_g.pth"
fi
if [ ! -d "$MODEL_ZOO/$MODEL_NAME" ] && [ -d "$LEGACY_MODEL_ZOO/$MODEL_NAME" ]; then
    mv "$LEGACY_MODEL_ZOO/$MODEL_NAME" "$MODEL_ZOO/$MODEL_NAME"
fi

if [ ! -f "$MODEL_ZOO/eva_vit_g.pth" ]; then
    wget -O "$MODEL_ZOO/eva_vit_g.pth" \
        https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
fi

python - <<PY
import os
from huggingface_hub import snapshot_download

target = "$MODEL_ZOO/$MODEL_NAME"
if not os.path.exists(os.path.join(target, "pytorch_model.bin.index.json")):
    os.makedirs(target, exist_ok=True)
    tmp_dir = "$MODEL_ZOO/_hf_tmp"
    snapshot_download(
        repo_id="Jzzhang/Uni-NaVid",
        allow_patterns=["$MODEL_NAME/*"],
        local_dir=tmp_dir,
        local_dir_use_symlinks=False,
    )
    src = os.path.join(tmp_dir, "$MODEL_NAME")
    if os.path.isdir(src):
        import shutil
        for name in os.listdir(src):
            s = os.path.join(src, name)
            d = os.path.join(target, name)
            if not os.path.exists(d):
                shutil.move(s, d)
        shutil.rmtree(tmp_dir, ignore_errors=True)
PY

echo "[uninavid 6/6] Writing checkpoint paths"
python - <<PY
import json
from pathlib import Path

config_path = Path("$MODEL_ZOO/$MODEL_NAME/config.json")
if config_path.exists():
    config = json.loads(config_path.read_text())
    config["image_processor"] = str(Path("$UPSTREAM_DIR/uninavid/processor/clip-patch14-224"))
    config["mm_vision_tower"] = str(Path("$MODEL_ZOO/eva_vit_g.pth"))
    config_path.write_text(json.dumps(config, indent=2) + "\\n")
else:
    raise FileNotFoundError(config_path)
PY

echo "[uninavid] Done. Activate with: conda activate $ENV_NAME"
