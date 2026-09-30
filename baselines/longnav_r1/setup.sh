#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
upstream="${root}/upstream"
revision=927b4a4c972e4cce5ea7c4c8cb0fd12a5ce4cbf0
url=https://github.com/UMich-CURLY/LongNav-R1.git
python="${LONGNAV_SETUP_PYTHON:-python3}"

if [[ -e "${upstream}" ]]; then
    [[ -d "${upstream}/.git" ]] || { echo "Existing upstream is not a Git checkout" >&2; exit 1; }
    [[ "$(git -C "${upstream}" remote get-url origin)" == "${url}" ]] || { echo "Unexpected upstream origin" >&2; exit 1; }
    [[ -z "$(git -C "${upstream}" status --porcelain -- . ':(exclude)src/longnav.egg-info')" ]] || { echo "Preserving modified upstream; setup stopped" >&2; exit 1; }
    [[ "$(git -C "${upstream}" rev-parse HEAD)" == "${revision}" ]] || { echo "Preserving a different upstream revision; setup stopped" >&2; exit 1; }
else
    git clone --no-checkout "${url}" "${upstream}"
    git -C "${upstream}" checkout --detach "${revision}"
fi

if [[ ! -x "${root}/.venv/bin/python" ]]; then
    "${python}" -m venv "${root}/.venv"
fi
python="${root}/.venv/bin/python"
"${python}" -m pip install 'pip>=24' 'setuptools>=77' wheel
"${python}" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"${python}" -m pip install -e "${upstream}" 'transformers>=4.57,<5' accelerate scipy qwen-vl-utils python-multipart requests
"${python}" -m pip check
"${python}" - <<'PY'
from pathlib import Path
import importlib.util
from longnav.config_schema import RolloutConfig, VLMConfig, VLMTrainingConfig
from longnav.utils.rollout_core import RLWorker

root = Path(next(iter(importlib.util.find_spec('longnav').submodule_search_locations)))
assert (root / 'conf/prompts/objectnav_prompt.txt').is_file()
print('LongNav-R1 imports and prompt verified:', root)
PY
echo "Ready: ${python} ${root}/serve_longnav.py"
