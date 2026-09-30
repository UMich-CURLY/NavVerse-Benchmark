#!/usr/bin/env bash
# Override INPUT_FOLDER or pass it as $1 to point at a different dataset.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
INPUT_FOLDER="${1:-${INPUT_FOLDER:-${REPO_ROOT}/navverse_data/grscenes_commercial/models/}}"
blender --background --python decimate_usd_models_face_blender.py -- --input_folder "${INPUT_FOLDER}"