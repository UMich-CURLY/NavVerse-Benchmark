#!/bin/bash
set -euo pipefail

# Adapted from the upstream Isaac Sim warmup flow for pip-installed Isaac Sim.
# Reference: https://github.com/isaac-sim/IsaacSim/blob/main/source/scripts/warmup.sh

CONDA_ENV="${CONDA_ENV:-navverse}"
TASKING_THREAD_CNT="${TASKING_THREAD_CNT:-$(( $(nproc --all) / 2 ))}"

if [[ "$TASKING_THREAD_CNT" -lt 1 ]]; then
  TASKING_THREAD_CNT=1
fi

export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export OMNI_KIT_ALLOW_ROOT="${OMNI_KIT_ALLOW_ROOT:-1}"

run_in_env() {
  conda run --no-capture-output -n "$CONDA_ENV" "$@"
}

echo "=== Isaac Sim warmup: Python app ==="
run_in_env python - <<'PY'
from isaacsim import SimulationApp
import omni

simulation_app = SimulationApp({"headless": True})

# Match the upstream hello_world warmup behavior more closely so Kit has
# time to initialize extensions and populate caches before shutdown.
for _ in range(100):
    simulation_app.update()
    if _ % 10 == 0:
        print(f"Warmup iteration {_} of 100", flush=True)

omni.kit.app.get_app().print_and_log("Warmup hello world complete.")
simulation_app.close()
PY
echo "=== Isaac Sim Python warmup complete ==="

echo "=== Isaac Sim warmup: full experience ==="
FULL_KIT_ARGS=(
  isaacsim.exp.full.kit
  --allow-root
  --no-window
  --/persistent/renderer/startupMessageDisplayed=true
  --/app/settings/persistent=0
  --/app/settings/loadUserConfig=0
  --/structuredLog/enable=0
  --/app/hangDetector/enabled=0
  --/crashreporter/skipOldDumpUpload=1
  --/app/content/emptyStageOnStart=1
  --/rtx/materialDb/syncLoads=1
  --/omni.kit.plugin/syncUsdLoads=1
  --/rtx/hydra/materialSyncLoads=1
  --/app/asyncRendering=0
  --/app/quitAfter=1000
  --/app/fastShutdown=1
  --/app/file/ignoreUnsavedOnExit=1
  --/app/warmupMode=1
  --/exts/omni.kit.registry.nucleus/registries/0/name=0
  "--/plugins/carb.tasking.plugin/threadCount=${TASKING_THREAD_CNT}"
)
run_in_env isaacsim "${FULL_KIT_ARGS[@]}"

echo "=== Isaac Sim warmup complete ==="
