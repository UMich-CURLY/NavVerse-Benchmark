#!/bin/bash
# Build a NavVerse SIF via sandbox patching so /workspace is writable.
set -euo pipefail

IMG="${1:-umich-curly/navverse:latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$SCRIPT_DIR/tmp"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$TMP_DIR" "$LOG_DIR"

IMAGE_NAME="${IMG%%:*}"
IMAGE_NAME="${IMAGE_NAME##*/}"
OCI="$TMP_DIR/${IMAGE_NAME}.oci.tar"
SANDBOX="$TMP_DIR/${IMAGE_NAME}.sandbox"
SIF="$SCRIPT_DIR/${IMAGE_NAME}.sif"

WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
WORKSPACE_MODE="${WORKSPACE_MODE:-777}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_TMP="${KEEP_TMP:-0}"
BUILD_DOCKER_IMAGE="${BUILD_DOCKER_IMAGE:-1}"

export APPTAINER_TMPDIR="$TMP_DIR"
export APPTAINER_CACHEDIR="$TMP_DIR/cache"
mkdir -p "$APPTAINER_CACHEDIR"

if command -v apptainer &>/dev/null; then
  SNG=apptainer
elif command -v singularity &>/dev/null; then
  SNG=singularity
else
  echo "Error: apptainer/singularity not found."
  exit 1
fi

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

run_root_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ sudo $*"
    return 0
  fi

  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo &>/dev/null; then
    sudo "$@"
  else
    echo "Error: sandbox patching needs sudo/root to chmod ${WORKSPACE_PATH}."
    exit 2
  fi
}

patch_workspace() {
  local target="$SANDBOX${WORKSPACE_PATH}"

  {
    echo "=== [3] Patching ${WORKSPACE_PATH} mode in sandbox ==="
    echo "    Target mode = ${WORKSPACE_MODE}"
    echo "    Sandbox path = $target"
    echo

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "--- Dry run ---"
      echo "+ chmod -R ${WORKSPACE_MODE} $target"
      echo "+ sudo chmod -R ${WORKSPACE_MODE} $target  # fallback if direct chmod fails"
      return 0
    fi

    if [[ ! -e "$target" ]]; then
      echo "Warning: $target not found in sandbox, skipping patch step."
      return 0
    fi

    echo "--- Before patch (sample) ---"
    find "$target" -maxdepth 2 -printf '%M %u:%g %p\n' | head -40 || true
    echo
    if chmod -R "${WORKSPACE_MODE}" "$target"; then
      echo "Applied chmod without sudo."
    else
      echo "Direct chmod failed, retrying with sudo."
      run_root_cmd chmod -R "${WORKSPACE_MODE}" "$target"
    fi
    echo
    echo "--- After patch (sample) ---"
    find "$target" -maxdepth 2 -printf '%M %u:%g %p\n' | head -40 || true
  } |& tee "$LOG_DIR/appt_patch_${IMAGE_NAME}.log"
}

if [[ "$BUILD_DOCKER_IMAGE" == "1" ]]; then
  echo "=== [0] Building docker image pipeline for '$IMG' ==="
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ bash $SCRIPT_DIR/build_docker.sh $IMG"
  else
    bash "$SCRIPT_DIR/build_docker.sh" "$IMG"
  fi
fi

echo "=== [1] Saving docker image '$IMG' to OCI archive ==="
docker image inspect "$IMG" >/dev/null 2>&1 || {
  echo "Error: docker image not found in local daemon: $IMG"
  exit 3
}
run_cmd rm -f "$OCI"
run_cmd rm -rf "$SANDBOX"
run_cmd docker save "$IMG" -o "$OCI"

echo "=== [2] Building sandbox from OCI archive ==="
run_cmd "$SNG" build --force --tmpdir "$TMP_DIR" --sandbox "$SANDBOX" "docker-archive://$OCI" |& tee "$LOG_DIR/appt_build_sandbox_${IMAGE_NAME}.log"

patch_workspace

echo "=== [4] Packing SIF from patched sandbox ==="
run_cmd "$SNG" build --force --tmpdir "$TMP_DIR" "$SIF" "$SANDBOX" |& tee "$LOG_DIR/appt_build_pack_${IMAGE_NAME}.log"

if [[ "$DRY_RUN" != "1" && ! -s "$SIF" ]]; then
  echo "Error: SIF not generated or empty: $SIF"
  exit 4
fi

if [[ "$KEEP_TMP" != "1" ]]; then
  echo "=== [5] Cleaning up tmp/ ==="
  run_cmd rm -rf "$TMP_DIR"
else
  echo "=== [5] Keeping tmp/ because KEEP_TMP=1 ==="
fi

echo "Done! Image is at: $SIF"
