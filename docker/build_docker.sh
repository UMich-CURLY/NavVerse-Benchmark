#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

IMAGE="${1:-navverse-release:local}"
if [[ "$IMAGE" != *:* ]]; then
  IMAGE="${IMAGE}:latest"
fi

REPO="${IMAGE%:*}"
TAG="${IMAGE##*:}"
BASE_IMAGE="${REPO}:customized-isaaclab"
CACHED_IMAGE="${REPO}:cached"
CONTAINER_NAME="navverse-warmup-$$"

if [[ "$TAG" == "customized-isaaclab" || "$TAG" == "cached" ]]; then
  echo "Error: final tag cannot be customized-isaaclab or cached."
  exit 1
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
trap 'docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true' EXIT

echo "=== [1/4] Building base Isaac Lab stage: $BASE_IMAGE ==="
docker build \
  --target customized-isaaclab \
  -t "$BASE_IMAGE" \
  -f docker/Dockerfile \
  .

echo "=== [2/4] Running Isaac Sim warmup in container ==="
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run \
  --runtime nvidia \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --shm-size=8g \
  --name "$CONTAINER_NAME" \
  -e ACCEPT_EULA=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -v "$PWD/docker/isaacsim_pip_warmup.sh:/tmp/isaacsim_pip_warmup.sh:ro" \
  --entrypoint /bin/bash \
  "$BASE_IMAGE" \
  -lc "bash /tmp/isaacsim_pip_warmup.sh"

echo "=== [3/4] Capturing warmed image: $CACHED_IMAGE ==="
docker commit "$CONTAINER_NAME" "$CACHED_IMAGE"

echo "=== [4/4] Building final image from warmed cache: $IMAGE ==="
docker build \
  --build-arg FINAL_BASE_IMAGE="$CACHED_IMAGE" \
  -t "$IMAGE" \
  -f docker/Dockerfile \
  .

echo
echo "Built $IMAGE"
