#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TOOLS_DIR="${SCRIPT_DIR}/.local_tools"
NVTT_DIR="${TOOLS_DIR}/nvtt_apt"
COMPRESSONATOR_DIR="${TOOLS_DIR}/compressonator"
COMPRESSONATOR_VERSION="4.5.52"
COMPRESSONATOR_TAG="V${COMPRESSONATOR_VERSION}"
COMPRESSONATOR_TAR="compressonatorcli-${COMPRESSONATOR_VERSION}-Linux.tar.gz"
COMPRESSONATOR_URL="https://github.com/GPUOpen-Tools/compressonator/releases/download/${COMPRESSONATOR_TAG}/${COMPRESSONATOR_TAR}"

mkdir -p "${TOOLS_DIR}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

install_nvtt_user() {
  mkdir -p "${NVTT_DIR}"
  cd "${NVTT_DIR}"
  if [ ! -x root/usr/bin/nvcompress ]; then
    apt-get download libnvtt-bin libnvtt2
    for deb in ./*.deb; do
      dpkg-deb -x "${deb}" root
    done
  fi
  cat > "${NVTT_DIR}/env.sh" <<EOF
export PATH="${NVTT_DIR}/root/usr/bin:\$PATH"
export LD_LIBRARY_PATH="${NVTT_DIR}/root/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}"
EOF
}

install_compressonator_user() {
  mkdir -p "${COMPRESSONATOR_DIR}"
  cd "${COMPRESSONATOR_DIR}"
  if [ ! -x "compressonatorcli-${COMPRESSONATOR_VERSION}-Linux/compressonatorcli" ]; then
    wget -q -O "${COMPRESSONATOR_TAR}" "${COMPRESSONATOR_URL}"
    tar -xzf "${COMPRESSONATOR_TAR}"
  fi
  cat > "${COMPRESSONATOR_DIR}/env.sh" <<EOF
export COMPRESSONATOR_ROOT="${COMPRESSONATOR_DIR}/compressonatorcli-${COMPRESSONATOR_VERSION}-Linux"
export PATH="\${COMPRESSONATOR_ROOT}:\$PATH"
export LD_LIBRARY_PATH="\${COMPRESSONATOR_ROOT}/pkglibs:\${LD_LIBRARY_PATH:-}"
EOF
}

install_system_packages_if_root() {
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update
    apt-get install -y libnvtt-bin wget ca-certificates
  fi
}

install_system_compressonator_if_root() {
  if [ "$(id -u)" -eq 0 ]; then
    mkdir -p "${COMPRESSONATOR_DIR}/deb"
    cd "${COMPRESSONATOR_DIR}/deb"
    wget -q -O compressonatorcli_${COMPRESSONATOR_VERSION}_amd64.deb "https://github.com/GPUOpen-Tools/compressonator/releases/download/${COMPRESSONATOR_TAG}/compressonatorcli_${COMPRESSONATOR_VERSION}_amd64.deb"
    apt-get install -y ./compressonatorcli_${COMPRESSONATOR_VERSION}_amd64.deb || true
    apt-get install -f -y
  fi
}

install_system_packages_if_root
install_system_compressonator_if_root
# nvtt is optional (not used by the BC7 pipeline); don't fail setup if it can't be fetched.
install_nvtt_user || echo "[WARN] nvtt install skipped (apt-get download unavailable); continuing without nvtt." >&2
install_compressonator_user

# shellcheck source=/dev/null
[ -f "${NVTT_DIR}/env.sh" ] && source "${NVTT_DIR}/env.sh"
# shellcheck source=/dev/null
source "${COMPRESSONATOR_DIR}/env.sh"

nvcompress -help >/dev/null 2>&1 || true
compressonatorcli 2>&1 | head -5 || true

echo "Texture tools are ready."
echo "For this shell, run:"
[ -f "${NVTT_DIR}/env.sh" ] && echo "  source ${NVTT_DIR}/env.sh"
echo "  source ${COMPRESSONATOR_DIR}/env.sh"
