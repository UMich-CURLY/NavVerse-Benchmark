#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/real-world-exp-server-bridge.service.in"
UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
UNIT_PATH="${UNIT_DIR}/real-world-exp-server-bridge.service"
START_SERVICE=true

if [[ "${1:-}" == "--no-start" ]]; then
    START_SERVICE=false
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--no-start]" >&2
    exit 2
fi

for candidate in \
    "${HOME}/miniconda3/envs/navverse/bin/python" \
    "${HOME}/anaconda3/envs/navverse/bin/python"; do
    if [[ -x "${candidate}" ]]; then
        PYTHON="${candidate}"
        break
    fi
done
if [[ -z "${PYTHON:-}" ]]; then
    echo "Could not find the navverse Python environment." >&2
    exit 1
fi

mkdir -p "${UNIT_DIR}"
sed \
    -e "s|@REPO_ROOT@|${REPO_ROOT}|g" \
    -e "s|@PYTHON@|${PYTHON}|g" \
    "${TEMPLATE}" > "${UNIT_PATH}"
systemctl --user daemon-reload

if [[ "${START_SERVICE}" == true ]]; then
    systemctl --user enable --now real-world-exp-server-bridge.service
    systemctl --user --no-pager status real-world-exp-server-bridge.service
else
    echo "Installed ${UNIT_PATH}; the service was not started or enabled."
fi
