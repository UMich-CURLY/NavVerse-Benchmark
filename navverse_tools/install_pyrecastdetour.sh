#!/usr/bin/env bash
set -euo pipefail

FILENAME="PyRecastDetour.cpython-311-x86_64-linux-gnu.so"
REVISION="e99b8a4a841785a04bb87a3751f4c31c8b766d80"
SHA256="1ce37c6fa6721d12e422e7021e5386f5dbab7ad52485c10264668ed683df252d"
INSTALL_DIR="$(python - <<'PY'
import platform
import site
import sys

if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11):
    raise SystemExit("The published PyRecastDetour binary requires CPython 3.11")
if sys.platform != "linux" or platform.machine() != "x86_64":
    raise SystemExit("The published PyRecastDetour binary requires Linux x86_64")
print(site.getsitepackages()[0])
PY
)"
DOWNLOAD_DIR="$(mktemp -d)"
trap 'rm -rf "$DOWNLOAD_DIR"' EXIT
curl --fail --location --silent --show-error \
    "https://raw.githubusercontent.com/tccoin/PyRecastDetour-Sources/$REVISION/dist/$FILENAME" \
    --output "$DOWNLOAD_DIR/$FILENAME"
echo "$SHA256  $DOWNLOAD_DIR/$FILENAME" | sha256sum --check --status
install -m 0644 "$DOWNLOAD_DIR/$FILENAME" "$INSTALL_DIR/$FILENAME"
python -c 'import PyRecastDetour; print("PyRecastDetour import OK")'
