#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
  echo "usage: GPT_CODEX_HOME=/path/to/private/home $0 command [args ...]" >&2
  exit 2
fi

export CODEX_HOME="${GPT_CODEX_HOME:-${HOME:?HOME must be set}/.codex-navverse-astra}"
mkdir -p "$CODEX_HOME"
chmod 700 "$CODEX_HOME"

config_file="$CODEX_HOME/config.toml"
if [[ ! -e "$config_file" ]]; then
  (umask 077; printf 'cli_auth_credentials_store = "file"\n' > "$config_file")
elif ! grep -Eq '^[[:space:]]*cli_auth_credentials_store[[:space:]]*=[[:space:]]*"file"([[:space:]]*(#.*)?)?$' "$config_file"; then
  echo "Set cli_auth_credentials_store = \"file\" in $config_file, or choose a fresh GPT_CODEX_HOME." >&2
  exit 2
fi

exec "$@"
