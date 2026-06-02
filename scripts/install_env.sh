#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA="${EXTRA:-}"

echo "[install_env] repo_root=${repo_root}"
echo "[install_env] python=$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"

"${PYTHON_BIN}" -m pip install -e . ${EXTRA}

echo "[install_env] done"
