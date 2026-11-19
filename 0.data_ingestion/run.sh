#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Do not cd: an explicit relative --config is relative to the caller's directory.
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}"
exec "${PYTHON:-python}" -m utils.pedcp_data_integration "$@"
