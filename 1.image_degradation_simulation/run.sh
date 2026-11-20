#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Resolve explicit relative --config paths from the caller, like ingestion.
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}"
exec "${PYTHON:-python}" -u -m utils.degradation "$@"
