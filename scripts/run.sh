#!/usr/bin/env bash
# Run ICSE'26 full experiments (all models defined in experiments.yaml).
#
# Restartable: completed runs are skipped automatically.
#
# Usage:
#   bash scripts/run.sh                 # normal run
#   bash scripts/run.sh --dry-run       # CPU-only smoke test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python -m prompt_piper.tools.compressor \
    --config "${REPO_ROOT}/resources/configs/icse_26/experiments.yaml" \
    "$@"
