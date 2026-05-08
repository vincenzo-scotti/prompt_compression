#!/usr/bin/env bash
# Run ICSE'26 preliminary optimisation experiments (Qwen/Qwen3.5-9B).
#
# Restartable: any run whose output JSON already exists in
#   ./results/icse_26/qwen3.5-9b/
# is automatically skipped.  Kill and re-run freely.
#
# Usage:
#   bash scripts/preliminary_run.sh                 # normal run
#   bash scripts/preliminary_run.sh --dry-run       # CPU-only smoke test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load .env from the repo root if present (exports all variables).
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.env"
    set +a
fi

python -m prompt_piper.tools.compressor \
    --config "${REPO_ROOT}/resources/configs/icse_26/preliminary_experiment.yaml" \
    "$@"
