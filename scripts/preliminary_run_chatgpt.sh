#!/usr/bin/env bash
# Run ICSE'26 preliminary optimisation experiments (gpt-4o via OpenAI API).
#
# Restartable: any run whose output JSON already exists in
#   ./results/icse_26/gpt-4o/
# is automatically skipped.  Kill and re-run freely.
#
# Requires: OPENAI_API_KEY set in the environment (or .env file sourced).
#
# Usage:
#   bash scripts/preliminary_run_chatgpt.sh                 # normal run
#   bash scripts/preliminary_run_chatgpt.sh --dry-run       # CPU-only smoke test
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

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Add it to .env or export it manually." >&2
    exit 1
fi

python -m prompt_piper.tools.compressor \
    --config "${REPO_ROOT}/resources/configs/icse_26/preliminary_experiment_chatgpt.yaml" \
    "$@"