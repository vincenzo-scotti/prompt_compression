#!/usr/bin/env bash
# Run the ICSE'26 experiments (final version) for all four models.
#
# Restartable: completed runs are skipped automatically.  Results and complete
# run logs go to results/icse_26_final/<model>/; every run log is validated
# (no missing data points) after each model.
#
# Usage:
#   bash scripts/run.sh                 # normal run
#   bash scripts/run.sh --dry-run       # CPU-only smoke test (mock evaluator)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Load .env from the repo root if present (API keys, LLAMA_CPP_API_KEY, ...).
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.env"
    set +a
fi

export HF_ALLOW_CODE_EVAL=1
mkdir -p logs

DRY=0
for arg in "$@"; do [ "${arg}" = "--dry-run" ] && DRY=1; done

run_model () {
    local name="$1" config="$2" outdir="$3"
    local extra=()
    if [ "${DRY}" = 1 ]; then
        # Mock results must never mix with real ones.
        outdir="results/dry_run/$(basename "${outdir}")"
        extra=(--output-dir "${outdir}")
    fi
    python -m prompt_piper.tools.compressor --config "${config}" "${@:4}" "${extra[@]}" \
        > "logs/${name}.log" 2>&1
    python -m prompt_piper.tools.validate_run "${outdir}"/*_log \
        > "logs/${name}_validation.log" 2>&1 \
        || { echo "Log validation failed for ${name}: see logs/${name}_validation.log"; exit 1; }
}

CFG=resources/configs/icse_26
OUT=results/icse_26_final

run_model gpt    ${CFG}/experiments_gpt.yaml          ${OUT}/gpt-5.4-nano-2026-03-17 "$@"
run_model gemini ${CFG}/experiments_gemini_flash.yaml ${OUT}/gemini-3.1-flash-lite   "$@"
run_model gemma  ${CFG}/experiments_gemma4_27b.yaml   ${OUT}/gemma4-26b-a4b-it       "$@"
run_model qwen   ${CFG}/experiments_qwen35_9b.yaml    ${OUT}/qwen3.5-9b              "$@"
