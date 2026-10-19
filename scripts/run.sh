#!/usr/bin/env bash
# Run the ICSE'26 experiments (final version) for all four models.
#
# Restartable: completed runs are skipped automatically.  Results and complete
# run logs go to results/icse_26_final/<model>/; every run log is validated
# (no missing data points) after each model.
#
# Usage:
#   bash scripts/run.sh                       # all four models
#   bash scripts/run.sh gemma qwen            # only the listed models (gpt, gemini, gemma, qwen)
#   bash scripts/run.sh gemma qwen --dry-run  # CPU-only smoke test (mock evaluator)
# Any other argument (e.g. --n-gen 5) is passed to the compressor.
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

# Keep the machine awake: a suspend drops open connections and stalls runs
# that last many hours (seen in the smoke test on 2026-10-08).
if [ -z "${PP_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
    export PP_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --who=prompt_piper \
        --why="PromptPiper experiments running" bash "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")" "$@"
fi

DRY=0
MODELS=()
OPTS=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY=1; OPTS+=("${arg}") ;;
        gpt|gemini|gemma|qwen) MODELS+=("${arg}") ;;
        *) OPTS+=("${arg}") ;;   # compressor options and their values
    esac
done
[ "${#MODELS[@]}" -eq 0 ] && MODELS=(gpt gemini gemma qwen)

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

for model in "${MODELS[@]}"; do
    echo "$(date '+%F %T')  starting ${model}"
    case "${model}" in
        gpt)    run_model gpt    ${CFG}/experiments_gpt.yaml          ${OUT}/gpt-5.4-nano-2026-03-17 "${OPTS[@]}" ;;
        gemini) run_model gemini ${CFG}/experiments_gemini_flash.yaml ${OUT}/gemini-3.8-flash   "${OPTS[@]}" ;;
        gemma)  run_model gemma  ${CFG}/experiments_gemma4_27b.yaml   ${OUT}/gemma4-26b-a4b-it       "${OPTS[@]}" ;;
        qwen)   run_model qwen   ${CFG}/experiments_qwen35_9b.yaml    ${OUT}/qwen3.5-9b              "${OPTS[@]}" ;;
    esac
    echo "$(date '+%F %T')  finished ${model} (log validation passed)"
done
