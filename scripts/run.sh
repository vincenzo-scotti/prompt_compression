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

# Load .env from the repo root if present (exports all variables).
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.env"
    set +a
fi

HF_ALLOW_CODE_EVAL=1

mkdir -p logs

# GPT
python -m prompt_piper.tools.compressor \
    --config resources/configs/icse_26/experiments_gpt.yaml \
    > logs/gpt.log 2>&1
python -m prompt_piper.tools.detailed_eval \
    --config resources/configs/icse_26/experiments_gpt.yaml \
    --results-dir results/icse_26/gpt-5.4-nano-2026-03-17 \
    > logs/gpt_details.log 2>&1
# Gemini
python -m prompt_piper.tools.compressor \
    --config resources/configs/icse_26/experiments_gemini_flash.yaml \
    > logs/gemini.log 2>&1
python -m prompt_piper.tools.detailed_eval \
    --config resources/configs/icse_26/experiments_gemini_flash.yaml \
    --results-dir results/icse_26/gemini-3-flash-preview \
    > logs/gemini_details.log 2>&1
# Gemma
python -m prompt_piper.tools.compressor \
    --config resources/configs/icse_26/experiments_gemma4_27b.yaml \
    > logs/gemma.log 2>&1
python -m prompt_piper.tools.detailed_eval \
    --config resources/configs/icse_26/experiments_gemma4_27b.yaml \
    --results-dir results/icse_26/gemma-4-26b-a4b-it \
    > logs/gemma_details.log 2>&1
# Qwen
python -m prompt_piper.tools.compressor \
    --config resources/configs/icse_26/experiments_qwen35_9b.yaml \
    > logs/qwen.log 2>&1
python -m prompt_piper.tools.detailed_eval \
    --config resources/configs/icse_26/experiments_qwen35_9b.yaml \
    --results-dir results/icse_26/qwen3.5-9b \
    > logs/qwen_details.log 2>&1
