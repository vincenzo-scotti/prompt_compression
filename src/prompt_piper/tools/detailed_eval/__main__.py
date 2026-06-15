"""Per-example detailed evaluation tool.

Re-evaluates all Pareto-optimal solutions, baselines, and transferability
results from a completed experiment at the single data-point level.

For each run found in *results-dir*, the tool:

1. Loads the stored Pareto bit vectors and baseline configurations.
2. Re-runs the same fitting phase used during the original experiment
   (when ``use_fitting: true`` in the config) so operator instances
   match exactly.
3. Replays every compression pipeline on the test dataset.
4. Scores each (compressed_prompt, target) pair individually.
5. Writes one JSONL file per run (plus one per transferability pair).

Output schema (one JSON object per line)
-----------------------------------------
run_name            str   — name of the run (or "transfer_X_to_Y")
solution_type       str   — "pareto" | "baseline_<name>" | "transfer"
solution_id         int   — index within the solution set
transformation_names list  — active operator names
transformation_ids  list  — operator IDs with section suffix (@instruction…)
example_id          int   — index in the test dataset
target              str   — ground-truth answer / label
original_prompt     str   — uncompressed prompt
compressed_prompt   str   — compressed prompt
original_tokens     int   — token count of the original prompt
compressed_tokens   int   — token count of the compressed prompt
compression_ratio   float — compressed_tokens / original_tokens
performance         float — per-example task score (0–1)
eval_elapsed_seconds float — wall-clock seconds for model inference (per example)
[source_run]        str   — transfer records only
[target_run]        str   — transfer records only
[baseline_config]   dict  — baseline hyper-parameters (baseline records only)

Usage
-----
    python -m prompt_piper.tools.detailed_eval \\
        --config resources/configs/icse_26/experiments_gemini_flash.yaml \\
        --results-dir results/icse_26/gemini-3-flash-preview \\
        --output detailed/gemini-3-flash-preview

    # Process only specific runs:
    python -m prompt_piper.tools.detailed_eval \\
        --config experiments.yaml \\
        --runs gsm8k_cot_nsga2 mmlu_pro_cs_nsga2

    # Smoke-test without a GPU:
    python -m prompt_piper.tools.detailed_eval \\
        --config experiments.yaml --dry-run

Load in Python::

    import pandas as pd
    df = pd.read_json("detailed/gsm8k_cot_nsga2_detailed.jsonl", lines=True)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from prompt_piper.tools.compressor.config import CompressorConfig

from .runner import DetailedEvalRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m prompt_piper.tools.detailed_eval",
        description="Re-evaluate compression solutions at the per-example level.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="YAML",
        help="Compressor config YAML (same file used for the original run).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="DIR",
        help="Directory containing result JSONs (default: value of config output.dir).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="Output directory for JSONL files (default: <results-dir>/detailed).",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        metavar="RUN",
        default=None,
        help="Specific run names to process (default: all runs in config).",
    )
    parser.add_argument(
        "--no-transfer",
        action="store_true",
        help="Skip transferability evaluation even when enabled in the config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use MockEvaluator — no model inference (smoke-test mode).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config = CompressorConfig.from_yaml(args.config)

    if args.runs:
        run_set = set(args.runs)
        config.runs = [r for r in config.runs if r.run_name in run_set]
        if not config.runs:
            print(f"No runs matched names: {args.runs}", file=sys.stderr)
            sys.exit(1)

    if args.no_transfer:
        config.transferability.enabled = False

    results_dir = Path(args.results_dir) if args.results_dir else Path(config.output.dir)
    output_dir = Path(args.output) if args.output else results_dir / "detailed"

    runner = DetailedEvalRunner(
        config=config,
        results_dir=results_dir,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )
    runner.run_all()


if __name__ == "__main__":
    main()
