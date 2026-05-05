"""Compressor CLI — entry point for prompt_piper.tools.compressor.

Usage examples
--------------
# Run from a YAML config file (recommended for multi-run experiments):
    python -m prompt_piper.tools.compressor --config experiments/config.yaml

# Single-run shorthand via CLI args:
    python -m prompt_piper.tools.compressor \\
        --model-type hf --model-name gpt2 \\
        --task hellaswag \\
        --algorithm nsga2 --pop-size 50 --n-gen 100 \\
        --n-search 200 --n-test 500 \\
        --output-dir ./results

# Override specific values from a YAML config:
    python -m prompt_piper.tools.compressor \\
        --config experiments/config.yaml \\
        --n-gen 200 --seed 99

# Dry run (uses MockEvaluator; no model needed):
    python -m prompt_piper.tools.compressor \\
        --config experiments/config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import CompressorConfig
from .runner import CompressorRunner


def _load_dotenv(verbose: bool = False) -> None:
    """Load variables from a .env file if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
        loaded = load_dotenv(override=False)  # don't override already-set vars
        if verbose and loaded:
            print("Loaded environment variables from .env")
    except ImportError:
        pass  # python-dotenv is optional


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m prompt_piper.tools.compressor",
        description=(
            "Search for optimal prompt compression pipelines using "
            "multi-objective evolutionary optimisation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Config file -------------------------------------------------------
    p.add_argument(
        "--config", "-c",
        metavar="PATH",
        help="Path to a YAML config file.  CLI flags override values in the file.",
    )

    # ---- Model -------------------------------------------------------------
    model = p.add_argument_group("model")
    model.add_argument("--model-type", metavar="TYPE",
                       help="lm-eval model type: hf | openai | vllm")
    model.add_argument("--model-name", metavar="NAME",
                       help="Model identifier (HuggingFace name, OpenAI string, …)")
    model.add_argument("--model-arg", metavar="KEY=VALUE", action="append",
                       dest="model_args", default=[],
                       help="Extra model kwargs, e.g. --model-arg device=cuda (repeatable)")
    model.add_argument("--apply-chat-template", action="store_true",
                       help="Apply the model's chat template to prompts "
                            "(required for instruct/chat models)")
    model.add_argument("--extra-stop-sequence", metavar="SEQ", action="append",
                       dest="extra_stop_sequences", default=[],
                       help="Additional stop token appended to the task's until list "
                            "(repeatable). E.g. --extra-stop-sequence '<|eot_id|>'")

    # ---- Task / dataset ----------------------------------------------------
    task = p.add_argument_group("task / dataset")
    task.add_argument("--task", metavar="NAME",
                      help="lm-eval task name (e.g. hellaswag, mmlu).  "
                           "Creates a single-run config when --config is absent.")
    task.add_argument("--num-fewshot", type=int, metavar="K",
                      help="Number of few-shot examples")
    task.add_argument("--search-split", metavar="SPLIT",
                      help="Dataset split used for optimisation (default: validation)")
    task.add_argument("--test-split", metavar="SPLIT",
                      help="Dataset split used for final evaluation (default: test)")
    task.add_argument("--n-search", type=int, metavar="N",
                      help="Number of samples for the optimisation phase")
    task.add_argument("--n-test", type=int, metavar="N",
                      help="Number of samples for the evaluation phase (0 = all)")

    # ---- Algorithm ---------------------------------------------------------
    alg = p.add_argument_group("algorithm")
    alg.add_argument("--algorithm", choices=["nsga2", "nsga3", "moead"],
                     help="Multi-objective optimisation algorithm")
    alg.add_argument("--pop-size", type=int, metavar="N",
                     help="Population size")
    alg.add_argument("--n-gen", type=int, metavar="N",
                     help="Number of generations")
    alg.add_argument("--seed", type=int, help="Random seed")
    alg.add_argument("--subset-seed", type=int,
                     help="Seed used for dataset subset sampling")
    alg.add_argument("--transformation-ids", metavar="ID", nargs="+",
                     help="Restrict search to specific operator IDs (e.g. O1 O3 O7)")
    alg.add_argument(
        "--baselines",
        nargs="+",
        metavar="NAME",
        choices=["no_compression", "all_transformations", "random_search"],
        help="Baselines to run alongside the search",
    )
    alg.add_argument("--no-llm-semantic", action="store_true",
                     help="Disable LLM backend for semantic transformations "
                          "(use heuristic fallbacks instead)")
    alg.add_argument(
        "--compress-field",
        metavar="FIELD",
        help=(
            "Dataset field to apply transformations to.  "
            "Omit (default) to compress the fully formatted prompt.  "
            "Pass 'prompt_0shot' to compress only the 0-shot prompt "
            "(few-shot examples are excluded from compression)."
        ),
    )

    # ---- Output ------------------------------------------------------------
    out = p.add_argument_group("output")
    out.add_argument("--output-dir", metavar="DIR",
                     help="Directory where result JSON files are written")
    out.add_argument("--overwrite", action="store_true",
                     help="Overwrite existing result files")

    # ---- Misc --------------------------------------------------------------
    p.add_argument("--dry-run", action="store_true",
                   help="Use MockEvaluator instead of a real model (for testing)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable verbose logging")

    return p


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

def _parse_model_args(raw: list[str]) -> dict:
    """Parse ``KEY=VALUE`` strings into a dict."""
    result = {}
    for item in raw:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--model-arg must be KEY=VALUE, got: {item!r}"
            )
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        # Best-effort type coercion: bool → int → float → str
        if v.lower() == "true":
            result[k] = True
        elif v.lower() == "false":
            result[k] = False
        else:
            for cast in (int, float, lambda x: x):
                try:
                    result[k] = cast(v)
                    break
                except (ValueError, TypeError):
                    continue
    return result


def _build_config(args: argparse.Namespace) -> CompressorConfig:
    """Assemble a :class:`CompressorConfig` from parsed CLI args."""

    # --- Load YAML base config if provided ---
    if args.config:
        cfg = CompressorConfig.from_yaml(args.config)
    elif args.task:
        # No YAML: build a minimal single-run config from CLI args.
        cfg = CompressorConfig.single_run(
            model_type=args.model_type or "hf",
            model_name=args.model_name or "gpt2",
            task=args.task,
        )
    else:
        print(
            "error: provide --config PATH or at least --task NAME",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Override model settings ---
    if args.model_type:
        cfg.model.type = args.model_type
    if args.model_name:
        cfg.model.name = args.model_name
    if args.model_args:
        cfg.model.kwargs.update(_parse_model_args(args.model_args))
    if args.apply_chat_template:
        cfg.model.kwargs["apply_chat_template"] = True

    # --- Output ---
    if args.output_dir:
        cfg.output.dir = args.output_dir
    if args.overwrite:
        cfg.output.overwrite = True

    # --- Per-run overrides ---
    run_overrides = {
        "num_fewshot": args.num_fewshot,
        "search_split": args.search_split,
        "test_split": args.test_split,
        "n_search": args.n_search,
        "n_test": args.n_test if args.n_test != 0 else None,
        "algorithm": args.algorithm,
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "seed": args.seed,
        "subset_seed": args.subset_seed,
        "transformation_ids": args.transformation_ids,
        "baselines": args.baselines,
        "use_llm_for_semantic": (
            False if args.no_llm_semantic else None
        ),
        "compress_field": args.compress_field,
        "extra_stop_sequences": args.extra_stop_sequences or None,
    }
    cfg = cfg.apply_cli_overrides(run_overrides)

    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _load_dotenv(verbose=args.verbose)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _build_config(args)

    runner = CompressorRunner(cfg, dry_run=args.dry_run)

    results = runner.run_all()

    print(f"\n{'='*60}")
    print(f"Completed {len(results)} run(s).")
    for r in results:
        print(f"\n{r.summary()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()