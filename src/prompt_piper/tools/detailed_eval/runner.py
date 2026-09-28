"""Per-example detailed evaluation runner.

Re-evaluates every stored solution (Pareto front, baselines, transferability)
on the test dataset at the single-example level and writes the results as JSONL
files — one record per (solution × example).

Each record contains:
    run_name, solution_type, solution_id, transformation_names,
    example_id, target, original_prompt, compressed_prompt,
    original_tokens, compressed_tokens, compression_ratio,
    performance, eval_elapsed_seconds
    [+ source_run / target_run for transfer records]

Read with pandas::

    df = pd.read_json("gsm8k_cot_nsga2_detailed.jsonl", lines=True)
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path

import numpy as np

from prompt_piper.optimization.runner import Experiment
from prompt_piper.optimization.tokenizer import WhitespaceTokenizer
from prompt_piper.tools.compressor.config import RunConfig
from prompt_piper.tools.compressor.results import CompressorResult, TransferabilityEval
from prompt_piper.tools.compressor.runner import CompressorRunner

logger = logging.getLogger(__name__)

_CUSTOM_BASELINES: frozenset[str] = frozenset({"truncated_prompt", "random_deletion", "llmlingua2"})
_FULL_OP_BASELINES: frozenset[str] = frozenset({"all_transformations", "summarization"})


class DetailedEvalRunner(CompressorRunner):
    """Re-evaluate all stored solutions at the per-example level.

    Inherits model loading, dataset fetching, fitting, and backend
    construction from :class:`~prompt_piper.tools.compressor.runner.CompressorRunner`
    so results are produced under exactly the same conditions as the
    original experiment.

    Parameters
    ----------
    config:
        Compressor configuration (same YAML used for the original run).
    results_dir:
        Directory that contains the ``{run_name}.json`` result files.
    output_dir:
        Destination for ``{run_name}_detailed.jsonl`` output files.
    dry_run:
        Use :class:`~prompt_piper.optimization.evaluator.MockEvaluator`
        instead of the real model (for smoke-testing without a GPU).
    """

    def __init__(self, config, results_dir, output_dir, dry_run: bool = False) -> None:
        super().__init__(config, dry_run)
        self._results_dir = Path(results_dir)
        self._output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> None:
        """Process every run and transferability pair defined in the config."""
        for run_cfg in self._config.runs:
            try:
                self._process_run(run_cfg)
            except Exception:
                logger.exception("Failed to process run %s", run_cfg.run_name)

        if self._config.transferability.enabled:
            run_map = {r.run_name: r for r in self._config.runs}
            for pair in self._config.transferability.pairs:
                src, tgt = pair[0], pair[1]
                if tgt not in run_map:
                    logger.warning("Transfer target %r not found in config runs; skipping", tgt)
                    continue
                try:
                    self._process_transfer(src, tgt, run_map[tgt])
                except Exception:
                    logger.exception("Failed to process transfer %s → %s", src, tgt)

    # ------------------------------------------------------------------
    # Run-level processing
    # ------------------------------------------------------------------

    def _process_run(self, run_cfg: RunConfig) -> None:
        result_path = self._results_dir / f"{run_cfg.run_name}.json"
        if not result_path.exists():
            logger.warning("Result JSON not found: %s — skipping", result_path)
            return

        out_path = self._output_dir / f"{run_cfg.run_name}_detailed.jsonl"
        if out_path.exists():
            logger.info("Output already exists: %s — skipping", out_path)
            return

        logger.info("=== Processing run: %s ===", run_cfg.run_name)
        result = CompressorResult.load(result_path)
        backend, evaluator, tokenizer, factory = self._prepare_backend(run_cfg)
        test_dataset = self._load_test_dataset(backend, run_cfg)

        exp_config = self._make_exp_config(run_cfg, n_subset=len(test_dataset))
        exp = Experiment(
            config=exp_config,
            dataset=test_dataset,
            evaluator=evaluator,
            tokenizer=tokenizer,
            transformations=self._resolve_transforms(run_cfg),
            transformation_factory=factory,
        )
        search_problem = exp._build_problem(test_dataset)
        full_problem = exp._build_full_problem(test_dataset)

        records: list[dict] = []

        # ── Pareto solutions ──────────────────────────────────────────
        for sol_idx, sol in enumerate(result.pareto_solutions):
            logger.info(
                "  pareto %d/%d  ops=%s",
                sol_idx + 1, len(result.pareto_solutions),
                sol.transformation_names,
            )
            bits = np.array(sol.bits, dtype=int)
            if len(bits) != search_problem.n_var:
                logger.warning(
                    "  Bit vector length mismatch: stored=%d, problem=%d — skipping solution",
                    len(bits), search_problem.n_var,
                )
                continue
            meta = {
                "run_name": run_cfg.run_name,
                "solution_type": "pareto",
                "solution_id": sol_idx,
                "transformation_names": sol.transformation_names,
                "transformation_ids": sol.transformation_ids,
            }
            records.extend(
                _score_bits_detailed(bits, search_problem, test_dataset, evaluator, tokenizer, meta)
            )

        # ── Baselines ─────────────────────────────────────────────────
        for b_eval in result.baselines:
            n_sol = len(b_eval.transformation_names)
            logger.info("  baseline: %s (%d solution(s))", b_eval.name, n_sol)
            problem = full_problem if b_eval.name in _FULL_OP_BASELINES else search_problem

            for sol_idx, sol_names in enumerate(b_eval.transformation_names):
                meta = {
                    "run_name": run_cfg.run_name,
                    "solution_type": f"baseline_{b_eval.name}",
                    "solution_id": sol_idx,
                    "transformation_names": sol_names,
                    "baseline_config": b_eval.config,
                }

                if b_eval.name in _CUSTOM_BASELINES:
                    compress_fn = _make_custom_compress_fn(b_eval.name, b_eval.config)
                    if compress_fn is not None:
                        records.extend(
                            _score_fn_detailed(
                                compress_fn, problem, test_dataset, evaluator, tokenizer, meta
                            )
                        )
                else:
                    bits = _bits_from_names(sol_names, problem)
                    records.extend(
                        _score_bits_detailed(bits, problem, test_dataset, evaluator, tokenizer, meta)
                    )

        _write_jsonl(records, out_path)

    def _process_transfer(
        self,
        source_name: str,
        target_name: str,
        target_run_cfg: RunConfig,
    ) -> None:
        out_path = self._output_dir / f"transfer_{source_name}_to_{target_name}_detailed.jsonl"
        if out_path.exists():
            logger.info("Transfer output already exists: %s — skipping", out_path)
            return

        # Prefer the dedicated transfer JSON (solutions already re-evaluated on
        # target task); fall back to source run for bit vectors.
        transfer_path = self._results_dir / f"transfer_{source_name}_to_{target_name}.json"
        source_path = self._results_dir / f"{source_name}.json"

        if transfer_path.exists():
            te = TransferabilityEval.load(transfer_path)
            source_solutions = te.solutions
        elif source_path.exists():
            source_result = CompressorResult.load(source_path)
            source_solutions = source_result.pareto_solutions
        else:
            logger.warning(
                "No source result found for transfer %s → %s; skipping",
                source_name, target_name,
            )
            return

        if not source_solutions:
            logger.warning("Source %s has no Pareto solutions; skipping", source_name)
            return

        logger.info(
            "=== Transfer %s → %s (%d solutions) ===",
            source_name, target_name, len(source_solutions),
        )
        backend, evaluator, tokenizer, factory = self._prepare_backend(target_run_cfg)
        test_dataset = self._load_test_dataset(backend, target_run_cfg)

        exp_config = self._make_exp_config(target_run_cfg, n_subset=len(test_dataset))
        exp = Experiment(
            config=exp_config,
            dataset=test_dataset,
            evaluator=evaluator,
            tokenizer=tokenizer,
            transformations=self._resolve_transforms(target_run_cfg),
            transformation_factory=factory,
        )
        problem = exp._build_problem(test_dataset)

        run_label = f"transfer_{source_name}_to_{target_name}"
        records: list[dict] = []

        for sol_idx, sol in enumerate(source_solutions):
            bits = np.array(sol.bits, dtype=int)
            if len(bits) != problem.n_var:
                logger.warning(
                    "  Bit vector length mismatch: stored=%d, problem=%d — skipping",
                    len(bits), problem.n_var,
                )
                continue
            meta = {
                "run_name": run_label,
                "source_run": source_name,
                "target_run": target_name,
                "solution_type": "transfer",
                "solution_id": sol_idx,
                "transformation_names": sol.transformation_names,
                "transformation_ids": sol.transformation_ids,
            }
            records.extend(
                _score_bits_detailed(bits, problem, test_dataset, evaluator, tokenizer, meta)
            )

        _write_jsonl(records, out_path)

    # ------------------------------------------------------------------
    # Shared setup helpers
    # ------------------------------------------------------------------

    def _prepare_backend(self, run_cfg: RunConfig):
        """Build the backend and optionally re-run fitting.

        Returns ``(backend, evaluator, tokenizer, factory)``.  Fitting is
        re-run on the search split so operator instances match the original
        experiment.  The *backend* dict is also returned so the caller can
        pass it to :meth:`_load_test_dataset` without loading the model twice.
        """
        backend = self._get_backend(run_cfg)
        evaluator = backend["evaluator"]
        tokenizer = backend.get("tokenizer") or WhitespaceTokenizer()
        factory = backend.get("transformation_factory")

        if run_cfg.use_fitting:
            search_dataset = self._get_dataset(backend, run_cfg, split=run_cfg.search_split)
            semantic_backend = backend.get("semantic_backend")
            search_idx, _ = self._split_indices(run_cfg, len(search_dataset), len(search_dataset))
            fitted, _ = self._run_fitting(
                run_cfg, [search_dataset[i] for i in search_idx], semantic_backend,
            )
            factory = self._compose_factory(fitted, semantic_backend)

        return backend, evaluator, tokenizer, factory

    def _load_test_dataset(self, backend: dict, run_cfg: RunConfig) -> list[dict]:
        """Load and sub-sample the test split exactly as the compressor does.

        Uses :meth:`CompressorRunner._split_indices`, so the test sample is
        disjoint from the search sample when both come from the same split.
        """
        dataset = self._get_dataset(backend, run_cfg, split=run_cfg.test_split)
        if run_cfg.search_split == run_cfg.test_split:
            n_search_total = len(dataset)
        else:
            n_search_total = len(self._get_dataset(backend, run_cfg, split=run_cfg.search_split))
        _, test_idx = self._split_indices(run_cfg, n_search_total, len(dataset))
        return [dataset[i] for i in test_idx]


# ---------------------------------------------------------------------------
# Per-example scoring (module-level so they can be used without a runner instance)
# ---------------------------------------------------------------------------

def _score_bits_detailed(
    bits: np.ndarray,
    problem,
    test_dataset: list[dict],
    evaluator,
    tokenizer,
    meta: dict,
) -> list[dict]:
    """Compress *test_dataset* with *bits*, evaluate each example, return records."""
    originals, compressed = problem._compress_bits(bits)
    t0 = time.perf_counter()
    scores = evaluator.evaluate_batch(originals, compressed, test_dataset)
    per_ex = (time.perf_counter() - t0) / max(len(originals), 1)
    return _build_records(originals, compressed, scores, test_dataset, tokenizer, per_ex, meta)


def _score_fn_detailed(
    compress_fn,
    problem,
    test_dataset: list[dict],
    evaluator,
    tokenizer,
    meta: dict,
) -> list[dict]:
    """Apply an arbitrary *compress_fn* to each example, evaluate, return records.

    Used for custom baselines (truncated_prompt, random_deletion, llmlingua2)
    whose compression is not represented as a bit vector.
    """
    if problem._sections is not None:
        originals = [s.assemble() for s in problem._sections]
    else:
        originals = [s[problem._compress_field] for s in problem._dataset]
    compressed = [compress_fn(t) for t in originals]
    t0 = time.perf_counter()
    scores = evaluator.evaluate_batch(originals, compressed, test_dataset)
    per_ex = (time.perf_counter() - t0) / max(len(originals), 1)
    return _build_records(originals, compressed, scores, test_dataset, tokenizer, per_ex, meta)


def _build_records(
    originals: list[str],
    compressed: list[str],
    scores: list[float],
    dataset: list[dict],
    tokenizer,
    per_ex_elapsed: float,
    meta: dict,
) -> list[dict]:
    records = []
    for j, (orig, comp, score) in enumerate(zip(originals, compressed, scores)):
        sample = dataset[j] if j < len(dataset) else {}
        orig_tok = tokenizer.count(orig)
        comp_tok = tokenizer.count(comp)
        records.append({
            **meta,
            "example_id": j,
            "target": sample.get("target", ""),
            "original_prompt": orig,
            "compressed_prompt": comp,
            "original_tokens": orig_tok,
            "compressed_tokens": comp_tok,
            "compression_ratio": comp_tok / max(orig_tok, 1),
            "performance": float(score),
            "eval_elapsed_seconds": per_ex_elapsed,
        })
    return records


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

def _bits_from_names(names: list[str], problem) -> np.ndarray:
    """Reconstruct a binary bit vector from a list of operator names.

    Handles both flat mode (no section prefix) and section-aware mode
    ("I:", "S:", "D:" prefixes produced by ``problem.names_from_bits``).
    """
    bits = np.zeros(problem.n_var, dtype=int)
    name_set = set(names)

    if problem._sections is None:
        # Legacy flat mode: single transform list, no prefixes.
        for j, op in enumerate(problem._transforms):
            if op.name in name_set:
                bits[j] = 1
    else:
        # Section-aware mode: I:/S:/D: prefixes.
        n_i = len(problem._instruction_ops)
        n_s = len(problem._shot_ops)
        for j, op in enumerate(problem._instruction_ops):
            if f"I:{op.name}" in name_set:
                bits[j] = 1
        for j, op in enumerate(problem._shot_ops):
            if f"S:{op.name}" in name_set:
                bits[n_i + j] = 1
        for j, op in enumerate(problem._data_ops):
            if f"D:{op.name}" in name_set:
                bits[n_i + n_s + j] = 1

    return bits


def _make_custom_compress_fn(name: str, config: dict):
    """Return a ``text → text`` function for custom (non-operator) baselines.

    Returns ``None`` if the required dependency is unavailable.
    """
    if name == "truncated_prompt":
        ratio = config.get("target_ratio", 0.5)
        return lambda text: text[: max(1, int(len(text) * ratio))]

    if name == "random_deletion":
        prob = config.get("deletion_prob", 0.3)
        seed = config.get("seed", 42)
        # Single shared RNG — examples must be processed in the same order as
        # the original baseline so the random decisions are reproducible.
        rng = random.Random(seed)

        def _delete(text: str) -> str:
            paragraphs = re.split(r"\n{2,}", text)
            kept = [p for p in paragraphs if rng.random() >= prob]
            return "\n\n".join(kept) if kept else text

        return _delete

    if name == "llmlingua2":
        try:
            from llmlingua import PromptCompressor  # type: ignore[import]
        except ImportError:
            logger.warning("llmlingua not installed — skipping llmlingua2 baseline")
            return None
        ratio = config.get("compression_ratio", 0.5)
        compressor = PromptCompressor(
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
        )
        return lambda text: compressor.compress_prompt(text, rate=ratio)["compressed_prompt"]

    logger.warning("Unknown custom baseline %r — skipping", name)
    return None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")
    logger.info("Written %d records → %s", len(records), path)
