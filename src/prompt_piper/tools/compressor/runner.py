"""CompressorRunner — orchestrates multi-run experiments with model reuse."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from prompt_piper.optimization.config import ExperimentConfig
from prompt_piper.optimization.evaluator import MockEvaluator
from prompt_piper.optimization.bits import bits_to_ids
from prompt_piper.optimization.recorder import RunRecorder
from prompt_piper.optimization.runner import Experiment, sample_indices
from prompt_piper.transformations import ALL_TRANSFORMATIONS

from .callback import GenerationCallback
from .config import CompressorConfig, RunConfig
from .results import BaselineEval, CompressorResult, SolutionEval, TransferabilityEval

logger = logging.getLogger(__name__)

_LEVEL_LABELS = ["@instruction", "@shots", "@data"]

#: Baselines that transform text directly instead of selecting operator bits.
_TEXT_BASELINES = frozenset({"truncated_prompt", "random_deletion", "llmlingua2"})


def _unbit_text_baselines(name: str, solutions: list[dict]) -> list[dict]:
    """Text-level baselines have no operator bits: store ``None`` instead of zeros."""
    if name in _TEXT_BASELINES:
        for sol in solutions:
            sol.update(layout=None, bits=None, transformation_ids=None)
    return solutions


def _bits_to_ids(transform_classes, bits) -> list[str]:
    """Map a level-aware bit vector to operator IDs like ``"O3@instruction"``.

    The bit vector layout is ``[N instruction | N shots | N data]`` where N is
    ``len(transform_classes)``.  Extra levels beyond three are ignored.
    """
    n = len(transform_classes)
    if n == 0:
        return []
    n_levels = len(bits) // n
    ids: list[str] = []
    for lvl, label in zip(range(n_levels), _LEVEL_LABELS):
        for cls, b in zip(transform_classes, bits[lvl * n: (lvl + 1) * n]):
            if b:
                ids.append(f"{cls.id}{label}")
    return ids


class CompressorRunner:
    """Load the model once; iterate over runs defined in *config*.

    Parameters
    ----------
    config:
        Top-level compressor configuration.
    dry_run:
        If ``True``, use :class:`~prompt_piper.optimization.evaluator.MockEvaluator`
        instead of the real LM (useful for testing the pipeline without a GPU).
    """

    def __init__(self, config: CompressorConfig, dry_run: bool = False) -> None:
        self._config = config
        self._dry_run = dry_run
        self._model = None   # loaded lazily, shared across runs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> list[CompressorResult]:
        """Run every entry in ``config.runs`` and write results to disk.

        Restartable: if ``output.overwrite`` is ``False`` and a result file
        ``{output_dir}/{run_name}.json`` already exists, that run is skipped
        and the existing result is loaded instead.

        After all primary runs complete, performs transferability analysis if
        ``config.transferability.enabled`` is ``True``.
        """
        if not self._config.runs:
            raise ValueError("No runs defined in config.")

        out_dir = Path(self._config.output.dir)
        results = []
        for run_cfg in self._config.runs:
            base_path = out_dir / f"{run_cfg.run_name}.json"
            if not self._config.output.overwrite and base_path.exists():
                logger.info(
                    "Skipping completed run %s — result already exists at %s",
                    run_cfg.run_name, base_path,
                )
                try:
                    results.append(CompressorResult.load(base_path))
                except Exception:
                    logger.warning(
                        "Could not load existing result for %s; re-running.",
                        run_cfg.run_name,
                    )
                else:
                    continue

            logger.info("Starting run: %s", run_cfg.run_name)
            try:
                result = self.run_single(run_cfg)
                out_path = self._output_path(run_cfg)
                result.save(out_path)
                logger.info("Saved result → %s", out_path)
                results.append(result)
            except NotImplementedError as exc:
                logger.warning("Skipping run %s — %s", run_cfg.run_name, exc)
            except Exception:
                logger.exception("Run %s failed", run_cfg.run_name)
                raise

        if self._config.transferability.enabled and self._config.transferability.pairs:
            self.run_transferability()

        return results

    def run_single(self, run_cfg: RunConfig) -> CompressorResult:
        """Execute one (task, algorithm) run and return a :class:`CompressorResult`.

        When ``run_cfg.baselines_only`` is ``True`` the NSGA-II optimisation
        phase is skipped; only the requested baselines are evaluated.
        """
        timestamp = CompressorResult.now_timestamp()
        recorder = self._open_recorder(run_cfg.run_name)
        try:
            return self._run_single_recorded(run_cfg, timestamp, recorder)
        finally:
            recorder.close()

    def _run_single_recorded(self, run_cfg: RunConfig, timestamp: str, recorder) -> CompressorResult:
        if run_cfg.fewshot_as_multiturn:
            raise ValueError(
                f"{run_cfg.run_name}: fewshot_as_multiturn=true is not supported. The few-shot "
                "examples are a compressible text section; the assembled prompt is sent as a "
                "single user turn (wrapped in the chat template when apply_chat_template=true)."
            )
        backend = self._get_backend(run_cfg)
        transform_classes = self._resolve_transforms(run_cfg)
        transform_ids = [T.id for T in transform_classes]
        transform_names = [T.name for T in transform_classes]
        semantic_backend = backend.get("semantic_backend")
        if semantic_backend is not None:
            semantic_backend.recorder = recorder

        # ---- Data: search subset and a disjoint test subset -----------------
        search_dataset = self._get_dataset(backend, run_cfg, split=run_cfg.search_split)
        test_full = self._get_dataset(backend, run_cfg, split=run_cfg.test_split)
        search_idx, test_idx = self._split_indices(run_cfg, len(search_dataset), len(test_full))
        test_dataset = [test_full[i] for i in test_idx]
        recorder.update_meta(
            run_name=run_cfg.run_name,
            timestamp=timestamp,
            config=self._serialise_config(run_cfg),
            dry_run=self._dry_run,
            generation_kwargs=getattr(backend.get("evaluator"), "_gen_kwargs", None),
            transformation_ids=transform_ids,
            source_splits={"search": run_cfg.search_split, "test": run_cfg.test_split},
            source_split_sizes={"search": len(search_dataset), "test": len(test_full)},
            same_source_split=run_cfg.search_split == run_cfg.test_split,
            sample_ids={"search": search_idx, "test": test_idx},
            disjoint=not (set(search_idx) & set(test_idx))
            if run_cfg.search_split == run_cfg.test_split else True,
        )

        # ---- Fitting phase (optional) ------------------------------------
        if run_cfg.use_fitting:
            t_fit_start = time.perf_counter()
            with recorder.phase("fitting", method="fitter"):
                # Fit on the search sample S only: no test samples or targets.
                fitted, fitting_report = self._run_fitting(
                    run_cfg, [search_dataset[i] for i in search_idx], semantic_backend
                )
            fitting_elapsed: float | None = time.perf_counter() - t_fit_start
            factory = self._compose_factory(fitted, semantic_backend)
        else:
            fitting_report = None
            fitting_elapsed = None
            factory = backend.get("transformation_factory")

        exp_config = self._make_exp_config(run_cfg, n_subset=run_cfg.n_search)
        exp = Experiment(
            config=exp_config,
            dataset=search_dataset,
            evaluator=backend["evaluator"],
            tokenizer=backend.get("tokenizer"),
            transformations=transform_classes,
            transformation_factory=factory,
            recorder=recorder,
        )

        bit_layouts = {"search": exp.bit_layout(), "full": exp.bit_layout(full=True)}
        seeds = self._seeds(run_cfg)
        from prompt_piper.transformations import lexical
        recorder.update_meta(bit_layouts=bit_layouts, seeds=seeds,
                             stopword_list=lexical.STOPWORD_SOURCE)
        self._log_samples(recorder, "search", run_cfg.search_split, [search_dataset[i] for i in search_idx])
        self._log_samples(recorder, "test", run_cfg.test_split, test_dataset)

        self._check_same_structure(exp, [search_dataset[i] for i in search_idx], test_dataset, recorder)

        n_budget = run_cfg.pop_size * run_cfg.n_gen
        expected: dict = {
            "algorithm": None if run_cfg.baselines_only else run_cfg.algorithm,
            "n_gen": None if run_cfg.baselines_only else run_cfg.n_gen,
            "pop_size": None if run_cfg.baselines_only else run_cfg.pop_size,
            "search_candidates": {
                b: n_budget for b in (run_cfg.baselines or [])
                if b in ("random_search", "multi_armed_bandit")
            },
            "test_labels": {},
            "baselines": list(run_cfg.baselines or []),
        }

        # ================================================================
        # BASELINES-ONLY mode: skip NSGA-II, run baselines on both splits
        # ================================================================
        if run_cfg.baselines_only:
            return self._run_baselines_only(
                run_cfg, exp, test_dataset,
                transform_ids, transform_names,
                timestamp, fitting_report, fitting_elapsed,
                recorder=recorder, expected=expected,
                search_idx=search_idx, test_idx=test_idx,
                bit_layouts=bit_layouts, seeds=seeds,
            )

        # ---- Search phase ------------------------------------------------
        callback = GenerationCallback(transform_names, recorder=recorder,
                                      layout=bit_layouts["search"])

        exp_result = exp.run(
            baselines=run_cfg.baselines,
            callback=callback,
        )
        # The primary algorithm's own search time only (exp.run also runs every
        # baseline's search, whose times are reported per baseline).
        search_elapsed = exp_result.elapsed_time
        if exp.search_indices != search_idx:
            raise RuntimeError("Search subset differs from the recorded search sample ids.")

        # ---- Test evaluation phase ----------------------------------------
        t_eval_start = time.perf_counter()
        pareto_labels = [f"pareto:{i}" for i in range(len(exp_result.pareto_solutions))]
        expected["test_labels"][run_cfg.algorithm] = pareto_labels
        with recorder.phase("test", method=run_cfg.algorithm):
            test_objectives = exp.evaluate_solutions(
                exp_result.pareto_solutions, test_dataset, split="test", labels=pareto_labels,
            )
        # Test time of the primary algorithm's Pareto front only; baselines'
        # test times are included in their own elapsed_seconds.
        eval_elapsed = time.perf_counter() - t_eval_start

        baseline_test, baseline_solutions = self._evaluate_baselines_on_test(
            exp, exp_result.baselines, test_dataset, recorder, expected,
        )

        # ---- Neutral-bit analysis (compression only, no model calls) -------
        analysis_problem = exp.compression_problem()
        noop_operators = {"search": bits_to_ids(
            bit_layouts["search"],
            [int(i in set(analysis_problem.noop_bits())) for i in range(analysis_problem.n_var)],
        )}
        recorder.update_meta(noop_operators=noop_operators)

        def effective_names(bits):
            keep = set(analysis_problem.effective_bits(bits))
            return analysis_problem.names_from_bits([int(i in keep) for i in range(len(bits))])

        # ---- Assemble result ----------------------------------------------
        pareto_evals = []
        for i, (bits, names) in enumerate(
            zip(exp_result.pareto_solutions, exp_result.pareto_transformation_names)
        ):
            ids = bits_to_ids(bit_layouts["search"], bits)
            pareto_evals.append(
                SolutionEval(
                    transformation_ids=ids,
                    transformation_names=names,
                    bits=bits.tolist(),
                    search_avg_tokens=float(exp_result.pareto_objectives[i, 0]),
                    search_avg_performance=float(exp_result.pareto_objectives[i, 1]),
                    test_avg_tokens=float(test_objectives[i, 0]),
                    test_avg_performance=float(test_objectives[i, 1]),
                    effective_transformation_names=effective_names(bits),
                )
            )

        baseline_evals = []
        for name, br in exp_result.baselines.items():
            b_test_tok, b_test_perf, b_test_elapsed = baseline_test.get(name, (None, None, None))
            bl_search_elapsed = exp_result.baseline_elapsed.get(name)
            total_elapsed = (
                (bl_search_elapsed or 0.0) + (b_test_elapsed or 0.0)
                if bl_search_elapsed is not None or b_test_elapsed is not None
                else None
            )
            baseline_evals.append(
                BaselineEval(
                    name=name,
                    transformation_names=br.transformation_names,
                    search_avg_tokens=float(br.avg_tokens.mean()),
                    search_avg_performance=float(br.avg_performance.mean()),
                    test_avg_tokens=b_test_tok,
                    test_avg_performance=b_test_perf,
                    elapsed_seconds=total_elapsed,
                    config=br.config,
                    solutions=baseline_solutions.get(name, []),
                )
            )

        recorder.update_meta(expected=expected)
        validation = self._validate(recorder)

        return CompressorResult(
            run_name=run_cfg.run_name,
            timestamp=timestamp,
            config=self._serialise_config(run_cfg),
            transformation_ids=transform_ids,
            transformation_names=transform_names,
            search_elapsed_seconds=search_elapsed,
            eval_elapsed_seconds=eval_elapsed,
            pareto_solutions=pareto_evals,
            baselines=baseline_evals,
            generation_history=exp_result.generation_history,
            extra={"dry_run": self._dry_run},
            fitting_report=fitting_report,
            fitting_elapsed_seconds=fitting_elapsed,
            log_dir=str(recorder.run_dir),
            search_sample_ids=search_idx,
            test_sample_ids=test_idx,
            validation=validation,
            bit_layouts=bit_layouts,
            seeds=seeds,
            noop_operators=noop_operators,
        )

    def _evaluate_baselines_on_test(self, exp, baselines: dict, test_dataset, recorder, expected):
        """Re-evaluate every baseline solution on the test split.

        Returns ``(summary, solutions)``: *summary* maps name → ``(mean tokens,
        mean performance, elapsed)``; *solutions* maps name → one dict per
        solution with bits, names, and search and test objectives.
        """
        # Baselines that need re-running (not bit-vector re-evaluation) on test data:
        # - custom baselines use score_transform_fn (no bit vector to replay)
        # - full-op baselines used a different problem instance during search;
        #   their bit vectors have a different length than the search problem.
        _CUSTOM_BASELINES = {"truncated_prompt", "random_deletion", "llmlingua2"}
        _FULL_OP_BASELINES = exp._FULL_OP_BASELINES  # {"summarization"}

        full_test_problem = None  # built lazily
        summary: dict[str, tuple[float, float, float]] = {}
        solutions: dict[str, list[dict]] = {}
        for name, br in baselines.items():
            t_bl_test = time.perf_counter()
            with recorder.phase("baseline_test", method=name):
                if name in _CUSTOM_BASELINES:
                    test_br = exp.run_baseline(name, problem=exp._build_problem(test_dataset, split="test"))
                    test_obj = test_br.objectives
                elif name in _FULL_OP_BASELINES:
                    if full_test_problem is None:
                        full_test_problem = exp._build_full_problem(test_dataset, split="test")
                    test_br = exp.run_baseline(name, problem=full_test_problem)
                    test_obj = test_br.objectives
                else:
                    labels = [f"{name}:{i}" for i in range(len(br.solutions))]
                    expected["test_labels"][name] = labels
                    test_obj = exp.evaluate_solutions(br.solutions, test_dataset, split="test", labels=labels)
            summary[name] = (
                float(test_obj[:, 0].mean()), float(test_obj[:, 1].mean()),
                time.perf_counter() - t_bl_test,
            )
            solutions[name] = [
                {
                    "layout": "full" if name in _FULL_OP_BASELINES else "search",
                    "bits": [int(b) for b in br.solutions[i]],
                    "transformation_ids": bits_to_ids(
                        exp.bit_layout(full=name in _FULL_OP_BASELINES), br.solutions[i],
                    ),
                    "transformation_names": br.transformation_names[i],
                    "search_avg_tokens": float(br.objectives[i, 0]),
                    "search_avg_performance": float(br.objectives[i, 1]),
                    "test_avg_tokens": float(test_obj[i, 0]) if i < len(test_obj) else None,
                    "test_avg_performance": float(test_obj[i, 1]) if i < len(test_obj) else None,
                }
                for i in range(len(br.solutions))
            ]
            _unbit_text_baselines(name, solutions[name])
            if name in ("random_search", "multi_armed_bandit", "all_transformations"):
                problem = exp.compression_problem()
                for sol in solutions[name]:
                    keep = set(problem.effective_bits(sol["bits"]))
                    sol["effective_transformation_names"] = problem.names_from_bits(
                        [int(i in keep) for i in range(len(sol["bits"]))]
                    )
        return summary, solutions

    def _run_baselines_only(
        self,
        run_cfg: RunConfig,
        exp: "Experiment",
        test_dataset: list[dict],
        transform_ids: list[str],
        transform_names: list[str],
        timestamp: str,
        fitting_report,
        fitting_elapsed: float | None = None,
        recorder=None,
        expected: dict | None = None,
        search_idx: list[int] | None = None,
        test_idx: list[int] | None = None,
        bit_layouts: dict | None = None,
        seeds: dict | None = None,
    ) -> CompressorResult:
        """Run baselines on both search and test splits; no NSGA-II optimisation."""
        search_problem = exp._build_problem(exp._sample_subset(), split="search")
        test_problem = exp._build_problem(test_dataset, split="test")
        full_search = exp._build_full_problem(exp._sample_subset(), split="search")
        full_test = exp._build_full_problem(test_dataset, split="test")
        expected = expected or {"search_candidates": {}, "test_labels": {}}

        t_eval_start = time.perf_counter()
        baseline_evals = []
        for name in (run_cfg.baselines or []):
            t_bl = time.perf_counter()
            full = name in exp._FULL_OP_BASELINES
            with recorder.phase("baseline_search", method=name):
                search_br = exp.run_baseline(name, problem=full_search if full else search_problem)
            with recorder.phase("baseline_test", method=name):
                if name in ("random_search", "multi_armed_bandit"):
                    labels = [f"{name}:{i}" for i in range(len(search_br.solutions))]
                    expected["test_labels"][name] = labels
                    test_obj = exp.evaluate_solutions(
                        search_br.solutions, test_dataset, split="test", labels=labels,
                    )
                else:
                    test_obj = exp.run_baseline(name, problem=full_test if full else test_problem).objectives
            bl_elapsed = time.perf_counter() - t_bl
            baseline_evals.append(
                BaselineEval(
                    name=name,
                    transformation_names=search_br.transformation_names,
                    search_avg_tokens=float(search_br.avg_tokens.mean()),
                    search_avg_performance=float(search_br.avg_performance.mean()),
                    test_avg_tokens=float(test_obj[:, 0].mean()),
                    test_avg_performance=float(test_obj[:, 1].mean()),
                    elapsed_seconds=bl_elapsed,
                    config=search_br.config,
                    solutions=_unbit_text_baselines(name, [
                        {
                            "layout": "full" if full else "search",
                            "bits": [int(b) for b in search_br.solutions[i]],
                            "transformation_ids": bits_to_ids(
                                exp.bit_layout(full=full), search_br.solutions[i],
                            ),
                            "transformation_names": search_br.transformation_names[i],
                            "search_avg_tokens": float(search_br.objectives[i, 0]),
                            "search_avg_performance": float(search_br.objectives[i, 1]),
                            "test_avg_tokens": float(test_obj[i, 0]) if i < len(test_obj) else None,
                            "test_avg_performance": float(test_obj[i, 1]) if i < len(test_obj) else None,
                        }
                        for i in range(len(search_br.solutions))
                    ]),
                )
            )
        eval_elapsed = time.perf_counter() - t_eval_start
        recorder.update_meta(expected=expected)
        validation = self._validate(recorder)

        return CompressorResult(
            run_name=run_cfg.run_name,
            timestamp=timestamp,
            config=self._serialise_config(run_cfg),
            transformation_ids=transform_ids,
            transformation_names=transform_names,
            search_elapsed_seconds=0.0,
            eval_elapsed_seconds=eval_elapsed,
            pareto_solutions=[],
            baselines=baseline_evals,
            generation_history=[],
            extra={"dry_run": self._dry_run, "baselines_only": True},
            fitting_report=fitting_report,
            fitting_elapsed_seconds=fitting_elapsed,
            log_dir=str(recorder.run_dir),
            search_sample_ids=search_idx or [],
            test_sample_ids=test_idx or [],
            validation=validation,
            bit_layouts=bit_layouts or {},
            seeds=seeds or {},
        )

    # ------------------------------------------------------------------
    # Logging and data-split helpers
    # ------------------------------------------------------------------

    def _open_recorder(self, name: str) -> RunRecorder:
        """Create ``{output.dir}/{name}_log``; keep any leftover as ``.incomplete-*``."""
        log_dir = Path(self._config.output.dir) / f"{name}_log"
        if log_dir.exists():
            ts = CompressorResult.now_timestamp().replace(":", "-")
            backup = log_dir.with_name(f"{log_dir.name}.incomplete-{ts}")
            log_dir.rename(backup)
            logger.warning("Existing log directory moved to %s", backup)
        return RunRecorder(log_dir)

    @staticmethod
    def _split_indices(run_cfg: RunConfig, n_search_total: int, n_test_total: int):
        """Return ``(search_idx, test_idx)``; disjoint when both use the same split.

        The search indices are exactly those drawn by
        :meth:`Experiment._sample_subset`.  When search and test come from the
        same source split, the test sample is drawn from the remaining items.
        """
        search_idx = sample_indices(n_search_total, run_cfg.n_search, run_cfg.subset_seed)
        same = run_cfg.search_split == run_cfg.test_split
        test_idx = sample_indices(
            n_test_total, run_cfg.n_test, run_cfg.subset_seed + 1,
            exclude=search_idx if same else (),
        )
        if same and set(search_idx) & set(test_idx):
            raise RuntimeError("Search and test samples overlap.")
        if len(search_idx) < run_cfg.n_search or (run_cfg.n_test is not None and len(test_idx) < run_cfg.n_test):
            raise ValueError(
                f"{run_cfg.run_name}: requested n_search={run_cfg.n_search}, n_test={run_cfg.n_test} "
                f"but only {len(search_idx)} / {len(test_idx)} samples are available "
                f"(source split sizes {n_search_total} / {n_test_total})."
            )
        return search_idx, test_idx

    @staticmethod
    def _check_same_structure(exp, search_samples, test_samples, recorder) -> None:
        """Search and test prompts must share every static part.

        The instruction, few-shot block, answer cue, separators and tail are
        compressed once per strategy; if they differed between S and T (e.g.
        a cue detected differently on the two corpora) the same strategy would
        act on different text in search and test.  Raises on any difference.
        """
        from prompt_piper.transformations.sections import PromptSplitter

        field = exp.config.compress_field or exp.config.prompt_field
        static = ("instruction", "shots", "output_cue", "sep_after_instruction",
                  "sep_after_shots", "sep_before_cue", "tail")
        signatures = {}
        for role, samples in (("search", search_samples), ("test", test_samples)):
            sections = PromptSplitter(samples, compress_field=field).split_all()
            sig = {tuple(getattr(sec, k) for k in static) for sec in sections}
            signatures[role] = sig
        same = len(signatures["search"]) == 1 and signatures["search"] == signatures["test"]
        recorder.update_meta(static_sections_identical_across_splits=same,
                             n_static_variants={k: len(v) for k, v in signatures.items()})
        if not same and len(signatures["search"] | signatures["test"]) > 1 and all(
            "_input_text" in s for s in search_samples + test_samples
        ):
            raise RuntimeError(
                "Static prompt sections differ between search and test samples "
                f"({ {k: len(v) for k, v in signatures.items()} } variants)."
            )

    @staticmethod
    def _seeds(run_cfg: RunConfig) -> dict:
        """Every seed that influences a run."""
        return {
            "subset_seed": run_cfg.subset_seed,          # search sample
            "test_seed": run_cfg.subset_seed + 1,        # test sample
            "algorithm_seed": run_cfg.seed,              # pymoo
            "random_search_seed": run_cfg.seed,
            "multi_armed_bandit_seed": run_cfg.seed,
            "random_deletion_seed": run_cfg.seed,
            "fitter_seed": run_cfg.seed,
        }

    @staticmethod
    def _log_samples(recorder: RunRecorder, role: str, split: str, samples: list[dict]) -> None:
        """Log the full content of every sample used in *role* (search / test)."""
        for position, sample in enumerate(samples):
            record = {"role": role, "source_split": split, "position": position}
            for key, value in sample.items():
                record["doc" if key == "_doc" else key.lstrip("_")] = value
            recorder.write("samples", record)

    @staticmethod
    def _validate(recorder: RunRecorder) -> dict:
        from prompt_piper.tools.validate_run import validate_run

        report = validate_run(recorder.run_dir)
        recorder.update_meta(validation=report)
        if report["ok"]:
            logger.info("Log validation passed: %s", report["counts"])
        else:
            logger.error(
                "Log validation FAILED (%d problems), e.g.: %s",
                report["n_errors"], report["errors"][:5],
            )
        return report

    # ------------------------------------------------------------------
    # Transferability analysis
    # ------------------------------------------------------------------

    def run_transferability(self) -> list[TransferabilityEval]:
        """Evaluate Pareto solutions from one run on another run's test dataset.

        For each ``(source_run, target_run)`` pair in
        ``config.transferability.pairs``:

        1. Load the source run's result file to obtain its Pareto bit vectors.
        2. Use the target run's ``RunConfig`` to build a backend and fetch the
           test dataset.
        3. Evaluate each source solution on the target task and save the result
           as ``{output_dir}/transfer_{source_run}_to_{target_run}.json``.

        Skips a pair if the output file already exists (respects
        ``output.overwrite``).

        Returns a list of :class:`~results.TransferabilityEval` objects.
        """
        out_dir = Path(self._config.output.dir)
        run_map = {r.run_name: r for r in self._config.runs}
        evals: list[TransferabilityEval] = []

        for pair in self._config.transferability.pairs:
            source_name, target_name = pair[0], pair[1]
            out_path = out_dir / f"transfer_{source_name}_to_{target_name}.json"

            if not self._config.output.overwrite and out_path.exists():
                logger.info(
                    "Skipping transferability %s → %s (result exists)",
                    source_name, target_name,
                )
                try:
                    evals.append(TransferabilityEval.load(out_path))
                except Exception:
                    pass
                continue

            source_result_path = out_dir / f"{source_name}.json"
            if not source_result_path.exists():
                logger.warning(
                    "Transferability skipped: source result not found at %s",
                    source_result_path,
                )
                continue

            if target_name not in run_map:
                logger.warning(
                    "Transferability skipped: target run %r not in config.runs",
                    target_name,
                )
                continue

            logger.info(
                "Transferability: evaluating %s solutions on %s",
                source_name, target_name,
            )
            try:
                te = self._evaluate_transfer(
                    source_name=source_name,
                    source_result_path=source_result_path,
                    target_run_cfg=run_map[target_name],
                    transform_classes=self._resolve_transforms(run_map[target_name]),
                )
                te.save(out_path)
                logger.info("Saved transferability result → %s", out_path)
                evals.append(te)
            except Exception:
                logger.exception(
                    "Transferability %s → %s failed", source_name, target_name
                )

        return evals

    def _evaluate_transfer(
        self,
        source_name: str,
        source_result_path: Path,
        target_run_cfg: RunConfig,
        transform_classes,
    ) -> TransferabilityEval:
        """Run the actual cross-task evaluation for one pair."""
        from prompt_piper.optimization.runner import Experiment

        source_result = CompressorResult.load(source_result_path)
        if not source_result.pareto_solutions:
            logger.warning("Source %s has no Pareto solutions; skipping.", source_name)
            return TransferabilityEval(
                source_run=source_name,
                target_run=target_run_cfg.run_name,
                timestamp=CompressorResult.now_timestamp(),
                solutions=[],
            )

        # Build target backend (reuses already-loaded model)
        target_backend = self._get_backend(target_run_cfg)
        test_full = self._get_dataset(target_backend, target_run_cfg, split=target_run_cfg.test_split)
        if target_run_cfg.search_split == target_run_cfg.test_split:
            search_full = test_full
        else:
            search_full = self._get_dataset(
                target_backend, target_run_cfg, split=target_run_cfg.search_split,
            )
        search_idx, test_idx = self._split_indices(target_run_cfg, len(search_full), len(test_full))
        target_dataset = [test_full[i] for i in test_idx]

        recorder = self._open_recorder(f"transfer_{source_name}_to_{target_run_cfg.run_name}")
        if target_backend.get("semantic_backend") is not None:
            target_backend["semantic_backend"].recorder = recorder
        labels = [f"pareto:{i}" for i in range(len(source_result.pareto_solutions))]
        recorder.update_meta(
            source_run=source_name,
            target_run=target_run_cfg.run_name,
            config=self._serialise_config(target_run_cfg),
            same_source_split=target_run_cfg.search_split == target_run_cfg.test_split,
            sample_ids={"search": search_idx, "test": test_idx},
            expected={"search_candidates": {}, "test_labels": {"transfer": labels}},
        )

        # Adaptable operators are fitted on the *target* search sample, exactly as
        # in the native target run: only the choice of operators is transferred.
        # (LLM-backed fitting reuses the target run's calls via the persistent cache.)
        semantic_backend = target_backend.get("semantic_backend")
        factory = target_backend.get("transformation_factory")
        fitting_report = None
        if target_run_cfg.use_fitting:
            with recorder.phase("fitting", method="fitter"):
                fitted, fitting_report = self._run_fitting(
                    target_run_cfg, [search_full[i] for i in search_idx], semantic_backend,
                )
            factory = self._compose_factory(fitted, semantic_backend)
        recorder.update_meta(fitting_report=fitting_report, fitted_on="target search sample")

        exp_config = self._make_exp_config(target_run_cfg, n_subset=len(target_dataset))
        exp = Experiment(
            config=exp_config,
            dataset=target_dataset,
            evaluator=target_backend["evaluator"],
            tokenizer=target_backend.get("tokenizer"),
            transformations=transform_classes,
            transformation_factory=factory,
            recorder=recorder,
        )

        recorder.update_meta(
            bit_layouts={"search": exp.bit_layout(), "full": exp.bit_layout(full=True)},
            source_bit_layouts=source_result.bit_layouts,
            seeds=self._seeds(target_run_cfg),
        )
        self._log_samples(recorder, "test", target_run_cfg.test_split, target_dataset)

        # Reconstruct bit vectors from source Pareto solutions
        import numpy as np
        source_bits = np.array([sol.bits for sol in source_result.pareto_solutions], dtype=int)
        if source_result.bit_layouts.get("search") not in (None, exp.bit_layout()):
            raise ValueError("Source and target runs use different bit layouts.")

        # Evaluate on target dataset
        try:
            with recorder.phase("transfer_test", method="transfer"):
                target_objectives = exp.evaluate_solutions(
                    source_bits, target_dataset, split="test", labels=labels,
                )
            self._validate(recorder)
        finally:
            recorder.close()

        solutions = []
        for sol, obj in zip(source_result.pareto_solutions, target_objectives):
            solutions.append(
                SolutionEval(
                    transformation_ids=sol.transformation_ids,
                    transformation_names=sol.transformation_names,
                    bits=sol.bits,
                    search_avg_tokens=sol.search_avg_tokens,
                    search_avg_performance=sol.search_avg_performance,
                    test_avg_tokens=float(obj[0]),
                    test_avg_performance=float(obj[1]),
                )
            )

        return TransferabilityEval(
            source_run=source_name,
            target_run=target_run_cfg.run_name,
            timestamp=CompressorResult.now_timestamp(),
            solutions=solutions,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model(self):
        """Load the lm-eval model once and cache it."""
        if self._model is None and not self._dry_run:
            from prompt_piper.backends.lm_eval import _load_model
            logger.info(
                "Loading model %s (%s)...",
                self._config.model.name,
                self._config.model.type,
            )
            kwargs = dict(self._config.model.kwargs)
            if self._config.model.quantization:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    **self._config.model.quantization
                )
            self._model = _load_model(
                self._config.model.type,
                self._config.model.name,
                **kwargs,
            )
            logger.info("Model loaded.")
        return self._model

    def _get_backend(self, run_cfg: RunConfig) -> dict[str, Any]:
        """Return a dict with evaluator, dataset getter, and transformation factory."""
        if self._dry_run:
            return {
                "evaluator": MockEvaluator(),
                "task": None,
                "transformation_factory": None,
                "semantic_backend": None,
            }

        from prompt_piper.backends.lm_eval import (
            LMEvalBackend,
            LMEvalEvaluator,
            LMEvalSemanticBackend,
            LMEvalTokenizer,
            _load_task,
        )

        from prompt_piper.backends.lm_eval import _is_chat_model

        model = self._get_model()
        task = _load_task(run_cfg.task, num_fewshot=run_cfg.num_fewshot)

        # Raise early when a loglikelihood task is paired with a chat API model
        # (OpenAI, Gemini, …) — these APIs do not expose prompt log-probs.
        if _is_chat_model(model):
            first_task = next(iter(task.values())) if isinstance(task, dict) else task
            output_type = getattr(first_task, "OUTPUT_TYPE", "generate_until")
            if output_type in ("multiple_choice", "loglikelihood"):
                raise NotImplementedError(
                    f"Task {run_cfg.task!r} uses output_type={output_type!r}, "
                    f"which requires loglikelihood — not supported by chat API "
                    f"models. Remove this run from the config or use an HF model."
                )

        evaluator = LMEvalEvaluator(
            model, task,
            generation_kwargs=run_cfg.generation_kwargs,
            extra_stop_sequences=run_cfg.extra_stop_sequences,
            metric_key=run_cfg.metric_key,
            apply_chat_template=run_cfg.apply_chat_template,
        )
        tokenizer = LMEvalTokenizer(model)
        semantic_backend = (
            LMEvalSemanticBackend(
                model,
                apply_chat_template=run_cfg.apply_chat_template,
                # Persistent cache shared by every run writing to this output dir
                # (one dir per model), so LLM rewrites are reused for search,
                # test, baselines, fitting and later runs.
                cache_path=str(Path(self._config.output.dir) / "semantic_llm_cache.jsonl"),
                model_id=self._config.model.name,
            )
            if run_cfg.use_llm_for_semantic else None
        )

        import inspect

        def factory(cls):
            if semantic_backend and cls.category == "semantic":
                if "backend" in inspect.signature(cls.__init__).parameters:
                    return cls(backend=semantic_backend)
            return cls()

        return {
            "evaluator": evaluator,
            "tokenizer": tokenizer,
            "task": task,
            "transformation_factory": factory,
            "semantic_backend": semantic_backend,
        }

    def _get_dataset(
        self, backend: dict, run_cfg: RunConfig, split: str
    ) -> list[dict]:
        """Return the sample dicts of *split*, each tagged with its index.

        ``_sample_id`` (position in the split) and ``_split`` are attached to
        every sample so that logged records identify samples unambiguously.
        """
        samples = self._load_dataset(backend, run_cfg, split)
        for i, sample in enumerate(samples):
            sample["_sample_id"] = i
            sample["_split"] = split
        return samples

    def _load_dataset(
        self, backend: dict, run_cfg: RunConfig, split: str
    ) -> list[dict]:
        """Return a list of sample dicts for the given split."""
        if self._dry_run or backend["task"] is None:
            samples = self._synthetic_dataset(n=run_cfg.n_search + (run_cfg.n_test or 0) + 10)
            if run_cfg.instruction_prefix:
                prefix = run_cfg.instruction_prefix.rstrip("\n")
                base_field = run_cfg.compress_field or "prompt_0shot"
                for sample in samples:
                    base_text = sample.get(base_field, sample.get("prompt_0shot", ""))
                    sample["verbose_instruction"] = prefix + "\n\n" + base_text
            return samples

        # When compressing a field other than the main prompt, ask the dataset
        # to also store the 0-shot template under that field name.
        # When instruction_prefix is set, we always need prompt_0shot so we can
        # build the verbose_instruction field.
        raw_compress = run_cfg.compress_field
        if run_cfg.instruction_prefix and not raw_compress:
            raw_compress = "prompt_0shot"
        template_field = (
            raw_compress
            if raw_compress and raw_compress != run_cfg.prompt_field
            else None
        )

        from prompt_piper.backends.lm_eval import LMEvalDataset
        ds = LMEvalDataset(
            backend["task"],
            split=split,
            num_fewshot=run_cfg.num_fewshot,
            prompt_field=run_cfg.prompt_field,
            template_field=template_field,
            apply_chat_template=run_cfg.apply_chat_template,
            fewshot_as_multiturn=run_cfg.fewshot_as_multiturn,
        )
        samples = ds.as_list()

        # Inject verbose_instruction when an instruction_prefix is configured.
        # The combined field becomes the compression target (see effective_compress_field).
        if run_cfg.instruction_prefix:
            prefix = run_cfg.instruction_prefix.rstrip("\n")
            base_field = raw_compress or "prompt_0shot"
            for sample in samples:
                base_text = sample.get(base_field, sample.get("prompt_0shot", ""))
                sample["verbose_instruction"] = prefix + "\n\n" + base_text

        return samples

    @staticmethod
    def _synthetic_dataset(n: int = 50) -> list[dict]:
        """Minimal synthetic dataset for dry-run / CI testing."""
        templates = [
            "Please kindly classify the sentiment of the following text.",
            "Your task is to analyze whether the text is positive or negative.",
            "Could you determine the sentiment label for the sentence below?",
            "In order to complete this task, read the context and answer.",
            "You are required to provide a concise answer to the question.",
        ]
        contexts = [
            "Example: The movie was great. Answer: positive\n\n",
            "Example: I hated the food. Answer: negative\n\n",
        ]
        return [
            {
                # Full formatted prompt (template + simulated few-shot context)
                "prompt": contexts[i % len(contexts)] + templates[i % len(templates)],
                # 0-shot prompt (no few-shot examples prepended)
                "prompt_0shot": templates[i % len(templates)],
                "target": "positive" if i % 2 == 0 else "negative",
            }
            for i in range(n)
        ]

    @staticmethod
    def _resolve_transforms(run_cfg: RunConfig):
        if run_cfg.transformation_ids is None:
            return ALL_TRANSFORMATIONS
        id_set = set(run_cfg.transformation_ids)
        result = [T for T in ALL_TRANSFORMATIONS if T.id in id_set]
        if not result:
            raise ValueError(f"No transformations matched ids: {run_cfg.transformation_ids}")
        return result

    @staticmethod
    def _make_exp_config(run_cfg: RunConfig, n_subset: int) -> ExperimentConfig:
        return ExperimentConfig(
            prompt_field=run_cfg.prompt_field,
            compress_field=run_cfg.effective_compress_field,
            n_subset=n_subset,
            subset_seed=run_cfg.subset_seed,
            algorithm=run_cfg.algorithm,
            pop_size=run_cfg.pop_size,
            n_gen=run_cfg.n_gen,
            seed=run_cfg.seed,
            mab_reward_weight=run_cfg.mab_reward_weight,
            transformation_ids=run_cfg.transformation_ids,
            search_transformation_ids=run_cfg.search_transformation_ids,
            preserve_answer_cue=run_cfg.preserve_answer_cue,
            verbose=False,
        )

    def _run_fitting(
        self,
        run_cfg: RunConfig,
        dataset: list[dict],
        semantic_backend,
    ) -> tuple[dict, dict]:
        """Run :class:`DatasetVocabularyFitter` and return ``(fitted, report)``.

        *fitted* is the ``{operator_id: Transformation}`` dict from
        :meth:`~DatasetVocabularyFitter.fit_all`.  *report* is the human-readable
        summary from :meth:`~DatasetVocabularyFitter.report`.

        The *semantic_backend* (an :class:`LMEvalSemanticBackend` or ``None``)
        is forwarded to the fitter so it can additionally discover structural
        patterns (header formats, tag names, chunk separators) via the LLM when
        ``use_llm_for_semantic=True``.  When ``None``, only corpus-based fitting
        is performed.
        """
        from prompt_piper.transformations.fitting import DatasetVocabularyFitter
        from prompt_piper.transformations.sections import PromptSplitter

        # The fitter sees the text the operators act on: the uncompressed
        # assembled prompt (instruction + shots + data + cue).  Frequency-based
        # fitters count only the per-sample data section (see frequency_field).
        compress_field = run_cfg.effective_compress_field or run_cfg.prompt_field
        sections = PromptSplitter(dataset, compress_field=compress_field).split_all()
        corpus = [
            {**sample, "_fit_text": sec.assemble(), "_fit_varying": sec.input_data}
            for sample, sec in zip(dataset, sections)
        ]
        logger.info("Running DatasetVocabularyFitter on %d samples…", len(corpus))
        fitter = DatasetVocabularyFitter(
            corpus,
            prompt_field="_fit_text",
            frequency_field="_fit_varying",
            template_field=run_cfg.effective_compress_field,
            compress_field=run_cfg.effective_compress_field,
            seed=run_cfg.seed,
            backend=semantic_backend,  # None → corpus-only; set → also LLM-backed
        )
        fitted = fitter.fit_all()
        report = fitter.report()
        logger.info(
            "Fitting complete: %d operators fitted, %d new stopwords, "
            "%d new abbreviations, %d new sections%s",
            len(fitted),
            len(report.get("stopwords", [])),
            len(report.get("abbreviations", [])),
            len(report.get("sections", [])),
            (
                f", {len(report.get('tag_names', []))} tag(s), "
                f"separator={report.get('chunk_separator')!r}"
                if "tag_names" in report else ""
            ),
        )
        return fitted, report

    def _compose_factory(self, fitted: dict, semantic_backend) -> Any:
        """Return a transformation factory that merges fitted instances with the LLM backend.

        Priority order for each operator class:

        1. **O28 (ChunkCompression) with LLM backend** — the fitted instance
           carries a corpus-discovered *separator*; the LLM backend handles
           per-chunk compression.  Both are combined into a single instance.
        2. **Any other fitted operator** — the fitted instance is returned
           as-is (corpus-adapted, no LLM needed).
        3. **Semantic operator with LLM backend** — instantiated with the
           backend (heuristic fallback overridden by LLM).
        4. **Everything else** — default no-arg constructor.
        """
        import inspect
        from prompt_piper.transformations.semantic import ChunkCompression

        def factory(cls):
            fitted_inst = fitted.get(cls.id)

            if fitted_inst is not None:
                # O28: blend the fitted separator with the semantic backend.
                if cls.id == "O28" and semantic_backend is not None:
                    return ChunkCompression(
                        backend=semantic_backend,
                        separator=fitted_inst._separator,
                    )
                return fitted_inst

            # Non-fitted semantic operator: inject the LLM backend.
            if semantic_backend is not None and cls.category == "semantic":
                if "backend" in inspect.signature(cls.__init__).parameters:
                    return cls(backend=semantic_backend)

            return cls()

        return factory

    def _output_path(self, run_cfg: RunConfig) -> Path:
        out_dir = Path(self._config.output.dir)
        path = out_dir / f"{run_cfg.run_name}.json"
        if path.exists() and not self._config.output.overwrite:
            # Append timestamp suffix to avoid overwriting.
            ts = CompressorResult.now_timestamp().replace(":", "-")
            path = out_dir / f"{run_cfg.run_name}_{ts}.json"
        return path

    def _serialise_config(self, run_cfg: RunConfig) -> dict:
        return {
            "model": {
                "type": self._config.model.type,
                "name": self._config.model.name,
                "kwargs": self._config.model.kwargs,
            },
            "run": {f: getattr(run_cfg, f) for f in run_cfg.__dataclass_fields__},
        }