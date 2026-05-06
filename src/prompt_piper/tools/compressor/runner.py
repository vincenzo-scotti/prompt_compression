"""CompressorRunner — orchestrates multi-run experiments with model reuse."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from prompt_piper.optimization.config import ExperimentConfig
from prompt_piper.optimization.evaluator import MockEvaluator
from prompt_piper.optimization.runner import Experiment
from prompt_piper.transformations import ALL_TRANSFORMATIONS

from .callback import GenerationCallback
from .config import CompressorConfig, RunConfig
from .results import BaselineEval, CompressorResult, SolutionEval

logger = logging.getLogger(__name__)


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
            except Exception:
                logger.exception("Run %s failed", run_cfg.run_name)
                raise
        return results

    def run_single(self, run_cfg: RunConfig) -> CompressorResult:
        """Execute one (task, algorithm) run and return a :class:`CompressorResult`.

        When ``run_cfg.baselines_only`` is ``True`` the NSGA-II optimisation
        phase is skipped; only the requested baselines are evaluated.
        """
        timestamp = CompressorResult.now_timestamp()

        backend = self._get_backend(run_cfg)
        transform_classes = self._resolve_transforms(run_cfg)
        transform_ids = [T.id for T in transform_classes]
        transform_names = [T.name for T in transform_classes]

        # ---- Fitting phase (optional) ------------------------------------
        search_dataset = self._get_dataset(backend, run_cfg, split=run_cfg.search_split)
        semantic_backend = backend.get("semantic_backend")

        if run_cfg.use_fitting:
            fitted, fitting_report = self._run_fitting(
                run_cfg, search_dataset, semantic_backend
            )
            factory = self._compose_factory(fitted, semantic_backend)
        else:
            fitting_report = None
            factory = backend.get("transformation_factory")

        exp_config = self._make_exp_config(run_cfg, n_subset=run_cfg.n_search)
        exp = Experiment(
            config=exp_config,
            dataset=search_dataset,
            evaluator=backend["evaluator"],
            tokenizer=backend.get("tokenizer"),
            transformations=transform_classes,
            transformation_factory=factory,
        )

        # ---- Test dataset ------------------------------------------------
        test_dataset = self._get_dataset(backend, run_cfg, split=run_cfg.test_split)
        if run_cfg.n_test is not None:
            import random
            rng = random.Random(run_cfg.subset_seed + 1)
            if run_cfg.n_test < len(test_dataset):
                test_dataset = rng.sample(test_dataset, run_cfg.n_test)

        # ================================================================
        # BASELINES-ONLY mode: skip NSGA-II, run baselines on both splits
        # ================================================================
        if run_cfg.baselines_only:
            return self._run_baselines_only(
                run_cfg, exp, test_dataset,
                transform_ids, transform_names,
                timestamp, fitting_report,
            )

        # ---- Search phase ------------------------------------------------
        callback = GenerationCallback(transform_names)

        t_search_start = time.perf_counter()
        exp_result = exp.run(
            baselines=run_cfg.baselines,
            callback=callback,
        )
        search_elapsed = time.perf_counter() - t_search_start

        # ---- Test evaluation phase ----------------------------------------
        t_eval_start = time.perf_counter()
        test_objectives = exp.evaluate_solutions(exp_result.pareto_solutions, test_dataset)

        # Baseline test evaluation — re-evaluate bit vectors on the test split.
        # Custom baselines (truncated_prompt, random_deletion) store all-zero
        # bit vectors which would evaluate as no-compression; run them on the
        # test problem directly instead.
        _CUSTOM_BASELINES = {"truncated_prompt", "random_deletion"}
        baseline_test: dict[str, tuple[float, float]] = {}
        for name, br in exp_result.baselines.items():
            if name in _CUSTOM_BASELINES:
                test_br = exp.run_baseline(name, problem=exp._build_problem(test_dataset))
                avg_tok = float(test_br.avg_tokens.mean())
                avg_perf = float(test_br.avg_performance.mean())
            else:
                bl_test_obj = exp.evaluate_solutions(br.solutions, test_dataset)
                avg_tok = float(bl_test_obj[:, 0].mean())
                avg_perf = float(bl_test_obj[:, 1].mean())
            baseline_test[name] = (avg_tok, avg_perf)

        eval_elapsed = time.perf_counter() - t_eval_start

        # ---- Assemble result ----------------------------------------------
        pareto_evals = []
        for i, (bits, names) in enumerate(
            zip(exp_result.pareto_solutions, exp_result.pareto_transformation_names)
        ):
            ids = [
                T.id for T, b in zip(transform_classes, bits) if b
            ]
            pareto_evals.append(
                SolutionEval(
                    transformation_ids=ids,
                    transformation_names=names,
                    bits=bits.tolist(),
                    search_avg_tokens=float(exp_result.pareto_objectives[i, 0]),
                    search_avg_performance=float(exp_result.pareto_objectives[i, 1]),
                    test_avg_tokens=float(test_objectives[i, 0]),
                    test_avg_performance=float(test_objectives[i, 1]),
                )
            )

        baseline_evals = []
        for name, br in exp_result.baselines.items():
            b_test_tok, b_test_perf = baseline_test.get(name, (None, None))
            baseline_evals.append(
                BaselineEval(
                    name=name,
                    transformation_names=br.transformation_names,
                    search_avg_tokens=float(br.avg_tokens.mean()),
                    search_avg_performance=float(br.avg_performance.mean()),
                    test_avg_tokens=b_test_tok,
                    test_avg_performance=b_test_perf,
                )
            )

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
        )

    def _run_baselines_only(
        self,
        run_cfg: RunConfig,
        exp: "Experiment",
        test_dataset: list[dict],
        transform_ids: list[str],
        transform_names: list[str],
        timestamp: str,
        fitting_report,
    ) -> CompressorResult:
        """Run baselines on both search and test splits; no NSGA-II optimisation."""
        search_problem = exp._build_problem(exp._sample_subset())
        test_problem = exp._build_problem(test_dataset)

        t_eval_start = time.perf_counter()
        baseline_evals = []
        for name in (run_cfg.baselines or []):
            search_br = exp.run_baseline(name, problem=search_problem)
            test_br = exp.run_baseline(name, problem=test_problem)
            baseline_evals.append(
                BaselineEval(
                    name=name,
                    transformation_names=search_br.transformation_names,
                    search_avg_tokens=float(search_br.avg_tokens.mean()),
                    search_avg_performance=float(search_br.avg_performance.mean()),
                    test_avg_tokens=float(test_br.avg_tokens.mean()),
                    test_avg_performance=float(test_br.avg_performance.mean()),
                )
            )
        eval_elapsed = time.perf_counter() - t_eval_start

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

        model = self._get_model()
        task = _load_task(run_cfg.task, num_fewshot=run_cfg.num_fewshot)
        evaluator = LMEvalEvaluator(
            model, task,
            generation_kwargs=run_cfg.generation_kwargs,
            extra_stop_sequences=run_cfg.extra_stop_sequences,
            metric_key=run_cfg.metric_key,
        )
        tokenizer = LMEvalTokenizer(model)
        semantic_backend = (
            LMEvalSemanticBackend(model) if run_cfg.use_llm_for_semantic else None
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
        """Return a list of sample dicts for the given split."""
        if self._dry_run or backend["task"] is None:
            # Generate synthetic data for dry runs
            return self._synthetic_dataset(n=max(run_cfg.n_search, run_cfg.n_test or 0) + 10)

        # When compressing a field other than the main prompt, ask the dataset
        # to also store the 0-shot template under that field name.
        template_field = (
            run_cfg.compress_field
            if run_cfg.compress_field and run_cfg.compress_field != run_cfg.prompt_field
            else None
        )

        from prompt_piper.backends.lm_eval import LMEvalDataset
        ds = LMEvalDataset(
            backend["task"],
            split=split,
            num_fewshot=run_cfg.num_fewshot,
            prompt_field=run_cfg.prompt_field,
            template_field=template_field,
        )
        return ds.as_list()

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
            compress_field=run_cfg.compress_field,
            n_subset=n_subset,
            subset_seed=run_cfg.subset_seed,
            algorithm=run_cfg.algorithm,
            pop_size=run_cfg.pop_size,
            n_gen=run_cfg.n_gen,
            seed=run_cfg.seed,
            transformation_ids=run_cfg.transformation_ids,
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

        logger.info("Running DatasetVocabularyFitter on %d samples…", len(dataset))
        fitter = DatasetVocabularyFitter(
            dataset,
            prompt_field=run_cfg.prompt_field,
            template_field=run_cfg.compress_field,
            compress_field=run_cfg.compress_field,
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