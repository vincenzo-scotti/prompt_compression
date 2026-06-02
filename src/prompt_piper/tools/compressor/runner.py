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
from .results import BaselineEval, CompressorResult, SolutionEval, TransferabilityEval

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

        backend = self._get_backend(run_cfg)
        transform_classes = self._resolve_transforms(run_cfg)
        transform_ids = [T.id for T in transform_classes]
        transform_names = [T.name for T in transform_classes]

        # ---- Fitting phase (optional) ------------------------------------
        search_dataset = self._get_dataset(backend, run_cfg, split=run_cfg.search_split)
        semantic_backend = backend.get("semantic_backend")

        if run_cfg.use_fitting:
            t_fit_start = time.perf_counter()
            fitted, fitting_report = self._run_fitting(
                run_cfg, search_dataset, semantic_backend
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
                timestamp, fitting_report, fitting_elapsed,
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
        # Baselines that need re-running (not bit-vector re-evaluation) on test data:
        # - custom baselines use score_transform_fn (no bit vector to replay)
        # - full-op baselines used a different problem instance during search;
        #   their bit vectors have a different length than the search problem.
        _CUSTOM_BASELINES = {"truncated_prompt", "random_deletion", "llmlingua2"}
        _FULL_OP_BASELINES = exp._FULL_OP_BASELINES  # {"all_transformations", "summarization"}

        full_test_problem = None  # built lazily
        baseline_test: dict[str, tuple[float, float, float]] = {}
        for name, br in exp_result.baselines.items():
            t_bl_test = time.perf_counter()
            if name in _CUSTOM_BASELINES:
                test_br = exp.run_baseline(name, problem=exp._build_problem(test_dataset))
                avg_tok = float(test_br.avg_tokens.mean())
                avg_perf = float(test_br.avg_performance.mean())
            elif name in _FULL_OP_BASELINES:
                if full_test_problem is None:
                    full_test_problem = exp._build_full_problem(test_dataset)
                test_br = exp.run_baseline(name, problem=full_test_problem)
                avg_tok = float(test_br.avg_tokens.mean())
                avg_perf = float(test_br.avg_performance.mean())
            else:
                bl_test_obj = exp.evaluate_solutions(br.solutions, test_dataset)
                avg_tok = float(bl_test_obj[:, 0].mean())
                avg_perf = float(bl_test_obj[:, 1].mean())
            baseline_test[name] = (avg_tok, avg_perf, time.perf_counter() - t_bl_test)

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
            fitting_elapsed_seconds=fitting_elapsed,
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
        fitting_elapsed: float | None = None,
    ) -> CompressorResult:
        """Run baselines on both search and test splits; no NSGA-II optimisation."""
        search_problem = exp._build_problem(exp._sample_subset())
        test_problem = exp._build_problem(test_dataset)

        t_eval_start = time.perf_counter()
        baseline_evals = []
        for name in (run_cfg.baselines or []):
            t_bl = time.perf_counter()
            search_br = exp.run_baseline(name, problem=search_problem)
            test_br = exp.run_baseline(name, problem=test_problem)
            bl_elapsed = time.perf_counter() - t_bl
            baseline_evals.append(
                BaselineEval(
                    name=name,
                    transformation_names=search_br.transformation_names,
                    search_avg_tokens=float(search_br.avg_tokens.mean()),
                    search_avg_performance=float(search_br.avg_performance.mean()),
                    test_avg_tokens=float(test_br.avg_tokens.mean()),
                    test_avg_performance=float(test_br.avg_performance.mean()),
                    elapsed_seconds=bl_elapsed,
                    config=search_br.config,
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
            fitting_elapsed_seconds=fitting_elapsed,
        )

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
        target_dataset = self._get_dataset(target_backend, target_run_cfg, split=target_run_cfg.test_split)
        if target_run_cfg.n_test is not None:
            import random
            rng = random.Random(target_run_cfg.subset_seed + 1)
            if target_run_cfg.n_test < len(target_dataset):
                target_dataset = rng.sample(target_dataset, target_run_cfg.n_test)

        exp_config = self._make_exp_config(target_run_cfg, n_subset=len(target_dataset))
        exp = Experiment(
            config=exp_config,
            dataset=target_dataset,
            evaluator=target_backend["evaluator"],
            tokenizer=target_backend.get("tokenizer"),
            transformations=transform_classes,
            transformation_factory=target_backend.get("transformation_factory"),
        )

        # Reconstruct bit vectors from source Pareto solutions
        import numpy as np
        source_bits = np.array([sol.bits for sol in source_result.pareto_solutions], dtype=int)

        # Evaluate on target dataset
        target_objectives = exp.evaluate_solutions(source_bits, target_dataset)

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
            LMEvalSemanticBackend(model, apply_chat_template=run_cfg.apply_chat_template)
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
        """Return a list of sample dicts for the given split."""
        if self._dry_run or backend["task"] is None:
            samples = self._synthetic_dataset(n=max(run_cfg.n_search, run_cfg.n_test or 0) + 10)
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
            transformation_ids=run_cfg.transformation_ids,
            search_transformation_ids=run_cfg.search_transformation_ids,
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