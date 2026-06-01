"""Experiment runner — ties together config, problem, algorithm, and baselines."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pymoo.optimize import minimize

from prompt_piper.transformations import (
    ALL_TRANSFORMATIONS,
    DATA_OPS,
    INSTRUCTION_OPS,
    SHOT_OPS,
)
from prompt_piper.transformations.base import Transformation
from prompt_piper.transformations.sections import PromptSplitter

from .algorithms import build_algorithm
from .baselines import (
    AllTransformationsBaseline,
    BaselineResult,
    LLMLingua2Baseline,
    MultiArmedBanditBaseline,
    NoCompressionBaseline,
    RandomDeletionBaseline,
    RandomSearchBaseline,
    SummarizationBaseline,
    TruncatedPromptBaseline,
)
from .config import ExperimentConfig
from .evaluator import Evaluator, MockEvaluator
from .problem import PromptCompressionProblem
from .tokenizer import Tokenizer, WhitespaceTokenizer

_BASELINE_REGISTRY = {
    "no_compression": NoCompressionBaseline,
    "all_transformations": AllTransformationsBaseline,
    "random_search": RandomSearchBaseline,
    "truncated_prompt": TruncatedPromptBaseline,
    "random_deletion": RandomDeletionBaseline,
    "summarization": SummarizationBaseline,
    "multi_armed_bandit": MultiArmedBanditBaseline,
    "llmlingua2": LLMLingua2Baseline,
}


@dataclass
class ExperimentResult:
    """All outputs from a completed experiment.

    Attributes
    ----------
    config:
        The configuration used.
    pareto_solutions:
        Binary matrix of shape ``(n_pareto, n_var)``.  Each row is a
        Pareto-optimal transformation mask.
    pareto_objectives:
        Matrix of shape ``(n_pareto, 2)`` with columns
        ``[avg_tokens, avg_performance]``.  Performance is stored as a
        *positive* value (not negated).
    pareto_transformation_names:
        Human-readable operator names for each Pareto solution.
    pymoo_result:
        Raw result object returned by ``pymoo.optimize.minimize``.
    elapsed_time:
        Wall-clock seconds for the optimisation run.
    baselines:
        Dict mapping baseline name → :class:`BaselineResult`.
    """

    config: ExperimentConfig
    pareto_solutions: np.ndarray
    pareto_objectives: np.ndarray
    pareto_transformation_names: list[list[str]]
    pymoo_result: Any
    elapsed_time: float
    baselines: dict[str, BaselineResult] = field(default_factory=dict)
    baseline_elapsed: dict[str, float] = field(default_factory=dict)
    generation_history: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def avg_tokens(self) -> np.ndarray:
        """Average token counts for each Pareto solution."""
        return self.pareto_objectives[:, 0]

    @property
    def avg_performance(self) -> np.ndarray:
        """Average task performance for each Pareto solution."""
        return self.pareto_objectives[:, 1]

    def best_compression(self) -> tuple[np.ndarray, float, float]:
        """Return the solution with the fewest tokens (most compressed).

        Returns
        -------
        tuple[ndarray, float, float]
            ``(bits, avg_tokens, avg_performance)``
        """
        idx = int(np.argmin(self.avg_tokens))
        return self.pareto_solutions[idx], self.avg_tokens[idx], self.avg_performance[idx]

    def best_performance(self) -> tuple[np.ndarray, float, float]:
        """Return the solution with the highest task performance.

        Returns
        -------
        tuple[ndarray, float, float]
            ``(bits, avg_tokens, avg_performance)``
        """
        idx = int(np.argmax(self.avg_performance))
        return self.pareto_solutions[idx], self.avg_tokens[idx], self.avg_performance[idx]

    def summary(self) -> str:
        """Return a short human-readable summary of the Pareto front."""
        lines = [
            f"Experiment: {self.config.algorithm.upper()} | "
            f"{self.config.n_gen} gen × {self.config.pop_size} pop | "
            f"{self.elapsed_time:.1f}s",
            f"Pareto front: {len(self.pareto_solutions)} solutions",
            f"  Token range:       {self.avg_tokens.min():.1f} – {self.avg_tokens.max():.1f}",
            f"  Performance range: {self.avg_performance.min():.3f} – {self.avg_performance.max():.3f}",
        ]
        if self.baselines:
            lines.append("Baselines:")
            for name, br in self.baselines.items():
                lines.append(
                    f"  {name}: tokens={br.avg_tokens.mean():.1f}, "
                    f"perf={br.avg_performance.mean():.3f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """Configure and run a prompt-compression optimisation experiment.

    Parameters
    ----------
    config:
        Experiment configuration.
    dataset:
        Full dataset as a list of sample dicts.
    evaluator:
        Task evaluator.  Defaults to :class:`MockEvaluator`.
    tokenizer:
        Token counter.  Defaults to :class:`WhitespaceTokenizer`.
    transformations:
        Ordered list of :class:`Transformation` *classes* to include in
        the search space.  Defaults to all 28 from the paper, filtered by
        ``config.transformation_ids`` if set.
    transformation_factory:
        Optional callable ``(cls: type[Transformation]) -> Transformation``
        invoked to instantiate each transformation.  Use this to inject
        backends (e.g. an LLM) into semantic transformations without
        changing the class interface.  When ``None`` (default), each class
        is instantiated with no arguments: ``cls()``.

    Example
    -------
    >>> config = ExperimentConfig(n_gen=20, pop_size=20, n_subset=10)
    >>> dataset = [{"prompt": "Classify this text.", "target": "positive"}]
    >>> exp = Experiment(config, dataset)
    >>> result = exp.run()
    >>> print(result.summary())
    """

    def __init__(
        self,
        config: ExperimentConfig,
        dataset: list[dict],
        evaluator: Evaluator | None = None,
        tokenizer: Tokenizer | None = None,
        transformations: list[type[Transformation]] | None = None,
        transformation_factory=None,
    ) -> None:
        self.config = config
        self._full_dataset = dataset
        self._evaluator = evaluator or MockEvaluator()
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        # Full operator pool (filtered by transformation_ids) — used by
        # all_transformations and summarization baselines.
        self._transform_classes = self._resolve_transformations(transformations)
        # Search-only pool (further filtered by search_transformation_ids) — used
        # by NSGA-II, random_search, and multi_armed_bandit.
        self._search_transform_classes = self._resolve_search_transformations()
        self._transformation_factory = transformation_factory or (lambda cls: cls())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        baselines: list[str] | None = None,
        callback=None,
    ) -> ExperimentResult:
        """Run the search and optionally one or more baselines.

        When ``config.algorithm == "mab"``, the Multi-Armed Bandit (Thompson
        Sampling) search is used as the primary algorithm instead of a pymoo
        evolutionary algorithm.  In that case *callback* is ignored and
        ``ExperimentResult.generation_history`` will be empty.

        Parameters
        ----------
        baselines:
            List of baseline names to run alongside the main search.
            Pass ``None`` (default) to skip baselines.
        callback:
            Optional pymoo :class:`~pymoo.core.callback.Callback` instance
            for per-generation history tracking.  Ignored when algorithm is
            ``"mab"``.

        Returns
        -------
        ExperimentResult
        """
        dataset_subset = self._sample_subset()
        problem = self._build_problem(dataset_subset)

        if self.config.algorithm == "mab":
            return self._run_mab(problem, dataset_subset, baselines)

        algorithm = build_algorithm(
            self.config.algorithm,
            n_var=problem.n_var,
            pop_size=self.config.pop_size,
        )

        t0 = time.perf_counter()
        minimize_kwargs = dict(
            seed=self.config.seed,
            verbose=self.config.verbose,
        )
        if callback is not None:
            minimize_kwargs["callback"] = callback
        res = minimize(
            problem,
            algorithm,
            ("n_gen", self.config.n_gen),
            **minimize_kwargs,
        )
        elapsed = time.perf_counter() - t0

        pareto_solutions, pareto_objectives, pareto_names = self._parse_result(res, problem)
        history = getattr(callback, "history", [])

        baseline_results, baseline_elapsed = self._run_baselines(
            baselines or [], problem, dataset_subset
        )

        return ExperimentResult(
            config=self.config,
            pareto_solutions=pareto_solutions,
            pareto_objectives=pareto_objectives,
            pareto_transformation_names=pareto_names,
            pymoo_result=res,
            elapsed_time=elapsed,
            baselines=baseline_results,
            baseline_elapsed=baseline_elapsed,
            generation_history=history,
        )

    def _run_mab(
        self,
        problem: PromptCompressionProblem,
        dataset_subset: list[dict],
        baselines: list[str] | None,
    ) -> ExperimentResult:
        """Run Multi-Armed Bandit (Thompson Sampling) as the primary search.

        Uses the same evaluation budget as the evolutionary approach:
        ``pop_size × n_gen`` evaluations.  Baselines are run on the same
        problem instance afterward.
        """
        n_eval = self.config.pop_size * self.config.n_gen
        mab = MultiArmedBanditBaseline(
            n_evaluations=n_eval,
            seed=self.config.seed,
        )

        t0 = time.perf_counter()
        mab_result = mab.run(problem)
        elapsed = time.perf_counter() - t0

        pareto_solutions = mab_result.solutions      # (n_pareto, n_var)
        pareto_objectives = mab_result.objectives    # (n_pareto, 2): [tokens, +perf]
        pareto_names = mab_result.transformation_names

        baseline_results, baseline_elapsed = self._run_baselines(
            baselines or [], problem, dataset_subset
        )

        return ExperimentResult(
            config=self.config,
            pareto_solutions=pareto_solutions,
            pareto_objectives=pareto_objectives,
            pareto_transformation_names=pareto_names,
            pymoo_result=None,
            elapsed_time=elapsed,
            baselines=baseline_results,
            baseline_elapsed=baseline_elapsed,
            generation_history=[],
        )

    def evaluate_solutions(
        self,
        solutions: np.ndarray,
        dataset: list[dict],
    ) -> np.ndarray:
        """Score a set of binary solution vectors on an arbitrary dataset.

        Used to re-evaluate search-phase Pareto solutions on held-out test
        data without re-running the optimisation.

        Parameters
        ----------
        solutions:
            Integer array of shape ``(n_solutions, n_var)``.
        dataset:
            List of sample dicts to evaluate on.

        Returns
        -------
        np.ndarray
            Shape ``(n_solutions, 2)`` — columns ``[avg_tokens, avg_performance]``
            (performance is *positive*).
        """
        problem = self._build_problem(dataset)
        objectives = np.empty((len(solutions), 2), dtype=float)
        for i, bits in enumerate(solutions):
            tokens, perf = problem.score_solution(bits)
            objectives[i] = [tokens, perf]
        return objectives

    def run_baseline(
        self,
        name: str,
        problem: PromptCompressionProblem | None = None,
    ) -> BaselineResult:
        """Run a single named baseline.

        Parameters
        ----------
        name:
            One of ``"no_compression"``, ``"all_transformations"``,
            ``"random_search"``.
        problem:
            Pre-built problem instance (shares the same dataset subset).
            If ``None``, a new subset is sampled and a new problem built.
        """
        if name not in _BASELINE_REGISTRY:
            raise ValueError(
                f"Unknown baseline {name!r}. Available: {sorted(_BASELINE_REGISTRY)}"
            )
        if problem is None:
            problem = self._build_problem(self._sample_subset())

        cls = _BASELINE_REGISTRY[name]
        if name in ("random_search", "multi_armed_bandit"):
            n_eval = self.config.pop_size * self.config.n_gen
            baseline = cls(n_evaluations=n_eval, seed=self.config.seed)
        elif name == "random_deletion":
            baseline = cls(seed=self.config.seed)
        else:
            baseline = cls()

        return baseline.run(problem)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Baselines that must run on the full operator set (including semantic ops)
    # so they remain meaningful even when the search is restricted to O1–O22.
    _FULL_OP_BASELINES: frozenset[str] = frozenset({"all_transformations", "summarization"})

    def _run_baselines(
        self,
        baselines: list[str],
        search_problem: PromptCompressionProblem,
        dataset_subset: list[dict],
    ) -> tuple[dict[str, BaselineResult], dict[str, float]]:
        """Run all baselines, routing full-op ones to a separate problem instance.

        ``all_transformations`` and ``summarization`` always use the complete
        operator set (``_transform_classes``) so that their results are
        independent of ``search_transformation_ids``.  All other baselines
        share the search problem and therefore operate in the same restricted
        operator space as NSGA-II.
        """
        results: dict[str, BaselineResult] = {}
        elapsed: dict[str, float] = {}
        if not baselines:
            return results, elapsed

        needs_full = any(name in self._FULL_OP_BASELINES for name in baselines)
        full_problem = self._build_full_problem(dataset_subset) if needs_full else None

        for name in baselines:
            t = time.perf_counter()
            problem = full_problem if name in self._FULL_OP_BASELINES else search_problem
            results[name] = self.run_baseline(name, problem)
            elapsed[name] = time.perf_counter() - t

        return results, elapsed

    def _resolve_transformations(
        self, classes: list[type[Transformation]] | None
    ) -> list[type[Transformation]]:
        """Filter and order transformation classes per config."""
        pool = classes if classes is not None else ALL_TRANSFORMATIONS
        if self.config.transformation_ids is not None:
            id_set = set(self.config.transformation_ids)
            pool = [T for T in pool if T.id in id_set]
            if not pool:
                raise ValueError(
                    f"No transformations matched ids {self.config.transformation_ids!r}"
                )
        return pool

    def _resolve_search_transformations(self) -> list[type[Transformation]]:
        """Further restrict the full pool to the search-only subset.

        When ``config.search_transformation_ids`` is ``None`` the full
        ``_transform_classes`` pool is returned unchanged.
        """
        if self.config.search_transformation_ids is None:
            return self._transform_classes
        id_set = set(self.config.search_transformation_ids)
        pool = [T for T in self._transform_classes if T.id in id_set]
        if not pool:
            raise ValueError(
                f"No transformations matched search_transformation_ids "
                f"{self.config.search_transformation_ids!r}"
            )
        return pool

    def _resolve_scoped_classes(
        self, all_classes: list[type[Transformation]]
    ) -> tuple[
        list[type[Transformation]],
        list[type[Transformation]],
        list[type[Transformation]],
    ]:
        """Return (instruction_classes, shot_classes, data_classes) for *all_classes*.

        Filters each scope list to the intersection with *all_classes* — the
        caller is responsible for pre-filtering *all_classes* to the desired
        operator pool (search-only or full).
        """
        allowed = set(all_classes)

        def _filter(scope_pool: list[type[Transformation]]) -> list[type[Transformation]]:
            return [cls for cls in scope_pool if cls in allowed]

        i_classes = _filter(INSTRUCTION_OPS)
        s_classes = _filter(SHOT_OPS)
        d_classes = _filter(DATA_OPS) if self.config.compress_input_data else []
        return i_classes, s_classes, d_classes

    def _sample_subset(self) -> list[dict]:
        """Draw a reproducible subset from the full dataset."""
        n = self.config.n_subset
        if n is None or n >= len(self._full_dataset):
            return self._full_dataset
        rng = random.Random(self.config.subset_seed)
        return rng.sample(self._full_dataset, n)

    def _build_problem(self, dataset: list[dict]) -> PromptCompressionProblem:
        """Build the search problem using the restricted operator set."""
        return self._build_problem_with_classes(dataset, self._search_transform_classes)

    def _build_full_problem(self, dataset: list[dict]) -> PromptCompressionProblem:
        """Build a problem with the full operator set (for all_transformations / summarization)."""
        return self._build_problem_with_classes(dataset, self._transform_classes)

    def _build_problem_with_classes(
        self, dataset: list[dict], transform_classes: list[type[Transformation]]
    ) -> PromptCompressionProblem:
        """Instantiate the pymoo problem using the given operator pool."""
        compress_field = self.config.compress_field or self.config.prompt_field
        i_classes, s_classes, d_classes = self._resolve_scoped_classes(transform_classes)
        # When compressing a sub-field (e.g. 'prompt_0shot'), the few-shot
        # demonstrations are loaded from sample['_shots'] for assembly/evaluation
        # but must remain intact — only data-section operators are applicable.
        if compress_field != self.config.prompt_field:
            s_classes = []

        # Build sections from the dataset using PromptSplitter.
        splitter = PromptSplitter(
            samples=dataset,
            compress_field=compress_field,
        )
        sections = splitter.split_all()

        i_ops = [self._transformation_factory(cls) for cls in i_classes]
        s_ops = [self._transformation_factory(cls) for cls in s_classes]
        d_ops = [self._transformation_factory(cls) for cls in d_classes]

        return PromptCompressionProblem(
            instruction_ops=i_ops,
            shot_ops=s_ops,
            data_ops=d_ops,
            sections=sections,
            dataset=dataset,
            evaluator=self._evaluator,
            tokenizer=self._tokenizer,
            prompt_field=self.config.prompt_field,
            compress_field=compress_field,
        )

    @staticmethod
    def _parse_result(
        res, problem: PromptCompressionProblem
    ) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
        """Extract Pareto solutions from a pymoo result object."""
        X = res.X.astype(int)  # (n_pareto, n_var)
        F = res.F.copy()       # (n_pareto, 2): [avg_tokens, -avg_perf]
        F[:, 1] = -F[:, 1]    # restore positive performance

        names = [problem.names_from_bits(row) for row in X]
        return X, F, names