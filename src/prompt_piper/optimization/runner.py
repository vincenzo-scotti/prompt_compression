"""Experiment runner — ties together config, problem, algorithm, and baselines."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pymoo.optimize import minimize

from prompt_piper.transformations import ALL_TRANSFORMATIONS
from prompt_piper.transformations.base import Transformation

from .algorithms import build_algorithm
from .baselines import (
    AllTransformationsBaseline,
    BaselineResult,
    NoCompressionBaseline,
    RandomSearchBaseline,
)
from .config import ExperimentConfig
from .evaluator import Evaluator, MockEvaluator
from .problem import PromptCompressionProblem
from .tokenizer import Tokenizer, WhitespaceTokenizer

_BASELINE_REGISTRY = {
    "no_compression": NoCompressionBaseline,
    "all_transformations": AllTransformationsBaseline,
    "random_search": RandomSearchBaseline,
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
    ) -> None:
        self.config = config
        self._full_dataset = dataset
        self._evaluator = evaluator or MockEvaluator()
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._transform_classes = self._resolve_transformations(transformations)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, baselines: list[str] | None = None) -> ExperimentResult:
        """Run the evolutionary search and optionally one or more baselines.

        Parameters
        ----------
        baselines:
            List of baseline names to run alongside the main search.
            Supported: ``"no_compression"``, ``"all_transformations"``,
            ``"random_search"``.  Pass ``None`` (default) to skip baselines.

        Returns
        -------
        ExperimentResult
        """
        dataset_subset = self._sample_subset()
        problem = self._build_problem(dataset_subset)
        algorithm = build_algorithm(
            self.config.algorithm,
            n_var=problem.n_var,
            pop_size=self.config.pop_size,
        )

        t0 = time.perf_counter()
        res = minimize(
            problem,
            algorithm,
            ("n_gen", self.config.n_gen),
            seed=self.config.seed,
            verbose=self.config.verbose,
        )
        elapsed = time.perf_counter() - t0

        pareto_solutions, pareto_objectives, pareto_names = self._parse_result(res, problem)

        baseline_results: dict[str, BaselineResult] = {}
        if baselines:
            for name in baselines:
                baseline_results[name] = self.run_baseline(name, problem)

        return ExperimentResult(
            config=self.config,
            pareto_solutions=pareto_solutions,
            pareto_objectives=pareto_objectives,
            pareto_transformation_names=pareto_names,
            pymoo_result=res,
            elapsed_time=elapsed,
            baselines=baseline_results,
        )

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
        if name == "random_search":
            n_eval = self.config.pop_size * self.config.n_gen
            baseline = cls(n_evaluations=n_eval, seed=self.config.seed)
        else:
            baseline = cls()

        return baseline.run(problem)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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

    def _sample_subset(self) -> list[dict]:
        """Draw a reproducible subset from the full dataset."""
        n = self.config.n_subset
        if n is None or n >= len(self._full_dataset):
            return self._full_dataset
        rng = random.Random(self.config.subset_seed)
        return rng.sample(self._full_dataset, n)

    def _build_problem(self, dataset: list[dict]) -> PromptCompressionProblem:
        """Instantiate the pymoo problem with pre-built transformation objects."""
        instances = [T() for T in self._transform_classes]
        return PromptCompressionProblem(
            transformations=instances,
            dataset=dataset,
            evaluator=self._evaluator,
            tokenizer=self._tokenizer,
            prompt_field=self.config.prompt_field,
        )

    @staticmethod
    def _parse_result(
        res, problem: PromptCompressionProblem
    ) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
        """Extract Pareto solutions from a pymoo result object."""
        X = res.X.astype(int)  # (n_pareto, n_var)
        F = res.F.copy()       # (n_pareto, 2): [avg_tokens, -avg_perf]
        F[:, 1] = -F[:, 1]    # restore positive performance

        names = [
            [t.name for t, b in zip(problem._transforms, row) if b]
            for row in X
        ]
        return X, F, names