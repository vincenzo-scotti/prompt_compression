"""Baseline strategies for comparison with the evolutionary search.

Three baselines are provided:

* :class:`NoCompressionBaseline` — apply no transformations (identity).
* :class:`AllTransformationsBaseline` — apply all transformations in sequence.
* :class:`RandomSearchBaseline` — uniformly random search over the binary
  solution space, using the same evaluation budget as the evolutionary run
  (``pop_size × n_gen`` evaluations).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .problem import PromptCompressionProblem


@dataclass
class BaselineResult:
    """Result from a single baseline run."""

    name: str
    #: Each row is a binary vector (one per non-dominated solution found).
    solutions: np.ndarray
    #: Corresponding objective values: (avg_tokens, avg_performance).
    #: Performance is stored as a *positive* value here (not negated).
    objectives: np.ndarray  # shape (n_solutions, 2)
    #: Human-readable transformation names for each solution.
    transformation_names: list[list[str]]

    @property
    def avg_tokens(self) -> np.ndarray:
        return self.objectives[:, 0]

    @property
    def avg_performance(self) -> np.ndarray:
        return self.objectives[:, 1]


def _names_from_bits(bits: np.ndarray, problem: PromptCompressionProblem) -> list[str]:
    return [t.name for t, b in zip(problem._transforms, bits) if b]


def _is_dominated(f: np.ndarray, population: np.ndarray) -> bool:
    """Return True if *f* is dominated by any member of *population*.

    Assumes minimisation for both objectives.
    """
    return bool(np.any(np.all(population <= f, axis=1) & np.any(population < f, axis=1)))


def _pareto_front(solutions: np.ndarray, objectives: np.ndarray):
    """Filter *solutions* to the non-dominated subset."""
    keep = []
    for i, f in enumerate(objectives):
        others = np.delete(objectives, i, axis=0)
        if not _is_dominated(f, others):
            keep.append(i)
    return solutions[keep], objectives[keep]


# ---------------------------------------------------------------------------
# No-compression baseline
# ---------------------------------------------------------------------------

class NoCompressionBaseline:
    """Apply no transformations; return the original prompt performance."""

    name = "no_compression"

    def run(self, problem: PromptCompressionProblem) -> BaselineResult:
        bits = np.zeros(problem.n_var, dtype=int)
        avg_tokens, avg_perf = problem.score_solution(bits)
        solutions = bits[np.newaxis, :]
        objectives = np.array([[avg_tokens, avg_perf]])
        return BaselineResult(
            name=self.name,
            solutions=solutions,
            objectives=objectives,
            transformation_names=[[]],
        )


# ---------------------------------------------------------------------------
# All-transformations baseline
# ---------------------------------------------------------------------------

class AllTransformationsBaseline:
    """Apply every transformation in sequence."""

    name = "all_transformations"

    def run(self, problem: PromptCompressionProblem) -> BaselineResult:
        bits = np.ones(problem.n_var, dtype=int)
        avg_tokens, avg_perf = problem.score_solution(bits)
        solutions = bits[np.newaxis, :]
        objectives = np.array([[avg_tokens, avg_perf]])
        names = [_names_from_bits(bits, problem)]
        return BaselineResult(
            name=self.name,
            solutions=solutions,
            objectives=objectives,
            transformation_names=names,
        )


# ---------------------------------------------------------------------------
# Random search baseline
# ---------------------------------------------------------------------------

class RandomSearchBaseline:
    """Uniform random search over the binary solution space.

    The evaluation budget matches the evolutionary run:
    ``n_evaluations = pop_size × n_gen``.

    Parameters
    ----------
    n_evaluations:
        Total number of random solutions to evaluate.
    seed:
        Random seed for reproducibility.
    """

    name = "random_search"

    def __init__(self, n_evaluations: int = 2500, seed: int = 42) -> None:
        self._n_eval = n_evaluations
        self._seed = seed

    def run(self, problem: PromptCompressionProblem) -> BaselineResult:
        rng = random.Random(self._seed)
        n_var = problem.n_var

        all_solutions = np.empty((self._n_eval, n_var), dtype=int)
        all_objectives = np.empty((self._n_eval, 2), dtype=float)

        for i in range(self._n_eval):
            bits = np.array([rng.randint(0, 1) for _ in range(n_var)], dtype=int)
            avg_tokens, avg_perf = problem.score_solution(bits)
            all_solutions[i] = bits
            all_objectives[i] = [avg_tokens, avg_perf]

        # Return only the Pareto-non-dominated subset.
        # For the Pareto filter we want to *minimise* tokens and *maximise*
        # performance → convert to minimise both by negating performance.
        obj_for_pareto = all_objectives.copy()
        obj_for_pareto[:, 1] = -obj_for_pareto[:, 1]
        pareto_solutions, pareto_obj_negated = _pareto_front(all_solutions, obj_for_pareto)
        pareto_objectives = pareto_obj_negated.copy()
        pareto_objectives[:, 1] = -pareto_objectives[:, 1]  # restore positive performance

        names = [_names_from_bits(s, problem) for s in pareto_solutions]
        return BaselineResult(
            name=self.name,
            solutions=pareto_solutions,
            objectives=pareto_objectives,
            transformation_names=names,
        )