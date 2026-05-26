"""Baseline strategies for comparison with the evolutionary search.

Seven baselines are provided:

* :class:`NoCompressionBaseline` — apply no transformations (identity).
* :class:`AllTransformationsBaseline` — apply all transformations in sequence.
* :class:`RandomSearchBaseline` — uniformly random search over the binary
  solution space, using the same evaluation budget as the evolutionary run
  (``pop_size × n_gen`` evaluations).
* :class:`TruncatedPromptBaseline` — truncate each prompt to a fixed fraction
  of its original character length (deterministic).
* :class:`RandomDeletionBaseline` — randomly delete paragraphs with a given
  probability (seeded for reproducibility).
* :class:`SummarizationBaseline` — apply all semantic operators (O23–O28);
  when the problem was built with ``use_llm_for_semantic=True`` these operators
  use the LLM backend, otherwise they fall back to heuristics.
* :class:`MultiArmedBanditBaseline` — Thompson Sampling bandit over the
  binary operator space; uses task performance as the reward signal and
  returns the Pareto-non-dominated subset of all visited solutions.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

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
    #: Hyperparameters used to construct this baseline.
    config: dict = field(default_factory=dict)

    @property
    def avg_tokens(self) -> np.ndarray:
        return self.objectives[:, 0]

    @property
    def avg_performance(self) -> np.ndarray:
        return self.objectives[:, 1]


def _names_from_bits(bits: np.ndarray, problem: PromptCompressionProblem) -> list[str]:
    return problem.names_from_bits(bits)


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

    @property
    def config(self) -> dict:
        return {}

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
            config=self.config,
        )


# ---------------------------------------------------------------------------
# All-transformations baseline
# ---------------------------------------------------------------------------

class AllTransformationsBaseline:
    """Apply every transformation in sequence."""

    name = "all_transformations"

    @property
    def config(self) -> dict:
        return {}

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
            config=self.config,
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

    @property
    def config(self) -> dict:
        return {"n_evaluations": self._n_eval, "seed": self._seed}

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
            config=self.config,
        )


# ---------------------------------------------------------------------------
# Truncated-prompt baseline
# ---------------------------------------------------------------------------

class TruncatedPromptBaseline:
    """Truncate each prompt to *target_ratio* of its original character length.

    Deterministic — no seed needed.

    Parameters
    ----------
    target_ratio:
        Fraction of characters to keep (default 0.5 = 50 %).
    """

    name = "truncated_prompt"

    def __init__(self, target_ratio: float = 0.5) -> None:
        self._ratio = target_ratio

    @property
    def config(self) -> dict:
        return {"target_ratio": self._ratio}

    def run(self, problem: PromptCompressionProblem) -> BaselineResult:
        ratio = self._ratio

        def _truncate(text: str) -> str:
            return text[:max(1, int(len(text) * ratio))]

        avg_tokens, avg_perf = problem.score_transform_fn(_truncate)
        solutions = np.zeros((1, problem.n_var), dtype=int)
        objectives = np.array([[avg_tokens, avg_perf]])
        return BaselineResult(
            name=self.name,
            solutions=solutions,
            objectives=objectives,
            transformation_names=[["TruncatedPrompt"]],
            config=self.config,
        )


# ---------------------------------------------------------------------------
# Random-deletion baseline
# ---------------------------------------------------------------------------

class RandomDeletionBaseline:
    """Randomly delete paragraphs (double-newline blocks) from each prompt.

    Each paragraph is dropped independently with probability *deletion_prob*.
    The same ``seed`` guarantees identical compression across runs.

    Parameters
    ----------
    deletion_prob:
        Probability of dropping each paragraph.
    seed:
        RNG seed for reproducibility.
    """

    name = "random_deletion"

    def __init__(self, deletion_prob: float = 0.3, seed: int = 42) -> None:
        self._prob = deletion_prob
        self._seed = seed

    @property
    def config(self) -> dict:
        return {"deletion_prob": self._prob, "seed": self._seed}

    def run(self, problem: PromptCompressionProblem) -> BaselineResult:
        prob = self._prob
        # A single RNG instance is shared across all samples in the dataset so
        # that the sequence of random decisions is fully determined by *seed*.
        rng = random.Random(self._seed)

        def _delete(text: str) -> str:
            paragraphs = re.split(r"\n{2,}", text)
            kept = [p for p in paragraphs if rng.random() >= prob]
            return "\n\n".join(kept) if kept else text

        avg_tokens, avg_perf = problem.score_transform_fn(_delete)
        solutions = np.zeros((1, problem.n_var), dtype=int)
        objectives = np.array([[avg_tokens, avg_perf]])
        return BaselineResult(
            name=self.name,
            solutions=solutions,
            objectives=objectives,
            transformation_names=[["RandomDeletion"]],
            config=self.config,
        )


# ---------------------------------------------------------------------------
# Multi-Armed Bandit baseline (Thompson Sampling)
# ---------------------------------------------------------------------------

class MultiArmedBanditBaseline:
    """Thompson Sampling bandit over the binary operator space.

    Each of the ``n_var`` operators is treated as an independent Bernoulli arm
    with a Beta(α, β) prior.  At every round the arm beliefs are sampled, a
    binary vector is constructed by thresholding at 0.5, and the observed task
    performance is used to update the beliefs of all *active* arms (those
    where the corresponding bit is 1).

    The total evaluation budget matches the evolutionary run:
    ``n_evaluations = pop_size × n_gen``.

    After the budget is exhausted the Pareto-non-dominated subset of all
    visited (solution, objective) pairs is returned — the same post-processing
    used by :class:`RandomSearchBaseline`.

    Parameters
    ----------
    n_evaluations:
        Total number of solutions to evaluate.
    seed:
        NumPy random seed for reproducibility.
    alpha_init:
        Initial α parameter of the Beta prior (default 1 → uniform).
    beta_init:
        Initial β parameter of the Beta prior (default 1 → uniform).
    """

    name = "multi_armed_bandit"

    def __init__(
        self,
        n_evaluations: int = 2500,
        seed: int = 42,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
    ) -> None:
        self._n_eval = n_evaluations
        self._seed = seed
        self._alpha_init = alpha_init
        self._beta_init = beta_init

    @property
    def config(self) -> dict:
        return {
            "n_evaluations": self._n_eval,
            "seed": self._seed,
            "alpha_init": self._alpha_init,
            "beta_init": self._beta_init,
        }

    def run(self, problem: PromptCompressionProblem) -> BaselineResult:
        rng = np.random.default_rng(self._seed)
        n_var = problem.n_var

        alpha = np.full(n_var, self._alpha_init, dtype=float)
        beta_ = np.full(n_var, self._beta_init, dtype=float)

        all_solutions = np.empty((self._n_eval, n_var), dtype=int)
        all_objectives = np.empty((self._n_eval, 2), dtype=float)

        for i in range(self._n_eval):
            # Draw a probability estimate from each arm's current belief.
            theta = rng.beta(alpha, beta_)
            bits = (theta > 0.5).astype(int)

            avg_tokens, avg_perf = problem.score_solution(bits)
            all_solutions[i] = bits
            all_objectives[i] = [avg_tokens, avg_perf]

            # Update beliefs for active arms only: performance is in [0, 1]
            # by the Evaluator contract, so it directly serves as a reward.
            active = bits.astype(bool)
            alpha[active] += avg_perf
            beta_[active] += (1.0 - avg_perf)

        # Extract Pareto-non-dominated front (minimise tokens, maximise perf).
        obj_for_pareto = all_objectives.copy()
        obj_for_pareto[:, 1] = -obj_for_pareto[:, 1]
        pareto_solutions, pareto_obj_neg = _pareto_front(all_solutions, obj_for_pareto)
        pareto_objectives = pareto_obj_neg.copy()
        pareto_objectives[:, 1] = -pareto_objectives[:, 1]

        names = [_names_from_bits(s, problem) for s in pareto_solutions]
        return BaselineResult(
            name=self.name,
            solutions=pareto_solutions,
            objectives=pareto_objectives,
            transformation_names=names,
            config=self.config,
        )


# ---------------------------------------------------------------------------
# Summarization baseline (LLM-backed semantic operators)
# ---------------------------------------------------------------------------

class SummarizationBaseline:
    """Apply all semantic operators (O23–O28) as an LLM-based summarization baseline.

    When the :class:`~prompt_piper.optimization.problem.PromptCompressionProblem`
    was built with ``use_llm_for_semantic=True`` the semantic operators carry an
    LLM backend and perform true neural summarization; otherwise they fall back
    to their heuristic mode.  Either way, no extra configuration is needed here.
    """

    name = "summarization"

    @property
    def config(self) -> dict:
        return {}

    def run(self, problem: PromptCompressionProblem) -> BaselineResult:
        bits = np.array(
            [1 if t.category == "semantic" else 0 for t in problem._transforms],
            dtype=int,
        )
        avg_tokens, avg_perf = problem.score_solution(bits)
        solutions = bits[np.newaxis, :]
        objectives = np.array([[avg_tokens, avg_perf]])
        names = [problem.names_from_bits(bits)]
        return BaselineResult(
            name=self.name,
            solutions=solutions,
            objectives=objectives,
            transformation_names=names,
            config=self.config,
        )
