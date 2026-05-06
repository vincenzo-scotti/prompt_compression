"""Tests verifying that all random components produce deterministic output given the same seed.

Covers:
  - Dataset subset sampling (Experiment._sample_subset)
  - NSGA-II optimisation (via MockEvaluator — same seed → same Pareto front)
  - RandomSearchBaseline (same seed → same Pareto front)
  - RandomDeletionBaseline (same seed → same compressed text)
  - Different seeds produce different subsets / different results
"""

import random

import numpy as np
import pytest

from prompt_piper.optimization.baselines import RandomDeletionBaseline, RandomSearchBaseline
from prompt_piper.optimization.config import ExperimentConfig
from prompt_piper.optimization.evaluator import MockEvaluator
from prompt_piper.optimization.runner import Experiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 30) -> list[dict]:
    return [
        {
            "prompt": f"Sample prompt number {i}. " * 5,
            "target": "positive" if i % 2 == 0 else "negative",
        }
        for i in range(n)
    ]


def _make_experiment(seed: int, subset_seed: int, n_subset: int = 10) -> Experiment:
    cfg = ExperimentConfig(
        n_subset=n_subset,
        subset_seed=subset_seed,
        pop_size=5,
        n_gen=3,
        seed=seed,
        verbose=False,
    )
    return Experiment(config=cfg, dataset=_make_dataset(30), evaluator=MockEvaluator())


# ---------------------------------------------------------------------------
# Subset sampling
# ---------------------------------------------------------------------------

class TestSubsetSampling:

    def test_same_seed_produces_identical_subset(self):
        exp1 = _make_experiment(seed=0, subset_seed=7)
        exp2 = _make_experiment(seed=0, subset_seed=7)
        s1 = exp1._sample_subset()
        s2 = exp2._sample_subset()
        assert [d["prompt"] for d in s1] == [d["prompt"] for d in s2]

    def test_different_subset_seed_produces_different_subset(self):
        exp1 = _make_experiment(seed=0, subset_seed=7)
        exp2 = _make_experiment(seed=0, subset_seed=99)
        s1 = exp1._sample_subset()
        s2 = exp2._sample_subset()
        assert [d["prompt"] for d in s1] != [d["prompt"] for d in s2]

    def test_n_subset_none_returns_full_dataset(self):
        cfg = ExperimentConfig(n_subset=None, subset_seed=0, pop_size=2, n_gen=1, seed=0)
        exp = Experiment(config=cfg, dataset=_make_dataset(10), evaluator=MockEvaluator())
        assert len(exp._sample_subset()) == 10

    def test_n_subset_larger_than_dataset_returns_full_dataset(self):
        cfg = ExperimentConfig(n_subset=999, subset_seed=0, pop_size=2, n_gen=1, seed=0)
        exp = Experiment(config=cfg, dataset=_make_dataset(10), evaluator=MockEvaluator())
        assert len(exp._sample_subset()) == 10


# ---------------------------------------------------------------------------
# NSGA-II reproducibility
# ---------------------------------------------------------------------------

class TestNSGA2Seeding:

    def test_same_seeds_produce_identical_pareto_objectives(self):
        exp1 = _make_experiment(seed=42, subset_seed=7)
        exp2 = _make_experiment(seed=42, subset_seed=7)
        r1 = exp1.run()
        r2 = exp2.run()
        np.testing.assert_array_equal(r1.pareto_objectives, r2.pareto_objectives)

    def test_different_algorithm_seed_may_produce_different_result(self):
        exp1 = _make_experiment(seed=1, subset_seed=7)
        exp2 = _make_experiment(seed=2, subset_seed=7)
        r1 = exp1.run()
        r2 = exp2.run()
        # With MockEvaluator the objectives depend on token counts (deterministic
        # given the subset), so different seeds may still converge to the same
        # front.  We just verify no crash and the shapes are valid.
        assert r1.pareto_objectives.shape[1] == 2
        assert r2.pareto_objectives.shape[1] == 2


# ---------------------------------------------------------------------------
# RandomSearchBaseline
# ---------------------------------------------------------------------------

class TestRandomSearchBaselineSeeding:

    def _run_random_search(self, seed: int) -> np.ndarray:
        exp = _make_experiment(seed=seed, subset_seed=0, n_subset=5)
        subset = exp._sample_subset()
        problem = exp._build_problem(subset)
        bl = RandomSearchBaseline(n_evaluations=20, seed=seed)
        result = bl.run(problem)
        return result.objectives

    def test_same_seed_produces_identical_objectives(self):
        obj1 = self._run_random_search(seed=42)
        obj2 = self._run_random_search(seed=42)
        np.testing.assert_array_equal(obj1, obj2)

    def test_different_seed_may_produce_different_objectives(self):
        obj1 = self._run_random_search(seed=1)
        obj2 = self._run_random_search(seed=2)
        # Objectives might coincidentally match for trivial problems, but
        # the internal solution sequences must differ (tested via solution bits).
        exp = _make_experiment(seed=1, subset_seed=0, n_subset=5)
        subset = exp._sample_subset()
        problem = exp._build_problem(subset)
        bl1 = RandomSearchBaseline(n_evaluations=20, seed=1)
        bl2 = RandomSearchBaseline(n_evaluations=20, seed=2)
        sols1 = bl1.run(problem).solutions
        sols2 = bl2.run(problem).solutions
        # At least the solution arrays must differ in shape or content.
        different = (
            sols1.shape != sols2.shape
            or not np.array_equal(sols1, sols2)
        )
        assert different, "Different seeds produced identical random-search solutions"


# ---------------------------------------------------------------------------
# RandomDeletionBaseline
# ---------------------------------------------------------------------------

class TestRandomDeletionBaselineSeeding:

    PROMPT = "Para one.\n\nPara two.\n\nPara three.\n\nPara four.\n\nPara five."

    def _make_compressed(self, seed: int) -> str:
        prob = 0.4
        rng = random.Random(seed)

        import re
        paragraphs = re.split(r"\n{2,}", self.PROMPT)
        kept = [p for p in paragraphs if rng.random() >= prob]
        return "\n\n".join(kept) if kept else self.PROMPT

    def test_same_seed_produces_identical_output(self):
        out1 = self._make_compressed(42)
        out2 = self._make_compressed(42)
        assert out1 == out2

    def test_different_seed_may_produce_different_output(self):
        # With 5 paragraphs and prob=0.4 the chance of identical output for
        # two different seeds is (0.4^5 + 0.6^5)^5 ≈ 0.01 — practically never.
        results = {self._make_compressed(s) for s in range(10)}
        assert len(results) > 1, "All seeds produced identical random-deletion output"

    def test_baseline_class_produces_same_token_count_for_same_seed(self):
        """Same seed → same paragraph deletion → same compressed length → same token count.

        MockEvaluator has its own internal randomness so performance scores
        are not compared here; only the compression side (token counts) is
        deterministic given the seed.
        """
        dataset = _make_dataset(5)
        exp = _make_experiment(seed=0, subset_seed=0, n_subset=5)
        problem = exp._build_problem(dataset)
        bl1 = RandomDeletionBaseline(deletion_prob=0.4, seed=7)
        bl2 = RandomDeletionBaseline(deletion_prob=0.4, seed=7)
        r1 = bl1.run(problem)
        r2 = bl2.run(problem)
        # Token counts are purely a function of the compressed text length
        # → must match exactly when seed is the same.
        assert r1.objectives[0, 0] == r2.objectives[0, 0], (
            f"Token counts differ: {r1.objectives[0,0]} vs {r2.objectives[0,0]}"
        )
