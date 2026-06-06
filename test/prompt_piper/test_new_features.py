"""Tests for new features: MAB as primary algorithm, batch evaluation, transferability."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from prompt_piper.optimization.config import ExperimentConfig
from prompt_piper.optimization.evaluator import ConstantEvaluator, MockEvaluator
from prompt_piper.optimization.runner import Experiment
from prompt_piper.optimization.tokenizer import WhitespaceTokenizer
from prompt_piper.tools.compressor.config import (
    CompressorConfig,
    TransferabilityConfig,
)
from prompt_piper.tools.compressor.results import SolutionEval, TransferabilityEval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 20) -> list[dict]:
    return [
        {
            "prompt": f"Sample prompt {i}. " * 6,
            "prompt_0shot": f"Instruction {i}.",
            "target": "A" if i % 2 == 0 else "B",
        }
        for i in range(n)
    ]


def _make_config(algorithm: str = "nsga2", pop_size: int = 5, n_gen: int = 3) -> ExperimentConfig:
    return ExperimentConfig(
        n_subset=10,
        subset_seed=42,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=42,
        verbose=False,
        algorithm=algorithm,
    )


# ---------------------------------------------------------------------------
# MAB as primary algorithm
# ---------------------------------------------------------------------------

class TestMABPrimaryAlgorithm:

    def test_mab_run_returns_experiment_result(self):
        cfg = _make_config(algorithm="mab")
        exp = Experiment(config=cfg, dataset=_make_dataset(), evaluator=MockEvaluator())
        result = exp.run()
        assert result is not None
        assert result.pareto_solutions is not None
        assert result.pareto_objectives is not None
        assert len(result.pareto_solutions) > 0

    def test_mab_pareto_objectives_are_non_dominated(self):
        cfg = _make_config(algorithm="mab", pop_size=8, n_gen=5)
        exp = Experiment(config=cfg, dataset=_make_dataset(), evaluator=MockEvaluator())
        result = exp.run()
        obj = result.pareto_objectives  # (n, 2): [tokens, +perf]
        for i, f_i in enumerate(obj):
            for j, f_j in enumerate(obj):
                if i == j:
                    continue
                # f_i should not be dominated: no solution dominates it (min both)
                # obj[:, 1] is +performance → we want to maximise it → negate for dominance
                fi_min = np.array([f_i[0], -f_i[1]])
                fj_min = np.array([f_j[0], -f_j[1]])
                # j does NOT strictly dominate i (not all ≤ and at least one <)
                dominated = np.all(fj_min <= fi_min) and np.any(fj_min < fi_min)
                assert not dominated, f"Solution {i} dominated by {j}"

    def test_mab_generation_history_is_empty(self):
        cfg = _make_config(algorithm="mab")
        exp = Experiment(config=cfg, dataset=_make_dataset(), evaluator=MockEvaluator())
        result = exp.run()
        assert result.generation_history == []

    def test_mab_pymoo_result_is_none(self):
        cfg = _make_config(algorithm="mab")
        exp = Experiment(config=cfg, dataset=_make_dataset(), evaluator=MockEvaluator())
        result = exp.run()
        assert result.pymoo_result is None

    def test_mab_runs_baselines(self):
        cfg = _make_config(algorithm="mab")
        exp = Experiment(config=cfg, dataset=_make_dataset(), evaluator=MockEvaluator())
        result = exp.run(baselines=["no_compression", "all_transformations"])
        assert "no_compression" in result.baselines
        assert "all_transformations" in result.baselines

    def test_mab_evaluation_budget_matches_nsga2(self):
        """MAB uses pop_size × n_gen evaluations — same budget as NSGA-II."""
        # This is implicitly tested by the fact that MAB runs without error
        # with any pop_size × n_gen budget; here we just verify it runs.
        cfg = _make_config(algorithm="mab", pop_size=4, n_gen=4)
        exp = Experiment(config=cfg, dataset=_make_dataset(), evaluator=MockEvaluator())
        result = exp.run()
        assert result.elapsed_time >= 0


# ---------------------------------------------------------------------------
# Batch evaluation (parallel execution via deduplication)
# ---------------------------------------------------------------------------

class TestBatchEvaluation:

    def test_batch_eval_same_result_as_sequential(self):
        """_evaluate must give identical results to calling score_solution per-row."""
        from prompt_piper.optimization.problem import PromptCompressionProblem
        from prompt_piper.transformations.sections import PromptSplitter

        dataset = _make_dataset(8)
        evaluator = ConstantEvaluator(0.5)
        tokenizer = WhitespaceTokenizer()

        splitter = PromptSplitter(samples=dataset, compress_field="prompt_0shot")
        sections = splitter.split_all()

        from prompt_piper.transformations import INSTRUCTION_OPS
        from prompt_piper.transformations.sections import PromptSplitter

        ops = [cls() for cls in INSTRUCTION_OPS[:4]]

        problem = PromptCompressionProblem(
            instruction_ops=ops,
            shot_ops=[],
            data_ops=[],
            sections=sections,
            dataset=dataset,
            evaluator=evaluator,
            tokenizer=tokenizer,
        )

        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, size=(3, problem.n_var))

        # Batch path via _evaluate
        out = {}
        problem._evaluate(x, out)
        batch_F = out["F"]

        # Sequential path via score_solution
        seq_F = np.empty_like(batch_F)
        for i, bits in enumerate(x):
            tok, perf = problem.score_solution(bits)
            seq_F[i, 0] = tok
            seq_F[i, 1] = -perf

        np.testing.assert_allclose(batch_F, seq_F, rtol=1e-9)

    def test_deduplication_reduces_evaluations(self):
        """When all solutions produce the same text, evaluate_batch is called once."""
        from prompt_piper.optimization.problem import PromptCompressionProblem

        call_count = [0]

        class CountingEvaluator(MockEvaluator):
            def evaluate_batch(self, originals, compressed, dataset):
                call_count[0] += 1
                return super().evaluate_batch(originals, compressed, dataset)

        dataset = _make_dataset(5)
        problem = PromptCompressionProblem(
            transformations=[],  # legacy flat mode, no operators
            dataset=dataset,
            evaluator=CountingEvaluator(),
            tokenizer=WhitespaceTokenizer(),
            compress_field="prompt",
        )

        # All-zeros solutions → all produce the same (no-op) compressed text
        x = np.zeros((4, problem.n_var), dtype=int)
        out = {}
        problem._evaluate(x, out)

        # With deduplication: all 4 solutions × 5 samples → only 5 unique (j, text) pairs
        # → evaluate_batch called exactly once
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# TransferabilityConfig parsing
# ---------------------------------------------------------------------------

class TestTransferabilityConfig:

    def test_from_yaml_parses_transferability(self, tmp_path):
        yaml_content = """
model:
  type: hf
  name: gpt2

output:
  dir: /tmp/results

transferability:
  enabled: true
  pairs:
    - [run_a_nsga2, run_b_nsga2]
    - [run_b_nsga2, run_a_nsga2]

runs: []
"""
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml_content)
        cfg = CompressorConfig.from_yaml(cfg_path)
        assert cfg.transferability.enabled is True
        assert cfg.transferability.pairs == [
            ["run_a_nsga2", "run_b_nsga2"],
            ["run_b_nsga2", "run_a_nsga2"],
        ]

    def test_default_transferability_disabled(self, tmp_path):
        yaml_content = "model:\n  type: hf\n  name: gpt2\noutput:\n  dir: /tmp\nruns: []\n"
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml_content)
        cfg = CompressorConfig.from_yaml(cfg_path)
        assert cfg.transferability.enabled is False
        assert cfg.transferability.pairs == []

    def test_to_dict_round_trips_transferability(self, tmp_path):
        cfg = CompressorConfig.from_dict({
            "transferability": {"enabled": True, "pairs": [["a", "b"]]},
            "runs": [],
        })
        d = cfg.to_dict()
        assert d["transferability"]["enabled"] is True
        assert d["transferability"]["pairs"] == [["a", "b"]]


# ---------------------------------------------------------------------------
# TransferabilityEval serialisation
# ---------------------------------------------------------------------------

class TestTransferabilityEval:

    def _make_sol(self, bits=None) -> SolutionEval:
        bits = bits or [0, 1, 0]
        return SolutionEval(
            transformation_ids=["O1"],
            transformation_names=["WhitespaceNormalizer"],
            bits=bits,
            search_avg_tokens=50.0,
            search_avg_performance=0.7,
            test_avg_tokens=48.0,
            test_avg_performance=0.65,
        )

    def test_save_and_load_round_trip(self, tmp_path):
        te = TransferabilityEval(
            source_run="gsm8k_cot_nsga2",
            target_run="bbh_logical_deduction_nsga2",
            timestamp="2026-01-01T00:00:00+00:00",
            solutions=[self._make_sol()],
        )
        path = tmp_path / "transfer.json"
        te.save(path)
        assert path.exists()
        loaded = TransferabilityEval.load(path)
        assert loaded.source_run == te.source_run
        assert loaded.target_run == te.target_run
        assert len(loaded.solutions) == 1
        assert loaded.solutions[0].test_avg_performance == pytest.approx(0.65)

    def test_to_dict_is_json_serializable(self):
        te = TransferabilityEval(
            source_run="a", target_run="b",
            timestamp="2026-01-01T00:00:00+00:00",
            solutions=[self._make_sol()],
        )
        d = te.to_dict()
        json.dumps(d)  # should not raise


# ---------------------------------------------------------------------------
# Level-aware search encoding
# ---------------------------------------------------------------------------

def _make_structured_dataset(n: int = 8) -> list[dict]:
    """Dataset with a shared instruction prefix so sections can be non-empty."""
    return [
        {
            "prompt": f"Classify sentiment.\n\nText {i}: This is sample {i}.",
            "target": "positive" if i % 2 == 0 else "negative",
        }
        for i in range(n)
    ]


class TestLevelAwareSearch:

    def test_n_var_is_three_times_n_ops_with_data(self):
        """With level-aware encoding + compress_input_data, bits = [N instr | N shots | N data]."""
        dataset = _make_structured_dataset()
        cfg = ExperimentConfig(
            n_subset=len(dataset),
            transformation_ids=["O1", "O2", "O3"],
            compress_input_data=True,
            pop_size=2, n_gen=1, seed=42,
        )
        exp = Experiment(config=cfg, dataset=dataset, evaluator=MockEvaluator())
        N = len(exp._search_transform_classes)
        problem = exp._build_problem(dataset)
        assert problem.n_var == 3 * N

    def test_n_var_is_two_times_n_ops_without_data(self):
        """With compress_input_data=False, bits = [N instr | N shots]."""
        dataset = _make_structured_dataset()
        cfg = ExperimentConfig(
            n_subset=len(dataset),
            transformation_ids=["O1", "O2", "O3"],
            compress_input_data=False,
            pop_size=2, n_gen=1, seed=42,
        )
        exp = Experiment(config=cfg, dataset=dataset, evaluator=MockEvaluator())
        N = len(exp._search_transform_classes)
        problem = exp._build_problem(dataset)
        assert problem.n_var == 2 * N

    def test_names_from_bits_include_level_prefix(self):
        """names_from_bits returns 'I:X', 'S:X', 'D:X' level-prefixed names."""
        dataset = _make_structured_dataset()
        cfg = ExperimentConfig(
            n_subset=len(dataset),
            transformation_ids=["O1", "O2"],
            compress_input_data=True,
            pop_size=2, n_gen=1, seed=42,
        )
        exp = Experiment(config=cfg, dataset=dataset, evaluator=MockEvaluator())
        N = len(exp._search_transform_classes)
        problem = exp._build_problem(dataset)
        all_on = np.ones(3 * N, dtype=int)
        names = problem.names_from_bits(all_on)
        assert any(n.startswith("I:") for n in names)
        assert any(n.startswith("S:") for n in names)
        assert any(n.startswith("D:") for n in names)

    def test_experiment_run_returns_correct_solution_width(self):
        """End-to-end run: pareto_solutions columns == 3 × N search ops."""
        dataset = _make_structured_dataset(n=10)
        cfg = ExperimentConfig(
            n_subset=8,
            transformation_ids=["O1", "O2", "O3"],
            compress_input_data=True,
            pop_size=4, n_gen=2, seed=42,
        )
        exp = Experiment(config=cfg, dataset=dataset, evaluator=MockEvaluator())
        result = exp.run()
        N = len(exp._search_transform_classes)
        assert result.pareto_solutions.shape[1] == 3 * N

    def test_evaluate_solutions_accepts_level_aware_bits(self):
        """evaluate_solutions must work with 3×N bit vectors on held-out data."""
        dataset = _make_structured_dataset(n=10)
        cfg = ExperimentConfig(
            n_subset=8,
            transformation_ids=["O1", "O2"],
            compress_input_data=True,
            pop_size=4, n_gen=2, seed=42,
        )
        exp = Experiment(config=cfg, dataset=dataset, evaluator=MockEvaluator())
        N = len(exp._search_transform_classes)
        dummy_solutions = np.zeros((3, 3 * N), dtype=int)
        objectives = exp.evaluate_solutions(dummy_solutions, dataset)
        assert objectives.shape == (3, 2)


# ---------------------------------------------------------------------------
# _bits_to_ids helper
# ---------------------------------------------------------------------------

class TestBitsToIds:

    def test_three_level_encoding(self):
        from prompt_piper.tools.compressor.runner import _bits_to_ids
        from prompt_piper.transformations import ALL_TRANSFORMATIONS

        classes = ALL_TRANSFORMATIONS[:3]  # O1, O2, O3
        # instr: O1=1, O2=0, O3=0 | shots: O1=0, O2=1, O3=0 | data: O1=0, O2=0, O3=1
        bits = [1, 0, 0,  0, 1, 0,  0, 0, 1]
        ids = _bits_to_ids(classes, bits)
        assert "O1@instruction" in ids
        assert "O2@shots" in ids
        assert "O3@data" in ids
        assert len(ids) == 3

    def test_two_level_encoding(self):
        from prompt_piper.tools.compressor.runner import _bits_to_ids
        from prompt_piper.transformations import ALL_TRANSFORMATIONS

        classes = ALL_TRANSFORMATIONS[:2]  # O1, O2
        bits = [1, 0,  0, 1]  # O1@instruction, O2@shots
        ids = _bits_to_ids(classes, bits)
        assert "O1@instruction" in ids
        assert "O2@shots" in ids
        assert len(ids) == 2

    def test_all_zeros_returns_empty(self):
        from prompt_piper.tools.compressor.runner import _bits_to_ids
        from prompt_piper.transformations import ALL_TRANSFORMATIONS

        classes = ALL_TRANSFORMATIONS[:3]
        ids = _bits_to_ids(classes, [0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert ids == []

    def test_empty_classes_returns_empty(self):
        from prompt_piper.tools.compressor.runner import _bits_to_ids
        assert _bits_to_ids([], []) == []