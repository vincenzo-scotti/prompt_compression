"""Tests for the LLMLingua2Baseline."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from prompt_piper.optimization.baselines import LLMLingua2Baseline
from prompt_piper.optimization.config import ExperimentConfig
from prompt_piper.optimization.evaluator import MockEvaluator
from prompt_piper.optimization.runner import Experiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 10) -> list[dict]:
    return [
        {"prompt": f"This is prompt number {i}. " * 4, "target": str(i % 2)}
        for i in range(n)
    ]


def _make_problem(dataset=None):
    dataset = dataset or _make_dataset()
    cfg = ExperimentConfig(n_subset=None, pop_size=2, n_gen=1, seed=0)
    exp = Experiment(config=cfg, dataset=dataset, evaluator=MockEvaluator())
    return exp._build_problem(dataset)


def _mock_llmlingua(compressed_text: str = "short compressed text"):
    """Return a (mock_module, mock_compressor) pair for patching sys.modules."""
    mock_compressor = MagicMock()
    mock_compressor.compress_prompt_llmlingua2.return_value = {
        "compressed_prompt": compressed_text,
    }
    mock_module = MagicMock()
    mock_module.PromptCompressor.return_value = mock_compressor
    return mock_module, mock_compressor


# ---------------------------------------------------------------------------
# Config and metadata
# ---------------------------------------------------------------------------

class TestLLMLingua2BaselineConfig:

    def test_name(self):
        assert LLMLingua2Baseline.name == "llmlingua2"

    def test_default_config(self):
        bl = LLMLingua2Baseline()
        cfg = bl.config
        assert cfg["model_name"] == "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        assert cfg["target_rate"] == 0.5
        assert cfg["force_tokens"] == ["\n", ".", "!", "?", ","]

    def test_custom_config(self):
        bl = LLMLingua2Baseline(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            target_rate=0.3,
            force_tokens=["\n", "."],
        )
        cfg = bl.config
        assert cfg["model_name"] == "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
        assert cfg["target_rate"] == 0.3
        assert cfg["force_tokens"] == ["\n", "."]

    def test_compressor_not_loaded_at_init(self):
        bl = LLMLingua2Baseline()
        assert bl._compressor is None


# ---------------------------------------------------------------------------
# Import and lazy initialisation
# ---------------------------------------------------------------------------

class TestLLMLingua2BaselineImport:

    def test_raises_import_error_when_llmlingua_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "llmlingua", None)
        bl = LLMLingua2Baseline()
        bl._compressor = None  # ensure not pre-loaded
        with pytest.raises(ImportError, match="pip install llmlingua"):
            bl._get_compressor()

    def test_compressor_loaded_lazily_on_first_call(self):
        mock_module, mock_compressor = _mock_llmlingua()
        with patch.dict(sys.modules, {"llmlingua": mock_module}):
            bl = LLMLingua2Baseline()
            assert bl._compressor is None
            bl._get_compressor()
            assert bl._compressor is mock_compressor

    def test_compressor_cached_after_first_call(self):
        mock_module, mock_compressor = _mock_llmlingua()
        with patch.dict(sys.modules, {"llmlingua": mock_module}):
            bl = LLMLingua2Baseline()
            c1 = bl._get_compressor()
            c2 = bl._get_compressor()
            assert c1 is c2
            mock_module.PromptCompressor.assert_called_once()

    def test_compressor_initialised_with_use_llmlingua2_true(self):
        mock_module, _ = _mock_llmlingua()
        with patch.dict(sys.modules, {"llmlingua": mock_module}):
            bl = LLMLingua2Baseline(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
            )
            bl._get_compressor()
            mock_module.PromptCompressor.assert_called_once_with(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
            )


# ---------------------------------------------------------------------------
# run() — return value structure
# ---------------------------------------------------------------------------

class TestLLMLingua2BaselineRun:

    def test_run_returns_baseline_result_with_correct_name(self):
        mock_module, _ = _mock_llmlingua()
        with patch.dict(sys.modules, {"llmlingua": mock_module}):
            bl = LLMLingua2Baseline()
            result = bl.run(_make_problem())
        assert result.name == "llmlingua2"

    def test_run_objectives_shape(self):
        mock_module, _ = _mock_llmlingua()
        with patch.dict(sys.modules, {"llmlingua": mock_module}):
            result = LLMLingua2Baseline().run(_make_problem())
        assert result.objectives.shape == (1, 2)

    def test_run_solutions_shape(self):
        mock_module, _ = _mock_llmlingua()
        with patch.dict(sys.modules, {"llmlingua": mock_module}):
            problem = _make_problem()
            result = LLMLingua2Baseline().run(problem)
        assert result.solutions.shape == (1, problem.n_var)
        assert np.all(result.solutions == 0)

    def test_run_transformation_names(self):
        mock_module, _ = _mock_llmlingua()
        with patch.dict(sys.modules, {"llmlingua": mock_module}):
            result = LLMLingua2Baseline().run(_make_problem())
        assert result.transformation_names == [["LLMLingua2"]]

    def test_run_config_preserved(self):
        mock_module, _ = _mock_llmlingua()
        with patch.dict(sys.modules, {"llmlingua": mock_module}):
            bl = LLMLingua2Baseline(target_rate=0.4, force_tokens=["\n"])
            result = bl.run(_make_problem())
        assert result.config["target_rate"] == 0.4
        assert result.config["force_tokens"] == ["\n"]

    def test_run_avg_tokens_positive(self):
        mock_module, _ = _mock_llmlingua(compressed_text="short text")
        with patch.dict(sys.modules, {"llmlingua": mock_module}):
            result = LLMLingua2Baseline().run(_make_problem())
        assert result.avg_tokens[0] > 0

    def test_run_calls_compress_prompt_llmlingua2_with_correct_args(self):
        mock_module, mock_compressor = _mock_llmlingua()
        with patch.dict(sys.modules, {"llmlingua": mock_module}):
            bl = LLMLingua2Baseline(target_rate=0.3, force_tokens=["\n", "."])
            bl.run(_make_problem(_make_dataset(n=2)))

        for call in mock_compressor.compress_prompt_llmlingua2.call_args_list:
            kwargs = call.kwargs if call.kwargs else {}
            args = call.args
            # rate and force_tokens must be forwarded
            rate = kwargs.get("rate", args[1] if len(args) > 1 else None)
            force = kwargs.get("force_tokens", args[2] if len(args) > 2 else None)
            assert rate == 0.3
            assert force == ["\n", "."]

    def test_compressed_text_shorter_reduces_tokens(self):
        mock_module_long, _ = _mock_llmlingua(
            compressed_text="word " * 50  # 50 tokens
        )
        mock_module_short, _ = _mock_llmlingua(
            compressed_text="word " * 10  # 10 tokens
        )
        problem = _make_problem()
        with patch.dict(sys.modules, {"llmlingua": mock_module_long}):
            result_long = LLMLingua2Baseline().run(problem)
        with patch.dict(sys.modules, {"llmlingua": mock_module_short}):
            result_short = LLMLingua2Baseline().run(problem)
        assert result_short.avg_tokens[0] < result_long.avg_tokens[0]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestLLMLingua2BaselineRegistry:

    def test_registered_under_llmlingua2_key(self):
        from prompt_piper.optimization.runner import _BASELINE_REGISTRY
        assert "llmlingua2" in _BASELINE_REGISTRY
        assert _BASELINE_REGISTRY["llmlingua2"] is LLMLingua2Baseline

    def test_run_baseline_instantiates_with_defaults(self):
        mock_module, _ = _mock_llmlingua()
        with patch.dict(sys.modules, {"llmlingua": mock_module}):
            cfg = ExperimentConfig(n_subset=5, pop_size=2, n_gen=1, seed=0)
            exp = Experiment(
                config=cfg,
                dataset=_make_dataset(),
                evaluator=MockEvaluator(),
            )
            result = exp.run_baseline("llmlingua2")
        assert result.name == "llmlingua2"

    def test_unknown_baseline_raises_value_error(self):
        cfg = ExperimentConfig(n_subset=5, pop_size=2, n_gen=1, seed=0)
        exp = Experiment(config=cfg, dataset=_make_dataset(), evaluator=MockEvaluator())
        with pytest.raises(ValueError, match="llmlingua2_typo"):
            exp.run_baseline("llmlingua2_typo")