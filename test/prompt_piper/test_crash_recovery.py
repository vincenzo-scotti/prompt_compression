"""Crash recovery: evaluations from interrupted runs are replayed, not re-paid."""
import json
import shutil

import pytest
import yaml

from prompt_piper.optimization.evaluator import MockEvaluator
from prompt_piper.tools.compressor.config import CompressorConfig
from prompt_piper.tools.compressor.runner import CompressorRunner
from prompt_piper.tools.validate_run import validate_run


def _read(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _config(tmp_path, model_name="dummy"):
    cfg = {
        "model": {"type": "hf", "name": model_name},
        "output": {"dir": str(tmp_path), "overwrite": False},
        "defaults": {
            "search_split": "test", "test_split": "test", "n_search": 8, "n_test": 6,
            "pop_size": 4, "n_gen": 3, "use_fitting": True, "use_llm_for_semantic": False,
            "compress_field": "prompt_0shot",
            "search_transformation_ids": [f"O{i}" for i in range(1, 23)],
            "baselines": ["no_compression", "all_transformations", "random_search",
                          "multi_armed_bandit", "truncated_prompt", "random_deletion"],
        },
        "runs": [{"name": "dry", "task": "dummy"}],
    }
    path = tmp_path / f"cfg_{model_name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return CompressorConfig.from_yaml(path)


@pytest.fixture
def count_calls(monkeypatch):
    calls = {"n": 0}
    original = MockEvaluator.evaluate_batch_detailed

    def counting(self, originals, compresseds, samples):
        calls["n"] += len(compresseds)
        return original(self, originals, compresseds, samples)

    monkeypatch.setattr(MockEvaluator, "evaluate_batch_detailed", counting)
    return calls


def _crash(tmp_path):
    """Simulate a crash before the result was saved: keep the log, drop the result."""
    (tmp_path / "dry.json").unlink()


def test_full_replay_makes_no_model_calls_and_reproduces_results(tmp_path, count_calls):
    first = CompressorRunner(_config(tmp_path), dry_run=True).run_all()[0]
    n_first = count_calls["n"]
    first_evals = _read(tmp_path / "dry_log" / "evaluations.jsonl")
    assert n_first == len(first_evals) > 0
    _crash(tmp_path)

    second = CompressorRunner(_config(tmp_path), dry_run=True).run_all()[0]
    assert count_calls["n"] == n_first                      # no new model calls
    evals = _read(tmp_path / "dry_log" / "evaluations.jsonl")
    assert len(evals) == len(first_evals)
    assert all("replayed_from" in e for e in evals)
    meta = json.loads((tmp_path / "dry_log" / "meta.json").read_text())
    assert meta["replay"]["used"] == len(evals)
    # identical trajectory and results
    assert [s.bits for s in second.pareto_solutions] == [s.bits for s in first.pareto_solutions]
    assert [s.test_avg_performance for s in second.pareto_solutions] == \
        [s.test_avg_performance for s in first.pareto_solutions]
    assert validate_run(tmp_path / "dry_log")["ok"]


def test_partial_replay_after_crash_mid_run(tmp_path, count_calls):
    CompressorRunner(_config(tmp_path), dry_run=True).run_all()
    _crash(tmp_path)
    # Cut the log mid-run, with a half-written last line, as a crash would.
    path = tmp_path / "dry_log" / "evaluations.jsonl"
    lines = path.read_text().splitlines()
    keep = len(lines) // 2
    path.write_text("\n".join(lines[:keep]) + "\n" + lines[keep][: len(lines[keep]) // 2])
    calls_before = count_calls["n"]

    CompressorRunner(_config(tmp_path), dry_run=True).run_all()
    evals = _read(tmp_path / "dry_log" / "evaluations.jsonl")
    replayed = [e for e in evals if "replayed_from" in e]
    assert len(replayed) == keep                             # everything logged is reused
    assert count_calls["n"] - calls_before == len(evals) - keep  # only the rest is paid
    assert validate_run(tmp_path / "dry_log")["ok"]
    assert len(list(tmp_path.glob("dry_log.incomplete-*"))) == 1


def test_different_model_is_never_replayed(tmp_path, count_calls):
    CompressorRunner(_config(tmp_path, "model-a"), dry_run=True).run_all()
    _crash(tmp_path)
    calls_before = count_calls["n"]
    CompressorRunner(_config(tmp_path, "model-b"), dry_run=True).run_all()
    evals = _read(tmp_path / "dry_log" / "evaluations.jsonl")
    assert not any("replayed_from" in e for e in evals)
    assert count_calls["n"] - calls_before == len(evals)
    meta = json.loads((tmp_path / "dry_log" / "meta.json").read_text())
    assert "fingerprint mismatch" in meta["replay"]["sources"][0]["reason"]


def test_evaluations_are_logged_in_chunks(tmp_path, monkeypatch):
    from prompt_piper.optimization.problem import PromptCompressionProblem

    original_init = PromptCompressionProblem.__init__

    def small_chunks(self, *a, **kw):
        kw["eval_chunk_size"] = 3
        original_init(self, *a, **kw)

    monkeypatch.setattr(PromptCompressionProblem, "__init__", small_chunks)
    CompressorRunner(_config(tmp_path), dry_run=True).run_all()
    evals = _read(tmp_path / "dry_log" / "evaluations.jsonl")
    assert max(e["batch_size"] for e in evals) <= 3
    assert validate_run(tmp_path / "dry_log")["ok"]
