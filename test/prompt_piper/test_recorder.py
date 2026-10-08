"""Tests for complete run logging and disjoint search/test sampling.

Covers:
  - sample_indices reproduces the historical search subsets
  - CompressorRunner._split_indices keeps search and test disjoint on one split
  - an end-to-end dry run logs every candidate and per-sample result, the
    result JSON can be recomputed from the logs, and validation passes
  - LMEvalEvaluator.evaluate_batch_detailed returns raw outputs and metrics
"""
import json
import random
import types
from collections import Counter

import yaml

from prompt_piper.backends import lm_eval as lm_eval_backend
from prompt_piper.backends.lm_eval import LMEvalEvaluator
from prompt_piper.optimization.bits import bits_to_ids, bits_to_names, ids_to_bits, make_layout
from prompt_piper.optimization.runner import sample_indices
from prompt_piper.transformations import ALL_TRANSFORMATIONS
from prompt_piper.tools.compressor.config import CompressorConfig, RunConfig
from prompt_piper.tools.compressor.runner import CompressorRunner
from prompt_piper.tools.validate_run import validate_run


def _read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestSampling:
    def test_same_indices_as_legacy_rng_sample(self):
        data = list(range(1319))
        legacy = random.Random(42).sample(data, 50)
        assert sample_indices(len(data), 50, 42) == legacy

    def test_full_pool_when_n_too_large(self):
        assert sample_indices(5, 10, 0, exclude=[1]) == [0, 2, 3, 4]

    def test_same_split_is_disjoint(self):
        cfg = RunConfig(task="t", search_split="test", test_split="test", n_search=50, n_test=50)
        s, t = CompressorRunner._split_indices(cfg, 250, 250)
        assert len(s) == 50 and len(t) == 50
        assert not set(s) & set(t)

    def test_different_splits_unchanged(self):
        cfg = RunConfig(task="t", search_split="validation", test_split="test", n_search=10, n_test=20)
        _, t = CompressorRunner._split_indices(cfg, 100, 500)
        assert t == random.Random(cfg.subset_seed + 1).sample(list(range(500)), 20)


class TestDryRunLogging:
    def _run(self, tmp_path):
        cfg = {
            "model": {"type": "hf", "name": "dummy"},
            "output": {"dir": str(tmp_path), "overwrite": False},
            "defaults": {
                "search_split": "test", "test_split": "test",
                "n_search": 8, "n_test": 6, "pop_size": 4, "n_gen": 3,
                "use_fitting": True, "use_llm_for_semantic": False,
                "compress_field": "prompt_0shot",
                "search_transformation_ids": [f"O{i}" for i in range(1, 23)],
                "baselines": [
                    "no_compression", "all_transformations", "random_search",
                    "multi_armed_bandit", "summarization", "truncated_prompt",
                    "random_deletion",
                ],
            },
            "runs": [{"name": "dry", "task": "dummy"}],
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(cfg))
        runner = CompressorRunner(CompressorConfig.from_yaml(path), dry_run=True)
        return runner.run_all()[0]

    def test_logs_complete_and_valid(self, tmp_path):
        result = self._run(tmp_path)
        log_dir = tmp_path / "dry_log"
        report = validate_run(log_dir)
        assert report["ok"], report["errors"]
        assert result.validation["ok"]

        meta = json.loads((log_dir / "meta.json").read_text())
        assert not set(meta["sample_ids"]["search"]) & set(meta["sample_ids"]["test"])
        assert result.search_sample_ids == meta["sample_ids"]["search"]

        cands = _read(log_dir / "candidates.jsonl")
        by = Counter((c["method"], c["split"]) for c in cands)
        assert by[("nsga2", "search")] == 4 * 3
        assert by[("random_search", "search")] == 4 * 3
        assert by[("multi_armed_bandit", "search")] == 4 * 3
        for c in cands:
            assert len(c["per_sample"]) == (8 if c["split"] == "search" else 6)

    def test_result_recomputable_from_logs(self, tmp_path):
        result = self._run(tmp_path)
        cands = _read(tmp_path / "dry_log" / "candidates.jsonl")
        test = {c["label"]: c for c in cands if c["split"] == "test" and c["method"] == "nsga2"}
        for i, sol in enumerate(result.pareto_solutions):
            c = test[f"pareto:{i}"]
            scores = [ps["score"] for ps in c["per_sample"]]
            tokens = [ps["tokens"] for ps in c["per_sample"]]
            assert abs(sum(scores) / len(scores) - sol.test_avg_performance) < 1e-12
            assert abs(sum(tokens) / len(tokens) - sol.test_avg_tokens) < 1e-12
        rs = next(b for b in result.baselines if b.name == "random_search")
        assert rs.solutions and all(s["test_avg_performance"] is not None for s in rs.solutions)

    def test_intermediate_fronts_logged(self, tmp_path):
        result = self._run(tmp_path)
        states = _read(tmp_path / "dry_log" / "states.jsonl")
        fronts = [s for s in states if s["kind"] == "front"]
        steps = Counter(s["algorithm"] for s in fronts)
        assert steps == {"nsga2": 3, "random_search": 12, "multi_armed_bandit": 12}
        # The last logged front equals the front each method returned.
        for name in ("random_search", "multi_armed_bandit"):
            last = max((s for s in fronts if s["algorithm"] == name), key=lambda s: s["step"])
            returned = next(b for b in result.baselines if b.name == name).solutions
            assert sorted(map(tuple, (p["bits"] for p in last["pareto_front"]))) == \
                sorted(map(tuple, (s["bits"] for s in returned)))
        last = max((s for s in fronts if s["algorithm"] == "nsga2"), key=lambda s: s["step"])
        assert sorted(map(tuple, (p["bits"] for p in last["pareto_front"]))) == \
            sorted(map(tuple, (s.bits for s in result.pareto_solutions)))

    def test_bits_round_trip_in_results(self, tmp_path):
        result = self._run(tmp_path)
        search, full = result.bit_layouts["search"], result.bit_layouts["full"]
        assert len(search) == 3 * 22 and len(full) == 3 * 28
        for sol in result.pareto_solutions:
            assert ids_to_bits(search, sol.transformation_ids) == sol.bits
            assert bits_to_names(search, sol.bits) == sol.transformation_names
        for b in result.baselines:
            for sol in b.solutions:
                if sol["layout"] is None:  # text-level baseline: no operator bits
                    assert b.name in ("truncated_prompt", "random_deletion", "llmlingua2")
                    assert sol["bits"] is None
                    continue
                layout = result.bit_layouts[sol["layout"]]
                assert ids_to_bits(layout, sol["transformation_ids"]) == sol["bits"]
                assert bits_to_names(layout, sol["bits"]) == sol["transformation_names"]
        layouts = {b.name: b.solutions[0]["layout"] for b in result.baselines}
        assert layouts["all_transformations"] == "search"   # O1–O22, as in Table IV
        assert layouts["summarization"] == "full"           # needs O23–O28
        all_t = next(b for b in result.baselines if b.name == "all_transformations")
        assert len(all_t.solutions[0]["transformation_ids"]) == 3 * 22

    def test_samples_seeds_sections_logged(self, tmp_path):
        result = self._run(tmp_path)
        log_dir = tmp_path / "dry_log"
        samples = _read(log_dir / "samples.jsonl")
        assert sorted(r["sample_id"] for r in samples if r["role"] == "search") == sorted(result.search_sample_ids)
        assert sorted(r["sample_id"] for r in samples if r["role"] == "test") == sorted(result.test_sample_ids)
        assert all("prompt_0shot" in r and "target" in r for r in samples)
        sections = _read(log_dir / "sections.jsonl")
        assert {(r["split"], r["sample_id"]) for r in sections} == (
            {("search", i) for i in result.search_sample_ids} | {("test", i) for i in result.test_sample_ids}
        )
        assert result.seeds["test_seed"] == result.seeds["subset_seed"] + 1

    def test_cache_hits_accounted(self, tmp_path):
        self._run(tmp_path)
        log_dir = tmp_path / "dry_log"
        cands = _read(log_dir / "candidates.jsonl")
        evals = _read(log_dir / "evaluations.jsonl")
        new = sum(not ps["cached"] for c in cands for ps in c["per_sample"])
        assert new == len(evals)
        # Within a method, no (split, sample, prompt) is sent to the model twice.
        keys = [(e["method"], e["split"], e["sample_id"], e["prompt_sha"]) for e in evals]
        assert len(keys) == len(set(keys))


class TestDetailedEvaluator:
    def test_generation_details(self, monkeypatch):
        ev = LMEvalEvaluator.__new__(LMEvalEvaluator)
        ev._model = None
        ev._metric_key = None
        ev._gen_kwargs = {"max_gen_toks": 16}
        task = types.SimpleNamespace(
            OUTPUT_TYPE="generate_until",
            config=types.SimpleNamespace(filter_list=None),
            process_results=lambda doc, filtered: {"exact_match": float(filtered[0] == "42")},
        )
        ev._task, ev._tasks = task, None
        ev._apply_chat_template = lambda p: p
        monkeypatch.setattr(lm_eval_backend, "_is_chat_model", lambda m: False)
        monkeypatch.setattr(lm_eval_backend, "_call_model", lambda m, meth, reqs: ["42", "7"])

        details = ev.evaluate_batch_detailed(["a", "b", ""], ["a", "b", ""], [{"_doc": {}}] * 3)
        assert [d["score"] for d in details] == [1.0, 0.0, 0.0]
        assert details[0]["output"] == "42" and details[0]["metrics"] == {"exact_match": 1.0}
        assert details[2]["error"] == "empty prompt"
        assert ev.evaluate_batch(["a", "b", ""], ["a", "b", ""], [{"_doc": {}}] * 3) == [1.0, 0.0, 0.0]


class TestBitLayout:
    def test_round_trip_all_single_bits(self):
        ops = ALL_TRANSFORMATIONS[:22]
        layout = make_layout(ops, ops, ops)
        for i in range(len(layout)):
            bits = [0] * len(layout)
            bits[i] = 1
            ids = bits_to_ids(layout, bits)
            assert ids == [f"{layout[i]['op_id']}@{layout[i]['section']}"]
            assert ids_to_bits(layout, ids) == bits

    def test_matches_problem_names(self):
        import numpy as np
        from prompt_piper.optimization.config import ExperimentConfig
        from prompt_piper.optimization.runner import Experiment

        data = [{"prompt": f"Classify text {i}. Answer:", "target": "a"} for i in range(4)]
        exp = Experiment(ExperimentConfig(n_subset=4, search_transformation_ids=[f"O{i}" for i in range(1, 23)]), data)
        problem = exp._build_problem(data)
        assert problem.bit_layout() == exp.bit_layout()
        rng = np.random.default_rng(0)
        for _ in range(20):
            bits = rng.integers(0, 2, problem.n_var)
            assert bits_to_names(problem.bit_layout(), bits) == problem.names_from_bits(bits)


class TestPersistentSemanticCache:
    def test_reused_across_instances_and_logged(self, tmp_path, monkeypatch):
        from prompt_piper.backends.lm_eval import LMEvalSemanticBackend
        from prompt_piper.optimization.recorder import RunRecorder

        calls = []

        def fake_call(model, method, reqs):
            calls.append(reqs)
            out = lm_eval_backend.GenerationText(" short ")
            out.usage = {"prompt_tokens": 9, "completion_tokens": 2}
            return [out]

        monkeypatch.setattr(lm_eval_backend, "_call_model", fake_call)
        monkeypatch.setattr(lm_eval_backend, "_is_chat_model", lambda m: False)
        monkeypatch.setattr(lm_eval_backend, "_hf_model_has_chat_template", lambda m: False)
        path = tmp_path / "cache.jsonl"

        b1 = LMEvalSemanticBackend(object(), cache_path=str(path), model_id="m")
        rec = RunRecorder(tmp_path / "log")
        b1.recorder = rec
        assert b1("long text", "Summarise") == "short"
        assert b1("long text", "Summarise") == "short"          # memory hit
        assert len(calls) == 1

        b2 = LMEvalSemanticBackend(object(), cache_path=str(path), model_id="m")
        b2.recorder = rec
        assert b2("long text", "Summarise") == "short"          # disk hit
        assert len(calls) == 1
        b3 = LMEvalSemanticBackend(object(), cache_path=str(path), model_id="other")
        b3("long text", "Summarise")                            # other model: miss
        assert len(calls) == 2
        rec.close()
        records = _read(tmp_path / "log" / "llm_calls.jsonl")
        assert [r["source"] for r in records] == ["llm", "memory", "disk"]
        assert records[0]["provider_usage"] == {"prompt_tokens": 9, "completion_tokens": 2}
        assert _read(path)[0]["provider_usage"]["completion_tokens"] == 2


class TestAnalysis:
    def test_analysis_on_dry_run(self, tmp_path):
        from prompt_piper import analysis as A

        (tmp_path / "mock-model").mkdir()
        result = TestDryRunLogging()._run(tmp_path / "mock-model")
        runs = A.load_runs(tmp_path)
        assert len(runs) == 1
        run = runs[0]
        pts = A.method_points(runs, selection="search")
        assert set(pts.method) == {b.name for b in result.baselines} | {"nsga2"}
        assert not pts.legacy_front_mean.any()
        # NSGA-II point = search-selected compromise, test values from the result
        i, sol = A.select_compromise(run, "nsga2", "search")
        row = pts[pts.method == "nsga2"].iloc[0]
        assert row.test_performance == sol["test_avg_performance"]
        ps = A.per_sample(run, "nsga2", i)
        assert abs(ps.score.mean() - sol["test_avg_performance"]) < 1e-12
        # every method's point is recomputable from per-sample logs
        for _, r in pts.iterrows():
            d = A.per_sample(run, r.method, int(r.solution_index))
            assert abs(d.score.mean() - r.test_performance) < 1e-12
            assert abs(d.tokens.mean() - r.test_tokens) < 1e-12
        conv = A.convergence(run)
        assert conv.groupby("method").size().to_dict() == {"nsga2": 3, "random_search": 12, "multi_armed_bandit": 12}
        c = A.cost(run)
        evals = _read(run.log_dir / "evaluations.jsonl")
        assert c["search_llm_calls"] + c["test_llm_calls"] == len(evals)
        assert A.hypervolume([(0.5, 0.5)]) == 0.25
        assert A.hypervolume([(0.5, 0.5), (0.2, 1.0)]) == 0.2 * 1.0 + 0.3 * 0.5


class TestNeutralBits:
    def test_effective_operators_in_results(self, tmp_path):
        result = TestDryRunLogging()._run(tmp_path)
        assert "search" in result.noop_operators
        layout = result.bit_layouts["search"]
        exp_problem = None
        for sol in result.pareto_solutions:
            assert set(sol.effective_transformation_names) <= set(sol.transformation_names)
        for b in result.baselines:
            if b.name in ("random_search", "multi_armed_bandit", "all_transformations"):
                assert all("effective_transformation_names" in s for s in b.solutions)

    def test_effective_bits_definition(self):
        from prompt_piper.optimization.config import ExperimentConfig
        from prompt_piper.optimization.runner import Experiment

        data = [{"prompt": f"Please   classify\\n\\nText {i}:  good\\n\\nAnswer:", "target": "a"} for i in range(4)]
        exp = Experiment(ExperimentConfig(n_subset=4, search_transformation_ids=[f"O{i}" for i in range(1, 23)]), data)
        p = exp.compression_problem()
        rng = __import__("numpy").random.default_rng(1)
        for _ in range(10):
            bits = list(rng.integers(0, 2, p.n_var))
            eff = set(p.effective_bits(bits))
            _, base = p._compress_bits(bits)
            for i, b in enumerate(bits):
                if b and i not in eff:          # neutral: removing it changes nothing
                    flipped = list(bits); flipped[i] = 0
                    assert p._compress_bits(flipped)[1] == base
        noop = set(p.noop_bits())
        for i in noop:
            single = [0] * p.n_var; single[i] = 1
            assert p._compress_bits(single)[1] == p._compress_bits([0] * p.n_var)[1]


def test_fewshot_as_multiturn_rejected(tmp_path):
    import pytest
    cfg = RunConfig(task="dummy", name="x", fewshot_as_multiturn=True)
    runner = CompressorRunner(CompressorConfig(), dry_run=True)
    runner._config.output.dir = str(tmp_path)
    with pytest.raises(ValueError, match="fewshot_as_multiturn"):
        runner.run_single(cfg)


def test_transfer_fits_operators_on_target_search_sample(tmp_path):
    cfg = {
        "model": {"type": "hf", "name": "dummy"},
        "output": {"dir": str(tmp_path), "overwrite": False},
        "defaults": {
            "search_split": "test", "test_split": "test", "n_search": 8, "n_test": 6,
            "pop_size": 4, "n_gen": 2, "use_fitting": True, "use_llm_for_semantic": False,
            "compress_field": "prompt_0shot", "baselines": ["no_compression"],
        },
        "runs": [{"name": "a", "task": "dummy", "subset_seed": 1},
                 {"name": "b", "task": "dummy", "subset_seed": 2}],
        "transferability": {"enabled": True, "pairs": [["a", "b"]]},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    CompressorRunner(CompressorConfig.from_yaml(path), dry_run=True).run_all()
    meta = json.loads((tmp_path / "transfer_a_to_b_log" / "meta.json").read_text())
    target = json.loads((tmp_path / "b.json").read_text())
    assert meta["fitted_on"] == "target search sample"
    assert meta["fitting_report"] == target["fitting_report"]       # same operators as the native run
    assert meta["sample_ids"]["test"] == target["test_sample_ids"]  # same test sample
    assert validate_run(tmp_path / "transfer_a_to_b_log")["ok"]


def test_provider_prompt_tokens_on_every_sample(tmp_path):
    from prompt_piper.optimization.config import ExperimentConfig
    from prompt_piper.optimization.evaluator import Evaluator
    from prompt_piper.optimization.recorder import RunRecorder
    from prompt_piper.optimization.runner import Experiment

    class UsageEvaluator(Evaluator):
        calls = 0

        def evaluate(self, original, compressed, sample):
            return 1.0

        def evaluate_batch_detailed(self, originals, compresseds, samples):
            UsageEvaluator.calls += len(compresseds)
            return [{"score": 1.0, "provider_usage": {"prompt_tokens": len(c) + 7}} for c in compresseds]

    data = [{"prompt": f"Classify text {i} please.", "target": "a", "_sample_id": i, "_split": "test"}
            for i in range(4)]
    rec = RunRecorder(tmp_path / "log")
    exp = Experiment(ExperimentConfig(n_subset=4, pop_size=4, n_gen=2), data,
                     evaluator=UsageEvaluator(), recorder=rec)
    with rec.phase("search", method="x"):
        exp.evaluate_solutions([[0] * exp.bit_layout().__len__()] * 2, data, split="search")
    rec.close()
    cands = _read(tmp_path / "log" / "candidates.jsonl")
    assert len(cands) == 2 and cands[1]["n_cache_hits"] == 4          # second is all cache hits
    for c in cands:
        for ps in c["per_sample"]:
            assert ps["provider_prompt_tokens"] is not None            # also for cache hits


def test_state_averages_backed_by_per_sample_values(tmp_path):
    TestDryRunLogging()._run(tmp_path)
    log_dir = tmp_path / "dry_log"
    cands = {c["id"]: c for c in _read(log_dir / "candidates.jsonl")}
    n = 0
    for st in _read(log_dir / "states.jsonl"):
        entries = (st.get("population", []) + st.get("offspring", []) if st["kind"] == "population"
                   else st.get("pareto_front", []) if st["kind"] == "front"
                   else [{"candidate_id": st["candidate_id"], "avg_performance": st["avg_performance"],
                          "avg_tokens": st["avg_tokens"]}])
        for e in entries:
            per = cands[e["candidate_id"]]["per_sample"]
            assert abs(sum(p["score"] for p in per) / len(per) - e["avg_performance"]) < 1e-12
            assert abs(sum(p["tokens"] for p in per) / len(per) - e["avg_tokens"]) < 1e-12
            n += 1
    assert n > 0


def test_review_fixes(tmp_path):
    from prompt_piper import analysis as A

    (tmp_path / "m").mkdir()
    result = TestDryRunLogging()._run(tmp_path / "m")
    layout = result.bit_layouts["search"]
    # generation history names derived from the correct layout
    for gen in result.generation_history:
        for p in gen["pareto_front"]:
            assert p["transformation_names"] == bits_to_names(layout, p["bits"])
    # search/eval times cover only the primary algorithm's phases
    phases = _read(tmp_path / "m" / "dry_log" / "phases.jsonl")
    search = sum(p["elapsed_seconds"] for p in phases if p["event"] == "end" and p["phase"] == "search")
    assert abs(result.search_elapsed_seconds - search) < 0.5
    # standalone cost per method
    run = A.load_runs(tmp_path)[0]
    sc = A.standalone_cost(run).set_index("method")
    assert sc.loc["nsga2", "candidates"] == 12 and sc.loc["random_search", "candidates"] == 12
    assert (sc.standalone_llm_calls >= sc.llm_calls_made_in_run).all()
    # per_sample on the search split selects the requested solution
    i, sol = A.select_compromise(run, "nsga2", "search")
    d = A.per_sample(run, "nsga2", i, split="search")
    assert abs(d.score.mean() - sol["search_avg_performance"]) < 1e-12


def test_cli_overrides_keep_transferability():
    cfg = CompressorConfig.from_dict({
        "runs": [{"name": "a", "task": "t"}, {"name": "b", "task": "t"}],
        "transferability": {"enabled": True, "pairs": [["a", "b"]]},
    })
    new = cfg.apply_cli_overrides({"seed": 7})
    assert new.transferability.enabled and new.transferability.pairs == [["a", "b"]]
    assert all(r.seed == 7 for r in new.runs)


def test_undersized_pool_raises():
    import pytest
    cfg = RunConfig(task="t", search_split="test", test_split="test", n_search=50, n_test=50)
    with pytest.raises(ValueError, match="only"):
        CompressorRunner._split_indices(cfg, 80, 80)


def test_each_method_pays_for_its_own_evaluations(tmp_path):
    from prompt_piper import analysis as A

    (tmp_path / "m").mkdir()
    TestDryRunLogging()._run(tmp_path / "m")
    run = A.load_runs(tmp_path)[0]
    sc = A.standalone_cost(run).set_index("method")
    # isolated caches: calls made in the run == calls the method needs alone
    assert (sc.standalone_llm_calls == sc.llm_calls_made_in_run).all()
    # no candidate resolves to an evaluation made by another method
    evals = {e["id"]: e for e in _read(tmp_path / "m" / "dry_log" / "evaluations.jsonl")}
    for c in _read(tmp_path / "m" / "dry_log" / "candidates.jsonl"):
        for ps in c["per_sample"]:
            assert evals[ps["eval_id"]]["method"] == c["method"]


def test_preserve_answer_cue_option():
    from prompt_piper.optimization.config import ExperimentConfig
    from prompt_piper.optimization.runner import Experiment

    data = [{"prompt": f"Instr.\n\nQ: question {i} with many words here\nA:", "target": "x",
             "_input_text": f"Q: question {i} with many words here\nA:"} for i in range(4)]
    cut = lambda t: t[: len(t) // 2]
    for preserve, expect_cue in ((False, False), (True, True)):
        exp = Experiment(ExperimentConfig(n_subset=4, preserve_answer_cue=preserve), data)
        p = exp._build_problem(data)
        assert p._sections[0].output_cue == "A:"
        texts = [s.with_cue(cut(s.body())) if preserve else cut(s.assemble()) for s in p._sections]
        assert all(t.endswith("A:") for t in texts) == expect_cue
        # the problem's own path produces the same texts
        captured = []
        p._evaluator.evaluate_batch_detailed = lambda o, c, s: captured.extend(c) or [{"score": 0.0}] * len(c)
        p.score_transform_fn(cut)
        assert captured == texts


def test_sections_log_rebuilds_prompts_exactly(tmp_path):
    from prompt_piper.transformations.sections import PromptSections

    TestDryRunLogging()._run(tmp_path)
    log_dir = tmp_path / "dry_log"
    texts = {t["sha"]: t["text"] for t in _read(log_dir / "texts.jsonl")}
    fields = ("instruction", "shots", "input_data", "output_cue", "sep_after_instruction",
              "sep_after_shots", "sep_before_cue", "tail")
    for r in _read(log_dir / "sections.jsonl"):
        assert PromptSections(**{k: r[k] for k in fields}).assemble() == texts[r["assembled_sha"]]
