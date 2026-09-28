"""Integration test on the real ICSE'26 benchmarks (skipped without cached datasets).

For every sample of every task, the uncompressed prompt assembled from the
instruction / shots / data / cue sections must equal the instruction prefix +
lm-eval's own prompt (including gen_prefix) character for character, the
static sections must be identical across samples, and the configured number of
few-shot examples must be present.
"""
import os

import pytest

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

pytest.importorskip("lm_eval")

from pathlib import Path

from prompt_piper.optimization.runner import Experiment
from prompt_piper.tools.compressor.config import CompressorConfig
from prompt_piper.tools.compressor.runner import CompressorRunner

CONFIG = Path(__file__).resolve().parents[2] / "resources/configs/icse_26/experiments_qwen35_9b.yaml"
# marker counted once per example (shots) + once for the question
MARKERS = {
    "mbpp_instruct": ("You are an expert Python programmer", 3),
    "mmlu_pro_computer_science": ("\nQuestion:\n", 5),
    "gsm8k_cot": ("Q: ", 8),
    "bbh_cot_fewshot_logical_deduction_seven_objects": ("Q: ", 3),
}


def _runs():
    cfg = CompressorConfig.from_yaml(CONFIG)
    return cfg, [r for r in cfg.runs if r.task in MARKERS]


@pytest.fixture(scope="module")
def loaded():
    from prompt_piper.backends.lm_eval import _load_task

    cfg, runs = _runs()
    runner = CompressorRunner(cfg, dry_run=False)
    out = {}
    for run in runs:
        try:
            backend = {"task": _load_task(run.task, num_fewshot=run.num_fewshot)}
            out[run.task] = (run, runner._get_dataset(backend, run, split=run.test_split))
        except Exception as exc:  # dataset not cached
            pytest.skip(f"{run.task}: {exc}")
    return out


@pytest.mark.parametrize("task", sorted(MARKERS))
def test_assembled_prompt_equals_lm_eval(loaded, task):
    run, data = loaded[task]
    exp = Experiment(CompressorRunner._make_exp_config(run, n_subset=len(data)), data)
    problem = exp._build_problem(data)
    prefix = run.instruction_prefix.rstrip("\n") + "\n\n" if run.instruction_prefix else ""
    for sec, sample in zip(problem._sections, data):
        assert sec.assemble() == prefix + sample["_reference_prompt"]
    first = problem._sections[0]
    for sec in problem._sections:
        assert (sec.instruction, sec.shots, sec.output_cue, sec.tail) == \
            (first.instruction, first.shots, first.output_cue, first.tail)
    marker, n_shots = MARKERS[task]
    assert ("\n" + first.shots).count(marker) == n_shots
    assert ("\n" + data[0]["_reference_prompt"]).count(marker) == n_shots + 1
    # the per-sample question is in the data section, not in the static parts
    assert first.input_data and first.input_data not in first.instruction


def test_bbh_cue_not_duplicated(loaded):
    run, data = loaded["bbh_cot_fewshot_logical_deduction_seven_objects"]
    exp = Experiment(CompressorRunner._make_exp_config(run, n_subset=len(data)), data)
    sec = exp._build_problem(data)._sections[0]
    assert sec.assemble().count("A: Let's think step by step.") == 3 + 1  # 3 shots + question
    assert "A: Let's think step by step." not in sec.input_data


def test_mbpp_gen_prefix_kept_uncompressed(loaded):
    run, data = loaded["mbpp_instruct"]
    exp = Experiment(CompressorRunner._make_exp_config(run, n_subset=len(data)), data)
    problem = exp._build_problem(data)
    assert problem._sections[0].tail.endswith("```python\n")
    bits = [1] * problem.n_var
    _, compressed = problem._compress_bits(bits)
    assert all(c.endswith("```python\n") for c in compressed)


@pytest.mark.parametrize("generation, expected", [
    ("{code}```", 1.0),                          # completion continuing the prefilled opener
    ("```python\n{code}```", 1.0),               # chat model opening its own block
    ("Here:\n```python\n{code}```\nDone.", 1.0),  # prose before the block
    ("{code}", 1.0),                             # bare code
    ("```python\ndef remove_Occ(s,ch):\n    return s\n```", 0.0),  # wrong code
    ("I cannot do that.", 0.0),
])
def test_mbpp_code_extraction(loaded, generation, expected):
    from prompt_piper.backends.lm_eval import LMEvalEvaluator

    run, data = loaded["mbpp_instruct"]
    from prompt_piper.backends.lm_eval import _load_task
    task = _load_task("mbpp_instruct", num_fewshot=3)
    doc = next(d for d in task.test_docs() if "remove_Occ" in d["test_list"][0])
    code = ("def remove_Occ(s,ch):\n    s=s.replace(ch,'',1)\n    i=s.rfind(ch)\n"
            "    return s[:i]+s[i+1:] if i>=0 else s\n")
    ev = LMEvalEvaluator.__new__(LMEvalEvaluator)
    ev._metric_key = None
    filtered = ev._apply_filter_list(generation.format(code=code), task, doc)
    assert float(task.process_results(doc, filtered)["pass_at_1"]) == expected


@pytest.mark.parametrize("task", sorted(MARKERS))
def test_search_and_test_share_static_sections(loaded, task):
    run, data = loaded[task]
    s_idx, t_idx = CompressorRunner._split_indices(run, len(data), len(data))
    assert not set(s_idx) & set(t_idx)
    exp = Experiment(CompressorRunner._make_exp_config(run, n_subset=len(s_idx)), data)

    class _Meta:
        meta = {}
        def update_meta(self, **kw):
            self.meta.update(kw)

    rec = _Meta()
    CompressorRunner._check_same_structure(exp, [data[i] for i in s_idx], [data[i] for i in t_idx], rec)
    assert rec.meta["static_sections_identical_across_splits"]
