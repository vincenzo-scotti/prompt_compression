"""Tests for LMEvalEvaluator helpers: _apply_filter_list, _pick_metric, _get_task.

All tests use lightweight stubs — no lm-eval installation or model required.
Filter patterns and prompt formats match real lm-eval task configurations:

  GSM8K  — generate_until, two filters (strict-match / flexible-extract)
  MMLU   — multiple_choice, filter_list=None
"""
import types
import pytest
from prompt_piper.backends.lm_eval import LMEvalEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluator(metric_key=None, task=None, tasks=None):
    """Build a bare LMEvalEvaluator without calling __init__."""
    ev = LMEvalEvaluator.__new__(LMEvalEvaluator)
    ev._model = None
    ev._metric_key = metric_key
    ev._gen_kwargs = {"until": ["Question:", "</s>", "<|im_end|>"], "max_gen_toks": 2048}
    ev._task = task
    ev._tasks = tasks
    return ev


def _make_task(filter_list=None, output_type="generate_until"):
    task = types.SimpleNamespace()
    task.OUTPUT_TYPE = output_type
    cfg = types.SimpleNamespace()
    cfg.filter_list = filter_list   # intentionally None for MMLU, list for GSM8K
    cfg.generation_kwargs = {"until": ["Question:", "</s>", "<|im_end|>"], "max_gen_toks": 2048}
    task.config = cfg
    return task


# Real GSM8K filter_list as returned by lm-eval (confirmed via _load_task('gsm8k')).
# strict-match: extract the number after "#### "
# flexible-extract: extract any number-like token, picking the last match
# take_first: a no-op in our pipeline (regex already selected the match)
GSM8K_FILTER_LIST = [
    {
        "name": "strict-match",
        "filter": [
            {"function": "regex", "regex_pattern": r"#### (\-?[0-9\.\,]+)"},
            {"function": "take_first"},
        ],
    },
    {
        "name": "flexible-extract",
        "filter": [
            {"function": "regex", "group_select": -1,
             "regex_pattern": r"(-?[$0-9.,]{2,})|(-?[0-9]+)"},
            {"function": "take_first"},
        ],
    },
]

# Representative GSM8K model outputs
GSM8K_FORMATTED = "Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = $18 every day.\n#### 18"
# No "#### N" marker: strict-match falls back, flexible-extract finds the number.
# Trailing space after "18" prevents the "." character class from extending the match.
GSM8K_NATURAL   = "The answer is $18 every day at the market"
GSM8K_COT_ONLY  = "She sells 9 eggs at $2 each."  # no explicit final number token ≥ 2 chars


# ---------------------------------------------------------------------------
# _apply_filter_list — GSM8K (two real filters)
# ---------------------------------------------------------------------------

class TestApplyFilterListGSM8K:

    def test_no_filter_list_none_returns_singleton(self):
        """MMLU returns filter_list=None — must gracefully return [generated]."""
        task = _make_task(filter_list=None, output_type="multiple_choice")
        ev = _make_evaluator(task=task)
        result = ev._apply_filter_list("The answer is A.", task)
        assert result == ["The answer is A."]

    def test_no_filter_list_empty_returns_singleton(self):
        task = _make_task(filter_list=[])
        ev = _make_evaluator(task=task)
        result = ev._apply_filter_list("some text", task)
        assert result == ["some text"]

    def test_no_config_attr_returns_singleton(self):
        """Task with no .config at all (e.g. legacy task) must not crash."""
        task = types.SimpleNamespace()
        task.OUTPUT_TYPE = "generate_until"
        ev = _make_evaluator(task=task)
        result = ev._apply_filter_list("output text", task)
        assert result == ["output text"]

    def test_gsm8k_two_filters_always_return_two_results(self):
        task = _make_task(filter_list=GSM8K_FILTER_LIST)
        ev = _make_evaluator(task=task)
        for generated in [GSM8K_FORMATTED, GSM8K_NATURAL, GSM8K_COT_ONLY, "no numbers"]:
            assert len(ev._apply_filter_list(generated, task)) == 2

    def test_gsm8k_strict_match_extracts_after_hash(self):
        task = _make_task(filter_list=GSM8K_FILTER_LIST)
        ev = _make_evaluator(task=task)
        result = ev._apply_filter_list(GSM8K_FORMATTED, task)
        assert result[0] == "18"

    def test_gsm8k_strict_match_fails_on_natural_language(self):
        """Model output without #### — strict-match returns lm-eval's "[invalid]"."""
        task = _make_task(filter_list=GSM8K_FILTER_LIST)
        ev = _make_evaluator(task=task)
        result = ev._apply_filter_list(GSM8K_NATURAL, task)
        assert result[0] == "[invalid]"     # lm-eval RegexFilter fallback: no #### found

    def test_gsm8k_flexible_extract_picks_last_number(self):
        """flexible-extract uses group_select=-1 → last number in text."""
        task = _make_task(filter_list=GSM8K_FILTER_LIST)
        ev = _make_evaluator(task=task)
        result = ev._apply_filter_list("She has 9 eggs and earns $18.", task)
        # Last number-like token is "$18" or "18" depending on regex group
        assert "18" in result[1]

    def test_gsm8k_flexible_extract_succeeds_on_natural_language(self):
        task = _make_task(filter_list=GSM8K_FILTER_LIST)
        ev = _make_evaluator(task=task)
        result = ev._apply_filter_list(GSM8K_NATURAL, task)
        assert result[1] != GSM8K_NATURAL   # flexible extracted something

    def test_take_first_step_is_harmless(self):
        """take_first is not a regex function — our code skips unknown functions
        without crashing and the result is still the regex-extracted string."""
        task = _make_task(filter_list=GSM8K_FILTER_LIST)
        ev = _make_evaluator(task=task)
        result = ev._apply_filter_list(GSM8K_FORMATTED, task)
        assert result[0] == "18"    # regex ran correctly despite take_first step


# ---------------------------------------------------------------------------
# _pick_metric — GSM8K metric keys
# ---------------------------------------------------------------------------

class TestPickMetricGSM8K:

    def test_no_key_returns_first_strict_match(self):
        """Default (metric_key=None) returns the first metric — strict-match for GSM8K."""
        ev = _make_evaluator(metric_key=None)
        metrics = {"exact_match,strict-match": 0.0, "exact_match,flexible-extract": 1.0}
        assert ev._pick_metric(metrics) == 0.0

    def test_flexible_extract_key_returns_flexible_value(self):
        ev = _make_evaluator(metric_key="exact_match,flexible-extract")
        metrics = {"exact_match,strict-match": 0.0, "exact_match,flexible-extract": 1.0}
        assert ev._pick_metric(metrics) == 1.0

    def test_strict_match_key(self):
        ev = _make_evaluator(metric_key="exact_match,strict-match")
        metrics = {"exact_match,strict-match": 1.0, "exact_match,flexible-extract": 1.0}
        assert ev._pick_metric(metrics) == 1.0

    def test_missing_key_raises_with_available_keys_in_message(self):
        ev = _make_evaluator(metric_key="exact_match,wrong-filter")
        with pytest.raises(KeyError) as exc_info:
            ev._pick_metric({"exact_match,strict-match": 0.5, "exact_match,flexible-extract": 0.5})
        assert "exact_match,wrong-filter" in str(exc_info.value)
        assert "exact_match,strict-match" in str(exc_info.value)

    def test_single_metric_task(self):
        """Tasks that return a single metric (e.g. acc) work with metric_key=None."""
        ev = _make_evaluator(metric_key=None)
        assert ev._pick_metric({"acc": 0.85}) == 0.85


# ---------------------------------------------------------------------------
# _get_task — single task and MMLU-style group
# ---------------------------------------------------------------------------

class TestGetTask:

    def test_single_task_always_returned_regardless_of_sample(self):
        task = _make_task()
        ev = _make_evaluator(task=task)
        for sample in [{"_task_name": "mmlu_anatomy"}, {}, {"_doc": {}}]:
            assert ev._get_task(sample) is task

    def test_mmlu_group_routes_by_task_name(self):
        """Mirrors real MMLU structure: dict of ConfigurableTask objects keyed by subtask name."""
        t_anatomy  = _make_task(output_type="multiple_choice")
        t_algebra  = _make_task(output_type="multiple_choice")
        ev = _make_evaluator(tasks={"mmlu_anatomy": t_anatomy, "mmlu_abstract_algebra": t_algebra})

        assert ev._get_task({"_task_name": "mmlu_anatomy"}) is t_anatomy
        assert ev._get_task({"_task_name": "mmlu_abstract_algebra"}) is t_algebra

    def test_mmlu_group_unknown_task_name_falls_back_to_first(self):
        t_anatomy = _make_task(output_type="multiple_choice")
        ev = _make_evaluator(tasks={"mmlu_anatomy": t_anatomy})
        assert ev._get_task({"_task_name": "unknown_subtask"}) is t_anatomy

    def test_mmlu_group_missing_task_name_falls_back_to_first(self):
        t_anatomy = _make_task(output_type="multiple_choice")
        ev = _make_evaluator(tasks={"mmlu_anatomy": t_anatomy})
        assert ev._get_task({}) is t_anatomy

    def test_output_type_resolved_per_subtask(self):
        t_mc  = _make_task(output_type="multiple_choice")
        t_gen = _make_task(output_type="generate_until")
        ev = _make_evaluator(tasks={"mc_task": t_mc, "gen_task": t_gen})
        assert ev._get_task({"_task_name": "mc_task"}).OUTPUT_TYPE  == "multiple_choice"
        assert ev._get_task({"_task_name": "gen_task"}).OUTPUT_TYPE == "generate_until"


# ---------------------------------------------------------------------------
# Integration: real GSM8K filter_list → pick metric round-trip
# ---------------------------------------------------------------------------

class TestGSM8KIntegration:
    """Simulate what _eval_generation does with the real GSM8K filter pipeline."""

    TARGET = "18"

    def _process_results(self, doc, results):
        """Stub matching lm-eval's GSM8K process_results: exact-match per filter.

        Note: flexible-extract's pattern includes '.' in [$0-9.,] so it can
        capture trailing periods (e.g. '$18.' from 'The answer is $18.').
        We strip trailing punctuation to mirror lm-eval's own normalisation.
        """
        def _normalise(s):
            return s.strip().rstrip(".,;").replace(",", "").replace("$", "").lstrip("0") or "0"

        strict_out   = _normalise(results[0]) if len(results) > 0 else ""
        flexible_out = _normalise(results[1]) if len(results) > 1 else ""
        return {
            "exact_match,strict-match":    1.0 if strict_out   == self.TARGET else 0.0,
            "exact_match,flexible-extract": 1.0 if flexible_out == self.TARGET else 0.0,
        }

    def _run(self, generated, metric_key):
        task = _make_task(filter_list=GSM8K_FILTER_LIST)
        task.process_results = self._process_results
        ev = _make_evaluator(metric_key=metric_key, task=task)
        filtered = ev._apply_filter_list(generated, task)
        metrics  = task.process_results({}, filtered)
        return ev._pick_metric(metrics)

    def test_formatted_output_passes_strict_match(self):
        assert self._run(GSM8K_FORMATTED, "exact_match,strict-match") == 1.0

    def test_formatted_output_passes_flexible_extract(self):
        assert self._run(GSM8K_FORMATTED, "exact_match,flexible-extract") == 1.0

    def test_natural_language_fails_strict_match(self):
        assert self._run(GSM8K_NATURAL, "exact_match,strict-match") == 0.0

    def test_natural_language_passes_flexible_extract(self):
        assert self._run(GSM8K_NATURAL, "exact_match,flexible-extract") == 1.0

    def test_wrong_answer_fails_both(self):
        wrong = "The answer is $99.\n#### 99"
        assert self._run(wrong, "exact_match,strict-match")    == 0.0
        assert self._run(wrong, "exact_match,flexible-extract") == 0.0
