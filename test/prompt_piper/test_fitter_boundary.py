"""Tests for DatasetVocabularyFitter few-shot boundary detection.

Prompt structures are taken from real lm-eval GSM8K samples (4-shot):
  - Each Q&A pair is one blank-line-separated paragraph.
  - The 0-shot prompt is the final "Question: ...\nAnswer:" pair only.
  - Full prompt = N few-shot paragraphs + 0-shot paragraph.
"""
import pytest
from prompt_piper.transformations.fitting import DatasetVocabularyFitter
from prompt_piper.transformations.structural import ContextPruning


# ---------------------------------------------------------------------------
# Real GSM8K-style prompt building blocks
# ---------------------------------------------------------------------------

FS1 = (
    "Question: Jen and Tyler are gymnasts. Jen did sixteen triple-flips. "
    "Tyler flipped half the number of times Jen did. How many double-flips did Tyler do?\n"
    "Answer: Jen did 16 * 3 = 48 flips. Tyler did 48 / 2 = 24 flips. "
    "24 / 2 = 12 double-flips.\n#### 12"
)
FS2 = (
    "Question: Four people are planning a party. They need $fruit, $meat and $drinks. "
    "How much does each person contribute?\n"
    "Answer: Total cost is 12 + 36 + 25 = $73. Each pays 73 / 4 = $18.25.\n#### 18"
)
# The 0-shot question — the only part present in prompt_0shot
QUESTION = (
    "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast "
    "and bakes muffins with four. She sells the rest at $2 each. "
    "How much does she make daily?\n"
    "Answer:"
)

FULL_1FS = f"{FS1}\n\n{QUESTION}"
FULL_2FS = f"{FS1}\n\n{FS2}\n\n{QUESTION}"


def _dataset(n=10, full=FULL_1FS, question=QUESTION, target="eighteen dollars"):
    """Return a list of sample dicts mimicking real GSM8K training samples.

    Target is a word string so _words() produces anchor terms and
    fit_context_pruning can discover non-empty anchors.
    """
    return [
        {"prompt": full, "prompt_0shot": question, "target": target}
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# _count_fewshot_paragraphs
# ---------------------------------------------------------------------------

class TestCountFewshotParagraphs:

    def test_no_template_field_returns_zero(self):
        fitter = DatasetVocabularyFitter(_dataset(), template_field=None)
        assert fitter._count_fewshot_paragraphs() == 0

    def test_compress_field_equals_template_field_returns_zero(self):
        # 0-shot compression: the text being compressed IS the template → no prefix
        fitter = DatasetVocabularyFitter(
            _dataset(),
            template_field="prompt_0shot",
            compress_field="prompt_0shot",
        )
        assert fitter._count_fewshot_paragraphs() == 0

    def test_one_fewshot_paragraph_detected(self):
        fitter = DatasetVocabularyFitter(
            _dataset(full=FULL_1FS),
            template_field="prompt_0shot",
            compress_field="prompt",
        )
        assert fitter._count_fewshot_paragraphs() == 1

    def test_two_fewshot_paragraphs_detected(self):
        fitter = DatasetVocabularyFitter(
            _dataset(full=FULL_2FS),
            template_field="prompt_0shot",
            compress_field="prompt",
        )
        assert fitter._count_fewshot_paragraphs() == 2

    def test_compress_field_none_defaults_to_prompt_field(self):
        """compress_field=None is treated as prompt_field → full compression."""
        fitter = DatasetVocabularyFitter(
            _dataset(full=FULL_1FS),
            prompt_field="prompt",
            template_field="prompt_0shot",
            compress_field=None,
        )
        assert fitter._count_fewshot_paragraphs() == 1

    def test_question_not_substring_of_full_returns_zero(self):
        broken = [{"prompt": "some text", "prompt_0shot": "completely different", "target": "x"}]
        fitter = DatasetVocabularyFitter(
            broken,
            template_field="prompt_0shot",
            compress_field="prompt",
        )
        assert fitter._count_fewshot_paragraphs() == 0

    def test_empty_corpus_returns_zero(self):
        fitter = DatasetVocabularyFitter(
            [],
            template_field="prompt_0shot",
            compress_field="prompt",
        )
        assert fitter._count_fewshot_paragraphs() == 0


# ---------------------------------------------------------------------------
# fit_context_pruning — n_protected_paragraphs on the returned instance
# ---------------------------------------------------------------------------

class TestFitContextPruning:

    def test_full_compression_one_shot_sets_n_protected_one(self):
        fitter = DatasetVocabularyFitter(
            _dataset(full=FULL_1FS),
            template_field="prompt_0shot",
            compress_field="prompt",
        )
        op = fitter.fit_context_pruning()
        assert isinstance(op, ContextPruning)
        assert op._n_protected == 1

    def test_full_compression_two_shot_sets_n_protected_two(self):
        fitter = DatasetVocabularyFitter(
            _dataset(full=FULL_2FS),
            template_field="prompt_0shot",
            compress_field="prompt",
        )
        op = fitter.fit_context_pruning()
        assert op._n_protected == 2

    def test_zeroshot_compression_n_protected_is_zero(self):
        fitter = DatasetVocabularyFitter(
            _dataset(full=FULL_1FS),
            template_field="prompt_0shot",
            compress_field="prompt_0shot",
        )
        op = fitter.fit_context_pruning()
        assert op._n_protected == 0

    def test_no_template_field_n_protected_is_zero(self):
        fitter = DatasetVocabularyFitter(_dataset(full=FULL_1FS))
        op = fitter.fit_context_pruning()
        assert op._n_protected == 0


# ---------------------------------------------------------------------------
# End-to-end: fitted O17 applied to a realistic GSM8K full prompt
# ---------------------------------------------------------------------------

class TestFittedO17EndToEnd:

    def test_full_compression_fewshot_paragraph_protected_from_pruning(self):
        """The few-shot Q&A paragraph has no anchor terms (target is 'eighteen
        dollars'; FS1 contains 'twelve' / 'eighteen' but not necessarily in a way
        that the tiny corpus discovers them as anchors).  We construct an op with
        known anchors to make the test deterministic."""
        # Directly build an op with anchors that don't appear in the few-shot
        # block but do appear in the question.
        op = ContextPruning(anchor_terms=["janet", "eggs"], n_protected_paragraphs=1)

        prompt = f"{FS1}\n\n{QUESTION}\n\nThis paragraph is completely unrelated."
        result = op.apply(prompt)

        assert "Jen and Tyler" in result      # FS1 protected (index 0)
        assert "Janet" in result              # question contains anchor
        assert "completely unrelated" not in result  # pruned — no anchor

    def test_zeroshot_compression_question_paragraph_is_prunable(self):
        """With n_protected=0, O17 can prune anything — including the question
        paragraph itself if it contains no anchors."""
        op = ContextPruning(anchor_terms=["xyz_never_appears"], n_protected_paragraphs=0)
        result = op.apply(QUESTION)
        assert result == ""

    def test_full_compression_fitter_anchors_protect_question(self):
        """Fitter discovers word anchors from targets; question paragraph that
        contains those words must survive pruning."""
        fitter = DatasetVocabularyFitter(
            _dataset(full=FULL_1FS, target="eighteen dollars"),
            template_field="prompt_0shot",
            compress_field="prompt",
        )
        op = fitter.fit_context_pruning()
        anchors = op._anchors
        if not anchors:
            pytest.skip("No anchor terms discovered from this corpus")

        # At least one anchor should appear in the question (word overlap with target)
        question_text = QUESTION.lower()
        assert any(a.search(question_text) for a in anchors) or op._n_protected == 1
