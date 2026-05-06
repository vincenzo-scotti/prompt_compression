"""Tests for ContextPruning boundary protection (n_protected_paragraphs)."""
import pytest
from prompt_piper.transformations.structural import ContextPruning


PARA_A = "Few-shot example 1: compute 2+2."
PARA_B = "Few-shot example 2: compute 3+3."
PARA_C = "Question: what is the square root of 144?"
PARA_IRRELEVANT = "This paragraph contains nothing on topic."

PROMPT_3 = f"{PARA_A}\n\n{PARA_B}\n\n{PARA_C}"
PROMPT_WITH_IRRELEVANT = f"{PARA_A}\n\n{PARA_B}\n\n{PARA_C}\n\n{PARA_IRRELEVANT}"


class TestContextPruningNProtected:

    def test_default_zero_protection_prunes_paragraphs_without_anchor(self):
        op = ContextPruning(anchor_terms=["question", "square"], n_protected_paragraphs=0)
        result = op.apply(PROMPT_WITH_IRRELEVANT)
        assert PARA_IRRELEVANT not in result
        assert PARA_C in result

    def test_protected_paragraphs_are_kept_even_without_anchor_terms(self):
        # PARA_A and PARA_B don't contain "question" or "square" but are protected.
        op = ContextPruning(anchor_terms=["question", "square"], n_protected_paragraphs=2)
        result = op.apply(PROMPT_WITH_IRRELEVANT)
        assert PARA_A in result
        assert PARA_B in result
        assert PARA_C in result
        assert PARA_IRRELEVANT not in result

    def test_unprotected_paragraphs_without_anchors_are_pruned(self):
        op = ContextPruning(anchor_terms=["question"], n_protected_paragraphs=2)
        result = op.apply(PROMPT_WITH_IRRELEVANT)
        assert PARA_IRRELEVANT not in result

    def test_n_protected_exceeds_paragraph_count_makes_o17_noop(self):
        # All paragraphs fall inside the protected window → nothing pruned.
        op = ContextPruning(anchor_terms=["xyz_never_appears"], n_protected_paragraphs=99)
        result = op.apply(PROMPT_3)
        for para in [PARA_A, PARA_B, PARA_C]:
            assert para in result

    def test_zero_protection_prunes_all_irrelevant(self):
        op = ContextPruning(anchor_terms=["xyz_never_appears"], n_protected_paragraphs=0)
        result = op.apply(PROMPT_3)
        assert result == ""

    def test_boilerplate_removal_without_anchors(self):
        op = ContextPruning(anchor_terms=None, n_protected_paragraphs=0)
        prompt = "Useful content here.\n\nDisclaimer: all rights reserved.\n\nMore useful content."
        result = op.apply(prompt)
        assert "Useful content here" in result
        assert "all rights reserved" not in result

    def test_boilerplate_in_protected_prefix_is_kept(self):
        # Even a boilerplate-looking paragraph is kept if it is protected.
        op = ContextPruning(anchor_terms=None, n_protected_paragraphs=1)
        prompt = "Disclaimer: all rights reserved.\n\nMore useful content."
        result = op.apply(prompt)
        assert "all rights reserved" in result

    def test_backward_compat_default_signature(self):
        # Calling with no n_protected_paragraphs must not raise and must behave
        # identically to the old single-argument form.
        op = ContextPruning(anchor_terms=["math"])
        result = op.apply("Math problem.\n\nIrrelevant text.")
        assert "Math problem" in result
        assert "Irrelevant text" not in result