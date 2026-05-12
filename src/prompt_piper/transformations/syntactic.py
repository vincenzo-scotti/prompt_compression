"""Syntactic transformations (O9–O13).

These restructure sentence-level syntax without changing vocabulary significantly.
"""

from __future__ import annotations

import re

from .base import Transformation, _ensure_nltk_resource

# ---------------------------------------------------------------------------
# O9 – SentenceMerge
# ---------------------------------------------------------------------------

_SHORT_SENTENCE_THRESHOLD = 10  # words


class SentenceMerge(Transformation):
    """O9 – Merge adjacent short sentences into a single sentence.

    Sentence boundaries are detected with NLTK's Punkt tokenizer, which
    correctly handles abbreviations (e.g. "Dr.", "U.S.") and decimal numbers
    that would otherwise confuse a simple period-based regex split.
    """

    id = "O9"
    name = "SentenceMerge"
    category = "syntactic"
    scope = frozenset({"instruction"})

    def __init__(self, max_words: int = _SHORT_SENTENCE_THRESHOLD) -> None:
        self._max_words = max_words

    def apply(self, prompt: str) -> str:
        _ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(prompt)
        merged: list[str] = []
        i = 0
        while i < len(sentences):
            current = sentences[i].rstrip(". ")
            if (
                i + 1 < len(sentences)
                and len(current.split()) <= self._max_words
                and len(sentences[i + 1].split()) <= self._max_words
            ):
                next_sent = sentences[i + 1].rstrip(". ")
                # Lower-case the start of the continuation.
                if next_sent:
                    next_sent = next_sent[0].lower() + next_sent[1:]
                merged.append(f"{current}; {next_sent}.")
                i += 2
            else:
                merged.append(sentences[i])
                i += 1
        return " ".join(merged)


# ---------------------------------------------------------------------------
# O10 – ImperativeRewrite
# ---------------------------------------------------------------------------

# Patterns of the form "Your task is to X" → "X" (imperative).
_IMPERATIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^Your\s+task\s+is\s+to\s+", re.I | re.M), ""),
    (re.compile(r"^Your\s+goal\s+is\s+to\s+", re.I | re.M), ""),
    (re.compile(r"^You\s+(need|have)\s+to\s+", re.I | re.M), ""),
    (re.compile(r"^You\s+are\s+(asked|required|expected|supposed)\s+to\s+", re.I | re.M), ""),
    (re.compile(r"^The\s+task\s+is\s+to\s+", re.I | re.M), ""),
    (re.compile(r"^The\s+goal\s+is\s+to\s+", re.I | re.M), ""),
    (re.compile(r"^I\s+(need|want)\s+you\s+to\s+", re.I | re.M), ""),
    (re.compile(r"^Please\s+", re.I | re.M), ""),
]


def _capitalise_first(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


class ImperativeRewrite(Transformation):
    """O10 – Convert descriptive task framing to imperative form.

    Parameters
    ----------
    patterns:
        List of ``(compiled_pattern, replacement)`` pairs applied in order.
        Defaults to the built-in :data:`_IMPERATIVE_PATTERNS`.  Pass a
        custom list (e.g. from
        :meth:`~prompt_piper.transformations.fitting.DatasetVocabularyFitter.fit_imperative_rewrite`)
        to add or replace dataset-specific framing patterns.
    """

    id = "O10"
    name = "ImperativeRewrite"
    category = "syntactic"
    scope = frozenset({"instruction"})

    def __init__(
        self,
        patterns: list[tuple[re.Pattern[str], str]] | None = None,
    ) -> None:
        self._patterns = patterns if patterns is not None else _IMPERATIVE_PATTERNS

    def apply(self, prompt: str) -> str:
        for pattern, replacement in self._patterns:
            prompt = pattern.sub(replacement, prompt)
        # Re-capitalise lines that lost their leading capital.
        lines = [_capitalise_first(line) if line else line for line in prompt.splitlines()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# O11 – EnumerationCompression
# ---------------------------------------------------------------------------

# Detect lines that look like standalone rules/sentences within a list block
# (e.g. starting with "- " or a numbered list) and collapse them with ";".
_LIST_ITEM = re.compile(r"^[\s]*[-*•]\s+(.+)$")
_NUMBERED_ITEM = re.compile(r"^[\s]*\d+[.)]\s+(.+)$")


class EnumerationCompression(Transformation):
    """O11 – Convert bulleted/numbered lists into compact semicolon-separated form."""

    id = "O11"
    name = "EnumerationCompression"
    category = "syntactic"
    scope = frozenset({"instruction"})

    def apply(self, prompt: str) -> str:
        lines = prompt.splitlines()
        result: list[str] = []
        list_items: list[str] = []

        def _flush() -> None:
            if list_items:
                result.append("; ".join(list_items) + ".")
                list_items.clear()

        for line in lines:
            m = _LIST_ITEM.match(line) or _NUMBERED_ITEM.match(line)
            if m:
                # Strip trailing punctuation so joined items read cleanly.
                list_items.append(m.group(1).rstrip(".;, "))
            else:
                _flush()
                result.append(line)
        _flush()
        return "\n".join(result)


# ---------------------------------------------------------------------------
# O12 – ClauseDeletion
# ---------------------------------------------------------------------------

# Remove common introductory/context-setting clauses that precede the
# actual instruction.
_CLAUSE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^Given\s+the\s+(context|information|passage|text)\s+(below|above|provided)[,.]?\s*", re.I | re.M),
    re.compile(r"^Based\s+on\s+the\s+(context|information|passage|text)\s+(below|above|provided)[,.]?\s*", re.I | re.M),
    re.compile(r"^Using\s+the\s+(context|information|passage|text)\s+(below|above|provided)[,.]?\s*", re.I | re.M),
    re.compile(r"^Refer(ring)?\s+to\s+the\s+(context|information|passage|text)\s+(below|above|provided)[,.]?\s*", re.I | re.M),
    re.compile(r"^Read\s+the\s+(following|passage|text|context)[,.]?\s*", re.I | re.M),
    re.compile(r"^Consider\s+the\s+(following|passage|text|context)[,.]?\s*", re.I | re.M),
    re.compile(r"^As\s+an?\s+\w+\s+(model|assistant|AI|system)[,.]?\s*", re.I | re.M),
    re.compile(r"^In\s+this\s+task[,.]?\s*", re.I | re.M),
    re.compile(r"^For\s+this\s+(task|exercise|problem)[,.]?\s*", re.I | re.M),
]


class ClauseDeletion(Transformation):
    """O12 – Remove non-essential introductory subordinate clauses."""

    id = "O12"
    name = "ClauseDeletion"
    category = "syntactic"
    scope = frozenset({"instruction"})

    def apply(self, prompt: str) -> str:
        for pattern in _CLAUSE_PATTERNS:
            prompt = pattern.sub("", prompt)
        lines = [_capitalise_first(line) if line.strip() else line for line in prompt.splitlines()]
        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# O13 – SentenceSimplification
# ---------------------------------------------------------------------------

# Replace complex constructions with simpler equivalents at the sentence level.
_SIMPLIFICATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "Provide a response that is X" → "Respond X"
    (re.compile(r"\bProvide\s+a\s+response\s+that\s+is\s+(\w+)\b", re.I), r"Respond \1"),
    # "Give a X answer" → "Answer X"
    (re.compile(r"\bGive\s+a\s+(\w+)\s+answer\b", re.I), r"Answer \1"),
    # "Write a X response" → "Respond X"
    (re.compile(r"\bWrite\s+a\s+(\w+)\s+response\b", re.I), r"Respond \1"),
    # "Make sure to X" → "X"
    (re.compile(r"\bMake\s+sure\s+to\s+", re.I), ""),
    # "Make sure that you X" → "X"
    (re.compile(r"\bMake\s+sure\s+that\s+you\s+", re.I), ""),
    # "Be sure to X" → "X"
    (re.compile(r"\bBe\s+sure\s+to\s+", re.I), ""),
    # "You should X" → "X"
    (re.compile(r"^You\s+should\s+", re.I | re.M), ""),
    # "It is important to X" → "X"
    (re.compile(r"\bIt\s+is\s+important\s+to\s+", re.I), ""),
    # "It is necessary to X" → "X"
    (re.compile(r"\bIt\s+is\s+necessary\s+to\s+", re.I), ""),
    # "Note that X" → "X" (strips meta-note wrapper)
    (re.compile(r"^Note\s+that\s+", re.I | re.M), ""),
]


class SentenceSimplification(Transformation):
    """O13 – Replace complex syntax patterns with simpler forms."""

    id = "O13"
    name = "SentenceSimplification"
    category = "syntactic"
    scope = frozenset({"instruction"})

    def apply(self, prompt: str) -> str:
        for pattern, replacement in _SIMPLIFICATION_PATTERNS:
            prompt = pattern.sub(replacement, prompt)
        lines = [_capitalise_first(line) if line.strip() else line for line in prompt.splitlines()]
        return "\n".join(lines).strip()