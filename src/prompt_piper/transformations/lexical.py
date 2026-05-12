"""Lexical transformations (O1–O8).

These operate at the word/token level without changing sentence structure.
"""

from __future__ import annotations

import re

from .base import Transformation, _ensure_nltk_resource

# ---------------------------------------------------------------------------
# O1 – RemoveExtraWhitespace
# ---------------------------------------------------------------------------

class RemoveExtraWhitespace(Transformation):
    """O1 – Remove redundant whitespace and collapse multiple blank lines.

    Horizontal whitespace (spaces/tabs) is collapsed to a single space on each
    line.  Multiple consecutive blank lines are collapsed to a single blank
    line so that paragraph structure is preserved.
    """

    id = "O1"
    name = "RemoveExtraWhitespace"
    category = "lexical"
    scope = frozenset({"instruction", "shots", "data"})

    def apply(self, prompt: str) -> str:
        # Collapse runs of spaces/tabs to a single space and strip each line.
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in prompt.splitlines()]
        # Collapse multiple consecutive blank lines to one; keep single blank lines.
        result: list[str] = []
        prev_blank = False
        for line in lines:
            if line:
                result.append(line)
                prev_blank = False
            elif not prev_blank:
                result.append(line)
                prev_blank = True
            # else: skip second-or-later consecutive blank line
        return "\n".join(result).strip()


# ---------------------------------------------------------------------------
# O2 – TrimSpaces
# ---------------------------------------------------------------------------

class TrimSpaces(Transformation):
    """O2 – Remove leading/trailing spaces from every line and the whole prompt."""

    id = "O2"
    name = "TrimSpaces"
    category = "lexical"
    scope = frozenset({"instruction", "shots", "data"})

    def apply(self, prompt: str) -> str:
        lines = [line.strip() for line in prompt.splitlines()]
        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# O3 – RemoveStopwords
# ---------------------------------------------------------------------------

def _load_stopwords() -> frozenset[str]:
    """Return the NLTK English stopword list, downloading it if needed.

    Falls back to a small built-in set when NLTK data cannot be obtained
    (e.g. in air-gapped environments).
    """
    try:
        _ensure_nltk_resource("corpora/stopwords", "stopwords")
        from nltk.corpus import stopwords
        return frozenset(stopwords.words("english"))
    except Exception:
        # Minimal fallback so the transformation still works without NLTK data.
        return frozenset(
            {
                "a", "an", "the",
                "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had",
                "do", "does", "did",
                "will", "would", "could", "should", "may", "might", "shall",
                "of", "in", "on", "at", "to", "for", "with", "by", "from",
                "that", "this", "these", "those",
                "it", "its",
                "i", "we", "you", "he", "she", "they",
                "me", "us", "him", "her", "them",
                "my", "our", "your", "his", "their",
                "which", "who", "whom",
                "also", "just", "very", "quite", "rather",
            }
        )


_STOPWORDS: frozenset[str] = _load_stopwords()

class RemoveStopwords(Transformation):
    """O3 – Remove non-essential function words."""

    id = "O3"
    name = "RemoveStopwords"
    category = "lexical"
    scope = frozenset({"instruction"})

    def __init__(self, stopwords: frozenset[str] | None = None) -> None:
        self._stopwords = stopwords if stopwords is not None else _STOPWORDS

    def apply(self, prompt: str) -> str:
        tokens = prompt.split(" ")
        kept = [t for t in tokens if t.lower().strip(".,;:!?\"'") not in self._stopwords]
        return " ".join(kept)


# ---------------------------------------------------------------------------
# O4 – ReplaceWithShortSynonym
# ---------------------------------------------------------------------------

_SHORT_SYNONYMS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bin order to\b", re.I), "to"),
    (re.compile(r"\bdue to the fact that\b", re.I), "because"),
    (re.compile(r"\bat this point in time\b", re.I), "now"),
    (re.compile(r"\bin the event that\b", re.I), "if"),
    (re.compile(r"\bprior to\b", re.I), "before"),
    (re.compile(r"\bsubsequent to\b", re.I), "after"),
    (re.compile(r"\bwith regard to\b", re.I), "about"),
    (re.compile(r"\bwith respect to\b", re.I), "about"),
    (re.compile(r"\bin addition to\b", re.I), "besides"),
    (re.compile(r"\ba large number of\b", re.I), "many"),
    (re.compile(r"\ba small number of\b", re.I), "few"),
    (re.compile(r"\bfor the purpose of\b", re.I), "to"),
    (re.compile(r"\bin spite of the fact that\b", re.I), "although"),
    (re.compile(r"\bhas the ability to\b", re.I), "can"),
    (re.compile(r"\bin the case of\b", re.I), "for"),
    (re.compile(r"\bmake use of\b", re.I), "use"),
    (re.compile(r"\bprovide assistance\b", re.I), "help"),
    (re.compile(r"\butilize\b", re.I), "use"),
    (re.compile(r"\bcommence\b", re.I), "start"),
    (re.compile(r"\bterminate\b", re.I), "end"),
    (re.compile(r"\bpurchase\b", re.I), "buy"),
    (re.compile(r"\bassist\b", re.I), "help"),
    (re.compile(r"\brequire\b", re.I), "need"),
    (re.compile(r"\battempt\b", re.I), "try"),
    (re.compile(r"\bdemonstrate\b", re.I), "show"),
]


class ReplaceWithShortSynonym(Transformation):
    """O4 – Replace verbose phrases with shorter equivalents."""

    id = "O4"
    name = "ReplaceWithShortSynonym"
    category = "lexical"
    scope = frozenset({"instruction"})

    def __init__(self, synonyms: list[tuple[re.Pattern[str], str]] | None = None) -> None:
        self._synonyms = synonyms if synonyms is not None else _SHORT_SYNONYMS

    def apply(self, prompt: str) -> str:
        for pattern, replacement in self._synonyms:
            prompt = pattern.sub(replacement, prompt)
        return prompt


# ---------------------------------------------------------------------------
# O5 – AbbreviationSubstitution
# ---------------------------------------------------------------------------

_ABBREVIATIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bclassification\b", re.I), "cls"),
    (re.compile(r"\bnatural language processing\b", re.I), "NLP"),
    (re.compile(r"\bnatural language\b", re.I), "NL"),
    (re.compile(r"\binformation\b", re.I), "info"),
    (re.compile(r"\bdefinition\b", re.I), "def"),
    (re.compile(r"\bdocument\b", re.I), "doc"),
    (re.compile(r"\bdocuments\b", re.I), "docs"),
    (re.compile(r"\bexample\b", re.I), "ex"),
    (re.compile(r"\bexamples\b", re.I), "exs"),
    (re.compile(r"\bquestion\b", re.I), "Q"),
    (re.compile(r"\banswer\b", re.I), "A"),
    (re.compile(r"\bmaximum\b", re.I), "max"),
    (re.compile(r"\bminimum\b", re.I), "min"),
    (re.compile(r"\bnumber\b", re.I), "num"),
    (re.compile(r"\bapproximately\b", re.I), "approx"),
    (re.compile(r"\brepresentation\b", re.I), "repr"),
    (re.compile(r"\bconfiguration\b", re.I), "config"),
    (re.compile(r"\bparameter\b", re.I), "param"),
    (re.compile(r"\bparameters\b", re.I), "params"),
    (re.compile(r"\bparagraph\b", re.I), "para"),
    (re.compile(r"\bsentiment\b", re.I), "sent"),
]


class AbbreviationSubstitution(Transformation):
    """O5 – Replace common words with standard abbreviations."""

    id = "O5"
    name = "AbbreviationSubstitution"
    category = "lexical"
    scope = frozenset({"instruction", "shots"})

    def __init__(self, abbreviations: list[tuple[re.Pattern[str], str]] | None = None) -> None:
        self._abbreviations = abbreviations if abbreviations is not None else _ABBREVIATIONS

    def apply(self, prompt: str) -> str:
        for pattern, abbrev in self._abbreviations:
            prompt = pattern.sub(abbrev, prompt)
        return prompt


# ---------------------------------------------------------------------------
# O6 – SymbolSubstitution
# ---------------------------------------------------------------------------

_SYMBOLS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\band\b", re.I), "&"),
    (re.compile(r"\bor\b", re.I), "|"),
    (re.compile(r"\bgreater than\b", re.I), ">"),
    (re.compile(r"\bless than\b", re.I), "<"),
    (re.compile(r"\bequal to\b", re.I), "="),
    (re.compile(r"\bplus\b", re.I), "+"),
    (re.compile(r"\bminus\b", re.I), "-"),
    (re.compile(r"\btherefore\b", re.I), "∴"),
    (re.compile(r"\bbecause\b", re.I), "∵"),
    (re.compile(r"\bapproximately\b", re.I), "≈"),
    (re.compile(r"\bversus\b", re.I), "vs"),
    (re.compile(r"\bwithout\b", re.I), "w/o"),
    (re.compile(r"\bwith\b", re.I), "w/"),
    (re.compile(r"\bnot equal\b", re.I), "≠"),
]


class SymbolSubstitution(Transformation):
    """O6 – Replace words with symbols."""

    id = "O6"
    name = "SymbolSubstitution"
    category = "lexical"
    scope = frozenset({"instruction"})

    def __init__(self, symbols: list[tuple[re.Pattern[str], str]] | None = None) -> None:
        self._symbols = symbols if symbols is not None else _SYMBOLS

    def apply(self, prompt: str) -> str:
        for pattern, symbol in self._symbols:
            prompt = pattern.sub(symbol, prompt)
        return prompt


# ---------------------------------------------------------------------------
# O7 – ContractionSubstitution
# ---------------------------------------------------------------------------

_CONTRACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdo not\b", re.I), "don't"),
    (re.compile(r"\bdoes not\b", re.I), "doesn't"),
    (re.compile(r"\bdid not\b", re.I), "didn't"),
    (re.compile(r"\bwill not\b", re.I), "won't"),
    (re.compile(r"\bwould not\b", re.I), "wouldn't"),
    (re.compile(r"\bcould not\b", re.I), "couldn't"),
    (re.compile(r"\bshould not\b", re.I), "shouldn't"),
    (re.compile(r"\bmust not\b", re.I), "mustn't"),
    (re.compile(r"\bcannot\b", re.I), "can't"),
    (re.compile(r"\bcan not\b", re.I), "can't"),
    (re.compile(r"\bis not\b", re.I), "isn't"),
    (re.compile(r"\bare not\b", re.I), "aren't"),
    (re.compile(r"\bwas not\b", re.I), "wasn't"),
    (re.compile(r"\bwere not\b", re.I), "weren't"),
    (re.compile(r"\bhave not\b", re.I), "haven't"),
    (re.compile(r"\bhas not\b", re.I), "hasn't"),
    (re.compile(r"\bhad not\b", re.I), "hadn't"),
    (re.compile(r"\bI am\b", re.I), "I'm"),
    (re.compile(r"\bI have\b", re.I), "I've"),
    (re.compile(r"\bI will\b", re.I), "I'll"),
    (re.compile(r"\bI would\b", re.I), "I'd"),
    (re.compile(r"\byou are\b", re.I), "you're"),
    (re.compile(r"\byou have\b", re.I), "you've"),
    (re.compile(r"\byou will\b", re.I), "you'll"),
    (re.compile(r"\bwe are\b", re.I), "we're"),
    (re.compile(r"\bwe have\b", re.I), "we've"),
    (re.compile(r"\bthey are\b", re.I), "they're"),
    (re.compile(r"\bthey have\b", re.I), "they've"),
    (re.compile(r"\bit is\b", re.I), "it's"),
    (re.compile(r"\bthat is\b", re.I), "that's"),
    (re.compile(r"\bthere is\b", re.I), "there's"),
]


class ContractionSubstitution(Transformation):
    """O7 – Replace expanded forms with contractions."""

    id = "O7"
    name = "ContractionSubstitution"
    category = "lexical"
    scope = frozenset({"instruction"})

    def __init__(self, contractions: list[tuple[re.Pattern[str], str]] | None = None) -> None:
        self._contractions = contractions if contractions is not None else _CONTRACTIONS

    def apply(self, prompt: str) -> str:
        for pattern, contraction in self._contractions:
            prompt = pattern.sub(contraction, prompt)
        return prompt


# ---------------------------------------------------------------------------
# O8 – RemovePolitenessMarkers
# ---------------------------------------------------------------------------

_POLITENESS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bplease\s+(kindly\s+)?", re.I),
    re.compile(r"\bkindly\s+", re.I),
    re.compile(r"\bif\s+you\s+(could|would|can|will|don't mind)\s+", re.I),
    re.compile(r"\bif\s+possible\b[,.]?\s*", re.I),
    re.compile(r"\bwhenever\s+you\s+(are\s+)?ready\b[,.]?\s*", re.I),
    re.compile(r"\bthank\s+you\b[.!]?\s*", re.I),
    re.compile(r"\bthanks\b[.!]?\s*", re.I),
    re.compile(r"\bi\s+would\s+(really\s+)?appreciate\s+(it\s+)?if\b", re.I),
    re.compile(r"\bfeel\s+free\s+to\s+", re.I),
    re.compile(r"\bcould\s+you\s+(please\s+)?", re.I),
    re.compile(r"\bwould\s+you\s+(please\s+|mind\s+)?", re.I),
]


class RemovePolitenessMarkers(Transformation):
    """O8 – Remove polite phrases that don't affect task semantics."""

    id = "O8"
    name = "RemovePolitenessMarkers"
    category = "lexical"
    scope = frozenset({"instruction"})

    def apply(self, prompt: str) -> str:
        for pattern in _POLITENESS_PATTERNS:
            prompt = pattern.sub("", prompt)
        # Capitalise first letter if it became lower-case after removal.
        lines = []
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped:
                stripped = stripped[0].upper() + stripped[1:]
            lines.append(stripped)
        return "\n".join(lines).strip()