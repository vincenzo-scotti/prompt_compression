"""Structural transformations (O14–O22).

These operate on prompt-level sections, examples, and labels.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .base import Transformation, _ensure_nltk_resource

# ---------------------------------------------------------------------------
# O14 – DeduplicateInstructions
# ---------------------------------------------------------------------------

def _sentence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class DeduplicateInstructions(Transformation):
    """O14 – Remove repeated or near-duplicate instruction sentences."""

    id = "O14"
    name = "DeduplicateInstructions"
    category = "structural"

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self._threshold = similarity_threshold

    def apply(self, prompt: str) -> str:
        _ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(prompt)
        kept: list[str] = []
        for sentence in sentences:
            if not any(_sentence_similarity(sentence, k) >= self._threshold for k in kept):
                kept.append(sentence)
        return " ".join(kept)


# ---------------------------------------------------------------------------
# O15 – RemovePromptHeader
# ---------------------------------------------------------------------------

# Matches Markdown-style headers (###, ##, #) and plain "ALL CAPS TITLE:" lines.
_HEADER_LINE = re.compile(
    r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 _\-]{2,}:?\s*)$", re.M
)


class RemovePromptHeader(Transformation):
    """O15 – Remove section headers that add no task-relevant content.

    Parameters
    ----------
    patterns:
        List of compiled patterns to match against each stripped line using
        ``fullmatch``.  When *None* (default), only the built-in
        ``_HEADER_LINE`` pattern is used.  Pass an explicit list to extend or
        replace the default (e.g. from :meth:`DatasetVocabularyFitter.fit_prompt_header`).
    """

    id = "O15"
    name = "RemovePromptHeader"
    category = "structural"

    def __init__(self, patterns: list[re.Pattern[str]] | None = None) -> None:
        self._patterns = patterns if patterns is not None else [_HEADER_LINE]

    def apply(self, prompt: str) -> str:
        lines = prompt.splitlines()
        cleaned = [
            line for line in lines
            if not any(p.fullmatch(line.strip()) for p in self._patterns)
        ]
        return "\n".join(cleaned).strip()


# ---------------------------------------------------------------------------
# O16 – RemoveMetaText
# ---------------------------------------------------------------------------

_META_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^The\s+following\s+(instructions?|prompt|text|guidelines?)\s+(explain|describe|outline|detail)[^.]*\.\s*", re.I | re.M),
    re.compile(r"^(This|The)\s+(prompt|instruction|task)\s+(is\s+(designed|intended|meant)\s+to|will)[^.]*\.\s*", re.I | re.M),
    re.compile(r"^Below\s+(you\s+will\s+find|is|are)\s+(a\s+)?(set\s+of\s+)?(instructions?|guidelines?|rules?)[^.]*\.\s*", re.I | re.M),
    re.compile(r"^(Here\s+are|Here\s+is)\s+(some|the|a\s+set\s+of)\s+(instructions?|guidelines?|rules?)[^.]*[.:]\s*", re.I | re.M),
    re.compile(r"^The\s+purpose\s+of\s+this\s+(prompt|task|instruction)[^.]*\.\s*", re.I | re.M),
    re.compile(r"^This\s+section\s+(explains?|describes?|outlines?)[^.]*\.\s*", re.I | re.M),
]


class RemoveMetaText(Transformation):
    """O16 – Remove meta-sentences that describe the prompt rather than give instructions."""

    id = "O16"
    name = "RemoveMetaText"
    category = "structural"

    def apply(self, prompt: str) -> str:
        for pattern in _META_PATTERNS:
            prompt = pattern.sub("", prompt)
        return prompt.strip()


# ---------------------------------------------------------------------------
# O17 – ContextPruning
# ---------------------------------------------------------------------------

# Removes paragraphs (separated by blank lines) that don't contain any of a
# set of task-relevant anchor terms.  The caller can supply custom anchors;
# otherwise we fall back to a generic heuristic (keep non-empty paragraphs
# unless they look like boilerplate).

_BOILERPLATE_MARKERS = re.compile(
    r"\b(disclaimer|copyright|all rights reserved|terms of (use|service)|privacy policy|note:|footnote)\b",
    re.I,
)


class ContextPruning(Transformation):
    """O17 – Remove paragraphs that appear irrelevant to the task.

    Parameters
    ----------
    anchor_terms:
        If provided, paragraphs that contain none of these terms are removed.
        If None (default), paragraphs that match boilerplate markers are removed.
    """

    id = "O17"
    name = "ContextPruning"
    category = "structural"

    def __init__(self, anchor_terms: list[str] | None = None) -> None:
        self._anchors: list[re.Pattern[str]] | None = (
            [re.compile(re.escape(t), re.I) for t in anchor_terms]
            if anchor_terms is not None
            else None
        )

    def apply(self, prompt: str) -> str:
        paragraphs = re.split(r"\n{2,}", prompt)
        kept: list[str] = []
        for para in paragraphs:
            if self._anchors is not None:
                if any(p.search(para) for p in self._anchors):
                    kept.append(para)
            else:
                if not _BOILERPLATE_MARKERS.search(para):
                    kept.append(para)
        return "\n\n".join(kept).strip()


# ---------------------------------------------------------------------------
# O18 – SectionPruning
# ---------------------------------------------------------------------------

# Remove whole named sections (identified by a header keyword) from the prompt.

_DEFAULT_PRUNABLE_SECTIONS = [
    "additional notes",
    "notes",
    "disclaimer",
    "background",
    "context",
    "extra information",
    "supplementary",
    "appendix",
    "remarks",
    "caveats",
]


def _build_section_pattern(heading: str) -> re.Pattern[str]:
    """Match a section from its header to the next header or end-of-string."""
    escaped = re.escape(heading)
    return re.compile(
        # Match various header styles: ### Heading, Heading:, HEADING
        rf"(?im)^(?:#{1,6}\s+)?{escaped}\s*:?\s*\n(?:(?!(?:#{1,6}\s+|\Z)).+\n?)*",
        re.DOTALL,
    )


def _build_tag_pattern(tag: str) -> re.Pattern[str]:
    """Match an XML/HTML block ``<tag …>…</tag>`` including all content."""
    t = re.escape(tag)
    # [^>]* allows optional attributes; [\s\S]*? is lazy to avoid over-matching.
    return re.compile(rf"<{t}[^>]*>[\s\S]*?</{t}>", re.I)


class SectionPruning(Transformation):
    """O18 – Remove entire named sections from the prompt.

    Parameters
    ----------
    sections:
        List of section heading keywords to remove (case-insensitive).
        Defaults to common low-value sections.
    tag_names:
        List of XML/HTML tag names whose entire block (opening tag, content,
        and closing tag) should be removed — e.g. ``["instructions",
        "context"]`` removes ``<instructions>…</instructions>`` blocks.
        When *None* (default), no tag-based removal is applied.
    """

    id = "O18"
    name = "SectionPruning"
    category = "structural"

    def __init__(
        self,
        sections: list[str] | None = None,
        tag_names: list[str] | None = None,
    ) -> None:
        headings = sections if sections is not None else _DEFAULT_PRUNABLE_SECTIONS
        self._headings = list(headings)
        self._patterns = [_build_section_pattern(h) for h in headings]
        self._tag_names = list(tag_names) if tag_names else []
        self._tag_patterns = [_build_tag_pattern(t) for t in self._tag_names]

    def apply(self, prompt: str) -> str:
        for pattern in self._patterns:
            prompt = pattern.sub("", prompt)
        for pattern in self._tag_patterns:
            prompt = pattern.sub("", prompt)
        return prompt.strip()


# ---------------------------------------------------------------------------
# O19 – ExampleRemoval
# ---------------------------------------------------------------------------

# Examples are assumed to be separated by blank lines and preceded by a header
# like "Example N:" or "Input:" / "Output:" blocks.

_EXAMPLE_BLOCK = re.compile(
    r"(?:^(?:Example\s*\d*\s*:|Input\s*:|Q\s*:)[^\n]*\n(?:[^\n]+\n?)*?)(?=\n\n|\Z)",
    re.I | re.M,
)


class ExampleRemoval(Transformation):
    """O19 – Remove the last few-shot example from the prompt.

    Removes the *last* detected example block so the remaining context is kept.
    """

    id = "O19"
    name = "ExampleRemoval"
    category = "structural"

    def __init__(self, n_to_remove: int = 1) -> None:
        self._n = n_to_remove

    def apply(self, prompt: str) -> str:
        matches = list(_EXAMPLE_BLOCK.finditer(prompt))
        if not matches:
            return prompt
        to_remove = matches[-self._n :]
        # Remove from right to left to preserve offsets.
        for m in reversed(to_remove):
            # Safety: if the matched block covers the entire prompt there is
            # nothing left after removal — skip rather than returning "".
            if m.start() == 0 and m.end() >= len(prompt.rstrip()):
                continue
            prompt = prompt[: m.start()] + prompt[m.end() :]
        return prompt.strip()


# ---------------------------------------------------------------------------
# O20 – ExampleSubsetSelection
# ---------------------------------------------------------------------------

class ExampleSubsetSelection(Transformation):
    """O20 – Keep only the first *k* few-shot examples.

    Uses the same block-detection heuristic as ExampleRemoval.
    """

    id = "O20"
    name = "ExampleSubsetSelection"
    category = "structural"

    def __init__(self, k: int = 2) -> None:
        self._k = k

    def apply(self, prompt: str) -> str:
        matches = list(_EXAMPLE_BLOCK.finditer(prompt))
        if len(matches) <= self._k:
            return prompt
        # Find the start of the (k+1)-th example and cut from there.
        cut_start = matches[self._k].start()
        cut_end = matches[-1].end()
        return (prompt[:cut_start] + prompt[cut_end:]).strip()


# ---------------------------------------------------------------------------
# O21 – ExampleSimplification
# ---------------------------------------------------------------------------

# Collapse verbose "Input: X\nOutput: Y" example blocks into "X → Y".

_DEFAULT_INPUT_MARKERS: list[str] = ["Input", "Question", "Q"]
_DEFAULT_OUTPUT_MARKERS: list[str] = ["Output", "Answer", "Label", "Sentiment", "A"]


def _build_io_pattern(
    input_markers: list[str],
    output_markers: list[str],
) -> re.Pattern[str]:
    """Compile a regex that matches input/output example blocks.

    The lookahead is also built from *input_markers* so that the output
    capture group stops correctly at the next example's opening line.
    """
    inp_alt = "|".join(re.escape(m) for m in input_markers)
    out_alt = "|".join(re.escape(m) for m in output_markers)
    return re.compile(
        rf"(?:{inp_alt})\s*:\s*(?P<inp>.+?)\n"
        rf"(?:{out_alt})\s*:\s*(?P<out>.+?)(?=\n\n|\n(?:{inp_alt})\s*:|\Z)",
        re.I | re.S,
    )


class ExampleSimplification(Transformation):
    """O21 – Compress verbose input/output examples into compact arrow notation.

    Parameters
    ----------
    input_markers:
        Words used as the input-side field label (e.g. ``"Input"``,
        ``"Question"``).  Defaults to :data:`_DEFAULT_INPUT_MARKERS`.
    output_markers:
        Words used as the output-side field label (e.g. ``"Output"``,
        ``"Answer"``).  Defaults to :data:`_DEFAULT_OUTPUT_MARKERS`.
    """

    id = "O21"
    name = "ExampleSimplification"
    category = "structural"

    def __init__(
        self,
        input_markers: list[str] | None = None,
        output_markers: list[str] | None = None,
    ) -> None:
        self._input_markers = input_markers if input_markers is not None else list(_DEFAULT_INPUT_MARKERS)
        self._output_markers = output_markers if output_markers is not None else list(_DEFAULT_OUTPUT_MARKERS)
        self._pattern = _build_io_pattern(self._input_markers, self._output_markers)

    def apply(self, prompt: str) -> str:
        def _compress(m: re.Match[str]) -> str:
            inp = m.group("inp").strip().replace("\n", " ")
            out = m.group("out").strip().replace("\n", " ")
            return f"{inp} → {out}"

        return self._pattern.sub(_compress, prompt)


# ---------------------------------------------------------------------------
# O22 – LabelShortening
# ---------------------------------------------------------------------------

_DEFAULT_LABEL_MAP: dict[str, str] = {
    "positive": "pos",
    "negative": "neg",
    "neutral": "neu",
    "true": "T",
    "false": "F",
    "yes": "Y",
    "no": "N",
    "entailment": "ent",
    "contradiction": "con",
    "not entailment": "non-ent",
    "not contradiction": "non-con",
}


class LabelShortening(Transformation):
    """O22 – Replace verbose output labels with compact abbreviations.

    Parameters
    ----------
    label_map:
        Mapping of full label → short label.  Matching is case-insensitive;
        replacement preserves the original capitalisation style if the pattern
        is all-caps.
    """

    id = "O22"
    name = "LabelShortening"
    category = "structural"

    def __init__(self, label_map: dict[str, str] | None = None) -> None:
        self._map = label_map if label_map is not None else _DEFAULT_LABEL_MAP
        self._patterns = [
            (re.compile(rf"\b{re.escape(label)}\b", re.I), short)
            for label, short in self._map.items()
        ]

    def apply(self, prompt: str) -> str:
        for pattern, short in self._patterns:
            prompt = pattern.sub(short, prompt)
        return prompt