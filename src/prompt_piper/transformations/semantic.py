"""Semantic transformations (O23–O28).

True semantic compression (summarisation, distillation) requires an LLM backend.
The rule-based implementations here are conservative approximations that apply
deterministic heuristics as stand-ins until an LLM-backed version is wired up.

Each class exposes a ``backend`` attribute (default ``None``).  When set to a
callable ``backend(text: str, instruction: str) -> str``, the class delegates to
it instead of the heuristic fallback.
"""

from __future__ import annotations

import re
from typing import Callable

from .base import Transformation

Backend = Callable[[str, str], str]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    return re.split(r"\n{2,}", text)


def _first_sentence(text: str) -> str:
    """Return the first sentence of *text*, or the whole text if none found."""
    m = re.match(r"[^.!?]+[.!?]", text)
    return m.group(0).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# O23 – ContextCompression
# ---------------------------------------------------------------------------

class ContextCompression(Transformation):
    """O23 – Compress contextual paragraphs to their first sentence (heuristic).

    With a backend: summarises each context paragraph to a short summary.

    Parameters
    ----------
    backend:
        Optional callable ``(text, instruction) -> str`` for LLM summarisation.
    context_marker:
        Regex pattern that identifies context paragraphs (e.g. "Context:").
        If None, all paragraphs longer than ``min_sentences`` are compressed.
    min_sentences:
        Minimum number of sentences in a paragraph before compression is applied.
    """

    id = "O23"
    name = "ContextCompression"
    category = "semantic"

    def __init__(
        self,
        backend: Backend | None = None,
        context_marker: str | None = None,
        min_sentences: int = 3,
    ) -> None:
        self.backend = backend
        self._marker = re.compile(context_marker, re.I) if context_marker else None
        self._min_sentences = min_sentences

    def _compress_paragraph(self, para: str) -> str:
        if self.backend is not None:
            return self.backend(para, "Summarise this passage in one concise sentence.")
        return _first_sentence(para)

    def apply(self, prompt: str) -> str:
        paragraphs = _split_paragraphs(prompt)
        result: list[str] = []
        for para in paragraphs:
            sentences = re.findall(r"[^.!?]+[.!?]", para)
            should_compress = (
                (self._marker and self._marker.search(para))
                or (not self._marker and len(sentences) >= self._min_sentences)
            )
            result.append(self._compress_paragraph(para) if should_compress else para)
        return "\n\n".join(result)


# ---------------------------------------------------------------------------
# O24 – InstructionSummarization
# ---------------------------------------------------------------------------

class InstructionSummarization(Transformation):
    """O24 – Compress a verbose task description to a short instruction.

    Heuristic: keep the first sentence of the instruction block.

    Parameters
    ----------
    backend:
        Optional callable for LLM-based summarisation.
    instruction_marker:
        Regex identifying the instruction block.  If None the whole prompt is
        treated as the instruction.
    """

    id = "O24"
    name = "InstructionSummarization"
    category = "semantic"

    def __init__(
        self,
        backend: Backend | None = None,
        instruction_marker: str | None = None,
    ) -> None:
        self.backend = backend
        self._marker = re.compile(instruction_marker, re.I) if instruction_marker else None

    def _summarise(self, text: str) -> str:
        if self.backend is not None:
            return self.backend(text, "Summarise this instruction in one sentence.")
        return _first_sentence(text)

    def apply(self, prompt: str) -> str:
        if self._marker is None:
            return self._summarise(prompt)

        paragraphs = _split_paragraphs(prompt)
        result: list[str] = []
        for para in paragraphs:
            result.append(self._summarise(para) if self._marker.search(para) else para)
        return "\n\n".join(result)


# ---------------------------------------------------------------------------
# O25 – InstructionAbstraction
# ---------------------------------------------------------------------------

# Replace lists of specific low-level rules with a single abstract guideline.
# The heuristic version collapses bullet-point rule sets into one sentence.

_RULE_BLOCK = re.compile(
    r"(?P<header>[^\n]+:\n)"
    r"(?P<items>(?:[ \t]*[-*•\d.]+[ \t]+[^\n]+\n?){2,})",
    re.I,
)

_DEFAULT_ABSTRACTION = "Follow the guidelines above."


class InstructionAbstraction(Transformation):
    """O25 – Replace detailed rule lists with a single abstract instruction.

    Parameters
    ----------
    backend:
        Optional callable for LLM-based abstraction.
    abstraction:
        Fallback string used when no backend is provided.
    """

    id = "O25"
    name = "InstructionAbstraction"
    category = "semantic"

    def __init__(
        self,
        backend: Backend | None = None,
        abstraction: str = _DEFAULT_ABSTRACTION,
    ) -> None:
        self.backend = backend
        self._abstraction = abstraction

    def _abstract(self, header: str, items: str) -> str:
        if self.backend is not None:
            return self.backend(header + items, "Replace these rules with one abstract guideline.")
        return self._abstraction

    def apply(self, prompt: str) -> str:
        def _replace(m: re.Match[str]) -> str:
            return self._abstract(m.group("header"), m.group("items"))

        return _RULE_BLOCK.sub(_replace, prompt)


# ---------------------------------------------------------------------------
# O26 – SchemaEncoding
# ---------------------------------------------------------------------------

# Replace verbose output-format instructions with a compact schema notation.
_OUTPUT_FORMAT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"Return\s+(?:a\s+)?JSON\s+(?:object\s+)?with\s+(?:the\s+)?fields?\s+([^.]+)\.",
            re.I,
        ),
        lambda m: "JSON{" + re.sub(r"\s+and\s+|\s*,\s*", ",", m.group(1).strip()) + "}",
    ),
    (
        re.compile(
            r"Output\s+(?:a\s+)?JSON\s+(?:object\s+)?(?:with|containing)\s+(?:the\s+)?(?:following\s+)?fields?\s*:?\s*([^.]+)\.",
            re.I,
        ),
        lambda m: "JSON{" + re.sub(r"\s+and\s+|\s*,\s*", ",", m.group(1).strip()) + "}",
    ),
    (
        re.compile(
            r"(?:Your\s+)?(?:output|response|answer)\s+(?:must|should)\s+be\s+(?:in\s+)?(?:valid\s+)?JSON\s+format\.",
            re.I,
        ),
        "Output: JSON.",
    ),
    (
        re.compile(
            r"Respond\s+(?:only\s+)?(?:in|with)\s+(?:valid\s+)?JSON(?:\s+format)?\.",
            re.I,
        ),
        "Output: JSON.",
    ),
]


class SchemaEncoding(Transformation):
    """O26 – Replace natural-language output-format descriptions with compact schema notation."""

    id = "O26"
    name = "SchemaEncoding"
    category = "semantic"

    def apply(self, prompt: str) -> str:
        for pattern, replacement in _OUTPUT_FORMAT_PATTERNS:
            if callable(replacement):
                prompt = pattern.sub(replacement, prompt)
            else:
                prompt = pattern.sub(replacement, prompt)
        return prompt


# ---------------------------------------------------------------------------
# O27 – ExampleDistillation
# ---------------------------------------------------------------------------

_EXAMPLE_BLOCK_PATTERN = re.compile(
    r"(?:^(?:Example\s*\d*\s*:|Input\s*:|Q\s*:)[^\n]*\n(?:[^\n]+\n?)*?)(?=\n\n|\Z)",
    re.I | re.M,
)

_DEFAULT_RULE_TEMPLATE = "Rule: classify according to the patterns shown above."


class ExampleDistillation(Transformation):
    """O27 – Replace multiple few-shot examples with a generalised rule.

    Parameters
    ----------
    backend:
        Optional callable for LLM-based distillation.
    min_examples:
        Only distil when at least this many example blocks are present.
    rule_template:
        Fallback rule string used when no backend is provided.
    """

    id = "O27"
    name = "ExampleDistillation"
    category = "semantic"

    def __init__(
        self,
        backend: Backend | None = None,
        min_examples: int = 2,
        rule_template: str = _DEFAULT_RULE_TEMPLATE,
    ) -> None:
        self.backend = backend
        self._min_examples = min_examples
        self._rule_template = rule_template

    def _distil(self, examples_text: str) -> str:
        if self.backend is not None:
            return self.backend(
                examples_text,
                "Distil these examples into a single generalised rule.",
            )
        return self._rule_template

    def apply(self, prompt: str) -> str:
        matches = list(_EXAMPLE_BLOCK_PATTERN.finditer(prompt))
        if len(matches) < self._min_examples:
            return prompt

        first_start = matches[0].start()
        last_end = matches[-1].end()
        examples_text = prompt[first_start:last_end]

        rule = self._distil(examples_text)
        return (prompt[:first_start] + rule + prompt[last_end:]).strip()


# ---------------------------------------------------------------------------
# O28 – ChunkCompression
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_SEPARATOR = re.compile(r"\n{2,}")


class ChunkCompression(Transformation):
    """O28 – Compress independent prompt chunks (paragraphs) individually.

    Each chunk is passed through a compression sub-pipeline or backend.

    Parameters
    ----------
    backend:
        Optional callable ``(chunk, instruction) -> str`` applied to each chunk.
    sub_pipeline:
        List of :class:`Transformation` instances applied in sequence to each
        chunk.  Ignored when *backend* is set.
    separator:
        Regex used to split the prompt into chunks.
    """

    id = "O28"
    name = "ChunkCompression"
    category = "semantic"

    def __init__(
        self,
        backend: Backend | None = None,
        sub_pipeline: list[Transformation] | None = None,
        separator: re.Pattern[str] = _DEFAULT_CHUNK_SEPARATOR,
    ) -> None:
        self.backend = backend
        self._sub_pipeline = sub_pipeline or []
        self._separator = separator

    def _compress_chunk(self, chunk: str) -> str:
        if self.backend is not None:
            return self.backend(chunk, "Compress this text as much as possible.")
        result = chunk
        for transform in self._sub_pipeline:
            result = transform(result)
        return result

    def apply(self, prompt: str) -> str:
        chunks = self._separator.split(prompt)
        compressed = [self._compress_chunk(c) for c in chunks if c.strip()]
        return "\n\n".join(compressed)