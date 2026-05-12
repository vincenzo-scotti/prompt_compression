"""Section-aware prompt splitting.

`PromptSplitter` decomposes a prompt into four named sections:

  * **instruction** — shared task description (static across all samples)
  * **shots** — few-shot Q&A demonstrations (static, may be empty)
  * **input_data** — per-sample test passage / question (dynamic)
  * **output_cue** — trailing suffix such as "Answer:" (not compressed)

`PromptSections` is a lightweight dataclass that holds the four parts and
provides an `assemble()` method that rebuilds the full prompt.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class PromptSections:
    """Four named sections of a single prompt sample."""

    instruction: str = ""
    shots: str = ""
    input_data: str = ""
    output_cue: str = ""

    def assemble(
        self,
        instruction: str | None = None,
        shots: str | None = None,
        input_data: str | None = None,
    ) -> str:
        """Rebuild the prompt from (optionally overridden) section texts.

        Pass compressed section text to replace the original.  ``None`` means
        keep the stored value.  ``output_cue`` is always appended as-is.
        """
        parts = [
            instruction if instruction is not None else self.instruction,
            shots if shots is not None else self.shots,
            input_data if input_data is not None else self.input_data,
            self.output_cue,
        ]
        return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Example-block pattern (reused from structural.py)
# ---------------------------------------------------------------------------

_EXAMPLE_BLOCK = re.compile(
    r"(?:^(?:Example\s*\d*\s*:|Input\s*:|Q\s*:)[^\n]*\n(?:[^\n]+\n?)*?)(?=\n\n|\Z)",
    re.I | re.M,
)


# ---------------------------------------------------------------------------
# PromptSplitter
# ---------------------------------------------------------------------------

class PromptSplitter:
    """Detect section boundaries and split prompts.

    Parameters
    ----------
    samples:
        List of dataset sample dicts.  Must contain the field named by
        *compress_field*.  May optionally contain ``"_shots"`` (plain-text
        few-shot block) and ``"prompt_0shot"``.
    compress_field:
        The field that contains the text to be compressed.  Typically
        ``"prompt"`` or ``"prompt_0shot"``.
    shots_field:
        Name of the field carrying the plain-text shots block (default
        ``"_shots"``).  When present, that field is used directly as the
        ``shots`` section; otherwise the splitter tries to detect embedded
        example blocks in the prefix.
    """

    def __init__(
        self,
        samples: list[dict],
        compress_field: str = "prompt_0shot",
        shots_field: str = "_shots",
    ) -> None:
        self._samples = samples
        self._compress_field = compress_field
        self._shots_field = shots_field

        # Derive static boundaries once from the corpus.
        self._instruction_prefix, self._output_cue = self._detect_boundaries()

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------

    def _texts(self) -> list[str]:
        return [str(s.get(self._compress_field, "")) for s in self._samples]

    @staticmethod
    def _lcp(strings: list[str]) -> str:
        """Longest common prefix of a list of strings."""
        if not strings:
            return ""
        return os.path.commonprefix(strings)

    @staticmethod
    def _lcs(strings: list[str]) -> str:
        """Longest common suffix of a list of strings."""
        if not strings:
            return ""
        reversed_strings = [s[::-1] for s in strings]
        return os.path.commonprefix(reversed_strings)[::-1]

    def _detect_boundaries(self) -> tuple[str, str]:
        """Return (instruction_prefix, output_cue) for the corpus.

        instruction_prefix: shared text before the first per-sample divergence,
        snapped back to the last ``\\n\\n`` boundary.

        output_cue: shared trailing suffix, snapped forward to the first ``\\n``
        after where the texts reconverge (the cue is usually a single line like
        ``"Answer:"``).

        Both return ``""`` when they cannot be reliably determined.
        """
        texts = self._texts()
        if len(texts) < 2:
            return "", ""

        # --- instruction prefix ---
        lcp = self._lcp(texts)
        # Exclude degenerate case where all texts are identical.
        if lcp and lcp == texts[0]:
            lcp = ""
        if lcp:
            boundary = lcp.rfind("\n\n")
            if boundary == -1:
                lcp = ""
            else:
                lcp = lcp[:boundary]
        instruction_prefix = lcp.strip()

        # --- output cue ---
        # Only treat the LCS as an output cue when it starts at a newline
        # boundary (i.e., the cue is on its own line such as "Answer:").
        # Single-character suffixes like "." are not meaningful cues.
        lcs = self._lcs(texts)
        if lcs and lcs == texts[0]:
            lcs = ""
        if lcs:
            nl = lcs.find("\n")
            if nl == -1:
                # No newline → not a line-level cue; discard.
                lcs = ""
            else:
                lcs = lcs[nl:].lstrip("\n")
        output_cue = lcs.strip()

        return instruction_prefix, output_cue

    # ------------------------------------------------------------------
    # Per-sample splitting
    # ------------------------------------------------------------------

    def split(self, sample: dict) -> PromptSections:
        """Split a single sample into its four sections.

        When no shared prefix is found (flat/unstructured dataset), the entire
        prompt text is placed in the ``instruction`` section so that instruction
        operators are applied to it.  This preserves backward-compatible behaviour.
        """
        text = str(sample.get(self._compress_field, ""))

        # Fast path: no shared structure detected → treat entire text as instruction.
        if not self._instruction_prefix and not self._output_cue:
            return PromptSections(
                instruction=text,
                shots="",
                input_data="",
                output_cue="",
            )

        # Strip shared prefix and suffix first.
        inner = text
        if self._instruction_prefix and inner.startswith(self._instruction_prefix):
            inner = inner[len(self._instruction_prefix):].lstrip("\n")
        if self._output_cue and inner.endswith(self._output_cue):
            inner = inner[: -len(self._output_cue)].rstrip("\n")

        # ------------------------------------------------------------------
        # Instruction section
        # ------------------------------------------------------------------
        # The instruction_prefix is what all samples share.  It may embed
        # example blocks that belong to the shots section — detect those.
        instruction_text = self._instruction_prefix
        shots_text = ""

        # Check if a plain-text shots block was pre-computed (_shots field).
        raw_shots = sample.get(self._shots_field, "")
        if raw_shots:
            shots_text = str(raw_shots).strip()
            # The instruction_prefix may start with the shots block; remove it
            # so instruction contains only the actual task description.
            if instruction_text.startswith(shots_text):
                instruction_text = instruction_text[len(shots_text):].lstrip("\n")
        else:
            # Try to detect example blocks embedded at the start of the prefix.
            # Any text before the first example block goes to instruction;
            # the example blocks themselves form the shots section.
            if instruction_text:
                matches = list(_EXAMPLE_BLOCK.finditer(instruction_text))
                if matches:
                    first_start = matches[0].start()
                    last_end = matches[-1].end()
                    # Text before first block → prepend to instruction
                    pre_block = instruction_text[:first_start].strip()
                    # The example blocks → shots
                    shots_text = instruction_text[first_start:last_end].strip()
                    # Text after last block (if any) stays with instruction
                    post_block = instruction_text[last_end:].strip()
                    instruction_text = "\n\n".join(p for p in [pre_block, post_block] if p)

        # ------------------------------------------------------------------
        # Input data section (the per-sample remainder)
        # ------------------------------------------------------------------
        input_data = inner.strip()

        return PromptSections(
            instruction=instruction_text,
            shots=shots_text,
            input_data=input_data,
            output_cue=self._output_cue,
        )

    def split_all(self) -> list[PromptSections]:
        """Split every sample in the corpus."""
        return [self.split(s) for s in self._samples]