"""pymoo Problem definition for prompt compression."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pymoo.core.problem import Problem

from prompt_piper.transformations.base import Transformation
from prompt_piper.transformations.sections import PromptSections
from .evaluator import Evaluator
from .tokenizer import Tokenizer


class PromptCompressionProblem(Problem):
    """Multi-objective optimisation problem for prompt compression.

    **Section-aware mode** (default when *sections* is provided):

    The binary vector is split into three consecutive regions::

        [ instruction_bits | shots_bits | data_bits ]
          len = n_i          len = n_s   len = n_d

    Each region controls operators for one prompt section.  Static sections
    (instruction, shots) are compressed once per unique
    ``(instruction_bits, shots_bits)`` combination and cached.

    **Legacy flat mode** (when *sections* is ``None`` and *transformations*
    is provided):

    A single 28-bit vector where bit ``i = 1`` applies operator ``i`` to the
    entire ``compress_field`` text — identical to the original pipeline.

    **Objectives** (both minimised):

    * ``F[:, 0]`` — average token count across the dataset subset.
    * ``F[:, 1]`` — negative average task performance.

    Parameters
    ----------
    instruction_ops:
        Transformation instances for the instruction section.
    shot_ops:
        Transformation instances for the shots section.
    data_ops:
        Transformation instances for the input_data section.
    sections:
        Pre-built :class:`~prompt_piper.transformations.sections.PromptSections`
        list (one per dataset sample).  When provided, section-aware mode is
        activated.
    transformations:
        *Legacy*: flat list of transformation instances.  Used when *sections*
        is ``None``.
    dataset:
        List of sample dicts.  Used only for evaluator calls (``_doc``, etc.).
    evaluator:
        :class:`Evaluator` used to score compressed prompts.
    tokenizer:
        :class:`Tokenizer` used to count tokens in compressed prompts.
    prompt_field:
        Key inside each sample dict that holds the prompt string.
    compress_field:
        Key inside each sample dict to apply transformations to (legacy mode
        only).  Ignored in section-aware mode.
    """

    def __init__(
        self,
        # Section-aware mode
        instruction_ops: list[Transformation] | None = None,
        shot_ops: list[Transformation] | None = None,
        data_ops: list[Transformation] | None = None,
        sections: list[PromptSections] | None = None,
        # Legacy flat mode
        transformations: list[Transformation] | None = None,
        # Common
        dataset: list[dict] | None = None,
        evaluator: Evaluator | None = None,
        tokenizer: Tokenizer | None = None,
        prompt_field: str = "prompt",
        compress_field: str | None = None,
    ) -> None:
        self._sections = sections

        if sections is not None:
            # Section-aware mode
            self._instruction_ops: list[Transformation] = instruction_ops or []
            self._shot_ops: list[Transformation] = shot_ops or []
            self._data_ops: list[Transformation] = data_ops or []
            # Combined for backward-compat access from baselines
            self._transforms: list[Transformation] = (
                self._instruction_ops + self._shot_ops + self._data_ops
            )
            n_var = len(self._transforms)
            self._static_cache: dict[tuple, tuple[str, str]] = {}
        else:
            # Legacy flat mode
            if transformations is None:
                raise ValueError(
                    "Either sections or transformations must be provided."
                )
            self._transforms = transformations
            self._instruction_ops = []
            self._shot_ops = []
            self._data_ops = []
            n_var = len(transformations)

        self._dataset: list[dict] = dataset or []
        self._evaluator: Evaluator = evaluator  # type: ignore[assignment]
        self._tokenizer: Tokenizer = tokenizer  # type: ignore[assignment]
        self._prompt_field = prompt_field
        self._compress_field = compress_field or prompt_field

        super().__init__(
            n_var=n_var,
            n_obj=2,
            xl=np.zeros(n_var, dtype=int),
            xu=np.ones(n_var, dtype=int),
            vtype=int,
        )

    # ------------------------------------------------------------------
    # pymoo interface
    # ------------------------------------------------------------------

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evaluate all solutions in a generation with deduplication.

        All compressed texts for every solution in the population are
        collected first, then deduplicated on (sample_index, text) before
        a single :meth:`Evaluator.evaluate_batch` call.  This avoids
        redundant LLM evaluations when multiple solutions produce identical
        compressed output for the same sample (e.g. the identity solution
        appearing in many individuals).
        """
        n_solutions = len(x)

        # Phase 1: compress without evaluating
        per_solution: list[tuple[list[str], list[str]]] = []
        for bits in x:
            originals, compressed = self._compress_bits(bits)
            per_solution.append((originals, compressed))

        # Phase 2: build deduplicated batch
        key_to_eval_idx: dict[tuple, int] = {}
        unique_originals: list[str] = []
        unique_compressed: list[str] = []
        unique_docs: list[dict] = []

        for originals, compressed in per_solution:
            for j, (orig, comp) in enumerate(zip(originals, compressed)):
                key = (j, comp)
                if key not in key_to_eval_idx:
                    key_to_eval_idx[key] = len(unique_compressed)
                    unique_originals.append(orig)
                    unique_compressed.append(comp)
                    unique_docs.append(self._dataset[j] if j < len(self._dataset) else {})

        # Phase 3: single batch evaluation
        unique_perfs = self._evaluator.evaluate_batch(
            unique_originals, unique_compressed, unique_docs
        )

        # Phase 4: reconstruct per-solution objectives
        F = np.empty((n_solutions, 2), dtype=float)
        for i, (originals, compressed) in enumerate(per_solution):
            n = len(originals)
            total_tokens = sum(self._tokenizer.count(c) for c in compressed)
            total_perf = sum(
                unique_perfs[key_to_eval_idx[(j, comp)]]
                for j, comp in enumerate(compressed)
            )
            F[i, 0] = total_tokens / n
            F[i, 1] = -(total_perf / n)
        out["F"] = F

    # ------------------------------------------------------------------
    # Public scoring
    # ------------------------------------------------------------------

    def score_solution(self, bits: np.ndarray) -> tuple[float, float]:
        """Return ``(avg_tokens, avg_performance)`` for a binary vector.

        Performance is *positive* here (not negated).
        """
        return self._score_bits(bits)

    def score_transform_fn(self, fn) -> tuple[float, float]:
        """Score an arbitrary compression callable on the dataset.

        Parameters
        ----------
        fn:
            Any callable ``(text: str) -> str``.

        Returns
        -------
        tuple[float, float]
            ``(avg_tokens, avg_performance)`` — performance is *positive*.
        """
        if self._sections is not None:
            originals = [s.assemble() for s in self._sections]
        else:
            originals = [s[self._compress_field] for s in self._dataset]

        compressed_texts = [fn(t) for t in originals]
        total_tokens = sum(self._tokenizer.count(c) for c in compressed_texts)
        perfs = self._evaluator.evaluate_batch(originals, compressed_texts, self._dataset)
        n = max(len(self._dataset), 1)
        return total_tokens / n, sum(perfs) / n

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_bits(self, bits) -> tuple[float, float]:
        originals, compressed = self._compress_bits(bits)
        n = max(len(originals), 1)
        total_tokens = sum(self._tokenizer.count(c) for c in compressed)
        perfs = self._evaluator.evaluate_batch(originals, compressed, self._dataset)
        return total_tokens / n, sum(perfs) / n

    # ---------- Compression (no evaluation) ----------

    def _compress_bits(self, bits) -> tuple[list[str], list[str]]:
        """Return ``(originals, compressed_texts)`` for *bits* without evaluating."""
        if self._sections is not None:
            return self._compress_bits_sections(bits)
        return self._compress_bits_flat(bits)

    def _compress_bits_flat(self, bits) -> tuple[list[str], list[str]]:
        pipeline = [t for t, b in zip(self._transforms, bits) if b]
        originals = [s[self._compress_field] for s in self._dataset]
        compressed: list[str] = []
        for original in originals:
            text = original
            for transform in pipeline:
                text = transform(text)
            compressed.append(text)
        return originals, compressed

    def _compress_bits_sections(self, bits) -> tuple[list[str], list[str]]:
        n_i = len(self._instruction_ops)
        n_s = len(self._shot_ops)

        instruction_bits = tuple(int(b) for b in bits[:n_i])
        shots_bits = tuple(int(b) for b in bits[n_i:n_i + n_s])
        data_bits = list(bits[n_i + n_s:])

        c_instr, c_shots = self._get_static_compressed(instruction_bits, shots_bits)

        data_pipeline = [op for op, b in zip(self._data_ops, data_bits) if b]
        compressed_texts: list[str] = []
        originals: list[str] = []
        for section in self._sections:  # type: ignore[union-attr]
            c_data = section.input_data
            for op in data_pipeline:
                c_data = op(c_data)
            assembled = section.assemble(
                instruction=c_instr,
                shots=c_shots,
                input_data=c_data,
            )
            compressed_texts.append(assembled)
            originals.append(section.assemble())
        return originals, compressed_texts

    def _get_static_compressed(
        self,
        instruction_bits: tuple,
        shots_bits: tuple,
    ) -> tuple[str, str]:
        """Return ``(compressed_instruction, compressed_shots)`` from cache."""
        static_key = (instruction_bits, shots_bits)
        if static_key not in self._static_cache:
            instr_pipeline = [
                op for op, b in zip(self._instruction_ops, instruction_bits) if b
            ]
            shot_pipeline = [
                op for op, b in zip(self._shot_ops, shots_bits) if b
            ]
            base_instruction = self._sections[0].instruction if self._sections else ""  # type: ignore[index]
            base_shots = self._sections[0].shots if self._sections else ""  # type: ignore[index]
            c_instr = base_instruction
            for op in instr_pipeline:
                c_instr = op(c_instr)
            c_shots = base_shots
            for op in shot_pipeline:
                c_shots = op(c_shots)
            self._static_cache[static_key] = (c_instr, c_shots)
        return self._static_cache[static_key]

    # ------------------------------------------------------------------
    # Helpers for baselines / runner
    # ------------------------------------------------------------------

    def names_from_bits(self, bits) -> list[str]:
        """Return human-readable names for the active operators in *bits*."""
        if self._sections is None:
            return [t.name for t, b in zip(self._transforms, bits) if b]
        n_i = len(self._instruction_ops)
        n_s = len(self._shot_ops)
        names: list[str] = []
        for t, b in zip(self._instruction_ops, bits[:n_i]):
            if b:
                names.append(f"I:{t.name}")
        for t, b in zip(self._shot_ops, bits[n_i:n_i + n_s]):
            if b:
                names.append(f"S:{t.name}")
        for t, b in zip(self._data_ops, bits[n_i + n_s:]):
            if b:
                names.append(f"D:{t.name}")
        return names