"""pymoo Problem definition for prompt compression."""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

from prompt_piper.transformations.base import Transformation
from .evaluator import Evaluator
from .tokenizer import Tokenizer


class PromptCompressionProblem(Problem):
    """Multi-objective optimisation problem for prompt compression.

    **Variables** — binary vector of length ``n_transformations``.
    Bit ``i = 1`` means transformation ``i`` is applied; ``0`` skips it.

    **Objectives** (both minimised):

    * ``F[:, 0]`` — average token count across the dataset subset.
    * ``F[:, 1]`` — negative average task performance (so maximising
      performance becomes minimising ``-performance``).

    Parameters
    ----------
    transformations:
        Ordered list of instantiated :class:`Transformation` objects.
        The ordering determines which bit in the binary vector controls
        which operator.
    dataset:
        List of sample dicts.  Each must contain at least the key given by
        ``prompt_field``.
    evaluator:
        :class:`Evaluator` used to score compressed prompts.
    tokenizer:
        :class:`Tokenizer` used to count tokens in compressed prompts.
    prompt_field:
        Key inside each sample dict that holds the prompt string.
    compress_field:
        Key inside each sample dict to apply transformations to.  When
        ``None`` (default) falls back to ``prompt_field``.  Setting this to
        a different field (e.g. ``"prompt_0shot"``) lets you compress the
        0-shot prompt while keeping the full formatted prompt (with any
        few-shot examples) in ``prompt_field``.  Token counts and evaluator
        calls both operate on the compressed value of this field.
    """

    def __init__(
        self,
        transformations: list[Transformation],
        dataset: list[dict],
        evaluator: Evaluator,
        tokenizer: Tokenizer,
        prompt_field: str = "prompt",
        compress_field: str | None = None,
    ) -> None:
        super().__init__(
            n_var=len(transformations),
            n_obj=2,
            xl=np.zeros(len(transformations), dtype=int),
            xu=np.ones(len(transformations), dtype=int),
            vtype=int,
        )
        self._transforms = transformations
        self._dataset = dataset
        self._evaluator = evaluator
        self._tokenizer = tokenizer
        self._prompt_field = prompt_field
        self._compress_field = compress_field or prompt_field

    # ------------------------------------------------------------------
    # pymoo interface
    # ------------------------------------------------------------------

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        F = np.empty((len(x), 2), dtype=float)
        for i, bits in enumerate(x):
            pipeline = [t for t, b in zip(self._transforms, bits) if b]
            avg_tokens, avg_perf = self._score_pipeline(pipeline)
            F[i, 0] = avg_tokens
            F[i, 1] = -avg_perf
        out["F"] = F

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _score_pipeline(self, pipeline: list[Transformation]) -> tuple[float, float]:
        """Return (avg_tokens, avg_performance) for *pipeline* on the dataset."""
        originals = [s[self._compress_field] for s in self._dataset]

        compressed_texts: list[str] = []
        for original in originals:
            text = original
            for transform in pipeline:
                text = transform(text)
            compressed_texts.append(text)

        total_tokens = sum(self._tokenizer.count(c) for c in compressed_texts)
        perfs = self._evaluator.evaluate_batch(originals, compressed_texts, self._dataset)

        n = len(self._dataset)
        return total_tokens / n, sum(perfs) / n

    def score_solution(self, bits: np.ndarray) -> tuple[float, float]:
        """Publicly exposed single-solution scorer (useful for baselines).

        Returns
        -------
        tuple[float, float]
            ``(avg_tokens, avg_performance)`` — note: performance is *positive*
            here (not negated), unlike the raw objective stored in ``F``.
        """
        pipeline = [t for t, b in zip(self._transforms, bits) if b]
        return self._score_pipeline(pipeline)