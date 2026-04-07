"""Task-performance evaluator abstractions."""

from __future__ import annotations

import abc
import random


class Evaluator(abc.ABC):
    """Score a compressed prompt against the original task.

    Parameters
    ----------
    original:
        The unmodified prompt string.
    compressed:
        The prompt after applying a transformation pipeline.
    sample:
        The full dataset sample dict (may contain target label, document, etc.).

    Returns
    -------
    float
        Performance score in [0, 1] where 1 is perfect performance.
    """

    @abc.abstractmethod
    def evaluate(self, original: str, compressed: str, sample: dict) -> float:
        """Return a performance score in [0, 1]."""

    def __call__(self, original: str, compressed: str, sample: dict) -> float:
        return self.evaluate(original, compressed, sample)


# ---------------------------------------------------------------------------
# Mock evaluator
# ---------------------------------------------------------------------------

class MockEvaluator(Evaluator):
    """Simulated evaluator for development and testing.

    Models the intuition that aggressive compression degrades performance:

        performance = max(0, 1 - alpha * compression_loss) + Gaussian_noise

    where ``compression_loss = 1 - len(compressed) / len(original)`` (fraction
    of characters removed).

    Parameters
    ----------
    alpha:
        Sensitivity of performance to compression.  ``alpha=0`` means
        performance is unaffected; ``alpha=1`` means removing all tokens
        drops performance to 0.
    noise_std:
        Standard deviation of Gaussian noise added to simulate model variance.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        noise_std: float = 0.05,
        seed: int | None = 42,
    ) -> None:
        self._alpha = alpha
        self._noise_std = noise_std
        self._rng = random.Random(seed)

    def evaluate(self, original: str, compressed: str, sample: dict) -> float:
        original_len = max(len(original), 1)
        compressed_len = len(compressed)
        compression_loss = 1.0 - compressed_len / original_len
        perf = max(0.0, 1.0 - self._alpha * compression_loss)
        noise = self._rng.gauss(0.0, self._noise_std)
        return float(min(1.0, max(0.0, perf + noise)))


class ConstantEvaluator(Evaluator):
    """Always returns a fixed score — useful for unit tests."""

    def __init__(self, score: float = 1.0) -> None:
        self._score = score

    def evaluate(self, original: str, compressed: str, sample: dict) -> float:
        return self._score