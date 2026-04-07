"""Token-counting abstractions."""

from __future__ import annotations

import abc


class Tokenizer(abc.ABC):
    """Count tokens in a string."""

    @abc.abstractmethod
    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""

    def __call__(self, text: str) -> int:
        return self.count(text)


class WhitespaceTokenizer(Tokenizer):
    """Approximate token count by splitting on whitespace."""

    def count(self, text: str) -> int:
        return len(text.split())


class CharTokenizer(Tokenizer):
    """Count characters (useful as a cheap proxy when no tokenizer is available)."""

    def count(self, text: str) -> int:
        return len(text)


# Optional: tiktoken-backed tokenizer (requires `pip install tiktoken`)
class TiktokenTokenizer(Tokenizer):
    """Exact token count using OpenAI's tiktoken library.

    Parameters
    ----------
    encoding:
        Tiktoken encoding name, e.g. ``"cl100k_base"`` (GPT-4) or
        ``"p50k_base"`` (GPT-3).
    """

    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "Install tiktoken to use TiktokenTokenizer: pip install tiktoken"
            ) from e
        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))