from __future__ import annotations

import abc


class Transformation(abc.ABC):
    """Base class for all prompt transformations.

    A transformation takes a prompt string and returns a (possibly shorter)
    prompt string.  Subclasses must implement :meth:`apply`.
    """

    #: Short identifier matching the paper table (e.g. "O1").
    id: str
    #: Human-readable name.
    name: str
    #: One of "lexical", "syntactic", "structural", "semantic".
    category: str

    @abc.abstractmethod
    def apply(self, prompt: str) -> str:
        """Return the transformed prompt."""

    def __call__(self, prompt: str) -> str:
        return self.apply(prompt)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r})"