"""Data-driven transformation fitting.

:class:`DatasetVocabularyFitter` analyses a corpus of prompts to discover
patterns specific to that dataset — recurring domain terms, common answer
labels, repeated section headings, high-frequency filler words — and returns
transformation instances pre-configured with those patterns, optionally
extending the built-in defaults.

The fitted instances plug directly into the ``transformation_factory`` hook
that :class:`~prompt_piper.optimization.runner.Experiment` already exposes::

    fitter = DatasetVocabularyFitter(train_dataset)
    factory = fitter.make_transformation_factory()
    result  = Experiment(config, eval_dataset,
                         transformation_factory=factory).run()
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Callable

from .base import Transformation
from .lexical import (
    AbbreviationSubstitution,
    RemoveStopwords,
    _ABBREVIATIONS,
    _STOPWORDS,
)
from .structural import (
    ContextPruning,
    LabelShortening,
    SectionPruning,
    _DEFAULT_LABEL_MAP,
    _DEFAULT_PRUNABLE_SECTIONS,
    _HEADER_LINE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _words(text: str) -> list[str]:
    """Lowercase alphabetic tokens only."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    from nltk.util import ngrams
    return list(ngrams(tokens, n))


def _doc_frequencies(sequences: list[list[tuple]]) -> Counter:
    """Count how many documents contain each sequence (at least once)."""
    freq: Counter = Counter()
    for doc_seqs in sequences:
        for seq in set(doc_seqs):
            freq[seq] += 1
    return freq


def _abbrev_from_phrase(words: tuple[str, ...]) -> str:
    """Uppercase initials of each word (e.g. ('multiple', 'choice') → 'MC')."""
    return "".join(w[0].upper() for w in words)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DatasetVocabularyFitter:
    """Mine a dataset to produce corpus-fitted transformation instances.

    Parameters
    ----------
    dataset:
        List of sample dicts (the same format used by ``Experiment``).
    prompt_field:
        Key that holds the prompt string in each sample.
    target_field:
        Key that holds the target/label string in each sample.
    sample_size:
        If set, randomly subsample this many documents before fitting.
        Useful when the training set is large.
    seed:
        Random seed for reproducible subsampling.
    """

    def __init__(
        self,
        dataset: list[dict],
        prompt_field: str = "prompt",
        target_field: str = "target",
        sample_size: int | None = None,
        seed: int = 42,
    ) -> None:
        corpus = dataset
        if sample_size is not None and sample_size < len(dataset):
            rng = random.Random(seed)
            corpus = rng.sample(dataset, sample_size)
        self._corpus = corpus
        self._prompt_field = prompt_field
        self._target_field = target_field

    # ------------------------------------------------------------------
    # Public fitters
    # ------------------------------------------------------------------

    def fit_label_shortening(
        self,
        extend_default: bool = True,
        max_short_len: int = 4,
    ) -> LabelShortening:
        """Build a label map from the unique target strings in the dataset.

        Labels whose length is already ≤ ``max_short_len`` are kept as-is.
        Multi-word labels are abbreviated to their uppercase initials.
        Single long words are truncated to ``max_short_len`` characters.

        Parameters
        ----------
        extend_default:
            If ``True``, the mined map is merged on top of
            ``_DEFAULT_LABEL_MAP``.  Set to ``False`` to use only the
            mined entries.
        max_short_len:
            Labels at or below this length are considered already short and
            are not shortened further.
        """
        base_map = dict(_DEFAULT_LABEL_MAP) if extend_default else {}
        targets = self._get_targets()

        for raw in sorted(set(targets)):
            label = raw.strip()
            if not label or label.lower() in base_map:
                continue
            if len(label) <= max_short_len:
                continue  # already short enough
            words = label.split()
            if len(words) >= 2:
                short = "".join(w[0].upper() for w in words)
            else:
                short = label[:max_short_len]
            if short and short != label:
                base_map[label.lower()] = short

        return LabelShortening(label_map=base_map)

    def fit_abbreviation_substitution(
        self,
        min_doc_freq: float = 0.3,
        min_phrase_chars: int = 10,
        max_ngram: int = 3,
        extend_default: bool = True,
    ) -> AbbreviationSubstitution:
        """Discover recurring multi-word phrases and abbreviate them.

        For each N-gram (2 ≤ N ≤ ``max_ngram``) that appears in at least
        ``min_doc_freq`` fraction of prompts and whose total character length
        is ≥ ``min_phrase_chars``, an uppercase-initials abbreviation is
        generated and added to the substitution list.

        Phrases already covered by the default list are skipped.

        Parameters
        ----------
        min_doc_freq:
            Minimum fraction of documents a phrase must appear in.
        min_phrase_chars:
            Minimum total character length (joined without spaces) of the
            phrase to be eligible.
        max_ngram:
            Largest N-gram size to consider (2 or 3 recommended).
        extend_default:
            Prepend the mined entries to ``_ABBREVIATIONS`` (default).
        """
        prompts = self._get_prompts()
        n_docs = len(prompts)
        threshold = max(1, int(min_doc_freq * n_docs))

        # Collect all existing target strings so we don't abbreviate them
        target_words: set[str] = set()
        for t in self._get_targets():
            target_words.update(_words(t))

        # Build an expanded stopword set that includes corpus-level filler
        # words (high doc-frequency words not found in targets). This avoids
        # abbreviating structural template words like "following" or "answer"
        # while still capturing genuine domain phrases like "sentiment analysis".
        unigram_doc_freq: Counter = Counter()
        for prompt in prompts:
            for w in set(_words(prompt)):
                unigram_doc_freq[w] += 1
        corpus_stopwords = _STOPWORDS | {
            w for w, c in unigram_doc_freq.items()
            if c / n_docs >= 0.85
            and len(w) >= 4
            and w not in target_words
        }

        # Existing patterns (to avoid duplicates)
        existing_patterns = {p.pattern for p, _ in _ABBREVIATIONS}

        new_entries: list[tuple[re.Pattern[str], str]] = []
        seen_phrases: set[tuple[str, ...]] = set()

        for n in range(2, max_ngram + 1):
            doc_ngrams = [_ngrams(_words(p), n) for p in prompts]
            freq = _doc_frequencies(doc_ngrams)

            # Sort by frequency descending for deterministic ordering
            for phrase, count in freq.most_common():
                if count < threshold:
                    break
                if phrase in seen_phrases:
                    continue
                joined = "".join(phrase)
                if len(joined) < min_phrase_chars:
                    continue
                # Skip if any word is a target label (would mangle answers)
                if any(w in target_words for w in phrase):
                    continue
                # Require at least one genuine content word: not in the
                # expanded corpus stopword set and length ≥ 5.
                content_words = [
                    w for w in phrase
                    if w not in corpus_stopwords and len(w) >= 5
                ]
                if not content_words:
                    continue
                abbrev = _abbrev_from_phrase(phrase)
                # Skip trivial or duplicate abbreviations
                if abbrev == " ".join(phrase).upper():
                    continue
                pattern_src = r"\b" + r"\s+".join(re.escape(w) for w in phrase) + r"\b"
                if pattern_src in existing_patterns:
                    continue
                compiled = re.compile(pattern_src, re.I)
                new_entries.append((compiled, abbrev))
                existing_patterns.add(pattern_src)
                seen_phrases.add(phrase)

        combined = new_entries + list(_ABBREVIATIONS) if extend_default else new_entries
        return AbbreviationSubstitution(abbreviations=combined)

    def fit_section_pruning(
        self,
        min_doc_freq: float = 0.3,
        extend_default: bool = True,
    ) -> SectionPruning:
        """Find section headings that recur across prompts and add them to the
        prunable set.

        A heading that appears in ≥ ``min_doc_freq`` fraction of all documents
        is considered structural boilerplate and can safely be removed.

        Parameters
        ----------
        min_doc_freq:
            Minimum fraction of documents the heading must appear in.
        extend_default:
            Merge the mined headings with ``_DEFAULT_PRUNABLE_SECTIONS``.
        """
        prompts = self._get_prompts()
        n_docs = len(prompts)
        threshold = max(1, int(min_doc_freq * n_docs))

        heading_counts: Counter = Counter()
        for prompt in prompts:
            seen_in_doc: set[str] = set()
            for line in prompt.splitlines():
                stripped = line.strip()
                if _HEADER_LINE.fullmatch(stripped):
                    # Normalise: lowercase, strip leading #, strip trailing :
                    normalised = re.sub(r"^#+\s*", "", stripped).lower().rstrip(":").strip()
                    if normalised and normalised not in seen_in_doc:
                        heading_counts[normalised] += 1
                        seen_in_doc.add(normalised)

        mined = [h for h, c in heading_counts.items() if c >= threshold]
        if extend_default:
            combined = list(_DEFAULT_PRUNABLE_SECTIONS) + [
                h for h in mined if h not in _DEFAULT_PRUNABLE_SECTIONS
            ]
        else:
            combined = mined
        return SectionPruning(sections=combined)

    def fit_stopwords(
        self,
        min_doc_freq: float = 0.8,
        min_word_len: int = 4,
        extend_default: bool = True,
    ) -> RemoveStopwords:
        """Discover high-frequency non-informative words in the prompt corpus.

        A word qualifies as a corpus-level stopword if it:

        * appears in ≥ ``min_doc_freq`` fraction of prompts,
        * is not present in any target string (so answer tokens are safe),
        * has length ≥ ``min_word_len``.

        Parameters
        ----------
        min_doc_freq:
            Minimum fraction of documents the word must appear in.
            A high value (≥ 0.8) is recommended to stay conservative.
        min_word_len:
            Minimum word length to consider (avoids removing short words
            that are already covered by the default set).
        extend_default:
            Union with ``_STOPWORDS`` (default).  Set to ``False`` to
            return only the mined stopwords.
        """
        prompts = self._get_prompts()
        n_docs = len(prompts)
        threshold = max(1, int(min_doc_freq * n_docs))

        # All words that appear in any target are off-limits
        protected: set[str] = set()
        for t in self._get_targets():
            protected.update(_words(t))

        word_doc_freq: Counter = Counter()
        for prompt in prompts:
            for w in set(_words(prompt)):
                word_doc_freq[w] += 1

        mined: set[str] = set()
        for word, count in word_doc_freq.items():
            if (
                count >= threshold
                and len(word) >= min_word_len
                and word not in protected
                and word not in _STOPWORDS
            ):
                mined.add(word)

        result = _STOPWORDS | mined if extend_default else frozenset(mined)
        return RemoveStopwords(stopwords=frozenset(result))

    def fit_context_pruning(
        self,
        top_k: int = 20,
    ) -> ContextPruning:
        """Extract the most discriminative terms from targets as anchor words.

        Paragraphs that contain none of the anchor terms are considered
        off-topic and pruned.  The anchor vocabulary is built from the most
        frequent words in the target strings, excluding very common English
        words.

        Parameters
        ----------
        top_k:
            Number of top target-vocabulary terms to use as anchors.
        """
        target_word_freq: Counter = Counter()
        for t in self._get_targets():
            target_word_freq.update(_words(t))

        # Filter out stopwords (uses NLTK list via _STOPWORDS) and very short words.
        anchors = [
            w for w, _ in target_word_freq.most_common(top_k + len(_STOPWORDS))
            if w not in _STOPWORDS and len(w) >= 3
        ][:top_k]

        return ContextPruning(anchor_terms=anchors if anchors else None)

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def fit_all(
        self,
        extend_default: bool = True,
        abbreviation_min_doc_freq: float = 0.3,
        stopword_min_doc_freq: float = 0.8,
        section_min_doc_freq: float = 0.3,
    ) -> dict[str, Transformation]:
        """Run all fitters and return a ``{operator_id: instance}`` dict.

        The dict can be passed to :meth:`make_transformation_factory` or
        inspected directly.

        Returns
        -------
        dict[str, Transformation]
            Keyed by operator id (``"O3"``, ``"O5"``, ``"O17"``, ``"O18"``,
            ``"O22"``).
        """
        return {
            "O3":  self.fit_stopwords(
                       min_doc_freq=stopword_min_doc_freq,
                       extend_default=extend_default,
                   ),
            "O5":  self.fit_abbreviation_substitution(
                       min_doc_freq=abbreviation_min_doc_freq,
                       extend_default=extend_default,
                   ),
            "O17": self.fit_context_pruning(),
            "O18": self.fit_section_pruning(
                       min_doc_freq=section_min_doc_freq,
                       extend_default=extend_default,
                   ),
            "O22": self.fit_label_shortening(extend_default=extend_default),
        }

    def make_transformation_factory(
        self,
        extend_default: bool = True,
        **fit_kwargs,
    ) -> Callable[[type[Transformation]], Transformation]:
        """Return a factory callable for use with ``Experiment``.

        The factory returns a fitted instance for the operators that benefit
        from corpus adaptation (O3, O5, O17, O18, O22) and falls back to the
        default no-arg constructor for all others.

        Parameters
        ----------
        extend_default:
            Passed through to every ``fit_*`` method.
        **fit_kwargs:
            Forwarded to :meth:`fit_all` (e.g. ``abbreviation_min_doc_freq``).

        Example
        -------
        ::

            fitter  = DatasetVocabularyFitter(train_dataset)
            factory = fitter.make_transformation_factory()
            result  = Experiment(config, eval_dataset,
                                 transformation_factory=factory).run()
        """
        fitted = self.fit_all(extend_default=extend_default, **fit_kwargs)

        def factory(cls: type[Transformation]) -> Transformation:
            if cls.id in fitted:
                return fitted[cls.id]
            return cls()

        return factory

    def report(
        self,
        extend_default: bool = True,
    ) -> dict:
        """Return a summary of what each fitter discovered.

        Useful for inspecting or displaying in a notebook before committing
        to a factory.

        Returns
        -------
        dict with keys ``"labels"``, ``"abbreviations"``, ``"sections"``,
        ``"stopwords"``, ``"context_anchors"``.
        """
        fitted = self.fit_all(extend_default=extend_default)

        label_inst: LabelShortening = fitted["O22"]
        base_labels = set(_DEFAULT_LABEL_MAP.keys())
        new_labels = {k: v for k, v in label_inst._map.items() if k not in base_labels}

        abbrev_inst: AbbreviationSubstitution = fitted["O5"]
        default_patterns = {p.pattern for p, _ in _ABBREVIATIONS}
        new_abbrevs = [
            (p.pattern, r)
            for p, r in abbrev_inst._abbreviations
            if p.pattern not in default_patterns
        ]

        section_inst: SectionPruning = fitted["O18"]
        new_sections = [
            h for h in section_inst._headings
            if h not in _DEFAULT_PRUNABLE_SECTIONS
        ]

        stopword_inst: RemoveStopwords = fitted["O3"]
        new_stopwords = sorted(stopword_inst._stopwords - _STOPWORDS)

        context_inst: ContextPruning = fitted["O17"]
        anchors = (
            [p.pattern for p in context_inst._anchors]
            if context_inst._anchors is not None
            else []
        )

        return {
            "labels":          new_labels,
            "abbreviations":   new_abbrevs,
            "sections":        new_sections,
            "stopwords":       new_stopwords,
            "context_anchors": anchors,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_prompts(self) -> list[str]:
        return [
            str(s.get(self._prompt_field, ""))
            for s in self._corpus
        ]

    def _get_targets(self) -> list[str]:
        return [
            str(s.get(self._target_field, ""))
            for s in self._corpus
            if s.get(self._target_field) is not None
        ]