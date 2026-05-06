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
    ReplaceWithShortSynonym,
    _ABBREVIATIONS,
    _SHORT_SYNONYMS,
    _STOPWORDS,
)
from .structural import (
    ContextPruning,
    ExampleSimplification,
    LabelShortening,
    RemovePromptHeader,
    SectionPruning,
    _DEFAULT_INPUT_MARKERS,
    _DEFAULT_LABEL_MAP,
    _DEFAULT_OUTPUT_MARKERS,
    _DEFAULT_PRUNABLE_SECTIONS,
    _HEADER_LINE,
)
from .syntactic import ImperativeRewrite, _IMPERATIVE_PATTERNS
from .semantic import ChunkCompression, _DEFAULT_CHUNK_SEPARATOR

# Matches "Word:" at the start of a line — used to detect example field markers.
_FIELD_MARKER = re.compile(r"^([\w][\w ]{0,15}?)\s*:\s*\S")

# Matches a literal XML/HTML opening tag returned by the LLM, e.g. "<instructions>".
_LITERAL_OPEN_TAG = re.compile(r"^<([\w][\w.-]*)>$")
# Matches a literal XML/HTML closing tag returned by the LLM, e.g. "</instructions>".
_LITERAL_CLOSE_TAG = re.compile(r"^</([\w][\w.-]*)>$")
# Matches a valid XML/HTML tag name (word chars plus hyphens/dots).
_TAG_NAME_RE = re.compile(r"^[\w][\w.-]*$")

#: Type of the optional LLM backend: takes (text, instruction) and returns a string.
Backend = Callable[[str, str], str]


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


def _merge_tag_pairs(patterns: list[str]) -> list[str]:
    """Merge literal ``<tag>`` / ``</tag>`` pairs into a single ``</?tag>`` pattern.

    When the LLM returns both the opening and the closing form of the same tag
    as two separate patterns, treating them independently is redundant and
    fragile.  This function detects such pairs and replaces them with a single
    combined pattern (``</?tagname>``) that matches both forms at once.

    Patterns that are not literal paired tags are passed through unchanged.
    """
    # First pass: collect positions of opening tags in the result list.
    result: list[str] = []
    open_positions: dict[str, int] = {}  # tag_name (lower) -> index in result

    for pat in patterns:
        m_close = _LITERAL_CLOSE_TAG.match(pat)
        if m_close:
            tag = m_close.group(1).lower()
            if tag in open_positions:
                # Replace the previously stored opening tag with the merged form.
                result[open_positions[tag]] = rf"</?{re.escape(tag)}>"
                open_positions.pop(tag)
                continue  # drop the closing tag — already merged
        result.append(pat)
        m_open = _LITERAL_OPEN_TAG.match(pat)
        if m_open:
            open_positions[m_open.group(1).lower()] = len(result) - 1

    return result


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
        backend: Backend | None = None,
        template_field: str | None = None,
         compress_field: str | None = None,
    ) -> None:
        corpus = dataset
        if sample_size is not None and sample_size < len(dataset):
            rng = random.Random(seed)
            corpus = rng.sample(dataset, sample_size)
        self._corpus = corpus
        self._prompt_field = prompt_field
        self._target_field = target_field
        self._seed = seed
        self._backend = backend
        self._template_field = template_field
        # When compressing the 0-shot field directly there is no few-shot prefix
        # in the text that operators will receive, so O17 must not protect any
        # paragraphs.  Treat template_field as inactive in that case.
        compress = compress_field or prompt_field
        self._effective_template_field = (
            template_field if template_field and template_field != compress else None
        )

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

        When ``template_field`` was supplied to the constructor, the few-shot
        prefix paragraphs are automatically protected from pruning — only the
        variable question portion (the 0-shot part) is subject to removal.

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

        return ContextPruning(
            anchor_terms=anchors if anchors else None,
            n_protected_paragraphs=self._count_fewshot_paragraphs(),
        )

    def fit_example_simplification(
        self,
        min_doc_freq: float = 0.3,
        extend_default: bool = True,
    ) -> ExampleSimplification:
        """Discover the input/output field names used in this dataset's examples.

        :class:`ExampleSimplification` (O21) converts ``Input: X\\nOutput: Y``
        blocks into ``X → Y``, but its built-in markers (``Input``,
        ``Question``, ``Q`` / ``Output``, ``Answer``, …) may not match a given
        benchmark's conventions (e.g. ``Passage:``/``Summary:``,
        ``Premise:``/``Hypothesis:``).

        This fitter scans prompts for consecutive ``"Word: content"`` line
        pairs.  The first word of each pair is counted as an input marker; the
        last word as an output marker.  Words appearing in at least
        ``min_doc_freq`` fraction of documents are returned as the discovered
        markers.

        Parameters
        ----------
        min_doc_freq:
            Minimum fraction of documents a marker must appear in.
        extend_default:
            When ``True`` (default), discovered markers are merged with the
            built-in :data:`_DEFAULT_INPUT_MARKERS` / :data:`_DEFAULT_OUTPUT_MARKERS`.
            When ``False``, only discovered markers are used (falls back to
            defaults if none are found).
        """
        prompts = self._get_prompts()
        n_docs = len(prompts)
        threshold = max(1, int(min_doc_freq * n_docs))

        input_counts: Counter = Counter()
        output_counts: Counter = Counter()

        for prompt in prompts:
            seen: set[tuple[str, str]] = set()
            lines = prompt.splitlines()
            i = 0
            while i < len(lines) - 1:
                m_cur = _FIELD_MARKER.match(lines[i])
                m_nxt = _FIELD_MARKER.match(lines[i + 1])
                if m_cur and m_nxt:
                    inp_word = m_cur.group(1).strip().lower()
                    out_word = m_nxt.group(1).strip().lower()
                    if (inp_word, out_word) not in seen:
                        input_counts[inp_word] += 1
                        output_counts[out_word] += 1
                        seen.add((inp_word, out_word))
                    i += 2
                else:
                    i += 1

        # Title-case to match typical prompt formatting.
        found_input = [w.title() for w, c in input_counts.most_common() if c >= threshold]
        found_output = [w.title() for w, c in output_counts.most_common() if c >= threshold]

        if extend_default:
            # Preserve default order, append new discoveries at the end.
            all_input = list(dict.fromkeys(_DEFAULT_INPUT_MARKERS + found_input))
            all_output = list(dict.fromkeys(_DEFAULT_OUTPUT_MARKERS + found_output))
        else:
            all_input = found_input or list(_DEFAULT_INPUT_MARKERS)
            all_output = found_output or list(_DEFAULT_OUTPUT_MARKERS)

        return ExampleSimplification(input_markers=all_input, output_markers=all_output)

    def fit_replace_with_short_synonym(
        self,
        n_samples: int = 5,
        extend_default: bool = True,
    ) -> ReplaceWithShortSynonym:
        """Discover domain-specific verbose phrases via LLM and fit :class:`ReplaceWithShortSynonym`.

        The built-in list covers generic English verbosity (``"in order to"``
        → ``"to"``, ``"utilize"`` → ``"use"``).  Domain-specific datasets
        accumulate their own wordy constructions that go unmatched.  The LLM
        is asked to identify such phrases directly from the corpus samples and
        return ``verbose phrase → short equivalent`` pairs.

        Falls back to the default synonym list when no backend is set or the
        LLM returns no parseable pairs.

        Parameters
        ----------
        n_samples:
            Number of prompts shown to the LLM.
        extend_default:
            When ``True`` (default), discovered pairs are appended after the
            built-in list.  When ``False``, only the discovered pairs are used.
        """
        instruction = (
            "Analyze the prompt samples below and identify verbose phrases or "
            "wordy constructions that appear repeatedly and could be replaced "
            "with shorter equivalents without changing meaning. "
            "Output format: one pair per line, written exactly as "
            "'verbose phrase → short equivalent' using the → character as separator. "
            "No bullet markers, no numbering, no explanations — just the pair. "
            "Example line: in order to → to\n"
            "Only include pairs where the short form is genuinely shorter and the "
            "phrase is directly observable in the samples. "
            "Do not invent pairs not grounded in the samples."
        )
        lines = self._llm_extract_lines(instruction, n_samples=n_samples)

        extra: list[tuple[re.Pattern[str], str]] = []
        for line in lines:
            for sep in (" → ", " -> ", " : "):
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        verbose, short = parts[0].strip(), parts[1].strip()
                        if verbose and short and len(short) < len(verbose):
                            try:
                                pat = re.compile(rf"\b{re.escape(verbose)}\b", re.I)
                                extra.append((pat, short))
                            except re.error:
                                pass
                    break

        synonyms = list(_SHORT_SYNONYMS) + extra if extend_default else extra or list(_SHORT_SYNONYMS)
        return ReplaceWithShortSynonym(synonyms=synonyms)

    def fit_imperative_rewrite(
        self,
        n_samples: int = 5,
        extend_default: bool = True,
    ) -> ImperativeRewrite:
        """Discover dataset-specific task-framing patterns via LLM and fit :class:`ImperativeRewrite`.

        The built-in patterns strip generic phrasings like ``"Your task is to"``
        or ``"I need you to"``.  Different benchmarks use their own conventions
        (e.g. ``"In this evaluation you are asked to"``, ``"For this problem"``).
        The LLM is shown sample prompts and asked to return regex patterns for
        these dataset-specific framings.

        Falls back to the default patterns when no backend is set or the LLM
        returns nothing parseable.

        Parameters
        ----------
        n_samples:
            Number of prompts shown to the LLM.
        extend_default:
            When ``True`` (default), discovered patterns are appended after
            the built-in list.  When ``False``, only discovered patterns are used.
        """
        instruction = (
            "Analyze the prompt samples below and identify verbose task-framing "
            "phrases at the start of lines that could be deleted entirely to "
            "leave just the imperative verb or direct instruction "
            "(e.g. 'Your task is to ' is deleted, keeping 'Classify...'). "
            "Output format: one Python regex pattern per line — the pattern must "
            "be ^-anchored, written for re.compile(..., re.IGNORECASE | re.MULTILINE), "
            "and must include a trailing \\s+ or space so the verb that follows is "
            "preserved. No bullet markers, no numbering, no explanations — "
            "just the raw regex string. "
            "Example line: ^In this evaluation you are asked to\\s+\n"
            "Only include patterns directly observable in the samples. "
            "Skip generic patterns already handled: 'Your task is to', "
            "'You need to', 'Please '."
        )
        raw = self._llm_extract_patterns(instruction, n_samples=n_samples)
        extra: list[tuple[re.Pattern[str], str]] = []
        for pat_str in raw:
            try:
                extra.append((re.compile(pat_str, re.I | re.M), ""))
            except re.error:
                pass

        patterns = list(_IMPERATIVE_PATTERNS) + extra if extend_default else extra or list(_IMPERATIVE_PATTERNS)
        return ImperativeRewrite(patterns=patterns)

    def fit_prompt_header(
        self,
        extend_default: bool = True,
        n_samples: int = 5,
    ) -> RemovePromptHeader:
        """Discover non-standard header formats via LLM and fit :class:`RemovePromptHeader`.

        The LLM is shown a sample of prompts and asked to return a bullet list
        of Python regex patterns (``fullmatch`` semantics) that identify header
        or title lines not already covered by the built-in markdown / ALL-CAPS
        heuristic — e.g. ``\\*\\*Title\\*\\*``, ``<section>``, ``===Title===``.

        Falls back to a plain default :class:`RemovePromptHeader` when no
        backend is set or the LLM returns no valid patterns.

        Parameters
        ----------
        extend_default:
            When ``True`` (default), discovered patterns are added on top of
            the built-in ``_HEADER_LINE`` pattern.  Set to ``False`` to use
            *only* the mined patterns (useful when the dataset uses a completely
            non-standard header style).
        n_samples:
            Number of prompts shown to the LLM.
        """
        instruction = (
            "Analyze the prompt samples below and identify any formats used to mark "
            "section headers or titles not already covered by standard Markdown "
            "(# headings) or ALL-CAPS lines. "
            "Output format: one Python regex pattern per line — the pattern must be "
            "suitable for re.fullmatch() with re.IGNORECASE, matching the entire "
            "header line (no leading/trailing context). "
            "No bullet markers, no numbering, no explanations — just the raw regex string. "
            "Example line: \\*\\*[\\w][\\w ]*\\*\\*\n"
            "Only include patterns directly observable in the samples. "
            "Focus on non-standard formats such as bold markers (**Title**), "
            "XML/HTML tags (<section>), or custom dividers (===Title===)."
        )
        raw = self._llm_extract_patterns(instruction, n_samples=n_samples)
        extra: list[re.Pattern[str]] = []
        for pat_str in raw:
            try:
                extra.append(re.compile(pat_str, re.I))
            except re.error:
                pass

        if extend_default:
            patterns = [_HEADER_LINE] + extra
        else:
            patterns = extra or [_HEADER_LINE]  # never leave the list empty
        return RemovePromptHeader(patterns=patterns)

    def fit_tagged_sections(
        self,
        n_samples: int = 5,
        extend_default: bool = True,
    ) -> SectionPruning:
        """Discover XML/HTML-style block tags via LLM and fit :class:`SectionPruning`.

        Asks the LLM to identify tag names (e.g. ``instructions``, ``context``,
        ``background``) used to wrap sections in the prompts, then configures
        :class:`SectionPruning` to remove entire ``<tag>…</tag>`` blocks —
        opening tag, content, and closing tag included.

        This complements the corpus-based :meth:`fit_section_pruning`, which
        targets plain-text and Markdown headings.  Both can be combined by
        passing the heading-based ``sections`` list alongside the discovered
        ``tag_names`` when constructing the final :class:`SectionPruning`
        instance (as :meth:`fit_all` does when a backend is available).

        Falls back to a default-only :class:`SectionPruning` when no backend
        is set or no tag names are found.

        Parameters
        ----------
        n_samples:
            Number of prompts shown to the LLM.
        extend_default:
            When ``True`` (default), the discovered tag-based patterns are
            added on top of the standard heading-based defaults.  When
            ``False``, heading-based removal is disabled and only the
            discovered tag patterns are applied.
        """
        tag_names = self._discover_tag_names(n_samples=n_samples)
        sections = _DEFAULT_PRUNABLE_SECTIONS if extend_default else []
        return SectionPruning(sections=list(sections), tag_names=tag_names)

    def fit_chunk_separator(
        self,
        n_samples: int = 5,
    ) -> ChunkCompression:
        """Discover the block-separator pattern used in this dataset via LLM.

        :class:`ChunkCompression` defaults to splitting on blank lines
        (``\\n{2,}``).  When prompts use a different convention (e.g.
        ``\\n---\\n``, ``\\n===\\n``, ``\\n***\\n``), this fitter detects the
        right pattern and returns a pre-configured :class:`ChunkCompression`
        instance.

        Falls back to the default separator when no backend is set or the LLM
        returns nothing parseable.

        Parameters
        ----------
        n_samples:
            Number of prompts shown to the LLM.
        """
        instruction = (
            "Analyze the prompt samples below and identify the exact text pattern "
            "used to separate major independent sections or blocks. "
            "Output format: a single Python regex pattern on one line, suitable for "
            "re.split() — no quotes, no r-prefix, no bullet markers, no explanation. "
            "Write escape sequences literally (e.g. \\n not a real newline). "
            "Example lines (pick the one that matches, or write your own): "
            "\\n\\n  |  \\n---\\n  |  \\n={3,}\\n  |  \\n\\*{3,}\\n\n"
            "If the samples simply use blank lines, output: \\n\\n"
        )
        raw = self._llm_extract_patterns(instruction, n_samples=n_samples)
        separator = _DEFAULT_CHUNK_SEPARATOR
        for pat_str in raw:
            try:
                separator = re.compile(pat_str)
                break  # take the first valid pattern
            except re.error:
                pass
        return ChunkCompression(separator=separator)

    # ------------------------------------------------------------------
    # Private LLM helpers
    # ------------------------------------------------------------------

    def _sample_prompts(self, n: int = 5, max_chars: int = 1500) -> list[str]:
        """Return up to *n* prompts from the corpus, each capped at *max_chars*."""
        prompts = self._get_prompts()
        rng = random.Random(self._seed)
        chosen = rng.sample(prompts, min(n, len(prompts)))
        return [p[:max_chars] for p in chosen]

    def _llm_extract_lines(
        self,
        instruction: str,
        n_samples: int = 5,
        max_chars_per_prompt: int = 1500,
    ) -> list[str]:
        """Call the LLM with sample prompts and return cleaned bullet-list lines.

        Strips list markers (``-``, ``*``, ``•``, numbers) and surrounding
        backticks from each non-empty line.  No further validation is applied
        — callers are responsible for interpreting the raw strings.

        Returns an empty list when no backend is configured or the corpus is
        empty.
        """
        if self._backend is None:
            return []
        samples = self._sample_prompts(n=n_samples, max_chars=max_chars_per_prompt)
        if not samples:
            return []
        context = "\n\n[SAMPLE SEPARATOR]\n\n".join(
            f"[Sample {i + 1}]\n{p}" for i, p in enumerate(samples)
        )
        response = self._backend(context, instruction)
        lines: list[str] = []
        for line in response.splitlines():
            line = line.strip().lstrip("-*•0123456789.) \t`").rstrip("`")
            if line:
                lines.append(line)
        return lines

    def _llm_extract_patterns(
        self,
        instruction: str,
        n_samples: int = 5,
        max_chars_per_prompt: int = 1500,
    ) -> list[str]:
        """Like :meth:`_llm_extract_lines` but validates each line as a Python regex.

        After validation, literal ``<tag>`` / ``</tag>`` pairs are merged into
        a single ``</?tag>`` pattern via :func:`_merge_tag_pairs` so that the
        caller never receives redundant paired patterns.

        Returns an empty list when no backend is configured or the LLM
        produces no valid patterns.
        """
        valid: list[str] = []
        for line in self._llm_extract_lines(instruction, n_samples, max_chars_per_prompt):
            try:
                re.compile(line)
                valid.append(line)
            except re.error:
                pass
        return _merge_tag_pairs(valid)

    def _discover_tag_names(self, n_samples: int = 5) -> list[str]:
        """Ask the LLM to identify XML/HTML tag names used in the prompt corpus.

        Returns a list of bare tag name strings (no angle brackets), validated
        to contain only word characters and hyphens/dots.  Returns an empty
        list when no backend is configured.
        """
        instruction = (
            "Analyze the prompt samples below and list the XML or HTML-style tag "
            "names used to delimit sections or blocks (e.g. instructions, context, "
            "background, system, user). "
            "Output format: one bare tag name per line — no angle brackets, no "
            "slash, no attributes, no bullet markers, no numbering, no explanation. "
            "Example line: instructions\n"
            "Only include tag names directly observable in the samples."
        )
        return [
            line for line in self._llm_extract_lines(instruction, n_samples=n_samples)
            if _TAG_NAME_RE.match(line)
        ]

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def fit_all(
        self,
        extend_default: bool = True,
        abbreviation_min_doc_freq: float = 0.3,
        stopword_min_doc_freq: float = 0.8,
        section_min_doc_freq: float = 0.3,
        operator_ids: list[str] | None = None,
    ) -> dict[str, Transformation]:
        """Run fitters and return a ``{operator_id: instance}`` dict.

        The dict can be passed to :meth:`make_transformation_factory` or
        inspected directly.

        Corpus-only fitters (no backend required): O3, O5, O17, O18, O21, O22.

        LLM-backed fitters (require ``backend``): O4, O10, O15, O28.  These
        are included automatically when a backend was supplied to the
        constructor.

        Parameters
        ----------
        operator_ids:
            Optional allowlist of operator IDs to fit (e.g. ``["O3", "O22"]``).
            When ``None`` (default), all eligible operators are fitted.
            IDs not in the eligible set (or requiring a backend that is absent)
            are silently ignored.

        Returns
        -------
        dict[str, Transformation]
            Keyed by operator id.
        """
        want: set[str] | None = set(operator_ids) if operator_ids is not None else None

        def _skip(oid: str) -> bool:
            return want is not None and oid not in want

        # O18 needs special treatment: corpus headings + optional LLM tag names.
        # Build it only when needed.
        o18: Transformation | None = None
        if not _skip("O18"):
            corpus_sections = self.fit_section_pruning(
                min_doc_freq=section_min_doc_freq,
                extend_default=extend_default,
            )
            if self._backend is not None:
                tag_names = self._discover_tag_names()
                o18 = SectionPruning(
                    sections=corpus_sections._headings,
                    tag_names=tag_names,
                )
            else:
                o18 = corpus_sections

        result: dict[str, Transformation] = {}

        if not _skip("O3"):
            result["O3"] = self.fit_stopwords(
                min_doc_freq=stopword_min_doc_freq,
                extend_default=extend_default,
            )
        if not _skip("O5"):
            result["O5"] = self.fit_abbreviation_substitution(
                min_doc_freq=abbreviation_min_doc_freq,
                extend_default=extend_default,
            )
        if not _skip("O17"):
            result["O17"] = self.fit_context_pruning()
        if o18 is not None:
            result["O18"] = o18
        if not _skip("O21"):
            result["O21"] = self.fit_example_simplification(extend_default=extend_default)
        if not _skip("O22"):
            result["O22"] = self.fit_label_shortening(extend_default=extend_default)

        if self._backend is not None:
            if not _skip("O4"):
                result["O4"] = self.fit_replace_with_short_synonym(extend_default=extend_default)
            if not _skip("O10"):
                result["O10"] = self.fit_imperative_rewrite(extend_default=extend_default)
            if not _skip("O15"):
                result["O15"] = self.fit_prompt_header(extend_default=extend_default)
            if not _skip("O28"):
                result["O28"] = self.fit_chunk_separator()

        return result

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
        ``"stopwords"``, ``"context_anchors"``, ``"example_fields"``, and —
        when a backend is available — ``"short_synonyms"``,
        ``"imperative_patterns"``, ``"header_patterns"``, ``"tag_names"``,
        and ``"chunk_separator"``.
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

        # O21 — example field markers discovered beyond built-in defaults
        ex_inst: ExampleSimplification = fitted["O21"]
        new_input_markers = [
            m for m in ex_inst._input_markers if m not in _DEFAULT_INPUT_MARKERS
        ]
        new_output_markers = [
            m for m in ex_inst._output_markers if m not in _DEFAULT_OUTPUT_MARKERS
        ]

        result = {
            "labels":          new_labels,
            "abbreviations":   new_abbrevs,
            "sections":        new_sections,
            "stopwords":       new_stopwords,
            "context_anchors": anchors,
            "example_fields":  {
                "input_markers":  new_input_markers,
                "output_markers": new_output_markers,
            },
        }

        if self._backend is not None:
            # O4 — verbose-phrase / short-synonym pairs beyond built-in list
            syn_inst: ReplaceWithShortSynonym = fitted["O4"]
            default_syn_patterns = {p.pattern for p, _ in _SHORT_SYNONYMS}
            result["short_synonyms"] = [
                (p.pattern, r)
                for p, r in syn_inst._synonyms
                if p.pattern not in default_syn_patterns
            ]

            # O10 — imperative-rewrite patterns beyond built-in list
            imp_inst: ImperativeRewrite = fitted["O10"]
            default_imp_patterns = {p.pattern for p, _ in _IMPERATIVE_PATTERNS}
            result["imperative_patterns"] = [
                p.pattern
                for p, _ in imp_inst._patterns
                if p.pattern not in default_imp_patterns
            ]

            header_inst: RemovePromptHeader = fitted["O15"]
            default_header_patterns = {_HEADER_LINE.pattern}
            result["header_patterns"] = [
                p.pattern for p in header_inst._patterns
                if p.pattern not in default_header_patterns
            ]
            section_inst: SectionPruning = fitted["O18"]
            result["tag_names"] = list(section_inst._tag_names)
            chunk_inst: ChunkCompression = fitted["O28"]
            result["chunk_separator"] = chunk_inst._separator.pattern

        return result

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

    def _count_fewshot_paragraphs(self) -> int:
        """Count blank-line-separated paragraphs that form the few-shot prefix.

        Uses the first corpus sample to find where the 0-shot question starts
        inside the full prompt.  Everything before that point is the few-shot
        block; its paragraph count is returned so :class:`ContextPruning` can
        protect it from pruning.

        Returns 0 when the effective template field is not set (either because
        ``template_field`` was not supplied, or because ``compress_field``
        equals ``template_field`` meaning there is no prefix to protect), or
        when the 0-shot text cannot be located inside the full prompt.
        """
        if self._effective_template_field is None or not self._corpus:
            return 0
        sample = self._corpus[0]
        full = str(sample.get(self._prompt_field, ""))
        question_only = str(sample.get(self._effective_template_field, ""))
        if not question_only:
            return 0
        idx = full.find(question_only)
        if idx <= 0:
            return 0
        prefix = full[:idx]
        return len([p for p in re.split(r"\n{2,}", prefix) if p.strip()])