"""pymoo Problem definition for prompt compression."""

from __future__ import annotations

import time
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
    split:
        Label of the data split this problem evaluates on (``"search"``,
        ``"test"``, …); attached to every logged record.
    shared_cache:
        Optional dict shared by the problems of a run, mapping
        ``(method, source_split, sample_id, prompt)`` → ``(score, evaluation_id,
        provider_prompt_tokens)``.  Keyed by method, so every method pays for
        all the evaluations it needs (no method reuses another's results);
        within a method, an identical prompt for the same sample is never
        evaluated twice (e.g. across its search and test phases).  Used only
        when every sample carries ``_sample_id`` and ``_split``.
    cache_namespace:
        Method that owns this problem's evaluations.  When ``None``, the
        recorder's current ``method`` context is used.
    preserve_answer_cue:
        When ``True``, text-level baselines (:meth:`score_transform_fn`) act
        only on instruction + shots + data and always keep the answer cue and
        tail (e.g. MBPP's code opener) unchanged, as the operator strategies do.
    recorder:
        Optional :class:`~prompt_piper.optimization.recorder.RunRecorder`.
        When set, every evaluation (including cache hits) and every scored
        candidate is logged with its per-sample breakdown.
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
        split: str = "search",
        recorder=None,
        shared_cache: dict | None = None,
        cache_namespace: str | None = None,
        preserve_answer_cue: bool = False,
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

        # Cross-generation caches: populated on first evaluation, reused on
        # subsequent calls.  Keys use (sample_index, compressed_text) so that
        # the same (sample, compression) pair is never sent to the LLM twice.
        self._eval_cache: dict[tuple[int, str], float] = {}
        self._token_cache: dict[str, int] = {}

        # Logging
        self.split = split
        self.recorder = recorder
        #: Stable identifier of each sample in its source split (falls back to
        #: the position in *dataset* when samples carry no ``_sample_id``).
        self._sample_ids: list = [
            s.get("_sample_id", j) if isinstance(s, dict) else j
            for j, s in enumerate(self._dataset)
        ]
        self._eval_ids: dict[tuple[int, str], int | None] = {}
        #: Provider-reported prompt tokens per evaluated pair (API / llama.cpp
        #: models); ``None`` when the backend reports no usage.
        self._provider_tokens: dict[tuple[int, str], int | None] = {}
        if recorder is not None:
            self._log_sections()
        self._call_counters: dict[str, int] = {}
        self._owned_pairs: set[tuple[int, str]] = set()
        has_ids = bool(self._dataset) and all(
            isinstance(s, dict) and "_sample_id" in s and "_split" in s for s in self._dataset
        )
        self._shared = shared_cache if has_ids else None
        self._cache_namespace = cache_namespace
        self.preserve_answer_cue = preserve_answer_cue

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
        """Evaluate all solutions in a generation with cross-generation caching.

        Two layers of deduplication are applied:

        1. **Cross-generation cache** (``_eval_cache``): any (sample_index,
           compressed_text) pair whose score is already known is skipped
           entirely.  This eliminates redundant LLM calls across generations
           when solutions converge to the same compressed output.
        2. **Within-generation deduplication**: novel pairs are collected once
           and sent to the evaluator in a single batched call.
        """
        per_solution = [self._compress_bits(bits) for bits in x]
        infos = [{"bits": bits} for bits in x]
        objectives = self._score_candidates(per_solution, infos)

        F = np.empty((len(x), 2), dtype=float)
        for i, (avg_tokens, avg_perf) in enumerate(objectives):
            F[i, 0] = avg_tokens
            F[i, 1] = -avg_perf
        out["F"] = F

    # ------------------------------------------------------------------
    # Public scoring
    # ------------------------------------------------------------------

    def score_solution(self, bits: np.ndarray, label: str | None = None) -> tuple[float, float]:
        """Return ``(avg_tokens, avg_performance)`` for a binary vector.

        Performance is *positive* here (not negated).  *label* is attached to
        the logged candidate record (e.g. ``"pareto:3"``).
        """
        return self._score_bits(bits, label=label)

    def score_transform_fn(self, fn, label: str | None = None) -> tuple[float, float]:
        """Score an arbitrary compression callable on the dataset.

        Parameters
        ----------
        fn:
            Any callable ``(text: str) -> str``.
        label:
            Optional label attached to the logged candidate record.

        Returns
        -------
        tuple[float, float]
            ``(avg_tokens, avg_performance)`` — performance is *positive*.
        """
        if self._sections is not None:
            originals = [s.assemble() for s in self._sections]
        else:
            originals = [s[self._compress_field] for s in self._dataset]

        if self.preserve_answer_cue and self._sections is not None:
            compressed_texts = [s.with_cue(fn(s.body())) for s in self._sections]
        else:
            compressed_texts = [fn(t) for t in originals]
        info = {"bits": None, "transform": getattr(fn, "__qualname__", repr(fn)), "label": label}
        return self._score_candidates([(originals, compressed_texts)], [info])[0]

    def reference_tokens(self) -> float:
        """Average token count of the uncompressed prompts (no evaluation)."""
        originals, _ = self._compress_bits(np.zeros(self.n_var, dtype=int))
        return sum(self._count_tokens(t) for t in originals) / max(len(originals), 1)

    def log_state(self, kind: str, payload: dict) -> None:
        """Log an algorithm-state snapshot (e.g. MAB beliefs) if recording.

        Front entries get the ``candidate_id`` of the candidate record that
        holds their per-sample values (the averages here are the objectives).
        """
        if self.recorder is not None:
            method = self.recorder.context.get("method")
            for entry in payload.get("pareto_front", []):
                entry["candidate_id"] = self.recorder.candidate_id(method, self.split, entry["bits"])
            if kind == "mab_round":
                payload = {**payload, "candidate_id": self.recorder.candidate_id(method, self.split, payload["bits"])}
            self.recorder.write("states", {
                **self.recorder.context, "split": self.split, "kind": kind, **payload,
            })

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Token count with a simple string-keyed cache."""
        if text not in self._token_cache:
            self._token_cache[text] = self._tokenizer.count(text)
        return self._token_cache[text]

    def _score_bits(self, bits, label: str | None = None) -> tuple[float, float]:
        return self._score_candidates(
            [self._compress_bits(bits)], [{"bits": bits, "label": label}]
        )[0]

    def _score_candidates(
        self,
        per_candidate: list[tuple[list[str], list[str]]],
        infos: list[dict],
    ) -> list[tuple[float, float]]:
        """Score candidates with cross-call caching and complete logging.

        Two layers of deduplication are applied:

        1. **Cross-call cache** (``_eval_cache``): any (sample_index,
           compressed_text) pair whose score is already known is skipped
           entirely.  This eliminates redundant LLM calls across generations
           when solutions converge to the same compressed output.
        2. **Within-call deduplication**: novel pairs are collected once and
           sent to the evaluator in a single batched call.

        Returns ``[(avg_tokens, avg_performance), ...]`` (performance positive).
        """
        rec = self.recorder
        method = rec.context.get("method", "unknown") if rec is not None else "unknown"
        call_index = self._call_counters.get(method, 0) + 1
        self._call_counters[method] = call_index
        t_call = time.perf_counter()

        # Collect pairs not already in the cross-call cache
        key_to_new_idx: dict[tuple, int] = {}
        new_keys: list[tuple[int, str]] = []
        new_originals: list[str] = []
        new_docs: list[dict] = []
        for originals, compressed in per_candidate:
            for j, (orig, comp) in enumerate(zip(originals, compressed)):
                key = (j, comp)
                if key not in self._eval_cache and self._shared is not None:
                    hit = self._shared.get(self._shared_key(j, comp))
                    if hit is not None:
                        self._eval_cache[key], self._eval_ids[key], self._provider_tokens[key] = hit
                if key not in self._eval_cache and key not in key_to_new_idx:
                    key_to_new_idx[key] = len(new_keys)
                    new_keys.append(key)
                    new_originals.append(orig)
                    new_docs.append(self._dataset[j] if j < len(self._dataset) else {})

        # Evaluate novel pairs, log them, and populate the cache
        if new_keys:
            t0 = time.perf_counter()
            details = self._evaluator.evaluate_batch_detailed(
                new_originals, [c for _, c in new_keys], new_docs
            )
            batch_elapsed = time.perf_counter() - t0
            batch_id = rec.next_id("batches") if rec is not None else None
            for (key, orig, doc), detail in zip(zip(new_keys, new_originals, new_docs), details):
                self._eval_cache[key] = float(detail["score"])
                usage = detail.get("provider_usage") or {}
                self._provider_tokens[key] = usage.get("prompt_tokens")
                self._eval_ids[key] = self._log_evaluation(
                    key, orig, doc, detail, batch_id, len(new_keys), batch_elapsed,
                ) if rec is not None else None
                if self._shared is not None:
                    self._shared[self._shared_key(*key)] = (
                        self._eval_cache[key], self._eval_ids[key], self._provider_tokens[key],
                    )

        # Reconstruct per-candidate objectives entirely from cache
        results: list[tuple[float, float]] = []
        for index, ((originals, compressed), info) in enumerate(zip(per_candidate, infos)):
            n = max(len(originals), 1)
            tokens = [self._count_tokens(c) for c in compressed]
            perfs = [self._eval_cache[(j, comp)] for j, comp in enumerate(compressed)]
            avg_tokens, avg_perf = sum(tokens) / n, sum(perfs) / n
            results.append((avg_tokens, avg_perf))
            if rec is not None:
                self._log_candidate(
                    info, call_index, index, compressed, tokens, perfs, key_to_new_idx,
                    avg_tokens, avg_perf, time.perf_counter() - t_call,
                )
        return results

    def _shared_key(self, j: int, comp: str) -> tuple:
        namespace = self._cache_namespace
        if namespace is None and self.recorder is not None:
            namespace = self.recorder.context.get("method")
        return (namespace, self._dataset[j]["_split"], self._dataset[j]["_sample_id"], comp)

    def _log_evaluation(self, key, original, doc, detail, batch_id, batch_size, batch_elapsed) -> int:
        rec = self.recorder
        j, comp = key
        eval_id = rec.next_id("evaluations")
        record = {
            **rec.context,
            "id": eval_id,
            "split": self.split,
            "sample_index": j,
            "sample_id": self._sample_ids[j] if j < len(self._sample_ids) else j,
            "target": doc.get("target") if isinstance(doc, dict) else None,
            "original_sha": rec.store_text(original),
            "prompt_sha": rec.store_text(comp),
            "original_tokens": self._count_tokens(original),
            "input_tokens": self._count_tokens(comp),
            "batch_id": batch_id,
            "batch_size": batch_size,
            "batch_elapsed_seconds": batch_elapsed,
        }
        for k, v in detail.items():
            record[k] = v
        output = detail.get("output")
        if isinstance(output, str):
            record["output_tokens"] = self._tokenizer.count(output)
        rec.write("evaluations", record)
        return eval_id

    def _log_candidate(self, info, call_index, index, compressed, tokens, perfs,
                       new_keys, avg_tokens, avg_perf, elapsed) -> None:
        rec = self.recorder
        bits = info.get("bits")
        per_sample = []
        n_hits = 0
        for j, (comp, tok, perf) in enumerate(zip(compressed, tokens, perfs)):
            key = (j, comp)
            # The first candidate using a pair evaluated in this call "owns"
            # the evaluation; every other use (earlier calls, or duplicates
            # within this call) is a cache hit.
            cached = key not in new_keys or key in self._owned_pairs
            if not cached:
                self._owned_pairs.add(key)
            n_hits += cached
            per_sample.append({
                "sample_index": j,
                "sample_id": self._sample_ids[j] if j < len(self._sample_ids) else j,
                "eval_id": self._eval_ids.get(key),
                "prompt_sha": rec.store_text(comp),
                "tokens": tok,
                "provider_prompt_tokens": self._provider_tokens.get(key),
                "score": perf,
                "cached": cached,
            })
        candidate_id = rec.next_id("candidates")
        if bits is not None:
            rec.register_candidate(rec.context.get("method"), self.split, bits, candidate_id)
        rec.write("candidates", {
            **rec.context,
            "id": candidate_id,
            "split": self.split,
            "call_index": call_index,
            "index_in_call": index,
            "label": info.get("label"),
            "bits": [int(b) for b in bits] if bits is not None else None,
            "transformation_names": self.names_from_bits(bits) if bits is not None else None,
            "transform": info.get("transform"),
            "n_samples": len(compressed),
            "avg_tokens": avg_tokens,
            "avg_performance": avg_perf,
            "n_new_evaluations": len(compressed) - n_hits,
            "n_cache_hits": n_hits,
            "elapsed_seconds_since_call_start": elapsed,
            "per_sample": per_sample,
        })

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

        data_pipeline = [op for op, b in zip(self._data_ops, data_bits) if b]
        compressed_texts: list[str] = []
        originals: list[str] = []
        for section in self._sections:  # type: ignore[union-attr]
            # Each sample's own instruction/shots are compressed; identical
            # texts (the usual static case) hit the cache and are compressed once.
            c_instr, c_shots = self._get_static_compressed(
                instruction_bits, shots_bits, section.instruction, section.shots,
            )
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
        base_instruction: str | None = None,
        base_shots: str | None = None,
    ) -> tuple[str, str]:
        """Return ``(compressed_instruction, compressed_shots)`` from cache.

        Keyed by the bits *and* the texts, so samples whose instruction or shots
        differ are never given another sample's text.
        """
        if base_instruction is None:
            base_instruction = self._sections[0].instruction if self._sections else ""  # type: ignore[index]
        if base_shots is None:
            base_shots = self._sections[0].shots if self._sections else ""  # type: ignore[index]
        static_key = (instruction_bits, shots_bits, base_instruction, base_shots)
        if static_key not in self._static_cache:
            instr_pipeline = [
                op for op, b in zip(self._instruction_ops, instruction_bits) if b
            ]
            shot_pipeline = [
                op for op, b in zip(self._shot_ops, shots_bits) if b
            ]
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

    def effective_bits(self, bits) -> list[int]:
        """Indices of active bits whose removal changes at least one prompt.

        A bit that is set but changes no prompt (on this problem's samples,
        given the other active bits) is neutral: the search cannot have
        selected it for its effect.  Compression only; no model calls.
        """
        bits = [int(b) for b in bits]
        _, base = self._compress_bits(bits)
        effective = []
        for i, b in enumerate(bits):
            if b:
                flipped = list(bits)
                flipped[i] = 0
                if self._compress_bits(flipped)[1] != base:
                    effective.append(i)
        return effective

    def noop_bits(self) -> list[int]:
        """Indices of bits that change no prompt when applied alone."""
        _, base = self._compress_bits([0] * self.n_var)
        out = []
        for i in range(self.n_var):
            bits = [0] * self.n_var
            bits[i] = 1
            if self._compress_bits(bits)[1] == base:
                out.append(i)
        return out

    def bit_layout(self) -> list[dict]:
        """Return the bit layout: one ``{index, section, op_id, op_name}`` per bit."""
        from .bits import make_flat_layout, make_layout

        if self._sections is None:
            return make_flat_layout(self._transforms)
        return make_layout(self._instruction_ops, self._shot_ops, self._data_ops)

    def _log_sections(self) -> None:
        """Log each sample's uncompressed sections once per split and sample."""
        rec = self.recorder
        logged = rec.__dict__.setdefault("_logged_sections", set())
        for j, sample in enumerate(self._dataset):
            key = (self.split, self._sample_ids[j])
            if key in logged:
                continue
            logged.add(key)
            record = {"split": self.split, "sample_index": j, "sample_id": self._sample_ids[j]}
            if self._sections is not None:
                sec = self._sections[j]
                record.update(
                    instruction=sec.instruction, shots=sec.shots,
                    input_data=sec.input_data, output_cue=sec.output_cue,
                    sep_after_instruction=sec.sep_after_instruction,
                    sep_after_shots=sec.sep_after_shots,
                    sep_before_cue=sec.sep_before_cue, tail=sec.tail,
                    assembled_sha=rec.store_text(sec.assemble()),
                )
            else:
                record["compress_field_sha"] = rec.store_text(sample[self._compress_field])
            rec.write("sections", record)

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