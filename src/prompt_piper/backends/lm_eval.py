"""LM Evaluation Harness backend.

This module wires lm-eval into prompt_piper in three ways:

1. **Dataset** (:class:`LMEvalDataset`) — extracts prompts and targets from
   any lm-eval task, producing ``{"prompt": ..., "target": ..., "_doc": ...}``
   dicts compatible with :class:`~prompt_piper.optimization.runner.Experiment`.

2. **Evaluator** (:class:`LMEvalEvaluator`) — scores a compressed prompt by
   running the model through the task's native metric (loglikelihood for
   multiple-choice tasks, generation + task metric for open-ended tasks).

3. **Semantic backend** (:class:`LMEvalSemanticBackend`) — exposes the model
   as a ``(text, instruction) -> str`` callable so that semantic
   transformations O23–O28 can delegate to a real LLM instead of the default
   heuristic fallbacks.

The :class:`LMEvalBackend` facade bundles all three and provides a
``build_experiment()`` convenience method.

Requirements
------------
``pip install lm-eval``   (tested against lm-eval >= 0.4)
"""

from __future__ import annotations

import logging
from typing import Callable, Iterator

import numpy as np

from prompt_piper.optimization.evaluator import Evaluator
from prompt_piper.optimization.tokenizer import Tokenizer, WhitespaceTokenizer
from prompt_piper.transformations.base import Transformation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _require_lm_eval():
    try:
        import lm_eval
        return lm_eval
    except ImportError as exc:
        raise ImportError(
            "LM Evaluation Harness is required: pip install lm-eval"
        ) from exc


def _load_model(model_type: str, model_name: str, **kwargs):
    """Load an lm-eval model by type and name."""
    _require_lm_eval()
    if model_type in ("hf", "huggingface"):
        from lm_eval.models.huggingface import HFLM
        quantization_config = kwargs.pop("quantization_config", None)
        if quantization_config is not None:
            # lm-eval bug: HFLM.__init__ reads quantization_config from the HF
            # model config and passes it explicitly to _create_model(**kwargs),
            # so a user-supplied quantization_config in **kwargs ends up
            # duplicated. Workaround: pre-load the model and hand the object to
            # HFLM, which skips _create_model when pretrained is not a string.
            from transformers import AutoModelForCausalLM
            pretrained = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=kwargs.pop("device_map", "auto"),
                trust_remote_code=kwargs.get("trust_remote_code", False),
                revision=kwargs.get("revision", "main"),
                torch_dtype=kwargs.get("dtype", "auto"),
            )
        else:
            pretrained = model_name
        return HFLM(pretrained=pretrained, **kwargs)
    if model_type == "openai":
        from lm_eval.models.openai_completions import OpenAICompletionsLM
        return OpenAICompletionsLM(model=model_name, **kwargs)
    if model_type == "vllm":
        from lm_eval.models.vllm_causallms import VLLM
        return VLLM(pretrained=model_name, **kwargs)
    raise ValueError(
        f"Unknown model type {model_type!r}. "
        "Supported: 'hf'/'huggingface', 'openai', 'vllm'."
    )


def _load_task(task_name: str, num_fewshot: int = 0):
    """Return an initialised lm-eval task object."""
    _require_lm_eval()
    from lm_eval.tasks import TaskManager

    tm = TaskManager()
    # lm-eval ≥ 0.4 API
    task_dict = tm.load_task_or_group([task_name])
    task = task_dict[task_name]
    # Some tasks need an explicit fewshot count at build time.
    if hasattr(task, "set_fewshot_seed"):
        task.set_fewshot_seed(1234)
    return task


def _make_instance(request_type: str, doc, arguments, idx: int = 0):
    """Create an lm-eval Instance, handling API differences across versions."""
    try:
        from lm_eval.api.instance import Instance
        return Instance(
            request_type=request_type,
            doc=doc,
            arguments=arguments,
            idx=idx,
        )
    except ImportError:
        # Older lm-eval versions that don't have Instance; return raw tuple.
        return arguments


def _call_model(model, method: str, requests: list):
    """Call ``model.loglikelihood`` or ``model.generate_until``, handling
    both Instance-based (lm-eval >= 0.4) and tuple-based (older) APIs."""
    fn = getattr(model, method)
    return fn(requests)


# ---------------------------------------------------------------------------
# 0. Model-backed tokenizer
# ---------------------------------------------------------------------------

class LMEvalTokenizer(Tokenizer):
    """Count tokens using the same tokenizer the model uses.

    Wraps the underlying tokenizer exposed by the lm-eval model object so that
    token-count objectives reflect the model's actual vocabulary rather than a
    whitespace approximation.

    Detection order
    ---------------
    1. **``model.tokenizer`` is a tokenizer object** — covers HuggingFace
       (``HFLM``), vLLM, OpenAI via lm-eval (uses tiktoken internally), and
       Anthropic Claude via lm-eval (uses ``client.get_tokenizer()``).
       Anything whose ``.encode(text)`` returns a sequence of tokens is handled
       here.  Plain Python ``str`` objects (which also have ``.encode()``) are
       excluded explicitly.
    2. **``model.encoding_name`` / ``model.tokenizer_name``** — tiktoken
       encoding name exposed by some lm-eval OpenAI wrappers.  ``tiktoken``
       must be installed (``pip install tiktoken``).
    3. **``model.tokenizer`` is a string** (model or encoding name) — covers
       self-hosted OpenAI-compatible servers (``LocalCompletionsAPI``).
       Tries tiktoken first, then falls back to loading a HuggingFace
       ``AutoTokenizer`` by that name (requires ``transformers``).
    4. **Fallback** — :class:`~prompt_piper.optimization.tokenizer.WhitespaceTokenizer`
       with a logged warning.

    Parameters
    ----------
    model:
        An lm-eval LM object (``HFLM``, ``OpenAICompletionsAPI``,
        ``LocalCompletionsAPI``, ``AnthropicLM``, ``VLLM``, …).

    Notes
    -----
    *Anthropic SDK deprecation*: ``client.get_tokenizer()`` was removed in
    Anthropic SDK ≥ 0.20.  If ``AnthropicLM`` fails to initialise because of
    this, lm-eval itself will error before ``LMEvalTokenizer`` is ever called.
    When using a recent SDK and a Claude model, pass an explicit ``tokenizer``
    to :meth:`LMEvalBackend.build_experiment` — e.g. a
    :class:`~prompt_piper.optimization.tokenizer.TiktokenTokenizer` with
    ``"cl100k_base"`` as a reasonable approximation.

    *Self-hosted models*: for models accessed via an OpenAI-compatible API
    whose name cannot be resolved to a tiktoken encoding (e.g. ``"llama3"``),
    detection falls through to :class:`WhitespaceTokenizer`.  Pass an explicit
    tokenizer or load the HuggingFace tokenizer manually in that case.
    """

    def __init__(self, model) -> None:
        self._counter = self._build_counter(model)

    # ------------------------------------------------------------------
    # Tokenizer interface
    # ------------------------------------------------------------------

    def count(self, text: str) -> int:
        return self._counter(text)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_counter(model):
        """Return a callable ``(text: str) -> int`` for the given model."""
        tok_attr = getattr(model, "tokenizer", None)

        # ---- 1. tokenizer object with .encode() -------------------------
        # Explicitly exclude plain strings — str.encode() returns bytes, not
        # tokens, so len(str.encode(text)) would silently be a byte count.
        if tok_attr is not None and not isinstance(tok_attr, str):
            encode_fn = getattr(tok_attr, "encode", None)
            if callable(encode_fn):
                def _obj_count(text: str, _enc=encode_fn) -> int:
                    return len(_enc(text))
                return _obj_count

        # ---- 2. tiktoken via encoding_name / tokenizer_name -------------
        # Some lm-eval OpenAI wrappers set model.encoding_name or
        # model.tokenizer_name to a tiktoken-resolvable string.
        try:
            import tiktoken  # noqa: PLC0415
            enc_name = (
                getattr(model, "encoding_name", None)
                or getattr(model, "tokenizer_name", None)
            )
            if enc_name and isinstance(enc_name, str):
                try:
                    enc = tiktoken.encoding_for_model(enc_name)
                except KeyError:
                    enc = tiktoken.get_encoding(enc_name)
                def _tiktoken_count(text: str, _enc=enc) -> int:
                    return len(_enc.encode(text))
                return _tiktoken_count
        except ImportError:
            pass

        # ---- 3. model.tokenizer is a string (e.g. LocalCompletionsAPI) --
        # Try to resolve the model/encoding name to a real tokenizer.
        if isinstance(tok_attr, str) and tok_attr:
            # 3a. tiktoken — works for OpenAI model names and encoding names.
            try:
                import tiktoken  # noqa: PLC0415
                try:
                    enc = tiktoken.encoding_for_model(tok_attr)
                except KeyError:
                    try:
                        enc = tiktoken.get_encoding(tok_attr)
                    except ValueError:
                        enc = None
                if enc is not None:
                    def _name_tiktoken_count(text: str, _enc=enc) -> int:
                        return len(_enc.encode(text))
                    return _name_tiktoken_count
            except ImportError:
                pass

            # 3b. HuggingFace AutoTokenizer — works for HF model names used
            # as the tokenizer string (e.g. "mistralai/Mistral-7B-v0.1").
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415
                hf_tok = AutoTokenizer.from_pretrained(tok_attr)
                logger.info(
                    "LMEvalTokenizer: loaded HuggingFace tokenizer %r from model name.",
                    tok_attr,
                )
                def _hf_name_count(text: str, _tok=hf_tok) -> int:
                    return len(_tok.encode(text))
                return _hf_name_count
            except Exception:
                pass

        # ---- 4. Fallback: whitespace ------------------------------------
        logger.warning(
            "LMEvalTokenizer: could not resolve a tokenizer for model %r; "
            "falling back to whitespace splitting.  For accurate token counts "
            "pass an explicit tokenizer= to build_experiment().",
            type(model).__name__,
        )
        return WhitespaceTokenizer().count


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

class LMEvalDataset:
    """Extract samples from an lm-eval task.

    Each sample is a dict with:

    * ``"prompt"`` — the fully formatted context string (with few-shot
      examples if ``num_fewshot > 0``).
    * ``"target"`` — the expected output / label.
    * ``"_doc"`` — the raw lm-eval document (needed by
      :class:`LMEvalEvaluator`).

    Parameters
    ----------
    task:
        An initialised lm-eval task object.
    split:
        Which split to use: ``"test"``, ``"validation"``/``"val"``,
        or ``"train"``.
    num_fewshot:
        Number of few-shot examples to prepend.
    limit:
        Cap the number of samples returned (``None`` = all).
    prompt_field:
        Key used for the full (few-shot) prompt in the output dict.
    template_field:
        When set, each sample also gets an entry under this key containing
        the 0-shot prompt (i.e. ``fewshot_context(doc, 0)`` — the prompt
        with data already substituted but no few-shot examples prepended).
        Use ``"prompt_0shot"`` as the conventional name and pass it as
        ``compress_field`` in :class:`ExperimentConfig` to compress that
        field instead of the full formatted prompt.
    """

    def __init__(
        self,
        task,
        split: str = "test",
        num_fewshot: int = 0,
        limit: int | None = None,
        prompt_field: str = "prompt",
        template_field: str | None = None,
    ) -> None:
        self._task = task
        self._split = split
        self._num_fewshot = num_fewshot
        self._limit = limit
        self._prompt_field = prompt_field
        self._template_field = template_field

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        docs = list(self._get_docs())
        if self._limit is not None:
            docs = docs[: self._limit]
        for doc in docs:
            yield self._doc_to_sample(doc)

    def __len__(self) -> int:
        docs = list(self._get_docs())
        if self._limit is not None:
            return min(len(docs), self._limit)
        return len(docs)

    def as_list(self) -> list[dict]:
        return list(self)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_task_name(
        cls,
        task_name: str,
        split: str = "test",
        num_fewshot: int = 0,
        limit: int | None = None,
        prompt_field: str = "prompt",
        template_field: str | None = None,
    ) -> "LMEvalDataset":
        """Convenience constructor — loads the task by name."""
        task = _load_task(task_name, num_fewshot=num_fewshot)
        return cls(task, split=split, num_fewshot=num_fewshot, limit=limit,
                   prompt_field=prompt_field, template_field=template_field)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_docs(self):
        split = self._split.lower()
        if split == "test":
            return self._task.test_docs()
        if split in ("validation", "val"):
            return self._task.validation_docs()
        if split == "train":
            return self._task.training_docs()
        raise ValueError(f"Unknown split {self._split!r}")

    def _doc_to_sample(self, doc: dict) -> dict:
        # fewshot_context signature varies; handle both common forms.
        try:
            prompt = self._task.fewshot_context(doc, self._num_fewshot)
        except TypeError:
            prompt = self._task.fewshot_context(
                doc, self._num_fewshot, rnd=None, description=None
            )
        target = self._task.doc_to_target(doc)
        sample = {
            self._prompt_field: prompt,
            "target": target,
            "_doc": doc,
        }
        if self._template_field is not None:
            try:
                template = self._task.fewshot_context(doc, 0)
            except TypeError:
                template = self._task.fewshot_context(doc, 0, rnd=None, description=None)
            sample[self._template_field] = template
        return sample


# ---------------------------------------------------------------------------
# 2. Evaluator
# ---------------------------------------------------------------------------

class LMEvalEvaluator(Evaluator):
    """Score a compressed prompt using an lm-eval model and task.

    Dispatches to the correct evaluation strategy based on the task's
    ``OUTPUT_TYPE``:

    * ``"multiple_choice"`` / ``"loglikelihood"`` — compare per-choice
      log-probabilities; return 1.0 if the predicted choice matches the gold
      label, 0.0 otherwise.
    * ``"generate_until"`` — generate text, then delegate to the task's
      ``process_results`` metric (returns the first numeric metric value).

    Parameters
    ----------
    model:
        An lm-eval LM object (e.g. ``HFLM``).
    task:
        The lm-eval task object corresponding to the benchmark.
    generation_kwargs:
        Extra kwargs for generation tasks (``until``, ``max_gen_toks``, …).
        Defaults to a reasonable stopping condition.
    """

    def __init__(
        self,
        model,
        task,
        generation_kwargs: dict | None = None,
        extra_stop_sequences: list[str] | None = None,
    ) -> None:
        self._model = model
        self._task = task
        if generation_kwargs is not None:
            self._gen_kwargs = generation_kwargs
        else:
            # Use the task's own generation_kwargs (e.g. GSM8K ships with
            # until=["Question:", "</s>"] which is needed for CoT generation).
            # Fall back to a conservative newline stop only when the task has
            # no config (e.g. legacy tasks or mocks).
            cfg = getattr(task, "config", None)
            task_kwargs = dict(getattr(cfg, "generation_kwargs", None) or {})
            task_kwargs.setdefault("max_gen_toks", 256)
            if not task_kwargs.get("until"):
                task_kwargs["until"] = ["\n", "\n\n"]
            self._gen_kwargs = task_kwargs

        # Merge model-specific stop tokens (e.g. "<|eot_id|>" for Llama-3)
        # into the until list without disturbing other generation kwargs.
        if extra_stop_sequences:
            until = list(self._gen_kwargs.get("until", []))
            for seq in extra_stop_sequences:
                if seq not in until:
                    until.append(seq)
            self._gen_kwargs = {**self._gen_kwargs, "until": until}

    def evaluate(self, original: str, compressed: str, sample: dict) -> float:
        doc = sample.get("_doc")
        if doc is None:
            logger.warning(
                "LMEvalEvaluator: sample missing '_doc' key; "
                "returning 0.0.  Use LMEvalDataset to build samples."
            )
            return 0.0

        output_type = getattr(self._task, "OUTPUT_TYPE", "generate_until")

        if output_type in ("multiple_choice", "loglikelihood"):
            return self._eval_multiple_choice(compressed, doc)
        return self._eval_generation(compressed, doc)

    def evaluate_batch(
        self,
        originals: list[str],
        compresseds: list[str],
        samples: list[dict],
    ) -> list[float]:
        """Score a batch of compressed prompts in a single model call."""
        if not samples:
            return []

        output_type = getattr(self._task, "OUTPUT_TYPE", "generate_until")
        if output_type in ("multiple_choice", "loglikelihood"):
            return self._eval_multiple_choice_batch(compresseds, samples)
        return self._eval_generation_batch(compresseds, samples)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _eval_multiple_choice(self, prompt: str, doc) -> float:
        """Loglikelihood-based scoring for MC tasks."""
        try:
            choices = self._task.doc_to_choice(doc)
        except AttributeError:
            # Task doesn't expose choices; fall back to generation.
            return self._eval_generation(prompt, doc)

        requests = [
            _make_instance("loglikelihood", doc, (prompt, choice), i)
            for i, choice in enumerate(choices)
        ]
        results = _call_model(self._model, "loglikelihood", requests)
        log_probs = [r[0] if isinstance(r, (tuple, list)) else r for r in results]
        predicted = int(np.argmax(log_probs))

        gold = self._task.doc_to_target(doc)
        # Gold may be an int (index) or a string (choice text).
        if isinstance(gold, int):
            return 1.0 if predicted == gold else 0.0
        # String: compare to the predicted choice text.
        return 1.0 if choices[predicted] == gold else 0.0

    def _eval_multiple_choice_batch(
        self, prompts: list[str], samples: list[dict]
    ) -> list[float]:
        """Batched loglikelihood scoring: one model call for all samples."""
        first_doc = samples[0].get("_doc")
        try:
            self._task.doc_to_choice(first_doc)
        except AttributeError:
            return self._eval_generation_batch(prompts, samples)

        # Empty prompts (fully compressed away) cannot be tokenised; score 0.
        scores: list[float] = [0.0] * len(prompts)
        valid = [(i, p, s) for i, (p, s) in enumerate(zip(prompts, samples)) if p.strip()]
        if not valid:
            return scores

        requests = []
        choices_counts: list[int] = []
        valid_indices: list[int] = []
        for orig_i, prompt, sample in valid:
            doc = sample.get("_doc")
            choices = self._task.doc_to_choice(doc)
            choices_counts.append(len(choices))
            valid_indices.append(orig_i)
            for i, choice in enumerate(choices):
                requests.append(_make_instance("loglikelihood", doc, (prompt, choice), i))

        results = _call_model(self._model, "loglikelihood", requests)

        idx = 0
        for orig_i, n_choices, (_, _, sample) in zip(
            valid_indices, choices_counts, valid
        ):
            chunk = results[idx : idx + n_choices]
            idx += n_choices
            log_probs = [r[0] if isinstance(r, (tuple, list)) else r for r in chunk]
            predicted = int(np.argmax(log_probs))
            doc = sample.get("_doc")
            gold = self._task.doc_to_target(doc)
            choices = self._task.doc_to_choice(doc)
            if isinstance(gold, int):
                scores[orig_i] = 1.0 if predicted == gold else 0.0
            else:
                scores[orig_i] = 1.0 if choices[predicted] == gold else 0.0
        return scores

    def _eval_generation(self, prompt: str, doc) -> float:
        """Generation-based scoring for open-ended tasks."""
        instance = _make_instance("generate_until", doc, (prompt, self._gen_kwargs), 0)
        generated = _call_model(self._model, "generate_until", [instance])[0]

        try:
            metrics: dict = self._task.process_results(doc, [generated])
            # Return the first numeric metric value, normalised to [0, 1].
            value = float(next(iter(metrics.values())))
            return max(0.0, min(1.0, value))
        except Exception as exc:
            logger.warning("LMEvalEvaluator: process_results failed (%s); returning 0.0", exc)
            return 0.0

    def _eval_generation_batch(
        self, prompts: list[str], samples: list[dict]
    ) -> list[float]:
        """Batched generation scoring: one model call for all samples."""
        # Empty prompts (fully compressed away) produce a zero-length token
        # tensor, causing an IndexError in HFLM._model_generate when it
        # accesses attention_mask[:, -1].  Skip them and return 0.0.
        scores: list[float] = [0.0] * len(prompts)
        valid = [(i, p, s) for i, (p, s) in enumerate(zip(prompts, samples)) if p.strip()]
        if not valid:
            return scores

        requests = [
            _make_instance(
                "generate_until",
                sample.get("_doc", {}),
                (prompt, self._gen_kwargs),
                i,
            )
            for i, (_, prompt, sample) in enumerate(valid)
        ]
        generated_texts = _call_model(self._model, "generate_until", requests)

        for (orig_i, _, sample), generated in zip(valid, generated_texts):
            doc = sample.get("_doc")
            try:
                metrics: dict = self._task.process_results(doc, [generated])
                value = float(next(iter(metrics.values())))
                scores[orig_i] = max(0.0, min(1.0, value))
            except Exception as exc:
                logger.warning(
                    "LMEvalEvaluator: process_results failed (%s); returning 0.0", exc
                )
        return scores


# ---------------------------------------------------------------------------
# 3. Semantic backend
# ---------------------------------------------------------------------------

class LMEvalSemanticBackend:
    """Use an lm-eval model as the ``backend`` for semantic transformations.

    Passed to O23–O28 transformations as their ``backend`` argument::

        from prompt_piper.transformations import ContextCompression
        backend = LMEvalSemanticBackend(model)
        op = ContextCompression(backend=backend)
        compressed = op("A long context paragraph...")

    Parameters
    ----------
    model:
        An lm-eval LM object.
    max_gen_toks:
        Maximum tokens to generate.
    stop_sequences:
        Token sequences that terminate generation.
    prompt_template:
        Format string with ``{instruction}`` and ``{text}`` placeholders used
        to build the model's input prompt.
    """

    def __init__(
        self,
        model,
        max_gen_toks: int = 256,
        stop_sequences: list[str] | None = None,
        prompt_template: str = "{instruction}\n\n{text}\n\nCompressed:",
    ) -> None:
        self._model = model
        self._max_gen_toks = max_gen_toks
        self._stop_sequences = stop_sequences or ["\n\n"]
        self._template = prompt_template

    def __call__(self, text: str, instruction: str) -> str:
        prompt = self._template.format(instruction=instruction, text=text)
        gen_kwargs = {
            "until": self._stop_sequences,
            "max_gen_toks": self._max_gen_toks,
        }
        instance = _make_instance("generate_until", {}, (prompt, gen_kwargs), 0)
        result = _call_model(self._model, "generate_until", [instance])
        return result[0].strip()


# ---------------------------------------------------------------------------
# 4. Facade
# ---------------------------------------------------------------------------

class LMEvalBackend:
    """Facade that bundles model + task into dataset, evaluator, and backend.

    Parameters
    ----------
    model:
        An lm-eval LM object (see :func:`build` for convenient construction).
    task:
        An initialised lm-eval task object.

    Example
    -------
    >>> backend = LMEvalBackend.build(
    ...     model_type="hf",
    ...     model_name="mistralai/Mistral-7B-v0.1",
    ...     task_name="hellaswag",
    ... )
    >>> config = ExperimentConfig(n_gen=50, pop_size=30, n_subset=200)
    >>> result = backend.build_experiment(config).run(
    ...     baselines=["no_compression", "random_search"]
    ... )
    >>> print(result.summary())
    """

    def __init__(self, model, task) -> None:
        self.model = model
        self.task = task
        self.tokenizer = LMEvalTokenizer(model)
        self.semantic_backend = LMEvalSemanticBackend(model)
        self.evaluator = LMEvalEvaluator(model, task)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model_type: str,
        model_name: str,
        task_name: str,
        num_fewshot: int = 0,
        **model_kwargs,
    ) -> "LMEvalBackend":
        """Load model and task by name and return a ready :class:`LMEvalBackend`.

        Parameters
        ----------
        model_type:
            One of ``"hf"``/``"huggingface"``, ``"openai"``, ``"vllm"``.
        model_name:
            Model identifier (HuggingFace hub name, OpenAI model string, …).
        task_name:
            lm-eval task name, e.g. ``"hellaswag"``, ``"mmlu"``, ``"gsm8k"``.
        num_fewshot:
            Number of few-shot examples.
        **model_kwargs:
            Extra arguments forwarded to the model constructor (e.g.
            ``device="cuda"``, ``dtype="float16"``).
        """
        model = _load_model(model_type, model_name, **model_kwargs)
        task = _load_task(task_name, num_fewshot=num_fewshot)
        return cls(model, task)

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def get_dataset(
        self,
        split: str = "test",
        num_fewshot: int = 0,
        limit: int | None = None,
        prompt_field: str = "prompt",
        template_field: str | None = None,
    ) -> LMEvalDataset:
        """Return a :class:`LMEvalDataset` for the given split."""
        return LMEvalDataset(
            self.task,
            split=split,
            num_fewshot=num_fewshot,
            limit=limit,
            prompt_field=prompt_field,
            template_field=template_field,
        )

    # ------------------------------------------------------------------
    # Experiment builder
    # ------------------------------------------------------------------

    def build_experiment(
        self,
        config,
        split: str = "test",
        num_fewshot: int = 0,
        transformations: list[type[Transformation]] | None = None,
        use_llm_for_semantic: bool = True,
        tokenizer: Tokenizer | None = None,
        transformation_factory=None,
    ):
        """Return a fully configured :class:`~prompt_piper.optimization.runner.Experiment`.

        Parameters
        ----------
        config:
            :class:`~prompt_piper.optimization.config.ExperimentConfig`.
        split:
            Dataset split to use for the optimisation search.
        num_fewshot:
            Few-shot count for prompt formatting.
        transformations:
            Transformation classes to include.  ``None`` = all 28.
        use_llm_for_semantic:
            If ``True`` (default), semantic transformations (O23–O28) are
            instantiated with the LLM backend.  If ``False``, the heuristic
            fallbacks are used.  Ignored when ``transformation_factory`` is
            provided.
        tokenizer:
            :class:`~prompt_piper.optimization.tokenizer.Tokenizer` used to
            count tokens in the compressed prompt.  Defaults to
            :class:`LMEvalTokenizer` which wraps the loaded model's own
            tokenizer (HuggingFace ``PreTrainedTokenizer`` for ``hf``/``vllm``
            models; ``tiktoken`` for OpenAI models).  Pass an explicit
            tokenizer to override, or ``False`` to fall back to the
            :class:`~prompt_piper.optimization.tokenizer.WhitespaceTokenizer`.
        transformation_factory:
            Optional callable ``(cls: type[Transformation]) -> Transformation``
            that overrides the internal factory.  Use this to inject a
            :class:`~prompt_piper.transformations.fitting.DatasetVocabularyFitter`
            factory (combined with LLM backend injection for semantic operators
            as needed).  When ``None`` (default), the internal factory derived
            from ``use_llm_for_semantic`` is used.

        Notes
        -----
        When ``config.compress_field`` differs from ``config.prompt_field``,
        the dataset is built with ``template_field=config.compress_field`` so
        each sample contains the 0-shot prompt under that key.
        """
        from prompt_piper.optimization.runner import Experiment

        template_field = (
            config.compress_field
            if config.compress_field and config.compress_field != config.prompt_field
            else None
        )
        dataset = self.get_dataset(
            split=split,
            num_fewshot=num_fewshot,
            limit=config.n_subset,
            prompt_field=config.prompt_field,
            template_field=template_field,
        ).as_list()

        factory = (
            transformation_factory
            if transformation_factory is not None
            else self._make_transformation_factory(use_llm_for_semantic)
        )

        # Resolve tokenizer: use model-native by default, allow explicit override.
        if tokenizer is False:
            resolved_tokenizer = None   # Experiment will use WhitespaceTokenizer
        elif tokenizer is None:
            resolved_tokenizer = self.tokenizer
        else:
            resolved_tokenizer = tokenizer

        return Experiment(
            config=config,
            dataset=dataset,
            evaluator=self.evaluator,
            tokenizer=resolved_tokenizer,
            transformations=transformations,
            transformation_factory=factory,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_transformation_factory(
        self, use_llm_for_semantic: bool
    ) -> Callable[[type[Transformation]], Transformation]:
        """Return a factory that injects the LLM backend into semantic ops."""
        import inspect
        backend = self.semantic_backend if use_llm_for_semantic else None

        def factory(cls: type[Transformation]) -> Transformation:
            if backend is not None and cls.category == "semantic":
                if "backend" in inspect.signature(cls.__init__).parameters:
                    return cls(backend=backend)
            return cls()

        return factory