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
        return HFLM(pretrained=model_name, **kwargs)
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
        When set, each sample also gets a ``template_field`` key containing
        the 0-shot prompt (i.e. ``fewshot_context(doc, 0)``).  Pass this
        field name as ``compress_field`` in :class:`ExperimentConfig` to
        compress the raw template instead of the full formatted prompt.
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
    ) -> None:
        self._model = model
        self._task = task
        self._gen_kwargs = generation_kwargs or {
            "until": ["\n", "\n\n"],
            "max_gen_toks": 64,
        }

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
            fallbacks are used.

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

        factory = self._make_transformation_factory(use_llm_for_semantic)

        return Experiment(
            config=config,
            dataset=dataset,
            evaluator=self.evaluator,
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