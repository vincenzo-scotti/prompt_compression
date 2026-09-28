"""Compressor tool configuration.

Supports both YAML file loading and CLI argument override.

YAML schema example
-------------------
.. code-block:: yaml

    model:
      type: hf                              # hf | openai | openai-chat-completions | openai-chat | vllm | gemini | llama-cpp
      name: mistralai/Mistral-7B-Instruct-v0.3
      kwargs:                               # forwarded to model constructor
        device: cuda
        dtype: float16
        apply_chat_template: true           # set true for instruct/chat models

    output:
      dir: ./results                        # output directory
      overwrite: false                      # overwrite existing result files

    # Settings shared across all runs (overridden per-run)
    defaults:
      num_fewshot: 0
      search_split: validation
      test_split: test
      n_search: 200                         # samples used for optimisation
      n_test: 500                           # samples used for final evaluation
      baselines:
        - no_compression
        - all_transformations
        - random_search
      use_llm_for_semantic: true
      use_fitting: false                    # run DatasetVocabularyFitter before search
      generation_kwargs: null               # null = use task's own generation config
      extra_stop_sequences: null            # e.g. ["<|eot_id|>"] for Llama-3
      algorithm: nsga2
      pop_size: 50
      n_gen: 100
      seed: 42
      subset_seed: 42
      transformation_ids: null              # null = all 28 operators

    runs:
      - name: hellaswag_nsga2
        task: hellaswag

      - name: mmlu_nsga2
        task: mmlu
        n_search: 300
        algorithm: nsga2

      - name: hellaswag_moead
        task: hellaswag
        algorithm: moead
        pop_size: 30
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    type: str = "hf"
    name: str = "gpt2"
    kwargs: dict[str, Any] = field(default_factory=dict)
    #: Optional quantization config expressed as a plain dict so it can be
    #: written in YAML.  Keys mirror the ``transformers.BitsAndBytesConfig``
    #: constructor (e.g. ``load_in_4bit``, ``bnb_4bit_quant_type``).
    #: The runner constructs the real ``BitsAndBytesConfig`` object at load time.
    quantization: dict[str, Any] | None = None


@dataclass
class OutputConfig:
    dir: str = "./results"
    overwrite: bool = False


@dataclass
class RunConfig:
    """Settings for a single (task, algorithm) run."""

    # Required
    task: str = ""
    # Optional — all fall back to CompressorConfig.defaults
    name: str | None = None
    num_fewshot: int = 0
    search_split: str = "validation"
    test_split: str = "test"
    n_search: int = 200
    n_test: int | None = 500
    baselines: list[str] = field(
        default_factory=lambda: [
            "no_compression",
            "all_transformations",
            "random_search",
        ]
    )
    use_llm_for_semantic: bool = True
    #: Run :class:`~prompt_piper.transformations.fitting.DatasetVocabularyFitter`
    #: on the search dataset before optimisation.  Corpus-based patterns (O3,
    #: O5, O17, O18, O22) are always fitted when this is ``True``.  When
    #: ``use_llm_for_semantic`` is also ``True``, the same LLM model is reused
    #: to additionally discover structural patterns (O15, O28; header formats,
    #: tag names, chunk separators).  The fitting report is persisted alongside
    #: the optimisation result.
    use_fitting: bool = False
    algorithm: str = "nsga2"
    pop_size: int = 50
    n_gen: int = 100
    seed: int = 42
    subset_seed: int = 42
    #: Weight of performance in the multi_armed_bandit reward
    #: ``w · performance + (1 − w) · compression`` (``1`` = performance only).
    mab_reward_weight: float = 0.5
    transformation_ids: list[str] | None = None
    #: Restrict the NSGA-II / MAB / random_search operator pool to these IDs.
    #: ``all_transformations`` and ``summarization`` baselines always use the
    #: full ``transformation_ids`` set.  ``None`` (default) = no restriction.
    search_transformation_ids: list[str] | None = None
    prompt_field: str = "prompt"
    #: Field to apply transformations to.  ``None`` means use ``prompt_field``
    #: (i.e. compress the fully formatted prompt).  Set to
    #: ``"prompt_0shot"`` to compress only the 0-shot prompt.
    compress_field: str | None = None
    #: Verbose task-specific instruction prepended to the compress_field text
    #: before the optimisation search.  When set the effective compress target
    #: becomes ``"verbose_instruction"`` (a synthetic field built at dataset
    #: load time as ``instruction_prefix + "\n\n" + <compress_field text>``).
    #: This gives compression operators realistic, verbose material to work on
    #: rather than the minimal benchmark-style prompts that lm-eval generates.
    instruction_prefix: str | None = None
    #: Override generation kwargs passed to :class:`~prompt_piper.backends.lm_eval.LMEvalEvaluator`.
    #: ``None`` (default) reads the task's own ``generation_kwargs`` config, which is
    #: correct for most tasks (e.g. GSM8K ships with ``until: ["Question:", ...]``).
    generation_kwargs: dict | None = None
    #: Model-specific stop tokens appended to the task's ``until`` list.
    #: Use this instead of overriding ``generation_kwargs`` when you only need
    #: to add a model's EOT token (e.g. ``["<|eot_id|>"]`` for Llama-3,
    #: ``["<|im_end|>"]`` for Qwen, ``"<|end|>"`` for Phi-3).
    extra_stop_sequences: list[str] | None = None
    #: Select which lm-eval metric to optimise when ``process_results`` returns
    #: multiple keys (e.g. ``"exact_match,flexible-extract"`` for GSM8K).
    #: ``None`` (default) picks the first key, which matches lm-eval's own
    #: primary-metric convention.
    metric_key: str | None = None
    #: When ``True``, skip the NSGA-II optimisation phase and run only the
    #: baselines listed in ``baselines``.  Useful for dedicated baseline
    #: evaluation runs that reuse the same model session as the main experiment.
    baselines_only: bool = False
    #: Wrap prompts in the model's chat template before evaluation.  Required
    #: for instruct/chat-tuned models (e.g. Qwen-Instruct, Llama-Instruct).
    #: Has no effect on OpenAI chat-completions models (handled by lm-eval).
    apply_chat_template: bool = False
    #: Format few-shot examples as separate conversation turns.  Only
    #: meaningful when ``apply_chat_template`` is also ``True``.
    fewshot_as_multiturn: bool = False
    #: Text-level baselines (truncated_prompt, random_deletion, llmlingua2)
    #: keep the answer cue and tail (e.g. MBPP's code opener) unchanged and act
    #: only on instruction + shots + data.  Default False: they act on the
    #: whole prompt (behaviour of the submitted paper).
    preserve_answer_cue: bool = False

    @property
    def run_name(self) -> str:
        return self.name or f"{self.task}_{self.algorithm}"

    @property
    def effective_compress_field(self) -> str | None:
        """Compression target field, accounting for instruction_prefix injection.

        When ``instruction_prefix`` is set the runner builds a synthetic
        ``"verbose_instruction"`` field and this property returns that name so
        all downstream components (problem, fitter, exp config) use it
        consistently without the caller needing to know about the injection.
        """
        if self.instruction_prefix:
            return "verbose_instruction"
        return self.compress_field


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

_RUN_FIELDS = {f for f in RunConfig.__dataclass_fields__}


@dataclass
class TransferabilityConfig:
    """Configuration for cross-task transferability analysis.

    After all primary runs finish, the runner evaluates Pareto solutions
    from a *source* run on the test dataset of a *target* run, revealing
    whether compression strategies discovered for one task generalise to
    another.

    Parameters
    ----------
    enabled:
        Set ``True`` to activate the analysis.
    pairs:
        List of ``[source_run_name, target_run_name]`` pairs.  Both names
        must appear in the enclosing ``CompressorConfig.runs`` list.
    """

    enabled: bool = False
    pairs: list[list[str]] = field(default_factory=list)


@dataclass
class CompressorConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    defaults: dict[str, Any] = field(default_factory=dict)
    runs: list[RunConfig] = field(default_factory=list)
    transferability: TransferabilityConfig = field(default_factory=TransferabilityConfig)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CompressorConfig":
        """Load configuration from a YAML file."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Install PyYAML: pip install pyyaml") from exc

        path = Path(path)
        with open(path) as f:
            raw: dict = yaml.safe_load(f) or {}

        return cls._from_dict(raw, config_dir=path.parent)

    @classmethod
    def from_dict(cls, d: dict) -> "CompressorConfig":
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, raw: dict, config_dir: Path | None = None) -> "CompressorConfig":
        model_raw = raw.get("model", {})
        model = ModelConfig(
            type=model_raw.get("type", "hf"),
            name=model_raw.get("name", "gpt2"),
            kwargs=model_raw.get("kwargs", {}),
            quantization=model_raw.get("quantization", None),
        )

        output_raw = raw.get("output", {})
        output = OutputConfig(
            dir=output_raw.get("dir", "./results"),
            overwrite=output_raw.get("overwrite", False),
        )

        defaults: dict = raw.get("defaults", {})

        # Load optional task library (shared per-task settings)
        task_library: dict[str, dict] = {}
        task_lib_path = raw.get("task_library")
        if task_lib_path and config_dir is not None:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("Install PyYAML: pip install pyyaml") from exc
            task_lib_full = config_dir / task_lib_path
            with open(task_lib_full) as f:
                lib_raw: dict = yaml.safe_load(f) or {}
            task_library = lib_raw.get("tasks", {})

        runs_raw: list[dict] = raw.get("runs", [])
        runs = [cls._build_run(r, defaults, task_library) for r in runs_raw]

        transfer_raw = raw.get("transferability", {})
        transferability = TransferabilityConfig(
            enabled=transfer_raw.get("enabled", False),
            pairs=transfer_raw.get("pairs", []),
        )

        return cls(model=model, output=output, defaults=defaults, runs=runs,
                   transferability=transferability)

    @staticmethod
    def _build_run(run_raw: dict, defaults: dict, task_library: dict | None = None) -> RunConfig:
        """Merge defaults → task-library → run-level overrides into a RunConfig."""
        merged = copy.deepcopy(defaults)
        # Task-library settings override global defaults but are overridden by per-run keys
        task_name = run_raw.get("task", "")
        if task_library and task_name in task_library:
            merged.update(copy.deepcopy(task_library[task_name]))
        merged.update(run_raw)
        # Keep only known fields
        kwargs = {k: v for k, v in merged.items() if k in _RUN_FIELDS}
        return RunConfig(**kwargs)

    # ------------------------------------------------------------------
    # CLI override
    # ------------------------------------------------------------------

    def apply_cli_overrides(self, overrides: dict[str, Any]) -> "CompressorConfig":
        """Return a new config with CLI overrides applied to all runs.

        Only non-None values in *overrides* are applied.
        """
        filtered = {k: v for k, v in overrides.items() if v is not None}
        if not filtered:
            return self

        new_runs = []
        for run in self.runs:
            run_dict = {f: getattr(run, f) for f in _RUN_FIELDS}
            run_dict.update(filtered)
            new_runs.append(RunConfig(**run_dict))

        return CompressorConfig(
            model=self.model,
            output=self.output,
            defaults=self.defaults,
            runs=new_runs,
            transferability=self.transferability,
        )

    # ------------------------------------------------------------------
    # Single-run convenience constructor (used by CLI for --task shorthand)
    # ------------------------------------------------------------------

    @classmethod
    def single_run(
        cls,
        model_type: str,
        model_name: str,
        task: str,
        **run_kwargs,
    ) -> "CompressorConfig":
        """Build a config for a single run directly from keyword arguments."""
        model = ModelConfig(type=model_type, name=model_name)
        run = RunConfig(task=task, **{k: v for k, v in run_kwargs.items() if k in _RUN_FIELDS})
        return cls(model=model, runs=[run])

    def to_dict(self) -> dict:
        """Serializable representation (no numpy, no non-JSON types)."""
        return {
            "model": {
                "type": self.model.type,
                "name": self.model.name,
                "kwargs": self.model.kwargs,
                "quantization": self.model.quantization,
            },
            "output": {"dir": self.output.dir, "overwrite": self.output.overwrite},
            "defaults": self.defaults,
            "runs": [
                {f: getattr(r, f) for f in _RUN_FIELDS}
                for r in self.runs
            ],
            "transferability": {
                "enabled": self.transferability.enabled,
                "pairs": self.transferability.pairs,
            },
        }