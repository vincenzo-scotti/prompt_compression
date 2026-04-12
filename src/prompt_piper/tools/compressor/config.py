"""Compressor tool configuration.

Supports both YAML file loading and CLI argument override.

YAML schema example
-------------------
.. code-block:: yaml

    model:
      type: hf                              # hf | openai | vllm
      name: mistralai/Mistral-7B-v0.1
      kwargs:                               # forwarded to model constructor
        device: cuda
        dtype: float16

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
    transformation_ids: list[str] | None = None
    prompt_field: str = "prompt"
    #: Field to apply transformations to.  ``None`` means use ``prompt_field``
    #: (i.e. compress the fully formatted prompt).  Set to
    #: ``"prompt_0shot"`` to compress only the 0-shot prompt.
    compress_field: str | None = None

    @property
    def run_name(self) -> str:
        return self.name or f"{self.task}_{self.algorithm}"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

_RUN_FIELDS = {f for f in RunConfig.__dataclass_fields__}


@dataclass
class CompressorConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    defaults: dict[str, Any] = field(default_factory=dict)
    runs: list[RunConfig] = field(default_factory=list)

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

        with open(path) as f:
            raw: dict = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> "CompressorConfig":
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, raw: dict) -> "CompressorConfig":
        model_raw = raw.get("model", {})
        model = ModelConfig(
            type=model_raw.get("type", "hf"),
            name=model_raw.get("name", "gpt2"),
            kwargs=model_raw.get("kwargs", {}),
        )

        output_raw = raw.get("output", {})
        output = OutputConfig(
            dir=output_raw.get("dir", "./results"),
            overwrite=output_raw.get("overwrite", False),
        )

        defaults: dict = raw.get("defaults", {})

        runs_raw: list[dict] = raw.get("runs", [])
        runs = [cls._build_run(r, defaults) for r in runs_raw]

        return cls(model=model, output=output, defaults=defaults, runs=runs)

    @staticmethod
    def _build_run(run_raw: dict, defaults: dict) -> RunConfig:
        """Merge defaults → run-level overrides into a RunConfig."""
        merged = copy.deepcopy(defaults)
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
            "model": {"type": self.model.type, "name": self.model.name, "kwargs": self.model.kwargs},
            "output": {"dir": self.output.dir, "overwrite": self.output.overwrite},
            "defaults": self.defaults,
            "runs": [
                {f: getattr(r, f) for f in _RUN_FIELDS}
                for r in self.runs
            ],
        }