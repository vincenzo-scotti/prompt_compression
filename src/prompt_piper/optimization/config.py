"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """All knobs for a single experiment run.

    Parameters
    ----------
    prompt_field:
        Key inside each dataset sample dict that holds the prompt string.
    target_field:
        Key inside each dataset sample dict that holds the ground-truth label/
        answer used by the evaluator.  May be ``None`` for mock evaluators.
    n_subset:
        Number of dataset samples used per fitness evaluation.  Set to ``None``
        to use the full dataset.
    subset_seed:
        Random seed for drawing the dataset subset.
    algorithm:
        Name of the pymoo algorithm to use.  One of ``"nsga2"``, ``"nsga3"``,
        ``"moead"``.
    pop_size:
        Population size for the evolutionary algorithm.
    n_gen:
        Number of generations.
    seed:
        Random seed passed to the algorithm.
    mab_reward_weight:
        Weight of performance in the Multi-Armed Bandit reward; the rest goes
        to compression.
    transformation_ids:
        Subset of operator IDs (e.g. ``["O1", "O3", "O7"]``) to include in
        the search.  ``None`` means all 28 operators.
    verbose:
        Whether to print pymoo progress.
    """

    # --- Dataset ---
    prompt_field: str = "prompt"
    #: Field to apply transformations to.  When ``None`` (default) falls back
    #: to ``prompt_field``.  Set to ``"prompt_0shot"`` (and ensure samples
    #: carry that key) to compress the 0-shot prompt instead of the fully
    #: formatted prompt (which may include few-shot examples).
    compress_field: str | None = None
    target_field: str | None = "target"
    n_subset: int | None = 100
    subset_seed: int = 42

    # --- Algorithm ---
    algorithm: str = "nsga2"
    pop_size: int = 50
    n_gen: int = 100
    seed: int = 42
    #: Weight of performance in the multi_armed_bandit reward
    #: ``w · performance + (1 − w) · compression`` (``1`` = performance only).
    mab_reward_weight: float = 0.5

    # --- Search space ---
    transformation_ids: list[str] | None = None  # None → all 28
    #: Further restrict the search to this subset of operator IDs.  When set,
    #: NSGA-II (and budget-matched baselines random_search / multi_armed_bandit)
    #: only explore these operators, and ``all_transformations`` applies all of
    #: them.  The ``summarization`` baseline always uses the full
    #: ``transformation_ids`` set (it needs the semantic operators).  ``None`` (default) means no
    #: extra restriction beyond ``transformation_ids``.
    search_transformation_ids: list[str] | None = None

    # --- Section-aware compression ---
    #: When True (default), also compress the per-sample input data section.
    #: When False, the input data section is passed through unchanged and no
    #: data-scoped operator bits are included in the search vector.
    compress_input_data: bool = True

    # --- Text-level baselines ---
    #: When True, truncated_prompt / random_deletion / llmlingua2 act only on
    #: instruction + shots + data and keep the answer cue and tail unchanged.
    preserve_answer_cue: bool = False

    # --- Misc ---
    verbose: bool = False