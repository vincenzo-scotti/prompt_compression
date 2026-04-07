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
    transformation_ids:
        Subset of operator IDs (e.g. ``["O1", "O3", "O7"]``) to include in
        the search.  ``None`` means all 28 operators.
    verbose:
        Whether to print pymoo progress.
    """

    # --- Dataset ---
    prompt_field: str = "prompt"
    target_field: str | None = "target"
    n_subset: int | None = 100
    subset_seed: int = 42

    # --- Algorithm ---
    algorithm: str = "nsga2"
    pop_size: int = 50
    n_gen: int = 100
    seed: int = 42

    # --- Search space ---
    transformation_ids: list[str] | None = None  # None → all 28

    # --- Misc ---
    verbose: bool = False