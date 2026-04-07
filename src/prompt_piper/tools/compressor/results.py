"""Serializable result objects for the compressor tool.

All results are JSON-serializable and self-contained: every file includes
the full configuration so experiments can be reproduced and aggregated later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Solution-level result
# ---------------------------------------------------------------------------

@dataclass
class SolutionEval:
    """Evaluation of a single Pareto-optimal transformation pipeline."""

    transformation_ids: list[str]
    transformation_names: list[str]
    bits: list[int]
    # Search-phase objectives (on the search subset)
    search_avg_tokens: float
    search_avg_performance: float
    # Test-phase objectives (on the held-out test split); None until evaluated
    test_avg_tokens: float | None = None
    test_avg_performance: float | None = None


# ---------------------------------------------------------------------------
# Baseline-level result
# ---------------------------------------------------------------------------

@dataclass
class BaselineEval:
    """Evaluation of a single baseline strategy."""

    name: str
    transformation_names: list[list[str]]
    # Search-phase (same dataset subset used by the optimiser)
    search_avg_tokens: float
    search_avg_performance: float
    # Test-phase
    test_avg_tokens: float | None = None
    test_avg_performance: float | None = None


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------

@dataclass
class CompressorResult:
    """Complete, self-contained result for one (task, algorithm) run.

    Attributes
    ----------
    run_name:
        Human-readable identifier (``"hellaswag_nsga2"``).
    timestamp:
        ISO-8601 UTC timestamp when the run started.
    config:
        Full serialised configuration dict (model + run settings).
    transformation_ids:
        Ordered IDs of the operators in the search space.
    transformation_names:
        Ordered names of the operators in the search space.
    search_elapsed_seconds:
        Wall-clock time for the optimisation phase.
    eval_elapsed_seconds:
        Wall-clock time for the test-evaluation phase.
    pareto_solutions:
        Pareto-optimal solutions with both search and test objectives.
    baselines:
        Baseline results with both search and test objectives.
    generation_history:
        Per-generation Pareto front snapshots from the optimiser.
    extra:
        Arbitrary extra metadata (e.g. hardware info, software versions).
    """

    run_name: str
    timestamp: str
    config: dict[str, Any]
    transformation_ids: list[str]
    transformation_names: list[str]
    search_elapsed_seconds: float
    eval_elapsed_seconds: float
    pareto_solutions: list[SolutionEval]
    baselines: list[BaselineEval]
    generation_history: list[dict]
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """Write the result to *path* as pretty-printed JSON.

        Returns the resolved path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CompressorResult":
        with open(path) as f:
            d = json.load(f)
        # Reconstruct nested dataclasses
        d["pareto_solutions"] = [SolutionEval(**s) for s in d["pareto_solutions"]]
        d["baselines"] = [BaselineEval(**b) for b in d["baselines"]]
        return cls(**d)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def now_timestamp() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    def summary(self) -> str:
        lines = [
            f"Run: {self.run_name}  [{self.timestamp}]",
            f"Search: {self.search_elapsed_seconds:.1f}s  |  "
            f"Eval: {self.eval_elapsed_seconds:.1f}s",
            f"Pareto front ({len(self.pareto_solutions)} solutions):",
        ]
        for sol in self.pareto_solutions:
            test_str = (
                f"  test=({sol.test_avg_tokens:.1f} tok, {sol.test_avg_performance:.3f} perf)"
                if sol.test_avg_tokens is not None
                else ""
            )
            lines.append(
                f"  search=({sol.search_avg_tokens:.1f} tok, "
                f"{sol.search_avg_performance:.3f} perf){test_str}"
                f"  ops={sol.transformation_names}"
            )
        if self.baselines:
            lines.append("Baselines:")
            for b in self.baselines:
                test_str = (
                    f"  test=({b.test_avg_tokens:.1f} tok, {b.test_avg_performance:.3f} perf)"
                    if b.test_avg_tokens is not None
                    else ""
                )
                lines.append(
                    f"  {b.name}: search=({b.search_avg_tokens:.1f} tok, "
                    f"{b.search_avg_performance:.3f} perf){test_str}"
                )
        return "\n".join(lines)