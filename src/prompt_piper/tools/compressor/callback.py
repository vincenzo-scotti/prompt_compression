"""pymoo callback for per-generation Pareto front tracking."""

from __future__ import annotations

from pymoo.core.callback import Callback


class GenerationCallback(Callback):
    """Record the Pareto-optimal front after every generation.

    The ``history`` list is populated in-place during optimisation and is
    available on the callback object after ``minimize()`` returns.

    Parameters
    ----------
    transform_names:
        Ordered list of transformation names corresponding to the binary
        decision vector.  Used to annotate each solution with human-readable
        operator names.

    Attributes
    ----------
    history : list[dict]
        One entry per generation::

            {
                "generation": int,
                "n_solutions": int,
                "pareto_front": [
                    {
                        "bits": [0, 1, ...],
                        "transformation_names": ["TrimSpaces", ...],
                        "avg_tokens": float,
                        "avg_performance": float,
                    },
                    ...
                ]
            }
    """

    def __init__(self, transform_names: list[str]) -> None:
        super().__init__()
        self._names = transform_names
        self.history: list[dict] = []

    def notify(self, algorithm) -> None:
        opt = algorithm.opt
        if opt is None or len(opt) == 0:
            return

        X = opt.get("X").astype(int)   # (n_pareto, n_var)
        F = opt.get("F").copy()         # (n_pareto, 2): [tokens, -perf]
        F[:, 1] = -F[:, 1]             # restore positive performance

        solutions = []
        for i, bits in enumerate(X):
            solutions.append(
                {
                    "bits": bits.tolist(),
                    "transformation_names": [
                        name for name, b in zip(self._names, bits) if b
                    ],
                    "avg_tokens": float(F[i, 0]),
                    "avg_performance": float(F[i, 1]),
                }
            )

        self.history.append(
            {
                "generation": int(algorithm.n_gen),
                "n_solutions": len(solutions),
                "pareto_front": solutions,
            }
        )