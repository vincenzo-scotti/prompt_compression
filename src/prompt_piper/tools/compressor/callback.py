"""pymoo callback for per-generation Pareto front tracking."""

from __future__ import annotations

import time

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
                "elapsed_seconds": float,
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

    def __init__(self, transform_names: list[str], recorder=None, layout=None) -> None:
        super().__init__()
        self._names = transform_names
        #: Bit layout (see ``optimization.bits``); when given, names are derived
        #: from it ("I:TrimSpaces", …).  Without it, *transform_names* must have
        #: one entry per bit.
        self._layout = layout
        self.history: list[dict] = []
        self._t0: float = time.perf_counter()
        #: Optional recorder: receives the *full* population (not only the
        #: Pareto front) and the offspring of every generation.
        self._recorder = recorder

    def _log_population(self, algorithm, elapsed: float) -> None:
        def dump(pop):
            if pop is None or len(pop) == 0:
                return []
            X = pop.get("X").astype(int)
            F = pop.get("F")
            rank = pop.get("rank")
            crowd = pop.get("crowding")
            rows = []
            for i in range(len(pop)):
                rows.append({
                    # Per-sample values behind the averages: candidates.jsonl[id].
                    "candidate_id": self._recorder.candidate_id(
                        self._recorder.context.get("method"), "search", X[i],
                    ),
                    "bits": X[i].tolist(),
                    "avg_tokens": float(F[i, 0]),
                    "avg_performance": float(-F[i, 1]),
                    "rank": None if rank is None or rank[i] is None else int(rank[i]),
                    "crowding": None if crowd is None or crowd[i] is None else float(crowd[i]),
                })
            return rows

        self._recorder.write("states", {
            **self._recorder.context,
            "kind": "population",
            "generation": int(algorithm.n_gen),
            "elapsed_seconds": elapsed,
            "n_evals": int(getattr(algorithm.evaluator, "n_eval", -1)),
            "population": dump(algorithm.pop),
            "offspring": dump(getattr(algorithm, "off", None)),
        })
        # Explicit Pareto front of this generation (search objectives).
        self._recorder.write("states", {
            **self._recorder.context,
            "kind": "front",
            "algorithm": self._recorder.context.get("method"),
            "step": int(algorithm.n_gen),
            "n_visited": int(getattr(algorithm.evaluator, "n_eval", -1)),
            "pareto_front": dump(algorithm.opt),
        })

    def _bit_names(self, bits) -> list[str]:
        if self._layout is not None:
            from prompt_piper.optimization.bits import bits_to_names
            return bits_to_names(self._layout, bits)
        if len(self._names) != len(bits):
            raise ValueError(f"{len(self._names)} names for {len(bits)} bits; pass the bit layout")
        return [name for name, b in zip(self._names, bits) if b]

    def notify(self, algorithm) -> None:
        elapsed = time.perf_counter() - self._t0
        if self._recorder is not None:
            self._log_population(algorithm, elapsed)
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
                    "transformation_names": self._bit_names(bits),
                    "avg_tokens": float(F[i, 0]),
                    "avg_performance": float(F[i, 1]),
                }
            )

        self.history.append(
            {
                "generation": int(algorithm.n_gen),
                "elapsed_seconds": round(elapsed, 3),
                "n_solutions": len(solutions),
                "pareto_front": solutions,
            }
        )