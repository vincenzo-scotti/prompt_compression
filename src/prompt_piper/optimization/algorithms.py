"""Factory for pymoo multi-objective algorithms.

All algorithms are configured for binary search spaces by default.
"""

from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.util.ref_dirs import get_reference_directions

_REGISTRY: dict[str, type] = {
    "nsga2": NSGA2,
    "nsga3": NSGA3,
    "moead": MOEAD,
}


def build_algorithm(
    name: str,
    n_var: int,
    pop_size: int = 50,
    **kwargs,
):
    """Return a configured pymoo algorithm for a binary problem.

    Parameters
    ----------
    name:
        Algorithm identifier.  One of ``"nsga2"``, ``"nsga3"``, ``"moead"``.
    n_var:
        Number of decision variables (needed to set the mutation probability).
    pop_size:
        Population size.
    **kwargs:
        Extra keyword arguments forwarded to the algorithm constructor.

    Returns
    -------
    pymoo algorithm instance ready to be passed to ``pymoo.optimize.minimize``.
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown algorithm {name!r}. Available: {sorted(_REGISTRY)}"
        )

    sampling = BinaryRandomSampling()
    crossover = TwoPointCrossover()
    mutation = BitflipMutation(prob=1.0 / n_var)

    if name == "nsga2":
        return NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
            **kwargs,
        )

    if name == "nsga3":
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=pop_size - 1)
        return NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
            **kwargs,
        )

    if name == "moead":
        ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size - 1)
        return MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=15,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            **kwargs,
        )

    # Should never reach here given the registry check above.
    raise ValueError(f"Algorithm {name!r} registered but not handled.")


def available_algorithms() -> list[str]:
    """Return the list of supported algorithm names."""
    return sorted(_REGISTRY)