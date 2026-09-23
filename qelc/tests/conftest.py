import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from qelc.config import rng_for
from qelc.data.cycles import Cycle
from qelc.qubo.builder import QUBO, build_qubo


def random_qubo(n: int, seed: int, *, density: float = 0.35) -> QUBO:
    """Random MWIS QUBO with the package's penalty rule (synthetic cycles)."""
    rng = rng_for(seed, 0xF00D)
    weights = rng.lognormal(mean=3.0, sigma=1.0, size=n)
    cycles = []
    # Fake cycles that share edges according to a random conflict pattern:
    # give cycle i a private edge (i, i + 1000) plus shared edges drawn from a pool.
    pool = [(int(a), int(a) + 1) for a in range(n * 2)]
    for i in range(n):
        edges = {(i, i + 1000)}
        for e in pool:
            if rng.random() < density / 2:
                edges.add(e)
        cycles.append(Cycle(nodes=(i, i + 1000), edges=frozenset(edges), savings=float(weights[i])))
    return build_qubo(cycles)


@pytest.fixture
def small_qubo():
    return random_qubo(8, seed=11)
