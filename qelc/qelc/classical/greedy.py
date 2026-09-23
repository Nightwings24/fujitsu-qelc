"""Greedy max-weight independent set: the classical heuristic baseline the
report's quantum-advantage claims must be measured against."""

from __future__ import annotations

import time

import numpy as np

from ..qubo.builder import QUBO
from ..solution import Solution, score


def greedy_mwis(qubo: QUBO) -> Solution:
    t0 = time.perf_counter()
    order = sorted(range(qubo.n), key=lambda i: (-qubo.weights[i], i))
    x = np.zeros(qubo.n, dtype=np.int8)
    blocked = set()
    neighbors = {i: set(qubo.conflict_graph.neighbors(i)) for i in range(qubo.n)}
    for i in order:
        if i not in blocked:
            x[i] = 1
            blocked |= neighbors[i]
    return score(x, qubo, "greedy", wall_time=time.perf_counter() - t0)
