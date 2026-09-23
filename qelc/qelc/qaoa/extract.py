"""Solution extraction: seeded sampling -> exact classical rescoring ->
feasible minimum -> greedy repair.

This replaces the old `Counter(sampling(100)).most_common(1)` extraction,
which for 2^18+ states degenerates to an arbitrary single unseeded draw.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..config import derive_seed
from ..qubo.builder import QUBO
from ..qubo.ising import Ising
from ..solution import Solution, score
from .circuit import run_state


def sample_bitstrings(isg: Ising, params: np.ndarray, *, n_shots: int = 10_000, seed: int = 0) -> Counter:
    """Seeded measurement of the QAOA state; returns Counter{state_int: count}."""
    state = run_state(isg, params)
    samples = state.sampling(n_shots, derive_seed(seed, 0x5A3B))
    return Counter(samples)


def _int_to_x(s: int, n: int) -> np.ndarray:
    return np.array([(s >> i) & 1 for i in range(n)], dtype=np.int8)


def greedy_repair(x: np.ndarray, qubo: QUBO) -> np.ndarray:
    """Constructive form of the penalty-dominance argument:

    1. While a conflict is violated, switch off its lighter-weight endpoint
       (strictly lowers QUBO cost — see qelc.qubo.builder docstring).
    2. Greedily add unused non-conflicting cycles by weight (never hurts).
    """
    x = np.asarray(x, dtype=np.int8).copy()
    neighbors = {i: set(qubo.conflict_graph.neighbors(i)) for i in range(qubo.n)}

    violated = qubo.violated_conflicts(x)
    while violated:
        i, j = violated[0]
        drop = i if qubo.weights[i] <= qubo.weights[j] else j
        x[drop] = 0
        violated = qubo.violated_conflicts(x)

    active = {i for i in range(qubo.n) if x[i]}
    for i in sorted(range(qubo.n), key=lambda i: (-qubo.weights[i], i)):
        if not x[i] and not (neighbors[i] & active):
            x[i] = 1
            active.add(i)
    return x


def extract_solution(samples: Counter, qubo: QUBO, *, source: str = "qaoa") -> Solution:
    """Best feasible assignment derivable from the sample set.

    Every unique sample is rescored exactly (cheap: O(n + |conflicts|) each).
    The best feasible sample competes with the repaired version of the best
    infeasible sample; the winner is returned.
    """
    n = qubo.n
    best_feasible: np.ndarray | None = None
    best_feasible_cost = np.inf
    best_any: np.ndarray | None = None
    best_any_cost = np.inf

    for s in samples:
        x = _int_to_x(int(s), n)
        c = qubo.value(x)
        if c < best_any_cost:
            best_any_cost, best_any = c, x
        if c < best_feasible_cost and qubo.is_feasible(x):
            best_feasible_cost, best_feasible = c, x

    candidates: list[tuple[np.ndarray, str]] = []
    if best_feasible is not None:
        candidates.append((best_feasible, source))
    if best_any is not None:
        repaired = greedy_repair(best_any, qubo)
        candidates.append((repaired, f"{source}-repaired"))

    if not candidates:  # empty sample set: all-zeros is always feasible
        return score(np.zeros(n, dtype=np.int8), qubo, source)

    best_x, best_src = min(candidates, key=lambda c: qubo.value(c[0]))
    sol = score(best_x, qubo, best_src)
    sol.meta["n_unique_samples"] = len(samples)
    return sol
