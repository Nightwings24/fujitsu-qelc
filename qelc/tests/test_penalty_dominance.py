"""Property tests for the QUBO construction.

1. Dominance: with P_ij = min(w_i, w_j) + eps, every brute-force optimum is
   feasible (the penalty proof in qelc.qubo.builder).
2. Round trip: QUBO cost == Ising energy for random assignments.
3. Objective identity: for feasible x, scaled QUBO cost == -savings / w_max.
"""

import numpy as np

from qelc.classical.exact import brute_force_optimum
from qelc.config import rng_for
from qelc.qubo.ising import qubo_to_ising

from conftest import random_qubo


def test_optimum_always_feasible():
    for seed in range(100):
        q = random_qubo(10, seed=seed)
        opt = brute_force_optimum(q)
        assert opt.feasible, f"seed {seed}: infeasible optimum {opt.bitstring}"


def test_qubo_ising_roundtrip():
    rng = rng_for(0, 1)
    for seed in range(20):
        q = random_qubo(9, seed=seed)
        isg = qubo_to_ising(q)
        for _ in range(50):
            x = rng.integers(0, 2, size=q.n)
            assert abs(q.value(x) - isg.energy(x)) < 1e-9


def test_feasible_cost_equals_negative_scaled_savings():
    for seed in range(20):
        q = random_qubo(9, seed=seed)
        rng = rng_for(seed, 2)
        for _ in range(30):
            x = rng.integers(0, 2, size=q.n)
            if q.is_feasible(x):
                assert abs(q.value(x) + q.savings(x) / q.w_max) < 1e-9
