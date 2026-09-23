"""Repair operator and extraction pipeline properties."""

from collections import Counter

import numpy as np

from qelc.classical.exact import brute_force_optimum
from qelc.config import rng_for
from qelc.qaoa.extract import extract_solution, greedy_repair

from conftest import random_qubo


def test_repair_always_feasible_and_never_worse():
    for seed in range(30):
        q = random_qubo(10, seed=seed)
        rng = rng_for(seed, 3)
        for _ in range(20):
            x = rng.integers(0, 2, size=q.n)
            fixed = greedy_repair(x, q)
            assert q.is_feasible(fixed)
            assert q.value(fixed) <= q.value(x) + 1e-12


def test_repair_keeps_feasible_savings():
    """A feasible input may only gain savings (greedy add phase)."""
    for seed in range(20):
        q = random_qubo(10, seed=seed)
        rng = rng_for(seed, 4)
        for _ in range(20):
            x = rng.integers(0, 2, size=q.n)
            if q.is_feasible(x):
                fixed = greedy_repair(x, q)
                assert q.savings(fixed) >= q.savings(x) - 1e-12


def test_extract_returns_feasible_and_at_least_best_sample():
    q = random_qubo(10, seed=7)
    rng = rng_for(7, 5)
    samples = Counter(int(s) for s in rng.integers(0, 1 << q.n, size=500))
    sol = extract_solution(samples, q)
    assert sol.feasible
    # extraction (with repair) must be at least as good as the best feasible raw sample
    best_raw = min(
        (q.value(np.array([(s >> i) & 1 for i in range(q.n)]))
         for s in samples
         if q.is_feasible(np.array([(s >> i) & 1 for i in range(q.n)]))),
        default=0.0,
    )
    assert sol.qubo_cost <= best_raw + 1e-12


def test_extract_empty_samples_returns_zero_solution():
    q = random_qubo(6, seed=1)
    sol = extract_solution(Counter(), q)
    assert sol.feasible and sol.savings == 0.0


def test_exact_beats_or_ties_everything():
    for seed in range(10):
        q = random_qubo(10, seed=seed)
        opt = brute_force_optimum(q)
        rng = rng_for(seed, 6)
        samples = Counter(int(s) for s in rng.integers(0, 1 << q.n, size=2000))
        sol = extract_solution(samples, q)
        assert opt.qubo_cost <= sol.qubo_cost + 1e-12
