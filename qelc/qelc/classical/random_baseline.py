"""Uniform-random sampling pushed through the SAME extraction/repair pipeline
as QAOA samples. If QAOA does not beat this, its distribution carries no
information — the fair null hypothesis for any sampling-based solver."""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from ..config import rng_for
from ..qubo.builder import QUBO
from ..solution import Solution


def random_baseline(qubo: QUBO, *, n_samples: int = 10_000, seed: int = 0) -> Solution:
    from ..qaoa.extract import extract_solution  # shared pipeline, late import avoids cycle

    t0 = time.perf_counter()
    rng = rng_for(seed, 0xBA5E)
    states = rng.integers(0, 1 << qubo.n, size=n_samples, dtype=np.int64)
    sol = extract_solution(Counter(states.tolist()), qubo, source="random")
    sol.wall_time = time.perf_counter() - t0
    return sol
