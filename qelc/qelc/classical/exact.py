"""Exact optima: vectorized brute force (ground truth for n <= ~26) and an
optional CBC MILP for larger instances."""

from __future__ import annotations

import time

import numpy as np

from ..qubo.builder import QUBO
from ..qubo.ising import dense_diagonal, qubo_to_ising
from ..solution import Solution, score

_CHUNK_BITS = 24  # evaluate at most 2^24 assignments (~128 MB of float64) at a time


def brute_force_optimum(qubo: QUBO, *, max_n: int = 26) -> Solution:
    """Global optimum by exhaustive evaluation, chunked to bound memory.

    This is the honest role for exhaustive evaluation: a *verifier* that
    ground-truths the quantum solver on small instances — not the solver
    itself.
    """
    n = qubo.n
    if n > max_n:
        raise ValueError(f"brute force capped at {max_n} variables, got {n}")
    t0 = time.perf_counter()
    isg = qubo_to_ising(qubo)

    if n <= _CHUNK_BITS:
        diag = dense_diagonal(isg)
        best = int(np.argmin(diag))
    else:
        best, best_val = 0, np.inf
        chunk = 1 << _CHUNK_BITS
        idx = np.arange(chunk, dtype=np.int64)
        z = np.empty((n, chunk), dtype=np.int8)
        for lo in range(0, 1 << n, chunk):
            block = lo + idx
            for i in range(n):
                z[i] = 1 - 2 * ((block >> i) & 1)
            vals = np.full(chunk, isg.offset)
            for i, hi in enumerate(isg.h):
                if hi != 0.0:
                    vals += hi * z[i]
            for (i, j), coeff in isg.J.items():
                vals += coeff * (z[i].astype(np.float64) * z[j])
            k = int(np.argmin(vals))
            if vals[k] < best_val:
                best_val, best = float(vals[k]), lo + k
    x = np.array([(best >> i) & 1 for i in range(n)], dtype=np.int8)
    return score(x, qubo, "exact", wall_time=time.perf_counter() - t0)


def milp_optimum(qubo: QUBO, *, time_limit_s: int = 120) -> Solution:
    """MWIS as a MILP via PuLP/CBC — scales past brute force; optional dep."""
    import pulp

    t0 = time.perf_counter()
    prob = pulp.LpProblem("mwis", pulp.LpMaximize)
    xs = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(qubo.n)]
    prob += pulp.lpSum(float(qubo.weights[i]) * xs[i] for i in range(qubo.n))
    for (i, j) in qubo.quadratic:
        prob += xs[i] + xs[j] <= 1
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s))
    x = np.array([int(round(v.value() or 0)) for v in xs], dtype=np.int8)
    sol = score(x, qubo, "milp", wall_time=time.perf_counter() - t0)
    sol.meta["status"] = pulp.LpStatus[status]
    return sol
