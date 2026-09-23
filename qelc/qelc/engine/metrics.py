"""Honest evaluation metrics: approximation ratio against a certified optimum,
probability of sampling the optimum, and netting efficiency."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from ..qubo.builder import QUBO
from ..solution import Solution


def approximation_ratio(sol: Solution, opt: Solution) -> float:
    """Ratio of achieved to optimal savings (feasible solutions only)."""
    if not (sol.feasible and opt.feasible):
        return float("nan")
    if opt.savings == 0:
        return 1.0 if sol.savings == 0 else float("inf")
    return sol.savings / opt.savings


def p_optimal(samples: Counter, qubo: QUBO, opt_cost: float, *, tol: float = 1e-9) -> float:
    """Probability mass of samples that hit the optimal QUBO cost."""
    total = sum(samples.values())
    if total == 0:
        return 0.0
    hits = 0
    for s, c in samples.items():
        x = np.array([(int(s) >> i) & 1 for i in range(qubo.n)], dtype=np.int8)
        if abs(qubo.value(x) - opt_cost) <= tol:
            hits += c
    return hits / total


def netting_efficiency(initial_debt: float, cleared: float) -> float:
    return 100.0 * cleared / initial_debt if initial_debt > 0 else 0.0


def report_table(solutions: dict, opt: Solution | None = None) -> pd.DataFrame:
    """Comparison table across solvers; the honest replacement for the
    report's unbenchmarked claims."""
    rows = []
    for name, sol in solutions.items():
        row = {
            "solver": name,
            "feasible": sol.feasible,
            "savings": sol.savings,
            "qubo_cost": sol.qubo_cost,
            "n_selected": int(np.sum(sol.x)),
            "wall_time_s": round(sol.wall_time, 4),
            "source": sol.source,
        }
        if opt is not None:
            row["approx_ratio"] = round(approximation_ratio(sol, opt), 6)
        rows.append(row)
    return pd.DataFrame(rows)
