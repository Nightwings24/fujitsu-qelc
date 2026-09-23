"""Common Solution container shared by quantum and classical solvers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Solution:
    x: np.ndarray               # 0/1 assignment, len n
    qubo_cost: float            # scaled QUBO value
    savings: float              # token units (real objective), feasible x only
    feasible: bool
    source: str                 # "qaoa" | "qaoa-repaired" | "greedy" | "exact" | "milp" | "random"
    wall_time: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def bitstring(self) -> str:
        return "".join(str(int(b)) for b in self.x)

    @property
    def selected(self) -> list:
        return [i for i, b in enumerate(self.x) if b]


def score(x: np.ndarray, qubo, source: str, wall_time: float = 0.0, **meta) -> Solution:
    x = np.asarray(x, dtype=np.int8)
    feasible = qubo.is_feasible(x)
    return Solution(
        x=x,
        qubo_cost=qubo.value(x),
        savings=qubo.savings(x) if feasible else float("nan"),
        feasible=feasible,
        source=source,
        wall_time=wall_time,
        meta=meta,
    )
