"""QUBO for max-savings non-conflicting cycle selection (MWIS).

Objective (scaled by 1/w_max for conditioning):

    C(x) = - sum_i (w_i / w_max) x_i
           + sum_{(i,j) in conflicts} P_ij x_i x_j

Penalty choice  P_ij = min(w_i, w_j) + eps  (eps > 0) guarantees feasibility
of every optimum: take any x with a violated conflict (i, j), w_i <= w_j, and
flip x_i to 0. The cost change is

    dC = + w_i - sum_{k: (i,k) conflict, x_k = 1} P_ik
       <= w_i - P_ij  =  w_i - min(w_i, w_j) - eps  =  -eps  <  0,

so every infeasible assignment is strictly improved by switching off the
lighter endpoint of any violated conflict. Hence all global optima are
feasible and the QUBO optimum coincides with the MWIS optimum. The same move
is the greedy repair operator (qelc.qaoa.extract.greedy_repair).

Compared to oversized penalties (e.g. w_i + w_j), this keeps the penalty
scale comparable to the reward scale, which avoids crowding all feasible
states into a narrow band of the spectrum — an easier landscape for QAOA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from ..data.cycles import Cycle, conflict_edges


@dataclass
class QUBO:
    linear: np.ndarray                 # (n,) scaled linear coefficients (negative)
    quadratic: dict                    # {(i, j) i<j: scaled penalty > 0}
    offset: float
    w_max: float                       # un-scaling factor back to token units
    weights: np.ndarray                # (n,) raw cycle savings (token units)
    cycles: list = field(default_factory=list)      # list[Cycle], may be empty for synthetic QUBOs
    conflict_graph: nx.Graph = field(default_factory=nx.Graph)

    @property
    def n(self) -> int:
        return len(self.linear)

    def value(self, x: np.ndarray) -> float:
        """Scaled QUBO cost of a 0/1 assignment."""
        x = np.asarray(x, dtype=float)
        v = float(self.linear @ x) + self.offset
        for (i, j), q in self.quadratic.items():
            v += q * x[i] * x[j]
        return v

    def savings(self, x: np.ndarray) -> float:
        """Raw objective in token units, ignoring penalties (only meaningful
        for feasible x)."""
        x = np.asarray(x, dtype=float)
        return float(self.weights @ x)

    def violated_conflicts(self, x: np.ndarray) -> list:
        x = np.asarray(x)
        return [(i, j) for (i, j) in self.quadratic if x[i] and x[j]]

    def is_feasible(self, x: np.ndarray) -> bool:
        return not self.violated_conflicts(x)


def build_qubo(
    cycles: list,
    *,
    penalty_eps_frac: float = 0.25,
    log_weights: bool = False,
) -> QUBO:
    """Build the MWIS QUBO from scored cycles.

    log_weights compresses dynamic range (experiment flag). It changes the
    optimization target away from raw savings, so headline results must use
    the default linear form.
    """
    if not cycles:
        raise ValueError("No cycles to build a QUBO from")

    raw = np.array([c.savings for c in cycles], dtype=float)
    w = np.log1p(raw) if log_weights else raw
    w_max = float(w.max())
    w_scaled = w / w_max
    eps = penalty_eps_frac * float(w_scaled.mean())

    conflicts = conflict_edges(cycles)
    quadratic = {
        (i, j): float(min(w_scaled[i], w_scaled[j]) + eps) for i, j in conflicts
    }

    Gc = nx.Graph()
    Gc.add_nodes_from(range(len(cycles)))
    for i in range(len(cycles)):
        Gc.nodes[i]["weight"] = float(raw[i])
    Gc.add_edges_from(conflicts)

    return QUBO(
        linear=-w_scaled,
        quadratic=quadratic,
        offset=0.0,
        w_max=w_max,
        weights=raw,
        cycles=list(cycles),
        conflict_graph=Gc,
    )
