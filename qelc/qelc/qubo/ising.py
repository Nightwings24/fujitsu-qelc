"""QUBO -> Ising conversion and exports to each runtime's operator type.

Substituting x_i = (1 - z_i) / 2 (z in {-1, +1}, z = +1 <-> x = 0) into

    C(x) = sum_i l_i x_i + sum_{i<j} q_ij x_i x_j + c

gives   H(z) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + offset  with

    J_ij   =  q_ij / 4
    h_i    = -l_i / 2 - sum_{j != i} q_ij / 4
    offset =  c + sum_i l_i / 2 + sum_{i<j} q_ij / 4

Bit convention used everywhere in this package: variable i <-> qubit i, and a
sampled basis-state integer s encodes x_i = (s >> i) & 1 (qulacs order).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .builder import QUBO


@dataclass
class Ising:
    n: int
    h: np.ndarray            # (n,)
    J: dict                  # {(i, j) i<j: coeff}
    offset: float

    def energy(self, x: np.ndarray) -> float:
        """Energy of a 0/1 assignment (z_i = 1 - 2 x_i)."""
        z = 1.0 - 2.0 * np.asarray(x, dtype=float)
        e = float(self.h @ z) + self.offset
        for (i, j), coeff in self.J.items():
            e += coeff * z[i] * z[j]
        return e


def qubo_to_ising(q: QUBO) -> Ising:
    n = q.n
    h = -q.linear / 2.0
    offset = q.offset + float(q.linear.sum()) / 2.0
    J = {}
    for (i, j), coeff in q.quadratic.items():
        J[(i, j)] = coeff / 4.0
        h[i] -= coeff / 4.0
        h[j] -= coeff / 4.0
        offset += coeff / 4.0
    return Ising(n=n, h=h, J=J, offset=offset)


def to_qulacs_observable(isg: Ising) -> "object":
    """qulacs Observable for expectation values (works on distributed backends)."""
    from qulacs import Observable

    obs = Observable(isg.n)
    for i, hi in enumerate(isg.h):
        if hi != 0.0:
            obs.add_operator(float(hi), f"Z {i}")
    for (i, j), coeff in isg.J.items():
        if coeff != 0.0:
            obs.add_operator(float(coeff), f"Z {i} Z {j}")
    return obs


def to_qubit_operator(isg: Ising) -> "object":
    """openfermion QubitOperator — the form QARP's QAOA accepts directly."""
    from openfermion import QubitOperator

    op = QubitOperator((), isg.offset)
    for i, hi in enumerate(isg.h):
        if hi != 0.0:
            op += QubitOperator(((int(i), "Z"),), float(hi))
    for (i, j), coeff in isg.J.items():
        if coeff != 0.0:
            op += QubitOperator(((int(i), "Z"), (int(j), "Z")), float(coeff))
    return op


def to_problem_graph(isg: Ising):
    """(weighted nx.Graph, linear_terms dict) for QARP's QAOABlock fallback path."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(isg.n))
    for (i, j), coeff in isg.J.items():
        G.add_edge(i, j, weight=float(coeff))
    linear_terms = {i: float(v) for i, v in enumerate(isg.h) if v != 0.0}
    return G, linear_terms


def dense_diagonal(isg: Ising) -> np.ndarray:
    """Reference-only: the full 2^n diagonal of H. Used in unit tests (n <= ~20)
    and by the vectorized brute-force baseline. Never used by the QAOA circuit."""
    n = isg.n
    size = 1 << n
    idx = np.arange(size, dtype=np.int64)
    diag = np.full(size, isg.offset, dtype=float)
    z = np.empty((n, size), dtype=np.int8)
    for i in range(n):
        z[i] = 1 - 2 * ((idx >> i) & 1)
    for i, hi in enumerate(isg.h):
        if hi != 0.0:
            diag += hi * z[i]
    for (i, j), coeff in isg.J.items():
        diag += coeff * (z[i].astype(float) * z[j])
    return diag
