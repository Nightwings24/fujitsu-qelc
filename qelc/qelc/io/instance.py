"""Problem-instance JSON files: built and exact-solved locally, executed on
the FX700. Loading needs only numpy + networkx (both ship with the QARP env);
no pandas or qulacs required cluster-side."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ProblemInstance:
    name: str
    n: int
    linear: list            # scaled QUBO linear coefficients
    quadratic: dict         # {"i,j": coeff}
    offset: float
    w_max: float
    weights: list           # raw cycle savings (token units)
    conflict_edges: list    # [(i, j), ...]
    partition: list | None = None       # 0/1 labels for cutting-demo instances
    exact_cost: float | None = None     # certified optimal QUBO cost (n <= 26)
    exact_bitstring: str | None = None
    provenance: dict | None = None


def save_instance(qubo, path: str, *, name: str, partition=None, exact=None, **provenance) -> None:
    inst = ProblemInstance(
        name=name,
        n=qubo.n,
        linear=[float(v) for v in qubo.linear],
        quadratic={f"{i},{j}": float(v) for (i, j), v in qubo.quadratic.items()},
        offset=float(qubo.offset),
        w_max=float(qubo.w_max),
        weights=[float(v) for v in qubo.weights],
        conflict_edges=[[int(i), int(j)] for (i, j) in qubo.quadratic],
        partition=list(partition) if partition is not None else None,
        exact_cost=(float(exact.qubo_cost) if exact is not None else None),
        exact_bitstring=(exact.bitstring if exact is not None else None),
        provenance=provenance or None,
    )
    with open(path, "w") as f:
        json.dump(asdict(inst), f, indent=2)


def load_instance(path: str):
    """Returns (QUBO, Ising, ProblemInstance). Reconstructs the QUBO without
    cycles (they are not needed for solving — only weights and conflicts)."""
    from ..qubo.builder import QUBO
    from ..qubo.ising import qubo_to_ising
    import networkx as nx

    with open(path) as f:
        d = json.load(f)
    inst = ProblemInstance(**d)

    quadratic = {tuple(map(int, k.split(","))): float(v) for k, v in inst.quadratic.items()}
    Gc = nx.Graph()
    Gc.add_nodes_from(range(inst.n))
    Gc.add_edges_from(quadratic.keys())
    qubo = QUBO(
        linear=np.array(inst.linear, dtype=float),
        quadratic=quadratic,
        offset=float(inst.offset),
        w_max=float(inst.w_max),
        weights=np.array(inst.weights, dtype=float),
        cycles=[],
        conflict_graph=Gc,
    )
    return qubo, qubo_to_ising(qubo), inst
