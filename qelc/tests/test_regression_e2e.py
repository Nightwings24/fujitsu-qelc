"""Deterministic end-to-end regression on a fixed synthetic instance.

Asserts the full pipeline (graph -> cycles -> QUBO -> QAOA -> extraction)
is deterministic and reaches the certified optimum on an easy instance.
"""

import numpy as np

from qelc.classical.exact import brute_force_optimum
from qelc.classical.greedy import greedy_mwis
from qelc.data.cycles import enumerate_cycles, select_top_k
from qelc.data.synthetic import barabasi_debt_graph
from qelc.engine.metrics import approximation_ratio
from qelc.qaoa.optimize import solve_qaoa
from qelc.qubo.builder import build_qubo


def build_fixture():
    G = barabasi_debt_graph(40, m=2, seed=7)
    cycles = enumerate_cycles(G, length_bound=5)
    top = select_top_k(cycles, 14)
    assert len(top) == 14
    return build_qubo(top)


def test_graph_and_cycles_deterministic():
    q1, q2 = build_fixture(), build_fixture()
    assert np.allclose(q1.linear, q2.linear)
    assert q1.quadratic == q2.quadratic


def test_qaoa_reaches_optimum_and_is_deterministic():
    q = build_fixture()
    opt = brute_force_optimum(q)

    sol1, _ = solve_qaoa(q, p=2, n_starts=4, maxiter=100, n_shots=10_000, seed=123)
    sol2, _ = solve_qaoa(q, p=2, n_starts=4, maxiter=100, n_shots=10_000, seed=123)

    assert sol1.bitstring == sol2.bitstring, "same seed must give identical solutions"
    assert sol1.feasible
    ratio = approximation_ratio(sol1, opt)
    assert ratio == 1.0, f"expected optimum on easy instance, got ratio {ratio}"


def test_greedy_is_feasible_and_bounded_by_optimum():
    q = build_fixture()
    opt = brute_force_optimum(q)
    g = greedy_mwis(q)
    assert g.feasible
    assert g.savings <= opt.savings + 1e-9
