"""Peeling engine invariants, audited independently."""

import networkx as nx

from qelc.data.synthetic import barabasi_debt_graph
from qelc.engine.audit import audit_log
from qelc.engine.peeling import PeelingConfig, run_peeling


def _run(solver: str, **kw):
    G = barabasi_debt_graph(60, m=3, seed=3)
    cfg = PeelingConfig(k=12, max_iters=8, solver=solver, master_seed=99, **kw)
    report = run_peeling(G, cfg)
    return G, report


def test_greedy_peeling_audit_passes():
    G, report = _run("greedy")
    assert report.total_cleared > 0
    audit = audit_log(G, report.log, report.total_cleared)
    assert audit["passed"], audit["failures"]


def test_exact_peeling_audit_passes():
    G, report = _run("exact")
    audit = audit_log(G, report.log, report.total_cleared)
    assert audit["passed"], audit["failures"]


def test_qaoa_peeling_audit_passes_and_deterministic():
    G, r1 = _run("qaoa", p=1, maxiter=40, n_starts=2, n_shots=2000)
    _, r2 = _run("qaoa", p=1, maxiter=40, n_starts=2, n_shots=2000)
    assert r1.total_cleared == r2.total_cleared
    audit = audit_log(G, r1.log, r1.total_cleared)
    assert audit["passed"], audit["failures"]


def test_gross_reduction_matches_ledger_delta():
    G, report = _run("greedy")
    # netting a length-L cycle by amt reduces gross debt by L*amt, so the
    # ledger delta must equal the per-cycle gross reduction from the log
    assert abs((report.initial_debt - report.final_debt) - report.gross_reduction) < 1e-6
    assert report.gross_reduction >= report.total_cleared
