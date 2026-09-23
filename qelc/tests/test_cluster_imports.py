"""Guard the FX700 (QARP venv) import surface.

The cluster venv ships numpy/networkx/scipy/openfermion + the QARP-built
qulacs, but NOT pandas/pyarrow/pulp/matplotlib. The cluster-side scripts
import `qelc` (transitively reaching qubo/qaoa/io/qvi), so those modules must
not pull in a local-only dependency at import time. This test blocks the
local-only libs and exercises the exact cluster call chain.
"""

import builtins
import importlib
from collections import Counter

import numpy as np
import pytest

BLOCKED = {"pandas", "pyarrow", "pulp", "matplotlib"}


@pytest.fixture
def cluster_env(monkeypatch):
    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"[cluster-sim] {name} unavailable in QARP venv")
        return real_import(name, *a, **k)

    # drop already-cached local-only modules so re-import goes through the guard
    import sys

    for mod in list(sys.modules):
        if mod.split(".")[0] in BLOCKED:
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", guard)
    yield


def test_cluster_import_chain_has_no_local_only_deps(cluster_env, tmp_path):
    # the modules the cluster scripts import
    from qelc.io.instance import load_instance, save_instance
    from qelc.qaoa.circuit import build_qaoa_circuit, expectation
    from qelc.qaoa.extract import extract_solution, sample_bitstrings
    from qelc.qubo.builder import build_qubo
    from qelc.qubo.ising import qubo_to_ising, to_qulacs_observable
    from qelc.qvi.grover import run_grover
    from qelc.data.cycles import Cycle

    # build a tiny instance in-memory (no CSV/pandas), round-trip through JSON
    cycles = [
        Cycle(nodes=(0, 1), edges=frozenset({(0, 1), (1, 0)}), savings=5.0),
        Cycle(nodes=(2, 3), edges=frozenset({(2, 3), (3, 2)}), savings=3.0),
        Cycle(nodes=(0, 1, 2), edges=frozenset({(0, 1), (1, 2), (2, 0)}), savings=4.0),
    ]
    qubo = build_qubo(cycles)
    path = tmp_path / "inst.json"
    save_instance(qubo, str(path), name="unit")
    qubo2, isg, inst = load_instance(str(path))

    obs = to_qulacs_observable(isg)
    build_qaoa_circuit(isg, [0.3], [0.5])
    e = expectation(isg, np.array([0.3, 0.5]))
    samples = sample_bitstrings(isg, np.array([0.3, 0.5]), n_shots=200, seed=0)
    sol = extract_solution(samples, qubo2)
    run_grover(4, seed=0, n_shots=200)

    assert obs.get_term_count() >= 1
    assert isinstance(e, float)
    assert sol.feasible
