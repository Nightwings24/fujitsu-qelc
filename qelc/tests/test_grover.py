"""Grover module: success probability must match sin^2((2m+1) theta) theory,
and the decomposed multi-controlled gates must be exactly correct."""

import numpy as np
import pytest
from qulacs import QuantumCircuit, QuantumState

from qelc.qvi.grover import (
    _mcz_all_ones,
    grover_circuit,
    n_qubits_for_key,
    optimal_iterations,
    run_grover,
    theoretical_success,
)
from qelc.qvi.qvi_model import compute_qvi


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_mcz_is_exact(k):
    """Decomposed MCZ == diagonal with -1 on the all-ones key subspace
    (ancillas returned clean)."""
    n = n_qubits_for_key(k)
    qc = QuantumCircuit(n)
    _mcz_all_ones(qc, list(range(k)), list(range(k, n)))

    rng = np.random.default_rng(k)
    vec = rng.normal(size=1 << k) + 1j * rng.normal(size=1 << k)
    vec /= np.linalg.norm(vec)
    full = np.zeros(1 << n, dtype=complex)
    full[: 1 << k] = vec  # ancillas |0>

    state = QuantumState(n)
    state.load(full)
    qc.update_quantum_state(state)
    got = state.get_vector()

    expected = full.copy()
    expected[(1 << k) - 1] *= -1
    assert np.allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize("k", [4, 6, 8])
def test_grover_matches_theory(k):
    r = run_grover(k, seed=1, n_shots=4000)
    assert abs(r.success_prob - r.theory_prob) < 0.05
    assert r.theory_prob > 0.9


def test_success_prob_formula():
    # exact for k=2: one iteration finds the target with certainty
    assert theoretical_success(2, optimal_iterations(2)) == pytest.approx(1.0)


def test_qvi_anchored_recommends_level3_at_192bit_floor():
    df = compute_qvi(required_bits=192)
    rec = df[df["recommended"]]
    assert list(rec["level"]) == ["Dilithium3"]


def test_qvi_raw_formula_actually_prefers_level5():
    """Documents the proposal's flaw: the original bits^2/cost formula picks
    the highest level once real (near-linear) costs are used."""
    df = compute_qvi(required_bits=192)
    assert df.loc[df["qvi_raw"].idxmax(), "level"] == "Dilithium5"
