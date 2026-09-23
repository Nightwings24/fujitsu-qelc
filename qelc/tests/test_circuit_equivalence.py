"""The gate-based QAOA circuit must reproduce exp(-i gamma H) exactly.

Reference: apply the dense diagonal phase directly to the statevector
(what the old notebook's DiagonalMatrix did — valid locally, unsupported on
the cluster). The gate circuit must match it to numerical precision, and the
Observable-based expectation must match the diagonal expectation.
"""

import numpy as np
from qulacs import QuantumCircuit, QuantumState

from qelc.qaoa.circuit import build_qaoa_circuit, expectation, run_state
from qelc.qubo.ising import dense_diagonal, qubo_to_ising

from conftest import random_qubo


def reference_state(isg, gammas, betas):
    n = isg.n
    diag = dense_diagonal(isg) - isg.offset  # circuit implements H without identity offset
    state = QuantumState(n)
    state.set_zero_state()
    h_layer = QuantumCircuit(n)
    for i in range(n):
        h_layer.add_H_gate(i)
    h_layer.update_quantum_state(state)
    for g, b in zip(gammas, betas):
        vec = state.get_vector() * np.exp(-1j * g * diag)
        state.load(vec)
        mix = QuantumCircuit(n)
        for i in range(n):
            mix.add_RX_gate(i, -2.0 * b)
        mix.update_quantum_state(state)
    return state


def test_gate_circuit_matches_diagonal_reference():
    for seed, p in [(0, 1), (1, 2), (2, 3)]:
        q = random_qubo(7, seed=seed)
        isg = qubo_to_ising(q)
        rng = np.random.default_rng(seed)
        gammas = rng.uniform(0.1, 1.2, p)
        betas = rng.uniform(0.1, 1.2, p)

        ref = reference_state(isg, gammas, betas)
        params = np.empty(2 * p)
        params[0::2], params[1::2] = gammas, betas
        got = run_state(isg, params)

        fidelity = abs(np.vdot(ref.get_vector(), got.get_vector()))
        assert fidelity > 1 - 1e-9, f"seed {seed} p={p}: fidelity {fidelity}"


def test_observable_expectation_matches_diagonal():
    q = random_qubo(8, seed=5)
    isg = qubo_to_ising(q)
    params = np.array([0.4, 0.7, 0.9, 0.3])
    e_obs = expectation(isg, params)

    state = run_state(isg, params)
    probs = np.abs(state.get_vector()) ** 2
    e_diag = float(probs @ dense_diagonal(isg))
    assert abs(e_obs - e_diag) < 1e-9
