"""Gate-based QAOA circuit for an Ising cost Hamiltonian.

The phase separator exp(-i gamma H) factorizes term-by-term because all terms
are diagonal and commute:

    exp(-i gamma h_i Z_i)        ->  RZ(i, 2 gamma h_i)
    exp(-i gamma J_ij Z_i Z_j)   ->  CNOT(i,j) RZ(j, 2 gamma J_ij) CNOT(i,j)

Gate count per layer: n RZ + 3|J| two-qubit-block + n RX — polynomial, and
every gate is in the supported set of the distributed (MPI) Qulacs backends,
unlike the all-qubit DiagonalMatrix this replaces.

qulacs convention: add_RZ_gate(i, angle) applies exp(+i angle/2 Z), so the
angle for exp(-i theta Z) is  -2 theta. Same sign for RX in the mixer:
exp(-i beta X) -> add_RX_gate(i, -2 beta).
"""

from __future__ import annotations

import numpy as np

from ..qubo.ising import Ising, to_qulacs_observable


def build_qaoa_circuit(isg: Ising, gammas, betas) -> "object":
    from qulacs import QuantumCircuit

    assert len(gammas) == len(betas)
    qc = QuantumCircuit(isg.n)
    for i in range(isg.n):
        qc.add_H_gate(i)
    for gamma, beta in zip(gammas, betas):
        # cost layer: exp(-i gamma H)
        for i, hi in enumerate(isg.h):
            if hi != 0.0:
                qc.add_RZ_gate(i, -2.0 * gamma * float(hi))
        for (i, j), coeff in isg.J.items():
            if coeff != 0.0:
                qc.add_CNOT_gate(i, j)
                qc.add_RZ_gate(j, -2.0 * gamma * float(coeff))
                qc.add_CNOT_gate(i, j)
        # mixer: exp(-i beta sum X)
        for i in range(isg.n):
            qc.add_RX_gate(i, -2.0 * beta)
    return qc


def run_state(isg: Ising, params: np.ndarray) -> "object":
    """Prepare |QAOA(params)> ; params = [g1, b1, g2, b2, ...].

    Set QELC_MULTI_CPU=1 (done by fx700/job.sh) to distribute the state
    vector across MPI ranks on the cluster — required beyond 30 qubits, where
    a single rank cannot hold the vector.
    """
    import os

    from qulacs import QuantumState

    p = len(params) // 2
    gammas, betas = params[0::2][:p], params[1::2][:p]
    if os.environ.get("QELC_MULTI_CPU") == "1":
        state = QuantumState(isg.n, use_multi_cpu=True)
    else:
        state = QuantumState(isg.n)
    state.set_zero_state()
    build_qaoa_circuit(isg, gammas, betas).update_quantum_state(state)
    return state


def expectation(isg: Ising, params: np.ndarray, *, _obs_cache: list = []) -> float:
    """<H> via qulacs Observable — no state-vector readout, so the identical
    computation runs on distributed backends.

    The one-slot cache pins the Ising object itself and compares by identity,
    so a recycled id can never alias a stale observable.
    """
    if _obs_cache and _obs_cache[0] is isg:
        obs = _obs_cache[1]
    else:
        obs = to_qulacs_observable(isg)
        _obs_cache[:] = [isg, obs]
    state = run_state(isg, params)
    return float(obs.get_expectation_value(state).real) + isg.offset
