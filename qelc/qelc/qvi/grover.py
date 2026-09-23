"""Grover key search built from gates supported on the distributed simulator.

Module II's honest, runnable core: measure the cost of quantum brute-force
key search empirically at small key sizes, verify success probabilities
against theory, and let the exponential wall speak for itself when
extrapolated to 128/192/256-bit security.

Native TOFFOLI is unsupported on the FX700 backends, so the multi-controlled
gates are decomposed to {H, T, Tdag, CNOT} via the textbook 6-CNOT Toffoli and
a V-chain over clean ancillas. Qubit budget for a k-bit key: k + max(k-3, 0)
ancillas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import derive_seed, rng_for


def _toffoli(qc, a: int, b: int, c: int) -> None:
    """Standard 6-CNOT Toffoli decomposition (H/T/Tdag/CNOT only)."""
    qc.add_H_gate(c)
    qc.add_CNOT_gate(b, c)
    qc.add_Tdag_gate(c)
    qc.add_CNOT_gate(a, c)
    qc.add_T_gate(c)
    qc.add_CNOT_gate(b, c)
    qc.add_Tdag_gate(c)
    qc.add_CNOT_gate(a, c)
    qc.add_T_gate(b)
    qc.add_T_gate(c)
    qc.add_H_gate(c)
    qc.add_CNOT_gate(a, b)
    qc.add_T_gate(a)
    qc.add_Tdag_gate(b)
    qc.add_CNOT_gate(a, b)


def _mcx(qc, controls: list, target: int, ancillas: list) -> None:
    """Multi-controlled X via V-chain of Toffolis over clean ancillas
    (computed then uncomputed, so ancillas return to |0>)."""
    m = len(controls)
    if m == 0:
        qc.add_X_gate(target)
        return
    if m == 1:
        qc.add_CNOT_gate(controls[0], target)
        return
    if m == 2:
        _toffoli(qc, controls[0], controls[1], target)
        return
    need = m - 2
    assert len(ancillas) >= need, f"need {need} ancillas for {m} controls"
    chain = []
    _toffoli(qc, controls[0], controls[1], ancillas[0])
    chain.append((controls[0], controls[1], ancillas[0]))
    for i in range(2, m - 1):
        _toffoli(qc, controls[i], ancillas[i - 2], ancillas[i - 1])
        chain.append((controls[i], ancillas[i - 2], ancillas[i - 1]))
    _toffoli(qc, controls[m - 1], ancillas[m - 3], target)
    for a, b, c in reversed(chain):
        _toffoli(qc, a, b, c)


def _mcz_all_ones(qc, qubits: list, ancillas: list) -> None:
    """Phase flip iff all `qubits` are |1> : H on the last, MCX, H."""
    *controls, target = qubits
    qc.add_H_gate(target)
    _mcx(qc, controls, target, ancillas)
    qc.add_H_gate(target)


def n_qubits_for_key(k: int) -> int:
    return k + max(k - 3, 0)


def grover_circuit(k: int, secret: int, n_iters: int):
    """Grover search for `secret` among 2^k keys. Key register = qubits
    0..k-1, ancillas above. Returns a qulacs QuantumCircuit."""
    from qulacs import QuantumCircuit

    n = n_qubits_for_key(k)
    key = list(range(k))
    anc = list(range(k, n))
    qc = QuantumCircuit(n)

    for q in key:
        qc.add_H_gate(q)

    zero_bits = [q for q in key if not ((secret >> q) & 1)]
    for _ in range(n_iters):
        # oracle: phase flip on |secret>
        for q in zero_bits:
            qc.add_X_gate(q)
        _mcz_all_ones(qc, key, anc)
        for q in zero_bits:
            qc.add_X_gate(q)
        # diffuser: reflection about the uniform superposition
        for q in key:
            qc.add_H_gate(q)
            qc.add_X_gate(q)
        _mcz_all_ones(qc, key, anc)
        for q in key:
            qc.add_X_gate(q)
            qc.add_H_gate(q)
    return qc


def optimal_iterations(k: int) -> int:
    theta = np.arcsin(2 ** (-k / 2))
    return max(1, int(np.floor(np.pi / (4 * theta))))


def theoretical_success(k: int, n_iters: int) -> float:
    theta = np.arcsin(2 ** (-k / 2))
    return float(np.sin((2 * n_iters + 1) * theta) ** 2)


@dataclass
class GroverRun:
    k: int
    n_qubits: int
    n_iters: int
    success_prob: float
    theory_prob: float
    gates_per_iteration: int
    wall_time_s: float
    time_per_iteration_s: float


def run_grover(k: int, *, seed: int = 0, n_iters: int | None = None, n_shots: int = 2000) -> GroverRun:
    """Build, run, and measure one Grover search with a seeded random secret."""
    from qulacs import QuantumState

    rng = rng_for(seed, 0x60E0, k)
    secret = int(rng.integers(0, 1 << k))
    iters = n_iters if n_iters is not None else optimal_iterations(k)

    t0 = time.perf_counter()
    qc = grover_circuit(k, secret, iters)
    n = n_qubits_for_key(k)
    state = QuantumState(n)
    state.set_zero_state()
    qc.update_quantum_state(state)
    samples = state.sampling(n_shots, derive_seed(seed, 0x60E1, k))
    wall = time.perf_counter() - t0

    mask = (1 << k) - 1
    hits = sum(1 for s in samples if (s & mask) == secret)
    per_iter = qc.get_gate_count() / iters if iters else 0

    return GroverRun(
        k=k,
        n_qubits=n,
        n_iters=iters,
        success_prob=hits / n_shots,
        theory_prob=theoretical_success(k, iters),
        gates_per_iteration=int(per_iter),
        wall_time_s=wall,
        time_per_iteration_s=wall / iters,
    )


def grover_resources(k_values, *, seed: int = 0, n_shots: int = 2000):
    """Empirical scaling table across key sizes; pandas DataFrame."""
    import pandas as pd

    rows = []
    for k in k_values:
        r = run_grover(k, seed=seed, n_shots=n_shots)
        rows.append(
            {
                "key_bits": r.k,
                "sim_qubits": r.n_qubits,
                "grover_iterations": r.n_iters,
                "gates_per_iteration": r.gates_per_iteration,
                "total_gates": r.gates_per_iteration * r.n_iters,
                "success_prob": r.success_prob,
                "theory_prob": r.theory_prob,
                "wall_time_s": round(r.wall_time_s, 4),
            }
        )
    return pd.DataFrame(rows)
