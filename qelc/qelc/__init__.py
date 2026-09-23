"""QELC v2 — Quantum-Enhanced Liquidity Clearinghouse.

Pure-Python problem layer (data -> cycles -> QUBO -> Ising) with two runtimes:
  * local: plain qulacs (gate-based QAOA, statevector)
  * FX700: the same Ising handed to Fujitsu QARP (see fx700/ scripts)

Design notes are in the project plan; the headline correctness rules:
  * objective weights are linear (max-normalized), so the QUBO optimum is the
    maximum-savings cycle selection;
  * conflicts (shared directed edge) make the problem a Max-Weight Independent
    Set; the penalty P_ij = min(w_i, w_j) + eps guarantees every QUBO optimum
    is feasible (see qelc.qubo.builder for the dominance argument);
  * everything is deterministic given a master seed.
"""

__version__ = "2.0.0"
