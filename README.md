# QELC - Quantum-Enhanced Liquidity Clearinghouse

A QAOA pipeline that nets settlement debt by cancelling payment cycles in a token-transfer graph, built to run on Fujitsu's **FX700 / QARP** distributed quantum simulator. Entry for the **Fujitsu Quantum Simulator Challenge 2025-26** (IIT Patna).

The design goal is credibility over hype: a deterministic, independently audited pipeline that reports measured numbers and benchmarks the quantum solver honestly against classical baselines, rather than claiming an advantage it did not observe.

---

## What problem this solves

Take a directed graph where each node is an account and an edge `u -> v` weighted `w` means "u owes v w units". Any directed cycle is circular debt (A owes B owes C owes A). That loop can be settled by subtracting the minimum edge weight around it from every edge - no account's net position changes, yet gross obligations shrink. This is multilateral netting, and it frees capital otherwise parked against those obligations.

The hard decision is *which* cycles to execute: two cycles sharing a directed edge compete for the same capacity. Model each candidate cycle as a node in a conflict graph, join cycles that share a debt-edge, and pick the highest-total-savings set of mutually non-conflicting cycles. That is exactly **Maximum-Weight Independent Set (MWIS)** on the conflict graph - the NP-hard core the quantum solver targets, and a canonical QAOA benchmark problem.

---

## Pipeline

```
CSV ledger -> enumerate cycles (len <= 5) -> select top-K (hub-restricted)
           -> MWIS QUBO -> Ising -> QubitOperator
           -> QAOA (optimize angles, sample) -> extract (rescore + repair)
           -> iterative peeling -> independent audit
```

Everything is deterministic given a single master seed. The problem layer is pure Python (numpy / networkx / scipy). Only the solve step differs between runtimes: plain qulacs locally, or `qarp.algorithms.QAOA` on the cluster (the only path Fujitsu scores).

### Key design decisions

- **Provably-feasible QUBO penalty.** The objective is `-sum(w_i/w_max) x_i + sum_conflicts P_ij x_i x_j` with `P_ij = min(w_i, w_j) + eps`. A one-line dominance argument shows every global optimum is feasible: flipping the lighter endpoint of any violated conflict strictly lowers cost by `eps`. So the QUBO optimum equals the MWIS optimum, and the penalty stays comparable to the reward scale (an easier QAOA landscape than an oversized barrier). The same flip is reused as the classical repair operator, and the proof is property-tested.
- **Warm-start angle optimization.** `p=1` grid scan -> COBYLA polish with multi-start -> INTERP extension to deeper `p` (Zhou et al. 2020), with an optional CVaR objective.
- **Robust extraction.** Every unique sampled bitstring is rescored exactly; the best feasible sample competes with the greedy-repaired best sample overall. This replaces the prototype's `most_common(1)` draw, which degenerates for large state spaces.
- **Corrected peeling engine.** Cycles execute at the current residual minimum (no over-counting); a stuck cycle goes into an exclusion set instead of corrupting the ledger; every stochastic step is seeded.
- **Independent audit.** A separate module replays the execution log against a pristine ledger and checks four invariants: no phantom edges, no overdraft, settlement-neutral net positions, and matching totals, with tolerances relative to billion-scale weights.

### Second module - Grover and QVI

`qelc/qvi/` replaces an earlier conceptual error (benchmarking a classical signature scheme "on the simulator") with something runnable: Grover key-search circuits built only from FX700-supported gates (Toffoli decomposed to `{H, T, Tdag, CNOT}`), verifying the `sqrt(2^k)` scaling empirically, plus a Quantum Viability Index over Dilithium levels anchored to published quantum attack-cost estimates.

---

## Results (honest framing)

Measured iterative peeling on real Ethereum-USDC settlement data, both solvers through the identical engine, both passing the independent audit:

| Solver      | Netting efficiency | Cleared | Cycles | Iterations | Wall time |
|-------------|-------------------:|--------:|-------:|-----------:|----------:|
| Greedy MWIS | 41.55%             | $3.60B  | 299    | 30         | 44 s      |
| QAOA        | 32.13%             | $2.81B  | 65     | 8          | 136 s     |

On this instance the classical greedy baseline beats QAOA. This project does not claim quantum advantage. Its value is a correct, deterministic, independently audited netting pipeline plus measured QAOA and Grover scaling on real Fujitsu QARP hardware. Approximation ratios are certified against exact / MILP optima for `n <= 26`; beyond that, QAOA quality at fixed `p` is explicitly heuristic.

The cluster experiment matrix runs QAOA through QARP from **16 to 34 qubits** (up to 16 MPI nodes), with automatic angle-transfer up the qubit ladder and a circuit-cutting demo (`EAPartitioning` + `QPDDecomposition`).

---

## Repository layout

```
qelc/                     Installable package (pip install -e "qelc/[local,milp,dev]")
  qubo/                   MWIS QUBO builder + feasibility proof, Ising conversion + operator exports
  qaoa/                   circuit, optimize (grid -> COBYLA -> INTERP), extract (rescore + repair)
  engine/                 iterative peeling loop, independent audit, metrics
  classical/              exact / greedy / random baselines (same engine, fair comparison)
  qvi/                    Grover circuits + Quantum Viability Index (Module II)
  data/                   CSV ETL loader, cycle enumeration, synthetic Barabasi-Albert graphs
  tests/                  28 tests (proofs, circuit equivalence, audit, regression)
fx700/                    Cluster kit - QARP driver, cutting demo, Slurm jobs, pre-built instances
notebooks/                qelc_v2_demo.ipynb (runnable demo) + prototype-v1 (kept for reference)
results/                  Sample outputs from local runs
docs/                     Report, self-review errata, and the HPC deployment runbook
```

---

## Quick start (local)

```bash
python3 -m venv .venv
.venv/bin/pip install -e "qelc/[local,milp,dev]"
.venv/bin/python -m pytest qelc/tests            # 28 tests, ~7 s
.venv/bin/python -m qelc.run_local --dataset synthetic --k 20 --mode both
```

Running on the FX700 cluster (the scored QARP path) is documented step by step in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## A note on what is not in this repository

To respect the Fujitsu Quantum Simulator Challenge participant agreement and to keep private data out of a public repository, the following are intentionally excluded:

- Fujitsu platform documentation and the QARP package (Fujitsu Confidential material).
- The raw multi-gigabyte SNAP `ethereum-exchanges` transfer dataset (available from the [SNAP project](https://snap.stanford.edu/); the loader in `qelc/data/loader.py` documents the expected schema).
- All account credentials, SSH keys, and personally identifiable information.

The code, problem formulation, solver design, tests, and results here are the team's own work.

---

## Tech stack

Python - QAOA / QUBO / Ising - openfermion `QubitOperator` - Qulacs (MPI-enabled via QARP) - Grover - networkx / numpy / scipy - pulp (MILP baseline) - pytest - Slurm / MPI on the Fujitsu FX700 (A64FX) cluster.

## License

[MIT](LICENSE).
