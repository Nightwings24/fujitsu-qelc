# QELC report — errata & required revisions

Apply these to the LaTeX source of `fujitsu-report.pdf` before the next
submission. Ordered by severity. Items marked **[data]** get their numbers
from the new `qelc` pipeline / FX700 runs.

## 1. Module II must be rewritten (conceptual error)

The report claims to "benchmark CRYSTALS-Dilithium standards on the Fujitsu
Simulator" and speaks of Dilithium's "Quantum Gas (Gate Count)". Dilithium is
a *classical* lattice-based signature scheme — it does not run on a quantum
simulator, and reviewers with quantum expertise will flag this immediately.

Replacement narrative (implemented in `qelc/qvi/`):
- **What runs on the simulator:** Grover key-search circuits at increasing key
  sizes k (built only from gates the FX700 backends support), verifying the
  sqrt(2^k) iteration scaling empirically and measuring the wall-time-per-
  qubit exponential wall. **[data: fx700 run G1 + local `grover_resources`]**
- **Security anchor:** published quantum attack-cost estimates for the AES
  instances that define NIST categories 1/3/5 (Grassl et al. 2016: ~2^86,
  ~2^118, ~2^151 T-gates; optionally Jaques et al. 2020). Our Grover runs
  demonstrate at small scale the scaling law those estimates extrapolate.
- **Cost side of QVI:** the *classical operational* cost of each Dilithium
  level (sign/verify time, signature+key bytes) from the NIST round-3
  reference benchmarks, or better, measured by the team via liboqs.

## 2. The QVI conclusion changes (and that's a feature)

With real cost data, the report's own formula QVI = bits^2 / (a·gates + b·
latency) selects **Dilithium5**, not Dilithium3: Dilithium's operational costs
grow roughly *linearly* with level (report Table 2 wrongly calls them
"exponential"), so the squared reward always wins. Present both:
- `qvi_raw` (original formula) — honestly report that it favors Level 5;
- `qvi_anchored` (security utility saturates at the mandated 192-bit floor) —
  selects **Dilithium3**, restoring the "Level 3 is the banking standard"
  conclusion with a defensible economic argument (bits above the compliance
  floor have bounded marginal value).
Table 2's Low/Medium/High qualitative entries must be replaced by the numeric
table from `qelc.qvi.compute_qvi()`. **[data]**

## 3. Methodology corrections (Module I)

1. **Penalty formula.** Report Eq. (2) says P_safe = sum_i S_i + delta; the
   implementation uses a pairwise penalty. Replace Eq. (2) with
   `P_ij = min(w_i, w_j) + eps` and add the dominance proof (any violated
   conflict is strictly improved by switching off its lighter endpoint, hence
   every QUBO optimum is feasible — QUBO optimum == MWIS optimum). One
   paragraph; the proof is in `qelc/qubo/builder.py` and is property-tested.
2. **Objective weights.** The old code optimized log-scaled savings, so the
   QUBO optimum was *not* the maximum-dollar netting. State that weights are
   linear (max-normalized); log scaling may be mentioned as a rejected
   alternative.
3. **Cycle enumeration.** "We traverse the graph classically to identify all
   elementary cycles" — must state the length bound (<= 5) and justify it
   (cycle value = min edge weight, so long cycles are low-value; enumeration
   stays polynomial in practice).
4. **Problem framing.** Name the batch problem: Maximum-Weight Independent
   Set on the cycle-conflict graph. This makes the benchmarking story crisp
   and connects to known QAOA-for-MIS literature.
5. **Dense-hub selection.** The report's "rich cycles in dense hubs" claim is
   now real: instances are built from cycles through the top hub node(s),
   which is what produces dense conflict graphs (unrestricted top-K is nearly
   conflict-free and trivial). Describe `select_top_k_hub`.

## 4. Data section must change **[data]**

- The $23.01B "initial debt" figure sums raw `value/1e18` across **28
  different ERC-20 tokens** (USDC has 6 decimals; token prices differ by
  orders of magnitude). Retract it.
- New protocol: netting runs per token (stablecoins give a defensible USD
  reading; report the token and its decimals), plus seeded Barabasi-Albert
  synthetic graphs (the topology the proposal promised) for controlled
  scaling experiments. Cite `qelc/data/loader.py` semantics.

## 5. Claims to remove or reword

- "For dense subgraphs, QAOA is mathematically proven to achieve a higher
  approximation ratio than basic greedy algorithms" — no such theorem exists
  for this problem; at p=2 nothing is proven at all. Replace with the
  *measured* approximation-ratio / P(optimal) tables vs greedy, exact, and a
  random-sampling-plus-repair null baseline. **[data]**
- "tunnel through high-cost barriers" — annealing language; QAOA does not
  tunnel. Reword to interference-based amplification of low-cost states.
- "reduces idle capital requirements by over 30%" — the audited pipeline
  achieved ~10-29% netting efficiency depending on instance; report measured
  numbers, cite McMahon et al. (C$240M/day) as *literature*, not as own
  results. **[data]**
- "Polynomial scaling (Circuit Depth)" in Table 3 — the circuit is polynomial
  but simulation cost is exponential in qubits (that is why the FX700 exists)
  and QAOA solution quality at fixed p is heuristic. State both honestly.
- FIFO / FIFO-Bypass / SCIP benchmarking is claimed but was never run. Either
  implement queue-replay baselines or replace the claim with the implemented
  baseline set (greedy MWIS, exact/MILP, random+repair). **[data]**

## 6. Compliance / architecture section **[data]**

- All scored results must state they were obtained **through QARP** (QAOA
  composite via `qarp.algorithms.QAOA`; cutting via `EAPartitioning` +
  `QPDDecomposition`) on MPI-enabled Qulacs — Fujitsu's proprietary
  mpi-qulacs is not a scored path.
- Replace the aspirational "Projected QARP Utilization" section with actual
  results: the E10 cutting demo (report crossing-cut count, 6^cuts sampling
  overhead, reconstruction error vs uncut, and the measured "cut-friendliness
  costs X% of objective" trade-off from `make_instances.py`).
- The old implementation's all-qubit DiagonalMatrix gate is unsupported on
  the distributed simulator (1-target only); the new gate-based cost layer
  (RZ + CNOT-RZ-CNOT) is what runs. Worth one sentence in the architecture
  section — it shows platform awareness.
- Update Figure 1 pipeline to match the implemented flow (hub-restricted
  candidate selection, QUBO->Ising->QubitOperator, QARP QAOA, seeded
  sampling + repair extraction, iterative peeling with exclusion sets,
  independent audit).

## 7. Reproducibility statement (add)

One paragraph: every stochastic step (graph generation, optimizer starts,
final-state sampling) is seeded from a single master seed; identical runs
produce byte-identical results; an independent auditor replays every executed
cycle against the pristine ledger and checks conservation invariants. (The
old notebook produced $813.82M vs $683.88M on identical reruns; judges can
verify the fix.)
