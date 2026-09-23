# QELC — FX700 / QARP execution kit

Everything the team runs on the Fujitsu FX700 cluster. **Only results obtained
through QARP are scored** (QSC setup guide, Important Notice #1), so every
scored run goes through `qarp.algorithms.QAOA` / QARP engines; plain-qulacs
paths exist only as documented fallbacks for final-state sampling.

## One-time setup (per the QSC_FX700_QARP setup guide)

1. Follow the guide's Part 1 (pyenv 3.12.10, venv, copy `qarp` into
   site-packages, `pip install -r requirements_mpi.txt`, build MPI-enabled
   Qulacs from the bundled source).
2. Install this project into the same venv:
   `pip install -e ~/qelc` (rsync the `qelc/` package + this `fx700/` dir to
   your home first; instances are pre-built JSON, no data download needed).
3. Smoke test on a compute node (salloc -N 1 -p Interactive):
   `python validate_qarp_energy.py --instance instances_usdc/E0_n16.json`
   This must print `MATCH` before anything else is submitted. It pins down
   QARP's QAOA angle/sign conventions against our locally verified circuit.

## Run order (experiment matrix)

| ID  | qubits | nodes | what                                                | job file          |
|-----|--------|-------|-----------------------------------------------------|-------------------|
| E0  | 16     | 1     | convention check, fixed params                      | jobs/e0.job       |
| E1  | 24     | 1     | p=1 grid->COBYLA, INTERP->p=2; vs shipped exact     | jobs/e1.job       |
| E2  | 26     | 1     | same, largest exactly-verified size                 | jobs/e2.job       |
| E3  | 28     | 1     | transferred init from E2, maxiter 40                | jobs/e3.job       |
| E4  | 30     | 2     | 2 nodes (16 GiB vector); p=1, maxiter <=20          | jobs/e4.job       |
| E5  | 31     | 2     | transferred angles, maxiter <=20                    | jobs/e5.job       |
| E6  | 32     | 4     | transferred angles, maxiter <=15                    | jobs/e6.job       |
| E7  | 33     | 8     | transferred angles, maxiter <=10                    | jobs/e7.job       |
| E8  | 34     | 16    | fixed angles, sample only (needs -t 48:00:00)       | jobs/e8.job       |
| E9  | 31     | 2/4/8 | strong scaling, identical config                    | jobs/e9_*.job     |
| E10 | 24     | 1     | QARP circuit-cutting demo (<=6 crossing RZZ, p=1)   | jobs/e10.job      |
| G1  | k=8-15 | 1+    | Grover key-search scaling (Module II)               | jobs/g1.job       |

Rules of thumb from the platform docs: power-of-2 MPI ranks only; Batch queue
(6 h default walltime, request more with -t); init overhead grows steeply with
node count (33-qubit VQE reference: ~3 h init on 512 nodes) — that is why
>=31-qubit rows use transferred angles with few or zero optimizer iterations.

`run_matrix.py` regenerates all job files; `collect_results.py` merges the
per-run JSON outputs into results.csv and the scaling plots.

## Instances

Two sets, both built locally by `make_instances.py` (needs the local qelc
install + dataset), both shipping the certified optimal cost/bitstring for
n <= 26 so approximation ratios compute on the cluster without re-solving:

- `instances_usdc/*.json` — **default**, real Ethereum-USDC settlement data.
- `instances/*.json` — synthetic Barabási–Albert (controlled scaling).

`run_matrix.py` bakes the chosen set into the job files (`--instances-dir`,
default `instances_usdc`). `run_qaoa_qarp.py` auto-writes
`params/<instance>_best.json` after each run so the next-larger job transfers
its angles automatically. `collect_results.py` (stdlib-only, no pandas) merges
`results/*.json` into `results.csv`.
