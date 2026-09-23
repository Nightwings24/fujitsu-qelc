#!/usr/bin/env python
"""G1 (Module II): Grover key-search scaling on the FX700.

Runs qelc.qvi.grover at key sizes beyond local reach and records wall time,
gate counts, and measured vs theoretical success probability. Key size k uses
2k-3 qubits (V-chain ancillas), so k=15 -> 27 qubits fits one node; larger k
needs multi-node MPI ranks.

    mpirun ... python -u run_grover_qvi.py --k-min 8 --k-max 15 --out results/g1.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json

try:
    from mpi4py import MPI  # noqa: F401

    _RANK = MPI.COMM_WORLD.Get_rank()
except ImportError:
    _RANK = 0

from qelc.qvi.grover import n_qubits_for_key, run_grover


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-min", type=int, default=8)
    ap.add_argument("--k-max", type=int, default=15)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iters", type=int, default=None,
                    help="cap Grover iterations (timing runs at large k)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for k in range(args.k_min, args.k_max + 1):
        if _RANK == 0:
            print(f"[g1] k={k} ({n_qubits_for_key(k)} qubits)...", flush=True)
        r = run_grover(k, seed=args.seed, n_iters=args.max_iters, n_shots=args.shots)
        rows.append(dataclasses.asdict(r))
        if _RANK == 0:
            print(f"[g1]   success {r.success_prob:.3f} (theory {r.theory_prob:.3f}), "
                  f"{r.wall_time_s:.1f}s total, {r.time_per_iteration_s:.3f}s/iter", flush=True)

    if _RANK == 0:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[g1] wrote {args.out}")


if __name__ == "__main__":
    main()
