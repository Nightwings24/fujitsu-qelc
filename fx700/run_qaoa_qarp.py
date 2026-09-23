#!/usr/bin/env python
"""QARP-scored QAOA driver for the FX700.

Loads a pre-built QELC instance (JSON), optimizes the QAOA angles through
qarp.algorithms.QAOA (the scored execution path), then samples the optimized
state to extract an actual netting solution.

Usage (inside a Slurm job; see jobs/*.job):
    python -u run_qaoa_qarp.py --instance instances/E1_n24.json --p 2 \
        --maxiter 60 --shots 10000 --seed 0 --out results/E1_seed0.json
    python -u run_qaoa_qarp.py --instance ... --init-params params.json --skip-optimize
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

# mpi4py must be imported for MPI execution even if unused directly
try:
    from mpi4py import MPI  # noqa: F401

    _RANK = MPI.COMM_WORLD.Get_rank()
except ImportError:
    _RANK = 0

from qelc.io.instance import load_instance
from qelc.qaoa.extract import extract_solution
from qelc.qubo.ising import to_qubit_operator


def log(msg: str) -> None:
    if _RANK == 0:
        print(msg, flush=True)


def _load_params(path: str | None) -> list | None:
    """Load transferred angles; tolerate a missing file so a not-yet-produced
    upstream result degrades to grid/random init instead of crashing the job."""
    if not path:
        return None
    if not os.path.exists(path):
        log(f"[qelc-fx700] WARNING: --init-params {path} not found; "
            "falling back to QARP's default (grid/random) initialization")
        return None
    with open(path) as f:
        return json.load(f)["params"]


def optimize_with_qarp(op, args) -> tuple[np.ndarray, float, dict]:
    """Scored path: QARP QAOA composite algorithm."""
    from qarp.algorithms import QAOA
    from qarp.optimizers import ScipyOptimizer

    init = _load_params(args.init_params)

    t0 = time.perf_counter()
    qaoa = QAOA(
        problem=op,
        n_layers=args.p,
        initial_parameters=init,
        optimizer=ScipyOptimizer("COBYLA", {"maxiter": args.maxiter}),
        verbose=(_RANK == 0),
    ).build()
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    energy, params = qaoa.run()
    t_run = time.perf_counter() - t0
    return np.asarray(params, dtype=float), float(energy), {"t_build": t_build, "t_optimize": t_run}


def sample_final_state(isg, params: np.ndarray, args) -> dict:
    """Measure the optimized QAOA state.

    Primary path: rebuild the (verified-equivalent) gate circuit with the
    QARP-optimized angles in the QARP-bundled Qulacs and use seeded sampling.
    The optimization itself ran through QARP; this step only reads out the
    state QARP prepared, which QAOA.run() does not expose.
    """
    from qelc.qaoa.extract import sample_bitstrings

    t0 = time.perf_counter()
    samples = sample_bitstrings(isg, params, n_shots=args.shots, seed=args.seed)
    return {"samples": {str(k): int(v) for k, v in samples.items()}, "t_sample": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-params", default=None, help="JSON file with {'params': [...]}")
    ap.add_argument("--skip-optimize", action="store_true", help="fixed-angle mode (E8)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qubo, isg, inst = load_instance(args.instance)
    log(f"[qelc-fx700] instance {inst.name}: n={inst.n}, conflicts={len(inst.conflict_edges)}")

    op = to_qubit_operator(isg)
    timings: dict = {}

    if args.skip_optimize:
        loaded = _load_params(args.init_params)
        if loaded is None:
            raise SystemExit(
                f"--skip-optimize needs angles, but {args.init_params!r} is missing. "
                "Run the smaller-qubit job first so it writes params/<inst>_best.json."
            )
        params = np.asarray(loaded, dtype=float)
        energy = float("nan")
    else:
        params, energy, timings = optimize_with_qarp(op, args)
        log(f"[qelc-fx700] QARP QAOA energy {energy:.6f} params {params.tolist()}")

    sampled = sample_final_state(isg, params, args)
    from collections import Counter

    samples = Counter({int(k): v for k, v in sampled["samples"].items()})
    sol = extract_solution(samples, qubo, source="qaoa-fx700")

    result = {
        "instance": inst.name,
        "n": inst.n,
        "p": args.p,
        "seed": args.seed,
        "params": params.tolist(),
        "qarp_energy": energy,
        "solution_bitstring": sol.bitstring,
        "solution_feasible": sol.feasible,
        "solution_savings": sol.savings,
        "solution_qubo_cost": sol.qubo_cost,
        "exact_cost": inst.exact_cost,
        "approx_ratio": (
            sol.qubo_cost / inst.exact_cost
            if inst.exact_cost not in (None, 0) and sol.feasible
            else None
        ),
        "timings": {**timings, "t_sample": sampled["t_sample"]},
        "top_samples": dict(sorted(samples.items(), key=lambda kv: -kv[1])[:100]),
    }
    if _RANK == 0:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        log(f"[qelc-fx700] wrote {args.out}")
        # Auto-publish this run's angles so downstream (larger-qubit) jobs can
        # transfer them without a manual copy step. File schema matches
        # _load_params: {"params": [...]}.
        os.makedirs("params", exist_ok=True)
        best_path = os.path.join("params", f"{inst.name}_best.json")
        with open(best_path, "w") as f:
            json.dump({"params": params.tolist()}, f)
        log(f"[qelc-fx700] wrote {best_path} (angles for angle-transfer)")


if __name__ == "__main__":
    main()
