#!/usr/bin/env python
"""E0: pin down QARP's QAOA conventions before burning node-hours.

Evaluates the QARP QAOA energy at FIXED angles and compares against the
locally verified gate-circuit expectation (qelc.qaoa.circuit), which is unit-
tested against the dense diagonal of H. If QARP uses a different angle sign /
factor-of-2 / endianness convention, this catches it immediately: try the
listed convention transforms and report which one matches.

Run on one compute node:
    python validate_qarp_energy.py --instance instances/E0_n16.json
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from qelc.io.instance import load_instance
from qelc.qaoa.circuit import expectation
from qelc.qubo.ising import to_qubit_operator

TEST_PARAMS = np.array([0.37, 0.61])  # p=1: [gamma, beta]


def qarp_energy_at(op, params: np.ndarray, p: int) -> float:
    """Energy of the QAOA state at fixed params through QARP.

    Uses the QAOA composite with maxiter=0-style evaluation: build with
    initial_parameters and an optimizer that performs no steps, then run.
    If ScipyOptimizer(maxiter=0) still perturbs parameters on your QARP
    version, fall back to maxiter=1 and compare against the local value at
    the optimizer's reported final parameters instead.
    """
    from qarp.algorithms import QAOA
    from qarp.optimizers import ScipyOptimizer

    qaoa = QAOA(
        problem=op,
        n_layers=p,
        initial_parameters=list(params),
        optimizer=ScipyOptimizer("COBYLA", {"maxiter": 0}),
        verbose=False,
    ).build()
    energy, final_params = qaoa.run()
    return float(energy), np.asarray(final_params, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    qubo, isg, inst = load_instance(args.instance)
    op = to_qubit_operator(isg)

    qarp_e, final_params = qarp_energy_at(op, TEST_PARAMS, p=1)
    print(f"[E0] QARP energy at params {final_params.tolist()}: {qarp_e:.10f}")

    # candidate convention transforms: (gamma sign, gamma scale, beta sign, beta scale)
    candidates = list(itertools.product([1, -1], [1.0, 0.5, 2.0], [1, -1], [1.0, 0.5, 2.0]))
    matches = []
    for gs, gf, bs, bf in candidates:
        local = expectation(isg, np.array([gs * gf * final_params[0], bs * bf * final_params[1]]))
        if abs(local - qarp_e) < args.tol:
            matches.append((gs, gf, bs, bf, local))

    if not matches:
        print("[E0] NO MATCH under tested conventions — inspect QARP QAOABlock source/examples")
        raise SystemExit(1)
    for gs, gf, bs, bf, local in matches:
        tag = "identity" if (gs, gf, bs, bf) == (1, 1.0, 1, 1.0) else f"gamma*{gs*gf}, beta*{bs*bf}"
        print(f"[E0] MATCH ({tag}): local {local:.10f} == qarp {qarp_e:.10f}")
    print("[E0] record the matching transform; apply it when transferring angles between runtimes")


if __name__ == "__main__":
    main()
