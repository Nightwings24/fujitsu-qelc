#!/usr/bin/env python
"""E10: QARP circuit-cutting demonstration.

Takes a cutting-friendly instance (built by make_instances.py with a two-block
partition and <= 6 crossing conflict edges), constructs the p=1 QAOA circuit
at fixed angles as a pytket circuit, cuts the crossing RZZ gates with QARP,
and reconstructs the cost expectation. On instances small enough to also run
uncut (<= ~26 qubits), reports reconstruction error vs the exact expectation.

    python -u run_cutting_demo.py --instance instances/E10_n24_cut.json \
        --params params/E2_best.json --shots 4000 --out results/E10.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from qelc.io.instance import load_instance
from qelc.qubo.ising import to_qubit_operator


def build_tket_qaoa(isg, gamma: float, beta: float):
    """p=1 QAOA circuit as a pytket Circuit (Rz/CX/Rx/H only).

    pytket angles are in half-turns: Rz(a) = exp(-i pi a Z / 2), so the angle
    for exp(-i theta Z) is a = 2 theta / pi.
    """
    from pytket import Circuit

    c = Circuit(isg.n)
    for q in range(isg.n):
        c.H(q)
    for i, hi in enumerate(isg.h):
        if hi != 0.0:
            c.Rz(2.0 * gamma * float(hi) / np.pi, i)
    for (i, j), coeff in isg.J.items():
        if coeff != 0.0:
            c.CX(i, j)
            c.Rz(2.0 * gamma * float(coeff) / np.pi, j)
            c.CX(i, j)
    for q in range(isg.n):
        c.Rx(2.0 * beta / np.pi, q)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--params", default=None, help="JSON {'params': [gamma, beta]}; "
                    "optional — the demo measures cut-vs-uncut fidelity at fixed "
                    "angles, so any angle works if a predecessor run's file is absent")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--compare-uncut", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qubo, isg, inst = load_instance(args.instance)
    if inst.partition is None:
        raise SystemExit("instance has no partition — rebuild with make_instances.py --cutting")
    import os

    if args.params and os.path.exists(args.params):
        with open(args.params) as f:
            gamma, beta = json.load(f)["params"][:2]
    else:
        # fidelity of reconstruction is angle-independent; use a fixed demo angle
        gamma, beta = 0.4, 0.6
        print(f"[E10] --params {args.params!r} absent; using fixed demo angles "
              f"(gamma={gamma}, beta={beta}) — reconstruction fidelity is angle-independent")

    labels = np.array(inst.partition)
    crossing = [
        (i, j) for (i, j) in isg.J if labels[i] != labels[j] and isg.J[(i, j)] != 0.0
    ]
    print(f"[E10] n={inst.n}, blocks {int((labels == 0).sum())}/{int((labels == 1).sum())}, "
          f"crossing RZZ terms: {len(crossing)}")
    if len(crossing) > 6:
        raise SystemExit("more than 6 crossing terms — QARP caps cuts at 6; rebuild the instance")

    circuit = build_tket_qaoa(isg, gamma, beta)
    op = to_qubit_operator(isg)

    from pytket.extensions.qulacs import QulacsBackend
    from qarp.cutting import EAPartitioning, QPDDecomposition

    block_a = [i for i in range(inst.n) if labels[i] == 0]
    block_b = [i for i in range(inst.n) if labels[i] == 1]

    t0 = time.perf_counter()
    cutter = EAPartitioning(
        original_circuit=circuit,
        max_size_subcircuits=[len(block_a), len(block_b)],
    )
    cut_result = cutter.cut(manual_setting=[block_a, block_b])
    t_cut = time.perf_counter() - t0

    t0 = time.perf_counter()
    post = QPDDecomposition(
        cutter_result=cut_result, observable=op, backend=QulacsBackend(), n_shots=args.shots
    )
    post.decompose()
    reconstructed = float(post.compute(parallelize=True))
    t_compute = time.perf_counter() - t0

    result = {
        "instance": inst.name,
        "n": inst.n,
        "n_crossing": len(crossing),
        "sampling_overhead": float(post.overhead()),
        "reconstructed_expectation": reconstructed,
        "t_cut": t_cut,
        "t_compute": t_compute,
        "shots": args.shots,
    }

    if args.compare_uncut:
        from qelc.qaoa.circuit import expectation as local_expectation

        exact = local_expectation(isg, np.array([gamma, beta]))
        result["uncut_expectation"] = exact
        result["abs_error"] = abs(reconstructed - exact)
        print(f"[E10] reconstructed {reconstructed:.6f} vs uncut {exact:.6f} "
              f"(|err| {abs(reconstructed - exact):.6f})")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[E10] wrote {args.out}")


if __name__ == "__main__":
    main()
