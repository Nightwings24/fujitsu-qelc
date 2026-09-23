#!/usr/bin/env python
"""Build the experiment-matrix problem instances LOCALLY (not on the cluster).

Instances with n <= 26 are exact-solved so the cluster can report certified
approximation ratios without re-solving. Cutting instances additionally store
a two-block partition with <= --max-crossing crossing conflict edges.

    python make_instances.py --out instances/ [--csv data/token_transfers.csv
        --token 0x... --decimals 6]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qelc.classical.exact import brute_force_optimum
from qelc.data.cycles import enumerate_cycles, select_top_k, select_top_k_hub, select_top_k_small_cut
from qelc.data.synthetic import barabasi_debt_graph
from qelc.io.instance import save_instance
from qelc.qubo.builder import build_qubo

MATRIX = [  # (name, n_qubits)
    ("E0_n16", 16),
    ("E1_n24", 24),
    ("E2_n26", 26),
    ("E3_n28", 28),
    ("E4_n30", 30),
    ("E5_n31", 31),
    ("E6_n32", 32),
    ("E7_n33", 33),
    ("E8_n34", 34),
]


def graph_for(args, seed: int):
    if args.csv:
        from qelc.data.loader import build_debt_graph, load_token_transfers

        edges = load_token_transfers(args.csv, token_address=args.token, decimals=args.decimals)
        G, _ = build_debt_graph(edges)
        return G
    return barabasi_debt_graph(150, m=4, reciprocal_frac=0.5, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="instances")
    ap.add_argument("--csv"), ap.add_argument("--token"), ap.add_argument("--decimals", type=int, default=18)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--length-bound", type=int, default=5)
    ap.add_argument("--max-crossing", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = graph_for(args, args.seed)
    cycles = enumerate_cycles(G, length_bound=args.length_bound)
    print(f"[instances] {len(cycles)} candidate cycles")

    for name, n in MATRIX:
        # Hub-restricted selection: cycles through the top hub node overlap
        # heavily, giving the dense conflict structure the proposal targets
        # (unrestricted top-K is nearly conflict-free and trivially solvable).
        top = select_top_k_hub(G, cycles, n, n_hubs=1)
        if len(top) < n:
            print(f"[instances] SKIP {name}: only {len(top)} hub cycles")
            continue
        qubo = build_qubo(top)
        exact = brute_force_optimum(qubo) if n <= 26 else None
        save_instance(
            qubo, str(out / f"{name}.json"), name=name, exact=exact,
            source=("csv:" + str(args.token) if args.csv else f"ba(150,4,rf=0.5,seed={args.seed})+hub1"),
            length_bound=args.length_bound,
        )
        tag = f" exact={exact.qubo_cost:.6f}" if exact else ""
        print(f"[instances] wrote {name}.json (n={n}, conflicts={len(qubo.quadratic)}){tag}")

    # cutting-demo instance: 24 qubits, two blocks, few crossing conflicts.
    # Drawn from two hubs so each block is dense internally but the blocks
    # barely interact — the realistic circuit-cutting scenario.
    vols = dict(G.degree(weight="weight"))
    hubs = sorted(vols, key=vols.get, reverse=True)[:2]
    hub_pool = [c for c in cycles if set(hubs) & set(c.nodes)]
    sel, labels, n_cross = select_top_k_small_cut(
        hub_pool, 24, pool_size=min(60, len(hub_pool)), max_crossing=args.max_crossing, seed=args.seed
    )
    qubo = build_qubo(sel)
    exact = brute_force_optimum(qubo)
    plain = build_qubo(select_top_k(cycles, 24))
    sacrifice = 1.0 - sum(c.savings for c in sel) / sum(c.savings for c in plain.cycles)
    save_instance(
        qubo, str(out / "E10_n24_cut.json"), name="E10_n24_cut", partition=labels, exact=exact,
        n_crossing=n_cross, objective_sacrifice_frac=round(sacrifice, 4),
    )
    print(f"[instances] wrote E10_n24_cut.json (crossing={n_cross}, "
          f"cut-friendliness sacrifice={sacrifice:.2%})")


if __name__ == "__main__":
    main()
