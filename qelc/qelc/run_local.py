"""Local end-to-end benchmark: build a debt graph, select top-K cycles, solve
the MWIS QUBO with every solver, and print the honest comparison table.

    python -m qelc.run_local --dataset synthetic --k 20 --solver all
    python -m qelc.run_local --dataset csv --csv data/token_transfers.csv \
        --token 0x... --decimals 6 --k 20
"""

from __future__ import annotations

import argparse
import json
import time

from .classical.exact import brute_force_optimum
from .classical.greedy import greedy_mwis
from .classical.random_baseline import random_baseline
from .data.cycles import enumerate_cycles, select_top_k
from .data.synthetic import barabasi_debt_graph
from .engine.metrics import report_table
from .engine.peeling import PeelingConfig, run_peeling
from .engine.audit import audit_log
from .qaoa.optimize import solve_qaoa
from .qubo.builder import build_qubo


def build_graph(args):
    if args.dataset == "synthetic":
        return barabasi_debt_graph(args.n_nodes, m=args.ba_m, seed=args.seed)
    from .data.loader import build_debt_graph, load_token_transfers

    edges = load_token_transfers(
        args.csv, token_address=args.token, decimals=args.decimals, max_rows=args.max_rows
    )
    G, _ = build_debt_graph(edges)
    return G


def single_batch(G, args) -> None:
    cycles = enumerate_cycles(G, length_bound=args.length_bound)
    top = select_top_k(cycles, args.k)
    print(f"[qelc] {len(cycles)} cycles found, using top {len(top)} (K={args.k})")
    qubo = build_qubo(top)
    print(f"[qelc] QUBO: {qubo.n} vars, {len(qubo.quadratic)} conflicts")

    solutions = {}
    opt = brute_force_optimum(qubo) if qubo.n <= 26 else None
    if opt is not None:
        solutions["exact"] = opt
    solutions["greedy"] = greedy_mwis(qubo)
    solutions["random"] = random_baseline(qubo, n_samples=args.shots, seed=args.seed)
    qaoa_sol, results = solve_qaoa(
        qubo, p=args.p, n_starts=args.n_starts, maxiter=args.maxiter,
        n_shots=args.shots, seed=args.seed,
    )
    solutions["qaoa"] = qaoa_sol

    table = report_table(solutions, opt)
    print("\n" + table.to_string(index=False))
    for r in results:
        print(f"[qaoa] p={r.p}: energy {r.energy:.6f} after {r.n_evals} evals ({r.wall_time:.2f}s)")


def peeling(G, args) -> None:
    for solver in args.peel_solvers.split(","):
        cfg = PeelingConfig(
            k=args.k, p=args.p, max_iters=args.max_iters, solver=solver,
            master_seed=args.seed, length_bound=args.length_bound,
            n_starts=args.n_starts, maxiter=args.maxiter, n_shots=args.shots,
        )
        t0 = time.time()
        report = run_peeling(G, cfg, verbose=args.verbose)
        audit = audit_log(G, report.log, report.total_cleared)
        print(
            f"[peel:{solver:>6}] cleared {report.total_cleared:,.2f} "
            f"(gross {report.gross_reduction:,.2f}, {report.netting_efficiency_pct:.2f}% of "
            f"{report.initial_debt:,.2f}) in {report.n_iterations} iters, "
            f"{report.n_cycles_executed} cycles, {time.time()-t0:.1f}s | "
            f"audit: {'PASS' if audit['passed'] else 'FAIL ' + str(audit['failures'][:2])}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "csv"], default="synthetic")
    ap.add_argument("--csv"), ap.add_argument("--token"), ap.add_argument("--decimals", type=int, default=18)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--n-nodes", type=int, default=120)
    ap.add_argument("--ba-m", type=int, default=3)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--length-bound", type=int, default=5)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["batch", "peel", "both"], default="batch")
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument("--peel-solvers", default="greedy,qaoa")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    G = build_graph(args)
    debt = sum(d["weight"] for _, _, d in G.edges(data=True))
    print(f"[qelc] graph: {G.number_of_nodes()}V {G.number_of_edges()}E, gross debt {debt:,.2f}")

    if args.mode in ("batch", "both"):
        single_batch(G, args)
    if args.mode in ("peel", "both"):
        peeling(G, args)


if __name__ == "__main__":
    main()
