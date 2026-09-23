"""Iterative peeling engine, corrected.

Each iteration: enumerate cycles on the residual ledger, select top-K,
solve the MWIS QUBO with the configured solver, execute the selected cycles,
subtract the netted amounts, repeat.

Fixes vs the original notebook:
  * the ledger is only ever mutated by valid executions — a stuck or
    unexecutable cycle goes into an exclusion set instead of having one of
    its debt edges deleted;
  * execution amount is recomputed as the *residual* minimum along the cycle
    at execution time, so overlapping selections partially execute when
    capacity allows and totals are never over-counted;
  * every stochastic component is seeded from the master seed keyed by
    (iteration, role) — two runs with the same config are identical;
  * the solver is pluggable (qaoa | greedy | exact | random) so classical
    baselines run through the exact same engine for a fair comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from ..classical.exact import brute_force_optimum
from ..classical.greedy import greedy_mwis
from ..classical.random_baseline import random_baseline
from ..config import derive_seed
from ..data.cycles import Cycle, enumerate_cycles, select_top_k
from ..qubo.builder import build_qubo
from ..solution import Solution


@dataclass
class PeelingConfig:
    k: int = 20
    p: int = 2
    max_iters: int = 30
    solver: str = "qaoa"          # qaoa | greedy | exact | random
    master_seed: int = 42
    length_bound: int = 5
    n_starts: int = 4
    maxiter: int = 100
    n_shots: int = 10_000
    min_execute: float = 1e-9     # smallest residual amount worth executing
    penalty_eps_frac: float = 0.25
    stall_patience: int = 3       # empty selections before excluding the top cycle


@dataclass
class PeelingReport:
    initial_debt: float
    final_debt: float
    total_cleared: float
    n_iterations: int
    n_cycles_executed: int
    wall_time: float
    log: list = field(default_factory=list)   # one dict per executed cycle
    iteration_stats: list = field(default_factory=list)

    @property
    def gross_reduction(self) -> float:
        """Reduction of total gross obligations. Netting a length-L cycle by
        `amt` lowers every one of its L edges, so this is sum(L_i * amt_i) =
        initial_debt - final_debt, whereas total_cleared counts each cycle's
        amt once (the liquidity that circulates to settle the loop)."""
        return sum(e["amount"] * e["length"] for e in self.log)

    @property
    def netting_efficiency_pct(self) -> float:
        """Share of initial gross obligations eliminated by netting."""
        return 100.0 * self.gross_reduction / self.initial_debt if self.initial_debt else 0.0


def _solve(qubo, cfg: PeelingConfig, iter_seed: int) -> Solution:
    if cfg.solver == "greedy":
        return greedy_mwis(qubo)
    if cfg.solver == "exact":
        return brute_force_optimum(qubo)
    if cfg.solver == "random":
        return random_baseline(qubo, n_samples=cfg.n_shots, seed=iter_seed)
    if cfg.solver == "qaoa":
        from ..qaoa.optimize import solve_qaoa

        sol, _ = solve_qaoa(
            qubo,
            p=cfg.p,
            n_starts=cfg.n_starts,
            maxiter=cfg.maxiter,
            n_shots=cfg.n_shots,
            seed=iter_seed,
        )
        return sol
    raise ValueError(f"unknown solver {cfg.solver!r}")


def run_peeling(G: nx.DiGraph, cfg: PeelingConfig, *, verbose: bool = False) -> PeelingReport:
    t0 = time.perf_counter()
    G = G.copy()
    initial_debt = float(sum(d["weight"] for _, _, d in G.edges(data=True)))

    excluded: set = set()
    stall_count = 0
    total_cleared = 0.0
    log: list = []
    iteration_stats: list = []
    it = 0

    for it in range(1, cfg.max_iters + 1):
        cycles = enumerate_cycles(G, length_bound=cfg.length_bound, excluded=frozenset(excluded))
        top = select_top_k(cycles, cfg.k)
        if not top:
            break

        qubo = build_qubo(top, penalty_eps_frac=cfg.penalty_eps_frac)
        sol = _solve(qubo, cfg, derive_seed(cfg.master_seed, it))

        if not sol.feasible or int(sol.x.sum()) == 0:
            # Solver produced nothing usable. Do NOT touch the ledger; retire
            # the current richest cycle from candidacy and move on.
            stall_count += 1
            if stall_count >= cfg.stall_patience:
                excluded.add(top[0].key)
                stall_count = 0
            iteration_stats.append({"iteration": it, "executed": 0, "cleared": 0.0, "stalled": True})
            continue
        stall_count = 0

        executed = 0
        cleared = 0.0
        order = sorted(sol.selected, key=lambda i: -top[i].savings)
        for i in order:
            cyc: Cycle = top[i]
            edge_list = list(zip(cyc.nodes, cyc.nodes[1:] + cyc.nodes[:1]))
            if not all(G.has_edge(u, v) for u, v in edge_list):
                excluded.add(cyc.key)
                continue
            amt = min(G[u][v]["weight"] for u, v in edge_list)
            if amt <= cfg.min_execute:
                excluded.add(cyc.key)
                continue
            for u, v in edge_list:
                G[u][v]["weight"] -= amt
                if G[u][v]["weight"] <= cfg.min_execute:
                    G.remove_edge(u, v)
            executed += 1
            cleared += amt
            log.append(
                {
                    "iteration": it,
                    "cycle_index": i,
                    "amount": amt,
                    "length": len(cyc.nodes),
                    "path": list(cyc.nodes),
                    "solver": cfg.solver,
                }
            )
        total_cleared += cleared
        iteration_stats.append(
            {"iteration": it, "executed": executed, "cleared": cleared, "stalled": False}
        )
        if verbose:
            residual = sum(d["weight"] for _, _, d in G.edges(data=True))
            print(f"[iter {it:>2}] executed {executed:>2} cycles | cleared {cleared:,.2f} | residual {residual:,.2f}")

    final_debt = float(sum(d["weight"] for _, _, d in G.edges(data=True)))
    return PeelingReport(
        initial_debt=initial_debt,
        final_debt=final_debt,
        total_cleared=total_cleared,
        n_iterations=it,
        n_cycles_executed=len(log),
        wall_time=time.perf_counter() - t0,
        log=log,
        iteration_stats=iteration_stats,
    )
