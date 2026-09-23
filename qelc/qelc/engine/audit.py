"""Independent audit: replay an execution log against the pristine ledger.

Invariants checked:
  1. No phantom edges — every netted edge existed when it was netted.
  2. No overdraft — no edge's weight ever goes negative.
  3. Netting invariant — a cycle execution changes no node's net position
     (inflow minus outflow), i.e. netting is settlement-neutral.
  4. Totals — the replayed cleared total equals the reported one.

Tolerances are *relative* to the magnitudes involved: real ledgers carry
billion-scale weights where float64 rounding alone is ~1e-7 absolute, so
absolute epsilons would flag rounding as fraud. `min_execute` must match the
engine's PeelingConfig.min_execute so replayed edge removals mirror the
engine's exactly.
"""

from __future__ import annotations

import networkx as nx


def audit_log(
    G_original: nx.DiGraph,
    log: list,
    reported_total: float,
    *,
    min_execute: float = 1e-9,
    rel_tol: float = 1e-9,
) -> dict:
    G = G_original.copy()
    replayed_total = 0.0
    n_removed = 0
    failures: list[str] = []

    net_before = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        net_before[v] += d["weight"]
        net_before[u] -= d["weight"]

    for entry in log:
        amt = float(entry["amount"])
        path = list(entry["path"])
        edges = list(zip(path, path[1:] + path[:1]))
        slack = rel_tol * max(1.0, amt)
        for u, v in edges:
            if not G.has_edge(u, v):
                failures.append(f"iter {entry['iteration']}: phantom edge {u}->{v}")
                break
            if G[u][v]["weight"] < amt - slack:
                failures.append(
                    f"iter {entry['iteration']}: overdraft on {u}->{v} "
                    f"({G[u][v]['weight']:.6f} < {amt:.6f})"
                )
                break
        else:
            for u, v in edges:
                G[u][v]["weight"] -= amt
                if G[u][v]["weight"] <= min_execute:
                    G.remove_edge(u, v)
                    n_removed += 1
            replayed_total += amt
            continue
        break

    net_after = {n: 0.0 for n in G_original.nodes()}
    for u, v, d in G.edges(data=True):
        net_after[v] += d["weight"]
        net_after[u] -= d["weight"]
    drift = max(
        (abs(net_before[n] - net_after[n]) for n in G_original.nodes()), default=0.0
    )
    # Removing a residual edge discards up to min_execute from two nodes'
    # positions; float64 rounding contributes ~rel_tol of the traded volume.
    initial_volume = sum(d["weight"] for _, _, d in G_original.edges(data=True))
    drift_tol = min_execute * 2.0 * max(n_removed, 1) + rel_tol * max(1.0, initial_volume)
    if drift > drift_tol:
        failures.append(
            f"netting invariant violated: max net-position drift {drift} (tol {drift_tol})"
        )

    total_tol = rel_tol * max(1.0, abs(reported_total)) * max(len(log), 1)
    if abs(replayed_total - reported_total) > total_tol:
        failures.append(
            f"total mismatch: replayed {replayed_total:.6f} vs reported {reported_total:.6f} "
            f"(tol {total_tol})"
        )

    return {
        "passed": not failures,
        "failures": failures,
        "replayed_total": replayed_total,
        "max_net_drift": drift,
        "edges_removed": n_removed,
    }
