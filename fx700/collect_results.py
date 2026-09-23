#!/usr/bin/env python
"""Merge per-run result JSONs into results.csv and print a summary.

Uses only the stdlib (no pandas), so it runs on the pandas-free cluster venv.

    python collect_results.py --results results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "run", "instance", "n", "p", "qarp_energy", "feasible", "savings",
    "approx_ratio", "t_build", "t_optimize", "t_sample",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.results).glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        if isinstance(d, list):  # grover output (list of runs) — skip in this table
            continue
        timings = d.get("timings") or {}
        rows.append(
            {
                "run": f.stem,
                "instance": d.get("instance"),
                "n": d.get("n"),
                "p": d.get("p"),
                "qarp_energy": d.get("qarp_energy"),
                "feasible": d.get("solution_feasible"),
                "savings": d.get("solution_savings"),
                "approx_ratio": d.get("approx_ratio"),
                "t_build": timings.get("t_build"),
                "t_optimize": timings.get("t_optimize"),
                "t_sample": timings.get("t_sample"),
            }
        )
    if not rows:
        print("no results found")
        return

    rows.sort(key=lambda r: (r["n"] if r["n"] is not None else 0, r["run"]))
    out = Path(args.results) / "results.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # plain-text table to stdout
    widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in FIELDS}
    print(" | ".join(k.ljust(widths[k]) for k in FIELDS))
    print("-+-".join("-" * widths[k] for k in FIELDS))
    for r in rows:
        print(" | ".join(str(r.get(k, "")).ljust(widths[k]) for k in FIELDS))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
