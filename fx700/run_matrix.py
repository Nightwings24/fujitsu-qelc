#!/usr/bin/env python
"""Generate the Slurm job files for the whole experiment matrix.

    python run_matrix.py                       # jobs point at instances_usdc/
    python run_matrix.py --instances-dir instances   # synthetic set

Angle transfer is automatic: run_qaoa_qarp.py writes params/<instance>_best.json
after every run, and each larger job's --init-params points at its predecessor's
auto-written file (missing files degrade to grid/random init, they do not crash).
Submit one job per sbatch, in README order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# name, instance, nodes, walltime, extra, kind, init_from (predecessor instance
# whose auto-written angles this job transfers; None = optimize from scratch)
RUNS = [
    ("e0",    "E0_n16",      1, "00:30:00", "--p 1 --maxiter 0 --shots 10000",  "validate", None),
    ("e1",    "E1_n24",      1, "02:00:00", "--p 2 --maxiter 60 --shots 10000", "qaoa",     None),
    ("e2",    "E2_n26",      1, "03:00:00", "--p 2 --maxiter 60 --shots 10000", "qaoa",     None),
    ("e3",    "E3_n28",      1, "05:00:00", "--p 2 --maxiter 40 --shots 10000", "qaoa",     "E2_n26"),
    ("e4",    "E4_n30",      2, "06:00:00", "--p 1 --maxiter 20 --shots 10000", "qaoa",     "E3_n28"),
    ("e5",    "E5_n31",      2, "06:00:00", "--p 1 --maxiter 20 --shots 4000",  "qaoa",     "E4_n30"),
    ("e6",    "E6_n32",      4, "06:00:00", "--p 1 --maxiter 15 --shots 4000",  "qaoa",     "E4_n30"),
    ("e7",    "E7_n33",      8, "06:00:00", "--p 1 --maxiter 10 --shots 2000",  "qaoa",     "E4_n30"),
    ("e8",    "E8_n34",     16, "48:00:00", "--p 1 --skip-optimize --shots 2000", "qaoa",   "E4_n30"),
    ("e9_n2", "E5_n31",      2, "06:00:00", "--p 1 --maxiter 10 --shots 2000",  "qaoa",     "E4_n30"),
    ("e9_n4", "E5_n31",      4, "06:00:00", "--p 1 --maxiter 10 --shots 2000",  "qaoa",     "E4_n30"),
    ("e9_n8", "E5_n31",      8, "06:00:00", "--p 1 --maxiter 10 --shots 2000",  "qaoa",     "E4_n30"),
    ("e10",   "E10_n24_cut", 1, "06:00:00", "--shots 4000 --compare-uncut",     "cutting",  "E2_n26"),
    ("g1",    None,          1, "06:00:00", "--k-min 8 --k-max 15 --shots 2000","grover",   None),
]

# E4 is 30 qubits: a 2^30 state vector is 16 GiB and work buffers push a single
# 32 GB node toward OOM, so E4 defaults to 2 nodes (see manual / troubleshooting).

TEMPLATE = """#!/bin/bash
#SBATCH -p Batch
#SBATCH -N {nodes}
#SBATCH -t {walltime}
#SBATCH -J qelc_{name}
#SBATCH -o out-{name}-%j.log

cd "$SLURM_SUBMIT_DIR"
mkdir -p results params
mpirun -N 1 -npernode 1 -n {nodes} ./job.sh python -u {driver}
"""

DRIVERS = {
    "qaoa": "run_qaoa_qarp.py --instance {idir}/{inst}.json --seed {seed} {extra}{init} --out results/{name}_seed{seed}.json",
    "validate": "validate_qarp_energy.py --instance {idir}/{inst}.json",
    "cutting": "run_cutting_demo.py --instance {idir}/{inst}.json{init} {extra} --out results/{name}.json",
    "grover": "run_grover_qvi.py {extra} --out results/{name}.json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--instances-dir",
        default="instances_usdc",
        help="which instance set the jobs point at (default: real USDC data)",
    )
    args = ap.parse_args()

    jobs = Path("jobs")
    jobs.mkdir(exist_ok=True)
    for name, inst, nodes, walltime, extra, kind, init_from in RUNS:
        # cutting uses --params; qaoa uses --init-params; both read the same
        # auto-written params/<predecessor>_best.json.
        if init_from and kind == "cutting":
            init = f" --params params/{init_from}_best.json"
        elif init_from:
            init = f" --init-params params/{init_from}_best.json"
        else:
            init = ""
        driver = DRIVERS[kind].format(
            idir=args.instances_dir, inst=inst, extra=extra, name=name, seed=0, init=init
        )
        (jobs / f"{name}.job").write_text(
            TEMPLATE.format(name=name, nodes=nodes, walltime=walltime, driver=driver)
        )
        print(f"wrote jobs/{name}.job  ({nodes} node(s), {walltime})  -> {args.instances_dir}")
    print("\nSubmit ONE per sbatch, in README order; E0 must MATCH before anything larger runs.")


if __name__ == "__main__":
    main()
