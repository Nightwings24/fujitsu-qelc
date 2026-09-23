# Running QELC on the Fujitsu Quantum Simulator (FX700) — runbook

This is the exact, ordered procedure to get our code producing **scored** (QARP)
results on the cluster. Follow it top to bottom the first time; later sessions
start at Part 4.

> **What "the simulator" is:** the FX700 is a 1,024-node ARM (A64FX) CPU cluster
> that *simulates* quantum circuits (state vector up to 40 qubits, or a tensor-
> network backend). It is not a physical quantum computer. Our QAOA runs on it
> through **QARP**, which is the only execution path Fujitsu scores.

---

## Part 0 — The Google Cloud VM constraint (read first)

Login is **IP-locked**: Fujitsu only accepts SSH connections from the exact
static global IP each member registered (Section 2.1.2 of the platform manual).
You registered those static IPs on **Google Cloud VMs**. Therefore:

- **You must run every `ssh`/`scp`/`rsync` to the cluster _from the GCP VM_,**
  not from a laptop. A laptop's IP is not registered and will be refused.
- Each of the 3 members has their own `{account name, access port, SSH key,
  static IP}` (in `Fujitsu QSC User Account.xlsx`, columns "User Account Name"
  and "Access Port Number", filled in by Fujitsu — all three are provisioned).
- The private SSH key whose public half you submitted must live on the GCP VM
  (in `~/.ssh/`). If you generated the keypair on a laptop, copy the **private**
  key to the VM (`scp id_ed25519 vm:~/.ssh/` then `chmod 600`).
- Keep the GCP VM running for the whole session; if its external IP is
  ephemeral it can change on stop/start — make sure it is a **reserved static**
  external IP (you said you set this up; confirm before a long run).

Everything below runs **on the GCP VM's shell**.

---

## Part 1 — SSH login (on the GCP VM)

1. Put your private key at `~/.ssh/id_ed25519` (`chmod 600`).
2. Create `~/.ssh/config` (substitute your account name and the port from the
   xlsx; `<GID>` = your 3-digit group id, which prefixes the login node name):

   ```
   # jump server (DMZ relay)
   Host qsim-gw
       HostName <JUMP_SERVER_IP>
       Port <ACCESS_PORT_from_xlsx>
       User <ACCOUNT_NAME_from_xlsx>
       IdentityFile ~/.ssh/id_ed25519
       StrictHostKeyChecking no

   # login server, reached via the jump server
   Host qsim
       HostName login-server
       User <ACCOUNT_NAME_from_xlsx>
       IdentityFile ~/.ssh/id_ed25519
       ProxyCommand ssh -W %h:%p qsim-gw
       StrictHostKeyChecking no
   ```

   (`StrictHostKeyChecking no` matches Fujitsu's sample config and avoids the
   two interactive host-key prompts a fresh GCP VM would otherwise show on the
   first hop-through — important if you later script the transfers.)

3. Log in (multi-hop is intended — do not work on the jump server):

   ```bash
   ssh qsim
   ```

   Success looks like `[<account>@loginvm-<GID> ~]$`. If you see
   `REMOTE HOST IDENTIFICATION HAS CHANGED`, run
   `ssh-keygen -R login-server && ssh-keygen -R <JUMP_SERVER_IP>` and retry.

You are now on the **login server**. Jobs run on compute nodes via Slurm; you
never compute on the login server directly.

---

## Part 2 — One-time environment build (QARP + qelc)

Do this once. It follows the MPI variant of `QSC_FX700_QARP_Setup_Guide.md`,
then adds our package. **Order note:** steps 2a–2b run now on a compute node;
step **2c installs our package and must wait until Part 3 has copied it over**,
so the sequence is 2a → 2b → Part 3 → 2c. Allocate an **Interactive** compute
node first (the build must happen on a compute node, and the docs directories
are only visible there):

```bash
# on the login server
salloc -N 1 -p Interactive --time=6:00:00
# you are now on a compute node: [<account>@fx-XX-XX-XX ~]$

cp /home/share/developer/QARPdemo/QARPdemo.tar ~/
tar xvf QARPdemo.tar
cd ~/QARPdemo
```

### 2a. pyenv + Python 3.12.10 + venv (per the guide)

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src
cd ~/QARPdemo

# CONFIRM the exact Boost directory name FIRST — the guide warns the source PDF
# renders it inconsistently (boost-1.90.0 vs boost1.90.0). Use what this prints:
ls /home/share/developer/ | grep boost      # -> substitute the real name below

# pyenv + Boost paths go in .bashrc (NOT .bash_profile — mpirun won't read that)
cat >> ~/.bashrc <<'RC'
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
export LD_LIBRARY_PATH=/home/share/developer/boost-1.90.0/lib:/usr/local/lib
export BOOST_ROOT=/home/share/developer/boost-1.90.0
RC
source ~/.bashrc

pyenv install 3.12.10
pyenv local 3.12.10
python -m venv venv
source venv/bin/activate
```

### 2b. QARP + MPI-enabled Qulacs (per the guide)

```bash
# QARP ships as pre-compiled .pyc — copy it into site-packages (cannot pip-install)
cp -r qarp venv/lib/python3.12/site-packages/

pip install --no-cache-dir --extra-index-url https://pypi.org/simple -r requirements_mpi.txt

# build MPI-enabled Qulacs from the bundled source
cd qulacs
C_COMPILER=mpicc CXX_COMPILER=mpic++ USE_MPI=Yes pip install --no-cache-dir --extra-index-url https://pypi.org/simple .
cd ~/QARPdemo
```

**Verify the QARP + MPI-Qulacs stack actually runs** (guide Section 3.1) before
layering anything on top — an install that imports but is mis-linked would
otherwise only surface after you have spent node-hours:

```bash
cd ~/QARPdemo/example && python mwe_vqe.py    # must converge to ~ -2.2023 (H2)
cd ~/QARPdemo
```

### 2c. Install our qelc package (do this AFTER Part 3's transfer)

`~/qelc` does not exist until Part 3 copies it from the GCP VM, and reaching the
GCP VM means leaving this compute node (which ends the `salloc` allocation and
deactivates the venv). So the order is: finish 2a+2b here → **do Part 3** →
come back, re-allocate a node, re-activate the venv, and only then install qelc:

```bash
# back on the login server AFTER Part 3 has put ~/qelc and ~/fx700 in place:
salloc -N 1 -p Interactive --time=6:00:00
source ~/QARPdemo/venv/bin/activate
pip install --no-cache-dir --extra-index-url https://pypi.org/simple -e ~/qelc
```

Our core deps are exactly numpy/networkx/scipy, which the QARP venv already has,
so this pulls in nothing heavy: pandas/pyarrow/qulacs live in the `local` extra
and are **not** installed here — the cluster uses the QARP-built qulacs, and the
instances arrive as JSON so no CSV tooling is needed.

> If openfermion is not already present (QARP examples use it, so it usually is):
> `pip install --extra-index-url https://pypi.org/simple openfermion`. Confirm
> the whole stack: `python -c "import openfermion, qarp, qulacs, mpi4py, qelc; print('ok')"`.

---

## Part 3 — Transfer our code + instances to the cluster

From the **GCP VM** (the machine with the registered IP). We ship the package,
the FX700 driver kit, and the pre-built USDC instances (already exact-solved
locally to n≤26, so the cluster just executes):

```bash
# from the GCP VM shell, in the directory holding qelc/ and fx700/
rsync -avz --exclude '__pycache__' --exclude '.venv' \
    qelc/  qsim:~/qelc/
rsync -avz --exclude '__pycache__' \
    fx700/ qsim:~/fx700/
```

The `fx700/instances_usdc/` directory (real Ethereum-USDC data) and
`fx700/instances/` (synthetic Barabási–Albert) both travel with it. Lead with
the USDC set for scoring — real settlement data reads stronger than synthetic.

---

## Part 4 — The run sequence

All jobs go through the **Batch** queue (IntrHPC is suspended during QSC), with
power-of-2 MPI ranks and a walltime you set per job. Regenerate the job files
once (they point at the driver scripts), then submit in order.

```bash
ssh qsim
cd ~/fx700
source ~/QARPdemo/venv/bin/activate
python run_matrix.py            # writes jobs/*.job
```

### Step 4.0 — E0: convention check (DO THIS FIRST, ~15 min)

This is the single most important gate. It evaluates the QARP QAOA energy at
fixed angles and compares against our locally-verified circuit, to pin down
QARP's angle/sign/factor-of-2 convention **before** you spend node-hours.

```bash
sbatch jobs/e0.job          # or run interactively on an salloc node:
# python validate_qarp_energy.py --instance instances_usdc/E0_n16.json
```

(The instance set is fixed into the job files when you run `python run_matrix.py`
— it defaults to `instances_usdc`; re-run with `--instances-dir instances` to
switch to the synthetic set.)

Read `out-e0-*.log`. It must print `MATCH`. If it prints a transform
(e.g. `gamma*-1`), that tells you QARP negates γ — record it; our sampling
fallback already uses our own circuit so results stay consistent, but you note
the convention in the report.

**Do not proceed to larger runs until E0 matches.**

### Step 4.1 — E1/E2: certified approximation ratios (24, 26 qubits, 1 node)

```bash
sbatch jobs/e1.job      # 24 qubits, p=1->2, ~1-2 h
sbatch jobs/e2.job      # 26 qubits, ~2-3 h
```

Each writes `results/e1_seed0.json` etc. with the QARP energy, the extracted
netting solution, and — because these instances ship with the exact optimum —
a real **approximation ratio**. This is your headline evidence.

### Step 4.2 — Scale up (28 → 34 qubits)

**Angle transfer is automatic** — each run writes `params/<instance>_best.json`,
and the next-larger job reads it (a missing file just falls back to fresh init,
never a crash). So there is **no manual copy step**; the only requirement is
**dependency order**: a job that transfers angles must be submitted only after
its predecessor has finished. Submit **one job per `sbatch`** (sbatch takes a
single script; extra arguments are silently swallowed):

```bash
sbatch jobs/e3.job          # 28 q, 1 node — transfers E2's angles; wait for it to finish
sbatch jobs/e4.job          # 30 q, 2 nodes (2^30 vector is 16 GiB, needs >1 node)
# E5-E8 all transfer E4's angles, so wait for E4 to finish, then:
sbatch jobs/e5.job          # 31 q, 2 nodes
sbatch jobs/e6.job          # 32 q, 4 nodes
sbatch jobs/e7.job          # 33 q, 8 nodes
sbatch jobs/e8.job          # 34 q, 16 nodes, sample-only (fixed angles), 48 h walltime
```

At ≥31 qubits we transfer angles from the 30-qubit optimum and do few or zero
optimizer iterations, because init overhead grows steeply (the platform's own
33-qubit reference was ~3 h just to initialize on 512 nodes).

Optional strong-scaling sweep (same 31-q problem on 2/4/8 nodes → time-to-
solution curve), after E4:

```bash
sbatch jobs/e9_n2.job; sbatch jobs/e9_n4.job; sbatch jobs/e9_n8.job
```

### Step 4.3 — QARP circuit-cutting demo (E10) and Grover (G1)

```bash
sbatch jobs/e10.job     # cuts a 24-q QUBO into 2x12; USDC instance has 4 crossing
                        # conflicts -> 4 gate cuts (<=6 cap); reports reconstruction
                        # error vs uncut and the 2.13% cut-friendliness sacrifice
sbatch jobs/g1.job      # Grover key search k=8..15, resource scaling for Module II
```

### Monitor jobs

```bash
squeue                  # your queue
squeues                 # everyone's (updates ~10s) — gauge contention
sacct                   # finished-job history
scancel <JOBID>         # kill one
```

---

## Part 5 — Retrieve and summarize results

```bash
# on the login server
cd ~/fx700
python collect_results.py --results results     # -> results/results.csv + summary

# from the GCP VM, pull everything back
rsync -avz qsim:~/fx700/results/ ./results_fx700/
```

`results.csv` has, per run: qubits, QARP energy, feasibility, savings (USD),
approximation ratio (where an exact optimum exists), and init/optimize/sample
wall times — the scaling story for the report.

---

## Troubleshooting

- **`Permission denied` / connection refused at `ssh qsim`** — you are not on
  the registered-IP host. Run from the GCP VM, and confirm its external IP still
  equals the one you registered (`curl -s ifconfig.me`).
- **Job PENDING forever** — a monthly maintenance reservation or a non-power-of-2
  rank count. Check `sinfo -T` and make `-N`/`-n` a power of two.
- **OOM near 30 qubits** — a 2^30 state vector is 16 GiB and work buffers push
  past a node's 32 GB (E4 already defaults to 2 nodes for this reason). If a run
  still OOMs, double `-N`/`-n` in its job file (keep it a power of two).
- **A larger job started before its predecessor finished** — it just optimizes
  from scratch (the missing `params/<pred>_best.json` triggers a graceful
  fallback with a warning), so it still produces a valid result, only without
  angle transfer. Re-run it after the predecessor if you want the transfer.
- **`ModuleNotFoundError: pandas` on the cluster** — should not happen (the
  cluster path is pandas-free, guarded by `tests/test_cluster_imports.py`, and
  `collect_results.py` uses only the stdlib); if it does, you invoked a
  local-only helper — use only the `fx700/*.py` drivers listed here.
- **Anaconda blocked** — expected; use `pip` with
  `--extra-index-url https://pypi.org/simple`, never conda.

---

## Sanity checklist before you burn node-hours

- [ ] `ssh qsim` works from the GCP VM.
- [ ] `python -c "import qarp, qulacs, mpi4py, openfermion, qelc; print('ok')"`
      inside the activated venv on a compute node.
- [ ] E0 prints `MATCH`.
- [ ] Only then: E1 → E2 → scale up.
