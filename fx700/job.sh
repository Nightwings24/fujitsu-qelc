#!/bin/bash
# MPI launcher wrapper (QARPdemo pattern): environment setup per rank, then
# exec the payload. Adjust QARPDEMO_DIR if your venv lives elsewhere.
set -eu

QARPDEMO_DIR="${QARPDEMO_DIR:-$HOME/QARPdemo}"

export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

export LD_LIBRARY_PATH=/home/share/developer/boost-1.90.0/lib:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export BOOST_ROOT=/home/share/developer/boost-1.90.0

# Platform-manual settings: single OpenMP thread for deterministic scipy,
# qulacs threads pinned to the node's 48 cores.
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=1
export QULACS_NUM_THREADS=48
export UCX_IB_MLX5_DEVX=no
# distribute the state vector across MPI ranks in qelc's sampling fallback
export QELC_MULTI_CPU=1

source "$QARPDEMO_DIR/venv/bin/activate"
exec "$@"
