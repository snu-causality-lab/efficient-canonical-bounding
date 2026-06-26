#!/bin/bash
#SBATCH --job-name=1086_post_K50
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

CPUS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$CPUS
export MKL_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS
export NUMEXPR_NUM_THREADS=$CPUS

mkdir -p results_K50
PY=${PYTHON_BIN:-python}
"$PY" run.py --n-grid 200 500 1000 5000 --S 3000 --burn 1000 --K 50 --seed 42 --out-dir results_K50
