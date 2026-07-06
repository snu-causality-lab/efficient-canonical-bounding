#!/bin/bash
#SBATCH --job-name=canonred-post_K50_excl
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Exclusive-node rerun to eliminate noisy-neighbor effects on solver_time.

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Force single-thread to minimize any residual variance from thread churn
# on a tiny LP (ECOS is single-threaded anyway for small problems).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p results_K50_exclusive
PY=${PYTHON_BIN:-python}
"$PY" run.py --n-grid 200 500 1000 5000 --S 3000 --burn 1000 --K 50 --seed 42 --out-dir results_K50_exclusive
