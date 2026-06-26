#!/bin/bash
#SBATCH --job-name=canonred-stable
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x_%A_%a.out
#SBATCH --array=42-61
set -euo pipefail

# Single-thread everything for reproducible, machine-independent timing.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ACCELERATE_MAXIMUM_THREADS=1 \
       PYTHONHASHSEED=0

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
PY=${PYTHON_BIN:-python}
SEED=${SLURM_ARRAY_TASK_ID}

echo "[slurm] node=$(hostname) seed=$SEED job=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID}"

# Stress scaling (Table 5) — pure timing, fast.
cd "$ROOT/experiments/paper_stress_scaling"
"$PY" run_stable.py --seed "$SEED" --require-slurm --n-repeats 25 --out-dir results_stable

# Posterior sweep (Table 3/4 timing + Fig 3/5/7 data) — S=10000 for smooth figures.
cd "$ROOT/experiments/paper_posterior_sweep"
"$PY" run_stable.py --seed "$SEED" --require-slurm --S 10000 --burn 5000 \
      --n-repeats 25 --out-dir results_stable

echo "[slurm] done seed=$SEED"
