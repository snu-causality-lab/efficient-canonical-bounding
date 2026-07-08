#!/bin/bash
#SBATCH --job-name=canonred-named
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x_%A_%a.out
#SBATCH --array=42-61
set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ACCELERATE_MAXIMUM_THREADS=1 \
       PYTHONHASHSEED=0

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
PY=${PYTHON_BIN:-python}
SEED=${SLURM_ARRAY_TASK_ID}

echo "[slurm] node=$(hostname) seed=$SEED job=${SLURM_JOB_ID}"
cd "$ROOT/experiments/named_graphs"
"$PY" run_stable.py --seed "$SEED" --require-slurm --n-repeats 25 --out-dir results_stable
echo "[slurm] done seed=$SEED"
