#!/bin/bash
#SBATCH --job-name=1086_lazy_phase1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 5 sizes (V=4..8) x 3 replicates = 15 tasks; max-lifted-dim caps lifted side
# at 1<<19 so V>=7 always lazy-only.
PY=${PYTHON_BIN:-python}
"$PY" -u run.py --sizes 4 5 6 7 8 --replicates 3 --max-lifted-dim 16384 --base-seed 42
