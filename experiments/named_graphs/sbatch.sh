#!/bin/bash
#SBATCH --job-name=canonred-named
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

CPUS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$CPUS
export MKL_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS
export NUMEXPR_NUM_THREADS=$CPUS

PY=${PYTHON_BIN:-python}
"$PY" run.py --p-samples 10 --seed 42
