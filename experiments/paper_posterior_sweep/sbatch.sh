#!/bin/bash
#SBATCH --job-name=canonred-post_sweep
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Paper §5 + App F: posterior + runtime sweep on NDE L_{2.5}.
# Covers Table 3, Table 4, Fig 3, Fig 5, Fig 6, Fig 7 in a single run.
# CPU-only. Activate the environment before sbatch, or set PYTHON_BIN.

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

CPUS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$CPUS
export MKL_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS
export NUMEXPR_NUM_THREADS=$CPUS

PY=${PYTHON_BIN:-python}
"$PY" run.py --n-grid 200 500 1000 5000 --S 3000 --burn 1000 --K 10 --seed 42
