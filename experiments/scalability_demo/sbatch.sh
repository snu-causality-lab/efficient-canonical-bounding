#!/bin/bash
#SBATCH --job-name=canonred-infeas
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-14

# Infeasible-feasible demo. Array: 5 sizes × 3 reps = 15 tasks.
# Phase A (analytical) always runs; Phase B (explicit LP build + solve)
# runs only when lifted_dim ≤ max_dim (default 2^18).

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

CPUS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$CPUS
export MKL_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS
export NUMEXPR_NUM_THREADS=$CPUS

SIZE_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
REP=$(( SLURM_ARRAY_TASK_ID % 3 ))
SIZES=(4 5 6 7 8)
NV=${SIZES[$SIZE_IDX]}

PY=${PYTHON_BIN:-python}
"$PY" run.py \
    --single-size $NV --single-rep $REP \
    --base-seed 42 --p-dir 0.4 --p-bidir 0.3 \
    --budget-s 300 --max-dim $((1 << 18)) \
    --out-csv infeasible_feasible_V${NV}_rep${REP}.csv
