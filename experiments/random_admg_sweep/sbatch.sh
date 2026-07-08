#!/bin/bash
#SBATCH --job-name=canonred-rand_admg
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-14

# Random ADMG sweep for graph-variation checks.
# Array index encodes (size, rep):  idx = (size_idx * 5) + rep
#   size_idx 0 -> |V|=4, idx 1 -> |V|=5, idx 2 -> |V|=6
# Each task writes rand_admg_sweep_V${n}_rep${r}.csv to cwd.
# Aggregate after all tasks complete:
#   awk '(NR==1 || FNR!=1)' rand_admg_sweep_V*.csv > rand_admg_sweep.csv

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

CPUS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$CPUS
export MKL_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS
export NUMEXPR_NUM_THREADS=$CPUS

SIZE_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
REP=$(( SLURM_ARRAY_TASK_ID % 5 ))
SIZES=(4 5 6)
NV=${SIZES[$SIZE_IDX]}

PY=${PYTHON_BIN:-python}
"$PY" run.py \
    --single-size $NV --single-rep $REP \
    --base-seed 42 --p-dir 0.4 --p-bidir 0.3 \
    --p-samples 5 --max-lifted-dim $((1 << 18)) \
    --out-csv rand_admg_sweep_V${NV}_rep${REP}.csv
