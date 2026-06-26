#!/bin/bash
#SBATCH --job-name=1086_lazy_phase1_arr
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-14
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 5 sizes (V=4..8), 3 replicates each — 15 tasks total
# idx = (V-4) * 3 + rep ; V in {4,5,6,7,8}, rep in {0,1,2}
IDX=$SLURM_ARRAY_TASK_ID
SIZE=$((4 + IDX / 3))
REP=$((IDX % 3))

mkdir -p partial
PY=${PYTHON_BIN:-python}
"$PY" -u run.py \
  --single-size $SIZE --single-rep $REP \
  --max-lifted-dim 524288 --base-seed 42 \
  --out-dir partial \
  --out-csv lazy_phase1_V${SIZE}_rep${REP}.csv
