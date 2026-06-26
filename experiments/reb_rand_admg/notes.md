# reb_rand_admg — Random ADMG sweep (U3kb, utei)

**Rebuttal target**: reviewer U3kb's "single-model only" critique and utei's "experiments only on the NDE graph" concern.

**Setup**:
- Semi-Markovian ADMGs with `|V| ∈ {4, 5, 6}`; 5 replicates each (15 trials total).
- Directed edge probability `p_dir=0.4`; bidirected `p_bidir=0.3`; ≥1 bidirected edge.
- Binary cardinality for all nodes.
- Query: `Q = P(Y_{X=1} = 1)` where `Y` = last topological node, `X` = earliest ancestor of `Y`.
- Information: full `P_obs(V)` — 2^|V| moment constraints.
- Prop 4.2 d_U reported for informational comparison (may differ from coarsest-safe reduced_dim).
- 5 feasible-p samples per trial for runtime statistics.

**Skip criterion**: `lifted_dim > 2^18 = 262144` → row marked `status=skipped_dim_cap`.

**Output**:
- Non-array run: `rand_admg_sweep.csv`.
- Slurm array: 15 per-task CSVs `rand_admg_sweep_V{n}_rep{r}.csv` + aggregate after.

**Slurm array**: `--array=0-14` splits into 15 parallel jobs.

**Expected resource use**:
- `|V|=4` (lifted ≤ 128): <1min, <1GB per trial.
- `|V|=5` (lifted ≤ ~10^4): 1–5min, few GB.
- `|V|=6` (lifted ≤ 2^18): 5–30min, 10–20GB. Some trials may hit the skip cap.

**Paper relevance**: directly addresses reviewer concern that the main-text experiments only use the NDE graph. Speedup column shows Prop 3.1 gains are not NDE-specific.
