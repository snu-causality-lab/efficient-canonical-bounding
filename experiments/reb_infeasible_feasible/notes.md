# reb_infeasible_feasible — Two-phase demo (U3kb)

**Important implementation note**: our `build_signature_quotient` enumerates `Omega_full` explicitly (Alg 1 Phase 2+3 materialise all states). So the *reduced-side construction itself* has complexity `O(|Omega_full|)`. A true "infeasible → feasible" demo needs Alg 1 Phase 1 to be implemented lazily (structurally determine signature classes without iterating). That is **deferred future work**.

To give the rebuttal a clean story today, this experiment reports two orthogonal things:

## Phase A — Analytical structural comparison (always reported)

For every `(|V|, rep)` trial we compute, **without building any LP**:
- `lifted_dim_analytical_L3` = the L_3 worst-case cardinality `Π_V d_V^{|Ω_{pa(V)}|}`.
- `prop42_d_U` = Prop 4.2's structural upper bound on the reduced side `Π_{V ∈ C(U)} d_V^{m_V(Q)}`.
- `structural_ratio` = lifted / d_U.

This shows the scale gap that motivates Prop 3.1 regardless of whether we can actually build the LP today.

## Phase B — Explicit build + solve (only when dim ≤ cap)

If `lifted_dim_analytical_L3 ≤ --max-dim` (default 2^18), we build both the lifted LP and the reduced LP via our (explicit-enumeration) pipeline, solve one feasible `p`, and record:
- `lifted_build_wall_s`, `reduced_build_wall_s`
- `lifted_solve_wall_s`, `reduced_solve_wall_s`
- `lifted_solver_time_s`, `reduced_solver_time_s`
- `lifted_peak_rss_delta_mb`
- `wall_speedup = lifted_solve_wall_s / reduced_solve_wall_s`
- `abs_lb_diff`, `abs_ub_diff` (bounds agreement)

If `lifted_dim_analytical_L3 > --max-dim`, `build_status = "skipped_dim_cap"` and Phase B columns are empty.

## Expected coverage

- `|V|=4, 5`: Phase B succeeds; actual-speedup story.
- `|V|=6` (sparse): Phase B usually succeeds; larger actual speedup.
- `|V|=6` (dense) / `|V|=7, 8`: Phase A only; structural-ratio story.

## Slurm: array 0-14

5 sizes × 3 reps = 15 tasks. 32 GB memory (explicit enumeration at |V|=6 can touch 10 GB). 30 min time limit per task.

## Aggregate

```
awk '(NR==1 || FNR!=1)' infeasible_feasible_V*.csv > infeasible_feasible.csv
```

## Output columns

| Column | Phase | Meaning |
|---|---|---|
| `lifted_dim_analytical_L3` | A | worst-case `Π_V d_V^{\|Ω_{pa(V)}\|}` |
| `prop42_d_U` | A | Prop 4.2 bound |
| `structural_ratio` | A | lifted / d_U |
| `build_status` | A/B | `ok`, `skipped_dim_cap`, or a timeout/error code |
| `lifted_dim_actual`, `reduced_dim` | B | as built |
| `lifted_build_wall_s`, `reduced_build_wall_s` | B | |
| `lifted_solve_wall_s`, `reduced_solve_wall_s` | B | |
| `lifted_solver_time_s`, `reduced_solver_time_s` | B | |
| `lifted_peak_rss_delta_mb` | B | peak RSS during lifted solve |
| `wall_speedup` | B | lifted_solve_wall_s / reduced_solve_wall_s |
| `abs_lb_diff`, `abs_ub_diff` | B | Prop 3.1 empirical check |
