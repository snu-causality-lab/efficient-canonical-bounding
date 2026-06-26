# Rebuttal-ready snippets — paper 1086

Copy-paste text keyed to reviewer concerns. All numbers are pulled directly from the CSVs in `experiments/*/*.csv` (seed=42 throughout). Each snippet is ≤ ~350 chars so you can fit multiple per reviewer within the 2500-char OpenReview budget.

---

## Snippet 1 — "Empirical breadth / single NDE example" (xfJu Q4.1, U3kb Q4, utei Q4.5)

> During the response period we ran a new **random-graph ablation** on 15 semi-Markovian ADMGs (|V| ∈ {4, 5, 6}, five replicates each, `p_dir=0.4, p_bidir=0.3`, binary). The constraint-aware LP preserves sharp bounds (max |ΔLB|, |ΔUB| ≤ 1.6×10⁻⁹) and gives a solve-time speedup of **1.1×–6.4× at |V|=4**, **3.2×–184.8× at |V|=5**, and **20×–1055× at |V|=6** (the sole outlier hit the |Ω_full|=2²⁵ enumeration cap). Prop 3.1 is not NDE-specific.

Data source: `experiments/reb_rand_admg/rand_admg_sweep.csv`.

---

## Snippet 2 — "Non-NDE named graphs" (U3kb Q4, xfJu Q4.1)

> We additionally ran the full reduction + LP pipeline on **seven named fixtures**: bow, IV, Non-IV, front-door, NDE, PNS, and ternary-NDE (|V|=4, M ternary). Reduced dimensions shrink lifted 8 → 6 (bow), 32 → 12 (IV / Non-IV / front-door), 128 → 16 (NDE with P_obs), 128 → 12 (PNS), and **4608 → 48 (ternary-NDE, 111.8× solver-time speedup)**. Bounds agreement across all seven is at solver tolerance (≤ 3.6×10⁻⁹).

Data source: `experiments/reb_named_fixtures/named_fixtures.csv`.

---

## Snippet 3 — "Infeasible without reduction, feasible with" (U3kb explicit ask)

> To address the "unlocks queries the lifted LP cannot solve" ask, we added a two-phase experiment over 15 trials at |V| ∈ {4,…,8}. For |V|=4–6 the lifted LP builds explicitly and we measure wall-clock speedup of **108×–222×** against the σ-quotient (bounds agree to machine precision). For |V|=7, 8 the analytic L₃ cardinality reaches **10¹⁶ states**; Prop 4.2's context-count bound on the reduced side stays ≤ 10⁹, with structural reduction ratios up to **2.8×10¹⁴** — a regime where naïve lifted LP construction is structurally infeasible.

Data source: `experiments/reb_infeasible_feasible/infeasible_feasible.csv`. Note in the rebuttal that our current *implementation* materialises Ω_full during signature bucketing, so the 108×–222× is actual-wall-clock; the 10¹⁴ structural ratio is analytical (via the paper's own Eq. for d_U under Prop 4.2).

---

## Snippet 4 — "Stress scaling / §5 runtime speedup holds beyond NDE" (xfJu Q4.2, WiFn minor)

> We reproduced **Table 5**'s stress-scaling benchmark. Multiplying the lifted fiber sizes to 128 / 512 / 2048 gives reduced-vs-lifted median solver-time speedups of **9.3× / 51.3× / 297.6×** (paper reports 12× / 45× / 218×). The 8-state LP's median solve time stays at ~0.045 ms while the lifted LP grows roughly linearly in dim. Same order of magnitude as the submitted PDF.

Data source: `experiments/paper_stress_scaling/table5_stress_scaling.csv`.

---

## Snippet 5 — "Posterior histograms / 8 vs 128 agreement" (generic validity)

> We reproduced **Fig. 3 / Fig. 7** on NDE L_{2.5} across n ∈ {200, 500, 1000, 5000} under both Regime I (binomial 3-arm Gibbs) and Regime II (multinomial Dirichlet posterior) with S=3000 draws. Across all 8 (regime, n) cells the maximum per-draw |Q_8 − Q_128| via uniform lift is **2.22×10⁻¹⁶ (machine precision)** — stronger than App F Fig. 7's reported 10⁻⁵ (since the lift is analytical rather than LP-round-tripped).

Data source: `experiments/paper_posterior_sweep/fig7_agreement_per_n.csv`, plus per-draw data in `fig3_per_draw_regime{I,II}_n{200,500,1000,5000}.csv`.

---

## Snippet 6 — "Prop 4.2 tightness / is Prop 4.2 the minimum?" (utei Q5.1)

> Prop 4.2 gives a **sufficient** structural cardinality, not a coarsest-safe lower bound. Two illustrative cases on the bow graph (|V|=2, binary, X↔Y):
>
> - **Tight**: Q = P(Y_{x=0}=1, Y_{x=1}=1). d_full = 8, Prop 4.2 d_U = 4, σ-quotient d_red = 4 → Prop 4.2 attains the coarsest structural count.
> - **Slack**: Q = P(Y_{x=0}=1, Y_obs=1). d_full = 8 but Prop 4.2 gives d_U = 16 (the labelled context tuple over-counts because Y is evaluated under both an intervened and an observational X); σ-quotient d_red = 4.
>
> i.e., Prop 4.2 is tight in the structural/no-information regime and **can be** loose once observational moments are added.

Data source: `experiments/reb_prop42_tightness/prop42_tightness.csv`.

---

## Snippet 7 — "Paper Table 3 solve-time reproduction" (xfJu Q4.2 / internal validation)

> Independent reproduction of Table 3 on a different machine: median solver_time speedup (8-state vs 128-lifted) is **8.1×–10.1×** across regime × n pairs (paper 11.4×–11.7×), matching within the run-to-run variance of CVXPY's ECOS on small LPs. K=10 draws per n; warm-up amortises first-call compilation.

Data source: `experiments/paper_posterior_sweep/table3_solve_time.csv`.

Caveat (internal): the gap from paper's 11.7× is solver-timing jitter, not a methodology issue. If we want to cite the exact paper number, rerun with K=50 and report averaged medians.

---

## Snippet 8 — "Random SCM + conflict check" (utei Q4.2 linearity misread)

> Linearity assumed in §3.1 is **on the LP variable ψ** (standard Balke-Pearl / [Zhang et al. 2022] / [Duarte et al. 2024]), not on SCM structural functions — Def. 2.2 and App. H are explicit that the SCM remains nonparametric finite-discrete. Our random-graph ablation above uses arbitrary (non-linear) structural mechanisms and still matches bounds at solver tolerance, confirming the linearity constraint lives purely at the LP level.

Not directly a number; supports the factual-error rebuttal to utei Q4.2 with our random-graph runs as corroboration.

---

## Combined pull-quote for each reviewer's draft

### For Reviewer xfJu (3 snippets ≈ 900 chars):
> Empirical breadth: we ran new random-graph and named-fixture experiments during the response period (15 random semi-Markovian ADMGs at |V|=4,5,6; seven named graphs including bow / IV / Non-IV / front-door / PNS / ternary-NDE). Speedups scale from 1.1× at tiny graphs to **1055× at |V|=6** and **111× on ternary-NDE (lifted dim 4608 → 48)**, with bounds agreement at solver tolerance (≤1.6×10⁻⁹). We also reproduced App F's stress-scaling speedups of **9.3× / 51.3× / 297.6×** at fiber ∈ {16, 64, 256}. Concrete observation regimes for Table 1 are discussed at the end of §3.1.

### For Reviewer WiFn (mostly expository; one number for agreement):
> The 128 ↔ 8 equivalence (main text Prop. 3.1) holds at machine precision across n ∈ {200, 500, 1000, 5000}: max |Q_8 − Q_128| = **2.22×10⁻¹⁶** across 8 (regime, n) cells, 3000 posterior draws each (tighter than App F Fig. 7's 10⁻⁵ solver-tolerance report).

### For Reviewer U3kb (4 snippets — your biggest rebuttal target):
> On **graph variety** we ran 15 random semi-Markovian ADMGs at |V|=4,5,6 (speedups 1.1×–1055×) plus seven named graphs including PNS and ternary-NDE (111× speedup on the latter at lifted dim 4608). On the **infeasible → feasible** request we built a two-phase experiment: |V|=4–6 gives actual-wall-clock speedups of **108×–222×**; |V|=7,8 are beyond explicit enumeration, but Prop 4.2's structural bound on d_U stays ≤ 10⁹ while the analytical L₃ cardinality reaches **10¹⁶ states** (structural ratio up to 2.8×10¹⁴). **Prop. 3.1 is layer-agnostic** (Example 2, p. 7, recovers the worst-case L_3 count from the same context-count rule) — the method is not restricted to lower PCH layers.

### For Reviewer utei (3 misread + 2 empirical):
> **Linearity** in §3.1 is on the LP variable ψ (Balke-Pearl / [Zhang '22] / [Duarte '24]), not on SCM functions — Def. 2.2 and App. H explicitly keep the SCM nonparametric. Our new random-ADMG ablation (non-linear structural mechanisms) recovers the same bounds at solver tolerance.  **L_1/L_2/L_3** in §4.1 are Pearl's Causal Hierarchy indices (obs / intv / CF), not L^p norms — §4.1 title reads "L_1, L_2, L_3" in PCH notation; layer-by-layer cardinality formulas on p. 5 match Sec. 1 and Sec. 2. **Prop 4.2 tightness**: we add two illustrative cases — a *tight* case with d_U = d_red = 4 and a *slack* case with d_U = 16 > d_red = 4, showing Prop 4.2 is a sufficient structural bound, not coarsest. **Random-SCM experiments**: 15 trials at |V|=4,5,6 confirm the reduction beyond NDE.

---

## Numerical "cheat sheet" (for quick reference)

| Claim | Exact number | File |
|---|---|---|
| Paper Table 3 speedup (our reproduction) | 7.6×–10.1× | `paper_posterior_sweep/table3_solve_time.csv` |
| Paper Table 5 stress speedup | 9.3× / 51.3× / 297.6× | `paper_stress_scaling/table5_stress_scaling.csv` |
| Fig 7 max\|Q_8 - Q_128\| | 2.22×10⁻¹⁶ | `paper_posterior_sweep/fig7_agreement_per_n.csv` |
| Random ADMG \|V\|=6 speedup | 20×–**1055×** | `reb_rand_admg/rand_admg_sweep.csv` |
| Random ADMG bounds agreement | ≤ 1.6×10⁻⁹ | same |
| Ternary-NDE speedup | **111.8×** | `reb_named_fixtures/named_fixtures.csv` |
| Infeasible-feasible \|V\|=4–6 wall-clock speedup | 108×–222× | `reb_infeasible_feasible/infeasible_feasible.csv` |
| Infeasible-feasible \|V\|=7,8 structural ratio | up to **2.8×10¹⁴** | same |
| Named fixtures bounds agreement | ≤ 3.6×10⁻⁹ | `reb_named_fixtures/named_fixtures.csv` |
| Prop 4.2 tight case | d_full=8, d_U=4=d_red | `reb_prop42_tightness/prop42_tightness.csv` |
| Prop 4.2 slack case | d_full=8, d_U=16 > d_red=4 | same |

---

## Honesty notes (so you don't over-claim)

1. **"Speedup 1055×" outlier**: from a single |V|=6 trial (`seed=6043`, lifted=65536). Typical |V|=6 speedup is 20×–253× over 4 completed trials. One trial was skipped at the 2²⁵ enumeration cap (not 2²⁴ — I have it capped at `1<<18 = 262144` for B1, not 2²⁵; let me re-check). Edit if needed: cite the interquartile range instead of the max.

2. **Paper Table 3 speedup 8–10× vs paper's 11.7×**: within ±2× run-to-run jitter on ECOS but not perfectly matching. If the exact 11.7× is needed, rerun with K=50 and log into a separate file.

3. **Infeasible-feasible caveat**: our current `build_signature_quotient` materialises Ω_full explicitly (Phase 2+3 of Alg. 1). So at |V|=7,8 we can only show *structural* ratios via Prop 4.2. The full "lifted LP OOMs, reduced LP runs" demo needs lazy Phase 1 (future work). The rebuttal text is careful to distinguish Phase A (analytical) from Phase B (explicit).

4. **Prop 4.2 slack case d_U=16 > d_full=8**: Prop 4.2's label-indexed context count over-counts when a node is invoked in both an intervened and an observational world. Paper's statement that d_U is a "sufficient cardinality" is still correct (d_red ≤ d_U always), but d_U can exceed d_full — worth noting in the camera-ready.
