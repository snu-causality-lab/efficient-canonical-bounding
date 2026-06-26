# paper_posterior_sweep — NDE L_{2.5} posterior + runtime

**Paper coverage**: Table 3 (§5.4), Table 4 (App F), Fig 3 (§5.3), Fig 5 (App F), Fig 6 (App F), Fig 7 (App F).

**Setup** (paper §5.1):
- Graph: NDE (X→W, W→Y, X→Y, X↔W); binary.
- Query: Q = P(Y_{0, W_1} = 1).
- Information set: L_{2.5} marginals p_{w1}, p_{y00}, p_{y01}.
- Ground truth: p* = (0.6, 0.3, 0.7); closed-form sharp bounds [0.3, 0.9].
- Prior: phi ~ Dir(α=1, 1_8).

**Regimes** (paper §5.2):
- Regime I (binomial 3-arm): K_r ~ Bin(n, p_r*); allocation-augmented Gibbs (Alg. 2).
- Regime II (multinomial t-data): N_t ~ Multinomial(n, phi*); closed-form Dirichlet posterior.

**Sweep defaults**: n ∈ {200, 500, 1000, 5000}; S=3000 posterior draws; burn=1000; K=10 runtime draws; seed=42.

**Output CSVs**:
- `table3_solve_time.csv` — ECOS solver_time (ms) med/p90/max for 8 vs 128, per (regime, n).
- `table4_wall_time.csv` — end-to-end wall-clock (s) med/p90/max for 8 vs 128.
- `fig3_per_draw_regime{I,II}_n{N}.csv` — per-draw Q_8, Q_128, Q_diff, L_closed, U_closed.
- `fig5_per_draw_bounds.csv` — long-format (regime, n, draw, L, U) for quantile-band plotting.
- `fig7_agreement_per_n.csv` — max|Q_8 - Q_128| and mean|·| per n.

**Paper-reported Table 3 numbers** (submitted PDF):
- n=200:  8-state med=0.154ms, 128-state med=1.761ms, speedup 11.44×
- n=500:  0.155 vs 1.807, 11.63×
- n=1000: 0.154 vs 1.812, 11.74×

Reproducing these closely (±~10% due to machine/timing jitter) validates the library.

**CachedSharpLP warm-up**: one preliminary solve on `p=(0.5,0.5,0.5)` to amortise CVXPY compilation; otherwise K=10 medians skew high at first.
