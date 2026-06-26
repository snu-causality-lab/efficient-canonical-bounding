"""Paper §5 + Appendix F posterior + runtime experiments on NDE L_{2.5}.

Covers the following paper outputs in one run:
  - Table 3 (main §5.4): solver_time for 8-state vs 128-state, median/p90/max
  - Table 4 (App F)    : wall-clock time for 8 vs 128, same grid + n=5000
  - Fig 3  (main §5.3) : posterior histogram data — per-draw Q_8, Q_128, (Q128-Q8)
  - Fig 5  (App F)     : per-draw lower/upper bounds [L^(s), U^(s)] for posterior bands
  - Fig 6  (App F)     : per-draw Q (same as Fig 3 column but saved separately)
  - Fig 7  (App F)     : max|Q_8 - Q_128| per n across K posterior draws

Two regimes per Def. 5.2 of the paper:
  - Regime I  (binomial-3-arm)  via allocation-augmented Gibbs sampler.
  - Regime II (multinomial)     via closed-form Dirichlet posterior.

Ground truth p* = (0.6, 0.3, 0.7); closed-form sharp bounds [0.3, 0.9].

Output CSVs (all written into the script's working directory):
  - table3_solve_time.csv       (columns: regime, n, med_8_ms, p90_8_ms, max_8_ms,
                                  med_128_ms, p90_128_ms, max_128_ms, speedup)
  - table4_wall_time.csv        (same columns as Table 3 but wall-clock)
  - fig3_per_draw_regime{I,II}_n{N}.csv  (one CSV per (regime,n); per-draw Q_8, Q_128)
  - fig5_per_draw_bounds_regime{I,II}.csv  (columns: regime, n, draw, L, U)
  - fig7_agreement_per_n.csv    (columns: regime, n, max_abs_q_diff)
  - meta.json                    (all parameters)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import closed_form_nde_l25, dump_csv, dump_meta, timed  # noqa: E402

from canonred.counterfactual import CFQuery, CFVar  # noqa: E402
from canonred.examples import nde_graph  # noqa: E402
from canonred.lp import CachedSharpLP, build_lp_lifted, build_lp_reduced  # noqa: E402
from canonred.observation import l25_marginal_constraints  # noqa: E402
from canonred.posterior import (  # noqa: E402
    binomial_marginal_gibbs,
    dirichlet_posterior,
    lift_phi_uniform,
)
from canonred.signature import build_signature_quotient  # noqa: E402
from canonred.worlds import World  # noqa: E402


# ---------------------------------------------------------------------
# NDE L_{2.5} setup (matches paper §3.2 and §5.1 exactly)
# ---------------------------------------------------------------------


def build_nde_l25():
    G = nde_graph()
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    w00 = World.from_dict("x0w0", {"X": 0, "W": 0})
    w01 = World.from_dict("x0w1", {"X": 0, "W": 1})

    y_nested = CFVar.nested(
        base="Y",
        edge_worlds={("X", "W"): w1, ("X", "Y"): w0, ("W", "Y"): w1},
        G=G,
    )
    Q = CFQuery(events=((y_nested, 1),), graph=G, name="Y_{0,W_1}=1")

    v_w1 = CFVar.standard("W", w1, G)
    v_y00 = CFVar.standard("Y", w00, G)
    v_y01 = CFVar.standard("Y", w01, G)
    cs = l25_marginal_constraints([(v_w1, 1), (v_y00, 1), (v_y01, 1)])
    return G, Q, cs


def phi_true_reduced(quotient, G) -> np.ndarray:
    """Ground-truth phi on the reduced simplex via the product phi*(a,b,c) matching p*."""
    from canonred.canonical import enumerate_states, eval_cfvar

    w1 = World.from_dict("x1", {"X": 1})
    w00 = World.from_dict("x0w0", {"X": 0, "W": 0})
    w01 = World.from_dict("x0w1", {"X": 0, "W": 1})
    v_w1 = CFVar.standard("W", w1, G)
    v_y00 = CFVar.standard("Y", w00, G)
    v_y01 = CFVar.standard("Y", w01, G)

    class_to_abc: dict[int, int] = {}
    for i, u in enumerate(enumerate_states(G)):
        t = quotient.state_class_assignment[i]
        if t in class_to_abc:
            continue
        a = int(eval_cfvar(u, v_w1, G))
        b = int(eval_cfvar(u, v_y00, G))
        c = int(eval_cfvar(u, v_y01, G))
        class_to_abc[t] = (a << 2) | (b << 1) | c
        if len(class_to_abc) == quotient.reduced_dim:
            break

    p_star = (0.6, 0.3, 0.7)
    probs = np.zeros(8, dtype=np.float64)
    for a in (0, 1):
        pa = p_star[0] if a == 1 else (1.0 - p_star[0])
        for b in (0, 1):
            pb = p_star[1] if b == 1 else (1.0 - p_star[1])
            for c in (0, 1):
                pc = p_star[2] if c == 1 else (1.0 - p_star[2])
                probs[(a << 2) | (b << 1) | c] = pa * pb * pc
    probs /= probs.sum()
    phi_true = np.array(
        [probs[class_to_abc[t]] for t in range(quotient.reduced_dim)], dtype=np.float64
    )
    return phi_true


# ---------------------------------------------------------------------
# Posterior helpers
# ---------------------------------------------------------------------


def draw_regime_I(quotient, G, cs, n: int, S: int, burn: int, seed: int) -> np.ndarray:
    """Regime I (binomial-3-arm): K_r ~ Bin(n, p_r*) then Gibbs.

    Returns phi samples of shape (S, reduced_dim).
    """
    rng = np.random.default_rng(seed)
    p_star = np.array([0.6, 0.3, 0.7])
    ks = [int(rng.binomial(n, float(p))) for p in p_star]
    obs = [(ks[i], n) for i in range(3)]
    return binomial_marginal_gibbs(
        quotient, observations=obs, alpha=1.0, S=S, burn=burn, seed=seed + 1
    )


def draw_regime_II(quotient, phi_true: np.ndarray, n: int, S: int, seed: int) -> np.ndarray:
    """Regime II (multinomial): N_t ~ Multinomial(n, phi_true) then closed-form Dirichlet."""
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, phi_true)
    return dirichlet_posterior(counts, alpha=1.0, S=S, seed=seed + 1)


# ---------------------------------------------------------------------
# LP runtime measurement (solver_time + wall-clock)
# ---------------------------------------------------------------------


def measure_runtimes(
    cached_r: CachedSharpLP,
    cached_l: CachedSharpLP,
    phi_samples: np.ndarray,
    M: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw K posterior samples' p; solve each LP pair once; return (solver_r, solver_l, wall_r, wall_l)."""
    idx = rng.integers(0, phi_samples.shape[0], size=K)
    solver_r = np.zeros(K)
    solver_l = np.zeros(K)
    wall_r = np.zeros(K)
    wall_l = np.zeros(K)
    for k, i in enumerate(idx):
        p_samp = M @ phi_samples[i]
        t0 = time.perf_counter()
        _, _, dR = cached_r.solve(p_samp)
        wall_r[k] = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, _, dL = cached_l.solve(p_samp)
        wall_l[k] = time.perf_counter() - t0
        solver_r[k] = dR["solver_time"]
        solver_l[k] = dL["solver_time"]
    return solver_r, solver_l, wall_r, wall_l


# ---------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", type=int, nargs="+", default=[200, 500, 1000, 5000])
    ap.add_argument("--S", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--regimes", nargs="+", choices=["I", "II"], default=["I", "II"])
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "n_grid": args.n_grid, "S": args.S, "burn": args.burn, "K": args.K,
        "seed": args.seed, "regimes": args.regimes,
        "p_star": (0.6, 0.3, 0.7), "alpha_prior": 1.0,
        "solver": "ECOS", "backup": "SCS",
    }
    dump_meta(meta, out / "meta.json")

    with timed("build"):
        G, Q, cs = build_nde_l25()
        coef_l = build_lp_lifted(G, Q, cs)
        quotient = build_signature_quotient(G, Q, cs)
        coef_r = build_lp_reduced(quotient)
        phi_true = phi_true_reduced(quotient, G)
        M_tilde = np.asarray(quotient.constraint_matrix, dtype=np.float64)
        cached_r = CachedSharpLP(coef_r)
        cached_l = CachedSharpLP(coef_l)
        # Warm up to amortise first-call compilation
        cached_r.solve(np.array([0.5, 0.5, 0.5]))
        cached_l.solve(np.array([0.5, 0.5, 0.5]))

    print(f"[setup] lifted_dim={coef_l.dim} reduced_dim={coef_r.dim}", flush=True)

    L_truth, U_truth = closed_form_nde_l25(0.6, 0.3, 0.7)
    print(f"[truth] closed-form [{L_truth}, {U_truth}]", flush=True)

    table3: list[dict] = []
    table4: list[dict] = []
    agreement: list[dict] = []
    fig5_bounds: list[dict] = []

    rng_master = np.random.default_rng(args.seed)

    for regime in args.regimes:
        print(f"\n=== Regime {regime} ===", flush=True)
        for n in args.n_grid:
            seed_n = int(rng_master.integers(1_000_000))
            with timed(f"regime{regime}_n{n}_posterior"):
                if regime == "I":
                    phi = draw_regime_I(quotient, G, cs, n, args.S, args.burn, seed_n)
                else:
                    phi = draw_regime_II(quotient, phi_true, n, args.S, seed_n)

            # Propagate to Q_8 and Q_128
            Q_8 = phi @ quotient.objective
            c_lifted = quotient.objective[list(quotient.state_class_assignment)]
            Q_128 = np.empty(phi.shape[0], dtype=np.float64)
            for s in range(phi.shape[0]):
                psi = lift_phi_uniform(phi[s], quotient)
                Q_128[s] = float(c_lifted @ psi)

            # Per-draw bounds (closed-form; paper's fig 5 semantic)
            p_posterior = phi @ M_tilde.T  # (S, 3)
            L_draws = np.empty(phi.shape[0])
            U_draws = np.empty(phi.shape[0])
            for s in range(phi.shape[0]):
                L_draws[s], U_draws[s] = closed_form_nde_l25(*p_posterior[s])

            # Runtime comparison
            with timed(f"regime{regime}_n{n}_runtime"):
                solver_r, solver_l, wall_r, wall_l = measure_runtimes(
                    cached_r, cached_l, phi, M_tilde, args.K,
                    np.random.default_rng(seed_n + 2),
                )

            # Tables
            table3.append({
                "regime": regime, "n": n,
                "med_8_ms": round(float(np.median(solver_r)) * 1e3, 4),
                "p90_8_ms": round(float(np.quantile(solver_r, 0.9)) * 1e3, 4),
                "max_8_ms": round(float(np.max(solver_r)) * 1e3, 4),
                "med_128_ms": round(float(np.median(solver_l)) * 1e3, 4),
                "p90_128_ms": round(float(np.quantile(solver_l, 0.9)) * 1e3, 4),
                "max_128_ms": round(float(np.max(solver_l)) * 1e3, 4),
                "speedup": round(float(np.median(solver_l) / max(np.median(solver_r), 1e-12)), 3),
            })
            table4.append({
                "regime": regime, "n": n,
                "med_8_wall_s": round(float(np.median(wall_r)), 6),
                "p90_8_wall_s": round(float(np.quantile(wall_r, 0.9)), 6),
                "max_8_wall_s": round(float(np.max(wall_r)), 6),
                "med_128_wall_s": round(float(np.median(wall_l)), 6),
                "p90_128_wall_s": round(float(np.quantile(wall_l, 0.9)), 6),
                "max_128_wall_s": round(float(np.max(wall_l)), 6),
                "speedup": round(float(np.median(wall_l) / max(np.median(wall_r), 1e-12)), 3),
            })
            agreement.append({
                "regime": regime, "n": n,
                "max_abs_q_diff": float(np.max(np.abs(Q_128 - Q_8))),
                "mean_abs_q_diff": float(np.mean(np.abs(Q_128 - Q_8))),
            })

            # Per-draw CSVs (Fig 3 data, Fig 6 data)
            per_draw = [
                {"draw": s, "Q_8": float(Q_8[s]), "Q_128": float(Q_128[s]),
                 "Q_diff": float(Q_128[s] - Q_8[s]),
                 "L_closed": float(L_draws[s]), "U_closed": float(U_draws[s])}
                for s in range(phi.shape[0])
            ]
            dump_csv(per_draw, out / f"fig3_per_draw_regime{regime}_n{n}.csv")

            # Fig 5 aggregated (per-n per-draw L, U)
            for s in range(phi.shape[0]):
                fig5_bounds.append({
                    "regime": regime, "n": n, "draw": s,
                    "L": float(L_draws[s]), "U": float(U_draws[s]),
                })

            q025, q50, q975 = np.quantile(Q_8, [0.025, 0.5, 0.975])
            print(
                f"  n={n:5d}  Q_8 [{q025:.4f},{q50:.4f},{q975:.4f}]  "
                f"max|Q128-Q8|={float(np.max(np.abs(Q_128-Q_8))):.2e}  "
                f"solve_8={float(np.median(solver_r))*1e3:.3f}ms  "
                f"solve_128={float(np.median(solver_l))*1e3:.3f}ms  "
                f"speedup={float(np.median(solver_l)/max(np.median(solver_r),1e-12)):.2f}x",
                flush=True,
            )

    dump_csv(table3, out / "table3_solve_time.csv")
    dump_csv(table4, out / "table4_wall_time.csv")
    dump_csv(agreement, out / "fig7_agreement_per_n.csv")
    dump_csv(fig5_bounds, out / "fig5_per_draw_bounds.csv")

    print(f"\n[done] outputs in {out}", flush=True)


if __name__ == "__main__":
    main()
