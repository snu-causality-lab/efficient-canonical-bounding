"""Reproduce the paper NDE L_{2.5} running example end-to-end.

This script is the canonred analog of the legacy
`allinone_posterior_visuals_cached.py`. It reproduces:

  1. Closed-form + LP sanity: 8-state / 128-state / closed-form bounds match
     at p^* = (0.6, 0.3, 0.7).
  2. Regime II (multinomial t-data): Dirichlet-posterior of phi on the reduced
     8-state simplex, propagated to Q_8 and Q_128 (uniform lift). Histograms
     saved per n.
  3. Regime I (binomial 3-arm): allocation-augmented Gibbs; same propagation
     and histograms.
  4. Solver-time comparison (Table 1 analog): 8-state vs 128-state median
     `solve_time` over K posterior draws per n.

Outputs (under `scripts/outputs/nde/`):
  - `summary_regime_{I,II}.csv`
  - `q_hist_regime{I,II}_n{N}.png`
  - `runtime.png`

Run:
  python -m scripts.reproduce_nde [--n-grid 200 500 1000] [--S 3000] [--K 10]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from canonred.counterfactual import CFQuery, CFVar
from canonred.examples import nde_graph
from canonred.lp import CachedSharpLP, build_lp_lifted, build_lp_reduced, solve_sharp_bounds
from canonred.observation import l25_marginal_constraints
from canonred.posterior import (
    binomial_marginal_gibbs,
    dirichlet_posterior,
    lift_phi_uniform,
)
from canonred.signature import build_signature_quotient
from canonred.viz import plot_posterior_histogram, plot_runtime_bar
from canonred.worlds import World

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "nde"


# ----------------------------------------------------------------------
# Query construction
# ----------------------------------------------------------------------


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
    return G, Q, cs, (v_w1, v_y00, v_y01)


# ----------------------------------------------------------------------
# Closed-form sanity
# ----------------------------------------------------------------------


def closed_form(p_w1: float, p_y00: float, p_y01: float) -> tuple[float, float]:
    L = max(0.0, p_y00 - p_w1) + max(0.0, p_y01 - (1.0 - p_w1))
    U = min(p_y00, 1.0 - p_w1) + min(p_y01, p_w1)
    return L, U


def run_sanity(G, Q, cs) -> None:
    print("=== Sanity: 8-state vs 128-state LP at p* = (0.6, 0.3, 0.7) ===")
    p_star = np.array([0.6, 0.3, 0.7])
    coef_l = build_lp_lifted(G, Q, cs)
    q = build_signature_quotient(G, Q, cs)
    coef_r = build_lp_reduced(q)

    lb_l, ub_l, _ = solve_sharp_bounds(coef_l, p_star)
    lb_r, ub_r, _ = solve_sharp_bounds(coef_r, p_star)
    L_cf, U_cf = closed_form(*p_star)

    print(f"  lifted  (dim={coef_l.dim}): [{lb_l:.6f}, {ub_l:.6f}]")
    print(f"  reduced (dim={coef_r.dim}): [{lb_r:.6f}, {ub_r:.6f}]")
    print(f"  closed-form              : [{L_cf:.6f}, {U_cf:.6f}]")
    print(f"  max |lifted - reduced|   : {max(abs(lb_l-lb_r), abs(ub_l-ub_r)):.2e}")
    print(f"  max |reduced - closed|   : {max(abs(lb_r-L_cf), abs(ub_r-U_cf)):.2e}\n")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _phi_true_product(p_w1: float, p_y00: float, p_y01: float) -> np.ndarray:
    probs = np.zeros(8, dtype=np.float64)
    for a in (0, 1):
        pa = p_w1 if a == 1 else (1.0 - p_w1)
        for b in (0, 1):
            pb = p_y00 if b == 1 else (1.0 - p_y00)
            for c in (0, 1):
                pc = p_y01 if c == 1 else (1.0 - p_y01)
                idx = (a << 2) + (b << 1) + c
                probs[idx] = pa * pb * pc
    probs /= probs.sum()
    return probs


def _infer_class_to_abc(quotient, G) -> dict[int, int]:
    from canonred.canonical import enumerate_states, eval_cfvar

    w1 = World.from_dict("x1", {"X": 1})
    w00 = World.from_dict("x0w0", {"X": 0, "W": 0})
    w01 = World.from_dict("x0w1", {"X": 0, "W": 1})
    v_w1 = CFVar.standard("W", w1, G)
    v_y00 = CFVar.standard("Y", w00, G)
    v_y01 = CFVar.standard("Y", w01, G)

    mapping: dict[int, int] = {}
    for i, u in enumerate(enumerate_states(G)):
        t = quotient.state_class_assignment[i]
        if t in mapping:
            continue
        a = int(eval_cfvar(u, v_w1, G))
        b = int(eval_cfvar(u, v_y00, G))
        c = int(eval_cfvar(u, v_y01, G))
        mapping[t] = (a << 2) | (b << 1) | c
        if len(mapping) == quotient.reduced_dim:
            break
    return mapping


def _propagate_q_to_lifted(phi_samples: np.ndarray, quotient) -> np.ndarray:
    """Return Q_128 per posterior draw via uniform lift."""
    c_lifted = quotient.objective[list(quotient.state_class_assignment)]
    Q_128 = np.empty(phi_samples.shape[0], dtype=np.float64)
    for s in range(phi_samples.shape[0]):
        psi = lift_phi_uniform(phi_samples[s], quotient)
        Q_128[s] = c_lifted @ psi
    return Q_128


def _runtime_comparison(
    quotient,
    coef_l,
    phi_samples: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    cached_r = CachedSharpLP(build_lp_reduced(quotient))
    cached_l = CachedSharpLP(coef_l)
    rt_r = np.zeros(K)
    rt_l = np.zeros(K)
    idx = rng.integers(0, phi_samples.shape[0], size=K)
    M_tilde = np.asarray(quotient.constraint_matrix, dtype=np.float64)
    for k, i in enumerate(idx):
        p_samp = M_tilde @ phi_samples[i]
        _, _, dR = cached_r.solve(p_samp)
        _, _, dL = cached_l.solve(p_samp)
        rt_r[k] = dR["solver_time"]
        rt_l[k] = dL["solver_time"]
    med_r = float(np.median(rt_r)) * 1000.0
    med_l = float(np.median(rt_l)) * 1000.0
    return med_r, med_l, (med_l / med_r if med_r > 0 else float("inf"))


def _per_n_row(n, Q_8, Q_128, med_r, med_l, speedup) -> dict:
    q025, q50, q975 = np.quantile(Q_8, [0.025, 0.5, 0.975])
    max_abs = float(np.max(np.abs(Q_128 - Q_8)))
    return {
        "n": n,
        "q_025": round(q025, 6),
        "q_500": round(q50, 6),
        "q_975": round(q975, 6),
        "max_abs_q_diff": max_abs,
        "solve_time_8_ms_median": round(med_r, 4),
        "solve_time_128_ms_median": round(med_l, 4),
        "speedup": round(speedup, 3),
    }


# ----------------------------------------------------------------------
# Regimes
# ----------------------------------------------------------------------


def run_regime_II(G, Q, cs, n_grid, S, seed, K) -> list[dict]:
    quotient = build_signature_quotient(G, Q, cs)
    coef_l = build_lp_lifted(G, Q, cs)

    p_star = (0.6, 0.3, 0.7)
    phi_true_8 = _phi_true_product(*p_star)
    class_to_abc = _infer_class_to_abc(quotient, G)
    phi_true_reduced = np.array(
        [phi_true_8[class_to_abc[t]] for t in range(quotient.reduced_dim)], dtype=np.float64
    )

    L_truth, U_truth = closed_form(*p_star)
    print(f"=== Regime II (multinomial t-data): p* = {p_star}, truth = [{L_truth:.4f}, {U_truth:.4f}] ===")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for n in n_grid:
        counts_data_rng = np.random.default_rng(rng.integers(1_000_000))
        counts = counts_data_rng.multinomial(n, phi_true_reduced)
        phi_samples = dirichlet_posterior(counts, alpha=1.0, S=S, seed=int(rng.integers(1_000_000)))
        Q_8 = phi_samples @ quotient.objective
        Q_128 = _propagate_q_to_lifted(phi_samples, quotient)
        med_r, med_l, speedup = _runtime_comparison(quotient, coef_l, phi_samples, K, rng)

        row = _per_n_row(n, Q_8, Q_128, med_r, med_l, speedup)
        print(
            f"  n={n:5d}  Q_8 [{row['q_025']:.4f},{row['q_500']:.4f},{row['q_975']:.4f}]  "
            f"max|Q128-Q8|={row['max_abs_q_diff']:.2e}  "
            f"solve_8={row['solve_time_8_ms_median']:.3f}ms  "
            f"solve_128={row['solve_time_128_ms_median']:.3f}ms  "
            f"speedup={row['speedup']:.2f}x"
        )
        # Plot histogram for this n
        plot_posterior_histogram(
            Q_8, Q_128,
            out_path=OUT_DIR / f"q_hist_regimeII_n{n}.png",
            title=f"Regime II (multinomial) — n={n}",
            bounds_truth=(L_truth, U_truth),
        )
        rows.append(row)
    return rows


def run_regime_I(G, Q, cs, n_grid, S, seed, K) -> list[dict]:
    quotient = build_signature_quotient(G, Q, cs)
    coef_l = build_lp_lifted(G, Q, cs)

    p_star = (0.6, 0.3, 0.7)
    L_truth, U_truth = closed_form(*p_star)
    print(f"\n=== Regime I (binomial 3-arm): p* = {p_star}, truth = [{L_truth:.4f}, {U_truth:.4f}] ===")

    rng = np.random.default_rng(seed + 1)
    rows: list[dict] = []
    for n in n_grid:
        data_rng = np.random.default_rng(rng.integers(1_000_000))
        # Observe K_r ~ Bin(n, p_r) for each of the 3 marginals
        ks = [int(data_rng.binomial(n, p)) for p in p_star]
        obs = [(ks[i], n) for i in range(3)]
        phi_samples = binomial_marginal_gibbs(
            quotient, observations=obs, alpha=1.0, S=S, burn=min(1000, S),
            seed=int(rng.integers(1_000_000)),
        )
        Q_8 = phi_samples @ quotient.objective
        Q_128 = _propagate_q_to_lifted(phi_samples, quotient)
        med_r, med_l, speedup = _runtime_comparison(quotient, coef_l, phi_samples, K, rng)

        row = _per_n_row(n, Q_8, Q_128, med_r, med_l, speedup)
        print(
            f"  n={n:5d}  Q_8 [{row['q_025']:.4f},{row['q_500']:.4f},{row['q_975']:.4f}]  "
            f"max|Q128-Q8|={row['max_abs_q_diff']:.2e}  "
            f"solve_8={row['solve_time_8_ms_median']:.3f}ms  "
            f"solve_128={row['solve_time_128_ms_median']:.3f}ms  "
            f"speedup={row['speedup']:.2f}x"
        )
        plot_posterior_histogram(
            Q_8, Q_128,
            out_path=OUT_DIR / f"q_hist_regimeI_n{n}.png",
            title=f"Regime I (binomial 3-arm) — n={n}",
            bounds_truth=(L_truth, U_truth),
        )
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def dump_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", type=int, nargs="+", default=[200, 500, 1000])
    ap.add_argument("--S", type=int, default=3000)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regimes", nargs="+", choices=["I", "II"], default=["I", "II"])
    ap.add_argument("--no-sanity", action="store_true")
    args = ap.parse_args()

    G, Q, cs, _ = build_nde_l25()
    if not args.no_sanity:
        run_sanity(G, Q, cs)

    t0 = time.perf_counter()
    all_rows: dict[str, list[dict]] = {}
    if "II" in args.regimes:
        all_rows["II"] = run_regime_II(G, Q, cs, args.n_grid, args.S, args.seed, args.K)
        dump_csv(OUT_DIR / "summary_regime_II.csv", all_rows["II"])
        plot_runtime_bar(
            all_rows["II"],
            out_path=OUT_DIR / "runtime_regime_II.png",
            title="LP solve time (Regime II)",
        )
    if "I" in args.regimes:
        all_rows["I"] = run_regime_I(G, Q, cs, args.n_grid, args.S, args.seed, args.K)
        dump_csv(OUT_DIR / "summary_regime_I.csv", all_rows["I"])
        plot_runtime_bar(
            all_rows["I"],
            out_path=OUT_DIR / "runtime_regime_I.png",
            title="LP solve time (Regime I)",
        )
    total = time.perf_counter() - t0
    print(f"\n[done] wall-clock {total:.2f}s")
    print(f"[outputs] {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
