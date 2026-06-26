"""Reproducible, stabilized re-measurement of the NDE L_{2.5} posterior + runtime.

Same outputs as run.py (Table 3/4, Fig 3/5/7) but with a hardened timing harness:
  - thread/BLAS pinning (set BEFORE numpy import, below)
  - ECOS tolerances pinned + strict (no silent SCS fallback)
  - batched amortized timing on a FIXED representative p (decoupled from
    posterior RNG), median-of-batches + IQR
  - per-seed sharded CSVs (a `seed` column; slurm-array friendly)
  - full provenance + LP fingerprints in meta.json
  - a CPU pre-flight check (refuses to time on a busy / non-SLURM box)

Build + posterior samplers are imported from run.py (unchanged, validated).
Run one seed per process; sweep seeds via a slurm array.
"""

# --- thread pinning MUST precede numpy import ---------------------------------
import os  # noqa: E402
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "ACCELERATE_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
# -----------------------------------------------------------------------------

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # experiments/
sys.path.insert(0, str(_HERE.parents[2]))  # repo root (for canonred if not pip-installed)

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    ECOS_OPTS, closed_form_nde_l25, dump_csv, dump_meta, lp_fingerprint,
    paired_speedup, preflight_cpu_check, provenance, time_solve_stable,
)
from run import (  # noqa: E402  (validated build + posterior helpers)
    build_nde_l25, draw_regime_I, draw_regime_II, phi_true_reduced,
)

from canonred.lp import CachedSharpLP, build_lp_lifted, build_lp_reduced  # noqa: E402
from canonred.posterior import lift_phi_uniform  # noqa: E402
from canonred.signature import build_signature_quotient  # noqa: E402


def _row_solver(res, tag):
    return {
        f"solver_med_{tag}_ms": round(res["solver_med_s"] * 1e3, 6),
        f"solver_iqr_lo_{tag}_ms": round(res["solver_iqr_lo_s"] * 1e3, 6),
        f"solver_iqr_hi_{tag}_ms": round(res["solver_iqr_hi_s"] * 1e3, 6),
    }


def _row_wall(res, tag):
    return {
        f"wall_med_{tag}_ms": round(res["wall_med_s"] * 1e3, 6),
        f"wall_iqr_lo_{tag}_ms": round(res["wall_iqr_lo_s"] * 1e3, 6),
        f"wall_iqr_hi_{tag}_ms": round(res["wall_iqr_hi_s"] * 1e3, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-grid", type=int, nargs="+", default=[200, 500, 1000, 5000])
    ap.add_argument("--S", type=int, default=10000)
    ap.add_argument("--burn", type=int, default=5000)
    ap.add_argument("--regimes", nargs="+", choices=["I", "II"], default=["I", "II"])
    # timing knobs
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-batch", type=int, default=200, help="inner solves/batch, 8-state")
    ap.add_argument("--n-batch-lifted", type=int, default=50, help="inner solves/batch, 128-state")
    ap.add_argument("--n-repeats", type=int, default=20, help="outer batches for median/IQR")
    ap.add_argument("--warm-start", dest="warm_start", action="store_true", default=True)
    ap.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    ap.add_argument("--solver", default="ECOS")
    # preflight
    ap.add_argument("--max-load-per-core", type=float, default=0.4)
    ap.add_argument("--require-slurm", action="store_true")
    ap.add_argument("--allow-busy", action="store_true", help="do not abort if machine busy")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    pf = preflight_cpu_check(max_load_per_core=args.max_load_per_core,
                             abort=not args.allow_busy, require_slurm=args.require_slurm)

    # ---- deterministic build ----
    G, Q, cs = build_nde_l25()
    coef_l = build_lp_lifted(G, Q, cs)
    quo = build_signature_quotient(G, Q, cs)
    coef_r = build_lp_reduced(quo)
    phi_true = phi_true_reduced(quo, G)
    M_tilde = np.asarray(quo.constraint_matrix, dtype=np.float64)
    p_fixed = M_tilde @ phi_true  # deterministic feasible representative p
    fp = {"reduced": lp_fingerprint(coef_r), "lifted": lp_fingerprint(coef_l)}
    print(f"[setup] dim_reduced={coef_r.dim} dim_lifted={coef_l.dim} "
          f"p_fixed={np.round(p_fixed, 4).tolist()}", flush=True)

    # ---- TIMING (LP size is n-independent: measure once per seed) ----
    eopts = ECOS_OPTS if args.solver == "ECOS" else {}
    cr = CachedSharpLP(coef_r, solver=args.solver, solver_opts=eopts, strict=True)
    cl = CachedSharpLP(coef_l, solver=args.solver, solver_opts=eopts, strict=True)
    res_r = time_solve_stable(cr, p_fixed, n_warmup=args.n_warmup,
                              n_batch=args.n_batch, n_repeats=args.n_repeats,
                              rotate_pool=args.warm_start)
    res_l = time_solve_stable(cl, p_fixed, n_warmup=args.n_warmup,
                              n_batch=args.n_batch_lifted, n_repeats=args.n_repeats,
                              rotate_pool=args.warm_start)
    spd_s = paired_speedup(res_l, res_r, "solver")
    spd_w = paired_speedup(res_l, res_r, "wall")
    solvers_used = "+".join(sorted(set(res_r["solvers_used"]) | set(res_l["solvers_used"])))
    host = provenance().get("host")

    common_cols = {
        "seed": seed, "dim_reduced": coef_r.dim, "dim_lifted": coef_l.dim,
        "n_warmup": args.n_warmup, "n_batch_8": args.n_batch,
        "n_batch_128": args.n_batch_lifted, "n_repeats": args.n_repeats,
        "warm_start": args.warm_start, "solver": args.solver,
        "solvers_used": solvers_used, "host": host,
        "load_per_core_1m": round(pf["load_per_core_1m"], 3),
        "slurm_job": pf.get("slurm_job_id"),
    }
    table3 = [{**common_cols, **_row_solver(res_r, "8"), **_row_solver(res_l, "128"),
               "speedup": round(spd_s["speedup"], 3),
               "speedup_iqr_lo": round(spd_s["speedup_iqr_lo"], 3),
               "speedup_iqr_hi": round(spd_s["speedup_iqr_hi"], 3)}]
    table4 = [{**common_cols, **_row_wall(res_r, "8"), **_row_wall(res_l, "128"),
               "speedup_wall": round(spd_w["speedup"], 3),
               "speedup_wall_iqr_lo": round(spd_w["speedup_iqr_lo"], 3),
               "speedup_wall_iqr_hi": round(spd_w["speedup_iqr_hi"], 3)}]
    dump_csv(table3, out / f"table3_solve_time_seed{seed}.csv")
    dump_csv(table4, out / f"table4_wall_time_seed{seed}.csv")
    print(f"[timing] solver speedup={spd_s['speedup']:.2f}x "
          f"IQR=[{spd_s['speedup_iqr_lo']:.2f},{spd_s['speedup_iqr_hi']:.2f}]  "
          f"8={res_r['solver_med_s']*1e3:.4f}ms 128={res_l['solver_med_s']*1e3:.4f}ms", flush=True)

    # ---- POSTERIOR + FIGURE DATA (per regime, n) ----
    rng_master = np.random.default_rng(seed)
    agreement: list[dict] = []
    fig5: list[dict] = []
    c_lifted = quo.objective[list(quo.state_class_assignment)]
    for regime in args.regimes:
        for n in args.n_grid:
            seed_n = int(rng_master.integers(1_000_000))
            if regime == "I":
                phi = draw_regime_I(quo, G, cs, n, args.S, args.burn, seed_n)
            else:
                phi = draw_regime_II(quo, phi_true, n, args.S, seed_n)
            Q_8 = phi @ quo.objective
            Q_128 = np.array([float(c_lifted @ lift_phi_uniform(phi[s], quo))
                              for s in range(phi.shape[0])])
            p_post = phi @ M_tilde.T
            L = np.empty(phi.shape[0]); U = np.empty(phi.shape[0])
            for s in range(phi.shape[0]):
                L[s], U[s] = closed_form_nde_l25(*p_post[s])
            dump_csv(
                [{"seed": seed, "draw": s, "Q_8": float(Q_8[s]), "Q_128": float(Q_128[s]),
                  "Q_diff": float(Q_128[s] - Q_8[s]), "L_closed": float(L[s]),
                  "U_closed": float(U[s])} for s in range(phi.shape[0])],
                out / f"fig3_per_draw_regime{regime}_n{n}_seed{seed}.csv",
            )
            for s in range(phi.shape[0]):
                fig5.append({"regime": regime, "n": n, "seed": seed, "draw": s,
                             "L": float(L[s]), "U": float(U[s])})
            d = np.abs(Q_128 - Q_8)
            agreement.append({"regime": regime, "n": n, "seed": seed,
                              "median_abs_q_diff": float(np.median(d)),
                              "p99_abs_q_diff": float(np.quantile(d, 0.99)),
                              "max_abs_q_diff": float(np.max(d))})
            q025, q50, q975 = np.quantile(Q_8, [0.025, 0.5, 0.975])
            print(f"  regime {regime} n={n:5d} Q_8[{q025:.4f},{q50:.4f},{q975:.4f}] "
                  f"max|Q128-Q8|={float(np.max(d)):.2e}", flush=True)

    dump_csv(agreement, out / f"fig7_agreement_per_n_seed{seed}.csv")
    dump_csv(fig5, out / f"fig5_per_draw_bounds_seed{seed}.csv")

    meta = {
        "command": " ".join(sys.argv),
        "seed": seed,
        "parameters": vars(args),
        "fingerprint": fp,
        "preflight": {k: pf.get(k) for k in
                      ("n_cpu", "loadavg", "load_per_core_1m", "slurm_job_id", "busy")},
        "provenance": provenance(),
        "timing": {"speedup_solver": spd_s, "speedup_wall": spd_w,
                   "reduced": {k: res_r[k] for k in res_r if not k.startswith("_")},
                   "lifted": {k: res_l[k] for k in res_l if not k.startswith("_")}},
    }
    dump_meta(meta, out / f"meta_seed{seed}.json")
    print(f"[done] seed={seed} -> {out}", flush=True)


if __name__ == "__main__":
    main()
