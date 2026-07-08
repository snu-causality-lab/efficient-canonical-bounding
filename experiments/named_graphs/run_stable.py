"""Reproducible re-measurement of named-fixture reduced-vs-lifted timing.

Same fixtures as run.py (bow / IV / Non-IV / front-door / NDE / PNS / ternary
NDE) under full P_obs constraints, but timed with the stabilized harness
(thread pinning, pinned ECOS opts + strict, batched amortized timing on a fixed
p, median + IQR). One seed per process; sweep seeds via a slurm array.

Replaces the earlier per-graph speedups (which were corrupted by unstable timing,
e.g. the ternary full solve was mis-reported as ~73 s when it is ~21 ms).
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
sys.path.insert(0, str(_HERE.parents[2]))  # repo root

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    ECOS_OPTS, dump_csv, dump_meta, lp_fingerprint, paired_speedup,
    preflight_cpu_check, provenance, time_solve_stable,
)
from run import FIXTURES  # noqa: E402  (validated fixture factories)

from canonred.lp import CachedSharpLP, build_lp_lifted, build_lp_reduced  # noqa: E402
from canonred.observation import pobs_constraints  # noqa: E402
from canonred.signature import build_signature_quotient  # noqa: E402


def _nb_lifted(dim: int) -> int:
    if dim < 256:
        return 50
    if dim < 1024:
        return 20
    if dim < 8192:
        return 5
    return 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-batch", type=int, default=200, help="inner solves/batch, reduced")
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--warm-start", dest="warm_start", action="store_true", default=True)
    ap.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    ap.add_argument("--solver", default="ECOS")
    ap.add_argument("--max-load-per-core", type=float, default=0.4)
    ap.add_argument("--require-slurm", action="store_true")
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    pf = preflight_cpu_check(max_load_per_core=args.max_load_per_core,
                             abort=not args.allow_busy, require_slurm=args.require_slurm)
    eopts = ECOS_OPTS if args.solver == "ECOS" else {}
    host = provenance().get("host")

    rows: list[dict] = []
    for factory in FIXTURES:
        name, G, Q = factory()
        cs = pobs_constraints(G)
        coef_l = build_lp_lifted(G, Q, cs)
        q = build_signature_quotient(G, Q, cs)
        coef_r = build_lp_reduced(q)
        p_fixed = coef_l.M @ (np.ones(coef_l.dim) / coef_l.dim)  # deterministic feasible p

        cr = CachedSharpLP(coef_r, solver=args.solver, solver_opts=eopts, strict=True)
        cl = CachedSharpLP(coef_l, solver=args.solver, solver_opts=eopts, strict=True)
        # agreement on the fixed p (sharp bounds must match)
        lb_r, ub_r, _ = cr.solve(p_fixed)
        lb_l, ub_l, _ = cl.solve(p_fixed)

        nb_l = _nb_lifted(coef_l.dim)
        res_r = time_solve_stable(cr, p_fixed, n_warmup=args.n_warmup,
                                  n_batch=args.n_batch, n_repeats=args.n_repeats,
                                  rotate_pool=args.warm_start)
        res_l = time_solve_stable(cl, p_fixed, n_warmup=args.n_warmup,
                                  n_batch=nb_l, n_repeats=args.n_repeats,
                                  rotate_pool=args.warm_start)
        spd = paired_speedup(res_l, res_r, "solver")
        rows.append({
            "fixture": name, "seed": seed, "n_nodes": len(G.nodes),
            "lifted_dim": coef_l.dim, "reduced_dim": coef_r.dim,
            "reduced_med_ms": round(res_r["solver_med_s"] * 1e3, 6),
            "reduced_iqr_lo_ms": round(res_r["solver_iqr_lo_s"] * 1e3, 6),
            "reduced_iqr_hi_ms": round(res_r["solver_iqr_hi_s"] * 1e3, 6),
            "lifted_med_ms": round(res_l["solver_med_s"] * 1e3, 6),
            "lifted_iqr_lo_ms": round(res_l["solver_iqr_lo_s"] * 1e3, 6),
            "lifted_iqr_hi_ms": round(res_l["solver_iqr_hi_s"] * 1e3, 6),
            "speedup": round(spd["speedup"], 3),
            "speedup_iqr_lo": round(spd["speedup_iqr_lo"], 3),
            "speedup_iqr_hi": round(spd["speedup_iqr_hi"], 3),
            "max_abs_lb_diff": float(abs(lb_l - lb_r)),
            "max_abs_ub_diff": float(abs(ub_l - ub_r)),
            "n_batch_reduced": args.n_batch, "n_batch_lifted": nb_l,
            "n_repeats": args.n_repeats, "warm_start": args.warm_start,
            "solver": args.solver,
            "solvers_used": "+".join(sorted(set(res_r["solvers_used"]) | set(res_l["solvers_used"]))),
            "host": host,
            "fp_lifted_M": lp_fingerprint(coef_l)["M_sha256"],
        })
        print(f"  {name:12s} lifted={coef_l.dim:5d} reduced={coef_r.dim:3d} "
              f"speedup={spd['speedup']:.2f}x IQR=[{spd['speedup_iqr_lo']:.2f},{spd['speedup_iqr_hi']:.2f}] "
              f"agree(lb,ub)=({abs(lb_l-lb_r):.1e},{abs(ub_l-ub_r):.1e})", flush=True)

    dump_csv(rows, out / f"named_fixtures_stable_seed{seed}.csv")
    dump_meta({"command": " ".join(sys.argv), "seed": seed, "parameters": vars(args),
               "preflight": {k: pf.get(k) for k in
                             ("n_cpu", "loadavg", "load_per_core_1m", "slurm_job_id", "busy")},
               "provenance": provenance()},
              out / f"meta_seed{seed}.json")
    print(f"[done] seed={seed} -> {out}", flush=True)


if __name__ == "__main__":
    main()
