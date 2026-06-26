"""Reproducible stress-scaling re-measurement (Appendix F, Table 5).

Same fiber-inflation experiment as run.py, with the hardened timing harness
(thread pinning, pinned ECOS opts + strict, batched amortized timing on a fixed
p, median + IQR, preflight, provenance). One seed per process.
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
from run import build_nde_l25, inflate  # noqa: E402

from canonred.lp import CachedSharpLP, build_lp_reduced  # noqa: E402
from canonred.signature import build_signature_quotient  # noqa: E402


def _nb_for(dim: int) -> int:
    return 50 if dim < 1024 else 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fibers", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-batch", type=int, default=200, help="inner solves/batch, 8-state")
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
    G, Q, cs = build_nde_l25()
    q = build_signature_quotient(G, Q, cs)
    coef_r = build_lp_reduced(q)
    M = np.asarray(q.constraint_matrix, dtype=np.float64)
    p_fixed = M @ (np.ones(coef_r.dim) / coef_r.dim)  # deterministic feasible p

    cached_r = CachedSharpLP(coef_r, solver=args.solver, solver_opts=eopts, strict=True)
    res_r = time_solve_stable(cached_r, p_fixed, n_warmup=args.n_warmup,
                              n_batch=args.n_batch, n_repeats=args.n_repeats,
                              rotate_pool=args.warm_start)
    host = provenance().get("host")
    print(f"[setup] reduced 8-state solver med={res_r['solver_med_s']*1e3:.4f}ms", flush=True)

    rows: list[dict] = []
    for fiber in args.fibers:
        coef_l = inflate(coef_r, fiber)
        nb = _nb_for(coef_l.dim)
        cached_l = CachedSharpLP(coef_l, solver=args.solver, solver_opts=eopts, strict=True)
        res_l = time_solve_stable(cached_l, p_fixed, n_warmup=args.n_warmup,
                                  n_batch=nb, n_repeats=args.n_repeats,
                                  rotate_pool=args.warm_start)
        spd = paired_speedup(res_l, res_r, "solver")
        rows.append({
            "fiber": fiber, "dim_lifted": coef_l.dim, "dim_reduced": coef_r.dim, "seed": seed,
            "solver_med_8_ms": round(res_r["solver_med_s"] * 1e3, 6),
            "solver_iqr_lo_8_ms": round(res_r["solver_iqr_lo_s"] * 1e3, 6),
            "solver_iqr_hi_8_ms": round(res_r["solver_iqr_hi_s"] * 1e3, 6),
            "solver_med_lifted_ms": round(res_l["solver_med_s"] * 1e3, 6),
            "solver_iqr_lo_lifted_ms": round(res_l["solver_iqr_lo_s"] * 1e3, 6),
            "solver_iqr_hi_lifted_ms": round(res_l["solver_iqr_hi_s"] * 1e3, 6),
            "speedup": round(spd["speedup"], 3),
            "speedup_iqr_lo": round(spd["speedup_iqr_lo"], 3),
            "speedup_iqr_hi": round(spd["speedup_iqr_hi"], 3),
            "n_warmup": args.n_warmup, "n_batch_8": args.n_batch, "n_batch_lifted": nb,
            "n_repeats": args.n_repeats, "warm_start": args.warm_start, "solver": args.solver,
            "solvers_used": "+".join(sorted(set(res_r["solvers_used"]) | set(res_l["solvers_used"]))),
            "host": host, "load_per_core_1m": round(pf["load_per_core_1m"], 3),
            "fp_lifted_M": lp_fingerprint(coef_l)["M_sha256"],
        })
        print(f"fiber={fiber:3d} dim={coef_l.dim:5d} "
              f"speedup={spd['speedup']:.1f}x IQR=[{spd['speedup_iqr_lo']:.1f},{spd['speedup_iqr_hi']:.1f}]",
              flush=True)

    dump_csv(rows, out / f"table5_stress_scaling_seed{seed}.csv")
    meta = {
        "command": " ".join(sys.argv), "seed": seed, "parameters": vars(args),
        "preflight": {k: pf.get(k) for k in
                      ("n_cpu", "loadavg", "load_per_core_1m", "slurm_job_id", "busy")},
        "provenance": provenance(),
        "reduced_fingerprint": lp_fingerprint(coef_r),
    }
    dump_meta(meta, out / f"meta_seed{seed}.json")
    print(f"[done] seed={seed} -> {out}", flush=True)


if __name__ == "__main__":
    main()
