"""Pobs-regime data for appendix figures (16-state safe quotient vs query-only).

For the NDE graph under full P_obs we compare three LPs at the SAME data:
  - 128-state (full lifted)                              -> reference bounds
  - 16-state  sigma-quotient (full P_obs, *safe*)        -> must match 128
  - 8-state   L_{2.5}-marginal reduction (*query-only*)  -> drops P_obs cells -> looser

We sample R random feasible canonical distributions psi over Omega_full, form
the induced data (p_obs for 128/16, the 3 marginals for the 8-state query-only),
solve all three, and record the bound discrepancies vs the 128 reference.

Output: pobs_discrepancy.csv  (one row per sample)
  safe_lb_diff, safe_ub_diff  = |L16-L128|, |U16-U128|   (~ solver tolerance)
  qonly_lb_diff, qonly_ub_diff = |L8q-L128|, |U8q-U128|  (nonzero: query-only loosens)
"""

import os  # noqa: E402
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # experiments/
sys.path.insert(0, str(_HERE.parents[2]))   # repo root
sys.path.insert(0, str(_HERE.parents[1] / "paper_posterior_sweep"))

import numpy as np  # noqa: E402

from _common import ECOS_OPTS, dump_csv  # noqa: E402
from run import build_nde_l25  # noqa: E402  (gives G, Q, cs = L2.5 marginals)

from canonred.lp import CachedSharpLP, build_lp_lifted, build_lp_reduced  # noqa: E402
from canonred.observation import pobs_constraints  # noqa: E402
from canonred.signature import build_signature_quotient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=2000, help="# random feasible psi")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(_HERE.parent))
    args = ap.parse_args()

    G, Q, cs_marg = build_nde_l25()          # cs_marg = 3 L2.5 marginals
    cs_pobs = pobs_constraints(G)            # 8 full P_obs cell constraints

    coef_l128 = build_lp_lifted(G, Q, cs_pobs)         # full, P_obs
    coef_marg128 = build_lp_lifted(G, Q, cs_marg)      # full, marginal rows (to form p_marg)
    coef_r16 = build_lp_reduced(build_signature_quotient(G, Q, cs_pobs))  # safe quotient
    coef_r8 = build_lp_reduced(build_signature_quotient(G, Q, cs_marg))   # query-only
    print(f"[pobs] dims: full={coef_l128.dim} safe={coef_r16.dim} query-only={coef_r8.dim}", flush=True)

    c128 = CachedSharpLP(coef_l128, solver_opts=ECOS_OPTS, strict=True)
    c16 = CachedSharpLP(coef_r16, solver_opts=ECOS_OPTS, strict=True)
    c8 = CachedSharpLP(coef_r8, solver_opts=ECOS_OPTS, strict=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.R):
        psi = rng.dirichlet(np.ones(coef_l128.dim))
        p_obs = coef_l128.M @ psi
        p_marg = coef_marg128.M @ psi
        try:
            L128, U128, _ = c128.solve(p_obs)
            L16, U16, _ = c16.solve(p_obs)
            L8, U8, _ = c8.solve(p_marg)
        except Exception:
            continue
        rows.append({
            "sample": i,
            "safe_lb_diff": abs(L16 - L128), "safe_ub_diff": abs(U16 - U128),
            "qonly_lb_diff": abs(L8 - L128), "qonly_ub_diff": abs(U8 - U128),
            "width_full": (U128 - L128), "width_qonly": (U8 - L8),
        })
    dump_csv(rows, Path(args.out_dir) / "pobs_discrepancy.csv")
    sd = np.array([max(r["safe_lb_diff"], r["safe_ub_diff"]) for r in rows])
    qd = np.array([max(r["qonly_lb_diff"], r["qonly_ub_diff"]) for r in rows])
    print(f"[pobs] safe(16) max|diff|: median={np.median(sd):.2e} p99={np.quantile(sd,0.99):.2e} max={sd.max():.2e}")
    print(f"[pobs] query-only(8) max|diff|: median={np.median(qd):.3f} p99={np.quantile(qd,0.99):.3f} "
          f"frac>1e-6={np.mean(qd>1e-6):.2%}")
    print(f"[done] {len(rows)} samples -> {args.out_dir}/pobs_discrepancy.csv", flush=True)


if __name__ == "__main__":
    main()
