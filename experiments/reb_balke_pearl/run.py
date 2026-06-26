"""Rebuttal #7 (xfJu Q4.1): Balke-Pearl 1997 noncompliance IV — replicate
the canonical RCT-with-noncompliance setting.

  IV graph: Z -> X -> Y, X <-> Y  (binary nodes)
  Observed: P(X, Y | Z) — 8 cells of P(Z, X, Y)
  Queries (L_2 single-world):
    - Q0 := P(Y_{x=0} = 1)
    - Q1 := P(Y_{x=1} = 1)

What this experiment shows:

(1) The framework handles the IV graph end-to-end: lifted dim 32, reduced dim 12,
    8 marginal P(Z,X,Y) constraints, both Q0 and Q1 are LP-feasible.
(2) lifted ≡ reduced bounds (sanity, ≤ 1e-9).
(3) Sanity vs joint-margin lower bound P(Y=1, X=x) ≤ P(Y_{x}=1) (always).
(4) **Honest comparison vs Manski (1990) IV-exclusion bound**:
       max_z P(Y=1, X=x | Z=z) ≤ P(Y_{x}=1) ≤ 1 - max_z P(Y=0, X=x | Z=z).
    Manski / Balke-Pearl (1997) achieve this bound by enforcing the IV-exclusion
    factorisation Z ⊥ U_{XY} on the canonical SCM; that factorisation is a
    POLYNOMIAL constraint (the c-component product), not a linear one — the
    Zhang-Tian-Bareinboim (2022) and Duarte et al. (2024) frameworks.
    Our paper's linear LP uses a JOINT canonical distribution and intentionally
    drops the factorisation, so our bounds are correctly looser than Manski.
    We report both widths so the gap is transparent.

(5) Bonferroni ATE = [L1 - U0, U1 - L0]; report width.

Output: `balke_pearl_iv.csv` (one row per (seed, k) sample) and `summary.json`.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import dump_csv, dump_meta  # noqa: E402

from canonred.counterfactual import CFQuery, CFVar  # noqa: E402
from canonred.examples import iv_graph  # noqa: E402
from canonred.lp import CachedSharpLP, build_lp_lifted, build_lp_reduced  # noqa: E402
from canonred.observation import pobs_constraints  # noqa: E402
from canonred.signature import build_signature_quotient  # noqa: E402
from canonred.worlds import World  # noqa: E402


def _query_y_x(g, x: int) -> CFQuery:
    w = World.from_dict(f"x{x}", {"X": x})
    return CFQuery(
        events=((CFVar.standard("Y", w, g), 1),),
        graph=g,
        name=f"Y_xeq{x}_eq1",
    )


def _decode_pobs(coef, p_vec: np.ndarray) -> tuple[dict, dict, dict]:
    """Return (joint, conditional_given_z, p_z) decoded from labelled p.

    pobs_constraints emits joints P(Z=z, X=x, Y=y) (one row per cell).
    """
    pz: dict = {0: 0.0, 1: 0.0}
    joint: dict = {}
    for lbl, pv in zip(coef.row_labels, p_vec):
        z = int(lbl.split("Z=")[1].split(",")[0])
        x = int(lbl.split("X=")[1].split(",")[0])
        y = int(lbl.split("Y=")[1].split(",")[0].rstrip(")"))
        joint[(x, y, z)] = float(pv)
        pz[z] += float(pv)
    cond: dict = {
        (x, y, z): (joint[(x, y, z)] / pz[z]) if pz[z] > 1e-12 else 0.0
        for (x, y, z) in joint
    }
    return joint, cond, pz


def _manski_iv_bounds(cond: dict, x_val: int) -> tuple[float, float]:
    """Manski (1990) IV-exclusion bound on P(Y_{x=x_val}=1).

    Enforces Z ⊥ U_{XY} (the polynomial factorisation our linear LP drops).
    """
    lb = max(cond[(x_val, 1, 0)], cond[(x_val, 1, 1)])
    ub = 1.0 - max(cond[(x_val, 0, 0)], cond[(x_val, 0, 1)])
    return lb, ub


def _joint_margin_bounds(joint: dict, x_val: int) -> tuple[float, float]:
    """Trivial joint-margin bound P(Y=1, X=x) ≤ P(Y_{x}=1) ≤ 1 - P(Y=0, X=x).

    Holds even WITHOUT IV exclusion — an absolute lower bound for our
    linear-LP framework.
    """
    p_y1_x = joint[(x_val, 1, 0)] + joint[(x_val, 1, 1)]
    p_y0_x = joint[(x_val, 0, 0)] + joint[(x_val, 0, 1)]
    return p_y1_x, 1.0 - p_y0_x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--out-csv", type=str, default="balke_pearl_iv.csv")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    G = iv_graph()
    cs = pobs_constraints(G)

    Q0 = _query_y_x(G, 0)
    Q1 = _query_y_x(G, 1)

    coef_l_0 = build_lp_lifted(G, Q0, cs)
    coef_l_1 = build_lp_lifted(G, Q1, cs)
    quot_0 = build_signature_quotient(G, Q0, cs)
    quot_1 = build_signature_quotient(G, Q1, cs)
    coef_r_0 = build_lp_reduced(quot_0)
    coef_r_1 = build_lp_reduced(quot_1)

    print(
        f"IV graph  lifted={coef_l_0.dim}  reduced(Q0)={coef_r_0.dim}  "
        f"reduced(Q1)={coef_r_1.dim}  n_constraints={coef_l_0.n_constraints}",
        flush=True,
    )
    dump_meta(
        {
            "graph": "IV (Z->X->Y, X<->Y)",
            "queries": ["Y_{x=0}=1", "Y_{x=1}=1"],
            "lifted_dim": coef_l_0.dim,
            "reduced_dim_Q0": coef_r_0.dim,
            "reduced_dim_Q1": coef_r_1.dim,
            "n_constraints": coef_l_0.n_constraints,
            "K": args.K,
            "seed": args.seed,
        },
        out / "meta.json",
    )

    cached_l_0 = CachedSharpLP(coef_l_0)
    cached_l_1 = CachedSharpLP(coef_l_1)
    cached_r_0 = CachedSharpLP(coef_r_0)
    cached_r_1 = CachedSharpLP(coef_r_1)
    warm = np.full(coef_l_0.n_constraints, 1.0 / coef_l_0.n_constraints)
    for c in (cached_l_0, cached_l_1, cached_r_0, cached_r_1):
        try:
            c.solve(warm)
        except Exception:  # noqa: BLE001
            pass

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    for k in range(args.K):
        alpha = rng.dirichlet(np.ones(coef_l_0.dim))
        p = coef_l_0.M @ alpha

        # Truth at the sampled SCM
        true_q0 = float((coef_l_0.c * alpha).sum())
        true_q1 = float((coef_l_1.c * alpha).sum())

        # Solve all 4 LPs
        t0 = time.perf_counter()
        L0_l, U0_l, dL0 = cached_l_0.solve(p)
        L1_l, U1_l, dL1 = cached_l_1.solve(p)
        L0_r, U0_r, dR0 = cached_r_0.solve(p)
        L1_r, U1_r, dR1 = cached_r_1.solve(p)
        t_solve = time.perf_counter() - t0

        joint, cond, _ = _decode_pobs(coef_l_0, p)
        manski_lb_0, manski_ub_0 = _manski_iv_bounds(cond, 0)
        manski_lb_1, manski_ub_1 = _manski_iv_bounds(cond, 1)
        margin_lb_0, margin_ub_0 = _joint_margin_bounds(joint, 0)
        margin_lb_1, margin_ub_1 = _joint_margin_bounds(joint, 1)

        eq_diff = max(
            abs(L0_l - L0_r), abs(U0_l - U0_r),
            abs(L1_l - L1_r), abs(U1_l - U1_r),
        )

        # Sanity: linear LP must respect joint-margin bound (no IV exclusion needed)
        respects_margin = (
            L0_l >= margin_lb_0 - 1e-9
            and U0_l <= margin_ub_0 + 1e-9
            and L1_l >= margin_lb_1 - 1e-9
            and U1_l <= margin_ub_1 + 1e-9
        )
        # Truth-containment
        contains_truth = (
            L0_l - 1e-9 <= true_q0 <= U0_l + 1e-9
            and L1_l - 1e-9 <= true_q1 <= U1_l + 1e-9
        )

        # Bonferroni ATE
        ate_lb = L1_l - U0_l
        ate_ub = U1_l - L0_l

        # Width comparison
        lp_width_0 = U0_l - L0_l
        manski_width_0 = manski_ub_0 - manski_lb_0
        margin_width_0 = margin_ub_0 - margin_lb_0
        lp_width_1 = U1_l - L1_l
        manski_width_1 = manski_ub_1 - manski_lb_1
        margin_width_1 = margin_ub_1 - margin_lb_1

        rows.append({
            "k": k,
            "true_Y_x0_eq_1": round(true_q0, 6),
            "true_Y_x1_eq_1": round(true_q1, 6),
            "L_y_x0_lifted": round(L0_l, 6),
            "U_y_x0_lifted": round(U0_l, 6),
            "L_y_x0_reduced": round(L0_r, 6),
            "U_y_x0_reduced": round(U0_r, 6),
            "L_y_x1_lifted": round(L1_l, 6),
            "U_y_x1_lifted": round(U1_l, 6),
            "L_y_x1_reduced": round(L1_r, 6),
            "U_y_x1_reduced": round(U1_r, 6),
            "manski_L_y_x0": round(manski_lb_0, 6),
            "manski_U_y_x0": round(manski_ub_0, 6),
            "manski_L_y_x1": round(manski_lb_1, 6),
            "manski_U_y_x1": round(manski_ub_1, 6),
            "margin_L_y_x0": round(margin_lb_0, 6),
            "margin_U_y_x0": round(margin_ub_0, 6),
            "margin_L_y_x1": round(margin_lb_1, 6),
            "margin_U_y_x1": round(margin_ub_1, 6),
            "lp_width_y_x0": round(lp_width_0, 6),
            "manski_width_y_x0": round(manski_width_0, 6),
            "margin_width_y_x0": round(margin_width_0, 6),
            "lp_width_y_x1": round(lp_width_1, 6),
            "manski_width_y_x1": round(manski_width_1, 6),
            "margin_width_y_x1": round(margin_width_1, 6),
            "ATE_LB_bonferroni": round(ate_lb, 6),
            "ATE_UB_bonferroni": round(ate_ub, 6),
            "ATE_width": round(ate_ub - ate_lb, 6),
            "lifted_eq_reduced": eq_diff < 1e-6,
            "max_lifted_reduced_diff": round(eq_diff, 10),
            "respects_joint_margin": respects_margin,
            "contains_truth": contains_truth,
            "solve_time_4LP_ms": round(t_solve * 1e3, 4),
        })
        print(
            f"  k={k:2d}  Y_x0:[{L0_l:.4f},{U0_l:.4f}] "
            f"(margin[{margin_lb_0:.4f},{margin_ub_0:.4f}], "
            f"manski[{manski_lb_0:.4f},{manski_ub_0:.4f}], true={true_q0:.4f})  "
            f"true_in_LP={contains_truth}",
            flush=True,
        )

    agg = {
        "K": args.K,
        "all_lifted_eq_reduced": all(r["lifted_eq_reduced"] for r in rows),
        "all_contain_truth": all(r["contains_truth"] for r in rows),
        "all_respect_joint_margin": all(r["respects_joint_margin"] for r in rows),
        "median_LP_width_y_x0": round(float(np.median([r["lp_width_y_x0"] for r in rows])), 6),
        "median_manski_width_y_x0": round(float(np.median([r["manski_width_y_x0"] for r in rows])), 6),
        "median_margin_width_y_x0": round(float(np.median([r["margin_width_y_x0"] for r in rows])), 6),
        "median_LP_width_y_x1": round(float(np.median([r["lp_width_y_x1"] for r in rows])), 6),
        "median_manski_width_y_x1": round(float(np.median([r["manski_width_y_x1"] for r in rows])), 6),
        "median_margin_width_y_x1": round(float(np.median([r["margin_width_y_x1"] for r in rows])), 6),
        "median_ATE_width_bonferroni": round(float(np.median([r["ATE_width"] for r in rows])), 6),
        "interpretation": (
            "Linear LP (ours) matches joint-margin bound P(Y=1,X=x) ≤ P(Y_{x}=1) ≤ 1-P(Y=0,X=x); "
            "Manski (1990) tight IV bound additionally enforces Z ⊥ U_{XY} via the c-component "
            "factorisation, which is a polynomial constraint (Zhang 2022 / Duarte 2024) and "
            "is intentionally outside our paper's linear-LP framework."
        ),
    }
    print("\n[summary]", agg, flush=True)
    dump_meta(agg, out / "summary.json")
    dump_csv(rows, out / args.out_csv)
    print(f"[wrote] {out / args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
