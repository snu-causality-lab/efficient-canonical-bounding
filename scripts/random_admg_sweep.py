"""Random ADMG sweep for graph-variation checks.

For each (|V|, replicate) with |V| in a small range, sample a semi-Markovian
ADMG and run:

    Query : Y_{x=1} where
              Y = last node in topological order
              X = earliest ancestor of Y (first node with Y as descendant)
    Constraints : full observational P_obs(V)

and report, per trial:
  - lifted_dim  (|Omega_full|)
  - reduced_dim (signature-quotient coarsest-safe)
  - speedup on LP solve (median over a small p-sample grid)
  - bounds agreement (max |lb_lifted - lb_reduced|, same for ub)

The script caps lifted_dim at `--max-lifted-dim` (default 2^17 = 131072) and
skips trials exceeding the cap (reporting "skipped").

Outputs: CSV at scripts/outputs/random_admg/sweep.csv.

Usage:
  python -m scripts.random_admg_sweep --sizes 4 5 --replicates 2
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from canonred.counterfactual import CFQuery, CFVar
from canonred.graph import ADMG
from canonred.lp import CachedSharpLP, build_lp_lifted, build_lp_reduced
from canonred.observation import pobs_constraints
from canonred.random_admg import random_admg
from canonred.signature import build_signature_quotient
from canonred.worlds import World

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "random_admg"


def pick_xy(G: ADMG) -> tuple[str, str]:
    """Choose (X, Y): Y = last non-sink-node-in-topo, X = earliest ancestor of Y."""
    topo = G.topo_order()
    y = topo[-1]
    anc = G.ancestors({y}, inclusive=False)
    if not anc:
        # Y is isolated; fall back to picking the first node as X.
        return topo[0], y
    # Earliest (in topo order) ancestor with at least one directed path to Y.
    for v in topo:
        if v in anc:
            return v, y
    return topo[0], y


def run_one(
    G: ADMG,
    *,
    rng: np.random.Generator,
    n_p_samples: int,
    max_lifted_dim: int,
) -> dict:
    lifted = G.lifted_canonical_size()
    row: dict = {
        "nodes": len(G.nodes),
        "n_bidir": len(G.bidirected),
        "lifted_dim": lifted,
    }
    if lifted > max_lifted_dim:
        row.update({"reduced_dim": None, "status": "skipped_dim_cap"})
        return row

    x_node, y_node = pick_xy(G)
    w = World.from_dict("x1", {x_node: 1})
    v_y = CFVar.standard(y_node, w, G)
    Q = CFQuery(events=((v_y, 1),), graph=G)
    cs = pobs_constraints(G)

    # Build both LPs.
    t0 = time.perf_counter()
    try:
        coef_l = build_lp_lifted(G, Q, cs)
    except MemoryError as e:
        row.update({"status": f"error_build_lifted: {e}"})
        return row
    build_lifted_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    quotient = build_signature_quotient(G, Q, cs)
    coef_r = build_lp_reduced(quotient)
    build_reduced_time = time.perf_counter() - t0

    row.update({
        "reduced_dim": quotient.reduced_dim,
        "n_constraints": coef_l.n_constraints,
        "build_time_lifted_s": round(build_lifted_time, 4),
        "build_time_reduced_s": round(build_reduced_time, 4),
    })

    # Sample feasible p: pick a random lifted convex combo, aggregate.
    cached_l = CachedSharpLP(coef_l)
    cached_r = CachedSharpLP(coef_r)
    diffs_lb: list[float] = []
    diffs_ub: list[float] = []
    rt_l: list[float] = []
    rt_r: list[float] = []
    for _ in range(n_p_samples):
        alpha = rng.dirichlet(np.ones(coef_l.dim))
        p = coef_l.M @ alpha
        p = np.clip(p, 0.0, 1.0)
        try:
            lb_l, ub_l, dL = cached_l.solve(p)
            lb_r, ub_r, dR = cached_r.solve(p)
        except Exception as e:  # noqa: BLE001
            row.update({"status": f"error_solve: {e}"})
            return row
        diffs_lb.append(abs(lb_l - lb_r))
        diffs_ub.append(abs(ub_l - ub_r))
        rt_l.append(dL["solver_time"])
        rt_r.append(dR["solver_time"])

    row.update({
        "status": "ok",
        "max_abs_lb_diff": round(max(diffs_lb), 10),
        "max_abs_ub_diff": round(max(diffs_ub), 10),
        "solver_time_lifted_ms_median": round(float(np.median(rt_l)) * 1000.0, 4),
        "solver_time_reduced_ms_median": round(float(np.median(rt_r)) * 1000.0, 4),
        "speedup": round(float(np.median(rt_l)) / float(np.median(rt_r)) if np.median(rt_r) > 0 else -1.0, 3),
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-samples", type=int, default=3, help="# of feasible p values per trial")
    ap.add_argument("--max-lifted-dim", type=int, default=1 << 17)
    ap.add_argument("--p-dir", type=float, default=0.4)
    ap.add_argument("--p-bidir", type=float, default=0.3)
    args = ap.parse_args()

    rows: list[dict] = []
    for n in args.sizes:
        for r in range(args.replicates):
            local_seed = args.seed + 1000 * n + r
            rng = np.random.default_rng(local_seed)
            G = random_admg(n, rng=rng, p_dir=args.p_dir, p_bidir=args.p_bidir)
            row = run_one(
                G, rng=rng, n_p_samples=args.p_samples, max_lifted_dim=args.max_lifted_dim
            )
            row.update({"seed": local_seed, "rep": r})
            rows.append(row)
            print(
                f"|V|={n} rep={r} seed={local_seed} "
                f"lifted={row.get('lifted_dim')} reduced={row.get('reduced_dim')} "
                f"status={row.get('status')} speedup={row.get('speedup')}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "sweep.csv"
    # union of all columns
    fields: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[wrote] {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
