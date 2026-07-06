"""Additional experiment #1 — Lazy Alg-1 Phase 1 demonstration.

Replaces `scalability_demo/`'s analytical-only Phase A for |V| ≥ 7
with **actual wall-clock build + LP solve** using the lazy σ-quotient
builder `build_signature_quotient_lazy_pobs` (canonred/signature.py).

Lazy builder avoids enumerating Omega_full. It iterates over the n_obs_cells
= prod_V d_V observed cells of P(V_1,...,V_n) and, per cell, runs DFS over
Q's expansion graph G* with cell-pinned response-fn entries to determine
which c-indicator values ∈ {0,1} are achievable, with early exit when both
0 and 1 are seen. Σ-classes ≤ 2 × n_cells regardless of |Omega_full|.

For each (|V|, replicate):
  Phase A: analytical lifted-L3 cardinality and Prop 4.2 d_U (always).
  Phase B: lifted build + LP solve (lifted_dim ≤ --max-lifted-dim, default 1<<19).
  Phase C: lazy reduced build + LP solve (always; this is the new lazy demo).
  Phase D: where both B and C succeed, verify lazy_LP ≡ lifted_LP bounds.

Headline column: `lazy_solve_wall_s` — finite for |V|=7,8 where the lifted
path is intractable.

Output columns
--------------
  n_nodes, seed, n_bidir, lifted_dim_analytical_L3, prop42_d_U, structural_ratio,
  lifted_buildable, lifted_build_wall_s, lifted_solve_wall_s, lifted_lb, lifted_ub,
  lazy_build_wall_s, lazy_reduced_dim, lazy_solve_wall_s, lazy_solver_time_s,
  lazy_lb, lazy_ub, abs_lb_diff_vs_lifted, abs_ub_diff_vs_lifted,
  speedup_vs_lifted, status, error.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import dump_csv, dump_meta  # noqa: E402

from canonred.context_set import prop42_cardinality  # noqa: E402
from canonred.counterfactual import CFQuery, CFVar  # noqa: E402
from canonred.graph import ADMG  # noqa: E402
from canonred.lp import build_lp_lifted, build_lp_reduced, solve_sharp_bounds  # noqa: E402
from canonred.observation import pobs_constraints  # noqa: E402
from canonred.random_admg import random_admg  # noqa: E402
from canonred.signature import build_signature_quotient_lazy_pobs  # noqa: E402
from canonred.worlds import World  # noqa: E402


def l3_query(G: ADMG) -> CFQuery:
    """Two-world cross-world L_3 query Q = P(Y_{x=0}=1, Y_{x=1}=1)."""
    topo = G.topo_order()
    y = topo[-1]
    anc = G.ancestors({y}, inclusive=False)
    x = min(anc, key=lambda v: topo.index(v)) if anc else topo[0]
    w0 = World.from_dict("x0", {x: 0})
    w1 = World.from_dict("x1", {x: 1})
    return CFQuery(
        events=((CFVar.standard(y, w0, G), 1), (CFVar.standard(y, w1, G), 1)),
        graph=G, name=f"Y_{x}=0_AND_Y_{x}=1",
    )


def run_one(n_nodes: int, seed: int, p_dir: float, p_bidir: float,
            max_lifted_dim: int) -> dict:
    rng = np.random.default_rng(seed)
    G = random_admg(n_nodes, rng=rng, p_dir=p_dir, p_bidir=p_bidir)
    Q = l3_query(G)
    cs = pobs_constraints(G)

    row: dict = {
        "n_nodes": n_nodes,
        "seed": seed,
        "n_bidir": len(G.bidirected),
    }

    # --- Phase A: analytical ---
    lifted_analytic = G.lifted_canonical_size()
    row["lifted_dim_analytical_L3"] = lifted_analytic
    try:
        d_U = prop42_cardinality(G, Q)
    except Exception as e:  # noqa: BLE001
        d_U = None
        row["prop42_err"] = str(e)
    row["prop42_d_U"] = d_U
    row["structural_ratio"] = (
        round(lifted_analytic / max(d_U, 1), 3) if d_U else None
    )

    # --- Phase C (run first; always feasible by design) ---
    t0 = time.perf_counter()
    try:
        lazy = build_signature_quotient_lazy_pobs(G, Q)
    except Exception as e:  # noqa: BLE001
        row["status"] = "lazy_build_err"
        row["error"] = str(e)
        return row
    row["lazy_build_wall_s"] = round(time.perf_counter() - t0, 6)
    row["lazy_reduced_dim"] = lazy.reduced_dim
    coef_lazy = build_lp_reduced(lazy)

    # Always pick a feasible p: if the lifted side is buildable, use lifted's
    # M @ alpha (gives an exact verification target). Otherwise, use a uniform
    # over cells, which is feasible by construction (all p_obs cells observable).
    p: np.ndarray | None = None
    coef_lifted = None
    lifted_buildable = lifted_analytic <= max_lifted_dim
    row["lifted_buildable"] = lifted_buildable

    if lifted_buildable:
        # --- Phase B: lifted build (only if cardinality permits) ---
        t0 = time.perf_counter()
        try:
            coef_lifted = build_lp_lifted(G, Q, cs)
        except MemoryError:
            row["lifted_buildable"] = False
            row["lifted_build_status"] = "oom"
        else:
            row["lifted_build_wall_s"] = round(time.perf_counter() - t0, 6)
            row["lifted_dim_actual"] = coef_lifted.dim

    if coef_lifted is not None:
        alpha = rng.dirichlet(np.ones(coef_lifted.dim))
        p = coef_lifted.M @ alpha
        p = np.clip(p, 0.0, 1.0)
    else:
        p = np.full(coef_lazy.n_constraints, 1.0 / coef_lazy.n_constraints)

    # --- Solve lifted (if available) ---
    if coef_lifted is not None:
        try:
            t0 = time.perf_counter()
            lb_l, ub_l, dL = solve_sharp_bounds(coef_lifted, p)
            row["lifted_solve_wall_s"] = round(time.perf_counter() - t0, 6)
            row["lifted_solver_time_s"] = round(dL["solver_time"], 6)
            row["lifted_lb"] = round(lb_l, 6)
            row["lifted_ub"] = round(ub_l, 6)
        except Exception as e:  # noqa: BLE001
            row["lifted_solve_err"] = str(e)
            lb_l = ub_l = None
    else:
        lb_l = ub_l = None

    # --- Solve lazy ---
    try:
        t0 = time.perf_counter()
        lb_z, ub_z, dR = solve_sharp_bounds(coef_lazy, p)
        row["lazy_solve_wall_s"] = round(time.perf_counter() - t0, 6)
        row["lazy_solver_time_s"] = round(dR["solver_time"], 6)
        row["lazy_lb"] = round(lb_z, 6)
        row["lazy_ub"] = round(ub_z, 6)
    except Exception as e:  # noqa: BLE001
        row["status"] = "lazy_solve_err"
        row["error"] = str(e)
        return row

    if lb_l is not None and ub_l is not None:
        row["abs_lb_diff_vs_lifted"] = round(abs(lb_l - lb_z), 10)
        row["abs_ub_diff_vs_lifted"] = round(abs(ub_l - ub_z), 10)
        if "lifted_solve_wall_s" in row and row["lifted_solve_wall_s"]:
            row["speedup_vs_lifted"] = round(
                (row["lifted_solve_wall_s"] + row["lifted_build_wall_s"])
                / max(row["lazy_solve_wall_s"] + row["lazy_build_wall_s"], 1e-9),
                3,
            )

    row["status"] = "ok"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--p-dir", type=float, default=0.4)
    ap.add_argument("--p-bidir", type=float, default=0.3)
    ap.add_argument("--max-lifted-dim", type=int, default=1 << 19,
                    help="Skip lifted build if lifted_dim_analytical > this.")
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--out-csv", type=str, default="lazy_phase1.csv")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_meta(
        {"sizes": args.sizes, "replicates": args.replicates,
         "base_seed": args.base_seed, "p_dir": args.p_dir, "p_bidir": args.p_bidir,
         "max_lifted_dim": args.max_lifted_dim,
         "approach": "lazy σ-quotient builder for pobs constraints"},
        out / "meta.json",
    )

    rows: list[dict] = []
    for n in args.sizes:
        for r in range(args.replicates):
            seed = args.base_seed + 1000 * n + r
            row = run_one(
                n_nodes=n, seed=seed,
                p_dir=args.p_dir, p_bidir=args.p_bidir,
                max_lifted_dim=args.max_lifted_dim,
            )
            row["rep"] = r
            rows.append(row)
            ratio = row.get("structural_ratio")
            print(
                f"|V|={n:1d} rep={r:1d}  lifted={row['lifted_dim_analytical_L3']:>22d}  "
                f"d_U={row.get('prop42_d_U')}  ratio={ratio}  "
                f"lazy_red={row.get('lazy_reduced_dim')}  "
                f"lazy_build={row.get('lazy_build_wall_s')}s  "
                f"lazy_solve={row.get('lazy_solve_wall_s')}s  "
                f"lifted_buildable={row.get('lifted_buildable')}  "
                f"diff_LB={row.get('abs_lb_diff_vs_lifted')}",
                flush=True,
            )

    dump_csv(rows, out / args.out_csv)
    print(f"[wrote] {out / args.out_csv} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
