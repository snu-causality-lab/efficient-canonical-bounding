"""Additional experiment: random ADMG sweep (paper, ).

For each (|V|, rep) sampled semi-Markovian ADMG:
  - Pick X = earliest non-sink with a directed path to Y = last topo node.
  - Define Q = P(Y_{X=1} = 1), a single-world L_2 query.
  - Information set = full P_obs(V).
  - Build both lifted LP (explicit enumeration) and reduced LP (σ-quotient).
  - Measure lifted_dim, reduced_dim, Prop 4.2 d_U, build time, solver_time,
    bounds agreement, speedup.

A trial is "skipped" if lifted_dim exceeds `--max-lifted-dim` (default 2^18).

Output CSV: `rand_admg_sweep.csv` with per-trial rows.

Usage:
  python run.py --sizes 4 5 6 --replicates 5
  # Single-(size, rep) mode for slurm arrays:
  python run.py --single-size 5 --single-rep 2

Slurm array: 3 sizes × 5 reps = 15 array tasks → submit as --array=0-14.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import dump_csv, dump_meta  # noqa: E402

from canonred.counterfactual import CFQuery, CFVar  # noqa: E402
from canonred.graph import ADMG  # noqa: E402
from canonred.lp import CachedSharpLP, build_lp_lifted, build_lp_reduced  # noqa: E402
from canonred.observation import pobs_constraints  # noqa: E402
from canonred.random_admg import random_admg  # noqa: E402
from canonred.signature import build_signature_quotient  # noqa: E402
from canonred.worlds import World  # noqa: E402


def pick_xy(G: ADMG) -> tuple[str, str]:
    topo = G.topo_order()
    y = topo[-1]
    anc = G.ancestors({y}, inclusive=False)
    if not anc:
        return topo[0], y
    for v in topo:
        if v in anc:
            return v, y
    return topo[0], y


def run_one(
    n_nodes: int, seed: int,
    p_dir: float, p_bidir: float,
    p_samples: int, max_lifted_dim: int,
) -> dict:
    rng = np.random.default_rng(seed)
    G = random_admg(n_nodes, rng=rng, p_dir=p_dir, p_bidir=p_bidir)
    lifted = G.lifted_canonical_size()
    row: dict = {
        "n_nodes": n_nodes, "seed": seed,
        "n_bidir": len(G.bidirected),
        "lifted_dim": lifted,
    }
    if lifted > max_lifted_dim:
        row.update({"status": "skipped_dim_cap"})
        return row

    x_node, y_node = pick_xy(G)
    w = World.from_dict("x1", {x_node: 1})
    v_y = CFVar.standard(y_node, w, G)
    Q = CFQuery(events=((v_y, 1),), graph=G)
    cs = pobs_constraints(G)

    t0 = time.perf_counter()
    try:
        coef_l = build_lp_lifted(G, Q, cs)
    except MemoryError as e:
        row.update({"status": f"mem_build_lifted: {e}"})
        return row
    t_build_l = time.perf_counter() - t0

    t0 = time.perf_counter()
    q = build_signature_quotient(G, Q, cs)
    coef_r = build_lp_reduced(q)
    t_build_r = time.perf_counter() - t0

    row.update({
        "reduced_dim": q.reduced_dim,
        "n_constraints": coef_l.n_constraints,
        "build_lifted_s": round(t_build_l, 4),
        "build_reduced_s": round(t_build_r, 4),
        "x_node": x_node, "y_node": y_node,
    })

    # Prop 4.2 d_U for informational comparison.
    from canonred.context_set import prop42_cardinality
    try:
        row["prop42_d_U"] = prop42_cardinality(G, Q)
    except Exception as e:  # noqa: BLE001
        row["prop42_d_U_error"] = str(e)

    cached_l = CachedSharpLP(coef_l)
    cached_r = CachedSharpLP(coef_r)
    warmup_p = np.full(coef_l.n_constraints, 1.0 / coef_l.n_constraints)
    try:
        cached_l.solve(warmup_p)
        cached_r.solve(warmup_p)
    except Exception:
        pass  # warmup may fail on infeasible p; ignore

    rt_l: list[float] = []
    rt_r: list[float] = []
    diffs_lb: list[float] = []
    diffs_ub: list[float] = []
    for _ in range(p_samples):
        alpha = rng.dirichlet(np.ones(coef_l.dim))
        p = coef_l.M @ alpha
        p = np.clip(p, 0.0, 1.0)
        try:
            lb_l, ub_l, dL = cached_l.solve(p)
            lb_r, ub_r, dR = cached_r.solve(p)
        except Exception as e:  # noqa: BLE001
            row.update({"status": f"solve_err: {type(e).__name__}"})
            return row
        rt_l.append(dL["solver_time"])
        rt_r.append(dR["solver_time"])
        diffs_lb.append(abs(lb_l - lb_r))
        diffs_ub.append(abs(ub_l - ub_r))

    row.update({
        "status": "ok",
        "max_abs_lb_diff": round(max(diffs_lb), 10),
        "max_abs_ub_diff": round(max(diffs_ub), 10),
        "med_lifted_ms": round(float(np.median(rt_l)) * 1e3, 4),
        "med_reduced_ms": round(float(np.median(rt_r)) * 1e3, 4),
        "speedup": round(float(np.median(rt_l) / max(np.median(rt_r), 1e-12)), 3),
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--p-dir", type=float, default=0.4)
    ap.add_argument("--p-bidir", type=float, default=0.3)
    ap.add_argument("--p-samples", type=int, default=5)
    ap.add_argument("--max-lifted-dim", type=int, default=1 << 18)
    ap.add_argument("--single-size", type=int, default=None)
    ap.add_argument("--single-rep", type=int, default=None)
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--out-csv", type=str, default="rand_admg_sweep.csv")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "sizes": args.sizes, "replicates": args.replicates,
        "base_seed": args.base_seed, "p_dir": args.p_dir, "p_bidir": args.p_bidir,
        "p_samples": args.p_samples, "max_lifted_dim": args.max_lifted_dim,
    }
    dump_meta(meta, out / "meta.json")

    tasks: list[tuple[int, int]] = []
    if args.single_size is not None and args.single_rep is not None:
        tasks.append((args.single_size, args.single_rep))
    else:
        for n in args.sizes:
            for r in range(args.replicates):
                tasks.append((n, r))

    rows: list[dict] = []
    for n, r in tasks:
        local_seed = args.base_seed + 1000 * n + r
        row = run_one(
            n_nodes=n, seed=local_seed,
            p_dir=args.p_dir, p_bidir=args.p_bidir,
            p_samples=args.p_samples, max_lifted_dim=args.max_lifted_dim,
        )
        row["rep"] = r
        rows.append(row)
        print(
            f"|V|={n} rep={r} seed={local_seed} "
            f"lifted={row.get('lifted_dim')} reduced={row.get('reduced_dim')} "
            f"status={row.get('status')} speedup={row.get('speedup')}",
            flush=True,
        )

    dump_csv(rows, out / args.out_csv)
    print(f"[wrote] {out / args.out_csv} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
