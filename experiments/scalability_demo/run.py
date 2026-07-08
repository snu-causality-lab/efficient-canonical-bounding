"""Additional experiment B2 (paper): "lifted infeasible, reduced feasible" demo.

Current implementation note — read before interpreting results.

  Our `build_signature_quotient` enumerates Omega_full explicitly to bucket
  by the (c_u, M_{:,u}) signature (Alg 1, Phase 2 + 3). So the "reduced-side
  build" itself has complexity O(|Omega_full|). A true "infeasible-to-feasible"
  demo — where the reduced-side constructs its quotient WITHOUT materialising
  every full state — requires Alg 1 Phase 1 to be implemented lazily; that's
  deferred.

  What this script does therefore:

    Phase A — Analytical cardinality comparison. For every (|V|, rep):
      report the L_3 worst-case lifted cardinality (via the formula
      prod_V d_V^{|Omega_pa(V)|}) vs. Prop 4.2's structural upper bound d_U
      on the reduced side. The RATIO shows the structural blow-up.

    Phase B — Actual build + solve, if both dims are under `--max-dim`.
      For graphs where enumeration is feasible (typically |V| ≤ 6 at our
      default density), we build the lifted LP, the σ-quotient reduced LP,
      solve both, record wall-clock and solver-time. For graphs too large
      to enumerate, we mark `build_status = "skipped_dim_cap"`.

  Interpretation in the appendix:
    - `lifted_dim_analytical` vs `prop42_d_U` gap (Phase A) tells the
      structural story at large |V|.
    - `lifted_wall_s` vs `reduced_wall_s` (Phase B) validates Prop 3.1
      empirically where it IS buildable.

For |V| ∈ {4, 5, 6} we expect Phase B to succeed; for |V| ∈ {7, 8} only
Phase A has data.

Usage
-----
  python run.py --sizes 4 5 6 7 8 --replicates 3
  python run.py --single-size 6 --single-rep 0

Array indexing:
  idx = size_idx * replicates + rep
  with sizes = [4, 5, 6, 7, 8] and replicates = 3, total = 15 tasks.
"""

from __future__ import annotations

import argparse
import resource
import signal
import sys
import time
from math import prod
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
from canonred.signature import build_signature_quotient  # noqa: E402
from canonred.worlds import World  # noqa: E402


class TimeoutException(Exception):
    pass


def _handler(signum, frame):
    raise TimeoutException()


def run_with_timeout(fn, seconds: float):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return fn()
    prev = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    val = float(r.ru_maxrss)
    return val / (1024.0 ** 2) if val > 1e9 else val / 1024.0


def l3_query(G: ADMG) -> CFQuery:
    """Two-world L_3 query: Q = P(Y_{x=0} = 1, Y_{x=1} = 1) for the last topo node."""
    topo = G.topo_order()
    y = topo[-1]
    anc = G.ancestors({y}, inclusive=False)
    x = min(anc, key=lambda v: topo.index(v)) if anc else topo[0]
    w0 = World.from_dict("x0", {x: 0})
    w1 = World.from_dict("x1", {x: 1})
    return CFQuery(
        events=((CFVar.standard(y, w0, G), 1), (CFVar.standard(y, w1, G), 1)),
        graph=G, name=f"Y_{x}=0 AND Y_{x}=1",
    )


def run_one(
    n_nodes: int, seed: int,
    p_dir: float, p_bidir: float,
    max_dim: int,
    budget_s: float,
) -> dict:
    rng = np.random.default_rng(seed)
    G = random_admg(n_nodes, rng=rng, p_dir=p_dir, p_bidir=p_bidir)
    Q = l3_query(G)
    cs = pobs_constraints(G)

    row: dict = {
        "n_nodes": n_nodes, "seed": seed,
        "n_bidir": len(G.bidirected),
        "max_dim": max_dim,
        "budget_s": budget_s,
    }

    # ---- Phase A: analytical cardinalities (always reported) ----
    lifted_analytic = G.lifted_canonical_size()
    row["lifted_dim_analytical_L3"] = lifted_analytic

    try:
        d_U = prop42_cardinality(G, Q)
        row["prop42_d_U"] = d_U
        row["structural_ratio"] = (
            round(lifted_analytic / max(d_U, 1), 3) if d_U else None
        )
    except Exception as e:  # noqa: BLE001
        row["prop42_err"] = str(e)
        d_U = None

    # ---- Phase B: actual build + solve (only if lifted is enumerable) ----
    if lifted_analytic > max_dim:
        row["build_status"] = "skipped_dim_cap"
        return row

    rss0 = _rss_mb()

    # Lifted side under budget
    t0 = time.perf_counter()
    try:
        def _build_lifted():
            return build_lp_lifted(G, Q, cs)
        coef_l = run_with_timeout(_build_lifted, budget_s)
    except TimeoutException:
        row["build_status"] = "lifted_build_timeout"
        row["lifted_wall_s"] = round(time.perf_counter() - t0, 4)
        return row
    except MemoryError:
        row["build_status"] = "lifted_build_oom"
        return row
    row["lifted_build_wall_s"] = round(time.perf_counter() - t0, 4)
    row["lifted_dim_actual"] = coef_l.dim

    # Reduced side under budget
    t0 = time.perf_counter()
    try:
        def _build_red():
            q = build_signature_quotient(G, Q, cs)
            return q, build_lp_reduced(q)
        q, coef_r = run_with_timeout(_build_red, budget_s)
    except TimeoutException:
        row["build_status"] = "reduced_build_timeout"
        row["reduced_wall_s"] = round(time.perf_counter() - t0, 4)
        return row
    row["reduced_build_wall_s"] = round(time.perf_counter() - t0, 4)
    row["reduced_dim"] = q.reduced_dim

    # Solve one feasible p on each side, measure wall-clock + solver_time
    alpha = rng.dirichlet(np.ones(coef_l.dim))
    p = coef_l.M @ alpha
    p = np.clip(p, 0.0, 1.0)

    rss1 = _rss_mb()
    try:
        t0 = time.perf_counter()
        lb_l, ub_l, dL = solve_sharp_bounds(coef_l, p)
        tl_wall = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        row["build_status"] = f"lifted_solve_err: {type(e).__name__}"
        return row

    try:
        t0 = time.perf_counter()
        lb_r, ub_r, dR = solve_sharp_bounds(coef_r, p)
        tr_wall = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        row["build_status"] = f"reduced_solve_err: {type(e).__name__}"
        return row

    row.update({
        "build_status": "ok",
        "lifted_solve_wall_s": round(tl_wall, 6),
        "lifted_solver_time_s": round(dL["solver_time"], 6),
        "lifted_peak_rss_delta_mb": round(max(0.0, _rss_mb() - rss1), 2),
        "reduced_solve_wall_s": round(tr_wall, 6),
        "reduced_solver_time_s": round(dR["solver_time"], 6),
        "lifted_lb": round(lb_l, 6), "lifted_ub": round(ub_l, 6),
        "reduced_lb": round(lb_r, 6), "reduced_ub": round(ub_r, 6),
        "abs_lb_diff": round(abs(lb_l - lb_r), 10),
        "abs_ub_diff": round(abs(ub_l - ub_r), 10),
        "wall_speedup": round(tl_wall / max(tr_wall, 1e-9), 3),
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--p-dir", type=float, default=0.4)
    ap.add_argument("--p-bidir", type=float, default=0.3)
    ap.add_argument("--max-dim", type=int, default=1 << 18,
                    help="Skip explicit build if lifted_dim_analytical > this.")
    ap.add_argument("--budget-s", type=float, default=300.0,
                    help="Wall-clock budget for each side's build (SIGALRM).")
    ap.add_argument("--single-size", type=int, default=None)
    ap.add_argument("--single-rep", type=int, default=None)
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--out-csv", type=str, default="infeasible_feasible.csv")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "sizes": args.sizes, "replicates": args.replicates,
        "base_seed": args.base_seed, "p_dir": args.p_dir, "p_bidir": args.p_bidir,
        "max_dim": args.max_dim, "budget_s": args.budget_s,
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
        seed = args.base_seed + 1000 * n + r
        row = run_one(
            n_nodes=n, seed=seed,
            p_dir=args.p_dir, p_bidir=args.p_bidir,
            max_dim=args.max_dim, budget_s=args.budget_s,
        )
        row["rep"] = r
        rows.append(row)
        print(
            f"|V|={n} rep={r} seed={seed}  "
            f"lifted_L3={row['lifted_dim_analytical_L3']}  "
            f"d_U={row.get('prop42_d_U')}  "
            f"ratio={row.get('structural_ratio')}  "
            f"status={row.get('build_status')}  "
            f"speedup={row.get('wall_speedup')}",
            flush=True,
        )

    dump_csv(rows, out / args.out_csv)
    print(f"[wrote] {out / args.out_csv} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
